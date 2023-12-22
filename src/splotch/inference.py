"""inference.py."""
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array, jit, pmap, random, vmap
from jax.lax import scan
from jax.tree_util import tree_map
from numpyro.infer import ELBO, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoGuide, AutoNormal
from numpyro.infer.hmc import HMCState, hmc
from numpyro.infer.svi import SVIState
from numpyro.infer.util import initialize_model
from numpyro.optim import Adam, _NumPyroOptim
from numpyro.util import fori_collect

from splotch.dataclasses import SplotchInputData, SplotchResult
from splotch.models import splotch_v1
from splotch.utils import get_mcmc_summary

KeyArray = Array


def get_padded_coordinates(
    splotch_input_data: SplotchInputData,
) -> tuple[np.ndarray, np.ndarray]:
    """Get padded coordinates.

    Args:
        splotch_input_data: Splotch input data.

    Returns:
        Padded coordinates and valid indices.
    """
    tissue_sections = splotch_input_data.metadata.index.get_level_values(
        "tissue_section"
    ).values
    coordinates = np.split(
        np.vstack(
            (
                splotch_input_data.metadata.x.values,
                splotch_input_data.metadata.y.values,
            )
        ).T,
        np.where(tissue_sections[:-1] != tissue_sections[1:])[0] + 1,
    )

    max_tissue_section_size = int(
        max(
            np.diff(
                np.hstack(
                    (
                        np.zeros(1),
                        np.where(tissue_sections[:-1] != tissue_sections[1:])[0] + 1,
                        len(tissue_sections) * np.ones(1),
                    )
                )
            )
        )
    )
    padded_coordinates = np.stack(
        [
            np.pad(
                c,
                ((0, max_tissue_section_size - len(c)), (0, 0)),
                constant_values=np.nan,
            )
            for c in coordinates
        ]
    )
    valid_coordinates = ~np.isnan(padded_coordinates)
    padded_coordinates[np.isnan(padded_coordinates)] = 0.0

    return padded_coordinates, valid_coordinates


def get_splotch_kwargs(
    splotch_input_data: SplotchInputData, use_zero_inflated: bool
) -> dict[str, int | np.ndarray | bool | dict[str, int]]:
    """Return.

    Args:
        splotch_input_data: TBA.
        use_zero_inflated: TBA.

    Returns:
        Dictionary with data.
    """
    num_spots = splotch_input_data.num_spots()
    num_aars = splotch_input_data.num_aars()
    num_levels = splotch_input_data.num_levels()
    num_categories_per_level = splotch_input_data.num_categories_per_level()
    annotations = splotch_input_data.annotations()
    levels = splotch_input_data.levels()
    size_factors = splotch_input_data.size_factors()
    padded_coordinates, valid_coordinates = get_padded_coordinates(splotch_input_data)

    return {
        "num_spots": num_spots,
        "num_aars": num_aars,
        "num_levels": num_levels,
        "num_categories_per_level": num_categories_per_level,
        "annotations": annotations,
        "padded_coordinates": padded_coordinates,
        "valid_coordinates": valid_coordinates,
        "levels": levels,
        "size_factors": size_factors,
        "use_zero_inflated": use_zero_inflated,
    }


def run_nuts(
    key: Array,
    genes: list[str],
    splotch_input_data: SplotchInputData,
    map_method: str = "map",
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    num_chains: int = 4,
    use_zero_inflated: bool = False,
) -> SplotchResult:
    """Run NUTS.

    Args:
        key: PRNGKey.
        genes: Genes of interest.
        splotch_input_data: Splotch input data.
        map_method: Map method. Possible values are pmap, vmap, and map. Defaults to map.
        num_warmup: Number of warmup iterations.
        num_samples: Number of sampling iterations.
        num_chains: Number of chains.
        use_zero_inflated: Whether to use the zero-inflated Poisson likelihood. Defaults to False.
    """

    def get_model_kwargs(model_kwargs: dict[str, Any], counts: Array) -> dict[str, Any]:
        return model_kwargs | {"counts": counts}

    model_kwargs = get_splotch_kwargs(splotch_input_data, use_zero_inflated)

    counts = jnp.asarray(splotch_input_data.counts(genes))

    key, key_ = random.split(key, 2)
    model_info = initialize_model(
        key_,
        splotch_v1,
        dynamic_args=True,
        model_kwargs=get_model_kwargs(model_kwargs, counts[:, 0]),
    )

    init_kernel, sample_kernel = hmc(
        potential_fn_gen=model_info.potential_fn, algo="NUTS"
    )

    def sample_posterior(hmc_state: HMCState, counts: Array) -> dict[str, Array]:
        samples: dict[str, Array] = fori_collect(
            lower=num_warmup,
            upper=num_warmup + num_samples,
            body_fun=partial(
                sample_kernel,
                model_args=(),
                model_kwargs=get_model_kwargs(model_kwargs, counts),
            ),
            init_val=hmc_state,
            transform=(
                lambda hmc_state: model_info.postprocess_fn(
                    **get_model_kwargs(model_kwargs, counts)
                )(hmc_state.z)
            ),
            progbar=False,
        )
        return samples

    key, key_ = random.split(key)
    hmc_states = vmap(
        lambda gene_key: vmap(
            lambda chain_key: init_kernel(
                model_info.param_info, num_warmup, rng_key=chain_key
            )
        )(random.split(gene_key, num_chains))
    )(random.split(key_, len(genes)))

    if map_method == "vmap":
        samples = vmap(
            lambda hmc_state, gene_counts: vmap(
                lambda x: sample_posterior(x, gene_counts)
            )(hmc_state),
            in_axes=(0, 1),
        )(hmc_states, counts)
    elif map_method == "pmap":
        samples = pmap(
            lambda hmc_state, gene_counts: pmap(
                lambda x: sample_posterior(x, gene_counts)
            )(hmc_state),
            in_axes=(0, 1),
        )(hmc_states, counts)
    elif map_method == "map":

        def get_hmc_state(
            hmc_states: HMCState, gene_idx: int, chain_idx: int
        ) -> HMCState:
            return tree_map(lambda x: x[gene_idx][chain_idx], hmc_states)

        gene_res = []
        for gene_idx in range(len(genes)):
            gene_counts = counts[:, gene_idx]
            chain_res = []
            for chain_idx in range(num_chains):
                hmc_state = get_hmc_state(hmc_states, gene_idx, chain_idx)
                samples = jit(sample_posterior)(hmc_state, gene_counts)
                chain_res.append(samples)
            gene_res.append(tree_map(lambda *x: jnp.stack(x), *chain_res))
        samples = tree_map(lambda *x: jnp.stack(x), *gene_res)
    else:
        msg = "map_method should be pmap, vmap or map"
        raise ValueError(msg)

    def get_summaries(samples: Array) -> pd.DataFrame:
        summary_dfs = [
            get_mcmc_summary(sample)
            .assign(gene=gene)
            .reset_index()
            .set_index(["gene", "index"])
            for gene, sample in zip(genes, samples)
        ]
        return pd.concat(summary_dfs, axis=0)

    return SplotchResult(
        splotch_input_data.metadata,
        genes,
        {"summary": tree_map(get_summaries, samples)},
        tree_map(
            lambda x: jnp.moveaxis(
                jnp.reshape(
                    jnp.moveaxis(x, 0, 2),
                    (x.shape[1] * x.shape[2], x.shape[0], *x.shape[3:]),
                ),
                0,
                1,
            ),
            samples,
        ),
    )


def run_svi(
    key: KeyArray,
    genes: list[str],
    splotch_input_data: SplotchInputData,
    map_method: str = "map",
    guide: AutoGuide | None = None,
    optim: _NumPyroOptim | None = None,
    loss: ELBO | None = None,
    num_steps: int = 10_000,
    num_samples: int = 1_000,
    use_zero_inflated: bool = False,
) -> SplotchResult:
    """Run SVI.

    Args:
        key: PRNGKey.
        genes: Genes of interest.
        splotch_input_data: Splotch input data.
        map_method: Map method. Possible values are pmap, vmap, and map. Defaults to map.
        guide: Automatic guide.
        optim: Optimizer. Defaults to numpyro.optim.Adam(step.size=0.1).
        loss: Loss function. Defaults to Trace_ELBO(num_particles=10).
        num_steps: Number of optimization steps. Defaults to 10_000.
        num_samples: Number of samples from the guide. Defaults to 1_000.
        use_zero_inflated: Whether to use the zero-inflated Poisson likelihood. Defaults to False.

    Returns:
        MCMC object.
    """
    guide = guide or AutoNormal(splotch_v1)
    optim = optim or Adam(step_size=0.1)
    loss = loss or Trace_ELBO(num_particles=10)

    model_kwargs = get_splotch_kwargs(splotch_input_data, use_zero_inflated)
    counts = np.asarray(splotch_input_data.counts(genes))

    def run_svi(key: KeyArray, counts: Array) -> tuple[dict[str, Array], Array, Array]:
        svi = SVI(splotch_v1, guide, optim, loss, counts=counts, **model_kwargs)
        key, key_ = random.split(key, 2)
        svi_state = svi.init(key_)

        def body_fn(svi_state: SVIState, _: None) -> tuple[SVIState, Array]:
            svi_state, losses = svi.update(svi_state)
            return svi_state, losses

        svi_state, losses = scan(body_fn, svi_state, None, length=num_steps)
        params = svi.get_params(svi_state)
        key, key_ = random.split(key, 2)
        samples = guide.sample_posterior(
            key_,
            params,
            (num_samples,),
        )
        return samples, params, losses

    key, key_ = random.split(key, 2)
    if map_method == "pmap":
        samples, params, losses = pmap(run_svi, in_axes=(0, 1))(
            random.split(key_, len(genes)), counts
        )
    elif map_method == "vmap":
        samples, params, losses = vmap(run_svi, in_axes=(0, 1))(
            random.split(key_, len(genes)), jnp.asarray(counts)
        )
    elif map_method == "map":
        gene_res = []
        keys = random.split(key_, len(genes))
        for gene_idx, key_ in zip(range(len(genes)), keys):
            gene_res.append(jit(run_svi)(key_, counts[:, gene_idx]))
        samples, params, losses = tree_map(lambda *x: jnp.stack(x), *gene_res)
    else:
        msg = "map_method should be pmap, vmap or map"
        raise ValueError(msg)

    return SplotchResult(
        splotch_input_data.metadata,
        genes,
        {"losses": losses, "params": params},
        samples,
    )
