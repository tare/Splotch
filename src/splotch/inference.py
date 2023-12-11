"""inference.py."""
from typing import Any

from jax import Array, random
from numpyro.infer import ELBO, MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoGuide, AutoNormal
from numpyro.optim import Adam, _NumPyroOptim

from splotch.models import splotch_v1
from splotch.utils import SplotchInputData, SplotchResult

KeyArray = Array


def run_nuts(
    key: KeyArray,
    gene_idx: int,
    splotch_input_data: SplotchInputData,
    nuts_kwargs: dict[str, Any] | None = None,
    mcmc_kwargs: dict[str, Any] | None = None,
) -> MCMC:
    """Run NUTS.

    Args:
        key: PRNGKey.
        gene_idx: Gene of interest.
        splotch_input_data: Splotch input data.
        nuts_kwargs: Keyword arguments to NUTS. Defaults to ``{}``.
        mcmc_kwargs: Keyword arguments to MCMC.
            Defaults to ``{"num_warmup": 1_000, "num_samples": 1_000, "num_chains": 4}``.

    Returns:
        MCMC object.
    """
    nuts_kwargs = nuts_kwargs or {}
    mcmc_kwargs = {"num_warmup": 1_000, "num_samples": 1_000, "num_chains": 4} | (
        mcmc_kwargs or {}
    )
    key, key_ = random.split(key, 2)
    nuts_kernel = NUTS(splotch_v1, **nuts_kwargs)
    mcmc = MCMC(nuts_kernel, **mcmc_kwargs)
    key, key_ = random.split(key, 2)
    mcmc.run(
        key_,
        gene_idx=gene_idx,
        splotch_input_data=splotch_input_data,
        use_zero_inflated=True,
    )

    return SplotchResult(splotch_input_data.metadata, mcmc, mcmc.get_samples())


def run_svi(
    key: KeyArray,
    gene_idx: int,
    splotch_input_data: SplotchInputData,
    guide: AutoGuide | None = None,
    optim: _NumPyroOptim | None = None,
    loss: ELBO | None = None,
    num_steps: int = 10_000,
    num_samples: int = 1_000,
) -> MCMC:
    """Run NUTS.

    Args:
        key: PRNGKey.
        gene_idx: Gene of interest.
        splotch_input_data: Splotch input data.
        guide: Automatic guide.
        optim: Optimizer. Defaults to numpyro.optim.Adam(step.size=0.1).
        loss: Loss function. Defaults to Trace_ELBO(num_particles=10).
        num_steps: Number of optimization steps. Defaults to 10_000.
        num_samples: Number of samples from the guide. Defaults to 1_000.

    Returns:
        MCMC object.
    """
    guide = guide or AutoNormal(splotch_v1)
    optim = optim or Adam(step_size=0.1)
    loss = loss or Trace_ELBO(num_particles=10)
    svi = SVI(
        splotch_v1,
        guide,
        optim,
        loss,
        gene_idx=gene_idx,
        splotch_input_data=splotch_input_data,
        use_zero_inflated=True,
    )

    key, key_ = random.split(key, 2)
    svi_result = svi.run(key_, num_steps)

    key, key_ = random.split(key, 2)
    posterior_samples = guide.sample_posterior(key_, svi_result.params, (num_samples,))

    return SplotchResult(splotch_input_data.metadata, svi_result, posterior_samples)
