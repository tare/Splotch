"""model.py."""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from splotch.utils import SplotchInputData


def splotch_v1(
    gene_idx: int, splotch_input_data: SplotchInputData, use_zero_inflated: bool = False
) -> None:
    """Splotch generative model.

    Args:
        gene_idx: Gene of interest.
        splotch_input_data: Splotch input data.
        use_zero_inflated: Whether to use the zero-inflated Poisson likelihood.
    """
    num_spots = splotch_input_data.num_spots()
    num_aars = splotch_input_data.num_aars()
    num_levels = splotch_input_data.num_levels()
    num_categories_per_level = splotch_input_data.num_categories_per_level()

    counts = splotch_input_data.counts()[:, gene_idx]
    annotations = splotch_input_data.annotations()
    levels = splotch_input_data.levels()
    size_factors = splotch_input_data.size_factors()

    if num_levels not in {1, 2, 3}:
        msg = "Only 1, 2, or 3 levels are supported"
        raise ValueError(msg)

    with numpyro.plate("aar", num_aars):
        sigma_level_1 = 2.0
        with numpyro.plate("level_1", num_categories_per_level["level_1"]):
            beta_level_1_raw = numpyro.sample("beta_level_1_raw", dist.Normal(0, 1))
            beta_level_1 = numpyro.deterministic(
                "beta_level_1", sigma_level_1 * beta_level_1_raw
            )

    if num_levels > 1:
        sigma_level_2 = numpyro.sample("sigma_level_2", dist.HalfNormal(1))
        with numpyro.plate("aar", num_aars), numpyro.plate(
            "level_2", num_categories_per_level["level_2"]
        ):
            beta_level_2_raw = numpyro.sample("beta_level_2_raw", dist.Normal(0, 1))
            beta_level_2 = numpyro.deterministic(
                "beta_level_2", sigma_level_2 * beta_level_2_raw
            )

    if num_levels > 2:
        sigma_level_3 = numpyro.sample("sigma_level_3", dist.HalfNormal(1))
        with numpyro.plate("aar", num_aars), numpyro.plate(
            "level_3", num_categories_per_level["level_3"]
        ):
            beta_level_3_raw = numpyro.sample("beta_level_3_raw", dist.Normal(0, 1))
            beta_level_3 = numpyro.deterministic(
                "beta_level_3", sigma_level_3 * beta_level_3_raw
            )

    if num_levels == 1:
        beta = numpyro.deterministic("beta", beta_level_1[levels[:, 0], annotations])
    elif num_levels == 2:
        beta = numpyro.deterministic(
            "beta",
            beta_level_1[levels[:, 0], annotations]
            + beta_level_2[levels[:, 1], annotations],
        )
    elif num_levels == 3:
        beta = numpyro.deterministic(
            "beta",
            beta_level_1[levels[:, 0], annotations]
            + beta_level_2[levels[:, 1], annotations]
            + beta_level_3[levels[:, 2], annotations],
        )

    # TODO (tare): add HSGP

    sigma_spot = 0.3
    sigma_spot_raw = numpyro.sample("sigma_spot", dist.HalfNormal(1))
    with numpyro.plate("spot", num_spots):
        spot_noise_raw = numpyro.sample("spot_noise_raw", dist.Normal(0, 1))

    rate = numpyro.deterministic(
        "lambda", jnp.exp(beta + sigma_spot * sigma_spot_raw * spot_noise_raw)
    )

    if use_zero_inflated:
        theta = numpyro.sample("theta", dist.Beta(1, 2))
        with numpyro.plate("spot", num_spots):
            numpyro.sample(
                "counts",
                dist.ZeroInflatedPoisson(theta, rate * size_factors),
                obs=counts,
            )
    else:
        with numpyro.plate("spot", num_spots):
            numpyro.sample("counts", dist.Poisson(rate * size_factors), obs=counts)
