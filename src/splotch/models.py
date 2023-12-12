"""model.py."""
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist


def splotch_v1(
    num_spots: int,
    num_aars: int,
    num_levels: int,
    num_categories_per_level: dict[str, int],
    counts: np.ndarray,
    annotations: np.ndarray,
    levels: np.ndarray,
    size_factors: np.ndarray,
    use_zero_inflated: bool = False,
) -> None:
    """Splotch generative model.

    Args:
        num_spots: Number of spots.
        num_aars: Number of AARs.
        num_levels: Number of levels.
        num_categories_per_level: Number of categories per level
        counts: Count for each spot.
        annotations: Annotation for each spot.
        levels: Level categories for each spot.
        size_factors: Size factor for each spot.
        use_zero_inflated: Whether to use the zero-inflated Poisson likelihood.
    """
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
