"""model.py."""
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import Array

# ruff: noqa: N803, N806, PLR0913, PLR0917, PLR2004

GP_INPUT_DIMENSIONS = 3


def gp(x: Array, L: Array, M: list[int], alpha: Array, length: Array) -> Array:
    """Hilbert space approximate Bayesian Gaussian process.

    https://doi.org/10.1007/s11222-022-10167-2

    Args:
        x: Input features.
        L: Interval in which the approximation is valid.
        M: Number of basis functions per dimension.
        alpha: TBA.
        length: TBA.

    Returns:
        Sample from Hilbert space approximate Gaussian process.
    """

    def se_spectral_density(w: Array, alpha: Array, length: Array) -> Array:
        """Get spectral density of the squared exponential covariance function.

        Args:
            w: Input features in the frequency domain.
            alpha: Signal magnitude parameter.
            length: Length-scale parameter.

        Returns:
            Spectral density values of the squared exponential covariance functions.
        """
        # see Eq. 1
        D = w.shape[-1]
        c = alpha * jnp.power(jnp.sqrt(2 * jnp.pi), D) * jnp.prod(length, -1)
        e = jnp.exp(-0.5 * jnp.power(w, 2) @ jnp.power(length, 2))
        return c * e

    def diag_spectral_density(alpha: Array, length: Array, L: Array, S: Array) -> Array:
        """Get diagonal matrix of the spectral density evaluated at the square root of the eigenvalues.

        Args:
            alpha: Signal magnitude parameter.
            length: Length-scale parameter.
            L:  Interval in which the approximation is valid.
            S: Set of possible combinations of univariate eigenfunctions over all dimensions.

        Returns:
            Diagonal matrix of the spectral density evaluated at the square root of the eigenvalues
        """
        # see Eq. 9
        sqrt_eigenvalues = jnp.pi * S / 2 / L
        return se_spectral_density(sqrt_eigenvalues, alpha, length)

    def eigenfunctions(x: Array, L: Array, S: Array) -> Array:
        """Get eigenfunction values.

        Args:
            x: Input features.
            L: Interval in which the approximation is valid.
            S: Set of possible combinations of univariate eigenfunctions over all dimensions.

        Returns:
            Eigenfunction values.
        """
        # see Eq. 10
        sqrt_eigenvalues = jnp.pi * S / 2 / L
        return jnp.prod(
            jnp.power(L, -0.5) * jnp.sin(sqrt_eigenvalues * jnp.expand_dims(x + L, -2)),
            -1,
        )

    assert (
        len(x.shape) == GP_INPUT_DIMENSIONS
    ), f"x should have {GP_INPUT_DIMENSIONS} dimensions"
    S = jnp.transpose(
        jnp.asarray(
            jnp.meshgrid(*[jnp.linspace(1.0, i, num=i) for i in M], indexing="ij")
        ).reshape(len(M), -1)
    )

    phi = eigenfunctions(x, L, S)
    spd = jnp.sqrt(diag_spectral_density(alpha, length, L, S))
    with numpyro.plate("basis", S.shape[0]):
        beta = numpyro.sample("beta", dist.Normal(0, 1))
    return numpyro.deterministic("f", jnp.einsum("ijk,ik->ij", phi, spd * beta))  # type: ignore[no-any-return]


def splotch_v1(
    num_spots: int,
    num_aars: int,
    num_levels: int,
    num_categories_per_level: dict[str, int],
    counts: np.ndarray,
    annotations: np.ndarray,
    padded_coordinates: np.ndarray,
    valid_coordinates: np.ndarray,
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
        padded_coordinates: TBA.
        valid_coordinates: TBA.
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

    padded_coordinates = padded_coordinates - jnp.min(
        padded_coordinates[valid_coordinates], axis=0
    )
    padded_coordinates = padded_coordinates - 0.5 * jnp.max(padded_coordinates, axis=0)
    num_basis = 5
    L = jnp.asarray([1.5 * jnp.max(padded_coordinates)])
    M = padded_coordinates.shape[-1] * [num_basis]
    alpha = numpyro.sample("alpha", dist.Gamma(2, 2))
    length = numpyro.sample("length", dist.Gamma(10, 10))
    with numpyro.plate("tissue_section", padded_coordinates.shape[-3], dim=-2):
        padded_f = numpyro.handlers.scope(gp, "gp")(
            padded_coordinates,
            L,
            M,
            alpha,
            jnp.repeat(length, 2),
        )

    f = numpyro.deterministic("f", padded_f[valid_coordinates[..., 0]])

    sigma_spot = 0.3
    sigma_spot_raw = numpyro.sample("sigma_spot", dist.HalfNormal(1))
    with numpyro.plate("spot", num_spots):
        spot_noise_raw = numpyro.sample("spot_noise_raw", dist.Normal(0, 1))
    spot_noise = numpyro.deterministic(
        "spot_noise", sigma_spot * sigma_spot_raw * spot_noise_raw
    )

    rate = numpyro.deterministic(
        "lambda",
        jnp.exp(beta + spot_noise + f),
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
