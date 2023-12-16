"""test_splotch.py."""
import numpy as np
import pandas as pd
import pytest
import scipy.stats
from splotch.utils import get_mcmc_summary, savagedickey


@pytest.mark.parametrize(
    "test_input", [(0, 2, 0, 2), (-1, 1, 0, 1), (0, 0.2, -0.5, 0.2)]
)
def test_savagedickey(test_input: tuple[float, float, float, float]) -> None:
    """Test savagedickey()."""
    mu_1, sigma_1, mu_2, sigma_2 = test_input
    mu_1_prior, sigma_1_prior, mu_2_prior, sigma_2_prior = 0, 2, 0, 2
    expected = scipy.stats.norm.pdf(
        0,
        mu_1_prior - mu_2_prior,
        np.sqrt(np.square(sigma_1_prior) + np.square(sigma_2_prior)),
    ) / scipy.stats.norm.pdf(
        0, mu_1 - mu_2, np.sqrt(np.square(sigma_1) + np.square(sigma_2))
    )
    samples_1 = np.random.default_rng(0).normal(mu_1, sigma_1, size=1_000)
    samples_2 = np.random.default_rng(1).normal(mu_2, sigma_2, size=1_000)
    assert np.isclose(
        savagedickey(
            samples_1, samples_2, mu_1_prior, sigma_1_prior, mu_2_prior, sigma_2_prior
        ),
        expected,
        rtol=1e-1,
    )


def test_get_mcmc_summary() -> None:
    """Test get_mcmc_summary()."""
    posterior_samples = {"mu": np.ones((4, 1000))}
    assert isinstance(get_mcmc_summary(posterior_samples["mu"]), pd.DataFrame)
