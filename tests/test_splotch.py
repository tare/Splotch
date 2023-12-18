"""test_splotch.py."""
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from splotch.utils import (
    detect_tissue_sections,
    get_mcmc_summary,
    read_annotation_files,
    read_count_files,
    savagedickey,
    separate_tissue_sections,
)


@pytest.fixture(scope="session")
def separated_tissue_sections_square_grid() -> np.ndarray:
    """Separated tissue sections on square grid."""
    num_spots_per_tissue_section = 5**2
    x1, y1 = (
        np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[:, None],
            (1, int(np.sqrt(num_spots_per_tissue_section))),
        ),
        np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[None, :],
            (int(np.sqrt(num_spots_per_tissue_section)), 1),
        ),
    )
    x2, y2 = (
        int(np.sqrt(num_spots_per_tissue_section))
        + 1
        + np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[:, None],
            (1, int(np.sqrt(num_spots_per_tissue_section))),
        ),
        np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[None, :],
            (int(np.sqrt(num_spots_per_tissue_section)), 1),
        ),
    )
    coordinates = np.concatenate(
        (
            np.stack((x1.flatten(), y1.flatten())).T,
            np.stack((x2.flatten(), y2.flatten())).T,
        )
    )
    return coordinates + 0.2 * np.random.default_rng(0).uniform(size=coordinates.shape)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def overlapping_tissue_sections_square_grid() -> np.ndarray:
    """Overlapping tissue sections on square grid."""
    num_spots_per_tissue_section = 5**2
    x1, y1 = (
        np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[:, None],
            (1, int(np.sqrt(num_spots_per_tissue_section))),
        ),
        np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[None, :],
            (int(np.sqrt(num_spots_per_tissue_section)), 1),
        ),
    )
    x2, y2 = (
        int(np.sqrt(num_spots_per_tissue_section))
        + 1
        + np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[:, None],
            (1, int(np.sqrt(num_spots_per_tissue_section))),
        ),
        np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[None, :],
            (int(np.sqrt(num_spots_per_tissue_section)), 1),
        ),
    )
    coordinates = np.concatenate(
        (
            np.stack((x1.flatten(), y1.flatten())).T,
            np.stack((x2.flatten(), y2.flatten())).T,
            np.asarray([[5, 1], [5, 2], [5, 3]]),
        )
    )
    return coordinates + 0.2 * np.random.default_rng(0).uniform(size=coordinates.shape)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def separated_tissue_sections_hexagonal_grid() -> np.ndarray:
    """Separated tissue sections on hexagonal grid."""
    num_spots_per_tissue_section = 5**2
    x1, y1 = (
        np.tile(
            np.arange(
                0, 2 * int(np.sqrt(num_spots_per_tissue_section)), step=2, dtype=int
            )[:, None],
            (1, int(np.sqrt(num_spots_per_tissue_section))),
        ),
        np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[None, :],
            (int(np.sqrt(num_spots_per_tissue_section)), 1),
        ),
    )
    x1 = x1 + (np.arange(int(np.sqrt(num_spots_per_tissue_section))) % 2)[None, :]
    x2, y2 = (
        int(np.sqrt(num_spots_per_tissue_section))
        + 1
        + np.tile(
            np.arange(
                0, 2 * int(np.sqrt(num_spots_per_tissue_section)), step=2, dtype=int
            )[:, None],
            (1, int(np.sqrt(num_spots_per_tissue_section))),
        ),
        np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[None, :],
            (int(np.sqrt(num_spots_per_tissue_section)), 1),
        ),
    )
    x2 = (
        int(np.sqrt(num_spots_per_tissue_section))
        + 1
        + x2
        + (np.arange(int(np.sqrt(num_spots_per_tissue_section))) % 2)[None, :]
    )
    coordinates = np.concatenate(
        (
            np.stack((x1.flatten(), y1.flatten())).T,
            np.stack((x2.flatten(), y2.flatten())).T,
        )
    )
    return coordinates + 0.2 * np.random.default_rng(0).uniform(size=coordinates.shape)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def overlapping_tissue_sections_hexagonal_grid() -> np.ndarray:
    """Overlapping tissue sections on hexagonal grid."""
    num_spots_per_tissue_section = 5**2
    x1, y1 = (
        np.tile(
            np.arange(
                0, 2 * int(np.sqrt(num_spots_per_tissue_section)), step=2, dtype=int
            )[:, None],
            (1, int(np.sqrt(num_spots_per_tissue_section))),
        ),
        np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[None, :],
            (int(np.sqrt(num_spots_per_tissue_section)), 1),
        ),
    )
    x1 = x1 + (np.arange(int(np.sqrt(num_spots_per_tissue_section))) % 2)[None, :]
    x2, y2 = (
        int(np.sqrt(num_spots_per_tissue_section))
        + 1
        + np.tile(
            np.arange(
                0, 2 * int(np.sqrt(num_spots_per_tissue_section)), step=2, dtype=int
            )[:, None],
            (1, int(np.sqrt(num_spots_per_tissue_section))),
        ),
        np.tile(
            np.arange(int(np.sqrt(num_spots_per_tissue_section)))[None, :],
            (int(np.sqrt(num_spots_per_tissue_section)), 1),
        ),
    )
    x2 = (
        int(np.sqrt(num_spots_per_tissue_section))
        + 1
        + x2
        + (np.arange(int(np.sqrt(num_spots_per_tissue_section))) % 2)[None, :]
    )
    coordinates = np.concatenate(
        (
            np.stack((x1.flatten(), y1.flatten())).T,
            np.stack((x2.flatten(), y2.flatten())).T,
            np.asarray([[10, 2], [11, 3], [11, 1]]),
        )
    )
    return coordinates + 0.2 * np.random.default_rng(0).uniform(size=coordinates.shape)  # type: ignore[no-any-return]


@pytest.fixture()
def count_files() -> tuple[BytesIO, BytesIO]:
    """Two count files."""
    b1 = BytesIO()
    pd.DataFrame(
        {"12_10": [1, 2, 3, 4, 5], "13_5": [1, 1, 1, 1, 1], "2_2": [0, 1, 1, 1, 1]},
        index=["a", "b", "c", "d", "e"],
    ).to_csv(b1, sep="\t")
    b1.seek(0)

    b2 = BytesIO()
    pd.DataFrame({"12_10": [1, 2, 3], "13_5": [1, 1, 1]}, index=["a", "b", "c"]).to_csv(
        b2, sep="\t"
    )
    b2.seek(0)

    return (b1, b2)


@pytest.fixture()
def annotation_files() -> tuple[BytesIO, BytesIO]:
    """Two annotation files."""
    b1 = BytesIO()
    pd.DataFrame(
        {"12_10": [1, 0, 0], "13_5": [0, 1, 0], "2_2": [0, 0, 1]},
        index=["a", "b", "c"],
    ).to_csv(b1, sep="\t")
    b1.seek(0)

    b2 = BytesIO()
    pd.DataFrame({"12_10": [0, 0, 1], "13_5": [0, 1, 0]}, index=["a", "b", "c"]).to_csv(
        b2, sep="\t"
    )
    b2.seek(0)

    return (b1, b2)


@pytest.fixture()
def annotation_files_invalid_1() -> tuple[BytesIO, BytesIO]:
    """Two invalid annotation files (invalid value)."""
    b1 = BytesIO()
    pd.DataFrame(
        {"12_10": [2, 0, 0], "13_5": [0, 1, 0], "2_2": [0, 0, 1]},
        index=["a", "b", "c"],
    ).to_csv(b1, sep="\t")
    b1.seek(0)

    b2 = BytesIO()
    pd.DataFrame({"12_10": [0, 0, 1], "13_5": [0, 1, 0]}, index=["a", "b", "c"]).to_csv(
        b2, sep="\t"
    )
    b2.seek(0)

    return (b1, b2)


@pytest.fixture()
def annotation_files_invalid_2() -> tuple[BytesIO, BytesIO]:
    """Two invalid annotation files (multiple active categories)."""
    b1 = BytesIO()
    pd.DataFrame(
        {"12_10": [1, 0, 0], "13_5": [0, 1, 0], "2_2": [0, 0, 1]},
        index=["a", "b", "c"],
    ).to_csv(b1, sep="\t")
    b1.seek(0)

    b2 = BytesIO()
    pd.DataFrame({"12_10": [0, 0, 1], "13_5": [1, 1, 0]}, index=["a", "b", "c"]).to_csv(
        b2, sep="\t"
    )
    b2.seek(0)

    return (b1, b2)


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


@pytest.mark.parametrize(
    ("test_input", "expected"), [((4, 1_000), (1, 7)), ((4, 1_000, 10), (10, 7))]
)
def test_get_mcmc_summary(
    test_input: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    """Test get_mcmc_summary()."""
    input_shape = test_input
    output_shape = expected
    posterior_samples = {"mu": np.ones(input_shape)}
    summary_df = get_mcmc_summary(posterior_samples["mu"])
    assert isinstance(summary_df, pd.DataFrame)
    assert summary_df.shape == output_shape
    assert set(summary_df.columns) == {
        "mean",
        "std",
        "median",
        "5.0%",
        "95.0%",
        "n_eff",
        "r_hat",
    }


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("separated_tissue_sections_square_grid", (2, 50)),
        ("separated_tissue_sections_hexagonal_grid", (2, 50)),
        ("overlapping_tissue_sections_square_grid", (1, 53)),
        ("overlapping_tissue_sections_hexagonal_grid", (1, 53)),
    ],
)
def test_detect_tissue_sections(
    test_input: str, expected: tuple[int, int], request: pytest.FixtureRequest
) -> None:
    """Test detect_tissue_sections()."""
    coordinates = request.getfixturevalue(test_input)
    tissue_sections = detect_tissue_sections(coordinates, 8)
    assert len(tissue_sections) == expected[0]
    assert (
        sum([len(tissue_section) for tissue_section in tissue_sections]) == expected[1]
    )


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (("separated_tissue_sections_square_grid", 100), (2, 50)),
        (("separated_tissue_sections_hexagonal_grid", 100), (2, 50)),
        (("overlapping_tissue_sections_square_grid", 100), (1, 53)),
        (("overlapping_tissue_sections_hexagonal_grid", 100), (1, 53)),
        (("separated_tissue_sections_square_grid", 15), (4, 50)),
        (("separated_tissue_sections_hexagonal_grid", 15), (4, 50)),
        (("overlapping_tissue_sections_square_grid", 15), (4, 53)),
        (("overlapping_tissue_sections_hexagonal_grid", 15), (4, 53)),
    ],
)
def test_separate_tissue_sections(
    test_input: str,
    expected: tuple[int, int],
    request: pytest.FixtureRequest,
) -> None:
    """Test separate_tissue_sections()."""
    coordinates = request.getfixturevalue(test_input[0])
    max_num_spots_per_tissue_section = test_input[1]
    tissue_sections = detect_tissue_sections(coordinates, 8)
    tissue_sections = separate_tissue_sections(
        coordinates, tissue_sections, 8, max_num_spots_per_tissue_section, 0
    )
    assert len(tissue_sections) == expected[0]
    assert (
        sum([len(tissue_section) for tissue_section in tissue_sections]) == expected[1]
    )


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (("count_files", 3 / 5), (5, 5)),
        (("count_files", 3 / 5 + 1e-4), (3, 5)),
    ],
)
def test_read_count_files(
    test_input: str,
    expected: tuple[int, int],
    request: pytest.FixtureRequest,
) -> None:
    """Test read_count_files()."""
    count_files = request.getfixturevalue(test_input[0])
    min_detection_rate = test_input[1]
    count_files_df = read_count_files(count_files, min_detection_rate)
    assert count_files_df.shape == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("annotation_files", (3, 5)),
    ],
)
def test_read_annotation_files(
    test_input: str,
    expected: tuple[int, int],
    request: pytest.FixtureRequest,
) -> None:
    """Test read_annotation_files()."""
    annotation_files = request.getfixturevalue(test_input)
    annotation_files_df = read_annotation_files(annotation_files)
    assert annotation_files_df.shape == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "annotation_files_invalid_1",
            (ValueError, "All annotation values should be 0 or 1"),
        ),
        (
            "annotation_files_invalid_2",
            (ValueError, "Each spot should have zero or one active category"),
        ),
    ],
)
def test_read_annotation_files_invalid(
    test_input: str,
    expected: tuple[Exception, str],
    request: pytest.FixtureRequest,
) -> None:
    """Test read_annotation_files() with invalid annotations files."""
    annotation_files = request.getfixturevalue(test_input)
    exception = expected[0]
    msg = expected[1]
    with pytest.raises(exception, match=msg):  # type: ignore[call-overload]
        read_annotation_files(annotation_files)
