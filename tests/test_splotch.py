"""test_splotch.py."""

import os
import re
from io import BytesIO
from typing import Literal

import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import scipy.stats
from jax import Array, random
from matplotlib.figure import Figure

from splotch.inference import get_splotch_kwargs, run_nuts, run_svi
from splotch.models import get_default_priors, splotch_v1
from splotch.registration import register
from splotch.utils import (
    SplotchInputData,
    SpotData,
    detect_tissue_sections,
    get_mcmc_summary,
    get_spot_adjacency_matrix,
    process_input_data,
    read_annotation_files,
    read_count_files,
    savagedickey,
    separate_tissue_sections,
)
from splotch.visualization import (
    plot_annotations_in_common_coordinate_system,
    plot_annotations_on_slides,
    plot_coefficients,
    plot_rates_in_common_coordinate_system,
    plot_rates_on_slides,
    plot_tissue_sections_on_slides,
    plot_variable_on_slides,
)

mpl.use("Agg")

# this is needed for the tests using pmap
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


@pytest.fixture(scope="session")
def separated_tissue_sections_square_grid() -> npt.NDArray[np.float64]:
    """Separated tissue sections on square grid.

    Returns:
        Coordinates.
    """
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
    return coordinates + 0.2 * np.random.default_rng(0).uniform(size=coordinates.shape)


@pytest.fixture(scope="session")
def overlapping_tissue_sections_square_grid() -> npt.NDArray[np.float64]:
    """Overlapping tissue sections on square grid.

    Returns:
        Coordinates.
    """
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
    return coordinates + 0.2 * np.random.default_rng(0).uniform(size=coordinates.shape)


@pytest.fixture(scope="session")
def separated_tissue_sections_hexagonal_grid() -> npt.NDArray[np.float64]:
    """Separated tissue sections on hexagonal grid.

    Returns:
        Coordinates.
    """
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
    x1 += (np.arange(int(np.sqrt(num_spots_per_tissue_section))) % 2)[None, :]
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
    return coordinates + 0.2 * np.random.default_rng(0).uniform(size=coordinates.shape)


@pytest.fixture(scope="session")
def overlapping_tissue_sections_hexagonal_grid() -> npt.NDArray[np.float64]:
    """Overlapping tissue sections on hexagonal grid.

    Returns:
        Coordinates.
    """
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
    x1 += (np.arange(int(np.sqrt(num_spots_per_tissue_section))) % 2)[None, :]
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
    return coordinates + 0.2 * np.random.default_rng(0).uniform(size=coordinates.shape)


@pytest.fixture
def count_files() -> tuple[BytesIO, BytesIO]:
    """Two count files.

    Returns:
        Two binary streams representing count files.
    """
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


@pytest.fixture
def annotation_files() -> tuple[BytesIO, BytesIO]:
    """Two annotation files.

    Returns:
        Two binary streams representing annotation files.
    """
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


@pytest.fixture
def annotation_files_invalid_1() -> tuple[BytesIO, BytesIO]:
    """Two invalid annotation files (invalid value).

    Returns:
        Two binary streams representing annotation files.
    """
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


@pytest.fixture
def annotation_files_invalid_2() -> tuple[BytesIO, BytesIO]:
    """Two invalid annotation files (multiple active categories).

    Returns:
        Two binary streams representing annotation files.
    """
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


@pytest.fixture
def spot_data() -> (
    tuple[
        npt.NDArray[np.bytes_],
        pd.Series,
        pd.Series,
        npt.NDArray[np.bytes_],
        npt.NDArray[np.int64],
        pd.DataFrame,
    ]
):
    """Spot data.

    Returns:
        Tuple containing genes, level values, metadata, coordinates, spot counts, and annotations.
    """
    genes = np.asarray(["g1", "g2"])
    level_values = pd.Series({"level_1": "a"})
    metadata = pd.Series({"annotation_file": "file.tsv", "image_file": "image.jpg"})
    coordinates_orig = np.asarray(["0_0", "3.0_2.0", "15_-1", "5_2"])
    spot_counts = np.asarray([[1, 2], [3, 4], [5, 5], [2, 2]])
    annotations_df = pd.DataFrame(
        {
            "0_0": [1, 0],
            "3.0_2.0": [1, 0],
            "15_-1": [0, 0],
            "5_2": [0, 1],
            "100_100": [0, 1],
        },
        index=["aar1", "aar2"],
    )

    return genes, level_values, metadata, coordinates_orig, spot_counts, annotations_df


@pytest.fixture
def spot_data_invalid_1() -> (
    tuple[
        npt.NDArray[np.bytes_],
        pd.Series,
        pd.Series,
        npt.NDArray[np.bytes_],
        npt.NDArray[np.int64],
        pd.DataFrame,
    ]
):
    """Invalid spot data.

    Coordinates do not following the required naming schema.

    Returns:
        Tuple containing genes, level values, metadata, coordinates, spot counts, and annotations.
    """
    genes = np.asarray(["g1", "g2"])
    level_values = pd.Series({"level_1": "a"})
    metadata = pd.Series({"annotation_file": "file.tsv", "image_file": "image.jpg"})
    coordinates_orig = np.asarray(["0_0", "3.0,2.0", "15_-1", "5_2"])
    spot_counts = np.asarray([[1, 2], [3, 4], [5, 5], [2, 2]])
    annotations_df = pd.DataFrame(
        {
            "0_0": [1, 0],
            "3.0_2.0": [1, 0],
            "15_-1": [0, 0],
            "5_2": [0, 1],
            "100_100": [0, 1],
        },
        index=["aar1", "aar2"],
    )

    return genes, level_values, metadata, coordinates_orig, spot_counts, annotations_df


@pytest.fixture
def spot_data_invalid_2() -> (
    tuple[
        npt.NDArray[np.bytes_],
        pd.Series,
        pd.Series,
        npt.NDArray[np.bytes_],
        npt.NDArray[np.int64],
        pd.DataFrame,
    ]
):
    """Spot data.

    Number of genes do not match.

    Returns:
        Tuple containing genes, level values, metadata, coordinates, spot counts, and annotations.
    """
    genes = np.asarray(["g1", "g2", "g3"])
    level_values = pd.Series({"level_1": "a"})
    metadata = pd.Series({"annotation_file": "file.tsv", "image_file": "image.jpg"})
    coordinates_orig = np.asarray(["0_0", "3.0_2.0", "15_-1", "5_2"])
    spot_counts = np.asarray([[1, 2], [3, 4], [5, 5], [2, 2]])
    annotations_df = pd.DataFrame(
        {
            "0_0": [1, 0],
            "3.0_2.0": [1, 0],
            "15_-1": [0, 0],
            "5_2": [0, 1],
            "100_100": [0, 1],
        },
        index=["aar1", "aar2"],
    )

    return genes, level_values, metadata, coordinates_orig, spot_counts, annotations_df


@pytest.fixture
def spot_data_invalid_3() -> (
    tuple[
        npt.NDArray[np.bytes_],
        pd.Series,
        pd.Series,
        npt.NDArray[np.bytes_],
        npt.NDArray[np.int64],
        pd.DataFrame,
    ]
):
    """Spot data.

    Number of coordinates do not match.

    Returns:
        Tuple containing genes, level values, metadata, coordinates, spot counts, and annotations.
    """
    genes = np.asarray(["g1", "g2"])
    level_values = pd.Series({"level_1": "a"})
    metadata = pd.Series({"annotation_file": "file.tsv", "image_file": "image.jpg"})
    coordinates_orig = np.asarray(["0_0", "3.0_2.0", "15_-1", "5_2"])
    spot_counts = np.asarray([[1, 2], [3, 4], [5, 5], [2, 2], [4, 4]])
    annotations_df = pd.DataFrame(
        {
            "0_0": [1, 0],
            "3.0_2.0": [1, 0],
            "15_-1": [0, 0],
            "5_2": [0, 1],
            "100_100": [0, 1],
        },
        index=["aar1", "aar2"],
    )

    return genes, level_values, metadata, coordinates_orig, spot_counts, annotations_df


@pytest.fixture
def splotchinputdata() -> (
    tuple[SplotchInputData, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """SplotchInputData.

    Returns:
        Tuple containing splotch input data, metadata, counts, and annotations.
    """
    count_files = ["count_file_1.tsv", "count_file_2.tsv"]
    annotation_files = ["annotation_file_1.tsv", "annotation_file_2.tsv"]

    metadata_df = pd.DataFrame(
        {
            "level_1": ["cond1", "cond2"],
            "count_file": count_files,
            "annotation_file": annotation_files,
            "image_file": ["image_1.jpeg", "image_2.jpeg"],
        }
    )
    counts_dfs = [
        pd.DataFrame(
            {"10_10": [2, 1], "11_10": [5, 10], "13_10": [1, 1], "12_10": [1, 2]},
            index=["g1", "g2"],
        ),
        pd.DataFrame(
            {"5_3": [10, 5], "5_2": [5, 1], "6_3": [3, 1], "6_2": [1, 4]},
            index=["g1", "g2"],
        ),
    ]

    for filename, count_df in zip(count_files, counts_dfs, strict=True):
        count_df.columns = pd.MultiIndex.from_product(
            [[filename], count_df.columns], names=["file", "coordinate"]
        )
        count_df.index.name = "gene"

    counts_df = pd.concat(counts_dfs, copy=False, axis=1, sort=True)

    annotation_dfs = [
        pd.DataFrame(
            {"10_10": [1, 0], "11_10": [0, 1], "13_10": [0, 1], "12_10": [1, 0]},
            index=["aar1", "aar2"],
        ),
        pd.DataFrame(
            {"5_3": [1, 0], "5_2": [0, 1], "6_3": [0, 1], "6_2": [1, 0]},
            index=["aar1", "aar2"],
        ),
    ]

    for filename, annotation_df in zip(annotation_files, annotation_dfs, strict=True):
        annotation_df.columns = pd.MultiIndex.from_product(
            [[filename], annotation_df.columns], names=["file", "coordinate"]
        )
        annotation_df.index.name = "aar"

    annotations_df = pd.concat(annotation_dfs, copy=False, axis=1, sort=True)

    splotch_input_data = process_input_data(
        metadata_df,
        counts_df,
        annotations_df,
        num_levels=1,
        min_total_count=1,
        min_num_spots_per_slide=1,
        num_of_neighbors=2,
        separate_overlapping_tissue_sections=True,
        min_num_spots_per_tissue_section=1,
    )

    return splotch_input_data, metadata_df, counts_df, annotations_df


@pytest.fixture
def splotchinputdata_low_coverage() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """SplotchInputData.

    Returns:
        Tuple containing metadata, counts, and annotations.
    """
    count_files = ["count_file_1.tsv", "count_file_2.tsv"]
    annotation_files = ["annotation_file_1.tsv", "annotation_file_2.tsv"]

    metadata_df = pd.DataFrame(
        {
            "level_1": ["cond1", "cond2"],
            "count_file": count_files,
            "annotation_file": annotation_files,
            "image_file": ["image_1.jpeg", "image_2.jpeg"],
        }
    )
    counts_dfs = [
        pd.DataFrame(
            {"10_10": [2, 1], "11_10": [5, 10], "13_10": [1, 1], "12_10": [1, 2]},
            index=["g1", "g2"],
        ),
        pd.DataFrame(
            {"5_3": [12, 5], "5_2": [5, 1], "6_3": [3, 1], "6_2": [1, 4]},
            index=["g1", "g2"],
        ),
    ]

    for filename, count_df in zip(count_files, counts_dfs, strict=True):
        count_df.columns = pd.MultiIndex.from_product(
            [[filename], count_df.columns], names=["file", "coordinate"]
        )
        count_df.index.name = "gene"

    counts_df = pd.concat(counts_dfs, copy=False, axis=1, sort=True)

    annotation_dfs = [
        pd.DataFrame(
            {"1_10": [1, 0], "1_11": [0, 1], "1_12": [0, 1], "1_13": [1, 0]},
            index=["aar1", "aar2"],
        ),
        pd.DataFrame(
            {"5_3": [1, 0], "5_2": [0, 1], "6_3": [0, 1], "6_2": [1, 0]},
            index=["aar1", "aar2"],
        ),
    ]

    for filename, annotation_df in zip(annotation_files, annotation_dfs, strict=True):
        annotation_df.columns = pd.MultiIndex.from_product(
            [[filename], annotation_df.columns], names=["file", "coordinate"]
        )
        annotation_df.index.name = "aar"

    annotations_df = pd.concat(annotation_dfs, copy=False, axis=1, sort=True)

    return metadata_df, counts_df, annotations_df


@pytest.fixture
def splotchinputdata_registration() -> SplotchInputData:
    """SplotchInputData.

    Returns:
        Processed input data.
    """
    count_files = ["count_file_1.tsv", "count_file_2.tsv"]
    annotation_files = ["annotation_file_1.tsv", "annotation_file_2.tsv"]

    metadata_df = pd.DataFrame(
        {
            "level_1": ["cond1", "cond2"],
            "count_file": count_files,
            "annotation_file": annotation_files,
            "image_file": ["image_1.jpeg", "image_2.jpeg"],
        }
    )
    counts_dfs = [
        pd.DataFrame(
            {"1_1": [2, 1], "2_2": [5, 10], "3_3": [1, 1], "4_4": [1, 2]},
            index=["g1", "g2"],
        ),
        pd.DataFrame(
            {
                "8_8": [10, 5],
                "7_7": [5, 1],
                "6_6": [3, 1],
                "5_5": [1, 4],
                "11_1": [10, 5],
                "11_0": [5, 1],
                "11_-1": [3, 1],
                "11_-2": [1, 4],
            },
            index=["g1", "g2"],
        ),
    ]

    for filename, count_df in zip(count_files, counts_dfs, strict=True):
        count_df.columns = pd.MultiIndex.from_product(
            [[filename], count_df.columns], names=["file", "coordinate"]
        )
        count_df.index.name = "gene"

    counts_df = pd.concat(counts_dfs, copy=False, axis=1, sort=True)

    annotation_dfs = [
        pd.DataFrame(
            {"1_1": [1, 0], "2_2": [0, 1], "3_3": [0, 1], "4_4": [1, 0]},
            index=["aar1", "aar2"],
        ),
        pd.DataFrame(
            {
                "8_8": [1, 0],
                "7_7": [0, 1],
                "6_6": [0, 1],
                "5_5": [1, 0],
                "11_1": [1, 0],
                "11_0": [0, 1],
                "11_-1": [0, 1],
                "11_-2": [1, 0],
            },
            index=["aar1", "aar2"],
        ),
    ]

    for filename, annotation_df in zip(annotation_files, annotation_dfs, strict=True):
        annotation_df.columns = pd.MultiIndex.from_product(
            [[filename], annotation_df.columns], names=["file", "coordinate"]
        )
        annotation_df.index.name = "aar"

    annotations_df = pd.concat(annotation_dfs, copy=False, axis=1, sort=True)

    return process_input_data(
        metadata_df,
        counts_df,
        annotations_df,
        num_levels=1,
        min_total_count=1,
        min_num_spots_per_slide=1,
        num_of_neighbors=2,
        separate_overlapping_tissue_sections=False,
        min_num_spots_per_tissue_section=1,
    )


@pytest.fixture
def splotchinputdata_inference() -> SplotchInputData:
    """SplotchInputData.

    Returns:
        Processed input data.
    """
    count_files = ["count_file_1.tsv", "count_file_2.tsv"]
    annotation_files = ["annotation_file_1.tsv", "annotation_file_2.tsv"]

    metadata_df = pd.DataFrame(
        {
            "level_1": ["cond1", "cond2"],
            "count_file": count_files,
            "annotation_file": annotation_files,
            "image_file": ["image_1.jpeg", "image_2.jpeg"],
        }
    )
    counts_dfs = [
        pd.DataFrame(
            {"1_1": [2, 1], "2_2": [5, 10], "3_3": [1, 1], "4_4": [1, 2]},
            index=["g1", "g2"],
        ),
        pd.DataFrame(
            {
                "8_8": [10, 5],
                "7_7": [5, 1],
                "6_6": [3, 1],
                "5_5": [1, 4],
                "11_1": [10, 5],
                "11_0": [5, 1],
                "11_-1": [3, 1],
                "11_-2": [1, 4],
            },
            index=["g1", "g2"],
        ),
    ]

    for filename, count_df in zip(count_files, counts_dfs, strict=True):
        count_df.columns = pd.MultiIndex.from_product(
            [[filename], count_df.columns], names=["file", "coordinate"]
        )
        count_df.index.name = "gene"

    counts_df = pd.concat(counts_dfs, copy=False, axis=1, sort=True)

    annotation_dfs = [
        pd.DataFrame(
            {"1_1": [1, 0], "2_2": [0, 1], "3_3": [0, 1], "4_4": [1, 0]},
            index=["aar1", "aar2"],
        ),
        pd.DataFrame(
            {
                "8_8": [1, 0],
                "7_7": [0, 1],
                "6_6": [0, 1],
                "5_5": [1, 0],
                "11_1": [1, 0],
                "11_0": [0, 1],
                "11_-1": [0, 1],
                "11_-2": [1, 0],
            },
            index=["aar1", "aar2"],
        ),
    ]

    for filename, annotation_df in zip(annotation_files, annotation_dfs, strict=True):
        annotation_df.columns = pd.MultiIndex.from_product(
            [[filename], annotation_df.columns], names=["file", "coordinate"]
        )
        annotation_df.index.name = "aar"

    annotations_df = pd.concat(annotation_dfs, copy=False, axis=1, sort=True)

    return process_input_data(
        metadata_df,
        counts_df,
        annotations_df,
        num_levels=1,
        min_total_count=1,
        min_num_spots_per_slide=1,
        num_of_neighbors=2,
        separate_overlapping_tissue_sections=False,
        min_num_spots_per_tissue_section=1,
    )


@pytest.fixture
def splotchinputdata_inference_2() -> SplotchInputData:
    """SplotchInputData.

    Returns:
        Processed input data.
    """
    count_files = ["count_file_1.tsv", "count_file_2.tsv"]
    annotation_files = ["annotation_file_1.tsv", "annotation_file_2.tsv"]

    metadata_df = pd.DataFrame(
        {
            "level_1": ["cond1", "cond2"],
            "count_file": count_files,
            "annotation_file": annotation_files,
            "image_file": ["image_2.jpeg", "image_1.jpeg"],
        }
    )
    counts_dfs = [
        pd.DataFrame(
            {"1_1": [2, 1], "2_2": [5, 10], "3_3": [1, 1], "4_4": [1, 2]},
            index=["g1", "g2"],
        ),
        pd.DataFrame(
            {
                "8_8": [10, 5],
                "7_7": [5, 1],
                "6_6": [3, 1],
                "5_5": [1, 4],
                "11_1": [10, 5],
                "11_0": [5, 1],
                "11_-1": [3, 1],
                "11_-2": [1, 4],
            },
            index=["g1", "g2"],
        ),
    ]

    for filename, count_df in zip(count_files, counts_dfs, strict=True):
        count_df.columns = pd.MultiIndex.from_product(
            [[filename], count_df.columns], names=["file", "coordinate"]
        )
        count_df.index.name = "gene"

    counts_df = pd.concat(counts_dfs, copy=False, axis=1, sort=True)

    annotation_dfs = [
        pd.DataFrame(
            {"1_1": [1, 0], "2_2": [0, 1], "3_3": [0, 1], "4_4": [1, 0]},
            index=["aar1", "aar2"],
        ),
        pd.DataFrame(
            {
                "8_8": [1, 0],
                "7_7": [0, 1],
                "6_6": [0, 1],
                "5_5": [1, 0],
                "11_1": [1, 0],
                "11_0": [0, 1],
                "11_-1": [0, 1],
                "11_-2": [1, 0],
            },
            index=["aar1", "aar2"],
        ),
    ]

    for filename, annotation_df in zip(annotation_files, annotation_dfs, strict=True):
        annotation_df.columns = pd.MultiIndex.from_product(
            [[filename], annotation_df.columns], names=["file", "coordinate"]
        )
        annotation_df.index.name = "aar"

    annotations_df = pd.concat(annotation_dfs, copy=False, axis=1, sort=True)

    return process_input_data(
        metadata_df,
        counts_df,
        annotations_df,
        num_levels=1,
        min_total_count=1,
        min_num_spots_per_slide=1,
        num_of_neighbors=2,
        separate_overlapping_tissue_sections=False,
        min_num_spots_per_tissue_section=1,
    )


@pytest.fixture
def splotchinputdata_inference_two_levels() -> SplotchInputData:
    """SplotchInputData.

    Returns:
        Processed input data.
    """
    count_files = ["count_file_1.tsv", "count_file_2.tsv"]
    annotation_files = ["annotation_file_1.tsv", "annotation_file_2.tsv"]

    metadata_df = pd.DataFrame(
        {
            "level_1": ["cond1", "cond2"],
            "level_2": ["cond3", "cond3"],
            "count_file": count_files,
            "annotation_file": annotation_files,
            "image_file": ["image_1.jpeg", "image_2.jpeg"],
        }
    )
    counts_dfs = [
        pd.DataFrame(
            {"1_1": [2, 1], "2_2": [5, 10], "3_3": [1, 1], "4_4": [1, 2]},
            index=["g1", "g2"],
        ),
        pd.DataFrame(
            {
                "8_8": [10, 5],
                "7_7": [5, 1],
                "6_6": [3, 1],
                "5_5": [1, 4],
                "11_1": [10, 5],
                "11_0": [5, 1],
                "11_-1": [3, 1],
                "11_-2": [1, 4],
            },
            index=["g1", "g2"],
        ),
    ]

    for filename, count_df in zip(count_files, counts_dfs, strict=True):
        count_df.columns = pd.MultiIndex.from_product(
            [[filename], count_df.columns], names=["file", "coordinate"]
        )
        count_df.index.name = "gene"

    counts_df = pd.concat(counts_dfs, copy=False, axis=1, sort=True)

    annotation_dfs = [
        pd.DataFrame(
            {"1_1": [1, 0], "2_2": [0, 1], "3_3": [0, 1], "4_4": [1, 0]},
            index=["aar1", "aar2"],
        ),
        pd.DataFrame(
            {
                "8_8": [1, 0],
                "7_7": [0, 1],
                "6_6": [0, 1],
                "5_5": [1, 0],
                "11_1": [1, 0],
                "11_0": [0, 1],
                "11_-1": [0, 1],
                "11_-2": [1, 0],
            },
            index=["aar1", "aar2"],
        ),
    ]

    for filename, annotation_df in zip(annotation_files, annotation_dfs, strict=True):
        annotation_df.columns = pd.MultiIndex.from_product(
            [[filename], annotation_df.columns], names=["file", "coordinate"]
        )
        annotation_df.index.name = "aar"

    annotations_df = pd.concat(annotation_dfs, copy=False, axis=1, sort=True)

    return process_input_data(
        metadata_df,
        counts_df,
        annotations_df,
        num_levels=2,
        min_total_count=1,
        min_num_spots_per_slide=1,
        num_of_neighbors=2,
        separate_overlapping_tissue_sections=False,
        min_num_spots_per_tissue_section=1,
    )


@pytest.fixture
def splotchinputdata_inference_three_levels() -> SplotchInputData:
    """SplotchInputData.

    Returns:
        Processed input data.
    """
    count_files = ["count_file_1.tsv", "count_file_2.tsv"]
    annotation_files = ["annotation_file_1.tsv", "annotation_file_2.tsv"]

    metadata_df = pd.DataFrame(
        {
            "level_1": ["cond1", "cond2"],
            "level_2": ["cond3", "cond3"],
            "level_3": ["cond4", "cond5"],
            "count_file": count_files,
            "annotation_file": annotation_files,
            "image_file": ["image_1.jpeg", "image_2.jpeg"],
        }
    )
    counts_dfs = [
        pd.DataFrame(
            {"1_1": [2, 1], "2_2": [5, 10], "3_3": [1, 1], "4_4": [1, 2]},
            index=["g1", "g2"],
        ),
        pd.DataFrame(
            {
                "8_8": [10, 5],
                "7_7": [5, 1],
                "6_6": [3, 1],
                "5_5": [1, 4],
                "11_1": [10, 5],
                "11_0": [5, 1],
                "11_-1": [3, 1],
                "11_-2": [1, 4],
            },
            index=["g1", "g2"],
        ),
    ]

    for filename, count_df in zip(count_files, counts_dfs, strict=True):
        count_df.columns = pd.MultiIndex.from_product(
            [[filename], count_df.columns], names=["file", "coordinate"]
        )
        count_df.index.name = "gene"

    counts_df = pd.concat(counts_dfs, copy=False, axis=1, sort=True)

    annotation_dfs = [
        pd.DataFrame(
            {"1_1": [1, 0], "2_2": [0, 1], "3_3": [0, 1], "4_4": [1, 0]},
            index=["aar1", "aar2"],
        ),
        pd.DataFrame(
            {
                "8_8": [1, 0],
                "7_7": [0, 1],
                "6_6": [0, 1],
                "5_5": [1, 0],
                "11_1": [1, 0],
                "11_0": [0, 1],
                "11_-1": [0, 1],
                "11_-2": [1, 0],
            },
            index=["aar1", "aar2"],
        ),
    ]

    for filename, annotation_df in zip(annotation_files, annotation_dfs, strict=True):
        annotation_df.columns = pd.MultiIndex.from_product(
            [[filename], annotation_df.columns], names=["file", "coordinate"]
        )
        annotation_df.index.name = "aar"

    annotations_df = pd.concat(annotation_dfs, copy=False, axis=1, sort=True)

    return process_input_data(
        metadata_df,
        counts_df,
        annotations_df,
        num_levels=3,
        min_total_count=1,
        min_num_spots_per_slide=1,
        num_of_neighbors=2,
        separate_overlapping_tissue_sections=False,
        min_num_spots_per_tissue_section=1,
    )


@pytest.fixture
def splotchinputdata_inference_wout_images() -> SplotchInputData:
    """SplotchInputData.

    Returns:
        Processed input data.
    """
    count_files = ["count_file_1.tsv", "count_file_2.tsv"]
    annotation_files = ["annotation_file_1.tsv", "annotation_file_2.tsv"]

    metadata_df = pd.DataFrame(
        {
            "level_1": ["cond1", "cond2"],
            "count_file": count_files,
            "annotation_file": annotation_files,
        }
    )
    counts_dfs = [
        pd.DataFrame(
            {"1_1": [2, 1], "2_2": [5, 10], "3_3": [1, 1], "4_4": [1, 2]},
            index=["g1", "g2"],
        ),
        pd.DataFrame(
            {
                "8_8": [10, 5],
                "7_7": [5, 1],
                "6_6": [3, 1],
                "5_5": [1, 4],
                "11_1": [10, 5],
                "11_0": [5, 1],
                "11_-1": [3, 1],
                "11_-2": [1, 4],
            },
            index=["g1", "g2"],
        ),
    ]

    for filename, count_df in zip(count_files, counts_dfs, strict=True):
        count_df.columns = pd.MultiIndex.from_product(
            [[filename], count_df.columns], names=["file", "coordinate"]
        )
        count_df.index.name = "gene"

    counts_df = pd.concat(counts_dfs, copy=False, axis=1, sort=True)

    annotation_dfs = [
        pd.DataFrame(
            {"1_1": [1, 0], "2_2": [0, 1], "3_3": [0, 1], "4_4": [1, 0]},
            index=["aar1", "aar2"],
        ),
        pd.DataFrame(
            {
                "8_8": [1, 0],
                "7_7": [0, 1],
                "6_6": [0, 1],
                "5_5": [1, 0],
                "11_1": [1, 0],
                "11_0": [0, 1],
                "11_-1": [0, 1],
                "11_-2": [1, 0],
            },
            index=["aar1", "aar2"],
        ),
    ]

    for filename, annotation_df in zip(annotation_files, annotation_dfs, strict=True):
        annotation_df.columns = pd.MultiIndex.from_product(
            [[filename], annotation_df.columns], names=["file", "coordinate"]
        )
        annotation_df.index.name = "aar"

    annotations_df = pd.concat(annotation_dfs, copy=False, axis=1, sort=True)

    return process_input_data(
        metadata_df,
        counts_df,
        annotations_df,
        num_levels=1,
        min_total_count=1,
        min_num_spots_per_slide=1,
        num_of_neighbors=2,
        separate_overlapping_tissue_sections=False,
        min_num_spots_per_tissue_section=1,
    )


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
    posterior_samples: dict[str, Array] = {"mu": jnp.ones(input_shape)}
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
    assert sum(len(tissue_section) for tissue_section in tissue_sections) == expected[1]


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
    test_input: tuple[str, int],
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
    assert sum(len(tissue_section) for tissue_section in tissue_sections) == expected[1]


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (("count_files", 3 / 5), (5, 5)),
        (("count_files", 3 / 5 + 1e-4), (3, 5)),
    ],
)
def test_read_count_files(
    test_input: tuple[str, float],
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
    expected: tuple[type[Exception], str],
    request: pytest.FixtureRequest,
) -> None:
    """Test read_annotation_files() with invalid annotations files."""
    annotation_files = request.getfixturevalue(test_input)
    exception = expected[0]
    msg = expected[1]
    with pytest.raises(exception, match=msg):
        read_annotation_files(annotation_files)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "spot_data",
            (
                np.asarray(["g1", "g2"]),
                np.asarray(["0_0", "3.0_2.0", "5_2"]),
                np.asarray([[0.0, 0.0], [3.0, 2.0], [5.0, 2.0]]),
                np.asarray([[1, 2], [3, 4], [2, 2]]),
                pd.DataFrame(
                    {"0_0": [1, 0], "3.0_2.0": [1, 0], "5_2": [0, 1]},
                    index=["aar1", "aar2"],
                ),
            ),
        ),
    ],
)
def test_spotdata(
    test_input: str,
    expected: tuple[
        npt.NDArray[np.bytes_],
        npt.NDArray[np.bytes_],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        pd.DataFrame,
    ],
    request: pytest.FixtureRequest,
) -> None:
    """Test SpotData."""
    (
        genes,
        level_values,
        metadata,
        coordinates_orig,
        spot_counts,
        annotations_df,
    ) = request.getfixturevalue(test_input)
    (
        expected_genes,
        expected_coordinates_orig,
        expected_coordinates,
        expected_spot_counts,
        expected_annotations_df,
    ) = expected
    spot_data = SpotData(
        genes, level_values, metadata, coordinates_orig, spot_counts, annotations_df
    )
    assert np.all(spot_data.genes == expected_genes)
    assert np.all(spot_data.coordinates_orig == expected_coordinates_orig)
    assert np.all(spot_data.coordinates == expected_coordinates)
    assert np.all(spot_data.spot_counts == expected_spot_counts)
    assert spot_data.annotations_df.equals(expected_annotations_df)

    indices = np.asarray([True, False, False])
    spot_data = spot_data.select(indices)
    assert np.all(spot_data.genes == expected_genes)
    assert np.all(spot_data.coordinates_orig == expected_coordinates_orig[indices])
    assert np.all(spot_data.coordinates == expected_coordinates[indices])
    assert np.all(spot_data.spot_counts == expected_spot_counts[indices, :])
    assert spot_data.annotations_df.equals(expected_annotations_df.iloc[:, indices])


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            "spot_data_invalid_1",
            (
                ValueError,
                "Unable not find x and y coordinates. Ensure the naming pattern is 'x_y'.",
            ),
        ),
        (
            "spot_data_invalid_2",
            (ValueError, "Number of genes do not match"),
        ),
        (
            "spot_data_invalid_3",
            (ValueError, "Number of coordinates do not match"),
        ),
    ],
)
def test_spotdata_invalid(
    test_input: str,
    expected: tuple[type[Exception], str],
    request: pytest.FixtureRequest,
) -> None:
    """Test SpotData with invalid data."""
    (
        genes,
        level_values,
        metadata,
        coordinates_orig,
        spot_counts,
        annotations_df,
    ) = request.getfixturevalue(test_input)
    exception = expected[0]
    msg = expected[1]
    with pytest.raises(exception, match=msg):
        SpotData(
            genes, level_values, metadata, coordinates_orig, spot_counts, annotations_df
        )


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            (
                np.asarray(
                    [[0, 0], [1, 0], [0, 1], [1, 1], [3, 0], [4, 0], [3, 1], [4, 1]],
                    dtype=np.float64,
                ),
                3,
            ),
            np.asarray(
                [
                    [False, True, True, True, False, False, False, False],
                    [True, False, True, True, False, False, False, False],
                    [True, True, False, True, False, False, False, False],
                    [True, True, True, False, False, False, False, False],
                    [False, False, False, False, False, True, True, True],
                    [False, False, False, False, True, False, True, True],
                    [False, False, False, False, True, True, False, True],
                    [False, False, False, False, True, True, True, False],
                ]
            ),
        ),
        (
            (
                np.asarray(
                    [[0, 0], [1, 0], [0, 1], [1, 1], [3, 0], [4, 0], [3, 1], [4, 1]],
                    dtype=np.float64,
                ),
                2,
            ),
            np.asarray(
                [
                    [False, True, True, False, False, False, False, False],
                    [True, False, False, True, False, False, False, False],
                    [True, False, False, True, False, False, False, False],
                    [False, True, True, False, False, False, False, False],
                    [False, False, False, False, False, True, True, False],
                    [False, False, False, False, True, False, False, True],
                    [False, False, False, False, True, False, False, True],
                    [False, False, False, False, False, True, True, False],
                ]
            ),
        ),
    ],
)
def test_get_spot_adjacency_matrix(
    test_input: tuple[npt.NDArray[np.float64], int], expected: npt.NDArray[np.bool]
) -> None:
    """Test get_spot_adjacency_matrix."""
    coordinates, num_of_neighbors = test_input
    assert np.all(get_spot_adjacency_matrix(coordinates, num_of_neighbors) == expected)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            (
                np.asarray(
                    [[0, 0], [1, 0], [0, 1], [1, 1], [3, 0], [4, 0], [3, 1], [4, 1]],
                    dtype=np.float64,
                ),
                0,
            ),
            (
                ValueError,
                re.escape(
                    "num_of_neighbors has to be between 1 and len(coordinates) - 2."
                ),
            ),
        ),
        (
            (
                np.asarray(
                    [[0, 0], [1, 0], [0, 1], [1, 1], [3, 0], [4, 0], [3, 1], [4, 1]],
                    dtype=np.float64,
                ),
                7,
            ),
            (
                ValueError,
                re.escape(
                    "num_of_neighbors has to be between 1 and len(coordinates) - 2."
                ),
            ),
        ),
    ],
)
def test_get_spot_adjacency_matrix_invalid(
    test_input: tuple[npt.NDArray[np.float64], int],
    expected: tuple[type[Exception], str],
) -> None:
    """Test get_spot_adjacency_matrix with invalid num_of_neighbors."""
    coordinates, num_of_neighbors = test_input
    exception, msg = expected
    with pytest.raises(exception, match=msg):
        get_spot_adjacency_matrix(coordinates, num_of_neighbors)


def test_splotchinputdata(
    splotchinputdata: tuple[SplotchInputData, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Test SplotchInputData."""
    splotch_input_data, metadata_df, counts_df, annotations_df = splotchinputdata

    assert splotch_input_data.num_levels() == 1
    assert splotch_input_data.num_categories_per_level() == {
        "level_1": metadata_df.level_1.nunique()
    }
    assert splotch_input_data.num_aars() == annotations_df.shape[0]
    assert splotch_input_data.num_spots() == annotations_df.shape[1]
    assert np.all(splotch_input_data.counts() == counts_df.to_numpy().T)
    assert np.all(splotch_input_data.counts(["g1"]) == counts_df.to_numpy()[[0], :].T)
    assert np.all(
        splotch_input_data.annotations() == np.argmax(annotations_df.to_numpy(), axis=0)
    )
    assert np.all(splotch_input_data.aars() == np.asarray(annotations_df.index))
    assert np.all(
        (counts_df.sum(0) / counts_df.sum(0).median()).to_numpy()
        == splotch_input_data.size_factors()
    )


def test_splotchinputdata_low_coverage(
    splotchinputdata_low_coverage: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test SplotchInputData with low coverage slides and spots."""
    metadata_df, counts_df, annotations_df = splotchinputdata_low_coverage
    with pytest.raises(ValueError, match="No valid tissue sections found"):
        process_input_data(
            metadata_df,
            counts_df,
            annotations_df,
            num_levels=1,
            min_total_count=5,
            min_num_spots_per_slide=1,
            num_of_neighbors=1,
            separate_overlapping_tissue_sections=False,
            min_num_spots_per_tissue_section=5,
        )
    assert (
        caplog.records[0].msg
        == "%s has less than %d valid spots. Maybe coordinates do not match in the count and annotation files."
    )
    assert caplog.records[1].msg == "Discarding %d spots due to low sequencing depth."
    assert caplog.records[2].msg == "Discarding a tissue section with %d spots"


def test_register(
    splotchinputdata_registration: SplotchInputData, caplog: pytest.LogCaptureFixture
) -> None:
    """Test register()."""
    key = random.PRNGKey(0)
    key, key_ = random.split(key, 2)
    register(key_, splotchinputdata_registration, num_steps=10_000)
    assert all(
        col in splotchinputdata_registration.metadata.columns
        for col in ["x_registration", "y_registration"]
    )
    assert np.all(
        np.isclose(splotchinputdata_registration.metadata.y_registration, 0, atol=1e-2)
    )

    register(
        key_, splotchinputdata_registration, num_steps=10_000, aars_of_interest=["aar1"]
    )
    assert all(
        col in splotchinputdata_registration.metadata.columns
        for col in ["x_registration", "y_registration"]
    )
    assert np.all(
        np.isclose(splotchinputdata_registration.metadata.y_registration, 0, atol=1e-2)
    )

    register(
        key_, splotchinputdata_registration, num_steps=2, aars_of_interest=["aar1"]
    )
    assert caplog.records[0].msg == "Not converged after %d iterations"


@pytest.mark.parametrize(
    "test_input",
    [
        "splotchinputdata_inference",
        "splotchinputdata_inference_two_levels",
        "splotchinputdata_inference_three_levels",
    ],
)
@pytest.mark.parametrize(
    "use_zero_inflated",
    [False, True],
)
@pytest.mark.parametrize(
    "map_method",
    ["map", "vmap", "pmap"],
)
def test_run_nuts(
    test_input: str,
    map_method: Literal["map", "vmap", "pmap"],
    use_zero_inflated: bool,  # noqa: FBT001
    request: pytest.FixtureRequest,
) -> None:
    """Test run_nuts()."""
    splotchinputdata_inference = request.getfixturevalue(test_input)
    key = random.PRNGKey(0)
    key, key_ = random.split(key, 2)
    splotch_result_nuts_batch_1 = run_nuts(
        key_,
        ["g1"],
        splotchinputdata_inference,
        map_method=map_method,
        num_warmup=10,
        num_samples=10,
        num_chains=4,
        use_zero_inflated=use_zero_inflated,
    )
    assert splotch_result_nuts_batch_1.genes == ["g1"]
    assert splotch_result_nuts_batch_1.posterior_samples["lambda"].shape == (
        1,
        40,
        12,
    )

    key, key_ = random.split(key, 2)
    splotch_result_nuts_batch_2 = run_nuts(
        key_,
        ["g2"],
        splotchinputdata_inference,
        map_method=map_method,
        num_warmup=10,
        num_samples=10,
        num_chains=4,
        use_zero_inflated=use_zero_inflated,
    )
    assert splotch_result_nuts_batch_2.genes == ["g2"]
    assert splotch_result_nuts_batch_2.posterior_samples["lambda"].shape == (
        1,
        40,
        12,
    )

    splotch_result_nuts = splotch_result_nuts_batch_1 + splotch_result_nuts_batch_2
    assert splotch_result_nuts.genes == ["g1", "g2"]
    assert splotch_result_nuts.posterior_samples["lambda"].shape == (
        2,
        40,
        12,
    )
    assert list(splotch_result_nuts.inference_metrics["summary"]["lambda"].columns) == [
        "mean",
        "std",
        "median",
        "5.0%",
        "95.0%",
        "n_eff",
        "r_hat",
    ]


@pytest.mark.parametrize(
    "test_input",
    [
        "splotchinputdata_inference",
        "splotchinputdata_inference_two_levels",
        "splotchinputdata_inference_three_levels",
    ],
)
@pytest.mark.parametrize(
    "use_zero_inflated",
    [False, True],
)
@pytest.mark.parametrize(
    "map_method",
    ["map", "vmap", "pmap"],
)
def test_run_svi(
    test_input: str,
    map_method: Literal["map", "vmap", "pmap"],
    use_zero_inflated: bool,  # noqa: FBT001
    request: pytest.FixtureRequest,
) -> None:
    """Test run_svi()."""
    splotchinputdata_inference = request.getfixturevalue(test_input)
    key = random.PRNGKey(0)
    key, key_ = random.split(key, 2)
    splotch_result_svi_batch_1 = run_svi(
        key_,
        ["g1"],
        splotchinputdata_inference,
        map_method=map_method,
        num_steps=50,
        num_samples=10,
        use_zero_inflated=use_zero_inflated,
    )
    assert splotch_result_svi_batch_1.genes == ["g1"]
    assert splotch_result_svi_batch_1.posterior_samples["lambda"].shape == (
        1,
        10,
        12,
    )

    key, key_ = random.split(key, 2)
    splotch_result_svi_batch_2 = run_svi(
        key_,
        ["g2"],
        splotchinputdata_inference,
        map_method=map_method,
        num_steps=50,
        num_samples=10,
        use_zero_inflated=use_zero_inflated,
    )
    assert splotch_result_svi_batch_2.genes == ["g2"]
    assert splotch_result_svi_batch_2.posterior_samples["lambda"].shape == (
        1,
        10,
        12,
    )

    splotch_result_svi = splotch_result_svi_batch_1 + splotch_result_svi_batch_2
    assert splotch_result_svi.genes == ["g1", "g2"]
    assert splotch_result_svi.posterior_samples["lambda"].shape == (
        2,
        10,
        12,
    )
    assert all(
        col in splotch_result_svi.inference_metrics for col in ["losses", "params"]
    )

    assert splotch_result_svi.inference_metrics["losses"].shape == (2, 50)


def test_run_nuts_invalid_map_method(
    splotchinputdata_inference: SplotchInputData,
) -> None:
    """Test run_nuts()."""
    key = random.PRNGKey(0)
    key, key_ = random.split(key, 2)
    map_method = "mapp"
    with pytest.raises(ValueError, match="map_method should be pmap, vmap or map"):
        run_nuts(
            key_,
            ["g1"],
            splotchinputdata_inference,
            map_method=map_method,  # type: ignore[arg-type]
            num_warmup=10,
            num_samples=10,
            num_chains=4,
        )


def test_run_svi_invalid_map_method(
    splotchinputdata_inference: SplotchInputData,
) -> None:
    """Test run_svi()."""
    key = random.PRNGKey(0)
    key, key_ = random.split(key, 2)
    map_method = "mapp"
    with pytest.raises(ValueError, match="map_method should be pmap, vmap or map"):
        run_svi(
            key_,
            ["g1"],
            splotchinputdata_inference,
            map_method=map_method,  # type: ignore[arg-type]
            num_steps=50,
            num_samples=10,
        )


def test_splotch_result_add(
    splotchinputdata_inference: SplotchInputData,
    splotchinputdata_inference_2: SplotchInputData,
) -> None:
    """Test run_nuts()."""
    key = random.PRNGKey(0)
    key, key_ = random.split(key, 2)

    splotch_result_nuts_batch_1 = run_nuts(
        key_,
        ["g1"],
        splotchinputdata_inference,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
    )
    splotch_result_nuts_batch_2 = run_nuts(
        key_,
        ["g2"],
        splotchinputdata_inference_2,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
    )
    with pytest.raises(ValueError, match="Metadata are not the same"):
        splotch_result_nuts_batch_1 + splotch_result_nuts_batch_2

    splotch_result_nuts_batch_1 = run_nuts(
        key_,
        ["g1"],
        splotchinputdata_inference,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
    )
    splotch_result_nuts_batch_2 = run_nuts(
        key_,
        ["g1"],
        splotchinputdata_inference,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
    )
    with pytest.raises(ValueError, match="Genes overlap"):
        splotch_result_nuts_batch_1 + splotch_result_nuts_batch_2

    splotch_result_nuts_batch_1 = run_nuts(
        key_,
        ["g1"],
        splotchinputdata_inference,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
    )
    splotch_result_svi_batch_2 = run_svi(
        key_,
        ["g2"],
        splotchinputdata_inference,
        num_steps=50,
        num_samples=10,
    )
    with pytest.raises(
        ValueError, match="Cannot combine results from different methods"
    ):
        splotch_result_nuts_batch_1 + splotch_result_svi_batch_2


def test_splotch_v1_invalid_number_of_levels(
    splotchinputdata_inference: SplotchInputData,
) -> None:
    """Test splotch_v1() with invalid number of levels."""
    genes = ["g1"]
    model_kwargs = get_splotch_kwargs(
        splotchinputdata_inference, get_default_priors(), use_zero_inflated=False
    ) | {"counts": jnp.asarray(splotchinputdata_inference.counts(genes))}
    model_kwargs["num_levels"] = 4

    with pytest.raises(ValueError, match="Only 1, 2, or 3 levels are supported"):
        splotch_v1(**model_kwargs)  # type: ignore[arg-type]


def test_visualization(
    splotchinputdata_inference_wout_images: SplotchInputData,
) -> None:
    """Test visualization routines."""
    assert isinstance(
        plot_annotations_on_slides(splotchinputdata_inference_wout_images), Figure
    )
    plot_tissue_sections_on_slides(splotchinputdata_inference_wout_images)

    key = random.PRNGKey(0)
    key, key_ = random.split(key, 2)
    register(key_, splotchinputdata_inference_wout_images, num_steps=10_000)

    assert isinstance(
        plot_annotations_in_common_coordinate_system(
            splotchinputdata_inference_wout_images
        ),
        Figure,
    )

    gene = "g1"
    splotch_result_nuts = run_nuts(
        key_,
        [gene],
        splotchinputdata_inference_wout_images,
        map_method="map",
        num_warmup=10,
        num_samples=10,
        num_chains=4,
    )

    assert isinstance(plot_rates_on_slides(splotch_result_nuts, gene), Figure)
    assert isinstance(plot_variable_on_slides(splotch_result_nuts, gene, "f"), Figure)
    assert isinstance(
        plot_coefficients(
            splotchinputdata_inference_wout_images, splotch_result_nuts, gene
        ),
        Figure,
    )
    assert isinstance(
        plot_rates_in_common_coordinate_system(splotch_result_nuts, gene), Figure
    )
