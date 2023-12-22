"""utils.py."""
import functools
import logging
import operator
from typing import Any, Mapping

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st
from jax import Array
from numpyro.diagnostics import summary
from scipy.spatial import distance_matrix

from splotch.dataclasses import SplotchInputData, SpotData

logger = logging.getLogger(__name__)

# ruff: noqa: PLR2004, PLR0913, PLR0914, PLR0917


def read_count_files(count_files: list[str], min_detection_rate: float) -> pd.DataFrame:
    """Read count files.

    Args:
        count_files: Count file names.
        min_detection_rate: Minimum detection rate.
            Discard any gene that has lower detection rate.

    Returns:
        Dataframe containing counts.
    """
    logger.info("Reading %d count files", len(count_files))
    counts_dfs = [
        pd.read_table(filename, header=0, index_col=0) for filename in count_files
    ]

    for filename, count_df in zip(count_files, counts_dfs):
        count_df.columns = pd.MultiIndex.from_product(
            [[filename], count_df.columns], names=["file", "coordinate"]
        )
        count_df.index.name = "gene"

    counts_df = pd.concat(counts_dfs, copy=False, axis=1, sort=True)
    logger.info("We have detected unique %d genes", counts_df.shape[0])
    counts_df = counts_df.fillna(0).astype(int)

    counts_df = counts_df[
        ((counts_df > 0).sum(axis=1) / counts_df.shape[1]) >= min_detection_rate
    ]
    logger.info(
        ("We have unique %d genes with sufficient detection rate (>= %f)"),
        counts_df.shape[0],
        min_detection_rate,
    )

    logger.info(
        "The median sequencing depth across the spots is %d",
        np.median(counts_df.sum(0)),
    )
    return counts_df


def read_annotation_files(annotation_files: list[str]) -> pd.DataFrame:
    """Read annotation files.

    Args:
        annotation_files: Annotation file names.

    Returns:
        Dataframe containing annotations.
    """
    logger.info("Reading %d annotation files", len(annotation_files))
    annotation_dfs = [
        pd.read_table(filename, header=0, index_col=0) for filename in annotation_files
    ]

    for filename, annotation_df in zip(annotation_files, annotation_dfs):
        annotation_df.columns = pd.MultiIndex.from_product(
            [[filename], annotation_df.columns], names=["file", "coordinate"]
        )
        annotation_df.index.name = "aar"

    annotation_df = pd.concat(annotation_dfs, copy=False, axis=1, sort=True)

    if not np.all(np.isin(annotation_df.values, [0, 1])):
        msg = "All annotation values should be 0 or 1"
        raise ValueError(msg)
    if not np.all(np.isin(annotation_df.values.sum(0), [0, 1])):
        msg = "Each spot should have zero or one active category"
        raise ValueError(msg)
    return annotation_df


def get_tissue_section_output_data(
    tissue_section_spot_data: SpotData,
    tissue_section_idx: int,
    count_file: str,
    median_spot_umi_count: int,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """TBA.

    Args:
        tissue_section_spot_data: TBA.
        tissue_section_idx: TBA.
        count_file: TBA.
        median_spot_umi_count: TBA.
    """
    spot_multiindex = pd.MultiIndex.from_arrays(
        [
            [count_file] * len(tissue_section_spot_data),
            [tissue_section_idx] * len(tissue_section_spot_data),
            tissue_section_spot_data.coordinates_orig,
        ],
        names=("count_file", "tissue_section", "coordinate"),
    )

    tissue_section_annotations_df = tissue_section_spot_data.annotations_df.T
    tissue_section_annotations_df.index = spot_multiindex

    size_factor_df = pd.DataFrame(
        tissue_section_spot_data.total_counts / median_spot_umi_count,
        index=spot_multiindex,
        columns=("size_factor",),
    )
    level_df = pd.DataFrame(
        np.tile(
            tissue_section_spot_data.level_values.values,
            (len(tissue_section_spot_data), 1),
        ),
        index=spot_multiindex,
        columns=tissue_section_spot_data.level_values.index,
    )
    metadata_df = pd.DataFrame(
        np.tile(
            tissue_section_spot_data.metadata.values, (len(tissue_section_spot_data), 1)
        ),
        index=spot_multiindex,
        columns=tissue_section_spot_data.metadata.index,
    )
    count_df = pd.DataFrame(
        tissue_section_spot_data.spot_counts,
        index=spot_multiindex,
        columns=tissue_section_spot_data.genes,
    )
    annotation_df = tissue_section_annotations_df
    coordinate_df = pd.DataFrame(
        tissue_section_spot_data.coordinates,
        index=spot_multiindex,
        columns=("x", "y"),
    )

    return size_factor_df, level_df, metadata_df, count_df, annotation_df, coordinate_df


def process_input_data(
    metadata_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    num_levels: int,
    min_total_count: int = 100,
    min_num_spots_per_slide: int = 10,
    num_of_neighbors: int = 8,
    separate_overlapping_tissue_sections: bool = True,
    max_num_spots_per_tissue_section: int = 120,
    min_num_spots_per_tissue_section: int = 10,
    seed: int = 0,
) -> SplotchInputData:
    """Process input data.

    Args:
        metadata_df: TBA.
        counts_df: TBA.
        annotations_df: TBA.
        num_levels: TBA.
        min_total_count: TBA. Defaults to 100.
        min_num_spots_per_slide: TBA. Defaults to 10.
        num_of_neighbors: TBA. Defaults to 8.
        separate_overlapping_tissue_sections: TBA. Defaults to True.
        max_num_spots_per_tissue_section: TBA. Defaults to 120.
        min_num_spots_per_tissue_section: TBA. Defaults to 10.
        seed: TBA.

    Returns:
        Splotch input data.
    """
    genes = list(counts_df.index)

    median_spot_umi_count = counts_df.sum(0).median()

    size_factor_dfs = []
    level_dfs = []
    metadata_dfs = []
    count_dfs = []
    annotation_dfs = []
    coordinate_dfs = []

    tissue_section_idx = 0

    for count_file in counts_df.columns.unique(level="file"):
        logger.info("Processing %s", count_file)

        coordinates_orig = np.asarray(list(counts_df[count_file].columns))
        spot_counts = counts_df[count_file].values.T

        annotation_file = metadata_df[metadata_df.count_file == count_file][
            "annotation_file"
        ].iloc[0]
        array_annotations_df = annotations_df[annotation_file]

        level_values = metadata_df[metadata_df.count_file == count_file][
            [f"level_{level_idx+1}" for level_idx in range(num_levels)]
        ].iloc[0]

        metadata = metadata_df[metadata_df.count_file == count_file][
            ["annotation_file", "image_file"]
        ].iloc[0]

        count_file_spot_data = SpotData(
            genes,
            level_values,
            metadata,
            coordinates_orig,
            spot_counts,
            array_annotations_df,
        )

        indices = count_file_spot_data.total_counts >= min_total_count

        if sum(indices) < min_num_spots_per_slide:
            logger.warning(
                "%s has less than %d valid spots. Maybe coordinates do not match in the count and annotation files.",
                count_file,
                sum(indices),
            )

        if sum(indices) == 0:
            logger.warning(
                "%s has zero valid spots. Skip the count file.",
                count_file,
            )
            continue

        if np.sum(~indices) > 0:
            logger.warning(
                "Discarding %d spots due to low sequencing depth.",
                np.sum(~indices),
            )
        count_file_spot_data = count_file_spot_data.select(indices)

        logger.info("Detecting distinct tissue sections on the slide")
        tissue_sections = detect_tissue_sections(
            count_file_spot_data.coordinates,
            num_of_neighbors,
        )

        if separate_overlapping_tissue_sections:
            logger.info("Separating tissue sections on the slide")
            tissue_sections = separate_tissue_sections(
                count_file_spot_data.coordinates,
                tissue_sections,
                num_of_neighbors,
                max_num_spots_per_tissue_section,
                seed,
            )

        tissue_section_labels = np.asarray(
            [
                [idx in tissue_section for tissue_section in tissue_sections].index(
                    True
                )
                for idx in range(len(count_file_spot_data))
            ]
        )

        for tissue_section_label in np.unique(tissue_section_labels):
            # find the indices of the spots of the current tissue section
            tissue_section_indices = tissue_section_labels == tissue_section_label

            # discard those tissue sections that are unexpectedly small
            if sum(tissue_section_indices) < min_num_spots_per_tissue_section:
                logger.warning(
                    "Discarding a tissue section with %d spots",
                    sum(tissue_section_indices),
                )
                continue

            # subset to the spots of the current tissue section
            tissue_section_spot_data = count_file_spot_data.select(
                tissue_section_indices
            )

            (
                tissue_section_size_factor_df,
                tissue_section_level_df,
                tissue_section_metadata_df,
                tissue_section_count_df,
                tissue_section_annotation_df,
                tissue_section_coordinate_df,
            ) = get_tissue_section_output_data(
                tissue_section_spot_data,
                tissue_section_idx,
                count_file,
                median_spot_umi_count,
            )
            size_factor_dfs.append(tissue_section_size_factor_df)
            level_dfs.append(tissue_section_level_df)
            metadata_dfs.append(tissue_section_metadata_df)
            count_dfs.append(tissue_section_count_df)
            annotation_dfs.append(tissue_section_annotation_df)
            coordinate_dfs.append(tissue_section_coordinate_df)

            tissue_section_idx += 1

    level_df = pd.concat(level_dfs, axis=0)
    for level in ["level_1", "level_2", "level_3"][0:num_levels]:
        level_df[level] = level_df[level].astype("category")

    return SplotchInputData(
        pd.concat(count_dfs, axis=0),
        pd.concat(annotation_dfs, axis=0),
        pd.concat(
            [
                level_df,
                pd.concat(metadata_dfs, axis=0),
                pd.concat(coordinate_dfs, axis=0),
                pd.concat(size_factor_dfs, axis=0),
            ],
            axis=1,
        ),
    )


def get_input_data(
    metadata: str,
    num_levels: int,
    min_detection_rate: float = 0.02,
    min_total_count: int = 100,
    min_num_spots_per_slide: int = 10,
    num_of_neighbors: int = 8,
    separate_overlapping_tissue_sections: bool = False,
    max_num_spots_per_tissue_section: int = 120,
    min_num_spots_per_tissue_section: int = 10,
    seed: int = 0,
) -> SplotchInputData:
    """Get Splotch input data.

    Args:
        metadata: TBA.
        num_levels: TBA.
        min_detection_rate: TBA. Defaults to 0.02.
        min_total_count: TBA. Defaults to 100.
        min_num_spots_per_slide: TBA. Defaults to 10.
        num_of_neighbors: TBA. Defaults to 8.
        separate_overlapping_tissue_sections: TBA. Defaults to False.
        max_num_spots_per_tissue_section: TBA. Defaults to 120.
        min_num_spots_per_tissue_section: TBA. Defaults to 10.
        seed: TBA. Defaults to 0.

    Returns:
        Splotch input data.
    """
    metadata_df = pd.read_csv(metadata, sep="\t")
    if num_levels not in {1, 2, 3}:
        msg = "num_levels has to be 1, 2, or 3"
        raise ValueError(msg)

    if len(metadata_df.count_file) != len(set(metadata_df.count_file)):
        msg = "count_file values are not unique"
        raise ValueError(msg)

    counts_df = read_count_files(
        metadata_df.count_file, min_detection_rate=min_detection_rate
    )
    annotations_df = read_annotation_files(metadata_df.annotation_file)

    return process_input_data(
        metadata_df,
        counts_df,
        annotations_df,
        num_levels,
        min_total_count,
        min_num_spots_per_slide,
        num_of_neighbors,
        separate_overlapping_tissue_sections,
        max_num_spots_per_tissue_section,
        min_num_spots_per_tissue_section,
        seed,
    )


def savagedickey(
    theta_1: np.ndarray,
    theta_2: np.ndarray,
    mu_1: float = 0.0,
    sigma_1: float = 2.0,
    mu_2: float = 0.0,
    sigma_2: float = 2.0,
) -> float:
    """Calculate the Savage-Dickey density ratio.

    Args:
      theta_1: Posteriors samples of theta_1.
      theta_2: Posterior samples of theta_2.
      mu_1: Mean of the prior of theta_1.
      sigma_1: Standard deviation of the prior of theta_1.
      mu_2: Mean of the prior of theta_2.
      sigma_2: Standard deviation of the prior of theta_2.

    Returns:
        Savage-Dickey density ratio value.
    """
    delta_theta = (theta_1[:, None] - theta_2).flatten()
    denominator: float = st.gaussian_kde(delta_theta, bw_method="scott").evaluate(0)[0]
    numerator: float = st.norm.pdf(
        0,
        loc=mu_1 - mu_2,
        scale=np.sqrt(np.square(sigma_1) + np.square(sigma_2)),
    )

    return numerator / denominator


def get_spot_adjacency_matrix(
    coordinates: np.ndarray, num_of_neighbors: int
) -> np.ndarray:
    """Get spot adjacency matrix.

    num_of_neighbors has to be between 1 and number of coordinates - 2.

    Args:
        coordinates: TBA.
        num_of_neighbors: TBA.

    Returns:
        TBA.

    Raises:
        ValueError if num_of_neighbors is outside of the reasonable range.
    """
    if num_of_neighbors < 1 or num_of_neighbors + 1 >= len(coordinates):
        msg = "num_of_neighbors has to be between 1 and len(coordinates) - 2."
        raise ValueError(msg)

    coordinates_distance_matrix = distance_matrix(coordinates, coordinates)
    threshold = np.min(
        np.sort(coordinates_distance_matrix, axis=0)[num_of_neighbors + 1, :]
    )
    return np.logical_and(  # type: ignore[no-any-return]
        coordinates_distance_matrix < threshold, coordinates_distance_matrix > 0
    )


def detect_tissue_sections(
    coordinates: np.ndarray,
    num_of_neighbors: int,
) -> list[set[int]]:
    """Detect tissue sections.

    Args:
        coordinates: Coordinates of the spots on the slide.
        num_of_neighbors: TBA.

    Returns:
        TBA.
    """
    adjacency_matrix = get_spot_adjacency_matrix(coordinates, num_of_neighbors)
    spot_graph = nx.from_numpy_array(adjacency_matrix)

    tissue_sections = list(
        nx.algorithms.components.connected.connected_components(spot_graph)
    )

    logger.info("Found %d candidate tissue sections", len(tissue_sections))

    return tissue_sections


def separate_tissue_sections(
    coordinates: np.ndarray,
    tissue_sections: list[set[int]],
    num_of_neighbors: int,
    max_num_spots_per_tissue_section: int,
    seed: int,
) -> list[set[int]]:
    """Separate tissue sections.

    Args:
        coordinates: TBA.
        tissue_sections: TBA.
        num_of_neighbors: TBA.
        max_num_spots_per_tissue_section: TBA.
        seed: TBA.

    Return:
        TBA.
    """
    adjacency_matrix = get_spot_adjacency_matrix(coordinates, num_of_neighbors)

    while (
        max(len(tissue_section) for tissue_section in tissue_sections)
        > max_num_spots_per_tissue_section
    ):
        tissue_sections = functools.reduce(
            operator.iconcat,
            [
                [tissue_section]
                if len(tissue_section) <= max_num_spots_per_tissue_section
                else separate_tissue_section(tissue_section, adjacency_matrix, seed)
                for tissue_section in tissue_sections
            ],
            [],
        )

    logger.info(
        "We have %d candidate tissue sections after the tissue section separation step",
        len(tissue_sections),
    )

    return tissue_sections


def separate_tissue_section(
    tissue_section: set[int],
    adjacency_matrix: np.ndarray,
    seed: int = 0,
) -> list[set[int]]:
    """Separate overlapping tissue sections.

    Args:
        tissue_section: TBA.
        adjacency_matrix: TBA.
        seed: TBA.

    Returns:
        TBA.
    """
    logger.warning(
        "Tissue section has %d spots. Let us try to break the tissue section into two.",
        len(tissue_section),
    )
    indices = np.asarray(sorted(tissue_section))
    spot_graph = nx.from_numpy_array(adjacency_matrix[np.ix_(indices, indices)])
    mapping = {idx: indices[idx] for idx in range(len(indices))}
    spot_graph = nx.relabel_nodes(spot_graph, mapping)
    return list(
        nx.algorithms.community.kernighan_lin.kernighan_lin_bisection(
            spot_graph, seed=seed
        )
    )


def get_mcmc_summary(posterior_samples: Array) -> pd.DataFrame:
    """Get MCMC summary DataFrame.

    Args:
        posterior_samples: Posterior samples.

    Returns:
        DataFrame.
    """

    def process_variable(data: Mapping[str, np.ndarray | np.float64]) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for statistic, values in data.items():
            if "index" not in res:
                if isinstance(values, np.ndarray):
                    res["index"] = [
                        tuple(map(int, x))
                        for x in zip(
                            *jnp.unravel_index(jnp.arange(values.size), values.shape)
                        )
                    ]
                else:
                    res["index"] = [None]
            if isinstance(values, np.ndarray):
                res[statistic] = values.flatten().tolist()
            else:
                res[statistic] = [values]
        return res

    return pd.concat(
        [
            pd.DataFrame.from_dict(process_variable(data))
            for data in summary(posterior_samples).values()
        ],
        axis=0,
        ignore_index=False,
    ).set_index("index")
