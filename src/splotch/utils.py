"""utils.py."""
import functools
import logging
import operator
from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st
from jax import Array
from jax.tree_util import tree_map
from numpyro.diagnostics import summary
from scipy.spatial import distance_matrix

# ruff: noqa: PLR2004


@dataclass
class SplotchInputData:
    """Splotch input data."""

    count_data: pd.DataFrame
    annotation_data: pd.DataFrame
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        """Initialize data dependent fields."""
        self.metadata["aar"] = self.annotation_data.columns[self.annotations()]

    def num_levels(self) -> int:
        """Get number of levels."""
        return len([col for col in self.metadata.columns if col.startswith("level_")])

    def num_categories_per_level(self) -> dict[str, int]:
        """Get number of categories per level."""
        return {
            col: self.metadata[col].cat.codes.max() + 1
            for col in self.metadata.columns
            if col.startswith("level_")
        }

    def num_aars(self) -> int:
        """Get number of AARs."""
        return self.annotation_data.shape[1]  # type: ignore[no-any-return]

    def num_spots(self) -> int:
        """Get number of spots."""
        return self.count_data.shape[0]  # type: ignore[no-any-return]

    def counts(self, genes: list[str] | None = None) -> np.ndarray:
        """Get counts matrix.

        Args:
            genes: TBA.

        Returns:
            Count matrix.
        """
        return (  # type: ignore[no-any-return]
            self.count_data.values
            if genes is None
            else self.count_data.loc[:, genes].values
        )

    def annotations(self) -> np.ndarray:
        """Get annotations."""
        return np.argmax(self.annotation_data.values, axis=1)  # type: ignore[no-any-return]

    def levels(self) -> np.ndarray:
        """Get levels."""
        return np.hstack(
            [
                self.metadata[f"level_{idx+1}"].cat.codes.values[:, None]
                for idx in range(self.num_levels())
            ]
        )

    def aars(self) -> np.ndarray:
        """Get AARs."""
        return self.annotation_data.columns.values  # type: ignore[no-any-return]

    def size_factors(self) -> np.ndarray:
        """Get size factors."""
        return self.metadata.size_factor.values  # type: ignore[no-any-return]


@dataclass
class SplotchResult:
    """Splotch result."""

    metadata: pd.DataFrame
    genes: list[str]
    inference_metrics: dict[str, Any]
    posterior_samples: dict[str, Array]

    def __post_init__(self) -> None:
        """Initialize data dependent fields."""
        self.rates = pd.DataFrame(
            jnp.reshape(
                jnp.moveaxis(self.posterior_samples["lambda"], 2, 1),
                (
                    self.posterior_samples["lambda"].shape[0]
                    * self.posterior_samples["lambda"].shape[2],
                    self.posterior_samples["lambda"].shape[1],
                ),
            ),
            index=pd.MultiIndex.from_tuples(
                ((gene, *item) for gene in self.genes for item in self.metadata.index),
                names=("gene", *self.metadata.index.names),
            ),
        )

    def __add__(self, other: "SplotchResult") -> "SplotchResult":
        """Combine two SplotchResult objects."""
        if not self.metadata.equals(other.metadata):
            msg = "Metadata are not the same"
            raise ValueError(msg)
        if len(set(self.genes) & set(other.genes)) > 0:
            msg = "Genes overlap"
            raise ValueError(msg)
        if self.inference_metrics.keys() != other.inference_metrics.keys():
            msg = "Cannot combine results from different methods"
            raise ValueError(msg)
        # SVI
        if "losses" in self.inference_metrics:
            return SplotchResult(
                self.metadata,
                self.genes + other.genes,
                tree_map(
                    lambda *x: jnp.concatenate(x),
                    *(self.inference_metrics, other.inference_metrics),
                ),
                tree_map(
                    lambda *x: jnp.vstack(x),
                    *(self.posterior_samples, other.posterior_samples),
                ),
            )
        # NUTS
        return SplotchResult(
            self.metadata,
            self.genes + other.genes,
            tree_map(
                lambda *x: pd.concat(x, axis=0),
                *(self.inference_metrics, other.inference_metrics),
            ),
            tree_map(
                lambda *x: jnp.vstack(x),
                *(self.posterior_samples, other.posterior_samples),
            ),
        )


def read_count_files(
    count_files: list[str], min_detection_rate: float = 0.02
) -> pd.DataFrame:
    """Read count files.

    Args:
        count_files: Count file names.
        min_detection_rate: Minimum detection rate.
            Discard any gene that has lower detection rate.

    Returns:
        Dataframe containing counts.
    """
    logging.info("Reading %d count files", len(count_files))
    counts_dfs = [
        pd.read_table(filename, header=0, index_col=0) for filename in count_files
    ]

    for filename, count_df in zip(count_files, counts_dfs):
        count_df.columns = pd.MultiIndex.from_product(
            [[filename], count_df.columns], names=["file", "coordinate"]
        )
        count_df.index.name = "gene"

    counts_df = pd.concat(counts_dfs, copy=False, axis=1, sort=True)
    logging.info("We have detected unique %d genes", counts_df.shape[0])
    counts_df = counts_df.fillna(0).astype(int)

    counts_df = counts_df[
        ((counts_df > 0).sum(axis=1) / counts_df.shape[1]) > min_detection_rate
    ]
    logging.info(
        ("We have unique %d genes with sufficient detection rate (>= %f)"),
        counts_df.shape[0],
        min_detection_rate,
    )

    logging.info(
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
    logging.info("Reading %d annotation files", len(annotation_files))
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


def process_array(
    counts_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process array.

    Args:
        counts_df: TBA.

    Returns:
        TBA.
    """
    coordinates_orig = np.asarray(list(counts_df.columns))
    coordinates = np.asarray(
        [
            [float(val) for val in coordinate.split("_")]
            for coordinate in coordinates_orig
        ]
    )
    spot_counts = counts_df.values.T
    total_counts = np.sum(spot_counts, axis=1)
    return coordinates_orig, coordinates, spot_counts, total_counts


def filter_spots(
    indices: np.ndarray,
    coordinates_orig: np.ndarray,
    coordinates: np.ndarray,
    spot_counts: np.ndarray,
    total_counts: np.ndarray,
    array_annotations_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Filter spot data.

    Args:
        indices: TBA.
        coordinates_orig: TBA.
        coordinates: TBA.
        spot_counts: TBA.
        total_counts: TBA.
        array_annotations_df: TBA.

    Returns:
        TBA.
    """
    coordinates_orig = coordinates_orig[indices]
    coordinates = coordinates[indices, :]
    spot_counts = spot_counts[indices, :]
    total_counts = total_counts[indices]
    array_annotations_df = array_annotations_df.loc[:, indices]
    return (
        coordinates_orig,
        coordinates,
        spot_counts,
        total_counts,
        array_annotations_df,
    )


def process_input_data(
    metadata_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    num_levels: int,
    min_total_count: int = 100,
    min_num_spots: int = 10,
    number_of_neighbors: int = 4,
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
        min_total_count: TBA.
        min_num_spots: TBA.
        number_of_neighbors: TBA.
        separate_overlapping_tissue_sections: TBA.
        max_num_spots_per_tissue_section: TBA.
        min_num_spots_per_tissue_section: TBA.
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

    for count_file, annotation_file in zip(
        counts_df.columns.unique(level="file"),
        annotations_df.columns.unique(level="file"),
    ):
        logging.info("Processing %s", count_file)
        coordinates_orig, coordinates, spot_counts, total_counts = process_array(
            counts_df[count_file]
        )

        array_level_values = metadata_df[metadata_df.count_file == count_file][
            [f"level_{level_idx+1}" for level_idx in range(num_levels)]
        ].iloc[0]

        array_metadata = metadata_df[metadata_df.count_file == count_file][
            ["annotation_file", "image_file"]
        ].iloc[0]

        array_annotations_df = annotations_df[annotation_file]

        # valid spot has enough UMIs, it is annotated, and it belongs to one aar
        valid_spots = np.asarray(
            [
                not (
                    total_count < min_total_count
                    or coordinate not in array_annotations_df.columns
                    or array_annotations_df[coordinate].sum() != 1
                )
                for coordinate, total_count in zip(coordinates_orig, total_counts)
            ]
        )

        if sum(valid_spots) < min_num_spots:
            logging.warning(
                "%s has less than %d valid spots. Maybe coordinates do not match in count files and annotations.",
                count_file,
                min_num_spots,
            )

        # subset to the valid spots
        (
            coordinates_orig,
            coordinates,
            spot_counts,
            total_counts,
            array_annotations_df,
        ) = filter_spots(
            valid_spots,
            coordinates_orig,
            coordinates,
            spot_counts,
            total_counts,
            array_annotations_df,
        )

        logging.info("Detecting distinct tissue sections on the slide")
        tissue_sections = detect_tissue_sections(
            coordinates,
            number_of_neighbors,
        )

        if separate_overlapping_tissue_sections:
            logging.info("Separating tissue sections on the slide")
            tissue_sections = separate_tissue_sections(
                coordinates,
                tissue_sections,
                number_of_neighbors,
                max_num_spots_per_tissue_section,
                seed,
            )

        tissue_section_labels = np.asarray(
            [
                [idx in tissue_section for tissue_section in tissue_sections].index(
                    True
                )
                for idx in range(len(coordinates))
            ]
        )

        for tissue_section_label in np.unique(tissue_section_labels):
            # find the indices of the spots of the current tissue section
            tissue_section_indices = tissue_section_labels == tissue_section_label

            # discard those tissue sections that are unexpectedly small
            if sum(tissue_section_indices) <= min_num_spots_per_tissue_section:
                logging.warning(
                    "Discarding a tissue section with %d spots",
                    sum(tissue_section_indices),
                )
                continue

            # subset to the spots of the current tissue section
            (
                tissue_section_coordinates_orig,
                tissue_section_coordinates,
                tissue_section_spot_counts,
                tissue_section_total_counts,
                tissue_section_annotations_df,
            ) = filter_spots(
                tissue_section_indices,
                coordinates_orig,
                coordinates,
                spot_counts,
                total_counts,
                array_annotations_df,
            )

            spot_multiindex = pd.MultiIndex.from_arrays(
                [
                    [count_file] * np.sum(tissue_section_indices),
                    [tissue_section_idx] * np.sum(tissue_section_indices),
                    tissue_section_coordinates_orig,
                ],
                names=("count_file", "tissue_section", "coordinate"),
            )

            tissue_section_annotations_df = tissue_section_annotations_df.T
            tissue_section_annotations_df.index = spot_multiindex

            size_factor_dfs.append(
                pd.DataFrame(
                    tissue_section_total_counts / median_spot_umi_count,
                    index=spot_multiindex,
                    columns=("size_factor",),
                )
            )
            level_dfs.append(
                pd.DataFrame(
                    np.tile(
                        array_level_values.values, (np.sum(tissue_section_indices), 1)
                    ),
                    index=spot_multiindex,
                    columns=array_level_values.index,
                )
            )
            metadata_dfs.append(
                pd.DataFrame(
                    np.tile(array_metadata.values, (np.sum(tissue_section_indices), 1)),
                    index=spot_multiindex,
                    columns=array_metadata.index,
                )
            )
            count_dfs.append(
                pd.DataFrame(
                    tissue_section_spot_counts, index=spot_multiindex, columns=genes
                )
            )
            annotation_dfs.append(tissue_section_annotations_df)
            coordinate_dfs.append(
                pd.DataFrame(
                    tissue_section_coordinates,
                    index=spot_multiindex,
                    columns=("x", "y"),
                )
            )

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
    min_num_spots: int = 10,
    number_of_neighbors: int = 4,
    separate_overlapping_tissue_sections: bool = False,
    max_num_spots_per_tissue_section: int = 120,
    min_num_spots_per_tissue_section: int = 10,
    seed: int = 0,
) -> SplotchInputData:
    """Get Splotch input data.

    Args:
        metadata: TBA.
        num_levels: TBA.
        min_detection_rate: TBA.
        min_total_count: TBA.
        min_num_spots: TBA.
        separate_overlapping_tissue_sections: TBA.
        number_of_neighbors: TBA.
        max_num_spots_per_tissue_section: TBA.
        min_num_spots_per_tissue_section: TBA.
        seed: TBA.

    Returns:
        Splotch input data.
    """
    metadata_df = pd.read_csv(metadata, sep="\t")
    if num_levels not in {1, 2, 3}:
        msg = "num_levels has to be 1, 2, or 3"
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
        min_num_spots,
        number_of_neighbors,
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
    coordinates: np.ndarray, number_of_neighbors: int
) -> np.ndarray:
    """Get spot adjacency matrix.

    Args:
        coordinates: TBA.
        number_of_neighbors: TBA.

    Returns:
        TBA.
    """
    coordinates_distance_matrix = distance_matrix(coordinates, coordinates)
    threshold = np.min(
        np.sort(coordinates_distance_matrix, axis=0)[number_of_neighbors + 1, :]
    )
    return np.logical_and(  # type: ignore[no-any-return]
        coordinates_distance_matrix < threshold, coordinates_distance_matrix > 0
    )


def detect_tissue_sections(
    coordinates: np.ndarray,
    number_of_neighbors: int,
) -> list[set[int]]:
    """Detect tissue sections.

    Args:
        coordinates: Coordinates of the spots on the slide.
        number_of_neighbors: TBA.

    Returns:
        TBA.
    """
    adjacency_matrix = get_spot_adjacency_matrix(coordinates, number_of_neighbors)
    spot_graph = nx.from_numpy_array(adjacency_matrix)

    tissue_sections = list(
        nx.algorithms.components.connected.connected_components(spot_graph)
    )

    logging.info("Found %d candidate tissue sections", len(tissue_sections))

    return tissue_sections


def separate_tissue_sections(
    coordinates: np.ndarray,
    tissue_sections: list[set[int]],
    number_of_neighbors: int,
    max_num_spots_per_tissue_section: int,
    seed: int,
) -> list[set[int]]:
    """Separate tissue sections.

    Args:
        coordinates: TBA.
        tissue_sections: TBA.
        number_of_neighbors: TBA.
        max_num_spots_per_tissue_section: TBA.
        seed: TBA.

    Return:
        TBA.
    """
    adjacency_matrix = get_spot_adjacency_matrix(coordinates, number_of_neighbors)

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

    logging.info(
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
    logging.warning(
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
