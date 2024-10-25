"""dataclasses.py."""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
from jax import Array
from jax.tree_util import tree_map

logger = logging.getLogger(__name__)

NUM_DIMENSIONS = 2


@dataclass
class SplotchInputData:
    """Splotch input data.

    Used to store Splotch input data.

    Args:
        count_data: Counts dataframe.
        annotation_data: Annotation dataframe.
        metadata: Metadata dataframe.
    """

    count_data: pd.DataFrame
    annotation_data: pd.DataFrame
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        """Initialize data dependent fields."""
        self.metadata["aar"] = self.annotation_data.columns[self.annotations()]

    def num_levels(self) -> int:
        """Get number of levels.

        Returns:
            Number of levels.
        """
        return len([col for col in self.metadata.columns if col.startswith("level_")])

    def num_categories_per_level(self) -> dict[str, int]:
        """Get number of categories per level.

        Returns:
            Number of categories per level.
        """
        return {
            col: self.metadata[col].cat.codes.max() + 1
            for col in self.metadata.columns
            if col.startswith("level_")
        }

    def num_aars(self) -> int:
        """Get number of AARs.

        Returns:
            Number of AARs.
        """
        return self.annotation_data.shape[1]

    def num_spots(self) -> int:
        """Get number of spots.

        Returns:
            Number of spots.
        """
        return self.count_data.shape[0]

    def counts(self, genes: list[str] | None = None) -> npt.NDArray[np.int64]:
        """Get counts matrix.

        Args:
            genes: List of genes of interest. If none, then consider all genes.

        Returns:
            Count matrix.
        """
        return (
            self.count_data.to_numpy()
            if genes is None
            else self.count_data.loc[:, genes].to_numpy()
        )

    def annotations(self) -> npt.NDArray[np.int64]:
        """Get annotations.

        Returns:
            Annotations.
        """
        return np.argmax(self.annotation_data.to_numpy(), axis=1)

    def levels(self) -> npt.NDArray[np.int8]:
        """Get levels.

        Returns:
            Levels.
        """
        return np.hstack(
            [
                self.metadata[f"level_{idx+1}"].cat.codes.to_numpy()[:, None]
                for idx in range(self.num_levels())
            ]
        )

    def aars(self) -> npt.NDArray[np.bytes_]:
        """Get AARs.

        Returns:
            AARs.
        """
        return self.annotation_data.columns.to_numpy()

    def size_factors(self) -> npt.NDArray[np.float64]:
        """Get size factors.

        Returns:
            Size factors.
        """
        return self.metadata.size_factor.to_numpy()


@dataclass
class SplotchResult:
    """Splotch result.

    Used to store results from `splotch.inference.run_nuts()` and `splotch.inference.run_svi()`

    Args:
        metadata: Metadata dataframe.
        genes: Genes.
        inference_metrics: Inference metrics corresponding to the genes.
        posterior_samples: Posterior samples corresponding to the genes.
    """

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
        """Combine two SplotchResult objects.

        Raises:
            ValueError: Mismatching metadata.
            ValueError: Genes overlap.
            ValueError: Mismatching inference methods.

        Returns:
            Combined SplotchResult.
        """
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


@dataclass
class SpotData:
    """Spot data.

    Used to group information across genes and spots.

    Args:
        genes: Genes.
        level_values: Level category values.
        metadata: Metadata dataframe.
        coordinates_orig: Coordinates as strings.
        spot_counts: Spot counts.
        annotations_df: Annotation dataframe.
    """

    genes: list[str]
    level_values: pd.Series
    metadata: pd.Series
    coordinates_orig: npt.NDArray[np.bytes_]
    spot_counts: npt.NDArray[np.int64]
    annotations_df: pd.DataFrame

    def __post_init__(self) -> None:
        """Post init steps.

        Raises:
            ValueError: Invalid coordinates.
            ValueError: Number of genes do not match.
            ValueError: Number of coordinates do not match.
        """
        self.coordinates = np.asarray(
            [
                [float(val) for val in coordinate.split("_")]
                for coordinate in self.coordinates_orig
                if len(coordinate.split("_")) == NUM_DIMENSIONS
            ]
        )
        if len(self.coordinates) != len(self.coordinates_orig):
            msg = "Unable not find x and y coordinates. Ensure the naming pattern is 'x_y'."
            raise ValueError(msg)
        if len(self.genes) != self.spot_counts.shape[1]:
            msg = "Number of genes do not match."
            raise ValueError(msg)
        if len(self.coordinates_orig) != self.spot_counts.shape[0]:
            msg = "Number of coordinates do not match."
            raise ValueError(msg)

        self.total_counts = np.sum(self.spot_counts, axis=1)

        logger.info("Start with %d spots", len(self.coordinates_orig))
        indices = np.asarray(
            [
                not (
                    coordinate not in self.annotations_df.columns
                    or self.annotations_df[coordinate].sum() != 1
                )
                for coordinate in self.coordinates_orig
            ]
        )
        if sum(~indices) > 0:
            logger.info("Discard %d spots due to annotation issues.", sum(~indices))
        self.coordinates_orig = self.coordinates_orig[indices]
        self.annotations_df = self.annotations_df[self.coordinates_orig]
        self.coordinates = self.coordinates[indices, :]
        self.spot_counts = self.spot_counts[indices, :]
        self.total_counts = self.total_counts[indices]

    def select(self, indices: npt.NDArray[np.bool]) -> "SpotData":
        """Return object with selected values.

        Returns:
            Selected SpotData.
        """
        return SpotData(
            self.genes,
            self.level_values,
            self.metadata,
            self.coordinates_orig[indices],
            self.spot_counts[indices, :],
            self.annotations_df.loc[:, indices],
        )

    def __len__(self) -> int:
        """Get the number of spots.

        Returns:
            Number of spots.
        """
        return len(self.coordinates_orig)
