"""dataclasses.py."""
import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from jax.tree_util import tree_map

logger = logging.getLogger(__name__)

NUM_DIMENSIONS = 2


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


@dataclass
class SpotData:
    """Spot data."""

    genes: list[str]
    level_values: pd.Series
    metadata: pd.Series
    coordinates_orig: np.ndarray  # type: ignore[type-arg]
    spot_counts: np.ndarray  # type: ignore[type-arg]
    annotations_df: pd.DataFrame

    def __post_init__(self) -> None:
        """Post init steps."""
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

    def select(self, indices: np.ndarray) -> "SpotData":
        """Return object with selected values."""
        return SpotData(
            self.genes,
            self.level_values,
            self.metadata,
            self.coordinates_orig[indices],
            self.spot_counts[indices, :],
            self.annotations_df.loc[:, indices],
        )

    def __len__(self) -> int:
        """Get the number of spots."""
        return len(self.coordinates_orig)
