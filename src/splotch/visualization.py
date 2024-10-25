"""visualization.py."""

from collections.abc import Callable
from math import ceil
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from PIL import Image

from splotch.dataclasses import SplotchInputData, SplotchResult

# ruff: noqa: PLR0914


LOAD_SLIDE_IMAGE_CALLABLE: TypeAlias = Callable[
    [SplotchInputData | SplotchResult, str], Image.Image
]
CALCULATE_SLIDE_COORDINATES_CALLABLE: TypeAlias = Callable[
    [SplotchInputData | SplotchResult, Image.Image, str],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
]
LOAD_SLIDE_COORDINATES_AND_IMAGE_CALLABLE: TypeAlias = Callable[
    [
        SplotchInputData | SplotchResult,
        str,
        float,
        LOAD_SLIDE_IMAGE_CALLABLE,
        CALCULATE_SLIDE_COORDINATES_CALLABLE,
    ],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Image.Image],
]


def load_slide_image(  # pragma: no cover
    splotch_data: SplotchInputData | SplotchResult, count_file: str
) -> Image.Image:
    """Load slide image.

    Args:
        splotch_data: Splotch data.
        count_file: Count file of interest.

    Returns:
        Image.
    """
    if "count_file" in splotch_data.metadata:
        return Image.open(
            splotch_data.metadata.query("count_file == @count_file").image_file.iloc[0]
        )
    return Image.open(
        splotch_data.metadata[
            splotch_data.metadata.index.get_level_values("count_file") == count_file
        ].image_file.iloc[0]
    )


def calculate_st_slide_coordinates(  # pragma: no cover
    splotch_data: SplotchInputData | SplotchResult,
    slide_image: Image.Image,
    count_file: str,  # noqa: ARG001
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate ST slide coordinates corresponding to image.

    Tested only using the old ST slides.

    Args:
        splotch_data: Splotch data.
        slide_image: Slide image.
        count_file: Count file of interest.

    Returns:
        X coordinates of the spots and and y coordinates of the spots.
    """
    xdim, _ = slide_image.size
    pixel_dim = 194.0 / (6200.0 / xdim)

    x = pixel_dim * (splotch_data.metadata.query("count_file == @count_file").x - 1)
    y = pixel_dim * (splotch_data.metadata.query("count_file == @count_file").y - 1)

    return x, y


def load_slide_coordinates_and_image(  # pragma: no cover
    splotch_data: SplotchInputData | SplotchResult,
    count_file: str,
    scale_factor: float = 0.05,
    load_slide_image: Callable[
        [SplotchInputData | SplotchResult, str], Image.Image
    ] = load_slide_image,
    calculate_slide_coordinates: Callable[
        [SplotchInputData | SplotchResult, Image.Image, str],
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ] = calculate_st_slide_coordinates,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Image.Image]:
    """Get slide image and corresponding spot coordinates.

    Args:
        splotch_data: Splotch data.
        count_file: Count file of interest.
        scale_factor: Factor to scale the HE image. Defaults to 0.05.
        load_slide_image: Function to load slide image.
            Defaults to splotch.visualization.load_slide_image().
        calculate_slide_coordinates: Function to calculate coordinates on the image.
            Defaults to splotch.visualization.calculate_st_slide_coordinates().

    Returns:
        X coordinates of the spots, y coordinates of the spots, and the image.
    """
    slide_image = load_slide_image(splotch_data, count_file)

    xdim, ydim = slide_image.size

    # downsample the image
    slide_image = slide_image.resize(
        (
            np.round(xdim * scale_factor).astype(int),
            np.round(ydim * scale_factor).astype(int),
        )
    )

    x, y = calculate_slide_coordinates(splotch_data, slide_image, count_file)

    return x, y, slide_image


def plot_coefficients(
    splotch_input_data: SplotchInputData, splotch_result: SplotchResult, gene: str
) -> Figure:
    """Plot coefficients.

    Args:
        splotch_input_data: Splotch input data.
        splotch_result: Splotch result data.
        gene: Gene of interest.

    Returns:
        Matplotlib figure object.
    """
    data = splotch_result.posterior_samples["beta_level_1"][
        splotch_result.genes.index(gene)
    ]
    aars = splotch_input_data.aars()

    num_cols = 5
    num_rows = ceil(len(aars) / num_cols)

    fig = plt.figure()
    fig.set_size_inches(3 * num_cols, 2 * num_rows)

    for aar_idx in range(len(aars)):
        ax = fig.add_subplot(num_rows, num_cols, aar_idx + 1)
        ax.set_title(f"{gene}\n{aars[aar_idx]}")
        for idx in range(data.shape[1]):
            ax.hist(
                data[:, idx, aar_idx],
                30,
                alpha=0.4,
                label=splotch_result.metadata.level_1.cat.categories[idx],
            )
        if aar_idx == 0:
            ax.legend()

    fig.tight_layout()

    return fig


def plot_rates_on_slides(
    splotch_result: SplotchResult,
    gene: str,
    load_slide_coordinates_and_image: LOAD_SLIDE_COORDINATES_AND_IMAGE_CALLABLE = load_slide_coordinates_and_image,
) -> Figure:
    """Plot rate estimates on slides.

    Args:
        splotch_result: Splotch result data.
        gene: Gene of interest.
        load_slide_coordinates_and_image: Function to load slide images.
            Defaults to splotch.visualization.load_slide_coordinates_and_image().

    Returns:
        Matplotlib figure object.
    """
    rates_s = splotch_result.rates.loc[gene, :].mean(1)
    vmin, vmax = 0, np.percentile(rates_s, 95)

    count_files = splotch_result.metadata.index.get_level_values("count_file").unique()

    num_cols = 5
    num_rows = ceil(len(count_files) / num_cols)

    fig = plt.figure()
    fig.set_size_inches(3 * num_cols, 3 * num_rows)

    for ax_idx, count_file in enumerate(count_files, start=1):
        ax = fig.add_subplot(num_rows, num_cols, ax_idx)

        if "image_file" in splotch_result.metadata:  # pragma: no cover
            x, y, tissue_image = load_slide_coordinates_and_image(
                splotch_result, count_file
            )  # type: ignore[call-arg]

            ax.imshow(tissue_image, origin="upper", interpolation="none", alpha=0.6)
        else:
            x = splotch_result.metadata.query("count_file == @count_file").x
            y = splotch_result.metadata.query("count_file == @count_file").y

        c = rates_s[count_file]

        cb = ax.scatter(
            x, y, s=4, c=c, cmap="viridis", vmin=vmin, vmax=vmax, marker="o", alpha=0.9
        )

        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            f"{count_file}\n"
            f"{splotch_result.metadata.query('count_file == @count_file').level_1.iloc[0]}"
        )

        cbar = plt.colorbar(cb, ax=ax, shrink=0.8)
        cbar.set_label(f"λ ({gene})")

    fig.tight_layout()

    return fig


def plot_variable_on_slides(
    splotch_result: SplotchResult,
    gene: str,
    variable: str = "f",
    load_slide_coordinates_and_image: LOAD_SLIDE_COORDINATES_AND_IMAGE_CALLABLE = load_slide_coordinates_and_image,
) -> Figure:
    """Plot variable of interest on slides.

    Variable has to be scalar per spot.

    Args:
        splotch_result: Splotch result data.
        gene: Gene of interest.
        variable: Variable of interest. Defaults to `f`.
        load_slide_coordinates_and_image: Function to load slide images.
            Defaults to splotch.visualization.load_slide_coordinates_and_image().

    Returns:
        Matplotlib figure object.
    """
    data = splotch_result.rates.loc[gene, :].mean(1)
    data = splotch_result.posterior_samples[variable][
        splotch_result.genes.index(gene)
    ].mean(0)
    vmin, vmax = np.percentile(data, 5), np.percentile(data, 95)

    count_files = splotch_result.metadata.index.get_level_values("count_file").unique()

    num_cols = 5
    num_rows = ceil(len(count_files) / num_cols)

    fig = plt.figure()
    fig.set_size_inches(3 * num_cols, 3 * num_rows)

    for ax_idx, count_file in enumerate(count_files, start=1):
        ax = fig.add_subplot(num_rows, num_cols, ax_idx)

        if "image_file" in splotch_result.metadata:  # pragma: no cover
            x, y, tissue_image = load_slide_coordinates_and_image(
                splotch_result, count_file
            )  # type: ignore[call-arg]

            ax.imshow(tissue_image, origin="upper", interpolation="none", alpha=0.6)
        else:
            x = splotch_result.metadata.query("count_file == @count_file").x
            y = splotch_result.metadata.query("count_file == @count_file").y

        c = data[
            splotch_result.metadata.index.get_level_values("count_file") == count_file
        ]

        cb = ax.scatter(
            x, y, s=4, c=c, cmap="RdBu_r", vmin=vmin, vmax=vmax, marker="o", alpha=0.9
        )

        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            f"{count_file}\n"
            f"{splotch_result.metadata.query('count_file == @count_file').level_1.iloc[0]}"
        )

        cbar = plt.colorbar(cb, ax=ax, shrink=0.8)
        cbar.set_label(f"{variable} ({gene})")

    fig.tight_layout()

    return fig


def plot_rates_in_common_coordinate_system(
    splotch_result: SplotchResult, gene: str
) -> Figure:
    """Plot rate estimates in common coordinate system.

    Before using this function, please run splotch.registration.register().

    Args:
        splotch_result: Splotch result data.
        gene: Gene of interest.

    Returns:
        Matplotlib figure object.
    """
    rates_s = splotch_result.rates.loc[gene, :].mean(1)
    vmin, vmax = 0, np.percentile(rates_s, 95)

    level_1_categories = splotch_result.metadata.level_1.unique()

    num_cols = 5
    num_rows = ceil(len(level_1_categories) / num_cols)

    fig = plt.figure()
    fig.set_size_inches(3 * num_cols, 3 * num_rows)

    for ax_idx, level_1_category in enumerate(level_1_categories, start=1):
        ax = fig.add_subplot(num_rows, num_cols, ax_idx)

        x = splotch_result.metadata.query("level_1 == @level_1_category").x_registration
        y = splotch_result.metadata.query("level_1 == @level_1_category").y_registration
        c = rates_s[splotch_result.metadata.query("level_1 == @level_1_category").index]

        cb = ax.scatter(
            x, y, s=4, c=c, cmap="viridis", vmin=vmin, vmax=vmax, marker="o", alpha=0.9
        )

        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f"{level_1_category}")

        cbar = plt.colorbar(cb, ax=ax, shrink=0.8)
        cbar.set_label(f"λ ({gene})")

    fig.tight_layout()

    return fig


def plot_annotations_in_common_coordinate_system(
    splotch_input_data: SplotchInputData,
) -> Figure:
    """Plot annotations in common coordinate system.

    Before using this function, please run splotch.registration.register().

    Args:
        splotch_input_data: Splotch input data.

    Returns:
        Matplotlib figure object.
    """
    fig = plt.figure()
    fig.set_size_inches(5, 5)

    ax = fig.add_subplot(1, 1, 1)

    aars = splotch_input_data.metadata.aar.unique()

    for aar in aars:
        ax.scatter(
            splotch_input_data.metadata.query("aar == @aar").x_registration,
            splotch_input_data.metadata.query("aar == @aar").y_registration,
            s=5,
            alpha=0.8,
            label=aar,
        )

    ax.legend()

    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title("AARs")

    fig.tight_layout()

    return fig


def plot_annotations_on_slides(
    splotch_input_data: SplotchInputData,
    load_slide_coordinates_and_image: LOAD_SLIDE_COORDINATES_AND_IMAGE_CALLABLE = load_slide_coordinates_and_image,
) -> Figure:
    """Plot annotations on slides.

    Args:
        splotch_input_data: Splotch input data.
        load_slide_coordinates_and_image: Function to load slide images.
            Defaults to splotch.visualization.load_slide_coordinates_and_image().

    Returns:
        Matplotlib figure object.
    """
    count_files = splotch_input_data.metadata.index.get_level_values(
        "count_file"
    ).unique()

    num_cols = 5
    num_rows = ceil(len(count_files) / num_cols)

    fig = plt.figure()
    fig.set_size_inches(3 * num_cols, 3 * num_rows)

    aars = list(splotch_input_data.annotation_data.columns)

    for ax_idx, count_file in enumerate(count_files, start=1):
        ax = fig.add_subplot(num_rows, num_cols, ax_idx)

        if "image_file" in splotch_input_data.metadata:  # pragma: no cover
            x, y, tissue_image = load_slide_coordinates_and_image(
                splotch_input_data, count_file
            )  # type: ignore[call-arg]

            ax.imshow(tissue_image, origin="upper", interpolation="none", alpha=0.6)
        else:
            x = splotch_input_data.metadata.query("count_file == @count_file").x
            y = splotch_input_data.metadata.query("count_file == @count_file").y

        count_file_annotations = splotch_input_data.metadata.aar[
            splotch_input_data.metadata.index.get_level_values("count_file")
            == count_file
        ]

        for aar in aars:
            ax.scatter(
                x[count_file_annotations == aar],
                y[count_file_annotations == aar],
                s=5,
                alpha=0.8,
                label=aar,
            )

        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            f"{count_file}\n"
            f"{splotch_input_data.metadata.query('count_file == @count_file').level_1.iloc[0]}"
        )

        if ax_idx == 1:
            ax.legend()

    fig.tight_layout()

    return fig


def plot_tissue_sections_on_slides(
    splotch_input_data: SplotchInputData,
    load_slide_coordinates_and_image: LOAD_SLIDE_COORDINATES_AND_IMAGE_CALLABLE = load_slide_coordinates_and_image,
) -> Figure:
    """Plot detected tissue sections on slides.

    Args:
        splotch_input_data: Splotch input data.
        load_slide_coordinates_and_image: Function to load slide images.
            Defaults to splotch.visualization.load_slide_coordinates_and_image().

    Returns:
        Matplotlib figure object.
    """
    count_files = splotch_input_data.metadata.index.get_level_values(
        "count_file"
    ).unique()

    num_cols = 5
    num_rows = ceil(len(count_files) / num_cols)

    fig = plt.figure()
    fig.set_size_inches(3 * num_cols, 3 * num_rows)

    for ax_idx, count_file in enumerate(count_files, start=1):
        ax = fig.add_subplot(num_rows, num_cols, ax_idx)

        if "image_file" in splotch_input_data.metadata:  # pragma: no cover
            x, y, tissue_image = load_slide_coordinates_and_image(
                splotch_input_data, count_file
            )  # type: ignore[call-arg]

            ax.imshow(tissue_image, origin="upper", interpolation="none", alpha=0.6)
        else:
            x = splotch_input_data.metadata.query("count_file == @count_file").x
            y = splotch_input_data.metadata.query("count_file == @count_file").y

        count_file_tissue_sections = splotch_input_data.metadata[
            splotch_input_data.metadata.index.get_level_values("count_file")
            == count_file
        ].index.get_level_values("tissue_section")

        for tissue_section_idx, tissue_section in enumerate(
            count_file_tissue_sections.unique(), start=1
        ):
            ax.scatter(
                x[count_file_tissue_sections == tissue_section],
                y[count_file_tissue_sections == tissue_section],
                s=5,
                alpha=0.8,
                label=f"Tissue section #{tissue_section_idx}",
            )

        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            f"{count_file}\n"
            f"{splotch_input_data.metadata.query('count_file == @count_file').level_1.iloc[0]}"
        )

        ax.legend()

    fig.tight_layout()

    return fig
