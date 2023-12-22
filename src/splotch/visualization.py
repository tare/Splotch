"""visualization.py."""
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from splotch.dataclasses import SplotchInputData, SplotchResult


def plot_coefficients(
    splotch_input_data: SplotchInputData, splotch_result: SplotchResult, gene: str
) -> Figure:
    """Plot coefficients.

    Args:
        splotch_input_data: TBA.
        splotch_result: TBA.
        gene: TBA.

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


def plot_rates_on_slides(splotch_result: SplotchResult, gene: str) -> Figure:
    """Plot rate estimates on slides.

    Args:
        splotch_result: TBA.
        gene: TBA.

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

        tissue_image = Image.open(
            splotch_result.metadata.query("count_file == @count_file").image_file.iloc[
                0
            ]
        )

        xdim, ydim = tissue_image.size

        # downsample the image
        tissue_image = tissue_image.resize(
            (np.round(xdim * 0.05).astype(int), np.round(ydim * 0.05).astype(int))
        )

        xdim, ydim = tissue_image.size
        pixel_dim = 194.0 / (6200.0 / xdim)

        x = pixel_dim * (
            splotch_result.metadata.query("count_file == @count_file").x - 1
        )
        y = pixel_dim * (
            splotch_result.metadata.query("count_file == @count_file").y - 1
        )
        c = rates_s[count_file]

        ax.imshow(tissue_image, origin="upper", interpolation="none", alpha=0.6)

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
    splotch_result: SplotchResult, gene: str, variable: str = "f"
) -> Figure:
    """Plot variable of interest on slides.

    Variable has to be scalar per spot.

    Args:
        splotch_result: TBA.
        gene: TBA.
        variable: TBA.

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

        tissue_image = Image.open(
            splotch_result.metadata.query("count_file == @count_file").image_file.iloc[
                0
            ]
        )

        xdim, ydim = tissue_image.size

        # downsample the image
        tissue_image = tissue_image.resize(
            (np.round(xdim * 0.05).astype(int), np.round(ydim * 0.05).astype(int))
        )

        xdim, ydim = tissue_image.size
        pixel_dim = 194.0 / (6200.0 / xdim)

        x = pixel_dim * (
            splotch_result.metadata.query("count_file == @count_file").x - 1
        )
        y = pixel_dim * (
            splotch_result.metadata.query("count_file == @count_file").y - 1
        )
        c = data[
            splotch_result.metadata.index.get_level_values("count_file") == count_file
        ]

        ax.imshow(tissue_image, origin="upper", interpolation="none", alpha=0.6)

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

    Args:
        splotch_result: TBA.
        gene: TBA.

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

    Args:
        splotch_input_data: TBA.

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


def plot_annotations_on_slides(splotch_input_data: SplotchInputData) -> Figure:
    """Plot annotations on slides.

    Args:
        splotch_input_data: TBA.

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

        tissue_image = Image.open(
            splotch_input_data.metadata.query(
                "count_file == @count_file"
            ).image_file.iloc[0]
        )

        xdim, ydim = tissue_image.size

        # downsample the image
        tissue_image = tissue_image.resize(
            (np.round(xdim * 0.05).astype(int), np.round(ydim * 0.05).astype(int))
        )

        xdim, ydim = tissue_image.size
        pixel_dim = 194.0 / (6200.0 / xdim)

        x = pixel_dim * (
            splotch_input_data.metadata.query("count_file == @count_file").x - 1
        )
        y = pixel_dim * (
            splotch_input_data.metadata.query("count_file == @count_file").y - 1
        )
        count_file_annotations = splotch_input_data.metadata.aar[
            splotch_input_data.metadata.index.get_level_values("count_file")
            == count_file
        ]

        ax.imshow(tissue_image, origin="upper", interpolation="none", alpha=0.6)

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


def plot_tissue_sections_on_slides(splotch_input_data: SplotchInputData) -> Figure:
    """Plot detected tissue sections on slides.

    Args:
        splotch_input_data: TBA.

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

        tissue_image = Image.open(
            splotch_input_data.metadata.query(
                "count_file == @count_file"
            ).image_file.iloc[0]
        )

        xdim, ydim = tissue_image.size

        # downsample the image
        tissue_image = tissue_image.resize(
            (np.round(xdim * 0.05).astype(int), np.round(ydim * 0.05).astype(int))
        )

        xdim, ydim = tissue_image.size
        pixel_dim = 194.0 / (6200.0 / xdim)

        x = pixel_dim * (
            splotch_input_data.metadata.query("count_file == @count_file").x - 1
        )
        y = pixel_dim * (
            splotch_input_data.metadata.query("count_file == @count_file").y - 1
        )
        count_file_tissue_sections = splotch_input_data.metadata[
            splotch_input_data.metadata.index.get_level_values("count_file")
            == count_file
        ].index.get_level_values("tissue_section")

        ax.imshow(tissue_image, origin="upper", interpolation="none", alpha=0.6)

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
