import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def convert_dem_to_slope(dem_data: np.array,
                         slope_png_filename: Path) -> None:
    altitude = dem_data

    smoothed_altitude = gaussian_filter(altitude, sigma=3)
    norm_smoothed_altitude = Normalize()(smoothed_altitude)

    grayscale_smoothed_altitude_image = np.uint8(plt.cm.gray(norm_smoothed_altitude) * 255)

    grad_y_smoothed, grad_x_smoothed = np.gradient(smoothed_altitude)
    slope_direction_smoothed = np.arctan2(grad_y_smoothed, grad_x_smoothed)

    norm_slope_direction = Normalize()(slope_direction_smoothed)
    direction_color_map = plt.cm.hsv(norm_slope_direction)

    # Combine the grayscale intensity and the direction color map
    combined_image = np.zeros((smoothed_altitude.shape[0], smoothed_altitude.shape[1], 3), dtype=np.uint8)
    for i in range(3):  # Iterate over RGB channels
        combined_image[..., i] = (grayscale_smoothed_altitude_image[..., 0] * direction_color_map[..., i]).astype(
            np.uint8)

    assert altitude.shape == combined_image.shape[:2]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    im = ax.imshow(combined_image)  # Sliced to give RGB channels
    fig.colorbar(im, ax=ax, label="Height [m]")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(slope_png_filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_ds(ds, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    patch = ds.attrs["patchname"]

    dates = ds.coords["date"].data

    for dt64 in dates:
        date = pd.to_datetime(dt64).date()

        date_index = (ds['date'] == dt64).values

        ds_date = ds.sel(t=date_index)

        factor = 3.5 / 10000  # The 3.5 is to increase brightness. The 10,000 is to convert from DN units. See: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L1C.html

        b02 = ds_date["B02"].values[0] * factor
        b03 = ds_date["B03"].values[0] * factor
        b04 = ds_date["B04"].values[0] * factor

        rgb_data = np.stack([b04, b03, b02], axis=-1)
        cld = ds_date["CLD"].values[0]
        snw = ds_date["SNW"].values[0]
        b11 = ds_date["B11"].values[0] * factor
        ndsi = (b03 - b11) / (
                b03 + b11)  # Snow index: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndsi/

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        ax = axs[0, 0]
        ax.set_title("RGB")
        ax.imshow(np.clip(rgb_data, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axs[0, 1]
        ax.set_title("Cloud probability")
        im = ax.imshow(cld, vmin=0, vmax=100)
        fig.colorbar(im, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axs[1, 1]
        ax.set_title("Snow probability")
        im = ax.imshow(snw, vmin=0, vmax=100)
        fig.colorbar(im, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axs[1, 0]
        ax.set_title("Normalised difference snow index")
        im = ax.imshow(ndsi, vmin=-0.2, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

        fig.suptitle(f"{patch}: {date}")
        fig.tight_layout()
        fig.savefig(outdir / f"{patch}_{date}.png")
        plt.close()


def save_dem_image(image: np.ndarray,
                   filename: Path,
                   ) -> None:
    """
    Utility function for saving png plot of digital elevation data
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    im = ax.imshow(image)  # Sliced to give RGB channels
    fig.colorbar(im, ax=ax, label="Height [m]")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_cls_image(image: np.ndarray,
                   filename: Path,
                   ) -> None:
    """
    Utility function for saving RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    cls_image = image[:, :, 0]
    ax.imshow(cls_image)
    ax.set_xticks([])
    ax.set_yticks([])
    date = extract_date_from_filename(filename)
    ax.set_title(date)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_rgb_image(image: np.ndarray,
                   filename: Path,
                   ) -> None:
    """
    Utility function for saving RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    factor = 3.5 / 10000  # The 3.5 is to increase brightness. The 10,000 is to convert from DN units. See: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L1C.html
    rgb_image = image[:, :, 3:0:-1]
    ax.imshow(np.clip(rgb_image * factor, 0, 1))  # Sliced to give RGB channels
    ax.set_xticks([])
    ax.set_yticks([])
    date = extract_date_from_filename(filename)
    ax.set_title(date)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def extract_date_from_filename(filename: Path):
    """
    Extracts the date from a given filename.

    :param filename: The input filename
    :return: The extracted date in "YYYY-MM-DD" format, or None if not found
    """
    parts = filename.stem.split('_')

    # Find the part that contains the date
    date_part = parts[2]

    # Convert to a more readable format
    formatted_date = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:]}"

    return formatted_date
