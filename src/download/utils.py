import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr
from sentinelhub import SHConfig, BBox

logger = logging.getLogger(__name__)


def bluebird_transform(image: np.ndarray) -> np.ndarray:
    """
    Utility function for transforming RGB images to bluebird images. Bluebird was my previous method of separating snow from cloud
    Should only run when the image_type request is for "snow", not "true colour".
    :param image:
    :return:
    """

    modified_image = image.copy()

    # Increase the blue channel
    modified_image[:, :, 2] = np.clip(modified_image[:, :, 2] * 1.9, 0, 255)

    # Increase red and green channels to create yellow
    modified_image[:, :, 0] = np.clip(modified_image[:, :, 0] / 1.6, 0, 255)  # Red channel
    modified_image[:, :, 1] = np.clip(modified_image[:, :, 1] / 1.6, 0, 255)  # Green channel

    return modified_image


def geojson_to_bbox(geojson_path: Path) -> Tuple[float, float, float, float]:
    """
    Utility function for getting the top left and bottom right coordinates from a geojson file.
    :param geojson_path:
    :return:
    """
    assert geojson_path.exists(), f"Error: geojson file at {geojson_path} does not exist. Try running script at src/produce_geojsons/produce_geojsons.py"

    with open(geojson_path) as f:
        geojson = json.load(f)

    # get top left and bottom right coordinates from geojson
    top_left = geojson['geometry']['coordinates'][0][0]
    bottom_right = geojson['geometry']['coordinates'][0][2]

    return top_left[0], top_left[1], bottom_right[0], bottom_right[1]


def get_config(client_id: str = '', client_secret: str = '') -> SHConfig:
    """
    Utility function for getting the config for the Copernicus API. Reads a config file to find client_id
    and client_secret. Get these from https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings

    :return: SHConfig object
    """
    if client_id == '' or client_secret == '':
        raise ValueError("Error: client_id and client_secret must be provided, please check the README for instructions on how to get these.")

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"

    return config


def bbox_to_grid_coords(aoi_bbox, aoi_size):
    """Get 2D arrays of lon and lat coords for grid cell centres in this bounding box
    """
    aoi_bbox = BBox._tuple_from_bbox(aoi_bbox)

    assert len(aoi_bbox) == 4
    assert len(aoi_size) == 2
    assert aoi_bbox[2] > aoi_bbox[0]
    assert aoi_bbox[3] > aoi_bbox[1]
    assert aoi_size[0] > 0
    assert aoi_size[1] > 0

    lon_start = aoi_bbox[0]
    lon_end = aoi_bbox[2]
    lon_range = lon_end - lon_start
    lon_dx = lon_range / aoi_size[0]  # Spacing of longitudinal grid lines
    lon_centres = np.linspace(lon_start + 0.5 * lon_dx, lon_end - 0.5 * lon_dx, aoi_size[0])

    lat_start = aoi_bbox[1]
    lat_end = aoi_bbox[3]
    lat_range = lat_end - lat_start
    lat_dx = lat_range / aoi_size[1]  # Spacing of latgitudinal grid lines
    lat_centres = np.linspace(lat_start + 0.5 * lat_dx, lat_end - 0.5 * lat_dx, aoi_size[1])

    lon_centres_2d, lat_centres_2d = np.meshgrid(lon_centres, lat_centres, indexing="xy")

    return lon_centres_2d, lat_centres_2d


def dataset_filename(geojson: Path, year: int) -> str:
    return f"{geojson.stem}_{year}.nc"


def load_dataset(filename) -> xr.Dataset:
    ds = xr.open_dataset(filename)
    return ds
