import logging
from datetime import time
from typing import Any, Optional, Tuple
import json
from pathlib import Path
import time

import pandas as pd
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from sentinelhub import SHConfig, DataCollection, SentinelHubCatalog, SentinelHubRequest, BBox, bbox_to_dimensions, CRS, \
    MimeType, Geometry
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm

logger = logging.getLogger()


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


def get_config(config_path: str = "config.json") -> SHConfig:
    """
    Utility function for getting the config for the Copernicus API. Reads a config file to find client_id
    and client_secret. Get these from https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings

    This looks for a simple json config file like:

    {
      "client_id": "..",
      "client_secret": "..."
    }

    :param config_path: Path to the configuration file
    :return: SHConfig object
    """
    assert Path(config_path).exists(), "Error - cannot find config.json to read authentication tokens"

    with open(config_path, 'r') as file:
        config_data = json.load(file)

    config = SHConfig()
    config.sh_client_id = config_data['client_id']
    config.sh_client_secret = config_data['client_secret']
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"

    return config


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


def download_dem_data(patchname, geojson_path, savedir) -> None:
    """Download elevation data for each patch
    """

    savedir = Path(savedir) / "downloads" / patchname / "dem"

    # If relative path is given, start from project root
    if not savedir.is_absolute():
        savedir = Path(__file__).parent.parent.parent / savedir

    if not savedir.exists():
        savedir.mkdir(parents=True)

    numpy_filename = savedir / f"{patchname}_dem.npy"

    if numpy_filename.exists():
        logger.info(f"{numpy_filename} already exists: skipping")
        return

    evalscript_dem = """
    //VERSION=3
    function setup() {
      return {
        input: ["DEM"],
        output:{
          id: "default",
          bands: 1,
          sampleType: SampleType.FLOAT32
        }
      }
    }

    function evaluatePixel(sample) {
      return [sample.DEM]
    }
    """

    snowpatch_aoi = geojson_to_bbox(geojson_path)

    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=10)

    config = get_config()
    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)

    dem_request = SentinelHubRequest(
        evalscript=evalscript_dem,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.DEM.define_from(
                    name="dem", service_url="https://sh.dataspace.copernicus.eu"
                ),
                time_interval=("2020-06-12", "2020-06-13"),  # Arbitrary dates, assume that elevation is constant
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi_bbox,
        size=aoi_size,
        config=config,
    )

    dem_data = dem_request.get_data()[0]

    png_filename = savedir / f"{patchname}_dem.png"
    slope_png_filename = savedir / f"{patchname}_slope.png"
    save_dem_image(dem_data, png_filename)
    convert_dem_to_slope(dem_data, slope_png_filename)

    np.save(numpy_filename, dem_data)




def download_l1c_data(start_date,
                      end_date,
                      resolution,
                      patchname,
                      geojson_path,
                      cadence,
                      savedir) -> None:
    """Download data from the copernicus api

    This is the migration to the new API. What is great about this is we dont need to download whole scenes, we can just download the area we want.
    ie, we dont need to spend ages waiting for tiles to download that we dont need. We need to specify the area we want to download, and the resolution.


    :param start_date: start date of time series
    :param end_date: end date of time series
    :param resolution: resolution of image
    :param patchname: name of patch
    :param geojson_path: path to geojson
    :param cadence: weekly or daily
    :param savedir: directory to save images to

    :return: None
    """

    # get each week/day between start and end date
    flyover_iterator = pd.date_range(start=start_date, end=end_date, freq=cadence)

    savedir = Path(savedir) / "downloads" / patchname / "l1c" / f"{start_date}-{end_date}"

    # If relative path is given, start from project root
    if not savedir.is_absolute():
        savedir = Path(__file__).parent.parent.parent / savedir

    if not savedir.exists():
        savedir.mkdir(parents=True)

    snowpatch_aoi = geojson_to_bbox(geojson_path)

    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

    config = get_config()
    catalog = SentinelHubCatalog(config=config)
    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)

    # start for loop to iterate through each day/week

    logger.info(f'Getting data for {patchname} from {start_date} to {end_date} at {resolution} m resolution')

    for i in tqdm(range(len(flyover_iterator) - 1), desc=f'Image in Time Series for {patchname} in {start_date[:4]}'):

        time.sleep(0.01)  # scared of getting banned from the api :)

        time_interval = (flyover_iterator[i].strftime('%Y-%m-%d'), flyover_iterator[i + 1].strftime('%Y-%m-%d'))

        search_iterator = catalog.search(
            DataCollection.SENTINEL2_L1C,
            bbox=aoi_bbox,
            time=time_interval,
            fields={"include": ["id", "properties.datetime"], "exclude": []},

        )

        results = list(search_iterator)

        if len(results) == 0:
            continue

        numpy_filename = savedir / (results[0]['id'] + '.npy')

        if numpy_filename.exists():
            logger.info(f"{numpy_filename} already exists: skipping")
            continue

        # fyi this will take the least cloudy image from the weekly flyover
        # From Murray - is the above always true? Does it depend on whether the cadence is daily or weekly?

        evalscript_all_bands = """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"],
                        units: "DN"
                    }],
                    output: {
                        bands: 13,
                        sampleType: "INT16"
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B01,
                        sample.B02,
                        sample.B03,
                        sample.B04,
                        sample.B05,
                        sample.B06,
                        sample.B07,
                        sample.B08,
                        sample.B8A,
                        sample.B09,
                        sample.B10,
                        sample.B11,
                        sample.B12];
            }
        """

        request_all_bands = SentinelHubRequest(
            evalscript=evalscript_all_bands,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C.define_from(
                        name="s2c", service_url="https://sh.dataspace.copernicus.eu"
                    ),
                    time_interval=(time_interval[0], time_interval[1]),
                    other_args={"dataFilter": {"mosaickingOrder": "leastCC"}})
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF),
            ],
            bbox=aoi_bbox,
            size=aoi_size,
            config=config,
        )

        # retry the request until no 500 error is returned, aka brute force the server until it gives us the data
        try:
            band_request = request_all_bands.get_data()
        except Exception as e:
            logger.error(e)
            continue

        image = band_request[0]
        image_path = savedir / (results[0]['id'] + '.png')
        save_rgb_image(image, image_path)
        np.save(numpy_filename, image)


def download_l2a_data(start_date,
                      end_date,
                      resolution,
                      patchname,
                      geojson_path,
                      cadence,
                      savedir) -> None:
    """Download data from the copernicus api

    This is the migration to the new API. What is great about this is we dont need to download whole scenes, we can just download the area we want.
    ie, we dont need to spend ages waiting for tiles to download that we dont need. We need to specify the area we want to download, and the resolution.


    :param start_date: start date of time series
    :param end_date: end date of time series
    :param resolution: resolution of image
    :param patchname: name of patch
    :param geojson_path: path to geojson
    :param cadence: weekly or daily
    :param savedir: directory to save images to

    :return: None
    """

    # get each week/day between start and end date
    flyover_iterator = pd.date_range(start=start_date, end=end_date, freq=cadence)

    savedir = Path(savedir) / "downloads" / patchname / "l2a" / f"{start_date}-{end_date}"

    # If relative path is given, start from project root
    if not savedir.is_absolute():
        savedir = Path(__file__).parent.parent.parent / savedir

    if not savedir.exists():
        savedir.mkdir(parents=True)

    snowpatch_aoi = geojson_to_bbox(geojson_path)

    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

    config = get_config()
    catalog = SentinelHubCatalog(config=config)
    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)

    # start for loop to iterate through each day/week

    logger.info(f'Getting data for {patchname} from {start_date} to {end_date} at {resolution} m resolution')

    for i in tqdm(range(len(flyover_iterator) - 1), desc=f'Image in Time Series for {patchname} in {start_date[:4]}'):

        time.sleep(0.01)  # scared of getting banned from the api :)

        time_interval = (flyover_iterator[i].strftime('%Y-%m-%d'), flyover_iterator[i + 1].strftime('%Y-%m-%d'))

        search_iterator = catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=aoi_bbox,
            time=time_interval,
            fields={"include": ["id", "properties.datetime"], "exclude": []},

        )

        results = list(search_iterator)

        if len(results) == 0:
            continue

        numpy_filename = savedir / (results[0]['id'] + '.npy')

        if numpy_filename.exists():
            logger.info(f"{numpy_filename} already exists: skipping")
            continue

        # fyi this will take the least cloudy image from the weekly flyover
        # From Murray - is the above always true? Does it depend on whether the cadence is daily or weekly?

        evalscript_cls = """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["SCL", "SNW", "CLD"],
                        units: "DN"
                    }],
                    output: {
                        bands: 3,
                        sampleType: "INT16"
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.SCL,
                        sample.SNW,
                        sample.CLD];
            }
        """

        request_all_bands = SentinelHubRequest(
            evalscript=evalscript_cls,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A.define_from(
                        name="s2a", service_url="https://sh.dataspace.copernicus.eu"
                    ),
                    time_interval=(time_interval[0], time_interval[1]),
                    other_args={"dataFilter": {"mosaickingOrder": "leastCC"}})
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            bbox=aoi_bbox,
            size=aoi_size,
            config=config,
        )

        # retry the request until no 500 error is returned, aka brute force the server until it gives us the data
        try:
            cls_response = request_all_bands.get_data()
        except Exception as e:
            logger.error(e)
            continue

        image = cls_response[0]
        image_path = savedir / (results[0]['id'] + '.png')
        save_cls_image(image, image_path)
        np.save(numpy_filename, image)


def main():
    cadence = {'weekly': 'W', 'daily': 'D'}

    years = [2018, 2019, 2020, 2021, 2022, 2023]

    # set to true for debugging etc, setting to false will get all data
    developer_mode = False
    if developer_mode:  # lightweight mode for development
        logger.info('You are in developer mode')
        years = [2020]
    else:
        logger.info('You are in full mode, downloading all data, this will take several hours, set and forget')

    geojsons = list(Path(__file__).parent.parent.parent.glob('output/geojson_bboxes/*.geojson'))

    assert len(geojsons) > 0

    for geojson in geojsons:
        logger.info(f"Downloading DEM for {geojson.stem}")
        download_dem_data(patchname=geojson.stem, geojson_path=geojson, savedir="output")

    for year in years:
        logger.info(f"Downloading L1C band data for year {year}")

        for geojson in geojsons:
            logger.info(f"Downloading L1C band data for {geojson.stem}")
            download_l1c_data(start_date=f'{year}-05-01',
                              end_date=f'{year}-09-30',
                              resolution=10,
                              patchname=geojson.stem,
                              geojson_path=geojson,
                              cadence=cadence['daily'],
                              savedir='output')

            logger.info(f"Downloading L2A band data for {geojson.stem}")
            download_l2a_data(start_date=f'{year}-05-01',
                              end_date=f'{year}-09-30',
                              resolution=10,
                              patchname=geojson.stem,
                              geojson_path=geojson,
                              cadence=cadence['daily'],
                              savedir='output')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    matplotlib.use('Agg')  # Fix for RuntimeError: main thread is not in main loop
    main()

