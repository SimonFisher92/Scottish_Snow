import logging
from datetime import time
from typing import Any, Optional, Tuple
import json
from pathlib import Path
import time

import pandas as pd
from sentinelhub import SHConfig, DataCollection, SentinelHubCatalog, SentinelHubRequest, BBox, bbox_to_dimensions, CRS, \
    MimeType, Geometry
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm

logger = logging.getLogger()


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


def save_numpy_array(image: np.ndarray,
                   filename: Path,
                   ) -> None:
    """
    Utility function for saving numpy array
    """
    np.save(filename, image)


def save_rgb_image(image: np.ndarray,
                   filename: Path,
                   ) -> None:
    """
    Utility function for saving RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    factor = 3.5 / 10000  # We are downloading data in "DN" units, which have a range of 0-10,000. The 3.5 is to increase brightness.
    rgb_image = image[:, :, 3:0:-1]
    ax.imshow(np.clip(rgb_image * factor, 0, 1))  # Sliced to give RGB channels
    ax.set_xticks([])
    ax.set_yticks([])
    date = extract_date_from_filename(filename)
    fig.title(date)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    
def extract_date_from_filename(filename):
    """
    Extracts the date from a given filename.

    :param filename: The input filename
    :return: The extracted date in "YYYY-MM-DD" format, or None if not found
    """
    parts = filename.split('_')

    # Find the part that contains the date
    date_part = next((part for part in parts if part.isdigit() and len(part) == 8), None)

    if date_part:
        # Convert to a more readable format, if needed
        formatted_date = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:]}"
        return formatted_date
    else:
        return None


def download_data(start_date,
                  end_date,
                  resolution,
                  patchname,
                  geojson_path,
                  cadence,
                  satellite,
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
    :param satellite: L1C or L2A (please use L2A for snow)
    :param savedir: directory to save images to

    :return: None
    """

    # get each week/day between start and end date
    flyover_iterator = pd.date_range(start=start_date, end=end_date, freq=cadence)

    savedir = Path(savedir) / "downloads" / patchname / f"{start_date}-{end_date}"

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
            satellite,
            bbox=aoi_bbox,
            time=time_interval,
            fields={"include": ["id", "properties.datetime"], "exclude": []},

        )

        results = list(search_iterator)

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

        request_true_color = SentinelHubRequest(
            evalscript=evalscript_all_bands,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=satellite.define_from(
                        name="s2", service_url="https://sh.dataspace.copernicus.eu"
                    ),
                    time_interval=(time_interval[0], time_interval[1]),
                    other_args={"dataFilter": {"mosaickingOrder": "leastCC"}})
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=aoi_bbox,
            size=aoi_size,
            config=config,
        )

        # retry the request until no 500 error is returned, aka brute force the server until it gives us the data
        try:
            images = request_true_color.get_data()

        except Exception as e:
            logger.error(e)
            continue

        if len(results) > 0:
            image = images[0]
            image_path = savedir / "rgb" / (results[0]['id'] + '.png')
            save_rgb_image(image, image_path)
            numpy_path = savedir / "numpy" / (results[0]['id'] + '.npy')
            save_numpy_array(image, numpy_path)


def main():
    cadence = {'weekly': 'W', 'daily': 'D'}

    product_level = {"L1C": DataCollection.SENTINEL2_L1C,
                     "L2A": DataCollection.SENTINEL2_L2A}

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

    for year in years:
        logger.info(f"Downloading data for year {year}")

        for geojson in geojsons:
            download_data(start_date=f'{year}-05-01',
                          end_date=f'{year}-09-30',
                          resolution=10,
                          patchname=geojson.stem,
                          geojson_path=geojson,
                          cadence=cadence['daily'],
                          satellite=product_level['L1C'],
                          savedir='output')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    matplotlib.use('Agg')  # Fix for RuntimeError: main thread is not in main loop
    main()

