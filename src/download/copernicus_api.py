import logging
from datetime import time

import pandas as pd
from sentinelhub import SHConfig, DataCollection, SentinelHubCatalog, SentinelHubRequest, BBox, bbox_to_dimensions, CRS, \
    MimeType, Geometry
from typing import Any, Optional, Tuple
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage.transform import resize
from skimage.util import img_as_ubyte
import time
import cv2

from tqdm import tqdm


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


def bicubic_upsampling(image: np.ndarray) -> np.ndarray:
    """
    Utility function for upsampling images using bicubic interpolation. This is used to increase the resolution of the image
    :param image:
    :return:
    """

    # Check if the image is a NumPy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input is not a NumPy array")

    # Check the data type and range of the image
    if image.dtype == np.uint8:
        # Image with 0-255 range
        pass
    elif image.dtype in [np.float32, np.float64]:
        # Image with 0.0-1.0 range
        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("Floating-point image has values outside the range 0.0-1.0")
    else:
        raise ValueError("Image data type not supported")

    try:

        upscaled = resize(image, (image.shape[0] * 3, image.shape[1] * 3), order=3, mode='reflect', anti_aliasing=True)

    except Exception as e:
        raise RuntimeError(f"Error during resizing: {e}")

    return img_as_ubyte(upscaled)


def geojson_to_bbox(geojson_path: Path) -> Tuple[float, float, float, float]:
    """
    Utility function for getting the top left and bottom right coordinates from a geojson file.
    :param geojson_path:
    :return:
    """

    with open(geojson_path) as f:
        geojson = json.load(f)

    # get top left and bottom right coordinates from geojson
    top_left = geojson['geometry']['coordinates'][0][0]
    bottom_right = geojson['geometry']['coordinates'][0][2]

    return top_left[0], top_left[1], bottom_right[0], bottom_right[1]


def get_config(client_id: str,
               client_secret: str) -> SHConfig:
    """
    Utility function for getting the config for the copernicus api
    :param client_id:
    :param client_secret:
    :return:
    """

    config = SHConfig()

    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.save("cdse")
    # Saved config can be later accessed with config = SHConfig("cdse")

    config = SHConfig("cdse")

    return config


def save_image(image: np.ndarray,
               filename: Path,
               image_type: str = 'true_color',
               factor: float = 1.0,
               upsample: bool = True, ) -> None:
    """
    Utility function for saving RGB images.
    :param image:
    :param filename:
    :param image_type:
    :param factor:
    :param upsample:
    :return:
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

    if upsample:
        image = bicubic_upsampling(image)

    if image_type == 'snow':
        image = bluebird_transform(image)

    ax.imshow(np.clip(image * factor, 0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)


def plot_image(
        image: np.ndarray,
        image_type: str = 'true_color',
        factor: float = 1.0,
        clip_range: Optional[Tuple[float, float]] = None,
        upsample: bool = True,
        **kwargs: Any
) -> None:
    """
    Utility function for plotting RGB images.
    :param image:
    :param image_type:
    :param factor:
    :param clip_range:
    :param upsample:
    :param kwargs:
    :return:
    """

    if upsample:
        image = bicubic_upsampling(image)

    if image_type == 'snow':
        image = bluebird_transform(image)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


def download_data(start_date,
                  end_date,
                  resolution,
                  upsampling,
                  image_type,
                  patchname,
                  geojson_path,
                  cadence,
                  satellite,
                  client_id,
                  client_secret,
                  savedir) -> None:
    """Download data from the copernicus api

    This is the migration to the new API. What is great about this is we dont need to download whole scenes, we can just download the area we want.
    ie, we dont need to spend ages waiting for tiles to download that we dont need. We need to specify the area we want to download, and the resolution.


    :param start_date: start date of time series
    :param end_date: end date of time series
    :param resolution: resolution of image
    :param upsampling: whether to upsample the image
    :param image_type: true_color or snow
    :param patchname: name of patch
    :param geojson_path: path to geojson
    :param cadence: weekly or daily
    :param satellite: L1C or L2A (please use L2A for snow)
    :param client_id: client id for copernicus api
    :param client_secret: client secret for copernicus api
    :param savedir: directory to save images to

    :return: None
    """

    # get each week/day between start and end date
    flyover_iterator = pd.date_range(start=start_date, end=end_date, freq=cadence)

    savedir = Path(savedir + f'/{image_type}/' + patchname)

    savedir = Path.cwd().parent.parent / savedir

    if not savedir.exists():
        savedir.mkdir(parents=True)

    snowpatch_aoi = geojson_to_bbox(geojson_path)

    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

    config = get_config(client_id, client_secret)
    catalog = SentinelHubCatalog(config=config)
    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)

    # start for loop to iterate through each day/week

    print(f'Getting data for {patchname} from {start_date} to {end_date} at {resolution} m resolution')
    for i in tqdm(range(len(flyover_iterator) - 1), desc=f'Image in Time Series for {patchname} in {year}'):

        time.sleep(0.01)  # scared of getting banned from the api :)

        time_interval = (flyover_iterator[i].strftime('%Y-%m-%d'), flyover_iterator[i + 1].strftime('%Y-%m-%d'))

        search_iterator = catalog.search(
            satellite,
            bbox=aoi_bbox,
            time=time_interval,
            fields={"include": ["id", "properties.datetime"], "exclude": []},

        )

        results = list(search_iterator)
        # print("Total number of results:", len(results))

        # fyi this will take the least cloudy image from the weekly flyover

        if image_type == 'true_color':
            evalscript = """//VERSION=3

                        function setup() {
                            return {
                                input: [{
                                    bands: ["B02", "B03", "B04"]
                                }],
                                output: {
                                    bands: 3
                                }
                            };
                        }

                        function evaluatePixel(sample) {
                            return [sample.B04, sample.B03, sample.B02];
                        }
                    """

        elif image_type == 'snow':

            evalscript = """
                //VERSION=3
        
                function setup() {
                    return {
                        input: [{
                            bands: ["B02", "B03", "SNW"]
                        }],
                        output: {
                            bands: 3
                        }
                    };
                }
        
                function evaluatePixel(sample) {
                    return [sample.B04, sample.B03, sample.SNW];
                }
            """

        request_true_color = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=satellite.define_from(
                        name="s2", service_url="https://sh.dataspace.copernicus.eu"
                    ),
                    time_interval=(time_interval[0], time_interval[1]),
                    other_args={"dataFilter": {"mosaickingOrder": "leastCC"}})
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=aoi_bbox,
            size=aoi_size,
            config=config,
        )

        # retry the request until no 500 error is returned, aka brute force the server until it gives us the data
        try:
            true_color_imgs = request_true_color.get_data()

        except Exception as e:
            print(e)
            continue




        image = true_color_imgs[0]

        if len(results) > 0:
            plot_image(image, image_type=image_type, factor=3 / 255, clip_range=(0, 1), upsample=upsampling)

            image_path = savedir / (results[0]['id'] + '.png')

            save_image(true_color_imgs[0], image_path, image_type=image_type, factor=3.5 / 255, upsample=upsampling)


if __name__ == "__main__":

    # get these from https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings
    # or message me and I'll send them to you
    # just dont commit them to github

    client_id = "sh-08ba0f4f-0eb9-4638-8b20-9fab1042711d"
    client_secret = "NLi765wHJK4j9AN3LeumESDGpgbVYVlS"

    cadence = {'weekly': 'W',

               'daily': 'D'}

    satellite = {"L1C": DataCollection.SENTINEL2_L1C,
                 "L2A": DataCollection.SENTINEL2_L2A}

    years = [2018, 2019, 2020, 2021, 2022, 2023]

    geojsons = list(Path.cwd().parent.parent.glob('geo_jsons/*.geojson'))

    # set to true for debugging etc, setting to false will get all data
    developer_mode = False

    for year in years:

        if developer_mode:  # lightweight mode for development
            print('You are in developer mode, only downloading limited data for Ciste Mhearad')
            download_data(start_date='2023-05-01',
                          end_date='2023-06-02',
                          resolution=10,
                          #upsampling options: True or False
                          upsampling=True,
                          #image_type options: 'true_color' or 'snow'
                          image_type='snow',
                          #patchname options: see ../geo_jsons
                          patchname='Ciste_Mhearad',
                          geojson_path=Path.cwd().parent.parent / 'geo_jsons' / 'Ciste_Mhearad.geojson',
                          #cadence options: 'weekly' or 'daily'
                          cadence=cadence['daily'],
                          #satellite options: 'L1C' or 'L2A', for snow use L2A
                          satellite=satellite['L2A'],
                          client_id=client_id,
                          client_secret=client_secret,
                          savedir='data')

            break

        for geojson in geojsons:
            print('You are in full mode, downloading all data, this will take several hours, set and forget')
            download_data(start_date=f'{year}-05-01',
                          end_date=f'{year}-09-30',
                          resolution=10,
                          upsampling=True,
                          image_type='snow',
                          patchname=geojson.stem,
                          geojson_path=geojson,
                          cadence=cadence['daily'],
                          satellite=satellite['L2A'],
                          client_id=client_id,
                          client_secret=client_secret,
                          savedir='data')
