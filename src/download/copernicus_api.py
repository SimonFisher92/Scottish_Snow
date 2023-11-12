import pandas as pd
from sentinelhub import SHConfig, DataCollection, SentinelHubCatalog, SentinelHubRequest, BBox, bbox_to_dimensions, CRS, MimeType, Geometry
from typing import Any, Optional, Tuple
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def geojson_to_bbox(geojson_path: Path) -> Tuple[float, float, float, float]:


    with open(geojson_path) as f:
        geojson = json.load(f)

    #get top left and bottom right coordinates from geojson
    top_left = geojson['geometry']['coordinates'][0][0]
    bottom_right = geojson['geometry']['coordinates'][0][2]

    return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])


def get_config(client_id: str,
               client_secret: str) -> SHConfig:

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
               factor: float = 1.0) -> None:

    """Utility function for saving RGB images."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    ax.imshow(np.clip(image * factor, 0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)

def plot_image(
    image: np.ndarray,
    factor: float = 1.0,
    clip_range: Optional[Tuple[float, float]] = None,
    **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
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


    Args:
        start_date (str): start date in format YYYY-MM-DD
        end_date (str): end date in format YYYY-MM-DD
        resolution (int): resolution of image in meters
        patchname (str): name of patch, this will be used to create a folder to save the images in
        cadence (str): cadence of flyover, either 'W' for weekly or 'D' for daily
        satellite (str): satellite to download data from, either 'L1C' or 'L2A'
        client_id (str): client id for copernicus api
        client_secret (str): client secret for copernicus api
        savedir (str): path to save directory

    Returns: None
    """

    #get each week/day between start and end date
    flyover_iterator = pd.date_range(start=start_date, end=end_date, freq=cadence)
    print(flyover_iterator)

    savedir = Path(savedir + '/' + patchname)


    savedir = Path.cwd().parent.parent / savedir


    if not savedir.exists():
        savedir.mkdir(parents=True)

    #TODO this should be able to take the geojsons, this is currently hardcoded

    snowpatch_aoi = geojson_to_bbox(geojson_path)


    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

    print(f'Image shape at {resolution} m resolution: {aoi_size} pixels')

    config = get_config(client_id, client_secret)
    catalog = SentinelHubCatalog(config=config)
    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)

    #start for loop to iterate through each week
    for i in range(len(flyover_iterator)-1):

        time_interval = (flyover_iterator[i].strftime('%Y-%m-%d'), flyover_iterator[i+1].strftime('%Y-%m-%d'))


        search_iterator = catalog.search(
            satellite,
            bbox=aoi_bbox,
            time=time_interval,
            fields={"include": ["id", "properties.datetime"], "exclude": []},

        )

        results = list(search_iterator)
        print("Total number of results:", len(results))

        #fyi this will take the least cloudy image from the weekly flyover

        evalscript_true_color = """
            //VERSION=3
    
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

        request_true_color = SentinelHubRequest(
            evalscript=evalscript_true_color,
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

        true_color_imgs = request_true_color.get_data()

        print(f"Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.")
        print(
            f"Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}")

        image = true_color_imgs[0]
        print(f"Image type: {image.dtype}")

        if len(results) > 0:
            plot_image(image, factor=3 / 255, clip_range=(0, 1))

            #create save path by combining path with image id and date
            image_path = savedir / (results[0]['id'] + '.png')


            save_image(true_color_imgs[0], image_path , factor=3.5 / 255)



if __name__ == "__main__":

    #get these from https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings
    #or message me and I'll send them to you
    #just dont commit them to github

    client_id = ""
    client_secret = ""

    cadence = {'weekly': 'W',
               'daily': 'D'}

    satellite = {"L1C": DataCollection.SENTINEL2_L1C,
                 "L2A": DataCollection.SENTINEL2_L2A}

    year = 2019

    download_data(start_date=f'{year}-05-01',
                  end_date=f'{year}-09-10',
                  resolution=10,
                  patchname= 'Ciste_Mhearad',
                  geojson_path=Path.cwd().parent / 'get_patches' / 'Ciste_Mhearad.geojson',
                  cadence=cadence['daily'],
                  satellite=satellite['L2A'],
                  client_id=client_id,
                  client_secret=client_secret,
                  savedir='data')
