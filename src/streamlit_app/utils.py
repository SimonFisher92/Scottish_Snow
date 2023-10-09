# Utilities required for Sentinel Hub API
# Eddie Boyle Sep 2023
import json
import math
from typing import Any, Optional, Tuple
import logging

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import folium
from sentinelhub import SHConfig, BBox, CRS, bbox_to_dimensions, SentinelHubRequest, DataCollection, MosaickingOrder, \
    MimeType
from streamlit_folium import st_folium

logger = logging.getLogger(__name__)


def plot_map_with_image(
        image: np.ndarray,
        bbox: list[list, list],  # SW, NE points in [lon, lat] ordering
        centre: list,  # Centre point as [lon, lat] ordering
        alpha: float = 1.0,
        name: str = None
):
    attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
    tiles = 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png'

    m = folium.Map(
        location=[centre[1], centre[0]],  # Folium works in (lat, lon) so we need to switch
        tiles=tiles,
        attr=attr,
        zoom_control=True,
        control_scale=True,
    )

    folium.Marker(
        location=centre[::-1],
        tooltip=f"{name} snow patch",
        icon=folium.Icon(),
    ).add_to(m)

    bbox = [x[::-1] for x in bbox]  # Convert from (lon, lat) to (lat, lon) for folium

    sw_point = bbox[0]
    ne_point = bbox[1]
    logger.debug(f"SW point of image: {sw_point}")
    logger.debug(f"NE point of image: {ne_point}")
    m.fit_bounds([sw_point, ne_point])  # SW, NE corners in (lat, lon) form

    folium.raster_layers.ImageOverlay(
        name="Sentinel-2 imagery",
        image=image,
        bounds=bbox,
        opacity=alpha,
        mercator_project=True,
    ).add_to(m)

    kw = {
        "color": "blue",
        "line_cap": "round",
        "fill": False,
        "weight": 1,
        "popup": "Sentinel-2 Image",
    }
    folium.Rectangle(
        bounds=[sw_point, ne_point],
        line_join="round",
        dash_array="5, 5",
        **kw,
    ).add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(m, height=600, width=600, use_container_width=True)


def read_bbox_geojson(path: str) -> dict[str, list]:
    # Parse the JSON data into a dictionary
    with open(path, "r") as infile:
        data = json.load(infile)

    # Dictionary to store name and coordinates
    name_coordinates = {}

    # Iterate through features and extract name and SW / NE coordinates
    for feature in data['features']:
        name = feature['properties']['name']
        coordinate_list = feature["geometry"]["coordinates"][0]
        lon_list = [x[0] for x in coordinate_list]
        lat_list = [x[1] for x in coordinate_list]
        sw = [min(lon_list), min(lat_list)]
        ne = [max(lon_list), max(lat_list)]
        coordinates = [sw, ne]
        name_coordinates[name] = coordinates

    return name_coordinates


def read_point_geojson(path: str) -> dict[str, list[float]]:
    # Parse the JSON data into a dictionary
    with open(path, "r") as infile:
        data = json.load(infile)

    # Dictionary to store name and coordinates
    name_coordinates = {}

    # Iterate through features and extract name and top left / bottom right coordinates
    for feature in data['features']:
        name = feature['properties']['name']
        coordinates = feature['geometry']['coordinates']
        name_coordinates[name] = coordinates

    return name_coordinates


def get_coords_from_sel(sel, name_bbox_coords) -> tuple[float, float, float, float]:
    assert sel in name_bbox_coords
    result = [
        name_bbox_coords[sel][0][0],
        name_bbox_coords[sel][0][1],
        name_bbox_coords[sel][1][0],
        name_bbox_coords[sel][1][1],
    ]
    return tuple(result)


@st.cache_data
def request_sentinel_image(gcm_coords_wgs84: tuple, config: SHConfig, start_date, end_date, cloud_mask: bool) -> np.ndarray:
    resolution = 10
    gcm_bbox = BBox(bbox=gcm_coords_wgs84, crs=CRS.WGS84)
    gcm_size = bbox_to_dimensions(gcm_bbox, resolution=resolution)

    logger.info(f"Image shape at {resolution} m resolution: {gcm_size} pixels")

    if cloud_mask:
        evalscript_true_color = """
        //VERSION=3
        function setup() {
          return {
            input: ["B02", "B03", "B04", "CLM"],
            output: { bands: 3 }
          }
        }

        function evaluatePixel(sample) {
          if (sample.CLM == 1) {
            return [0.75 + sample.B04, sample.B03, sample.B02]
          }
          return [sample.B04, sample.B03, sample.B02];
        }
        """
    else:
        # Evalscript to select the RGB (B04, B03, B02) Sentinel-2 L1C bands.
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

    # PNG image from Jun 2020. Least cloud cover mosaicking used. Reflectance values in UINT8 format (values in 0-255 range).
    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C.define_from(
                    "s2l1c", service_url=config.sh_base_url
                ),
                time_interval=(start_date, end_date),
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=gcm_bbox,
        size=gcm_size,
        config=config,
    )

    return request_true_color.get_data()[0]
