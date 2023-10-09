# shub_api_snow.py
# Python script to retrieve image using Sentinel Hub API
# Requires sentinelhub Python package/virtual environment (https://sentinelhub-py.readthedocs.io/en/latest/)
# Requires OAuth client credentials to be configured in CDSE dashboard (https://shapps.dataspace.copernicus.eu/dashboard/)
# This code is adapted from https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html
# Eddie Boyle Sep 2023
# Murray Cutforth Oct 2023

import logging
from datetime import datetime

import numpy as np
import streamlit as st

from sentinelhub import (
    SHConfig,
)

from utils import read_bbox_geojson, read_point_geojson, plot_map_with_image, get_coords_from_sel, \
    request_sentinel_image

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(layout="centered")


# Credentials
config = SHConfig()
config.sh_client_id = st.sidebar.text_input("Enter your SentinelHub client id", type="password")
config.sh_client_secret = st.sidebar.text_input("Enter your SentinelHub client secret", type="password")
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"


# Mapping of name to bbox
name_bbox_coords = read_bbox_geojson("all_patches_bbox.geojson")  # Mapping of name to [sw, ne] points in [lon, lat]
name_point_coords = read_point_geojson("all_patches_point.geojson")  # Mapping of name to [lon, lat]
logger.debug(f"name_bbox_coords: {name_bbox_coords}")
logger.debug(f"name_point_coords: {name_point_coords}")


# Structure page
st.title("Sentinel-2 RGB @ 10m resolution")
start_date = st.sidebar.date_input("Choose start date", value=datetime(2020, 6, 1))
end_date = st.sidebar.date_input("Choose end date", value=datetime(2020, 6, 30))
sel = st.sidebar.selectbox("Choose a snow patch", [name for name in name_bbox_coords])
alpha = st.sidebar.slider('Set opacity of Sentinel-2 overlay', min_value=0.0, max_value=1.0, value=0.75, step=0.01)
cloud_mask = st.sidebar.button("Show cloud mask (click me)")


# Retrieve and display image
if config.sh_client_id and config.sh_client_secret:
    gcm_coords_wgs84 = get_coords_from_sel(sel, name_bbox_coords)
    logger.debug(f"Bounding box: {gcm_coords_wgs84}")
    image = request_sentinel_image(gcm_coords_wgs84, config, start_date, end_date, cloud_mask)
    image = np.clip(image * 3.5 / 255, 0, 1)

    plot_map_with_image(image, name_bbox_coords[sel], name_point_coords[sel], alpha=alpha, name=sel)
