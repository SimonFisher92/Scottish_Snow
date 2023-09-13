from pathlib import Path
import logging
import argparse

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sentinelsat import SentinelAPI

from src.download.download import download_products
from src.download.product_list import get_product_ids


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="A script to download Sentinel-2 data for snowpatch analysis")

    parser.add_argument("--data_dir", type=str, help="Path to dir where data should be cached", default=".")
    parser.add_argument("--geojson_path", type=str, help="Path to geojson file containing polygons covering all areas which data should be downloaded for")
    parser.add_argument("--download_full", action="store_true", help="Store the full data product, rather than just the 20m SCL band. Uses WAY more disk space.")
    parser.add_argument("--target_tile", type=str, default=None, help="Optional field to restrict data to a single tile, such as T30VVJ")
    parser.add_argument("--max_cloud_cover", type=int, default=50, help="Only get results with total cloud cover % less than this")
    parser.add_argument("--api_user", type=str, help="Username for Copernicus Sentinel API")
    parser.add_argument("--api_password", type=str, help="Password for Copernicus Sentinel API")

    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s")
    args = parse_args()

    logger.info("Starting data download code")

    prod_ids = get_product_ids(args)

    download_products(prod_ids, args)

    logger.info("All products downloaded")
    logger.info("Code completed normally")


if __name__ == "__main__":
    main()
