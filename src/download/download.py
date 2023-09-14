from typing import List, Tuple
from dataclasses import dataclass
from math import atan2
from pathlib import Path
import logging
import queue
import concurrent.futures
import argparse
import time

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sentinelsat import SentinelAPI, make_path_filter


logger = logging.getLogger(__name__)


def _check_previous_download(cdir, title):
    """If a .SAFE folder already exists, skip this product
    """
    downloaded_titles = set(x.stem for x in Path(cdir).glob("*.SAFE"))
    return title in downloaded_titles


def _download_prod_inner(prod_id, product_info, cdir, api, path_filter, retry_delay=1800):
    """This function tries to download a given product. Will retry every 30 mins until download succeeds. In particular, this function will keep retrying if a product is offline.
    """
    if _check_previous_download(cdir, product_info["title"]):
        logger.info(f"Found existing download of {product_info['title']}; skipping.")
        return

    logger.info(f"Attempting download of {prod_id} to {cdir}")

    while True:
        try:
            path_filter = make_path_filter(path_filter)
            api.download(prod_id, cdir, nodefilter=path_filter)
            break
        except Exception as e:
            logger.error(f"Download failed with {e}. Retrying in 30mins.")
            time.sleep(retry_delay)


def download_products(prod_ids: List, args: argparse.Namespace):
    """Download all products

    Note: I don't have a reference, but I have found that up to 20 products at a time can be triggered for LTA
    before hitting a quota.
    """
    api = SentinelAPI(args.api_user, args.api_password)

    product_infos = {}
    for prod_id in tqdm(prod_ids, desc="Retrieving product info"):
        product_infos[prod_id] = api.get_product_odata(prod_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for pid in prod_ids:
            executor.submit(_download_prod_inner, pid, product_infos[pid], args.data_dir, api, args.product_filter)

    logger.info("All products downloaded")

