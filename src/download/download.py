from typing import List
import logging
import concurrent.futures
import argparse
import time

from tqdm import tqdm
from sentinelsat import SentinelAPI, make_path_filter


logger = logging.getLogger(__name__)


def _download_prod_inner(prod_id, cdir, api, path_filter, retry_delay=3600):
    """This function tries to download a given product. Will retry every 30 mins until download succeeds. In particular, this function will keep retrying if a product is offline.
    """
    logger.info(f"Attempting download of {prod_id} to {cdir}")

    while True:
        try:
            if path_filter:
                path_filter = make_path_filter(path_filter)
                api.download(prod_id, cdir, nodefilter=path_filter)
            else:
                api.download(prod_id, cdir)
            break
        except Exception as e:
            # This catches the LTA error if product is offline, as well as anything else (transient network errors?)
            logger.error(f"Download failed with {e}. Retrying in {retry_delay}s.")
            time.sleep(retry_delay)


def download_products(prod_ids: List, args: argparse.Namespace):
    """Download all products, including triggering the LTA.

    There is a quota which we believe limits one user to requesting 20 products per 12 hours.

    TODO: if a product is already downloaded, but is now offline, how does the caching work? Does the code get stuck requesting retrieval again?
    """
    api = SentinelAPI(args.api_user, args.api_password)
    cdir = args.data_dir
    path_filter = args.product_filter

    product_infos = {}
    for prod_id in tqdm(prod_ids, desc="Retrieving product info"):
        product_infos[prod_id] = api.get_product_odata(prod_id)

    logger.info(f"There are a total of {sum(v['Online'] for v in product_infos.values())}/{len(product_infos)} online products")

    if args.num_threads <= 0:
        # Run downloads one at a time, no concurrency
        for pid in tqdm(prod_ids, desc="Downloading products (no concurrency)"):
            logger.debug(f"Calling download_prod_inner with pid={pid}, cdir={cdir}, api={api}, filter={path_filter}")
            _download_prod_inner(prod_id=pid, cdir=cdir, api=api, path_filter=path_filter)
            logger.debug(f"Download complete for pid={pid}")

    else:
        # Try to request products concurrently - I'm suspicious of this. Is the api object thread safe?
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            for pid in prod_ids:
                executor.submit(_download_prod_inner, prod_id=pid, cdir=cdir, api=api, path_filter=path_filter)

    logger.info("All products downloaded")

