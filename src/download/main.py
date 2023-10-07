import logging
import argparse

from src.download.download import download_products
from src.download.product_list import get_product_ids


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
            description="A script to download Sentinel-2 data for snowpatch analysis. Example command: \npython -m src.download.main --data_dir='data' --geojson_path='input/cairngorms_footprint.geojson' --product_filter='*B0[234]_10m.jp2' --target_tile='T30VVJ' --api_user=<> --api_password=<>")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to dir where data should be cached", default=".")
    parser.add_argument("--geojson_path", type=str, required=True, help="Path to geojson file containing polygons covering all areas which data should be downloaded for")
    parser.add_argument("--product_filter", type=str, default="*SCL_20m.jp2", help=r"Path filter which is passed to sentinelsat.SentinelAPI.download(). The default is '*SCL_20m.jp2' which only downloads SCL masks at 20m resolution. For all data use '*'. For 10m resolution RGB use '*B0[234]_10m.jp2'. See documentation at https://sentinelsat.readthedocs.io/en/latest/api_overview.html#downloading-parts-of-products.")
    parser.add_argument("--target_tile", type=str, default=None, help="Optional field to restrict data to a single tile, such as T30VVJ")
    parser.add_argument("--max_cloud_cover", type=int, default=50, help="Only get results with total cloud cover %% less than this")
    parser.add_argument("--month_range", type=str, default=None, help="Only get results from months in this range. Should be a string such as 4-10")
    parser.add_argument("--api_user", type=str, required=True, help="Username for Copernicus Sentinel API")
    parser.add_argument("--api_password", type=str, required=True, help="Password for Copernicus Sentinel API")

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
