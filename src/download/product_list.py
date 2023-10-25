import logging
from typing import List

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt


logger = logging.getLogger(__name__)


def get_product_ids(args) -> List[str]:
    """Get a list containing the product ids of all available products which we can find.

    For a particular footprint, you still often get adjacent tiles (I think due to the overlap
    between tiles), so this function lets you filter to a particular tile).

    See https://eatlas.org.au/data/uuid/f7468d15-12be-4e3f-a246-b2882a324f59 for a useful
    interactive map of the tile grid.
    """
    api = SentinelAPI(args.api_user, args.api_password)
    footprint_geojson_filename = args.geojson_path
    target_tile = args.target_tile
    max_cloud_cover = args.max_cloud_cover
    month_range = args.month_range
    year = args.year

    footprint = geojson_to_wkt(read_geojson(footprint_geojson_filename))

    products = api.query(footprint,
                     platformname="Sentinel-2",
                     cloudcoverpercentage=(0, max_cloud_cover),)

    products_df = api.to_dataframe(products)

    logger.info(f"Initial query found {len(products_df)} products")

    if target_tile:
        # Filter to tile in post-processing - note that "tileid" column is not set for L2A products!
        # Instead look for the tile id in the title
        products_df = products_df[products_df["title"].str.contains(f"_{target_tile}_")]

        logger.info(f"After filtering to tile {target_tile}, there are {len(products_df)} products")
        assert len(products_df["tileid"].value_counts()) == 1

    if year:
        # Filter to year given in args
        assert 2000 < year < 2050
        mask_rows = year == products_df["beginposition"].dt.year
        products_df = products_df[mask_rows]
        logger.info(f"After filtering to year {year}, there are {len(products_df)} products")

    if month_range:
        # Filter to range of months given in args
        month_range = [int(x) for x in month_range.split("-")]
        assert len(month_range) == 2
        mask_rows = (month_range[0] <= products_df["beginposition"].dt.month) & (products_df["beginposition"].dt.month <= month_range[1])
        products_df = products_df[mask_rows]
        logger.info(f"After filtering to month range {month_range}, there are {len(products_df)} products")

    assert len(products_df["orbitdirection"].value_counts()) == 1
    assert len(products_df["instrumentname"].value_counts()) == 1

    products_2a_df = products_df[(products_df["producttype"] == "S2MSI2A") | (products_df["producttype"] == "S2MSI2Ap")]

    logger.info(f"After filtering to level 2A products, there are {len(products_2a_df)} products")

    return list(products_2a_df.index)

