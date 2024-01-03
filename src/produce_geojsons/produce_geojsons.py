import logging
import json
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def write_all_geojson_bbox(coords_dict: dict, offset: float = 0.00675) -> None:
    '''Write out bounding box geojson for dict of all snow patch coords

    Note: 1 degree of latitude is 111,111 metres, so a 750m offset is 0.00675 degrees.
    The default offset above gives 1.5km bounding boxes.

    :param coords_dict: input patch names and coordinates
    :param offset: the offset from the centre of the snowpatch required to the make the bounding box ROI (larger is bigger)
    :return: geojsons for specific patches
    '''

    all_data = []

    savedir = Path(__file__).parent.parent.parent / "output" / "geojson_bboxes"  # Path resolves correctly independent of where script is run from

    if not savedir.exists():
        savedir.mkdir(parents=True)

    for name, coords in coords_dict.items():

        geojson_data = construct_geojson_data(coords, name, offset)

        all_data.append(geojson_data)

        # Save individual GeoJSON to a file
        filename = savedir / f'{name}.geojson'
        with open(filename, 'w') as f:
            json.dump(geojson_data, f, indent=4)
            logging.info(f"Written file to {filename}")


def construct_geojson_data(coords: list[float, float], name: str, offset: float) -> dict:
    # Use a longitude offset which is set to give square-ish boxes, because distance per degree of longitude is not constant
    lon_correction_factor = compute_lon_correction_factor(coords[1])

    # Coords in (longitude, latitude)
    top_left = [coords[0] - offset * lon_correction_factor, coords[1] + offset]
    top_right = [coords[0] + offset * lon_correction_factor, coords[1] + offset]
    bottom_right = [coords[0] + offset * lon_correction_factor, coords[1] - offset]
    bottom_left = [coords[0] - offset * lon_correction_factor, coords[1] - offset]

    # Create GeoJSON
    geojson_data = {
        "type": "Feature",
        "properties": {"name": f"{name}"},
        "geometry": {
            "type": "Polygon",  # arbitrary
            "coordinates": [
                [
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left,
                    top_left  # Closing the polygon by repeating the first point
                ]
            ]
        }
    }

    return geojson_data


def compute_lon_correction_factor(latitude: float) -> float:
    lon_correction_factor = 1.0 / math.cos(2 * np.pi * latitude / 360)
    return lon_correction_factor


if __name__ == "__main__":
    # Read name and coords from snowpatch_coords.json
    snowpatch_coords_filepath = Path("snowpatch_coords.json")
    with open(snowpatch_coords_filepath, 'r') as f:
        snowpatch_coords = json.load(f)

    logger.info(f"Found coords for {len(snowpatch_coords)} snowpatches")
    logger.info(f"Snow patch names: {snowpatch_coords.keys()}")

    write_all_geojson_bbox(snowpatch_coords, offset=0.00675)

