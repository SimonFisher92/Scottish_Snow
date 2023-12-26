import logging
import json
import math
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def write_geojson_bbox(coords_dict: dict, offset: float = 0.00675):
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

    for name, centroid in coords_dict.items():

        # Use a longitude offset which is set to give square boxes, because distance per degree of longitude is not constant
        correction = 1.0 / math.cos(2 * 3.141592 * centroid[1] / 360)

        # Coords in (longitude, latitude)
        top_left = [centroid[0] - offset * correction, centroid[1] + offset]
        top_right = [centroid[0] + offset * correction, centroid[1] + offset]
        bottom_right = [centroid[0] + offset * correction, centroid[1] - offset]
        bottom_left = [centroid[0] - offset * correction, centroid[1] - offset]

        # Create GeoJSON
        geojson_data = {
            "type": "Feature",
            "properties": {"name": f"{name}"},
            "geometry": {
                "type": "Polygon", #arbitrary
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

        all_data.append(geojson_data)

        # Save individual GeoJSON to a file
        filename = savedir / f'{name}.geojson'
        with open(filename, 'w') as f:
            json.dump(geojson_data, f, indent=4)
            logging.info(f"Written file to {filename}")


def get_coords_from_file(filepath: str) -> dict:
    '''
    :param filepath: path to file containing coordinates of snowpatches from Iain, badly coded during Bluebird
    as they were intended to directly pull up playground data for webscraping
    :return: a dictionary of patch name, lat and lon
    '''

    coords=[]
    with open(filepath, 'r') as file:
        for line in file:
            coords.append(line.strip())

    coords_dict = {}

    indices = range(0, len(coords)-1, 2)

    for i in indices:
        coords_dict[coords[i].split('=')[1].strip("'\"")] = [float(coords[i+1].split('=')[4].split('&')[0]),
                                                            float(coords[i+1].split('=')[3].split('&')[0])]

    return coords_dict


if __name__ == "__main__":
    test_cords = get_coords_from_file('bluebird_coords.txt')
    logging.info(test_cords)
    write_geojson_bbox(test_cords)

