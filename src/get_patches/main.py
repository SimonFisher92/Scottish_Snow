import logging
import json

logging.basicConfig(level=logging.INFO)


def snowpatch_geojson(coords_dict: dict, offset: float):
    '''
    :param coords_dict: input patch names and coordinates
    :param offset: the offset from the centre of the snowpatch required to the make the bounding box ROI (larger is bigger)
    :return: geojsons for specific patches
    '''

    for name, centroid in coords_dict.items():

        top_left = [centroid[0] + offset, centroid[1] - offset]
        top_right = [centroid[0] + offset,centroid[1] + offset]
        bottom_right = [centroid[0] - offset, centroid[1] + offset]
        bottom_left = [centroid[0] - offset, centroid[1] - offset]

        # Create GeoJSON
        geojson_data = {
            "type": "Feature",
            "properties": {},
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

        # Save GeoJSON to a file
        with open(f'{name}.geojson', 'w') as f:
            json.dump(geojson_data, f, indent=4)


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
        coords_dict[coords[i].split('=')[1].strip("'\"")] = [float(coords[i+1].split('=')[3].split('&')[0]),
                                                            float(coords[i+1].split('=')[4].split('&')[0])]

    return coords_dict


if __name__ == "__main__":
    test_cords = get_coords_from_file('bluebird_coords')
    logging.info(test_cords)
    snowpatch_geojson(test_cords, offset=1e-3)

