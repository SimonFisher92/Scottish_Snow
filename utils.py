import logging

logging.basicConfig(level=logging.INFO)

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

