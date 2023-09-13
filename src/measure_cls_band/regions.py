from typing import List, Dict, Tuple
import logging

import utm

from src.measure_cls_band.polygon import Polygon

logger = logging.getLogger(__name__)


def get_rois_lat_lon():
    """This function specifies ROI polygons in lat/lon coordinates, which are hardcoded in.

    Note: At present, all polygons must be convex
    """
    name_to_coords = {
        "northern_cairngorms": [
            (57.114301, -3.929280),
            (57.162312, -3.714076),
            (57.152050, -3.358325),
            (57.006323, -3.397939),
            (57.003096, -3.902027)
        ],
        "monadh_liath": [
            (57.236986, -4.106196),
            (57.105345, -3.974958),
            (57.019243, -4.283127),
            (57.130311, -4.418831)
        ],
        "southern_cairngorms": [
            (57.048123, -3.037983),
            (56.871318, -3.177420),
            (56.814204, -3.460366),
            (57.006297, -3.398833)
        ],
        "beinn_dearg": [
            (56.898559, -3.939602),
            (56.901257, -3.836427),
            (56.873347, -3.821048),
            (56.828310, -3.831984),
            (56.850181, -3.917934),
        ]
    }

    # Check convexity
    for name, coords in name_to_coords.items():
        p = Polygon(vertices=coords)
        assert p.is_convex(), f"{name} is not convex"
        logger.debug(f"Convexity check passed for {name}")

    # Re-order vertices to clockwise in lat/lon coords if necessary
    for name, coords in name_to_coords.items():
        p = Polygon(vertices=coords)
        if p.area() > 0:
            name_to_coords[name] = coords[::-1]

    for coords in name_to_coords.values():
        p = Polygon(vertices=coords)
        assert p.area() <= 0.0

    return name_to_coords


def convert_list_to_utm(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Convert a list of lat/lon coordinates to UTM coordinates. Discards the UTM zone information."""
    return [utm.from_latlon(*coord)[:2] for coord in coords]


def get_rois() -> Dict[str, Polygon]:
    """Return dictionary of all ROIs in UTM coordinates
    """
    rois_lat_lon = get_rois_lat_lon()
    rois_utm = {k: convert_list_to_utm(v) for k, v in rois_lat_lon.items()}
    rois = {k: Polygon(vertices=v) for k, v in rois_utm.items()}

    for k, p in rois.items():
        logger.debug(f"Region {k} has area {p.area()} m^2")
        
    return rois

