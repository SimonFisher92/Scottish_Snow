from typing import List, Tuple
from dataclasses import dataclass
from math import atan2
from pathlib import Path

import numpy as np
import xarray as xr

from src.measure_cls_band.polygon import Polygon



def compute_a(v0: np.array, v1: np.array) -> np.array:
    """a is a unit length vector pointing from v0 to v1
    """
    a = (v1 - v0) / np.linalg.norm(v1 - v0)
    return a


def compute_cm(a: np.array, v0: np.array, p_grid) -> float:
    """cm defines the point v0 + cm * a, which is the closest point on the line to p
    """
    assert len(p_grid.shape) == 3
    assert p_grid.shape[2] == 2

    cm = np.dot(p_grid, a) - np.dot(a, v0)
    return cm


def compute_m(v0, cm, a) -> np.array:
    """m is the closest point to p on the line

    Args:
        v0: (2,) array
        cm: (N, N) array
        a: (2,) array
    """
    m = np.expand_dims(cm, axis=-1) * a + v0

    assert m.shape[:2] == cm.shape
    assert len(m.shape) == 3
    assert m.shape[2] == 2

    return m


def is_inside_vectorised(x_grid, y_grid, v0, v1) -> np.array:
    """Vectorised function which which takes a 2D grid of x- and y-coordinates,
    and returns a boolean array denoting whether they are inside the half
    plane defined by the vectors v0 and v1.

    The half plane is defined by the line passing through these two points.
    The "outside" region is the half plane which the vector (v1 - v0), rotated
    by pi/2 anticlockwise, points INTO.
    """
    a = compute_a(v0, v1)  # a is the unit vector from v0 to v1

    p_grid = np.stack([x_grid, y_grid], axis=-1)

    cm = compute_cm(a, v0, p_grid)  # cm is a grid of scalars

    m = compute_m(v0, cm, a)  # m is a grid of vectors

    R = np.array([[0, -1], [1, 0]], dtype=np.float32)

    # Now R @ a points towards the INSIDE region
    # p - m points from the boundary to the point
    # If they point in the same direction, then p must be inside
    # Otherwise, p is outside
    s = np.dot((p_grid - m), R @ a)  # s is an inside/outside indicator function, grid of scalars
    
    return s >= 0
    



def convex_polygon_mask(polygon: Polygon, data: xr.Dataset):
    """In this function we want to obtain a boolean mask of all cells which are inside
    the polygon. Polygon must be provided in UTM coords, using same zone as measurement array.
    """
    assert polygon.is_convex()

    p = Polygon(vertices=[
        (437835.64, 6318955.21),
        (490385.33, 6319486.61),
        (470214.34, 6360601.50),
        (438212.63, 6380287.29),
        ])

    # Start with array of all true values
    mask_np = np.ones(shape=(len(data.x), len(data.y)), dtype=bool)

    # Equivalent of np.meshgrid to get 2D grids of coordinates
    x_da = data.x
    y_da = data.y
    x_grid = x_da.expand_dims(y=y_da.sizes["y"]).to_numpy()
    y_grid = y_da.expand_dims(x=x_da.sizes["x"]).to_numpy().T  # Note the transpose! This caused me some pain.

    for v_n0, v_n1 in zip(polygon.vertices, polygon.vertices[1:] + [polygon.vertices[0]]):
        mask_np = np.logical_and(mask_np, is_inside_vectorised(x_grid, y_grid, np.array(v_n0), np.array(v_n1)))

    data["mask"] = xr.DataArray(data=mask_np, dims=["y", "x"], coords={"lon": (["x", "y"], x_grid), "lat": (["x", "y"], y_grid)})



