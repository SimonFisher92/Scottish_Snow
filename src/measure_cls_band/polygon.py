from typing import List, Tuple
from dataclasses import dataclass
from math import atan2
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

TWO_PI = 2 * np.pi


@dataclass
class Polygon:
    vertices: List[Tuple[float, float]]


    def area(self) -> float:
        """Apply shoelace formula for area of polygon. Result will be negative if vertices are clockwise.
        """
        area = 0.0

        for v_n0, v_n1 in zip(self.vertices, self.vertices[1:] + [self.vertices[0]]):
            x0 = v_n0[0]
            x1 = v_n1[0]
            y0 = v_n0[1]
            y1 = v_n1[1]
            area += (x0 + x1) * (y0 - y1)

        area /= - 2.0

        return area

    def is_convex(self) -> bool:
        """Copied from stack overflow.
        
        Return True if the polynomial defined by the sequence of 2D
        points is 'strictly convex': points are valid, side lengths non-
        zero, interior angles are strictly between zero and a straight
        angle, and the polygon does not intersect itself.

        NOTES:  1.  Algorithm: the signed changes of the direction angles
                    from one side to the next side must be all positive or
                    all negative, and their sum must equal plus-or-minus
                    one full turn (2 pi radians). Also check for too few,
                    invalid, or repeated points.
                2.  No check is explicitly done for zero internal angles
                    (180 degree direction-change angle) as this is covered
                    in other ways, including the `n < 3` check.
        """
        try:  # needed for any bad points or direction changes
            # Check for too few points
            if len(self.vertices) < 3:
                return False
            # Get starting information
            old_x, old_y = self.vertices[-2]
            new_x, new_y = self.vertices[-1]
            new_direction = atan2(new_y - old_y, new_x - old_x)
            angle_sum = 0.0
            # Check each point (the side ending there, its angle) and accum. angles
            for ndx, newpoint in enumerate(self.vertices):
                # Update point coordinates and side directions, check side length
                old_x, old_y, old_direction = new_x, new_y, new_direction
                new_x, new_y = newpoint
                new_direction = atan2(new_y - old_y, new_x - old_x)
                if old_x == new_x and old_y == new_y:
                    return False  # repeated consecutive points
                # Calculate & check the normalized direction-change angle
                angle = new_direction - old_direction
                if angle <= -np.pi:
                    angle += TWO_PI  # make it in half-open interval (-Pi, Pi]
                elif angle > np.pi:
                    angle -= TWO_PI
                if ndx == 0:  # if first time through loop, initialize orientation
                    if angle == 0.0:
                        return False
                    orientation = 1.0 if angle > 0.0 else -1.0
                else:  # if other time through loop, check orientation is stable
                    if orientation * angle <= 0.0:  # not both pos. or both neg.
                        return False
                # Accumulate the direction-change angle
                angle_sum += angle
            # Check that the total number of full turns is plus-or-minus 1
            return abs(round(angle_sum / TWO_PI)) == 1
        except (ArithmeticError, TypeError, ValueError):
            return False  # any exception means not a proper convex polygon
