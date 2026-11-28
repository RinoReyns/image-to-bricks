"""Step 5: lowest point at y = 0, centred in x and z, scaled to a height."""

from typing import NamedTuple

import numpy as np

from image_to_bricks.images_to_model.clean import compute_volume
from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.type_aliases import FloatArray, JsonDict

ROUNDING_DIGITS = 5  # measurements in the report
FINE_ROUNDING_DIGITS = 6  # volume and position in the report


class Placement(NamedTuple):
    """vertices * scale + offset is the placed mesh (uniform scale, offset [3])."""

    scale: float
    offset: FloatArray


def compute_placement(vertices: FloatArray, height: float) -> Placement:
    """Scale and offset that put the lowest point at y = 0, centre x and z and make the mesh `height` tall (same unit
    as height). Raises ValueError for a height <= 0 or a flat mesh."""
    if height <= 0:
        raise ValueError("height must be positive")
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    extent = float(high[1] - low[1])
    if extent <= 0:
        raise ValueError("mesh has no height")
    scale = height / extent
    centre = (low + high) / 2.0
    offset = np.array([-centre[0] * scale, -low[1] * scale, -centre[2] * scale])
    return Placement(scale, offset)


def apply_placement(vertices: FloatArray, scale: float, offset: FloatArray) -> FloatArray:
    """vertices [n, 3] * scale + offset."""
    return np.asarray(np.asarray(vertices, dtype=np.float64) * scale + offset, dtype=np.float64)


def measure_mesh(mesh: Mesh) -> JsonDict:
    """Width (x), height (y), depth (z), volume, floor height and x / z centre, rounded for the report."""
    low, high = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    return {
        "width": round(float(high[0] - low[0]), ROUNDING_DIGITS),
        "height": round(float(high[1] - low[1]), ROUNDING_DIGITS),
        "depth": round(float(high[2] - low[2]), ROUNDING_DIGITS),
        "volume": round(compute_volume(mesh), FINE_ROUNDING_DIGITS),
        "min_y": round(float(low[1]), FINE_ROUNDING_DIGITS),
        "centre_x": round(float((low[0] + high[0]) / 2), FINE_ROUNDING_DIGITS),
        "centre_z": round(float((low[2] + high[2]) / 2), FINE_ROUNDING_DIGITS),
    }
