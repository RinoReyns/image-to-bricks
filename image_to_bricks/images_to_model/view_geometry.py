"""Orthographic views around the character: the four camera directions and how mesh points project into them.

Coordinates: y up; a camera at "+z" looks at the origin along -z (the glTF front), "+x" from the character's left
when it faces +z, and so on, 90 degrees apart.
"""

import numpy as np
from PIL import Image

from image_to_bricks.images_to_model.type_aliases import BoolArray, FloatArray
from image_to_bricks.images_to_model.views import FRONT_VIEW

DIRECTIONS = ("+z", "+x", "-z", "-x")  # camera positions, 90 degrees apart around the up (y) axis
DIRECTION_VECTORS = {"+z": (0, 0, 1), "-z": (0, 0, -1), "+x": (1, 0, 0), "-x": (-1, 0, 0)}
QUARTER_TURNS_TO_BACK = 2  # the back view is two quarter turns from the front
QUARTER_TURNS_PER_TURN = len(DIRECTIONS)
OPAQUE_ALPHA = 128  # alpha 0-255: at or above = part of the character
FULL_BYTE = 255.0


def project_vertices(vertices: FloatArray, direction: str) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Orthographic camera at `direction`, looking at the origin, y up. Returns (screen x, screen y, depth; larger
    depth = closer to the camera), each [n]."""
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    mapping = {"+z": (x, z), "-z": (-x, -z), "+x": (-z, x), "-x": (z, -x)}
    screen_x, depth = mapping[direction]
    return screen_x, y, depth


def assign_view_directions(front_index: int, side_sign: int) -> dict[str, str]:
    """Camera direction of each named view when the front picture is DIRECTIONS[front_index]; side_sign +1 or -1
    says on which side 'left' is."""
    return {
        FRONT_VIEW: DIRECTIONS[front_index % QUARTER_TURNS_PER_TURN],
        "back": DIRECTIONS[(front_index + QUARTER_TURNS_TO_BACK) % QUARTER_TURNS_PER_TURN],
        "left": DIRECTIONS[(front_index + side_sign) % QUARTER_TURNS_PER_TURN],
        "right": DIRECTIONS[(front_index - side_sign) % QUARTER_TURNS_PER_TURN],
    }


def split_rgba(image: Image.Image) -> tuple[FloatArray, BoolArray]:
    """(RGB 0-1 [rows, columns, 3], character mask [rows, columns]) of a picture."""
    pixels = np.asarray(image.convert("RGBA"), dtype=np.float64)
    return pixels[..., :3] / FULL_BYTE, np.asarray(pixels[..., 3] >= OPAQUE_ALPHA, dtype=bool)


def compute_mask_iou(first: BoolArray, second: BoolArray) -> float:
    """Intersection over union of two masks of the same shape (0 when both are empty)."""
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0
