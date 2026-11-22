"""Flat side views ("elevations") of the finished model, drawn from the placed pieces.

Views are named by the FIGURE's own sides, using --front (model.front): 'front' = the face, 'back', 'left' =
the figure's left side, 'right' = the figure's right side. Each picture is drawn as a camera standing on that
side sees it (z up), so some grid axes run right-to-left; the axis numbers stay FIGURE-grid coordinates used by
--paint / --thicken-box / --accent / --retile-box (z = plate layer above the base, x / y = stud; the base plates
and base margin are not counted), so a coordinate read from the picture can be pasted into a box option
unchanged. Layer pages of the PDF count from the base plate corner instead (see COLOR_FIX_PLAN.md, step 34).
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from image_to_bricks.consts import LEGO_RGB
from image_to_bricks.instructions.view_sides import (
    FIGURE_SIDES,
    GroundVector,
    front_or_default,
    screen_right_vector,
    side_vector,
)
from image_to_bricks.model import LegoModel, PlacedBrick, piece_mask

PIXELS_PER_STUD = 22
PIXELS_PER_PLATE = 9
LEFT_MARGIN_PIXELS = 34  # room for the z labels
TOP_MARGIN_PIXELS = 30  # room for the title and the horizontal labels
CROP_PADDING_CELLS = 1  # empty cells kept around the figure in a view
MIN_COLUMNS_FOR_DIRECTION = 2  # a view needs two columns to say which way its axis runs
UNKNOWN_COLOUR_RGB = (255, 0, 255)
SIDE_TEXT = {"front": "front (face)", "back": "back", "left": "left side", "right": "right side"}


def colour_cube(bricks: Sequence[PlacedBrick]) -> NDArray[np.int32]:
    """Colour id per cell, int32 [z, y, x] in model-grid coordinates (0 = empty), from the placed pieces."""
    layer_count = max(b.z + b.h_plates for b in bricks)
    depth = max(b.y + b.d for b in bricks)
    width = max(b.x + b.w for b in bricks)
    cube = np.zeros((layer_count, depth, width), np.int32)
    for b in bricks:
        piece_cells = cube[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w]
        if b.notch >= 0:
            piece_cells[:, piece_mask(b)] = b.color  # L piece: not the notch cell
        else:
            piece_cells[:] = b.color
    return cube


@dataclass(frozen=True)
class Projection:
    """What a camera on one side sees: colour id per (z, picture column) and the grid coordinate of each
    picture column, columns ordered left to right as the viewer sees them."""

    face_ids: NDArray[np.int32]  # [z, column], 0 = nothing
    column_coordinates: NDArray[np.int_]  # [column], figure-grid x or y of each picture column
    horizontal_axis: str  # 'x' or 'y'


def project_nearest(figure_cube: NDArray[np.int32], camera_side: GroundVector) -> Projection:
    """Nearest filled cell per line of sight for a camera standing at camera_side (a ground unit vector along
    x or y) looking at the figure. figure_cube: colour ids [z, y, x]."""
    looks_along_y = camera_side[0] == 0
    cells = figure_cube if looks_along_y else figure_cube.transpose(0, 2, 1)  # -> [z, depth, horizontal]
    horizontal_axis = "x" if looks_along_y else "y"
    toward_camera_sign = camera_side[1] if looks_along_y else camera_side[0]
    right = screen_right_vector(camera_side)
    rightward_sign = right[0] if looks_along_y else right[1]
    if toward_camera_sign < 0:
        cells = cells[:, ::-1, :]  # depth index now grows toward the camera
    column_coordinates: NDArray[np.int_] = np.arange(cells.shape[2])
    if rightward_sign < 0:
        cells = cells[:, :, ::-1]
        column_coordinates = column_coordinates[::-1]
    filled = cells > 0
    nearest_depth = cells.shape[1] - 1 - np.argmax(filled[:, ::-1, :], axis=1)
    face_ids = np.take_along_axis(cells, nearest_depth[:, None, :], axis=1)[:, 0, :]
    face_ids[~filled.any(axis=1)] = 0
    return Projection(face_ids.astype(np.int32), column_coordinates, horizontal_axis)


def describe_view(side: str, camera_side: GroundVector, projection: Projection) -> str:
    """Title line: which side of the figure, where the camera stands, how the horizontal axis runs."""
    depth_axis = "y" if projection.horizontal_axis == "x" else "x"
    camera_sign = camera_side[1] if depth_axis == "y" else camera_side[0]
    coordinates = projection.column_coordinates
    increasing = len(coordinates) < MIN_COLUMNS_FOR_DIRECTION or coordinates[1] > coordinates[0]
    return (
        f"{side} view = the figure's {SIDE_TEXT[side]}, camera at {'high' if camera_sign > 0 else 'low'} "
        f"{depth_axis}; horizontal = {projection.horizontal_axis} "
        f"({'increasing' if increasing else 'decreasing'} to the right), vertical = z"
    )


def render_view(
    cube: NDArray[np.int32], side: str, *, base_layers: int = 0, margin: int = 0, front: str | None = None
) -> Image.Image:
    """Picture of one side of the figure ('front', 'back', 'left', 'right' = the figure's own sides, from
    front = the --front value, default view_sides.DEFAULT_FRONT). cube: colour ids [z, y, x] in model-grid
    coordinates; base_layers / margin are removed so the labels are figure-grid numbers. Cropped to the figure."""
    camera_side = side_vector(side, front_or_default(front))
    projection = project_nearest(cube[base_layers:, margin:, margin:], camera_side)
    face = projection.face_ids
    drawn_columns = np.nonzero((face > 0).any(axis=0))[0]
    drawn_rows = np.nonzero((face > 0).any(axis=1))[0]
    if len(drawn_columns) == 0:
        return Image.new("RGB", (80, 40), "white")
    first_column = max(int(drawn_columns.min()) - CROP_PADDING_CELLS, 0)
    last_column = min(int(drawn_columns.max()) + CROP_PADDING_CELLS, face.shape[1] - 1)
    lowest_z = max(int(drawn_rows.min()) - CROP_PADDING_CELLS, 0)
    highest_z = min(int(drawn_rows.max()) + CROP_PADDING_CELLS, face.shape[0] - 1)
    image = Image.new(
        "RGB",
        (
            LEFT_MARGIN_PIXELS + (last_column - first_column + 1) * PIXELS_PER_STUD + 6,
            TOP_MARGIN_PIXELS + (highest_z - lowest_z + 1) * PIXELS_PER_PLATE + 6,
        ),
        "white",
    )
    draw = ImageDraw.Draw(image)
    draw.text((2, 2), describe_view(side, camera_side, projection), fill="black")
    for z in range(lowest_z, highest_z + 1):
        pixel_y = TOP_MARGIN_PIXELS + (highest_z - z) * PIXELS_PER_PLATE
        for column in range(first_column, last_column + 1):
            colour_id = int(face[z, column])
            if colour_id:
                pixel_x = LEFT_MARGIN_PIXELS + (column - first_column) * PIXELS_PER_STUD
                draw.rectangle(
                    [pixel_x, pixel_y, pixel_x + PIXELS_PER_STUD - 1, pixel_y + PIXELS_PER_PLATE - 1],
                    fill=LEGO_RGB.get(colour_id, UNKNOWN_COLOUR_RGB),
                    outline=(80, 80, 80),
                )
        if z % 5 == 0:
            draw.line(
                [
                    LEFT_MARGIN_PIXELS - 4,
                    pixel_y + PIXELS_PER_PLATE // 2,
                    LEFT_MARGIN_PIXELS - 1,
                    pixel_y + PIXELS_PER_PLATE // 2,
                ],
                fill="black",
            )
            draw.text((2, pixel_y - 1), str(z), fill="black")
    for column in range(first_column, last_column + 1):
        pixel_x = LEFT_MARGIN_PIXELS + (column - first_column) * PIXELS_PER_STUD
        draw.text((pixel_x + 5, 16), str(int(projection.column_coordinates[column])), fill="black")
    return image


def save_elevations(model: LegoModel, outdir: str, model_name: str, margin: int = 0) -> list[str]:
    """Writes <model_name>_view_front / back / left / right.png (the figure's own sides, from model.front)
    into outdir; returns the paths."""
    cube = colour_cube(model.bricks)
    front = getattr(model, "front", None)
    paths = []
    for side in FIGURE_SIDES:
        path = os.path.join(outdir, f"{model_name}_view_{side}.png")
        render_view(cube, side, base_layers=model.base_layers, margin=margin, front=front).save(path)
        paths.append(path)
    return paths
