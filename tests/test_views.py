"""Flat views: named by the figure's own sides (from --front), drawn as seen, axis numbers = figure grid.

Fixture: a Black body with two asymmetric markers, built in figure-relative coordinates and placed in the grid
for every --front value: a Red "raised right paw" on the figure's RIGHT side and a Yellow "nose" on its face.
(Checked once against photos of a real build: front view = the photo with the raised paw on the viewer's left,
the figure's left side = the photo with the face to the viewer's left.)
"""

import os
import sys
import tempfile

import support

support.stub_mesh_libraries()
from types import SimpleNamespace as NS

import numpy as np

from image_to_bricks.consts import LEGO_RGB
from image_to_bricks.instructions import elevation, view_sides

BLACK, RED, YELLOW, ORANGE = 6, 2, 5, 9
GRID_SIZE = 9
CENTRE = 4


def place(forward: int, leftward: int, front: str):
    """Grid (x, y) of a point given in figure-relative studs from the centre (forward = towards the face,
    leftward = towards the figure's own left)."""
    facing = view_sides.FRONT_VECTORS[front]
    left = view_sides.figure_left_vector(front)
    return (CENTRE + forward * facing[0] + leftward * left[0], CENTRE + forward * facing[1] + leftward * left[1])


def marker_figure(front: str) -> np.ndarray:
    """Colour ids [z, y, x]: 3x3 Black body (z 0-2), Red paw 2 studs to the figure's right, Yellow nose 2 studs
    in front, both at z 1."""
    cube = np.zeros((3, GRID_SIZE, GRID_SIZE), np.int32)
    for forward in (-1, 0, 1):
        for leftward in (-1, 0, 1):
            x, y = place(forward, leftward, front)
            cube[:, y, x] = BLACK
    paw_x, paw_y = place(0, -2, front)
    nose_x, nose_y = place(2, 0, front)
    cube[1, paw_y, paw_x] = RED
    cube[1, nose_y, nose_x] = YELLOW
    return cube


def horizontal_position(image, colour_id: int):
    """Mean picture x of the pixels of a colour (below the title), as a fraction of the drawn area, or None."""
    pixels = np.array(image.convert("RGB"))[elevation.TOP_MARGIN_PIXELS :, elevation.LEFT_MARGIN_PIXELS :]
    columns = np.nonzero((pixels == LEGO_RGB[colour_id]).all(axis=-1))[1]
    return None if len(columns) == 0 else columns.mean() / pixels.shape[1]


def check_marker_sides(front: str) -> list:
    """Raised right paw and nose land on the side a viewer standing there sees them."""
    cube = marker_figure(front)
    views = {side: elevation.render_view(cube, side, front=front) for side in view_sides.FIGURE_SIDES}
    paw = {side: horizontal_position(image, RED) for side, image in views.items()}
    nose = {side: horizontal_position(image, YELLOW) for side, image in views.items()}
    return [
        paw["front"] is not None and paw["front"] < 0.4,  # facing the figure: its right paw on our left
        nose["front"] is not None,
        paw["back"] is not None and paw["back"] > 0.6,  # from behind: its right paw on our right
        nose["back"] is None,
        paw["right"] is not None and nose["right"] > 0.6,  # its right side: paw seen, face to our right
        paw["left"] is None and nose["left"] < 0.4,  # its left side: paw hidden, face to our left
    ]


def test_marker_sides_for_every_front():
    return all(all(check_marker_sides(front)) for front in view_sides.FRONT_VECTORS)


def test_default_front_is_minus_y():
    cube = marker_figure("-y")
    return (
        np.array(elevation.render_view(cube, "left")) == np.array(elevation.render_view(cube, "left", front="-y"))
    ).all()


def test_axis_labels_are_grid_coordinates_in_picture_order():
    cube = marker_figure("-y")
    front_view = elevation.project_nearest(cube, view_sides.side_vector("front", "-y"))
    back_view = elevation.project_nearest(cube, view_sides.side_vector("back", "-y"))
    figure_left = elevation.project_nearest(cube, view_sides.side_vector("left", "-y"))
    return (
        front_view.horizontal_axis == "x"
        and list(front_view.column_coordinates) == list(range(GRID_SIZE))
        and list(back_view.column_coordinates) == list(range(GRID_SIZE))[::-1]
        and figure_left.horizontal_axis == "y"
        and list(figure_left.column_coordinates) == list(range(GRID_SIZE))
    )


def test_nearest_cell_wins():
    cube = np.zeros((1, 3, 1), np.int32)
    cube[0, 0, 0], cube[0, 2, 0] = BLACK, ORANGE
    front = elevation.project_nearest(cube, (0, -1)).face_ids[0, 0]
    back = elevation.project_nearest(cube, (0, 1)).face_ids[0, 0]
    return front == BLACK and back == ORANGE


def test_title_names_the_figure_side():
    title = elevation.describe_view("left", (1, 0), elevation.project_nearest(marker_figure("-y"), (1, 0)))
    return title.startswith("left view = the figure's left side, camera at high x")


def brick(x, y, z, w, d, h, c):
    return NS(x=x, y=y, z=z, w=w, d=d, h_plates=h, color=c, notch=-1)


def test_save_elevations_removes_base_and_margin_and_uses_model_front():
    margin, base_layers = 3, 2
    bricks = [
        brick(0, 0, 0, 20, 20, 2, BLACK),  # base (base_layers=2)
        brick(margin, margin, base_layers, 2, 1, 1, BLACK),
        brick(margin, margin, base_layers + 1, 1, 1, 1, RED),
    ]  # red at figure x 0, low x
    folder = tempfile.mkdtemp()
    paths = elevation.save_elevations(NS(bricks=bricks, base_layers=base_layers, front="+x"), folder, "t", margin)
    written = len(paths) == 4 and all(os.path.getsize(p) > 100 for p in paths)
    cube = elevation.colour_cube(bricks)
    back_with_face_to_plus_x = elevation.render_view(cube, "back", base_layers=base_layers, margin=margin, front="+x")
    return written and horizontal_position(back_with_face_to_plus_x, RED) is not None


def side_coloured_block() -> tuple:
    """6x6x6 block [z, y, x] whose four sides have one colour each (top layer grey) and the colour per side."""
    from image_to_bricks.instructions import fast_render

    grid = np.ones((6, 6, 6), bool)
    rgb = np.full((6, 6, 6, 3), 128, np.uint8)
    colour_by_side = {(0, -1): (255, 0, 0), (0, 1): (0, 255, 0), (-1, 0): (0, 0, 255), (1, 0): (255, 255, 0)}
    rgb[:, 0, :], rgb[:, -1, :] = colour_by_side[(0, -1)], colour_by_side[(0, 1)]
    rgb[:, :, 0], rgb[:, :, -1] = colour_by_side[(-1, 0)], colour_by_side[(1, 0)]
    rgb[-1] = 128
    return fast_render, grid, rgb, colour_by_side


def test_iso_camera_sides_match_the_rendered_faces():
    """The two block sides that fill most of each isometric picture face the camera position ISO_CAMERA_SIDES
    gives (shading darkens faces, so count pixels by their dominant channels)."""
    fast_render, grid, rgb, colour_by_side = side_coloured_block()
    for view, (camera_x, camera_y) in fast_render.ISO_CAMERA_SIDES.items():
        pixels = np.array(
            fast_render.iso_render(grid, rgb, fast_render.IsoView(view=view, elev=20, outline=None))
        ).astype(int)
        red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        counts = {
            (0, -1): ((red > 150) & (green < 60) & (blue < 60)).sum(),
            (0, 1): ((green > 150) & (red < 60)).sum(),
            (-1, 0): ((blue > 150) & (red < 60)).sum(),
            (1, 0): ((red > 150) & (green > 150) & (blue < 60)).sum(),
        }
        biggest_two = set(sorted(counts, key=counts.get)[-2:])
        if biggest_two != {(camera_x, 0), (0, camera_y)}:
            return False
    return True


def test_iso_names_follow_front():
    from image_to_bricks.instructions import outputs_handler

    facing_minus_y = [title for _, title, _ in outputs_handler.isometric_view_names("-y")]
    default = [title for _, title, _ in outputs_handler.isometric_view_names(None)]
    facing_plus_x = [title for _, title, _ in outputs_handler.isometric_view_names("+x")]
    return (
        facing_minus_y == ["Front-left", "Rear-left", "Rear-right", "Front-right"]
        and default == facing_minus_y
        and facing_plus_x == ["Front-right", "Front-left", "Rear-left", "Rear-right"]
    )


def test_cli_writes_views():
    source = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image_to_bricks", "cli.py"), encoding="utf-8"
    ).read()
    return '"--no-views"' in source and "elevation.save_elevations" in source


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    results = [(test.__name__, bool(test())) for test in tests]
    for test_name, passed in results:
        print(("ok    " if passed else "FAIL  ") + test_name)
    print("all passed" if all(p for _, p in results) else "FAILED")
    sys.exit(0 if all(p for _, p in results) else 1)
