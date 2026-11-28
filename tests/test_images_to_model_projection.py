"""images_to_model step 4: colours projected from the pictures onto the mesh (front found by outline, depth test,
unseen vertices filled from their neighbours)."""

import numpy as np
import support
from images_to_model_helpers import build_asymmetric_figure, build_icosphere, render_four_views

from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.projection import (
    ColouredMesh,
    ProjectionPainter,
    choose_front_direction,
    fill_unseen,
    project_colours,
)
from image_to_bricks.images_to_model.type_aliases import FloatArray
from image_to_bricks.images_to_model.view_geometry import split_rgba


def build_front_back_figure() -> tuple[Mesh, FloatArray]:
    """Asymmetric figure: red nose on the front (+z), the back half of the body blue."""
    mesh, colours = build_asymmetric_figure()
    colours[:8][mesh.vertices[:8, 2] < 0] = [0.0, 0.0, 1.0]
    return mesh, colours


def build_coloured_ellipsoid() -> tuple[Mesh, FloatArray]:
    """Dense ellipsoid: front (z > 0.1) red, back (z < -0.1) blue, character's left (x > 0.25) green, rest grey."""
    sphere = build_icosphere(4, 1.0)
    mesh = Mesh(sphere.vertices * np.array([0.4, 1.0, 0.3]) + np.array([0, 1.0, 0]), sphere.faces)
    x, z = mesh.vertices[:, 0], mesh.vertices[:, 2]
    colours = np.full((len(x), 3), 0.5)
    colours[z > 0.1] = [1, 0, 0]
    colours[z < -0.1] = [0, 0, 1]
    colours[x > 0.25] = [0, 1, 0]
    return mesh, colours


def test_projection_recovers_front_back_and_side_colours() -> None:
    mesh, colours = build_coloured_ellipsoid()
    projected, info = project_colours(mesh, render_four_views(mesh, colours))
    assert info["front_direction"] == "+z" and info["vertices_seen"] > 0.95
    error = np.abs(projected - colours).max(axis=1)
    assert (error < 0.25).mean() > 0.9  # region borders blur a little
    for region in (colours[:, 0] == 1, colours[:, 2] == 1, colours[:, 1] == 1):
        assert np.abs(projected[region].mean(axis=0) - colours[region][0]).max() < 0.2


def test_projection_finds_a_turned_mesh() -> None:
    mesh, colours = build_front_back_figure()
    turned = Mesh(mesh.vertices[:, [2, 1, 0]] * np.array([1, 1, -1]), mesh.faces)  # nose now towards +x
    front_mask = split_rgba(render_four_views(mesh, colours)["front"])[1]
    assert choose_front_direction(turned, front_mask) == "+x"


def test_unseen_vertices_take_their_neighbours_colour() -> None:
    sphere = build_icosphere(2, 1.0)
    known = sphere.vertices[:, 1] > 0
    colours = np.where(known[:, None], [0.2, 0.4, 0.6], 0.0)
    assert np.allclose(fill_unseen(sphere, colours, known), [0.2, 0.4, 0.6])


def test_painter_returns_a_coloured_mesh_and_its_info() -> None:
    mesh, colours = build_front_back_figure()
    painter = ProjectionPainter()
    result = painter.paint(mesh, render_four_views(mesh, colours))
    assert isinstance(result, ColouredMesh) and result.colours.shape == (len(mesh.vertices), 3)
    assert painter.last_info is not None and painter.last_info["views"] == ["back", "front", "left", "right"]


def test_coloured_mesh_exports_a_glb_with_vertex_colours() -> None:
    import trimesh  # noqa: PLC0415 (this test needs the real trimesh)

    mesh, colours = build_front_back_figure()
    path = f"{support.make_temp_dir('projection_export')}/figure.glb"
    ColouredMesh(mesh.vertices, mesh.faces, colours).export(path)
    loaded = trimesh.load(path, force="mesh", process=False)
    assert isinstance(loaded, trimesh.Trimesh) and isinstance(loaded.visual, trimesh.visual.ColorVisuals)
    assert len(loaded.faces) == len(mesh.faces)
    assert loaded.visual.vertex_colors[:, :3].max() == 255  # the red nose


if __name__ == "__main__":
    support.run_tests(globals())
