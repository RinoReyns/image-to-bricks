"""assign_voxel_colors: surface cells take the colour of the nearest mesh point, from vertex colours or from a UV
texture (GLB round trip), interior cells copy the nearest surface colour, debug images, --color-alias, --paint.
Needs the real trimesh. Run: python tests/test_colour_sampling.py"""

import os
from dataclasses import replace

import numpy as np
import support
import trimesh
from PIL import Image

from image_to_bricks.colours import colour_sampling
from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.mesh import loading, voxelize
from image_to_bricks.options import BuildOptions, ColourOptions, Paint, VoxelOptions

COLOUR_ID_BY_NAME = {name: colour_id for colour_id, name in COLOR_NAMES.items()}
RED = COLOUR_ID_BY_NAME["Red"]
BLUE = COLOUR_ID_BY_NAME["Blue"]
YELLOW = COLOUR_ID_BY_NAME["Yellow"]
LIGHT_BLUISH_GRAY = COLOUR_ID_BY_NAME["Light Bluish Gray"]
RED_RGB = (200, 30, 30)
BLUE_RGB = (30, 60, 200)
CUBE_SIDE_MM = 32.0
TEXTURE_SIZE_PIXELS = 8
UV_MARGIN = 0.05
TARGET_STUDS = 4
ONLY_RED_AND_BLUE = ColourOptions(allowed_colour_ids=[RED, BLUE])
SUPERSAMPLED = VoxelOptions(supersample=3)


def export_and_load(mesh: trimesh.Trimesh, file_name: str) -> trimesh.Trimesh:
    """Write the mesh as GLB and read it back with the pipeline's loader (the real input path)."""
    glb_path = os.path.join(support.make_temp_dir("sampling"), file_name)
    mesh.export(glb_path)
    with support.quiet_stdout():
        return loading.load_mesh(glb_path)


def make_vertex_coloured_tower() -> trimesh.Trimesh:
    """Cube: lower half blue, upper half red (two boxes, each with its own vertex colours)."""
    lower = trimesh.creation.box(extents=(CUBE_SIDE_MM, CUBE_SIDE_MM, CUBE_SIDE_MM / 2))
    lower.apply_translation(-lower.bounds[0])
    lower.visual.vertex_colors = BLUE_RGB + (255,)
    upper = lower.copy()
    upper.apply_translation((0.0, 0.0, CUBE_SIDE_MM / 2))
    upper.visual.vertex_colors = RED_RGB + (255,)
    return trimesh.util.concatenate([lower, upper])


def make_textured_cube() -> trimesh.Trimesh:
    """Cube whose UV v coordinate follows the height; the texture is red in its upper half, blue below. UVs stay
    inside (UV_MARGIN, 1 - UV_MARGIN): exactly 1.0 wraps to 0 like a repeating GPU texture."""
    cube = trimesh.creation.box(extents=(CUBE_SIDE_MM,) * 3)
    cube.apply_translation(-cube.bounds[0])
    relative_position = cube.vertices / CUBE_SIDE_MM
    uv = UV_MARGIN + (1 - 2 * UV_MARGIN) * relative_position[:, [0, 2]]
    texture = Image.new("RGB", (TEXTURE_SIZE_PIXELS, TEXTURE_SIZE_PIXELS), BLUE_RGB)
    for row in range(TEXTURE_SIZE_PIXELS // 2):  # image row 0 is the top = v 1
        for column in range(TEXTURE_SIZE_PIXELS):
            texture.putpixel((column, row), RED_RGB)
    cube.visual = trimesh.visual.TextureVisuals(uv=uv, image=texture)
    return cube


def colour_mesh(mesh: trimesh.Trimesh, debug_dir: str = None, colour: ColourOptions = ONLY_RED_AND_BLUE):
    """Scale, voxelise and colour the mesh like build_model does. Returns (grid [z, y, x], lego ids)."""
    with support.quiet_stdout():
        scaled = loading.scale_mesh(mesh, TARGET_STUDS)
        grid, origin, pitch = voxelize.voxelize_mesh(scaled, voxel=SUPERSAMPLED)
        lego = colour_sampling.assign_voxel_colors(
            grid,
            scaled,
            colour_sampling.VoxelFrame(origin, pitch),
            colour_sampling.ColourDebug(debug_dir, "cube"),
            BuildOptions(colour=colour),
        )
    return grid, lego


def layer_colours(grid: np.ndarray, lego: np.ndarray, layer_index: int) -> set:
    return {COLOR_NAMES[int(colour_id)] for colour_id in np.unique(lego[layer_index][grid[layer_index]])}


def assert_blue_bottom_and_red_top(grid: np.ndarray, lego: np.ndarray) -> None:
    filled_layers = np.nonzero(grid.any(axis=(1, 2)))[0]
    bottom_layer, top_layer = int(filled_layers[0]), int(filled_layers[-1])
    assert layer_colours(grid, lego, bottom_layer) == {"Blue"}, layer_colours(grid, lego, bottom_layer)
    assert layer_colours(grid, lego, top_layer) == {"Red"}, layer_colours(grid, lego, top_layer)


def test_vertex_colours_are_sampled_onto_the_surface():
    grid, lego = colour_mesh(export_and_load(make_vertex_coloured_tower(), "tower.glb"))
    assert_blue_bottom_and_red_top(grid, lego)


def test_texture_colours_are_sampled_onto_the_surface():
    grid, lego = colour_mesh(export_and_load(make_textured_cube(), "textured.glb"))
    assert_blue_bottom_and_red_top(grid, lego)


def test_mesh_with_fewer_triangles_than_search_candidates_is_coloured():
    """Regression: a 12-triangle cube crashed with IndexError (16 nearest triangles were always requested)."""
    cube = trimesh.creation.box(extents=(CUBE_SIDE_MM,) * 3)
    assert len(cube.faces) < colour_sampling.NEAREST_TRIANGLE_CANDIDATES
    cube.visual.vertex_colors = RED_RGB + (255,)
    grid, lego = colour_mesh(export_and_load(cube, "low_poly.glb"))
    assert (lego[grid] == RED).all()


def test_every_filled_cell_gets_a_colour_including_the_interior():
    grid, lego = colour_mesh(export_and_load(make_vertex_coloured_tower(), "tower.glb"))
    assert set(np.unique(lego[grid]).tolist()) <= {RED, BLUE}
    assert (lego[~grid] == 0).all(), "empty cells have no colour"


def test_missing_data_grey_takes_the_nearest_real_colour():
    no_data_grey = (192, 196, 205)
    tower = make_vertex_coloured_tower()
    upper_box_vertices = slice(len(tower.vertices) // 2, None)  # concatenate keeps the lower box's vertices first
    tower.visual.vertex_colors[upper_box_vertices] = no_data_grey + (255,)
    grey_allowed = ColourOptions(allowed_colour_ids=[RED, BLUE, LIGHT_BLUISH_GRAY])
    grid, lego_without_fill = colour_mesh(export_and_load(tower, "tower_no_data.glb"), colour=grey_allowed)
    assert LIGHT_BLUISH_GRAY in lego_without_fill[grid], "without --missing-rgb the grey stays grey"
    with support.module_settings(colour_sampling, MISSING_TOL=10):
        grid, lego = colour_mesh(
            export_and_load(tower, "tower_no_data.glb"), colour=replace(grey_allowed, missing_rgb=no_data_grey)
        )
    assert LIGHT_BLUISH_GRAY not in lego[grid], "the grey 'no data' part copies the real colour next to it"


def test_colour_alias_builds_with_the_second_colour():
    aliased = replace(ONLY_RED_AND_BLUE, colour_alias={RED: YELLOW})
    grid, lego = colour_mesh(export_and_load(make_vertex_coloured_tower(), "tower.glb"), colour=aliased)
    assert RED not in lego[grid] and YELLOW in lego[grid]


def test_paint_box_recolours_surface_cells_inside_it():
    whole_top_layer = (slice(-1, None), slice(None), slice(None))
    painted = replace(ONLY_RED_AND_BLUE, paints=[Paint(whole_top_layer, YELLOW, False)])
    grid, lego = colour_mesh(export_and_load(make_vertex_coloured_tower(), "tower.glb"), colour=painted)
    top_layer = int(np.nonzero(grid.any(axis=(1, 2)))[0][-1])
    assert layer_colours(grid, lego, top_layer) == {"Yellow"}


def test_colour_debug_images_are_written():
    debug_dir = support.make_temp_dir("debug")
    colour_mesh(export_and_load(make_vertex_coloured_tower(), "tower.glb"), debug_dir=debug_dir)
    for stage in ("dbg1_raw_rgb", "dbg2_enhanced_rgb", "dbg3_mapped_ids", "dbg4_final_ids"):
        image_path = os.path.join(debug_dir, f"cube_{stage}.png")
        assert os.path.getsize(image_path) > 0, image_path


def test_unsupported_visual_kind_is_rejected():
    plain_box = trimesh.creation.box(extents=(8.0, 8.0, 8.0))
    plain_box.visual = trimesh.visual.ColorVisuals()  # no colours at all -> kind None
    try:
        colour_sampling._prepare_visual_data(plain_box)
    except ValueError as error:
        assert "Unsupported visual type" in str(error)
        return
    raise AssertionError("a mesh without colours must be rejected with a clear message")


if __name__ == "__main__":
    support.run_tests(globals())
