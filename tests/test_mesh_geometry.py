"""Mesh front end: load_mesh (scene node transform), rotate_mesh, scale_mesh, voxelize_mesh.
Needs the real trimesh. Run: python tests/test_mesh_geometry.py"""

import os

import numpy as np
import support
import trimesh

from image_to_bricks.consts import PLATE_MM, STUD_MM
from image_to_bricks.mesh import loading, voxelize
from image_to_bricks.options import VoxelOptions

BOX_EXTENTS_MM = (32.0, 24.0, 24.0)  # 4 x 3 studs, 3 cells high in the Z-stretched space
HALF_CELL_OFFSET_MM = STUD_MM / 2  # puts the box faces on cell borders (cell centres sit on multiples of a stud)


def make_box_on_cell_borders() -> trimesh.Trimesh:
    """Box of BOX_EXTENTS_MM whose faces lie exactly on voxel cell borders, so its true volume is 4 x 3 x 3 cells."""
    box = trimesh.creation.box(extents=BOX_EXTENTS_MM)
    box.apply_translation(-box.bounds[0] + HALF_CELL_OFFSET_MM)
    return box


def filled_extent_cells(grid: np.ndarray) -> tuple:
    """(z, y, x) size of the bounding box of the filled cells."""
    filled = np.argwhere(grid)
    return tuple((filled.max(axis=0) - filled.min(axis=0) + 1).tolist())


def test_load_mesh_applies_the_scene_node_transform():
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    scene = trimesh.Scene()
    node_translation = trimesh.transformations.translation_matrix((100.0, 0.0, 0.0))
    scene.add_geometry(box, transform=node_translation)
    glb_path = os.path.join(support.make_temp_dir("load"), "moved_box.glb")
    scene.export(glb_path)
    with support.quiet_stdout():
        loaded = loading.load_mesh(glb_path)
    assert isinstance(loaded, trimesh.Trimesh)
    assert np.allclose(loaded.centroid, (100.0, 0.0, 0.0), atol=1e-4), loaded.centroid


def test_load_mesh_accepts_a_single_mesh_file():
    stl_path = os.path.join(support.make_temp_dir("load"), "box.stl")
    trimesh.creation.box(extents=(4.0, 5.0, 6.0)).export(stl_path)
    with support.quiet_stdout():
        loaded = loading.load_mesh(stl_path)
    assert np.allclose(sorted(loaded.extents), (4.0, 5.0, 6.0))


def test_load_mesh_rejects_a_scene_without_geometry():
    def load_empty_scene(*_arguments, **_keyword_arguments) -> trimesh.Scene:
        return trimesh.Scene()  # trimesh cannot export an empty GLB, so the loader is replaced instead

    with support.module_settings(trimesh, load=load_empty_scene), support.quiet_stdout():
        try:
            loading.load_mesh("empty.glb")
        except ValueError as error:
            assert "no mesh geometry" in str(error)
            return
    raise AssertionError("an empty scene must not load as a mesh")


def test_rotate_mesh_turns_y_up_into_z_up_around_the_centroid():
    tall_in_y = trimesh.creation.box(extents=(2.0, 10.0, 4.0))
    tall_in_y.apply_translation((5.0, 5.0, 5.0))
    with support.quiet_stdout():
        rotated = loading.rotate_mesh(tall_in_y, rotate_x=90)
    assert np.allclose(rotated.extents, (2.0, 4.0, 10.0)), rotated.extents
    assert np.allclose(rotated.centroid, tall_in_y.centroid)


def test_rotate_mesh_does_not_change_the_input_mesh():
    original = trimesh.creation.box(extents=(2.0, 10.0, 4.0))
    vertices_before = original.vertices.copy()
    with support.quiet_stdout():
        loading.rotate_mesh(original, rotate_x=90, rotate_y=45, rotate_z=30)
    assert np.array_equal(original.vertices, vertices_before)


def test_rotate_mesh_with_zero_angles_keeps_the_mesh():
    original = trimesh.creation.box(extents=(2.0, 10.0, 4.0))
    with support.quiet_stdout():
        rotated = loading.rotate_mesh(original)
    assert np.allclose(rotated.vertices, original.vertices)


def test_scale_mesh_fits_the_widest_horizontal_side_to_the_target_studs():
    target_studs = 12
    box = trimesh.creation.box(extents=(3.0, 6.0, 9.0))
    with support.quiet_stdout():
        scaled = loading.scale_mesh(box, target_studs)
    uniform_scale = target_studs * STUD_MM / 6.0
    expected_extents = (3.0 * uniform_scale, 6.0 * uniform_scale, 9.0 * uniform_scale * STUD_MM / PLATE_MM)
    assert np.allclose(scaled.extents, expected_extents), scaled.extents
    assert np.allclose(scaled.bounds[0], 0.0), "the scaled mesh must sit on the origin"


def test_scale_mesh_rejects_a_mesh_without_horizontal_extent():
    flat_vertical_line = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], faces=[[0, 1, 2]], process=False)
    with support.quiet_stdout():
        try:
            loading.scale_mesh(flat_vertical_line, 10)
        except ValueError as error:
            assert "zero horizontal extent" in str(error)
            return
    raise AssertionError("a degenerate mesh must raise ValueError")


def test_voxelize_without_supersampling_fattens_the_box_by_half_a_cell():
    with support.quiet_stdout():
        grid, _, pitch = voxelize.voxelize_mesh(make_box_on_cell_borders(), voxel=VoxelOptions(supersample=1))
    assert np.allclose(pitch, STUD_MM)
    assert int(grid.sum()) > 4 * 3 * 3, "the plain surface voxeliser marks every cell the surface touches"


def test_voxelize_with_supersampling_keeps_the_true_volume():
    for supersample in (3, 5):
        with support.quiet_stdout():
            grid, _, _ = voxelize.voxelize_mesh(
                make_box_on_cell_borders(), voxel=VoxelOptions(supersample=supersample, threshold=0.4)
            )
        assert int(grid.sum()) == 4 * 3 * 3, (supersample, int(grid.sum()))
        assert filled_extent_cells(grid) == (3, 3, 4), filled_extent_cells(grid)


def test_voxelize_returns_the_grid_as_z_y_x():
    tall_box = trimesh.creation.box(extents=(8.0, 16.0, 40.0))
    tall_box.apply_translation(-tall_box.bounds[0] + HALF_CELL_OFFSET_MM)
    with support.quiet_stdout():
        grid, _, _ = voxelize.voxelize_mesh(tall_box, voxel=VoxelOptions(supersample=3))
    assert filled_extent_cells(grid) == (5, 2, 1), filled_extent_cells(grid)


def test_downsample_of_an_empty_fine_grid_is_one_empty_cell():
    empty_fine_grid = np.zeros((6, 6, 6), bool)
    coarse, origin = voxelize.downsample_voxels(
        empty_fine_grid, (0, 0, 0), 1.0, 3.0, VoxelOptions(threshold=0.4, thin_threshold=None)
    )
    assert coarse.shape == (1, 1, 1) and not coarse.any()
    assert np.allclose(origin, 0.0)


if __name__ == "__main__":
    support.run_tests(globals())
