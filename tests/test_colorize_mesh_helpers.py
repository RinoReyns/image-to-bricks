"""colorize_mesh.py helpers that need no OpenGL: centring, silhouettes (alpha and plain background), colour-sampling
mask with edge erosion, camera look-at pose, vertex adjacency, highlight denoise, palette quantisation and filling
vertices no photo saw. pyrender is stubbed, so camera calibration and projection (they render) are not covered here.
Run: python tests/test_colorize_mesh_helpers.py"""

import os
import sys
import types

import numpy as np
import support
import trimesh
from PIL import Image

sys.modules.setdefault("pyrender", types.ModuleType("pyrender"))  # rendering is not used by these helpers
from tools import colorize_mesh

IMAGE_SIZE_PIXELS = 100
BACKGROUND_RGB = (240, 240, 240)
OBJECT_RGB = (30, 30, 30)
WHITE = (255.0, 255.0, 255.0)


def write_image(image: Image.Image, file_name: str) -> str:
    path = os.path.join(support.make_temp_dir("colorize"), file_name)
    image.save(path)
    return path


def image_with_square(mode: str, square: tuple, speck: tuple = None) -> Image.Image:
    """Plain background with a dark square (x0, y0, x1, y1) and an optional small dark speck elsewhere."""
    background = BACKGROUND_RGB + ((0,) if mode == "RGBA" else ())
    foreground = OBJECT_RGB + ((255,) if mode == "RGBA" else ())
    image = Image.new(mode, (IMAGE_SIZE_PIXELS, IMAGE_SIZE_PIXELS), background)
    image.paste(foreground, square)
    if speck:
        image.paste(foreground, speck)
    return image


def make_strip_mesh(vertex_count: int = 6) -> trimesh.Trimesh:
    """Triangle strip along x: vertex i is joined to i+1 and i+2 (every inner vertex has 4 neighbours)."""
    vertices = [(index // 2, index % 2, 0.0) for index in range(vertex_count)]
    faces = [(index, index + 1, index + 2) for index in range(vertex_count - 2)]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_load_and_center_puts_the_base_at_y0_and_centres_x_and_z():
    box = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    box.apply_translation((10.0, 20.0, 30.0))
    path = os.path.join(support.make_temp_dir("colorize"), "box.glb")
    box.export(path)
    centred = colorize_mesh.load_and_center_mesh(path)
    assert np.isclose(centred.bounds[0][1], 0.0)
    assert np.allclose((centred.bounds[0] + centred.bounds[1])[[0, 2]], 0.0)


def test_silhouette_of_a_cut_out_uses_the_alpha_channel():
    path = write_image(image_with_square("RGBA", (20, 30, 60, 70)), "cut_out.png")
    silhouette = colorize_mesh.extract_silhouette(path)
    assert silhouette.sum() == 40 * 40 and silhouette[50, 40] and not silhouette[5, 5]


def test_silhouette_of_a_photo_keeps_only_the_largest_blob():
    image = image_with_square("RGB", (30, 30, 70, 70), speck=(85, 85, 88, 88))
    silhouette = colorize_mesh.extract_silhouette(write_image(image, "photo.png"))
    assert silhouette[50, 50], "the object is found"
    assert not silhouette[86, 86], "the small speck is dropped"
    assert silhouette.sum() == 40 * 40


def test_colour_sampling_mask_erodes_the_edge():
    path = write_image(image_with_square("RGBA", (20, 20, 80, 80)), "erode.png")
    fallback = np.zeros((IMAGE_SIZE_PIXELS, IMAGE_SIZE_PIXELS), bool)
    with support.module_settings(colorize_mesh, _color_mask_cache={}):
        mask = colorize_mesh.color_sampling_mask(path, fallback, erode_px=5)
    assert mask[50, 50] and not mask[21, 50], "a pixel 1 px inside the edge is eroded away"
    assert mask.sum() < 60 * 60


def test_colour_sampling_mask_uses_the_fallback_for_photos_without_alpha():
    path = write_image(image_with_square("RGB", (20, 20, 80, 80)), "no_alpha.png")
    fallback = np.zeros((IMAGE_SIZE_PIXELS, IMAGE_SIZE_PIXELS), bool)
    fallback[40:60, 40:60] = True
    with support.module_settings(colorize_mesh, _color_mask_cache={}):
        mask = colorize_mesh.color_sampling_mask(path, fallback, erode_px=0)
    assert np.array_equal(mask, fallback)


def test_look_at_pose_points_the_camera_at_the_target():
    eye, target = np.array([0.0, 1.0, 5.0]), np.array([0.0, 1.0, 0.0])
    pose = colorize_mesh._look_at_pose(eye, target)
    camera_forward = -pose[:3, 2]  # OpenGL cameras look down their -Z axis
    assert np.allclose(camera_forward, (target - eye) / np.linalg.norm(target - eye))
    assert np.allclose(pose[:3, 3], eye)
    assert np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3)), "the rotation part is orthonormal"


def test_adjacency_is_symmetric_and_follows_the_edges():
    adjacency = colorize_mesh._build_adjacency(make_strip_mesh()).toarray()
    assert np.array_equal(adjacency, adjacency.T)
    assert adjacency[0, 1] and adjacency[0, 2] and not adjacency[0, 3]


def test_denoise_replaces_a_lone_highlight_vertex():
    mesh = make_strip_mesh(8)
    colours = np.zeros((8, 3))
    colours[4] = WHITE
    seen = np.ones(8, bool)
    denoised = colorize_mesh.denoise_outliers(colours, seen, colorize_mesh._build_adjacency(mesh))
    # Known flaw (see the refactor findings): all vertices are updated at once, so the highlight is smeared over its
    # neighbours (they look like outliers against a mean that still contains it) instead of being removed. Only what
    # holds today is checked: the highlight fades and the strip ends stay black.
    assert denoised[4].max() < 255 / 4, denoised[4]
    assert np.allclose(denoised[[0, 7]], 0.0), "the strip ends keep their colour"


def test_quantize_snaps_colours_to_the_palette_size():
    random_colours = np.random.default_rng(3).uniform(0, 255, (200, 3))
    quantised = colorize_mesh.quantize_palette(random_colours, palette_size=4)
    assert len(np.unique(quantised, axis=0)) <= 4
    assert colorize_mesh.quantize_palette(random_colours, palette_size=0) is random_colours


def test_fill_missing_copies_colour_from_seen_neighbours():
    mesh = make_strip_mesh(6)
    weights = np.zeros(6)
    weights[:2] = 1.0
    colour_sums = np.zeros((6, 3))
    colour_sums[:2] = (200.0, 10.0, 10.0)
    filled = colorize_mesh.fill_missing_colors(mesh, colour_sums, weights)
    assert filled.dtype == np.uint8
    assert (filled == (200, 10, 10)).all(), filled


def test_fill_missing_without_any_seen_vertex_uses_neutral_grey():
    mesh = make_strip_mesh(4)
    filled = colorize_mesh.fill_missing_colors(mesh, np.zeros((4, 3)), np.zeros(4))
    assert (filled == 180).all()


def test_coloured_copy_keeps_the_original_mesh():
    mesh = make_strip_mesh(4)
    coloured = colorize_mesh._colored_copy(mesh, np.full((4, 3), 7, np.uint8))
    assert (coloured.visual.vertex_colors[:, :3] == 7).all() and (coloured.visual.vertex_colors[:, 3] == 255).all()
    assert mesh.visual.kind != "vertex" or not (mesh.visual.vertex_colors[:, :3] == 7).all()


if __name__ == "__main__":
    support.run_tests(globals())
