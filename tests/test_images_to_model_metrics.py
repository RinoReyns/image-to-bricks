"""Verification metrics of the model step: surface distances, F-scores, volume IoU, outlines, renders, PSNR / SSIM
and matching a turned model to the input pictures."""

import numpy as np
import support
from images_to_model_helpers import (
    build_asymmetric_figure,
    build_box,
    build_icosphere,
    colour_by_height,
    is_close,
    render_input_view,
)

from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.verification.metrics import (
    align_to_reference,
    build_axis_rotations,
    compare_surfaces,
    compare_view,
    compare_with_views,
    compute_lego_volume_iou,
    compute_masked_psnr,
    compute_ssim,
    compute_volume_iou,
    normalise_view,
    rasterize_silhouette,
    render_vertex_colours,
    summarise_views,
    voxelize_solid,
)
from image_to_bricks.images_to_model.view_geometry import assign_view_directions, compute_mask_iou


def test_identical_surfaces_score_perfect() -> None:
    sphere = build_icosphere(3, 0.5)
    result = compare_surfaces(sphere, sphere, count=20000)
    assert result["chamfer_l1"] < 0.01
    assert result["fscore_0.02"] > 0.99 and result["normal_consistency"] > 0.97


def test_a_shifted_surface_scores_worse() -> None:
    sphere = build_icosphere(3, 0.5)
    shifted = Mesh(sphere.vertices + np.array([0.1, 0, 0]), sphere.faces)
    near = compare_surfaces(sphere, sphere, count=20000)
    far = compare_surfaces(shifted, sphere, count=20000)
    assert far["chamfer_l1"] > near["chamfer_l1"] + 0.03 and far["fscore_0.01"] < near["fscore_0.01"]


def test_voxelized_box_has_its_volume_and_iou() -> None:
    box = build_box(1.0, 2.0, 0.5)
    grid = voxelize_solid(box, np.array([-1, -0.5, -1]), np.array([1, 2.5, 1]), 60)
    assert is_close(grid.sum() * (2 / 60) * (3 / 60) * (2 / 60), 1.0, rel=0.06)
    assert compute_volume_iou(box, box, 48) == 1.0
    assert is_close(compute_volume_iou(build_box(1.0, 1.0, 0.5), box, 48), 0.5, abs_tol=0.03)


def test_voxelized_sphere_has_its_volume() -> None:
    grid = voxelize_solid(build_icosphere(3, 1.0), np.array([-1.1] * 3), np.array([1.1] * 3), 64)
    assert is_close(grid.sum() * (2.2 / 64) ** 3, 4.15, rel=0.03)


def test_rotations_and_alignment_recover_the_orientation() -> None:
    assert len(build_axis_rotations()) == 24
    reference = build_box(0.6, 1.0, 0.3)
    lying = Mesh(reference.vertices[:, [1, 0, 2]] * np.array([1, -1, 1]) * 2.5 + 7, reference.faces)
    aligned, _, chamfer = align_to_reference(lying, reference)
    assert chamfer < 0.02  # sampling noise of 4000 random points
    assert compute_volume_iou(aligned, reference, 40) > 0.9


def test_silhouette_and_render_of_a_box() -> None:
    box = build_box(0.5, 1.0, 0.25)
    front, side = rasterize_silhouette(box, "+z", 128), rasterize_silhouette(box, "+x", 128)
    assert front.sum() > side.sum() * 1.7
    rgb, mask = render_vertex_colours(box, colour_by_height(box), "+z", 128)
    assert compute_mask_iou(mask, front) > 0.95
    rows = np.nonzero(mask.any(axis=1))[0]
    assert rgb[rows[0] + 2, 64, 0] > rgb[rows[-1] - 2, 64, 0]  # top is red, bottom green


def test_render_hides_back_faces_with_the_zbuffer() -> None:
    near = build_box(1.0, 1.0, 0.1)
    far = Mesh(near.vertices - np.array([0, 0, 1.0]), near.faces)
    mesh = Mesh(np.vstack([near.vertices, far.vertices]), np.vstack([near.faces, far.faces + len(near.vertices)]))
    colours = np.vstack([np.tile([1.0, 0, 0], (8, 1)), np.tile([0, 0, 1.0], (8, 1))])
    assert render_vertex_colours(mesh, colours, "+z", 64)[0][32, 32].tolist() == [1.0, 0.0, 0.0]
    assert render_vertex_colours(mesh, colours, "-z", 64)[0][32, 32].tolist() == [0.0, 0.0, 1.0]


def test_normalised_views_ignore_scale_and_offset() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 50:60] = True
    big = np.zeros((300, 300), dtype=bool)
    big[100:160, 20:50] = True
    _, first = normalise_view(np.zeros((100, 100, 3)), mask, 64)
    _, second = normalise_view(np.zeros((300, 300, 3)), big, 64)
    assert compute_mask_iou(first, second) > 0.95


def test_psnr_and_ssim_basics() -> None:
    image = np.random.default_rng(0).random((64, 64, 3))
    mask = np.ones((64, 64), dtype=bool)
    assert compute_masked_psnr(image, image, mask) == 99.0
    assert compute_masked_psnr(image, image * 0.5, mask) < 15
    assert is_close(compute_ssim(image, image), 1.0) and compute_ssim(image, np.zeros_like(image)) < 0.1


def test_matching_finds_the_direction_a_turned_model_faces() -> None:
    mesh, colours = build_asymmetric_figure()
    views = {
        "front": render_input_view(mesh, colours, "+z"),
        "left": render_input_view(mesh, colours, "+x"),
        "back": render_input_view(mesh, colours, "-z"),
    }
    turned = Mesh(mesh.vertices[:, [2, 1, 0]] * np.array([1, 1, -1]), mesh.faces)  # rotated 90 degrees about y
    result = compare_with_views(turned, colours, views, size=128)
    for metrics in result.views.values():
        assert metrics["silhouette_iou"] > 0.9 and metrics["psnr"] > 20
    assert result.assignment["front"] != "+z"
    summary = summarise_views(result.views, seen=("front",))
    assert summary["seen_silhouette_iou"] > 0.9 and summary["unseen_psnr"] > 20


def test_a_wrong_colour_is_penalised() -> None:
    mesh, colours = build_asymmetric_figure()
    image = render_input_view(mesh, colours, "+z")
    rgb, mask = render_vertex_colours(mesh, colours, "+z", 128)
    good, bad = compare_view(image, rgb, mask), compare_view(image, 1 - rgb, mask)
    assert good["psnr"] > bad["psnr"] + 5 and good["ssim"] > bad["ssim"]


def test_view_directions_are_consistent() -> None:
    assert assign_view_directions(0, 1) == {"front": "+z", "back": "-z", "left": "+x", "right": "-x"}
    assert assign_view_directions(1, -1)["right"] == "-z"


def test_voxelize_accepts_counts_per_axis() -> None:
    box = build_box(1.0, 2.0, 0.5)
    grid = voxelize_solid(box, np.array([-1, -0.5, -1]), np.array([1, 2.5, 1]), (20, 60, 40))
    assert grid.shape == (20, 60, 40)
    assert is_close(grid.sum() * (2 / 20) * (3 / 60) * (2 / 40), 1.0, rel=0.1)


def test_lego_grid_volume_iou() -> None:
    box = build_box(0.4, 1.0, 0.3)
    assert compute_lego_volume_iou(box, box) == 1.0
    assert is_close(compute_lego_volume_iou(build_box(0.4, 0.5, 0.3), box), 0.5, abs_tol=0.03)
    shifted = Mesh(box.vertices + np.array([0.005, 0, 0]), box.faces)  # 1.5 mm on a 30 cm build
    assert compute_lego_volume_iou(shifted, box) > 0.9


if __name__ == "__main__":
    support.run_tests(globals())
