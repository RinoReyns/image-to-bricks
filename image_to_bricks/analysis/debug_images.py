"""--debug: images of the raw cell grid and a check that the placed pieces cover it exactly."""

import os

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from image_to_bricks.consts import PLATE_MM, STUD_MM
from image_to_bricks.instructions.outputs_handler import draw_voxel_scene
from image_to_bricks.model import LegoModel, grid_to_xyz_bool


def visualize_voxel_grid(grid: NDArray[np.bool_], outdir: str, model_name: str) -> None:
    """Save raw voxel-grid diagnostics before brick placement.

    The 3D image renders the grid using the pipeline convention that z=0 is
    the bottom layer.  The two slice images make that convention easy to
    verify without perspective or camera-angle ambiguity.
    """
    filled_xyz = grid_to_xyz_bool(grid)
    depth_z, _, _ = grid.shape

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(filled_xyz, facecolors="orange", edgecolor="black", linewidth=0.1)
    width_x, depth_y, _ = filled_xyz.shape
    ax.set_box_aspect((width_x, depth_y, max(1, depth_z) * (PLATE_MM / STUD_MM)))
    ax.view_init(elev=22, azim=-60)
    ax.set_title("Raw voxelized mesh (Z increases upward)")
    fig.savefig(os.path.join(outdir, f"{model_name}_voxelized.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_voxel_debug_scene(grid: NDArray[np.bool_], path: str, color: str, title: str) -> None:
    """Save a consistently oriented 3D voxel diagnostic image."""
    filled_xyz = grid_to_xyz_bool(grid)
    facecolors = np.full(filled_xyz.shape, color, dtype=object)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_voxel_scene(ax, filled_xyz, facecolors, grid.shape[0])
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def verify_brick_placement(model: LegoModel, outdir: str, model_name: str) -> bool:
    """Compare brick occupancy to the input voxel grid cell by cell.

    This detects holes, overhangs, and overlapping brick volumes even when
    the total number of occupied cells happens to be unchanged.
    """
    coverage = np.zeros(model.grid.shape, dtype=np.uint16)
    depth_z, height_y, width_x = model.grid.shape

    for brick in model.bricks:
        z_end = brick.z + brick.h_plates
        y_end = brick.y + brick.d
        x_end = brick.x + brick.w
        if not (
            0 <= brick.z < z_end <= depth_z and 0 <= brick.y < y_end <= height_y and 0 <= brick.x < x_end <= width_x
        ):
            raise ValueError(f"Brick placement is outside the voxel grid: {brick}")
        coverage[brick.z : z_end, brick.y : y_end, brick.x : x_end] += 1

    reconstructed = coverage > 0
    missing = model.grid & ~reconstructed
    extra = reconstructed & ~model.grid
    overlaps = coverage > 1

    print("Brick placement verification:")
    print(f"  expected occupied cells: {int(model.grid.sum())}")
    print(f"  reconstructed cells: {int(reconstructed.sum())}")
    print(f"  missing cells: {int(missing.sum())}")
    print(f"  extra cells: {int(extra.sum())}")
    print(f"  overlapping cells: {int(overlaps.sum())}")

    _save_voxel_debug_scene(
        reconstructed,
        os.path.join(outdir, f"{model_name}_placed_bricks.png"),
        "deepskyblue",
        "Occupancy reconstructed from placed bricks",
    )
    _save_voxel_debug_scene(
        missing,
        os.path.join(outdir, f"{model_name}_missing_from_bricks.png"),
        "crimson",
        "Cells missing from brick placement",
    )
    _save_voxel_debug_scene(
        extra,
        os.path.join(outdir, f"{model_name}_extra_from_bricks.png"),
        "royalblue",
        "Cells outside the input voxel grid",
    )

    return not (missing.any() or extra.any() or overlaps.any())
