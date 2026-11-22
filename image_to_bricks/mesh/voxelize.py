"""Mesh -> cell grid (bool [z, y, x], 1 cell = 1 stud x 1 plate), hollowing, visible cells."""

from typing import Any

import numpy as np
import trimesh
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import binary_dilation, binary_erosion, label

from image_to_bricks.consts import STUD_MM
from image_to_bricks.options import (
    VoxelOptions,
)

MAX_RECONNECT_ROUNDS = 100000  # safety stop of the reconnecting loop (each round joins two pieces)
MIN_PIECES_TO_JOIN = 2
FACE_STEPS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))  # to the 6 face neighbours

type Cell3 = tuple[int, int, int]


def _face_structure() -> NDArray[np.bool_]:
    """3x3x3 structure: a cell and its 6 face neighbours (face-connected pieces)."""
    face = np.zeros((3, 3, 3), bool)
    face[1, 1, :] = face[1, :, 1] = face[:, 1, 1] = True
    return face


def _path_to_target(
    region: NDArray[np.bool_], source: NDArray[np.bool_], target: NDArray[np.bool_]
) -> list[Cell3] | None:
    """Breadth-first growth from the `source` cells through `region` until a `target` cell is touched; the cells
    of that path from the target back to (not including) the source, or None if no target is reachable."""
    seen = source.copy()
    parent: dict[Cell3, Cell3] = {}
    current: list[Cell3] = [(int(c[0]), int(c[1]), int(c[2])) for c in np.argwhere(source)]
    hit: Cell3 | None = None
    while current and hit is None:
        next_cells: list[Cell3] = []
        for cell in current:
            for step in FACE_STEPS:
                neighbour = (cell[0] + step[0], cell[1] + step[1], cell[2] + step[2])
                if min(neighbour) < 0 or any(n >= size for n, size in zip(neighbour, region.shape, strict=True)):
                    continue
                if seen[neighbour] or not region[neighbour]:
                    continue
                seen[neighbour] = True
                parent[neighbour] = cell
                if target[neighbour]:
                    hit = neighbour
                    break
                next_cells.append(neighbour)
            if hit is not None:
                break
        current = next_cells
    if hit is None:
        return None
    path = []
    while hit in parent:
        path.append(hit)
        hit = parent[hit]
    return path


def _join_two_pieces(
    final: NDArray[np.bool_], final_labels: NDArray[Any], allowed_labels: NDArray[Any], allowed_count: int
) -> bool:
    """In the first piece of `allowed` (labels 1..allowed_count) that holds several pieces of `final` (labels
    final_labels), join the biggest of them to the nearest other one through allowed cells (in place). True if a
    join was made."""
    for allowed_label in range(1, allowed_count + 1):
        region = allowed_labels == allowed_label
        piece_ids = np.unique(final_labels[region & final])
        if len(piece_ids) < MIN_PIECES_TO_JOIN:
            continue
        main = max((int((final_labels == i).sum()), i) for i in piece_ids)[1]
        others = np.isin(final_labels, [i for i in piece_ids if i != main])
        path = _path_to_target(region, final_labels == main, others)
        if path is None:
            continue
        for cell in path:
            final[cell] = True
        return True
    return False


def _reconnect(final: NDArray[np.bool_], allowed: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Add cells of `allowed` (bool [x, y, z], face-connected paths) so that every connected piece of `allowed`
    that still has at least one cell in `final` is one connected piece again (thin parts must not fall apart)."""
    final = final.copy()
    face = _face_structure()
    allowed_labels, allowed_count = label(allowed, structure=face)
    for _ in range(MAX_RECONNECT_ROUNDS):
        final_labels, piece_count = label(final, structure=face)
        if piece_count <= 1 or not _join_two_pieces(final, final_labels, allowed_labels, allowed_count):
            break
    return final


def downsample_voxels(
    fine: NDArray[np.bool_],
    fine_origin: ArrayLike,
    fine_pitch: float,
    pitch: float,
    voxel: VoxelOptions,
) -> tuple[NDArray[np.bool_], NDArray[Any]]:
    """Fine voxel grid -> grid of cell size `pitch`. The surface voxeliser marks every cell the surface
    touches, which fattens every part by about half a cell per side (gaps close, thin parts inflate); a
    coarse cell is kept only if at least voxel.threshold of its sub-cells are filled.
    Thin parts (tail, blade, brim edge) would fall apart under that rule, so cells that have no kept cell
    next to them use the gentler voxel.thin_threshold instead (None = same as threshold), and the result is
    reconnected through cells of that lower threshold, so nothing that was one piece falls apart.
    voxel.box = (slice z, slice y, slice x) in the coarse grid [z, y, x] with voxel.box_threshold: inside it the
    thin threshold is lowered further (a rapier blade is ~0.04 of a cell: lost at 0.05, kept at 0.02) without
    adding speckles (whiskers, hair) elsewhere. voxel.supersample is not used here.
    fine: bool [x, y, z]; fine_origin: world position (xyz) of the centre of fine[0, 0, 0].
    Returns (bool [x, y, z], world position of the centre of its [0, 0, 0] cell). Coarse cell centres
    sit on multiples of `pitch`, like the fine ones on multiples of `fine_pitch` (odd N: aligned)."""
    threshold, thin_threshold, box = voxel.threshold, voxel.thin_threshold, voxel.box
    sub_cells = round(pitch / fine_pitch) ** 3
    fine_cells = np.argwhere(fine)
    if len(fine_cells) == 0:
        return np.zeros((1, 1, 1), bool), np.zeros(3)
    world = np.asarray(fine_origin, float) + fine_cells * fine_pitch
    coarse_index = np.rint(world / pitch).astype(np.int64)
    lowest_index = coarse_index.min(axis=0)
    shape = tuple((coarse_index.max(axis=0) - lowest_index + 1).tolist())
    counts = np.zeros(shape, np.int32)
    np.add.at(counts, tuple((coarse_index - lowest_index).T), 1)
    thick = counts >= threshold * sub_cells
    if thin_threshold is None or (thin_threshold >= threshold and box is None):
        return thick, lowest_index * pitch
    needed_counts = np.full(shape, thin_threshold * sub_cells)
    if box is not None and voxel.box_threshold is not None:
        inside_box = np.zeros(shape[::-1], bool)  # [z, y, x]
        inside_box[box] = True
        box_counts = min(voxel.box_threshold, thin_threshold) * sub_cells
        needed_counts = np.where(inside_box.transpose(2, 1, 0), box_counts, needed_counts)
    thin_kept = counts >= needed_counts
    near_thick = binary_dilation(thick, structure=np.ones((3, 3, 3), bool))
    final = thick | (thin_kept & ~near_thick)
    return _reconnect(final, thin_kept), lowest_index * pitch


def voxelize_mesh(
    mesh: trimesh.Trimesh, pitch: float = STUD_MM, voxel: VoxelOptions | None = None
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Return (grid, origin, pitch).

    grid   : bool array [z, y, x], index 0 at the physical bottom.
    origin : (3,) xyz position of the CENTRE of voxel [0,0,0] in the
             (Z-stretched) mesh space.  trimesh places voxel centres on
             multiples of `pitch`, so voxel i spans [c - pitch/2, c + pitch/2],
             NOT [i*pitch, (i+1)*pitch].  Colour sampling needs this.
    pitch  : (3,) cell size along x, y, z.
    """
    print("Voxelizing...")
    voxel = voxel if voxel is not None else VoxelOptions()
    if voxel.supersample > 1:
        factor = voxel.supersample
        fine_voxels = mesh.voxelized(pitch=pitch / factor).fill()
        fine_transform = np.asarray(fine_voxels.transform, dtype=np.float64)
        matrix, origin = downsample_voxels(fine_voxels.matrix, fine_transform[:3, 3], pitch / factor, pitch, voxel)
        real_pitch: NDArray[np.float64] = np.full(3, pitch, dtype=np.float64)
        print(
            f"  supersampled x{factor}, keep cells at least {voxel.threshold:.2f} inside "
            f"({int(fine_voxels.matrix.sum())} fine cells -> {int(matrix.sum())})"
        )
    else:
        voxels = mesh.voxelized(pitch=pitch).fill()
        matrix = voxels.matrix  # [x, y, z] bool
        transform = np.asarray(voxels.transform, dtype=np.float64)
        origin = np.asarray(transform[:3, 3].copy(), dtype=np.float64)
        real_pitch = np.diag(transform)[:3].copy()
    grid = np.transpose(matrix, (2, 1, 0))  # -> [z, y, x]
    print(f"Voxel grid shape (z, y, x): {grid.shape}, filled cells: {grid.sum()}")
    print(f"Voxel origin (centre of [0,0,0]): {origin}, pitch: {real_pitch}")
    return grid, origin, real_pitch


def voxels_to_lego_units(
    grid: NDArray[np.bool_], *, hollow: bool = False, shell_thickness_studs: int = 1
) -> NDArray[np.bool_]:
    """Grid is already 1 voxel == 1 stud (x,y) x 1 plate (z) thanks to the
    anisotropic pre-scale in scale_mesh(). With hollow=True, keep only a wall
    shell_thickness_studs thick (measured sideways, within each layer) plus a
    1-plate skin on top/bottom faces; the rest of the inside is removed."""
    print("Converting voxels to LEGO units...")
    if not hollow:
        return grid
    solid = grid.astype(bool)
    # sideways erosion, layer by layer (4-neighbour cross in x/y)
    cross = np.zeros((1, 3, 3), bool)
    cross[0, 1, :] = cross[0, :, 1] = True
    inner = binary_erosion(solid, structure=cross, iterations=max(1, shell_thickness_studs), border_value=0)
    # 1-plate skin where the layer above/below is empty
    padded = np.pad(solid, ((1, 1), (0, 0), (0, 0)), constant_values=False)
    inner &= padded[:-2] & padded[2:]
    shell: NDArray[np.bool_] = solid & ~inner
    print(f"  shell {shell_thickness_studs} stud(s): kept {int(shell.sum())} of {int(solid.sum())} voxels")
    return shell


def surface_mask(grid: NDArray[Any]) -> NDArray[np.bool_]:
    """True for filled voxels that have at least one empty 6-neighbour
    (or touch the grid boundary).  Only these are visible on the model."""
    valid = grid.astype(bool)
    padded = np.pad(valid, 1, constant_values=False)
    interior = np.ones_like(valid)
    depth_z, depth_y, depth_x = valid.shape
    for step_z, step_y, step_x in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        interior &= padded[
            1 + step_z : 1 + step_z + depth_z, 1 + step_y : 1 + step_y + depth_y, 1 + step_x : 1 + step_x + depth_x
        ]
    is_surface: NDArray[np.bool_] = valid & ~interior
    return is_surface
