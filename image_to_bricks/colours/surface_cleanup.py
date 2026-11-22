"""Clean-up of the palette colours on the surface: smoothing, small regions, trim rings, overrides and the
automatic rules.

Arrays are [z, y, x]; lego = palette colour id per cell (0 = none).
"""

from collections import Counter
from collections.abc import Collection, Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, label

from image_to_bricks.colours import color_cleanup, trim_snap
from image_to_bricks.colours.palette_matching import rgb_to_lab
from image_to_bricks.consts import LEGO_RGB
from image_to_bricks.options import Box, ColourOptions

SMOOTH_MAJORITY = 4  # neighbours (of 6) that must agree to recolour a voxel

# Thin trim (hat band, boot cuffs): a layer whose outer ring is mostly this colour becomes a full ring.
TRIM_MIN_CELLS = 4  # ...and at least this many trim cells

FACE_STEPS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))  # to the 6 face neighbours
# the same steps in the order recolour_from_neighbours votes (decides ties between equally common colours)
VOTE_STEPS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
LAB_NEUTRAL = 128  # OpenCV 8-bit Lab: a and b of a grey

type TrimRings = dict[int, tuple[NDArray[np.bool_], NDArray[np.bool_]]]  # {colour id: (ring mask, stray mask)}
type RegionOverride = tuple[slice, slice, slice, int]  # (z slice, y slice, x slice, new colour id)


def _face_neighbours(
    cell: tuple[int, int, int], shape: tuple[int, ...], steps: Sequence[tuple[int, int, int]] = FACE_STEPS
) -> list[tuple[int, int, int]]:
    """The (z, y, x) face neighbours of a cell that lie inside the grid, in `steps` order."""
    z, y, x = cell
    return [
        (z + step_z, y + step_y, x + step_x)
        for step_z, step_y, step_x in steps
        if 0 <= z + step_z < shape[0] and 0 <= y + step_y < shape[1] and 0 <= x + step_x < shape[2]
    ]


def smooth_lego_colors(lego: NDArray[Any], grid: NDArray[np.bool_], iters: int = 2, majority: int = 5) -> NDArray[Any]:
    """Smooth palette colour ids by a majority vote of the 6-connected filled neighbours, `iters` times: a cell
    takes the most common neighbour colour if at least `majority` neighbours have it (4 = aggressive,
    5 = recommended, 6 = conservative). lego: uint8 ids [z, y, x]; grid: bool [z, y, x] cells taking part."""
    lego = lego.copy()
    for _ in range(iters):
        smoothed = lego.copy()
        for z, y, x in np.argwhere(grid):
            neighbour_colours = [
                lego[neighbour] for neighbour in _face_neighbours((z, y, x), grid.shape) if grid[neighbour]
            ]
            if not neighbour_colours:
                continue
            colour, count = Counter(neighbour_colours).most_common(1)[0]
            if colour != lego[z, y, x] and count >= majority:
                smoothed[z, y, x] = colour
        lego = smoothed
    return lego


def denoise_voxel_colors(
    colors: NDArray[Any], grid: NDArray[np.bool_], chroma_gap: float = 15.0, iters: int = 3
) -> NDArray[np.uint8]:
    """RGB [z, y, x, 3] with grey-ish outliers replaced: a cell whose most colourful filled neighbour is more than
    chroma_gap (Lab chroma) more colourful takes that neighbour's colour, `iters` times (bleeds saturated
    colours, so it is off by default)."""
    valid = grid.astype(bool)
    current = colors.astype(np.uint8).copy()
    padding = [(1, 1)] * 3
    for _ in range(iters):
        lab = cv2.cvtColor(current.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape((*grid.shape, 3)).astype(np.float32)
        chroma = np.hypot(lab[..., 1] - LAB_NEUTRAL, lab[..., 2] - LAB_NEUTRAL)
        best_neighbour_chroma = np.full(grid.shape, -1.0, dtype=np.float32)
        best_neighbour_colour = np.zeros_like(current)
        for step in FACE_STEPS:
            padded_valid = np.pad(valid, padding, constant_values=False)
            padded_chroma = np.pad(chroma, padding, constant_values=-1.0)
            padded_colour = np.pad(current, [*padding, (0, 0)], constant_values=0)
            shifted = tuple(slice(1 + offset, 1 + offset + size) for offset, size in zip(step, grid.shape, strict=True))
            neighbour_valid = padded_valid[shifted]
            neighbour_chroma = padded_chroma[shifted]
            is_better = neighbour_valid & (neighbour_chroma > best_neighbour_chroma)
            best_neighbour_chroma[is_better] = neighbour_chroma[is_better]
            best_neighbour_colour[is_better] = padded_colour[shifted][is_better]
        is_outlier = valid & (best_neighbour_chroma - chroma > chroma_gap)
        current[is_outlier] = best_neighbour_colour[is_outlier]
    return current


def unify_layer_regions(
    lego: NDArray[Any],
    grid: NDArray[np.bool_],
    min_region_size: int = 20,
    connectivity: int = 1,
) -> NDArray[Any]:
    """Makes large connected regions on each Z layer use a single LEGO colour.

    Regions are found in XY only, so disconnected objects (tail vs cape)
    remain independent. connectivity 1 = side neighbours, else diagonals too.
    """
    lego = lego.copy()
    structure = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        if connectivity == 1
        else np.ones((3, 3), dtype=np.uint8)
    )
    for z in range(grid.shape[0]):
        labels, region_count = label(grid[z], structure)
        for region in range(1, region_count + 1):
            region_mask = labels == region
            if region_mask.sum() < min_region_size:
                continue
            values, counts = np.unique(lego[z][region_mask], return_counts=True)
            lego[z][region_mask] = values[np.argmax(counts)]
    return lego


def _region_structure(connectivity: int) -> NDArray[np.uint8]:
    """3x3x3 structure: a cell and its face neighbours (connectivity 1) or all 26 neighbours."""
    if connectivity != 1:
        return np.ones((3, 3, 3), dtype=np.uint8)
    structure = np.zeros((3, 3, 3), dtype=np.uint8)
    structure[1, 1, 0] = structure[1, 1, 2] = 1
    structure[1, 0, 1] = structure[1, 2, 1] = 1
    structure[0, 1, 1] = structure[2, 1, 1] = 1
    structure[1, 1, 1] = 1
    return structure


def absorb_small_color_regions_3d(
    lego: NDArray[Any], grid: NDArray[np.bool_], min_region_size: int = 15, connectivity: int = 1
) -> NDArray[Any]:
    """Every connected region of one colour with fewer than min_region_size cells takes the most common colour
    around it (26 neighbours). All decisions read the colours before any change."""
    original = lego.copy()  # frozen snapshot -- all decisions read from this
    result = lego.copy()  # only this gets written to
    valid = grid.astype(bool)
    structure = _region_structure(connectivity)
    for colour_id in np.unique(original[valid]):
        colour_mask = valid & (original == colour_id)
        if not colour_mask.any():
            continue
        labelled, region_count = label(colour_mask, structure=structure)
        if region_count == 0:
            continue
        sizes = np.bincount(labelled.ravel())
        for region_id in range(1, region_count + 1):
            if sizes[region_id] >= min_region_size:
                continue
            region_mask = labelled == region_id
            dilated = binary_dilation(region_mask, structure=np.ones((3, 3, 3), dtype=bool))
            border_colours = original[dilated & valid & ~region_mask]
            if len(border_colours) == 0:
                continue
            values, counts = np.unique(border_colours, return_counts=True)
            result[region_mask] = values[np.argmax(counts)]
    return result


def recolour_from_neighbours(
    lego: NDArray[Any], valid: NDArray[np.bool_], mask: NDArray[np.bool_], avoid: Collection[int]
) -> NDArray[Any]:
    """Give each cell in `mask` the most common colour among its 6 neighbours (not in `avoid`); keep it if none."""
    recoloured = lego.copy()
    for z, y, x in np.argwhere(mask):
        votes: dict[int, int] = {}
        for neighbour in _face_neighbours((z, y, x), lego.shape, VOTE_STEPS):
            colour = int(lego[neighbour])
            if valid[neighbour] and not mask[neighbour] and colour not in avoid and colour != 0:
                votes[colour] = votes.get(colour, 0) + 1
        if votes:
            recoloured[z, y, x] = max(votes, key=lambda colour: votes[colour])
    return recoloured


def apply_color_overrides(
    lego: NDArray[Any],
    grid: NDArray[np.bool_],
    global_overrides: dict[int, int] | None = None,
    region_overrides: Sequence[RegionOverride] | None = None,
) -> NDArray[Any]:
    """global_overrides: {old_lego_id: new_lego_id} -- blanket recolor.
    region_overrides: list of (z_slice, y_slice, x_slice, new_lego_id) --
        force a specific voxel-space box to one colour. Applied last, so
        it always wins over every automated cleanup step above it.
    """
    lego = lego.copy()
    valid = grid.astype(bool)
    if global_overrides:
        for old_id, new_id in global_overrides.items():
            lego[valid & (lego == old_id)] = new_id
    if region_overrides:
        for z_slice, y_slice, x_slice, new_id in region_overrides:
            region = np.zeros_like(valid)
            region[z_slice, y_slice, x_slice] = True
            lego[valid & region] = new_id
    return lego


def _cluster_ring(
    cells: NDArray[np.bool_], valid: NDArray[np.bool_], min_fraction: float, min_cells: int
) -> tuple[int, NDArray[np.bool_]] | None:
    """(layer, ring) of one trim cluster: in its layers by number of trim cells, the first whose part outline the
    cluster covers well (the busiest layer can cut through a wide brim, whose outline the band does not follow)."""
    cells_per_layer = cells.sum(axis=(1, 2))
    for z in [int(layer) for layer in np.argsort(-cells_per_layer, kind="stable") if cells_per_layer[layer] > 0]:
        part_labels, _ = label(valid[z])
        ids, counts = np.unique(part_labels[cells[z]], return_counts=True)
        ring = trim_snap.part_outline(part_labels == ids[np.argmax(counts)])
        covered = int((cells[z] & ring).sum())
        if covered >= min_cells and covered >= min_fraction * int(ring.sum()):
            return z, ring
    return None


def trim_ring_mask(
    lego: NDArray[Any],
    grid: NDArray[np.bool_],
    trim_ids: Sequence[int],
    min_fraction: float = 0.15,
    min_cells: int = 4,
) -> TrimRings:
    """One full ring per trim feature. A feature = a 3D-connected cluster of cells with a trim colour (a hat band
    tilted over several layers, one boot cuff). In the layer where the cluster has most cells, the outline of the
    part (4-connected cross-section) that holds them becomes trim, if the cluster covers at least `min_fraction`
    of that outline (and `min_cells` cells). Trim cells of the cluster off the ring are 'stray'.
    Returns {id: (ring mask, stray mask)}."""
    valid = grid.astype(bool)
    rings: TrimRings = {}
    for trim_id in trim_ids:
        ring_cells = np.zeros(valid.shape, bool)
        stray_cells = np.zeros(valid.shape, bool)
        cluster_labels, cluster_count = label(valid & (lego == trim_id), structure=np.ones((3, 3, 3), bool))
        for cluster in range(1, cluster_count + 1):
            cells = cluster_labels == cluster
            if cells.sum() < min_cells:
                continue
            found = _cluster_ring(cells, valid, min_fraction, min_cells)
            if found is not None:
                z, ring = found
                ring_cells[z] |= ring
                stray_cells |= cells
        stray_cells &= ~ring_cells
        rings[trim_id] = (ring_cells, stray_cells)
    return rings


def _cleanup_locks(lego: NDArray[Any], trim_ids: Sequence[int], colour: ColourOptions) -> NDArray[np.bool_]:
    """Cells the automatic colour clean-up must not change: --paint / --paint-deep boxes, --accent boxes and every
    cell of a trim colour (rings and the trim that is spread over several layers, e.g. a tilted boot cuff)."""
    locked: NDArray[np.bool_] = np.isin(lego, list(trim_ids)) | _hand_painted_mask(lego.shape, colour)
    return locked


def _hand_painted_mask(shape: tuple[int, ...], colour: ColourOptions) -> NDArray[np.bool_]:
    """bool [z, y, x]: cells inside a --paint / --paint-deep or --accent box."""
    painted = np.zeros(shape, bool)
    boxes: list[Box] = [paint.box for paint in colour.paints] + [accent.box for accent in colour.accents]
    for box in boxes:
        painted[box] = True
    return painted


def apply_automatic_colour_cleanup(
    lego: NDArray[Any], surface: NDArray[np.bool_], trim_ids: Sequence[int], colour: ColourOptions
) -> NDArray[Any]:
    """--neutral-enclaves / --dark-enclaves / --despeckle (color_cleanup.py), after smoothing and the trim rings:
    neutral enclaves, dark enclaves, then lone patches. Prints one line per rule that ran."""
    despeckle, dark_enclaves = colour.despeckle_max_cells, colour.dark_enclave_max_cells
    if not (despeckle or colour.neutral_enclaves or dark_enclaves):
        return lego
    locked = _cleanup_locks(lego, trim_ids, colour)
    if colour.neutral_enclaves:
        palette_lab = {colour_id: rgb_to_lab(np.array([rgb]))[0] for colour_id, rgb in LEGO_RGB.items()}
        lego, changed = color_cleanup.neutral_enclaves(lego, surface, locked, palette_lab)
        print(f"Colour clean-up: {changed} cell(s) of grey / white enclosed by colourful cells recoloured")
    if dark_enclaves:
        painted = _hand_painted_mask(lego.shape, colour)
        lego, changed = color_cleanup.dark_enclaves(lego, surface, locked, dark_enclaves, painted)
        print(
            f"Colour clean-up: {changed} cell(s) in regions of <= {dark_enclaves} cells inside Black parts made Black"
        )
    if despeckle:
        lego, changed = color_cleanup.despeckle(lego, surface, locked, despeckle)
        print(f"Colour clean-up: {changed} cell(s) in lone patches of <= {despeckle} cells per layer recoloured")
    return lego


def snap_trim_to_rings(
    lego: NDArray[Any], grid: NDArray[np.bool_], surface: NDArray[np.bool_], trim: TrimRings, colour: ColourOptions
) -> tuple[NDArray[Any], TrimRings]:
    """--trim-snap: ring-less trim features get a ring in the nearest ring layer (trim_snap.py); every trim cell
    off the rings takes the colour of the nearest non-trim surface cell, so each ring is one layer. Returns
    (lego, trim with empty stray masks)."""
    options = trim_snap.SnapOptions(max_layers=colour.trim_snap_layers, min_area_ratio=colour.trim_snap_min_area_ratio)
    trim, added = trim_snap.snap_all(lego, grid.astype(bool), surface, trim, options)
    strays = np.zeros(grid.shape, bool)
    for _, stray in trim.values():
        strays |= stray
    lego = color_cleanup.nearest_fill(lego, surface & ~np.isin(lego, list(trim)), strays)
    print(
        f"Trim snap: {added} ring(s) added in the nearest trim ring layer (within {colour.trim_snap_layers} layers); "
        f"{int(strays.sum())} trim cell(s) off the rings recoloured"
    )
    return lego, {trim_id: (rings, np.zeros(grid.shape, bool)) for trim_id, (rings, _) in trim.items()}
