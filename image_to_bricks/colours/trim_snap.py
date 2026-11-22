"""Trim rings for trim that did not form a ring of its own (--trim-snap N).

`surface_cleanup.trim_ring_mask` makes one ring per trim feature, in the layer where the feature has most cells, if
the feature already covers enough of that layer's outline. A matching feature on another part (the second boot
cuff, sitting at an angle or higher up on the merged body) can stay a few scattered cells. Here every such feature
snaps to the nearest existing ring layer of the same colour (at most N layers away): in that layer the part right
below / above it (the cross-section part under most of its cells, same x / y; none = no snap) gets a full ring
too, unless it already has one. Every ring is then the whole visible band of its part in that layer (the outline
plus the cells seen from above or below, e.g. the top of a boot inside the cuff), and every other trim cell is a
stray (it takes the neighbours' colour), so each ring is exactly one layer and matching parts look the same.

Two checks keep a snapped ring on a part that matches (a second boot, not the tail beside it):
- similar size: the part's cross-section area must be at least `min_area_ratio` x (and at most 1 / that) of a
  part that already has an original ring in that layer (a boot of 16-19 cells vs a 10-cell tail: 0.53);
- same body: the trim feature must reach the part through filled cells between its layers and the ring layer
  (red on a separate piece hanging over a boot does not ring the boot).

Arrays: lego [z, y, x] colour ids; valid / surface / rings [z, y, x] bool.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, label

from image_to_bricks.colours.color_cleanup import nearest_fill

TRIM_SNAP_LAYERS = 12  # farthest ring layer (in plates) a ring-less trim feature may snap to (--trim-snap N)
TRIM_SNAP_MIN_AREA_RATIO = 0.65  # snapped part area / ringed part area must be in [r, 1/r] (--trim-snap-similarity)
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def part_outline(part: NDArray[Any]) -> NDArray[Any]:
    """Outline [y, x] of a cross-section part: its cells with a 4-neighbour outside the part."""
    outline: NDArray[Any] = part & ~binary_erosion(part, structure=_CROSS, border_value=0)
    return outline


def part_under(layer: NDArray[Any], cells_yx: NDArray[Any]) -> NDArray[Any]:
    """The 4-connected part of `layer` [y, x] that lies directly below / above most of cells_yx [n, 2] (same x / y);
    empty mask if none of the cells is over a part (trim beside a part, e.g. by a tail, does not snap to it)."""
    parts, _ = label(layer, structure=_CROSS)
    under = parts[cells_yx[:, 0], cells_yx[:, 1]]
    under = under[under > 0]
    if not len(under):
        return np.zeros(layer.shape, bool)
    part_mask: NDArray[Any] = parts == np.bincount(under).argmax()
    return part_mask


@dataclass(frozen=True)
class SnapOptions:
    """max_layers: farthest ring layer (plates) a feature may snap to; min_area_ratio: size similarity to a part
    with an original ring in that layer (0 = no size check)."""

    max_layers: int = TRIM_SNAP_LAYERS
    min_area_ratio: float = TRIM_SNAP_MIN_AREA_RATIO


DEFAULT_SNAP_OPTIONS = SnapOptions()


def ringed_part_areas(layer: NDArray[Any], layer_rings: NDArray[Any]) -> list[int]:
    """Cross-section areas (cells) of the 4-connected parts of `layer` [y, x] that hold a ring cell."""
    parts, _ = label(layer, structure=_CROSS)
    return [int((parts == k).sum()) for k in np.unique(parts[layer_rings]) if k]


def is_similar_size(area: int, reference_areas: list[int], min_area_ratio: float) -> bool:
    """True if `area` is within [r, 1 / r] of at least one reference area (always true for r = 0)."""
    if min_area_ratio <= 0:
        return True
    return any(min(area, reference) / max(area, reference) >= min_area_ratio for reference in reference_areas)


def is_same_body(valid: NDArray[Any], part: NDArray[Any], ring_layer: int, feature_cells: NDArray[Any]) -> bool:
    """True if the trim feature (cells [n, 3] as z, y, x) and `part` (mask [y, x] in ring_layer) are joined by
    face-connected filled cells inside the layers from the ring layer to the feature."""
    lowest = min(ring_layer, int(feature_cells[:, 0].min()))
    highest = max(ring_layer, int(feature_cells[:, 0].max()))
    bodies, _ = label(valid[lowest : highest + 1])
    part_bodies = set(np.unique(bodies[ring_layer - lowest][part]).tolist()) - {0}
    feature_bodies = bodies[feature_cells[:, 0] - lowest, feature_cells[:, 1], feature_cells[:, 2]]
    return bool(part_bodies & set(feature_bodies.tolist()))


def snap_rings(
    lego: NDArray[Any],
    valid: NDArray[Any],
    trim_id: int,
    rings: NDArray[Any],
    options: SnapOptions = DEFAULT_SNAP_OPTIONS,
) -> tuple[NDArray[Any], int]:
    """Rings of `trim_id` extended by the ring-less features that snap to a ring layer (see the module docstring).
    Returns (new ring mask, rings added)."""
    ring_layers = np.nonzero(rings.any(axis=(1, 2)))[0]
    out, added = rings.copy(), 0
    if not len(ring_layers):
        return out, 0
    reference_areas = {int(z): ringed_part_areas(valid[z], rings[z]) for z in ring_layers}
    features, feature_count = label(valid & (lego == trim_id) & ~rings, structure=np.ones((3, 3, 3), bool))
    for feature_index in range(1, feature_count + 1):
        cells = np.argwhere(features == feature_index)
        gap = np.abs(cells[:, :1] - ring_layers[None, :]).min(axis=0)
        if gap.min() > options.max_layers:
            continue
        z = int(ring_layers[int(np.argmin(gap))])
        part = part_under(valid[z], cells[:, 1:])
        if not part.any() or (out[z] & part).any():
            continue
        if not is_similar_size(int(part.sum()), reference_areas[z], options.min_area_ratio):
            continue
        if is_same_body(valid, part, z, cells):
            out[z] |= part_outline(part)
            added += 1
    return out, added


def visible_bands(valid: NDArray[Any], surface: NDArray[Any], rings: NDArray[Any]) -> NDArray[Any]:
    """rings widened to every visible (surface) cell of each ringed part in its layer."""
    out = rings.copy()
    for z in np.nonzero(rings.any(axis=(1, 2)))[0]:
        parts, _ = label(valid[z], structure=_CROSS)
        for k in np.unique(parts[rings[z]]):
            if k:
                out[z] |= (parts == k) & surface[z]
    return out


def snap_all(
    lego: NDArray[Any],
    valid: NDArray[Any],
    surface: NDArray[Any],
    trim: dict[int, tuple[NDArray[Any], NDArray[Any]]],
    options: SnapOptions = DEFAULT_SNAP_OPTIONS,
) -> tuple[dict[int, tuple[NDArray[Any], NDArray[Any]]], int]:
    """trim = {id: (ring mask, stray mask)} from trim_ring_mask -> the same with snapped rings added, every ring
    widened to its part's visible band, and every trim cell off the rings as stray. Returns (new trim, rings added)."""
    out, added = {}, 0
    for tid, (rings, _stray) in trim.items():
        rings2, n = snap_rings(lego, valid, tid, rings, options)
        rings2 = visible_bands(valid, surface, rings2)
        out[tid] = (rings2, valid & (lego == tid) & ~rings2)
        added += n
    return out, added


def clear_strays(
    lego: NDArray[Any], surface: NDArray[Any], trim: dict[int, tuple[NDArray[Any], NDArray[Any]]]
) -> tuple[NDArray[Any], int]:
    """Trim-colour surface cells off the rings take the colour of the nearest non-trim surface cell. Used again at
    the end of colouring: smoothing and region merging can spread a trim colour next to a ring.
    Returns (new lego, cells recoloured)."""
    rings = np.zeros(lego.shape, bool)
    for ring, _ in trim.values():
        rings |= ring
    is_trim = np.isin(lego, list(trim))
    strays = surface & is_trim & ~rings
    return nearest_fill(lego, surface & ~is_trim, strays), int(strays.sum())
