"""Automatic colour clean-up of the surface (--despeckle N, --neutral-enclaves, --dark-enclaves N).

Runs at the end of colouring, on the LEGO colour ids of the voxel grid. Two kinds of outliers:
1. lone patches: in one layer, up to N surface cells of a colour that none of their surface neighbours in that
   layer share; every such patch costs extra small plates. They take the most common neighbour colour.
2. neutral enclaves: a region of a light neutral colour (grey, white) whose border is (almost) all colourful, e.g. a
   light highlight on orange fur mapped to Light Bluish Gray. It takes the colours of the nearest border cells.
3. dark enclaves: a small region (up to N surface cells, 3D) of another colour inside a Black part whose border is
   mostly Black, e.g. feather colour sampled onto a hat crown: it reads as a hole in the dark part and becomes Black.
Locked cells (paint and accent boxes, every trim-colour cell) never change: trim such as a boot cuff that sits at an
angle is spread over several layers in small patches and is still wanted.

Arrays: lego [z, y, x] uint8 colour ids; surface / locked [z, y, x] bool.
"""

from collections import Counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from image_to_bricks.consts import BLACK_ID

DESPECKLE_MAX = 2  # largest lone patch (cells in one layer) that is recoloured (--despeckle N)
DESPECKLE_PASSES = 3  # repeat, since one recolouring can leave a new lone patch next to it
ENCLAVE_BORDER_SHARE = 0.9  # share of an enclave's border cells that must be colourful
DARK_ENCLAVE_BORDER_SHARE = 0.7  # share of a dark enclave's border cells that must be Black (--dark-enclaves)
MIN_LONE_PATCH_NEIGHBOURS = 2  # a patch with fewer surface neighbours in its layer is an edge, not a speckle
NEUTRAL_CHROMA = 12.0  # palette colours with Lab chroma below this count as neutral (grey, white, black)
_IN_LAYER = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def nearest_fill(lego: NDArray[Any], source: NDArray[Any], target: NDArray[Any]) -> NDArray[Any]:
    """Copy of lego where every target cell takes the colour of its nearest source cell (3D distance)."""
    if not source.any() or not target.any():
        return lego.copy()
    _, idx = distance_transform_edt(~source, return_indices=True)
    out = lego.copy()
    t = np.nonzero(target)
    out[t] = lego[idx[0][t], idx[1][t], idx[2][t]]
    return out


def _layer_patches(colours: NDArray[Any], surface: NDArray[Any]) -> tuple[NDArray[Any], int]:
    """(label array, count) of same-colour, 4-connected surface patches in one layer [y, x]."""
    labels = np.zeros(colours.shape, np.int32)
    total = 0
    for c in np.unique(colours[surface]):
        lab, n = label(surface & (colours == c), structure=_IN_LAYER)
        labels[lab > 0] = lab[lab > 0] + total
        total += n
    return labels, total


def _lone_patch_colour(colours: NDArray[Any], surface: NDArray[Any], patch: NDArray[Any], locked: NDArray[Any]) -> int:
    """Most common colour of the surface cells around `patch` in its layer, or -1 if the patch is not lone (a
    neighbour shares its colour, it has fewer than 2 neighbours, or it holds a locked cell)."""
    if locked[patch].any():
        return -1
    ring = binary_dilation(patch, structure=_IN_LAYER) & surface & ~patch
    own = colours[patch][0]
    around = colours[ring]
    if len(around) < MIN_LONE_PATCH_NEIGHBOURS or (around == own).any():
        return -1
    return int(Counter(around.tolist()).most_common(1)[0][0])


def despeckle(
    lego: NDArray[Any], surface: NDArray[Any], locked: NDArray[Any], max_cells: int = DESPECKLE_MAX
) -> tuple[NDArray[Any], int]:
    """Lone patches of at most max_cells per layer take their neighbours' most common colour.
    Returns (new lego, cells recoloured)."""
    out, changed = lego.copy(), 0
    for _ in range(DESPECKLE_PASSES):
        changed_pass = 0
        for z in range(out.shape[0]):
            if not surface[z].any():
                continue
            labels, n = _layer_patches(out[z], surface[z])
            sizes = np.bincount(labels.ravel(), minlength=n + 1)
            for k in np.nonzero((sizes > 0) & (sizes <= max_cells))[0]:
                if k == 0:
                    continue
                patch = labels == k
                new = _lone_patch_colour(out[z], surface[z], patch, locked[z])
                if new >= 0:
                    out[z][patch] = new
                    changed_pass += int(patch.sum())
        changed += changed_pass
        if not changed_pass:
            break
    return out, changed


def neutral_enclaves(
    lego: NDArray[Any],
    surface: NDArray[Any],
    locked: NDArray[Any],
    palette_lab: dict[int, NDArray[Any]],
) -> tuple[NDArray[Any], int]:
    """Light neutral regions (3D surface-connected, not Black) whose border cells are at least ENCLAVE_BORDER_SHARE
    colourful take the colours of the nearest colourful surface cells. palette_lab {id: Lab (3,)}.
    Returns (new lego, cells recoloured)."""
    chroma = {c: float(np.hypot(lab[1], lab[2])) for c, lab in palette_lab.items()}
    neutral = [c for c, ch in chroma.items() if ch < NEUTRAL_CHROMA and c != BLACK_ID]
    colourful = np.isin(lego, [c for c, ch in chroma.items() if ch >= NEUTRAL_CHROMA]) & surface
    target = np.zeros(lego.shape, bool)
    for c in neutral:
        regions, n = label(surface & (lego == c), structure=np.ones((3, 3, 3), bool))
        for k in range(1, n + 1):
            region = regions == k
            border = binary_dilation(region, structure=np.ones((3, 3, 3), bool)) & surface & ~region
            if not locked[region].any() and border.any() and colourful[border].mean() >= ENCLAVE_BORDER_SHARE:
                target |= region
    return nearest_fill(lego, colourful, target), int(target.sum())


def dark_enclaves(
    lego: NDArray[Any],
    surface: NDArray[Any],
    locked: NDArray[Any],
    max_cells: int,
    hand_painted: NDArray[Any] | None = None,
) -> tuple[NDArray[Any], int]:
    """Face-connected surface regions of one non-Black colour with at most max_cells cells whose surface border
    is at least DARK_ENCLAVE_BORDER_SHARE Black become Black. Locked cells are never changed and do not count in
    the border; regions touching a hand_painted cell (--paint / --accent boxes) are left alone (a painted black
    mouth and pupils must not turn the muzzle between them black). Returns (new lego, cells changed)."""
    face_connected = np.zeros((3, 3, 3), bool)
    face_connected[1, 1, :] = face_connected[1, :, 1] = face_connected[:, 1, 1] = True
    out, changed = lego.copy(), 0
    for colour in np.unique(lego[surface]):
        if colour == BLACK_ID:
            continue
        regions, region_count = label(surface & (lego == colour), structure=face_connected)
        sizes = np.bincount(regions.ravel(), minlength=region_count + 1)
        for region_index in np.nonzero((sizes > 0) & (sizes <= max_cells))[0]:
            if region_index == 0:
                continue
            region = regions == region_index
            around = binary_dilation(region, structure=face_connected) & ~region
            if hand_painted is not None and (around & hand_painted).any():
                continue
            border = around & surface & ~locked
            if locked[region].any() or not border.any():
                continue
            if (lego[border] == BLACK_ID).mean() >= DARK_ENCLAVE_BORDER_SHARE:
                out[region] = BLACK_ID
                changed += int(region.sum())
    return out, changed
