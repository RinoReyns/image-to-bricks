"""RGB -> palette colour: Lab conversion, CIEDE2000, nearest colour, k-means groups, region growing."""

from collections import deque
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from image_to_bricks.consts import BLACK_ID, COLOR_NAMES, LEGO_RGB
from image_to_bricks.options import ColourOptions

DEFAULT_COLOUR = ColourOptions()


L_WEIGHT = 1.0  # lightness weight in matching (<1 = lightness matters less)


SAT_BOOST = 1.3  # chroma multiplier before matching (photos/mesh are faded)


LIGHT_GAMMA = 0.85  # <1 lifts dark tones (shadows) before matching


KMEANS_CHROMA_W = 2.0  # weight of a/b vs L when clustering (keeps small coloured details)


BLACK_L_MAX = 52  # clusters darker than this (Lab L, after enhance)...


BLACK_C_MAX = 20  # ...and less saturated than this become Black

HALF_TURN_DEGREES = 180  # hue angles: half and full turn
FULL_TURN_DEGREES = 360


def rgb_to_lab(rgb: ArrayLike) -> NDArray[np.float32]:
    """RGB (0-255, [...,3]) -> true CIE Lab floats (L 0-100, a/b signed)."""
    arr = np.asarray(rgb, np.float32).reshape(-1, 1, 3) / 255.0
    return np.asarray(cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3), dtype=np.float32)


def ciede2000(
    lab1: NDArray[np.floating[Any]], lab2: NDArray[np.floating[Any]], lightness_factor: float = 1.0
) -> NDArray[Any]:
    """Vectorised CIEDE2000 colour difference. lab1 (N,3), lab2 (M,3) -> (N,M)."""
    lightness_1, a1, b1 = [lab1[:, None, i] for i in range(3)]
    lightness_2, a2, b2 = [lab2[None, :, i] for i in range(3)]
    chroma_1, chroma_2 = np.hypot(a1, b1), np.hypot(a2, b2)
    chroma_mean = (chroma_1 + chroma_2) / 2
    chroma_compensation = 0.5 * (1 - np.sqrt(chroma_mean**7 / (chroma_mean**7 + 25.0**7)))
    a1_prime, a2_prime = a1 * (1 + chroma_compensation), a2 * (1 + chroma_compensation)
    chroma_1_prime, chroma_2_prime = np.hypot(a1_prime, b1), np.hypot(a2_prime, b2)
    hue_1_prime = np.degrees(np.arctan2(b1, a1_prime)) % FULL_TURN_DEGREES
    hue_2_prime = np.degrees(np.arctan2(b2, a2_prime)) % FULL_TURN_DEGREES
    delta_lightness, delta_chroma = lightness_2 - lightness_1, chroma_2_prime - chroma_1_prime
    delta_hue_degrees = hue_2_prime - hue_1_prime
    delta_hue_degrees = np.where(
        delta_hue_degrees > HALF_TURN_DEGREES,
        delta_hue_degrees - FULL_TURN_DEGREES,
        np.where(delta_hue_degrees < -HALF_TURN_DEGREES, delta_hue_degrees + FULL_TURN_DEGREES, delta_hue_degrees),
    )
    delta_hue_degrees = np.where(chroma_1_prime * chroma_2_prime == 0, 0, delta_hue_degrees)
    delta_hue = 2 * np.sqrt(chroma_1_prime * chroma_2_prime) * np.sin(np.radians(delta_hue_degrees / 2))
    lightness_mean, chroma_mean_prime = (lightness_1 + lightness_2) / 2, (chroma_1_prime + chroma_2_prime) / 2
    hue_sum = hue_1_prime + hue_2_prime
    hue_mean = (
        np.where(
            np.abs(hue_1_prime - hue_2_prime) > HALF_TURN_DEGREES,
            np.where(hue_sum < FULL_TURN_DEGREES, hue_sum + FULL_TURN_DEGREES, hue_sum - FULL_TURN_DEGREES),
            hue_sum,
        )
        / 2
    )
    hue_mean = np.where(chroma_1_prime * chroma_2_prime == 0, hue_sum, hue_mean)
    hue_weight = (
        1
        - 0.17 * np.cos(np.radians(hue_mean - 30))
        + 0.24 * np.cos(np.radians(2 * hue_mean))
        + 0.32 * np.cos(np.radians(3 * hue_mean + 6))
        - 0.20 * np.cos(np.radians(4 * hue_mean - 63))
    )
    rotation_degrees = 30 * np.exp(-(((hue_mean - 275) / 25) ** 2))
    rotation_chroma = 2 * np.sqrt(chroma_mean_prime**7 / (chroma_mean_prime**7 + 25.0**7))
    lightness_scale = 1 + 0.015 * (lightness_mean - 50) ** 2 / np.sqrt(20 + (lightness_mean - 50) ** 2)
    chroma_scale, hue_scale = 1 + 0.045 * chroma_mean_prime, 1 + 0.015 * chroma_mean_prime * hue_weight
    rotation_term = -np.sin(np.radians(2 * rotation_degrees)) * rotation_chroma
    distance: NDArray[Any] = np.sqrt(
        (delta_lightness / (lightness_factor * lightness_scale)) ** 2
        + (delta_chroma / chroma_scale) ** 2
        + (delta_hue / hue_scale) ** 2
        + rotation_term * (delta_chroma / chroma_scale) * (delta_hue / hue_scale)
    )
    return distance


_PAL_IDS = np.array(list(LEGO_RGB.keys()))


_PAL_LAB = rgb_to_lab(np.array(list(LEGO_RGB.values())))


def nearest_lego_color(rgb: ArrayLike, allowed: Sequence[int] | None = None) -> int:
    """Closest palette id by CIEDE2000. L_WEIGHT < 1 makes lightness matter less.
    allowed: palette ids to choose from (None or empty = the whole palette)."""
    if allowed:
        keep = np.isin(_PAL_IDS, allowed)
        ids, labs = _PAL_IDS[keep], _PAL_LAB[keep]
    else:
        ids, labs = _PAL_IDS, _PAL_LAB
    distances = ciede2000(rgb_to_lab(rgb), labs, lightness_factor=1.0 / L_WEIGHT if L_WEIGHT else 1.0)
    return int(ids[np.argmin(distances[0])])


def enhance_colors(colors: NDArray[np.uint8], mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Counter faded/shaded input: multiply chroma by SAT_BOOST and lift
    dark tones with LIGHT_GAMMA (applied in Lab, only on masked voxels)."""
    out = colors.copy()
    lab = rgb_to_lab(colors[mask])
    lab[:, 0] = 100.0 * (np.clip(lab[:, 0], 0, 100) / 100.0) ** LIGHT_GAMMA
    lab[:, 1:] *= SAT_BOOST
    rgb = cv2.cvtColor(lab.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    out[mask] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def cluster_to_lego(
    colors: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    k: int,
    allowed_ids: Sequence[int] | None = None,
    max_colours: int = DEFAULT_COLOUR.max_colours,
) -> NDArray[Any]:
    """K-means the masked voxel colours in Lab; each cluster gets ONE LEGO
    colour (from its median), so light and shadow of the same material
    don't split into different bricks. allowed_ids: palette ids to use (None = all); max_colours: keep the N
    most-used colours and re-match the other clusters to them (0 = no limit)."""
    lab = rgb_to_lab(colors[mask]).astype(np.float32)
    feat = lab.copy()
    feat[:, 1:] *= KMEANS_CHROMA_W  # hue/chroma differences (e.g. red trim) count more
    k = int(min(k, len(np.unique(lab, axis=0))))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    # bestLabels None is valid at run time (not with KMEANS_USE_INITIAL_LABELS); the opencv stubs lack it
    _, labels, _ = cv2.kmeans(feat, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)  # type: ignore[call-overload]
    labels = labels.ravel()
    ids = np.zeros(len(lab), np.uint8)
    rgb_masked = colors[mask]

    def pick(median_rgb: NDArray[Any], candidates: Sequence[int] | None = None) -> int:
        median_lab = rgb_to_lab(median_rgb)[0]
        is_dark_neutral = median_lab[0] <= BLACK_L_MAX and np.hypot(median_lab[1], median_lab[2]) <= BLACK_C_MAX
        if (
            is_dark_neutral
            and (candidates is None or BLACK_ID in candidates)
            and (not allowed_ids or BLACK_ID in allowed_ids)
        ):
            return BLACK_ID  # dark neutral -> never grey
        return nearest_lego_color(median_rgb, candidates if candidates is not None else allowed_ids)

    clusters: list[list[Any]] = []
    for c in np.argsort(-np.bincount(labels, minlength=k)):
        sel = labels == c
        if sel.any():
            med = np.median(rgb_masked[sel], axis=0)
            clusters.append([sel, med, pick(med)])

    # Generic palette limit: keep the max_colours most-used LEGO colours and
    # re-match the remaining clusters to those (fewer, cleaner colour patches).
    used: dict[int, int] = {}
    for sel, _, cid in clusters:
        used[cid] = used.get(cid, 0) + int(sel.sum())
    if max_colours and len(used) > max_colours:
        kept = sorted(used, key=lambda colour_id: used[colour_id], reverse=True)[:max_colours]
        print(f"Limiting to {max_colours} colours: {[COLOR_NAMES[c] for c in kept]}")
        for cl in clusters:
            if cl[2] not in kept:
                cl[2] = pick(cl[1], kept)

    print("Colour clusters (voxels, median RGB -> LEGO):")
    for sel, med, cid in clusters:
        ids[sel] = cid
        print(f"  {int(sel.sum()):5d}  {tuple(int(v) for v in med)} -> {COLOR_NAMES[cid]}")
    lego = np.zeros(mask.shape, np.uint8)
    lego[mask] = ids
    return lego


NEIGHBOUR_STEPS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))  # 6-connected (z, y, x)

type Voxel = tuple[int, int, int]  # (z, y, x)


def _grow_region(
    rgb_colors: NDArray[np.uint8], grid: NDArray[np.bool_], visited: NDArray[np.bool_], start: Voxel, threshold: float
) -> tuple[list[Voxel], NDArray[np.float32]]:
    """Breadth-first region from `start` over filled, unvisited 6-neighbours whose RGB lies within `threshold` of
    the region's running mean (marks them in `visited`). Returns (voxels, mean RGB)."""
    layer_count, depth, width = grid.shape
    queue = deque([start])
    visited[start] = True
    region = []
    region_sum = rgb_colors[start].astype(np.float32).copy()
    region_count = 1
    while queue:
        z, y, x = queue.popleft()
        region.append((z, y, x))
        region_mean = region_sum / region_count
        for dz, dy, dx in NEIGHBOUR_STEPS:
            neighbour = (z + dz, y + dy, x + dx)
            if not (0 <= neighbour[0] < layer_count and 0 <= neighbour[1] < depth and 0 <= neighbour[2] < width):
                continue
            if not grid[neighbour] or visited[neighbour]:
                continue
            rgb = rgb_colors[neighbour].astype(np.float32)
            if np.linalg.norm(rgb - region_mean) <= threshold:
                visited[neighbour] = True
                queue.append(neighbour)
                region_sum += rgb
                region_count += 1
    return region, region_sum / region_count


def map_regions_to_lego(
    rgb_colors: NDArray[np.uint8],
    grid: NDArray[np.bool_],
    color_threshold: float = 22,
    min_region_size: int = 6,
    allowed: Sequence[int] | None = None,
) -> NDArray[np.uint8]:
    """Segment RGB colors into coherent regions using region growing
    with a running mean.

    Each connected region receives exactly one LEGO color.
    """
    lego = np.zeros(grid.shape, dtype=np.uint8)
    visited = np.zeros(grid.shape, dtype=bool)
    for z0, y0, x0 in np.argwhere(grid):
        start = (int(z0), int(y0), int(x0))
        if visited[start]:
            continue
        region, mean_rgb = _grow_region(rgb_colors, grid, visited, start, color_threshold)
        if len(region) < min_region_size:  # small regions are likely texture noise
            for voxel in region:
                lego[voxel] = nearest_lego_color(rgb_colors[voxel], allowed)
            continue
        lego_color = nearest_lego_color(mean_rgb, allowed)
        for voxel in region:
            lego[voxel] = lego_color
    return lego
