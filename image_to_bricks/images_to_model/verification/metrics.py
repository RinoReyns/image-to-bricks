"""Compare generated meshes: geometry against a reference, outlines and colours against the input pictures.

Pure numpy / scipy / Pillow (no nvdiffrast), so every metric is unit-tested without a GPU. All lengths are divided by
the reference height, so 0.01 means 1 % of the figure's height.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from image_to_bricks.images_to_model.clean import compute_face_areas
from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.place import apply_placement, compute_placement
from image_to_bricks.images_to_model.type_aliases import BoolArray, FloatArray, IntArray, JsonDict
from image_to_bricks.images_to_model.view_geometry import (
    DIRECTIONS,
    QUARTER_TURNS_PER_TURN,
    assign_view_directions,
    compute_mask_iou,
    project_vertices,
    split_rgba,
)

VIEW_SIZE = 256  # pixels of the renders the pictures are compared with
SUBJECT_HEIGHT_SHARE = 0.9  # renders and normalised views show the character 90 % of the picture tall
F_THRESHOLDS = (0.01, 0.02, 0.05)  # F-score distances, in reference heights
SURFACE_SAMPLES = 100000
ALIGNMENT_SAMPLES = 4000
VOLUME_IOU_RESOLUTION = 96
VOLUME_PADDING_SHARE = 0.02  # of the bounding box, around both meshes
COLUMN_JITTER = 1.234567e-6  # keeps ray columns off exact triangle edges
LEGO_HEIGHT_MM = 300.0  # build height used for the LEGO-grid IoU (a typical 30 cm figure)
LEGO_STUD_MM = 8.0
LEGO_PLATE_MM = 3.2
HAUSDORFF_PERCENTILE = 95
ALIGNMENT_SEEDS = (3, 4)  # reference, candidate samples while aligning
COMPARISON_SEEDS = (1, 2)  # mesh, reference samples while scoring
TINY = 1e-12
TINY_DETERMINANT = 1e-18
TINY_EXTENT = 1e-9
PSNR_OF_IDENTICAL = 99.0
SSIM_SIGMA = 1.5
SSIM_C1 = 0.01**2
SSIM_C2 = 0.03**2
PSNR_SCORE_DIVISOR = 50  # view matching score: silhouette IoU + PSNR / 50
DISTANCE_DIGITS = 5
SCORE_DIGITS = 4
PSNR_DIGITS = 2
MASK_THRESHOLD = 128  # resized mask pixels 0-255 at or above count as inside


@dataclass(frozen=True)
class ViewMatch:
    """Result of compare_with_views: camera direction of each named view, metrics per view, and the 4 renders
    {direction: (rgb [size, size, 3], mask [size, size])}."""

    assignment: dict[str, str]
    views: dict[str, JsonDict]
    renders: dict[str, tuple[FloatArray, BoolArray]]


def sample_surface(mesh: Mesh, count: int, seed: int = 0) -> tuple[FloatArray, FloatArray]:
    """Area-weighted random points on the surface [count, 3] with their face normals [count, 3]."""
    rng = np.random.default_rng(seed)
    areas = compute_face_areas(mesh)
    chosen = rng.choice(len(mesh.faces), size=count, p=areas / areas.sum())
    corners = mesh.vertices[mesh.faces[chosen]]
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    points = corners[:, 0] + u[:, None] * (corners[:, 1] - corners[:, 0]) + v[:, None] * (corners[:, 2] - corners[:, 0])
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), TINY)
    return points, np.asarray(normals, dtype=np.float64)


def measure_height(mesh: Mesh) -> float:
    """Extent along y."""
    return float(np.ptp(mesh.vertices[:, 1]))


def compare_surfaces(
    mesh: Mesh, reference: Mesh, count: int = SURFACE_SAMPLES, thresholds: Sequence[float] = F_THRESHOLDS
) -> JsonDict:
    """Chamfer distance, accuracy, completeness, 95 % Hausdorff, F-scores and normal consistency, in reference
    heights."""
    scale = measure_height(reference)
    points, normals = sample_surface(mesh, count, seed=COMPARISON_SEEDS[0])
    ref_points, ref_normals = sample_surface(reference, count, seed=COMPARISON_SEEDS[1])
    to_ref, nearest_ref = cKDTree(ref_points).query(points)
    to_mesh, nearest_mesh = cKDTree(points).query(ref_points)
    to_ref, to_mesh = to_ref / scale, to_mesh / scale
    hausdorff = max(np.percentile(to_ref, HAUSDORFF_PERCENTILE), np.percentile(to_mesh, HAUSDORFF_PERCENTILE))
    result = {
        "chamfer_l1": round(float((to_ref.mean() + to_mesh.mean()) / 2), DISTANCE_DIGITS),
        "accuracy_mean": round(float(to_ref.mean()), DISTANCE_DIGITS),
        "completeness_mean": round(float(to_mesh.mean()), DISTANCE_DIGITS),
        "hausdorff_95": round(float(hausdorff), DISTANCE_DIGITS),
    }
    for tau in thresholds:
        precision, recall = float((to_ref < tau).mean()), float((to_mesh < tau).mean())
        fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[f"fscore_{tau:g}"] = round(fscore, SCORE_DIGITS)
    agreement = np.abs(np.einsum("ij,ij->i", normals, ref_normals[nearest_ref]))
    agreement_back = np.abs(np.einsum("ij,ij->i", ref_normals, normals[nearest_mesh]))
    result["normal_consistency"] = round(float((agreement.mean() + agreement_back.mean()) / 2), SCORE_DIGITS)
    return result


@dataclass(frozen=True)
class ColumnGrid:
    """Vertical ray columns through the centres of an x / z grid: lowest corner [3], cell size [3] and cells per
    axis (nx, ny, nz)."""

    low: FloatArray
    step: FloatArray
    counts: tuple[int, int, int]


def collect_column_hits(mesh: Mesh, columns: ColumnGrid) -> tuple[IntArray, FloatArray]:
    """Where rays along +y through every (x, z) column centre cross the surface: (column index x * nz + z, y of
    the crossing), one entry per crossing."""
    low, step = columns.low, columns.step
    nx, nz = columns.counts[0], columns.counts[2]
    limits = np.array([nx, nz])
    triangles = mesh.vertices[mesh.faces]
    flat = triangles[:, :, [0, 2]]
    first = np.clip(np.ceil((flat.min(axis=1) - low[[0, 2]]) / step[[0, 2]] - 0.5).astype(int), 0, limits)
    last = np.clip(np.floor((flat.max(axis=1) - low[[0, 2]]) / step[[0, 2]] - 0.5).astype(int), -1, limits - 1)
    column_blocks, height_blocks = [], []
    for index in np.nonzero((last >= first).all(axis=1))[0]:
        xs = low[0] + (np.arange(first[index, 0], last[index, 0] + 1) + 0.5) * step[0] + COLUMN_JITTER
        zs = low[2] + (np.arange(first[index, 1], last[index, 1] + 1) + 0.5) * step[2] + COLUMN_JITTER
        grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
        a, b, c = flat[index]
        det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(det) < TINY_DETERMINANT:
            continue
        w1 = ((grid_x - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (grid_z - a[1])) / det
        w2 = ((b[0] - a[0]) * (grid_z - a[1]) - (grid_x - a[0]) * (b[1] - a[1])) / det
        inside = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
        if not inside.any():
            continue
        ys = triangles[index, :, 1]
        hit_y = ys[0] + w1 * (ys[1] - ys[0]) + w2 * (ys[2] - ys[0])
        ii, kk = np.nonzero(inside)
        column_blocks.append((ii + first[index, 0]) * nz + kk + first[index, 1])
        height_blocks.append(hit_y[ii, kk])
    if not column_blocks:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(column_blocks).astype(np.int64), np.concatenate(height_blocks)


def voxelize_solid(mesh: Mesh, low: FloatArray, high: FloatArray, resolution: int | Sequence[int]) -> BoolArray:
    """Solid occupancy grid [x, y, z] by ray parity along y (needs a closed mesh). resolution: cells per axis, one
    number or (nx, ny, nz)."""
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    counts = tuple(int(n) for n in np.broadcast_to(resolution, 3))
    step = (high - low) / np.array(counts)
    column_ids, heights = collect_column_hits(mesh, ColumnGrid(low, step, (counts[0], counts[1], counts[2])))
    grid = np.zeros(counts, dtype=bool)
    if len(column_ids) == 0:
        return grid
    centres_y = low[1] + (np.arange(counts[1]) + 0.5) * step[1]
    order = np.lexsort((heights, column_ids))
    column_ids, heights = column_ids[order], heights[order]
    starts = np.flatnonzero(np.r_[True, column_ids[1:] != column_ids[:-1]])
    ends = np.r_[starts[1:], len(column_ids)]
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        crossings = np.searchsorted(heights[start:end], centres_y)
        column = int(column_ids[start])
        grid[column // counts[2], :, column % counts[2]] = crossings % 2 == 1
    return grid


def compute_lego_volume_iou(mesh: Mesh, reference: Mesh, height_mm: float = LEGO_HEIGHT_MM) -> float:
    """Volume IoU on a LEGO grid (1 cell = 1 stud wide and 1 plate tall, for a build height_mm tall), which says
    whether both meshes would give the same bricks."""
    unit = measure_height(reference) / height_mm
    cell = np.array([LEGO_STUD_MM, LEGO_PLATE_MM, LEGO_STUD_MM]) * unit
    low = np.minimum(mesh.vertices.min(axis=0), reference.vertices.min(axis=0)) - cell
    high = np.maximum(mesh.vertices.max(axis=0), reference.vertices.max(axis=0)) + cell
    counts = np.ceil((high - low) / cell).astype(int)
    high = low + counts * cell
    first = voxelize_solid(mesh, low, high, counts.tolist())
    second = voxelize_solid(reference, low, high, counts.tolist())
    union = np.logical_or(first, second).sum()
    return round(float(np.logical_and(first, second).sum() / union), SCORE_DIGITS) if union else 0.0


def compute_volume_iou(mesh: Mesh, reference: Mesh, resolution: int = VOLUME_IOU_RESOLUTION) -> float:
    """Volume IoU of two closed meshes on a shared grid of resolution cells per axis."""
    low = np.minimum(mesh.vertices.min(axis=0), reference.vertices.min(axis=0))
    high = np.maximum(mesh.vertices.max(axis=0), reference.vertices.max(axis=0))
    pad = (high - low) * VOLUME_PADDING_SHARE + TINY_EXTENT
    first = voxelize_solid(mesh, low - pad, high + pad, resolution)
    second = voxelize_solid(reference, low - pad, high + pad, resolution)
    union = np.logical_or(first, second).sum()
    return round(float(np.logical_and(first, second).sum() / union), SCORE_DIGITS) if union else 0.0


def build_axis_rotations() -> list[FloatArray]:
    """The 24 rotations that map coordinate axes onto coordinate axes."""
    rotations: list[FloatArray] = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            matrix[range(3), order] = signs
            if np.isclose(np.linalg.det(matrix), 1.0):
                rotations.append(matrix)
    return rotations


def place_like(mesh: Mesh, reference: Mesh) -> Mesh:
    """The mesh scaled and moved to the reference's height, floor and x / z centre."""
    scale, offset = compute_placement(mesh.vertices, measure_height(reference))
    placed = apply_placement(mesh.vertices, scale, offset)
    low, high = reference.vertices.min(axis=0), reference.vertices.max(axis=0)
    placed += np.array([(low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2])
    return Mesh(placed, mesh.faces)


def align_to_reference(mesh: Mesh, reference: Mesh, count: int = ALIGNMENT_SAMPLES) -> tuple[Mesh, FloatArray, float]:
    """Try the 24 axis rotations, keep the one with the lowest Chamfer distance.
    Returns (aligned mesh, rotation matrix, Chamfer distance of the best rotation in reference heights)."""
    ref_points, _ = sample_surface(reference, count, seed=ALIGNMENT_SEEDS[0])
    ref_tree = cKDTree(ref_points)
    best_mesh, best_rotation, best_score = mesh, np.eye(3), np.inf
    for rotation in build_axis_rotations():
        candidate = place_like(Mesh(mesh.vertices @ rotation.T, mesh.faces), reference)
        points, _ = sample_surface(candidate, count, seed=ALIGNMENT_SEEDS[1])
        score = ref_tree.query(points)[0].mean() + cKDTree(points).query(ref_points)[0].mean()
        if score < best_score:
            best_mesh, best_rotation, best_score = candidate, rotation, score
    return best_mesh, best_rotation, round(float(best_score / 2 / measure_height(reference)), DISTANCE_DIGITS)


def compute_pixel_coordinates(
    vertices: FloatArray, direction: str, size: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(pixel x, pixel y, depth) of every vertex in a size x size render from direction, the character
    SUBJECT_HEIGHT_SHARE of the picture tall and centred."""
    screen_x, screen_y, depth = project_vertices(vertices, direction)
    height = max(float(np.ptp(screen_y)), TINY_EXTENT)
    scale = SUBJECT_HEIGHT_SHARE * size / height
    centre_x = (screen_x.min() + screen_x.max()) / 2
    centre_y = (screen_y.min() + screen_y.max()) / 2
    pixel_x = (screen_x - centre_x) * scale + size / 2
    pixel_y = size / 2 - (screen_y - centre_y) * scale
    return pixel_x, pixel_y, depth


def rasterize_silhouette(mesh: Mesh, direction: str, size: int = VIEW_SIZE) -> BoolArray:
    """Outline of the mesh seen from direction [size, size]."""
    pixel_x, pixel_y, _ = compute_pixel_coordinates(mesh.vertices, direction, size)
    canvas = Image.new("1", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    for face in mesh.faces:
        draw.polygon([(pixel_x[i], pixel_y[i]) for i in face], fill=1)
    return np.asarray(canvas, dtype=bool)


def render_vertex_colours(
    mesh: Mesh, colours: FloatArray, direction: str, size: int = VIEW_SIZE
) -> tuple[FloatArray, BoolArray]:
    """Z-buffered orthographic render with per-vertex colours (RGB 0-1). Returns (rgb [size, size, 3], mask)."""
    pixel_x, pixel_y, depth = compute_pixel_coordinates(mesh.vertices, direction, size)
    zbuffer = np.full((size, size), -np.inf)
    image = np.zeros((size, size, 3))
    for face in mesh.faces:
        xs, ys = pixel_x[face], pixel_y[face]
        x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), size - 1)
        y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), size - 1)
        if x0 > x1 or y0 > y1:
            continue
        det = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(det) < TINY:
            continue
        grid_x, grid_y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w1 = ((grid_x - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (grid_y - ys[0])) / det
        w2 = ((xs[1] - xs[0]) * (grid_y - ys[0]) - (grid_x - xs[0]) * (ys[1] - ys[0])) / det
        w0 = 1 - w1 - w2
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * depth[face[0]] + w1 * depth[face[1]] + w2 * depth[face[2]]
        window = zbuffer[y0 : y1 + 1, x0 : x1 + 1]
        nearer = inside & (z > window)
        if not nearer.any():
            continue
        window[nearer] = z[nearer]
        colour = w0[..., None] * colours[face[0]] + w1[..., None] * colours[face[1]] + w2[..., None] * colours[face[2]]
        image[y0 : y1 + 1, x0 : x1 + 1][nearer] = colour[nearer]
    return image, np.asarray(np.isfinite(zbuffer), dtype=bool)


def normalise_view(rgb: FloatArray, mask: BoolArray, size: int = VIEW_SIZE) -> tuple[FloatArray, BoolArray]:
    """Crop to the character, scale its height to SUBJECT_HEIGHT_SHARE of size, centre it (removes framing
    differences). Returns (rgb [size, size, 3], mask [size, size])."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.zeros((size, size, 3)), np.zeros((size, size), dtype=bool)
    top, bottom, left, right = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    scale = SUBJECT_HEIGHT_SHARE * size / (bottom - top)
    width = max(round(float((right - left) * scale)), 1)
    height = max(round(float((bottom - top) * scale)), 1)
    crop_rgb = Image.fromarray((np.clip(rgb[top:bottom, left:right], 0, 1) * 255).astype(np.uint8))
    crop_mask = Image.fromarray(mask[top:bottom, left:right].astype(np.uint8) * 255)
    resized_rgb = np.asarray(crop_rgb.resize((width, height), Image.Resampling.BILINEAR)) / 255.0
    resized_mask = np.asarray(crop_mask.resize((width, height), Image.Resampling.BILINEAR)) >= MASK_THRESHOLD
    out_rgb, out_mask = np.zeros((size, size, 3)), np.zeros((size, size), dtype=bool)
    offset_y, offset_x = (size - height) // 2, (size - width) // 2
    source_x0, target_x0 = max(-offset_x, 0), max(offset_x, 0)
    span = min(width - source_x0, size - target_x0)
    out_rgb[offset_y : offset_y + height, target_x0 : target_x0 + span] = resized_rgb[:, source_x0 : source_x0 + span]
    out_mask[offset_y : offset_y + height, target_x0 : target_x0 + span] = resized_mask[:, source_x0 : source_x0 + span]
    return out_rgb, out_mask


def compute_masked_psnr(first: FloatArray, second: FloatArray, mask: BoolArray) -> float:
    """PSNR (dB, values 0-1) over the masked pixels; PSNR_OF_IDENTICAL for identical pixels, 0 for an empty mask."""
    if not mask.any():
        return 0.0
    error = float(((first[mask] - second[mask]) ** 2).mean())
    return PSNR_OF_IDENTICAL if error == 0 else float(10 * np.log10(1.0 / error))


def compute_ssim(first: FloatArray, second: FloatArray, sigma: float = SSIM_SIGMA) -> float:
    """Grey-level SSIM (Wang et al. 2004, Gaussian window) of two RGB pictures, values 0-1."""
    a, b = first.mean(axis=-1), second.mean(axis=-1)
    mu_a, mu_b = ndimage.gaussian_filter(a, sigma), ndimage.gaussian_filter(b, sigma)
    var_a = ndimage.gaussian_filter(a * a, sigma) - mu_a**2
    var_b = ndimage.gaussian_filter(b * b, sigma) - mu_b**2
    cov = ndimage.gaussian_filter(a * b, sigma) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + SSIM_C1) * (2 * cov + SSIM_C2)
    ssim = numerator / ((mu_a**2 + mu_b**2 + SSIM_C1) * (var_a + var_b + SSIM_C2))
    return float(ssim.mean())


def compare_view(image: Image.Image, render_rgb: FloatArray, render_mask: BoolArray) -> JsonDict:
    """Silhouette IoU, PSNR (inside both masks) and SSIM (character on black) of one picture against a render."""
    image_rgb, image_mask = normalise_view(*split_rgba(image))
    model_rgb, model_mask = normalise_view(render_rgb, render_mask)
    both = image_mask & model_mask
    ssim = compute_ssim(image_rgb * image_mask[..., None], model_rgb * model_mask[..., None])
    return {
        "silhouette_iou": round(compute_mask_iou(image_mask, model_mask), SCORE_DIGITS),
        "psnr": round(compute_masked_psnr(image_rgb, model_rgb, both), PSNR_DIGITS),
        "ssim": round(ssim, SCORE_DIGITS),
    }


def compare_with_views(
    mesh: Mesh, colours: FloatArray, views: dict[str, Image.Image], size: int = VIEW_SIZE
) -> ViewMatch:
    """Match a coloured mesh to the input pictures (front / left / back / right). The model may face any of the 4
    horizontal directions and 'left' may mean either side, so all 8 assignments are tried and the best
    (silhouette IoU + PSNR / PSNR_SCORE_DIVISOR, summed over the views) is kept."""
    renders = {direction: render_vertex_colours(mesh, colours, direction, size) for direction in DIRECTIONS}
    best: tuple[float, dict[str, str], dict[str, JsonDict]] | None = None
    for front_index, side_sign in itertools.product(range(QUARTER_TURNS_PER_TURN), (1, -1)):
        assignment = assign_view_directions(front_index, side_sign)
        scores = {name: compare_view(image, *renders[assignment[name]]) for name, image in views.items()}
        total = sum(item["silhouette_iou"] + item["psnr"] / PSNR_SCORE_DIVISOR for item in scores.values())
        if best is None or total > best[0]:
            best = (total, assignment, scores)
    if best is None:
        raise ValueError("no assignment of views was tried")
    return ViewMatch(best[1], best[2], renders)


def summarise_views(view_metrics: dict[str, JsonDict], seen: Sequence[str]) -> JsonDict:
    """Mean of each metric over the views the model was given ("seen_...") and over the views it never saw
    ("unseen_...", None without such views)."""
    summary: JsonDict = {}
    groups = (("seen", [n for n in view_metrics if n in seen]), ("unseen", [n for n in view_metrics if n not in seen]))
    for label, names in groups:
        for key in ("silhouette_iou", "psnr", "ssim"):
            values = [view_metrics[name][key] for name in names]
            summary[f"{label}_{key}"] = round(float(np.mean(values)), SCORE_DIGITS) if values else None
    return summary
