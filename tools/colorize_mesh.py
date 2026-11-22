#!/usr/bin/env python3
r"""Colour an untextured mesh from reference photos (colorize_mesh.py).

Colors an untextured 3D mesh (.glb/.gltf/.obj/...) using a handful of
reference photos of the same object from different (unknown) angles.

Pipeline:
  1. Load mesh, center it (x/z centered, feet/base at y=0)
  2. Extract a clean foreground silhouette from each photo
     (background-color subtraction + shadow removal + largest-blob filter)
  3. For each photo INDEPENDENTLY, search for the camera pose (azimuth,
     elevation, distance, vertical FOV, look-at height) whose rendered
     silhouette best matches that photo's silhouette (IoU), using a fast
     decimated copy of the mesh for the search
  4. With calibrated cameras, render a depth buffer per photo (full-res
     mesh) to determine which vertices are actually visible (not
     self-occluded), project visible vertices into the photo, and sample
     color there, weighted by how face-on each vertex is to that camera
  5. Per vertex, keep the color of the photo that sees it most face-on
  6. Fill vertices never seen by any camera via neighbor-color propagation
  7. Export a vertex-colored .glb

Dependencies:
    pip install trimesh numpy scipy pillow pyrender PyOpenGL

Rendering backend: on Linux this script auto-selects EGL (headless
software rendering via Mesa/llvmpipe -- works on servers/containers with
no GPU/display). On Windows/macOS it leaves PYOPENGL_PLATFORM unset so
pyrender uses its normal windowed backend, which needs a real display
(or a virtual one, e.g. Xvfb, if run on a headless Windows/macOS CI box).

Usage:
    python colorize_mesh.py model.glb photo1.jpg photo2.jpg ... \\
        --outdir out --out-name model_colored.glb [--azimuth photo2.jpg=36 ...]

Notes / limitations:
    - This produces PER-VERTEX color, not a UV texture. Quality depends
      on mesh vertex density; very low-poly meshes will look blotchy.
    - Each photo's camera is calibrated independently by silhouette
      matching alone, so there's no guarantee of avoiding a front/back
      "flip" on highly symmetric shapes -- more views, or ones with a
      distinctive asymmetric pose/feature, calibrate more reliably.
      --azimuth PHOTO=DEGREES fixes the azimuth of a photo whose
      calibration landed on the wrong side (check the --debug overlays).
    - Photos should be roughly consistent (same object, plain/uniform
      background so a silhouette can be extracted, similar lighting).
    - Surfaces never visible in any photo are filled by propagating
      color inward from their nearest visible neighbors -- a guess,
      not real information.
"""

import argparse
import os
import pickle
import platform as _platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2

# EGL headless rendering is a Linux/Mesa thing (needed on servers/containers
# with no real display). On Windows/macOS there's a real display available,
# so pyrender's default backend (pyglet) works out of the box -- forcing EGL
# there just breaks it with "Unable to load EGL library". Only opt into EGL
# automatically on Linux, and only if the caller hasn't already set something.
if _platform.system() == "Linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix

type Mask = NDArray[np.bool_]
type Floats = NDArray[np.float64]
type CameraParams = NDArray[np.float64]  # [azimuth_deg, elevation_deg, distance, fov_deg, look_at_height]

UP = np.array([0.0, 1.0, 0.0])
COVERED_WEIGHT = 1e-6  # a vertex with a larger weight got a direct photo sample

# Silhouettes, see extract_silhouette.
ALPHA_FOREGROUND_MIN = 10  # alpha above this is the object
BORDER_PX = 20  # photo border strip whose median colour is the background
BACKGROUND_DISTANCE_MIN = 45  # RGB distance from the background above which a pixel is the object

# Camera calibration, see calibrate_camera_for_image.
MASK_THRESHOLD = 127  # resized 0..255 silhouette: above this is the object
CALIBRATION_RES_PX = 110  # silhouettes are matched at this size
MIN_CAMERA_DISTANCE = 0.3
FOV_RANGE_DEG = (2, 100)  # open interval of acceptable vertical fields of view
INVALID_LOSS = 2.0  # loss of an impossible camera (worse than any real match)
DISTANCE_PENALTY_WEIGHT = 0.02  # weight of the silhouette distance-transform penalty next to 1 - IoU
GRID_AZIMUTHS_DEG = range(0, 360, 10)
GRID_ELEVATIONS_DEG = (0, 10, 20)
GRID_DISTANCES = (3.0, 4.0, 5.0)
GRID_FOVS_DEG = (20, 30, 40)
GRID_LOOK_AT_HEIGHT = 1.0
REFINED_CANDIDATES = 8  # best grid points refined with Nelder-Mead
NELDER_MEAD_OPTIONS = {"xatol": 1e-2, "fatol": 1e-4, "maxiter": 250, "adaptive": True}

# Projection, see project_colors.
MIN_FORWARD_DISTANCE = 0.01  # vertices closer to the camera plane are not sampled
OCCLUSION_TOLERANCE = 0.015  # depth slack, as a share of the camera distance
OVERLAY_RES_PX = 380  # --debug calibration overlays

# Colour is sampled only from SOLID foreground pixels, shrunk by EDGE_ERODE_PX.
# Background-removed PNGs keep a grey RGB (~193,197,206) under transparent
# pixels and soft alpha at the edges; sampling there painted the mesh grey.
# Small calibration errors also shift projections a few px, so the erosion
# keeps edge vertices from reaching the background.
COLOR_ALPHA_MIN = 200
EDGE_ERODE_PX = 12
_color_mask_cache: dict[tuple[str, int], Mask] = {}


# ---------------------------------------------------------------------------
# Stage 1: load + center mesh
# ---------------------------------------------------------------------------
def load_and_center_mesh(path: str) -> trimesh.Trimesh:
    """The mesh with x / z centred and its lowest point at y = 0."""
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as one mesh")
    bounds = mesh.bounds
    center_xz = (bounds[0][[0, 2]] + bounds[1][[0, 2]]) / 2
    mesh.apply_translation([-center_xz[0], -bounds[0][1], -center_xz[1]])
    return mesh


# ---------------------------------------------------------------------------
# Stage 2: silhouette extraction (background subtraction + shadow removal)
# ---------------------------------------------------------------------------
def extract_silhouette(image_path: str) -> Mask:
    """Object mask of a photo: its alpha channel, or (no alpha) the pixels far from the median border colour,
    largest blob only."""
    img = Image.open(image_path)

    if img.mode == "RGBA":
        alpha = np.array(img.getchannel("A"))
        return np.asarray(alpha > ALPHA_FOREGROUND_MIN)

    # fallback for JPGs
    im = np.array(img.convert("RGB")).astype(np.float32)

    border = np.concatenate(
        [
            im[:BORDER_PX].reshape(-1, 3),
            im[-BORDER_PX:].reshape(-1, 3),
            im[:, :BORDER_PX].reshape(-1, 3),
            im[:, -BORDER_PX:].reshape(-1, 3),
        ]
    )

    bg = np.median(border, axis=0)
    dist = np.linalg.norm(im - bg, axis=-1)

    mask = dist > BACKGROUND_DISTANCE_MIN

    labeled, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        mask = labeled == (np.argmax(sizes) + 1)

    return np.asarray(mask)


# ---------------------------------------------------------------------------
# Stage 3: per-image camera calibration via silhouette IoU matching
# ---------------------------------------------------------------------------
def _look_at_pose(eye: Floats, target: Floats, up: Floats = UP) -> Floats:
    """4x4 camera-to-world pose of an OpenGL camera at eye looking at target."""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = r, u, -f, eye
    return pose


def _camera_eye_and_target(az_deg: float, el_deg: float, dist: float, look_h: float) -> tuple[Floats, Floats]:
    """Camera position on a sphere around the vertical axis (azimuth from +z toward +x) and its look-at point."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    horiz = dist * np.cos(el)
    eye = np.array([horiz * np.sin(az), look_h + dist * np.sin(el), horiz * np.cos(az)])
    target = np.array([0, look_h, 0])
    return eye, target


def _white_decimated(mesh: trimesh.Trimesh, decimate_faces: int) -> trimesh.Trimesh:
    """Fast low-poly white copy of the mesh for silhouette renders."""
    decimated = mesh.simplify_quadric_decimation(face_count=decimate_faces)
    decimated.visual = trimesh.visual.ColorVisuals(
        decimated, vertex_colors=np.tile([255, 255, 255, 255], (len(decimated.vertices), 1)).astype(np.uint8)
    )
    return decimated


@dataclass
class _SilhouetteMatch:
    """Renders the decimated mesh from a camera and scores it against the target silhouette (lower is better)."""

    small_mask: Mask
    target_dt: NDArray[Any]  # distance (px) of every pixel to the target silhouette
    scene: Any  # pyrender.Scene
    camera: Any  # pyrender.PerspectiveCamera
    camera_node: Any
    renderer: Any  # pyrender.OffscreenRenderer

    def loss(self, params: Sequence[float]) -> float:
        """(1 - IoU) + DISTANCE_PENALTY_WEIGHT * mean distance of the mismatched pixels; INVALID_LOSS for an
        impossible camera."""
        az_deg, el_deg, dist, fov_deg, look_h = params

        if dist <= MIN_CAMERA_DISTANCE or not (FOV_RANGE_DEG[0] < fov_deg < FOV_RANGE_DEG[1]):
            return INVALID_LOSS

        self.camera.yfov = np.deg2rad(fov_deg)
        eye, target = _camera_eye_and_target(az_deg, el_deg, dist, look_h)
        self.scene.set_pose(self.camera_node, _look_at_pose(eye, target))

        _, depth = self.renderer.render(self.scene)

        pred = depth > 0

        inter = np.logical_and(pred, self.small_mask).sum()
        union = np.logical_or(pred, self.small_mask).sum()
        iou = inter / max(union, 1)

        # Pixels where the rendered model sticks outside the target silhouette
        outside = pred & (~self.small_mask)
        penalty = self.target_dt[outside].mean() if outside.any() else 0.0

        # Pixels where the target silhouette is missing in the render
        missing = self.small_mask & (~pred)

        if missing.any():
            penalty += self.target_dt[missing].mean()

        return float((1.0 - iou) + DISTANCE_PENALTY_WEIGHT * penalty)


def _start_silhouette_match(decimated_mesh: trimesh.Trimesh, target_mask: Mask, calib_res: int) -> _SilhouetteMatch:
    """Scene, camera and renderer for matching one photo's silhouette at calib_res px."""
    small_mask = (
        np.array(
            Image.fromarray((target_mask * 255).astype(np.uint8)).resize(
                (calib_res, calib_res), Image.Resampling.BILINEAR
            )
        )
        > MASK_THRESHOLD
    )

    small_mask_u8 = small_mask.astype(np.uint8) * 255

    target_dt = cv2.distanceTransform(
        255 - small_mask_u8,
        cv2.DIST_L2,
        5,
    )

    pyr_mesh = pyrender.Mesh.from_trimesh(decimated_mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyr_mesh)
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(35), aspectRatio=1.0)
    cam_node = scene.add(cam, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(calib_res, calib_res)
    return _SilhouetteMatch(small_mask, target_dt, scene, cam, cam_node, renderer)


def _grid_candidates(match: _SilhouetteMatch) -> list[tuple[float, CameraParams]]:
    """(loss, params) of every grid camera, best first."""
    candidates = []
    for az in GRID_AZIMUTHS_DEG:
        for el in GRID_ELEVATIONS_DEG:
            for dist in GRID_DISTANCES:
                for fov in GRID_FOVS_DEG:
                    look_h = GRID_LOOK_AT_HEIGHT
                    loss = match.loss((az, el, dist, fov, look_h))
                    candidates.append((loss, np.array([az, el, dist, fov, look_h], dtype=float)))
    candidates.sort(key=lambda x: x[0])
    return candidates


def calibrate_camera_for_image(
    decimated_mesh: trimesh.Trimesh,
    target_mask: Mask,
    calib_res: int = CALIBRATION_RES_PX,
) -> CameraParams:
    """Returns [azimuth_deg, elevation_deg, distance, fov_deg, look_at_height]: a grid search, then Nelder-Mead
    from the REFINED_CANDIDATES best grid points."""
    match = _start_silhouette_match(decimated_mesh, target_mask, calib_res)
    best_loss = 1e9
    best_params = None
    for _, x0 in _grid_candidates(match)[:REFINED_CANDIDATES]:
        res = minimize(match.loss, x0, method="Nelder-Mead", options=NELDER_MEAD_OPTIONS)
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
    match.renderer.delete()
    if best_params is None:
        raise RuntimeError("camera calibration found no candidate")
    return np.asarray(best_params, dtype=np.float64)  # az_deg, el_deg, dist, fov_deg, look_h


def calibrate_all_images(
    mesh: trimesh.Trimesh,
    image_paths: Sequence[str],
    masks: dict[str, Mask],
    decimate_faces: int = 15000,
    azimuths: dict[str, float] | None = None,
) -> dict[str, CameraParams]:
    """Calibrate every photo's camera on a decimated copy. azimuths: {photo file name: degrees} replaces the
    calibrated azimuth of those photos (--azimuth, for a calibration that landed on the wrong side)."""
    azimuths = azimuths or {}
    decimated = _white_decimated(mesh, decimate_faces)
    calibrations = {}
    for path in image_paths:
        t0 = time.time()
        params = calibrate_camera_for_image(decimated, masks[path])
        name = os.path.basename(path)
        if name in azimuths:
            params[0] = azimuths[name]
        calibrations[path] = params
        print(
            f"  calibrated {name}: az={params[0]:.1f} el={params[1]:.1f} "
            f"dist={params[2]:.2f} fov={params[3]:.1f} look_h={params[4]:.2f} "
            f"({time.time() - t0:.1f}s)"
        )

    return calibrations


# ---------------------------------------------------------------------------
# Stage 4-5: project photo colors onto full-res mesh, weighted by visibility
# ---------------------------------------------------------------------------
def color_sampling_mask(image_path: str, fallback_mask: Mask, erode_px: int = EDGE_ERODE_PX) -> Mask:
    """Pixels colour may be sampled from: alpha >= COLOR_ALPHA_MIN (or fallback_mask without alpha), shrunk by
    erode_px. Cached per photo and erosion."""
    key = (image_path, erode_px)
    if key in _color_mask_cache:
        return _color_mask_cache[key]
    img = Image.open(image_path)
    mask = np.array(img.getchannel("A")) >= COLOR_ALPHA_MIN if img.mode == "RGBA" else fallback_mask.copy()
    if erode_px > 0:
        k = 2 * erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.erode(mask.astype(np.uint8), kernel) > 0
    _color_mask_cache[key] = mask
    return mask


@dataclass
class _DepthCamera:
    """Full-resolution mesh scene for occlusion depth buffers (square, depth_res px)."""

    scene: Any  # pyrender.Scene
    camera: Any  # pyrender.PerspectiveCamera
    camera_node: Any
    renderer: Any  # pyrender.OffscreenRenderer
    depth_res: int

    @classmethod
    def start(cls, mesh: trimesh.Trimesh, depth_res: int) -> "_DepthCamera":
        """Grey copy of the mesh in a scene with one camera."""
        render_mesh = mesh.copy()
        render_mesh.visual = trimesh.visual.ColorVisuals(
            render_mesh, vertex_colors=np.tile([200, 200, 200, 255], (len(mesh.vertices), 1)).astype(np.uint8)
        )
        pyr_mesh = pyrender.Mesh.from_trimesh(render_mesh, smooth=False)
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
        scene.add(pyr_mesh)
        cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(30), aspectRatio=1.0)
        cam_node = scene.add(cam, pose=np.eye(4))
        return cls(scene, cam, cam_node, pyrender.OffscreenRenderer(depth_res, depth_res), depth_res)


@dataclass(frozen=True)
class Photos:
    """The reference photos: their paths, object masks and calibrated cameras (both keyed by path)."""

    paths: Sequence[str]
    masks: dict[str, Mask]
    calibrations: dict[str, CameraParams]


@dataclass(frozen=True)
class _ViewSample:
    """One photo's sample per vertex: colour (N, 3), weight (N,) = how face-on (0 = not usable), visible count."""

    colour: NDArray[Any]
    weight: Floats
    visible_count: int


def _bilinear_sample(photo: NDArray[np.float32], px_full: Floats, py_full: Floats) -> NDArray[Any]:
    """RGB of the photo at sub-pixel positions (bilinear)."""
    img_res = photo.shape[0]
    x0 = np.clip(np.floor(px_full).astype(int), 0, img_res - 2)
    y0 = np.clip(np.floor(py_full).astype(int), 0, img_res - 2)
    fx_ = np.clip(px_full - x0, 0, 1)
    fy_ = np.clip(py_full - y0, 0, 1)
    c00, c10 = photo[y0, x0], photo[y0, x0 + 1]
    c01, c11 = photo[y0 + 1, x0], photo[y0 + 1, x0 + 1]
    return np.asarray(
        c00 * (1 - fx_)[:, None] * (1 - fy_)[:, None]
        + c10 * fx_[:, None] * (1 - fy_)[:, None]
        + c01 * (1 - fx_)[:, None] * fy_[:, None]
        + c11 * fx_[:, None] * fy_[:, None]
    )


def _sample_view(
    mesh: trimesh.Trimesh, depth_camera: _DepthCamera, path: str, photos: Photos, erode_px: int
) -> _ViewSample:
    """Colour of every vertex in one photo: visible (not occluded, in frame) vertices get a weight = how face-on
    they are, 0 where the photo's eroded silhouette does not cover them."""
    depth_res = depth_camera.depth_res
    n_verts = len(mesh.vertices)
    az_deg, el_deg, dist, fov_deg, look_h = photos.calibrations[path]
    eye, target = _camera_eye_and_target(az_deg, el_deg, dist, look_h)
    pose = _look_at_pose(eye, target)
    depth_camera.camera.yfov = np.deg2rad(fov_deg)
    depth_camera.scene.set_pose(depth_camera.camera_node, pose)
    _, depth_buf = depth_camera.renderer.render(depth_camera.scene)

    view = np.linalg.inv(pose)
    verts_h = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]
    cam_space = (view @ verts_h.T).T[:, :3]
    vis_forward = -cam_space[:, 2]
    fx = fy = (depth_res / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    cx = cy = depth_res / 2.0
    px = fx * (cam_space[:, 0] / (-cam_space[:, 2])) + cx
    py = cy - fy * (cam_space[:, 1] / (-cam_space[:, 2]))

    in_frame = (
        (vis_forward > MIN_FORWARD_DISTANCE) & (px >= 0) & (px < depth_res - 1) & (py >= 0) & (py < depth_res - 1)
    )
    ix = np.clip(px.astype(int), 0, depth_res - 1)
    iy = np.clip(py.astype(int), 0, depth_res - 1)
    buf_depth = depth_buf[iy, ix]
    occl_tol = OCCLUSION_TOLERANCE * dist
    visible = in_frame & (buf_depth > 0) & (vis_forward <= buf_depth + occl_tol)

    view_dir = eye[None, :] - mesh.vertices
    view_dir = view_dir / (np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-9)
    align = np.clip(np.sum(mesh.vertex_normals.copy() * view_dir, axis=1), 0, 1)

    w = np.zeros(n_verts)
    w[visible] = align[visible]

    photo = np.array(Image.open(path).convert("RGB")).astype(np.float32)
    img_res = photo.shape[0]
    fg_mask = color_sampling_mask(path, photos.masks[path], erode_px)
    px_full = px * (img_res / depth_res)
    py_full = py * (img_res / depth_res)

    xi = np.clip(px_full.astype(int), 0, img_res - 1)
    yi = np.clip(py_full.astype(int), 0, img_res - 1)
    w[~fg_mask[yi, xi]] = 0  # never sample outside the photo's own silhouette

    return _ViewSample(_bilinear_sample(photo, px_full, py_full), w, int(visible.sum()))


def project_colors(
    mesh: trimesh.Trimesh, photos: Photos, depth_res: int = 1536, erode_px: int = EDGE_ERODE_PX
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """For each vertex, gathers a candidate color+weight from every photo that sees it (weight = how face-on,
    0 when occluded, out of frame or outside the photo's eroded silhouette) and keeps the color of the photo with
    the highest weight. Returns (colors (N, 3), weights (N,): 1 = covered, 0 = never seen)."""
    n_verts = len(mesh.vertices)
    depth_camera = _DepthCamera.start(mesh, depth_res)

    per_view_color = []
    per_view_weight = []

    for path in photos.paths:
        t0 = time.time()
        sample = _sample_view(mesh, depth_camera, path, photos, erode_px)
        per_view_color.append(sample.colour)
        per_view_weight.append(sample.weight)
        print(f"  projected {os.path.basename(path)}: {sample.visible_count} visible verts ({time.time() - t0:.1f}s)")

    depth_camera.renderer.delete()

    colours = np.stack(per_view_color, axis=0)  # [V, N, 3]
    weights = np.stack(per_view_weight, axis=0)  # [V, N]

    best_view = np.argmax(weights, axis=0)

    best_weight = weights[best_view, np.arange(n_verts)]

    color_accum = np.zeros((n_verts, 3), dtype=np.float32)

    has_any = best_weight > COVERED_WEIGHT

    color_accum[has_any] = colours[best_view[has_any], np.arange(n_verts)[has_any]]

    weight_accum = has_any.astype(np.float32)

    return color_accum, weight_accum


# ---------------------------------------------------------------------------
# Stage 5b: color approximation -- outlier denoising + optional palette quantization
# ---------------------------------------------------------------------------
MIN_NEIGHBOURS_FOR_OUTLIER = 3  # a vertex is compared with its covered neighbours only when it has this many


def _build_adjacency(mesh: trimesh.Trimesh) -> csr_matrix:
    """Symmetric vertex adjacency (1 per mesh edge)."""
    n = len(mesh.vertices)
    edges = mesh.edges_unique
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    return coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n)).tocsr()


def denoise_outliers(
    colors: Floats,
    has_color: Mask,
    adj: csr_matrix,
    threshold: float = 32.0,
    iters: int = 4,
) -> Floats:
    """Fixes stray specular-highlight vertices (e.g. a few near-white samples
    on what should be solid black leather) by comparing each directly-covered
    vertex to its covered neighbors' mean color and overwriting only the ones
    that stick out by more than `threshold`. Leaves real color boundaries
    alone since those are supported by many neighbors agreeing, not a lone
    outlier vertex."""
    cur = colors.copy()
    cov = has_color.copy()
    for _ in range(iters):
        neighbor_sum = adj.dot(cur * cov[:, None])
        neighbor_count = adj.dot(cov.astype(np.float64))
        has_neighbors = neighbor_count >= MIN_NEIGHBOURS_FOR_OUTLIER
        neighbor_mean = np.zeros_like(cur)
        neighbor_mean[has_neighbors] = neighbor_sum[has_neighbors] / neighbor_count[has_neighbors, None]
        deviation = np.linalg.norm(cur - neighbor_mean, axis=1)
        outlier = cov & has_neighbors & (deviation > threshold)
        if not outlier.any():
            break
        cur[outlier] = neighbor_mean[outlier]
    return cur


def quantize_palette(colors: Floats, palette_size: int, seed: int = 0) -> Floats:
    """Snaps every vertex color to its nearest of `palette_size` K-means
    centroids -- a flat, 'approximate' posterized look that also mops up any
    remaining noise. Set palette_size <= 0 to skip."""
    if palette_size <= 0:
        return colors
    centroids, labels = kmeans2(colors.astype(np.float64), palette_size, minit="++", seed=seed)
    return np.asarray(centroids[labels], dtype=np.float64)


# ---------------------------------------------------------------------------
# Stage 6: fill never-seen vertices via neighbor propagation
# ---------------------------------------------------------------------------
UNSEEN_FALLBACK_RGB = (180, 180, 180)  # when no vertex was seen at all
DEBUG_CHANGE_MIN = 0.5  # --debug: a colour channel change above this counts as touched


def fill_missing_colors(
    mesh: trimesh.Trimesh,
    color_accum: NDArray[Any],
    weight_accum: NDArray[Any],
    adj: csr_matrix | None = None,
    max_iters: int = 25,
) -> NDArray[np.uint8]:
    """Colours (uint8) of every vertex: seen ones keep theirs, unseen ones take the mean of their filled
    neighbours, ring by ring (max_iters rings), the rest the mean seen colour."""
    n = len(mesh.vertices)
    has_color = weight_accum > COVERED_WEIGHT
    colors = np.zeros((n, 3))
    colors[has_color] = color_accum[has_color] / weight_accum[has_color, None]

    adj = adj if adj is not None else _build_adjacency(mesh)

    filled = has_color.copy()
    cur = colors.copy()
    for _ in range(max_iters):
        missing = ~filled
        if not missing.any():
            break
        neighbor_sum = adj.dot(cur * filled[:, None])
        neighbor_count = adj.dot(filled.astype(np.float64))
        can_fill = missing & (neighbor_count > 0)
        if not can_fill.any():
            break
        cur[can_fill] = neighbor_sum[can_fill] / neighbor_count[can_fill, None]
        filled[can_fill] = True

    if (~filled).any():
        fallback = colors[has_color].mean(axis=0) if has_color.any() else np.array(UNSEEN_FALLBACK_RGB)
        cur[~filled] = fallback

    return np.clip(cur, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Debug: intermediate-artifact dumping (--debug)
# ---------------------------------------------------------------------------
def _colored_copy(mesh: trimesh.Trimesh, colors_uint8: NDArray[np.uint8]) -> trimesh.Trimesh:
    """Copy of the mesh with these vertex colours (opaque)."""
    m = mesh.copy()
    alpha = np.full((len(mesh.vertices), 1), 255, dtype=np.uint8)
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.hstack([colors_uint8, alpha]))
    return m


def render_turnaround(mesh: trimesh.Trimesh, out_path: str, n_views: int = 4, res: int = 450) -> None:
    """Renders an n-view turnaround strip of a (colored) mesh with simple
    two-light studio lighting. Used for every debug snapshot so different
    stages/runs are directly visually comparable."""
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[214, 216, 222, 255], ambient_light=[0.55, 0.55, 0.58])
    scene.add(pyr_mesh)
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(28), aspectRatio=1.0)
    cam_node = scene.add(cam, pose=np.eye(4))

    bounds = mesh.bounds
    look_h = (bounds[0][1] + bounds[1][1]) / 2
    dist = max(bounds[1] - bounds[0]) * 2.2
    renderer = pyrender.OffscreenRenderer(res, res)
    imgs = []
    for i in range(n_views):
        az = np.deg2rad(360 * i / n_views)
        eye = np.array([dist * np.sin(az), look_h + dist * 0.15, dist * np.cos(az)])
        target = np.array([0, look_h, 0])
        pose = _look_at_pose(eye, target)
        scene.set_pose(cam_node, pose)
        key = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
        key_node = scene.add(key, pose=_look_at_pose(eye + np.array([1.5, 1.5, 1.5]), target))
        fill = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.1)
        fill_node = scene.add(fill, pose=_look_at_pose(-eye + np.array([0, 1, 0]), target))
        color, _ = renderer.render(scene)
        scene.remove_node(key_node)
        scene.remove_node(fill_node)
        imgs.append(color)
    Image.fromarray(np.concatenate(imgs, axis=1)).save(out_path)
    renderer.delete()


def save_debug_masks(masks: dict[str, Mask], debug_dir: str) -> None:
    """mask_<photo>.png per silhouette."""
    os.makedirs(debug_dir, exist_ok=True)
    for path, mask in masks.items():
        name = os.path.splitext(os.path.basename(path))[0]
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(debug_dir, f"mask_{name}.png"))


def save_debug_calibration_overlays(
    mesh: trimesh.Trimesh,
    image_paths: Sequence[str],
    calibrations: dict[str, CameraParams],
    decimate_faces: int,
    debug_dir: str,
) -> None:
    """For each photo, renders the calibrated camera's predicted silhouette
    in red on top of the actual photo. A good calibration hugs the real
    outline; a bad one (wrong pose/FOV/azimuth) will show an obviously
    offset or mismatched red blob -- this is the single most useful debug
    image when colors end up on the wrong part of the model."""
    res = OVERLAY_RES_PX
    os.makedirs(debug_dir, exist_ok=True)
    decimated = _white_decimated(mesh, decimate_faces)
    pyr_mesh = pyrender.Mesh.from_trimesh(decimated, smooth=False)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
    scene.add(pyr_mesh)
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(35), aspectRatio=1.0)
    cam_node = scene.add(cam, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(res, res)

    for path in image_paths:
        az_deg, el_deg, dist, fov_deg, look_h = calibrations[path]
        eye, target = _camera_eye_and_target(az_deg, el_deg, dist, look_h)
        cam.yfov = np.deg2rad(fov_deg)
        scene.set_pose(cam_node, _look_at_pose(eye, target))
        _, depth = renderer.render(scene)
        pred_mask = depth > 0

        real = np.array(Image.open(path).convert("RGB").resize((res, res)))
        overlay = real.copy()
        overlay[pred_mask] = (overlay[pred_mask] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
        name = os.path.splitext(os.path.basename(path))[0]
        Image.fromarray(overlay).save(os.path.join(debug_dir, f"calib_overlay_{name}.png"))
    renderer.delete()


def save_debug_coverage_map(mesh: trimesh.Trimesh, has_color: Mask, debug_dir: str) -> None:
    """Green = vertex got a direct photo sample. Red = never seen by any
    camera, filled in by neighbor propagation. Shrinking the red area is
    the clearest sign that more/better-angled photos are helping."""
    os.makedirs(debug_dir, exist_ok=True)
    colors = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
    colors[has_color] = [60, 200, 90]
    colors[~has_color] = [220, 60, 60]
    m = _colored_copy(mesh, colors)
    render_turnaround(m, os.path.join(debug_dir, "coverage_map_turnaround.png"))
    covered = has_color.sum()
    print(
        f"  [debug] coverage: {covered}/{len(mesh.vertices)} ({100 * covered / len(mesh.vertices):.1f}%) "
        f"-> {debug_dir}/coverage_map_turnaround.png"
    )


def save_debug_stage(
    mesh: trimesh.Trimesh,
    colors: NDArray[Any],
    stage_name: str,
    debug_dir: str,
    uncovered_mask: Mask | None = None,
) -> None:
    """Dumps a turnaround render for one pipeline stage (raw projection,
    post-denoise, post-fill, post-quantize, ...). Run this after each
    parameter change and diff the same stage_name across debug folders to
    see exactly where a tweak changed something."""
    os.makedirs(debug_dir, exist_ok=True)
    c = np.clip(colors, 0, 255).astype(np.uint8).copy()
    if uncovered_mask is not None:
        c[uncovered_mask] = [160, 160, 160]  # neutral gray = "no data yet at this stage"
    m = _colored_copy(mesh, c)
    out_path = os.path.join(debug_dir, f"{stage_name}_turnaround.png")
    render_turnaround(m, out_path)
    print(f"  [debug] {stage_name} -> {out_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColorizeSettings:
    """Settings of one colorize run (see the command-line help): output file name (None = <model>_colored.glb),
    calibration mesh size, depth-buffer resolution, outlier clean-up, palette size (0 = off), whether to use the
    calibration cache, debug images, the edge erosion of the colour-sampling mask and fixed azimuths
    ({photo file name: degrees})."""

    out_name: str | None = None
    decimate_faces: int = 15000
    depth_res: int = 1536
    outlier_threshold: float = 32.0
    outlier_iters: int = 4
    palette_size: int = 0
    cache_calibration: bool = True
    debug: bool = False
    edge_erode_px: int = EDGE_ERODE_PX
    azimuths: dict[str, float] = field(default_factory=dict)


def _expand_image_paths(image_paths: Sequence[str]) -> list[str]:
    """A single folder argument means every file in it."""
    if len(image_paths) == 1 and os.path.isdir(image_paths[0]):
        folder = image_paths[0]
        return [os.path.join(folder, image) for image in os.listdir(folder)]
    return list(image_paths)


def _calibrations(
    mesh: trimesh.Trimesh,
    model_path: str,
    image_paths: Sequence[str],
    masks: dict[str, Mask],
    cache: tuple[str, ColorizeSettings],
) -> dict[str, CameraParams]:
    """Camera per photo: from the calibration cache (same model) or calibrated now (and cached). cache = (cache
    file path, settings)."""
    calib_cache_path, settings = cache
    cached_calibrations: dict[str, CameraParams] = {}
    if settings.cache_calibration and os.path.exists(calib_cache_path):
        with open(calib_cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("model_path") == model_path:
            cached_calibrations = cached.get("calibrations", {})

    calibrations = {p: cached_calibrations[p] for p in image_paths if p in cached_calibrations}
    to_calibrate = [p for p in image_paths if p not in calibrations]
    if to_calibrate:
        print(f"Calibrating cameras for {len(to_calibrate)} image(s) (slow step)...")
        new_calibs = calibrate_all_images(mesh, to_calibrate, masks, settings.decimate_faces, settings.azimuths)
        calibrations.update(new_calibs)
        if settings.cache_calibration:
            cached_calibrations.update(new_calibs)
            with open(calib_cache_path, "wb") as f:
                pickle.dump({"model_path": model_path, "calibrations": cached_calibrations}, f)
    else:
        print("Reusing cached camera calibration for all images.")
    return calibrations


def _finish_colours(
    mesh: trimesh.Trimesh,
    projected: tuple[NDArray[Any], NDArray[Any]],
    settings: ColorizeSettings,
    debug_dir: str | None,
) -> NDArray[np.uint8]:
    """Projected (colours, weights) -> final vertex colours: denoise, fill unseen vertices, optional palette."""
    color_accum, weight_accum = projected
    has_color = weight_accum > COVERED_WEIGHT
    coverage = has_color.sum()
    print(f"  {coverage}/{len(mesh.vertices)} vertices directly covered ({100 * coverage / len(mesh.vertices):.1f}%)")

    adj = _build_adjacency(mesh)

    raw_colors = np.zeros_like(color_accum)
    raw_colors[has_color] = color_accum[has_color] / weight_accum[has_color, None]

    if debug_dir:
        save_debug_coverage_map(mesh, has_color, debug_dir)
        save_debug_stage(mesh, raw_colors, "01_raw_projection", debug_dir, uncovered_mask=~has_color)

    print("Denoising outlier vertices (specular highlights, stray samples)...")
    denoised = denoise_outliers(
        raw_colors, has_color, adj, threshold=settings.outlier_threshold, iters=settings.outlier_iters
    )
    denoised_accum = denoised * has_color[:, None]
    denoised_weight = has_color.astype(np.float64)
    if debug_dir:
        n_changed = int(np.any(np.abs(denoised - raw_colors) > DEBUG_CHANGE_MIN, axis=1)[has_color].sum())
        print(f"  [debug] outlier correction touched {n_changed}/{coverage} covered vertices")
        save_debug_stage(mesh, denoised, "02_denoised", debug_dir, uncovered_mask=~has_color)

    print("Filling uncovered vertices...")
    final_colors = fill_missing_colors(mesh, denoised_accum, denoised_weight, adj=adj)
    if debug_dir:
        save_debug_stage(mesh, final_colors.astype(np.float64), "03_filled", debug_dir)

    palette_size = settings.palette_size
    if palette_size > 0:
        print(f"Quantizing to a {palette_size}-color palette...")
        final_colors = np.clip(quantize_palette(final_colors.astype(np.float64), palette_size), 0, 255).astype(np.uint8)
        if debug_dir:
            save_debug_stage(mesh, final_colors.astype(np.float64), "04_quantized", debug_dir)
    return final_colors


def colorize_mesh(
    model_path: str, image_paths: Sequence[str], outdir: str, settings: ColorizeSettings | None = None
) -> str:
    """Colour the mesh at model_path from the photos (or one folder of photos); writes the vertex-coloured GLB to
    outdir and returns its path. See ColorizeSettings."""
    settings = settings if settings is not None else ColorizeSettings()
    os.makedirs(outdir, exist_ok=True)
    out_name = settings.out_name or (os.path.splitext(os.path.basename(model_path))[0] + "_colored.glb")
    out_path = os.path.join(outdir, out_name)
    calib_cache_path = os.path.join(outdir, "_calibration_cache.pkl")
    debug_dir = os.path.join(outdir, "debug") if settings.debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[debug] artifacts will be saved under {debug_dir}/")

    print("Loading and centering mesh...")
    mesh = load_and_center_mesh(model_path)
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    print("Extracting silhouettes...")
    image_paths = _expand_image_paths(image_paths)
    masks = {p: extract_silhouette(p) for p in image_paths}
    if debug_dir:
        save_debug_masks(masks, os.path.join(debug_dir, "masks"))
        print(f"  [debug] silhouette masks -> {debug_dir}/masks/")

    calibrations = _calibrations(mesh, model_path, image_paths, masks, (calib_cache_path, settings))

    if debug_dir:
        save_debug_calibration_overlays(
            mesh, image_paths, calibrations, settings.decimate_faces, os.path.join(debug_dir, "calibration")
        )
        print(
            f"  [debug] calibration overlays -> {debug_dir}/calibration/ "
            f"(red = predicted silhouette; should hug the real outline)"
        )

    print("Projecting photo colors onto mesh (best view per vertex)...")
    photos = Photos(image_paths, masks, calibrations)
    projected = project_colors(mesh, photos, settings.depth_res, settings.edge_erode_px)
    final_colors = _finish_colours(mesh, projected, settings, debug_dir)

    alpha = np.full((len(mesh.vertices), 1), 255, dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.hstack([final_colors, alpha]))
    mesh.export(out_path)
    print(f"Exported: {out_path}")

    if debug_dir:
        render_turnaround(mesh, os.path.join(debug_dir, "05_final_result_turnaround.png"))
        print(f"  [debug] final result -> {debug_dir}/05_final_result_turnaround.png")

    return out_path


def _azimuth(text: str) -> tuple[str, float]:
    """argparse type for --azimuth: 'PHOTO=DEGREES' -> (photo file name, degrees)."""
    name, separator, degrees = text.rpartition("=")
    try:
        value = float(degrees)
    except ValueError:
        value = None
    if not separator or not name or value is None:
        raise argparse.ArgumentTypeError(f"--azimuth must look like photo-20.png=36 (PHOTO=DEGREES), got {text!r}")
    return os.path.basename(name), value


def build_argument_parser() -> argparse.ArgumentParser:
    """The command-line options."""
    parser = argparse.ArgumentParser(description="Color an untextured mesh using reference photos.")
    parser.add_argument("model", help="Path to the untextured mesh (.glb, .gltf, .obj, ...)")
    parser.add_argument("images", nargs="+", help="Reference photos of the same object, plain background")
    parser.add_argument("--outdir", default="colorize_output")
    parser.add_argument("--out-name", default=None, help="Output filename (default: <model>_colored.glb)")
    parser.add_argument(
        "--decimate-faces", type=int, default=15000, help="Face count for the fast calibration mesh copy"
    )
    parser.add_argument(
        "--depth-res", type=int, default=1536, help="Resolution of the occlusion depth buffer during final projection"
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=32.0,
        help="Color-distance (0-441) from local neighbor average above which a "
        "vertex is treated as a stray outlier (e.g. a specular highlight) "
        "and overwritten. Lower = more aggressive cleanup.",
    )
    parser.add_argument("--outlier-iters", type=int, default=4, help="Number of outlier-correction passes.")
    parser.add_argument(
        "--palette-size",
        type=int,
        default=0,
        help="If > 0, snap all colors to this many K-means palette colors for a "
        "flat/posterized 'approximate' look. 0 disables (default).",
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignore/overwrite cached camera calibration")
    parser.add_argument(
        "--edge-erode-px",
        type=int,
        default=EDGE_ERODE_PX,
        help="Shrink each photo's foreground by this many px before sampling "
        "colour (avoids grey background bleeding onto the mesh).",
    )
    parser.add_argument(
        "--azimuth",
        type=_azimuth,
        action="append",
        default=[],
        metavar="PHOTO=DEGREES",
        help="Fix the camera azimuth of one photo (file name, e.g. photo-20.png=36) after calibration, when the "
        "--debug overlay shows it on the wrong side. Repeatable. Only applies to photos calibrated in this run "
        "(use --no-cache to recalibrate cached ones).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dump intermediate artifacts (silhouette masks, per-image calibration "
        "overlays, a coverage map, and a turnaround render after each pipeline "
        "stage) into <outdir>/debug/ so you can visually diff runs after "
        "changing parameters.",
    )
    return parser


def main() -> None:
    """Colour a mesh from the command line."""
    args = build_argument_parser().parse_args()
    settings = ColorizeSettings(
        out_name=args.out_name,
        decimate_faces=args.decimate_faces,
        depth_res=args.depth_res,
        outlier_threshold=args.outlier_threshold,
        outlier_iters=args.outlier_iters,
        palette_size=args.palette_size,
        cache_calibration=not args.no_cache,
        debug=args.debug,
        edge_erode_px=args.edge_erode_px,
        azimuths=dict(args.azimuth),
    )
    colorize_mesh(args.model, args.images, args.outdir, settings)


if __name__ == "__main__":
    main()
