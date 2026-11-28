"""Step 4: colour the mesh by projecting the pictures onto it (numpy only: no GPU, no nvdiffrast).

Each picture (front / left / back / right) is an orthographic view of the character. The mesh is turned to the front
picture (the one of 4 horizontal directions whose outline matches it best), each vertex takes the colour under it in
every picture where it is visible (depth test) and faces the camera, the pictures are blended by how squarely the
surface faces them, and vertices no picture sees get their neighbours' colours. Lighting in the pictures is kept.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse

from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.type_aliases import BoolArray, FloatArray, IntArray, JsonDict
from image_to_bricks.images_to_model.view_geometry import (
    DIRECTION_VECTORS,
    DIRECTIONS,
    assign_view_directions,
    compute_mask_iou,
    project_vertices,
    split_rgba,
)
from image_to_bricks.images_to_model.views import FRONT_VIEW, VIEW_NAMES

FRONT_DIRECTION = "+z"  # TRELLIS.2 / glTF front
FRONT_PREFERENCE = 0.01  # outline-IoU bonus for FRONT_DIRECTION when the front and back outlines tie
DEPTH_TOLERANCE = 0.01  # of the mesh height
FACING_POWER = 2.0  # weight of a picture = (cosine between surface normal and camera) ** FACING_POWER
MAX_FILL_ROUNDS = 200  # rings of neighbours that unseen colours spread over
NEUTRAL_GREY = 0.5  # colour when no vertex was seen at all
TINY = 1e-12  # guards divisions by zero lengths / weights
MIN_SEEN_WEIGHT = 1e-6  # a vertex with less total picture weight counts as unseen
MIN_SCREEN_EXTENT = 1e-9
SEEN_SHARE_DIGITS = 4


class ColouredMesh:
    """Mesh with one RGB colour (0-1) per vertex; run_pipeline re-places its vertices, export writes a GLB with
    vertex colours."""

    def __init__(self, vertices: FloatArray, faces: IntArray, colours: FloatArray) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.colours = np.asarray(colours, dtype=np.float64)

    def export(self, path: str) -> None:
        """Write a GLB with sRGB vertex colours (needs trimesh)."""
        import trimesh  # noqa: PLC0415 (only needed when a model is written)

        opaque = np.full(len(self.colours), 255)
        rgba = np.column_stack([np.clip(self.colours, 0, 1) * 255, opaque]).astype(np.uint8)
        trimesh.Trimesh(self.vertices, self.faces, vertex_colors=rgba, process=False).export(path)


@dataclass(frozen=True)
class PictureFit:
    """How mesh coordinates map to a picture: the mesh's projected box is scaled to the character's height and
    centred on it. pixel x = (screen x - mesh_x) * scale + image_x; pixel y = image_y - (screen y - mesh_y) * scale."""

    scale: float
    mesh_x: float
    mesh_y: float
    image_x: float
    image_y: float


class ProjectedVertices(NamedTuple):
    """Every vertex in picture pixels: x, y and depth (larger = closer to the camera)."""

    pixel_x: FloatArray
    pixel_y: FloatArray
    depth: FloatArray


def compute_vertex_normals(mesh: Mesh) -> FloatArray:
    """Unit normal per vertex [n, 3]: the area-weighted sum of its faces' normals."""
    corners = mesh.vertices[mesh.faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(mesh.vertices)
    for corner in range(3):
        np.add.at(normals, mesh.faces[:, corner], face_normals)
    return np.asarray(normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), TINY), dtype=np.float64)


def compute_fit(vertices: FloatArray, direction: str, mask: BoolArray) -> PictureFit:
    """Fit of the mesh seen from direction onto the character mask [rows, columns] of a picture."""
    screen_x, screen_y, _ = project_vertices(vertices, direction)
    rows, cols = np.nonzero(mask)
    return PictureFit(
        scale=float((rows.max() + 1 - rows.min()) / max(float(np.ptp(screen_y)), MIN_SCREEN_EXTENT)),
        mesh_x=float((screen_x.min() + screen_x.max()) / 2),
        mesh_y=float((screen_y.min() + screen_y.max()) / 2),
        image_x=float((cols.min() + cols.max() + 1) / 2),
        image_y=float((rows.min() + rows.max() + 1) / 2),
    )


def fit_projection(vertices: FloatArray, direction: str, mask: BoolArray) -> ProjectedVertices:
    """Pixel coordinates of every vertex in a picture seen from direction."""
    screen_x, screen_y, depth = project_vertices(vertices, direction)
    fit = compute_fit(vertices, direction, mask)
    pixel_x = (screen_x - fit.mesh_x) * fit.scale + fit.image_x
    pixel_y = fit.image_y - (screen_y - fit.mesh_y) * fit.scale
    return ProjectedVertices(pixel_x, pixel_y, depth)


def rasterize_depth(mesh: Mesh, projected: ProjectedVertices, shape: tuple[int, int]) -> FloatArray:
    """Nearest depth per pixel [rows, columns] (larger = closer to the camera); -inf where the mesh does not cover."""
    height, width = shape
    zbuffer = np.full(shape, -np.inf)
    for face in mesh.faces:
        xs, ys = projected.pixel_x[face], projected.pixel_y[face]
        x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), width - 1)
        y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), height - 1)
        if x0 > x1 or y0 > y1:
            continue
        det = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(det) < TINY:
            continue
        grid_x, grid_y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w1 = ((grid_x - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (grid_y - ys[0])) / det
        w2 = ((xs[1] - xs[0]) * (grid_y - ys[0]) - (grid_x - xs[0]) * (ys[1] - ys[0])) / det
        inside = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
        if not inside.any():
            continue
        depth = projected.depth
        z = (1 - w1 - w2) * depth[face[0]] + w1 * depth[face[1]] + w2 * depth[face[2]]
        window = zbuffer[y0 : y1 + 1, x0 : x1 + 1]
        nearer = inside & (z > window)
        window[nearer] = z[nearer]
    return zbuffer


def draw_outline(mesh: Mesh, direction: str, mask: BoolArray) -> BoolArray:
    """Silhouette of the mesh seen from direction, fitted onto the picture of mask [rows, columns]."""
    projected = fit_projection(mesh.vertices, direction, mask)
    canvas = Image.new("1", (mask.shape[1], mask.shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for face in mesh.faces:
        draw.polygon([(projected.pixel_x[i], projected.pixel_y[i]) for i in face], fill=1)
    return np.asarray(canvas, dtype=bool)


def measure_outline_iou(mesh: Mesh, direction: str, mask: BoolArray) -> float:
    """How well the mesh outline seen from direction matches the character mask of a picture (IoU 0-1)."""
    return compute_mask_iou(draw_outline(mesh, direction, mask), mask)


def choose_front_direction(mesh: Mesh, front_mask: BoolArray) -> str:
    """Which of the 4 horizontal directions shows the front picture (best outline match, FRONT_DIRECTION preferred on
    ties)."""
    scores = {
        direction: measure_outline_iou(mesh, direction, front_mask)
        + (FRONT_PREFERENCE if direction == FRONT_DIRECTION else 0)
        for direction in DIRECTIONS
    }
    return max(scores, key=lambda direction: scores[direction])


def sample_view(
    mesh: Mesh, normals: FloatArray, picture: tuple[FloatArray, BoolArray], direction: str
) -> tuple[FloatArray, FloatArray]:
    """Colour [n, 3] and weight [n] per vertex from one picture (RGB, mask) seen from direction; weight 0 where the
    vertex is hidden, outside the character or facing away."""
    rgb, mask = picture
    projected = fit_projection(mesh.vertices, direction, mask)
    zbuffer = rasterize_depth(mesh, projected, (mask.shape[0], mask.shape[1]))
    cols = np.clip(np.floor(projected.pixel_x).astype(int), 0, mask.shape[1] - 1)
    rows = np.clip(np.floor(projected.pixel_y).astype(int), 0, mask.shape[0] - 1)
    tolerance = DEPTH_TOLERANCE * float(np.ptp(mesh.vertices[:, 1]))  # depth is in mesh units
    visible = (projected.depth >= zbuffer[rows, cols] - tolerance) & mask[rows, cols]
    facing = np.clip(normals @ np.array(DIRECTION_VECTORS[direction], dtype=float), 0, 1) ** FACING_POWER
    weights = np.where(visible, facing, 0.0)
    return rgb[rows, cols], weights


def fill_unseen(mesh: Mesh, colours: FloatArray, known: BoolArray) -> FloatArray:
    """Give vertices no picture saw the mean colour of their already coloured neighbours, ring by ring; what is left
    after MAX_FILL_ROUNDS gets the mean seen colour."""
    count = len(mesh.vertices)
    edges = np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    pairs = (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])
    adjacency = sparse.coo_matrix((np.ones(len(edges) * 2), pairs), shape=(count, count)).tocsr()
    adjacency.data[:] = 1.0
    colours, known = colours.copy(), known.copy()
    for _ in range(MAX_FILL_ROUNDS):
        if known.all():
            break
        sums = adjacency @ (colours * known[:, None])
        counts = adjacency @ known.astype(float)
        newly = ~known & (counts > 0)
        if not newly.any():
            break
        colours[newly] = sums[newly] / counts[newly, None]
        known = known | newly
    colours[~known] = colours[known].mean(axis=0) if known.any() else NEUTRAL_GREY
    return colours


def project_colours(mesh: Mesh, views: dict[str, Image.Image]) -> tuple[FloatArray, JsonDict]:
    """Per-vertex RGB (0-1) [n, 3] from the named RGBA pictures (front required). Returns (colours, info: front
    direction, views used, share of vertices seen)."""
    pictures = {name: split_rgba(image) for name, image in views.items()}
    front = choose_front_direction(mesh, pictures[FRONT_VIEW][1])
    assignment = assign_view_directions(DIRECTIONS.index(front), 1)
    normals = compute_vertex_normals(mesh)
    total, weight_sum = np.zeros((len(mesh.vertices), 3)), np.zeros(len(mesh.vertices))
    for name, picture in pictures.items():
        colours, weights = sample_view(mesh, normals, picture, assignment[name])
        total += colours * weights[:, None]
        weight_sum += weights
    known = weight_sum > MIN_SEEN_WEIGHT
    colours = np.where(known[:, None], total / np.maximum(weight_sum, TINY)[:, None], 0.0)
    filled = fill_unseen(mesh, colours, known)
    info = {
        "front_direction": front,
        "views": sorted(pictures),
        "vertices_seen": round(float(known.mean()), SEEN_SHARE_DIGITS),
    }
    return filled, info


class ProjectionPainter:
    """Colours a mesh from the named pictures; keeps the info of the last run (front direction, views, seen share)."""

    view_names: tuple[str, ...] = VIEW_NAMES

    def __init__(self) -> None:
        self.last_info: JsonDict | None = None

    def paint(self, mesh: Mesh, views: dict[str, Image.Image]) -> ColouredMesh:
        """The mesh with projected vertex colours."""
        colours, self.last_info = project_colours(mesh, views)
        return ColouredMesh(mesh.vertices, mesh.faces, colours)
