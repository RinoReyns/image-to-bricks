"""Step 3: clean the generated mesh (weld, drop debris, rebuild broken surfaces as a solid, reduce, repair, orient)
and report every count.

Pure numpy / scipy so every step is unit-testable; the face reducer is injected (fast-simplification in production,
a fake in tests).
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.type_aliases import BoolArray, FloatArray, IntArray, JsonDict

MERGE_DIGITS = 8  # vertices closer than 1e-8 (after rounding) are welded
SMALL_PIECE_FRACTION = 0.005  # pieces with fewer faces than this share of all faces are debris
MAX_REDUCE_ATTEMPTS = 5
REDUCE_TARGET_SHRINK_PER_ATTEMPT = 0.995  # each new attempt aims 0.5 % lower (the reducer may overshoot)
MAX_REPAIR_ROUNDS = 3
NONMANIFOLD_REBUILD_FRACTION = 1e-4  # of all edges: more edges used by 3+ faces -> rebuild as a solid
NONMANIFOLD_REBUILD_MINIMUM = 10
SOLID_RESOLUTION = 256  # voxels along the longest side
SOLID_CLOSING = 4  # voxels: seals the 3-4 voxel leaks in raw TRELLIS.2 surfaces
SOLID_SMOOTHING = 1.0  # Gaussian sigma, voxels
SOLID_OPEN_DISTANCE = 3.0  # voxels from any filled interior: farther = open surface, kept as a wall
GRID_MARGIN_BEYOND_CLOSING = 4  # voxels of empty border around the mesh, on top of the closing distance
SURFACE_SAMPLE_SPACING = 0.5  # voxels between surface samples (dense enough to mark every voxel crossed)
DISTANCE_CLAMP = 3.0  # voxels: the signed distance is exact only near the surface, clamped beyond
NEAR_SURFACE_BAND = 2  # voxels around the surface where the exact distance is computed
MIN_WALL_DISTANCE = 1.0  # voxels: open walls keep at least this depth, to survive the smoothing
STAGED_REDUCTION_FACTOR = 3  # like TRELLIS.2's own export: reduce to 3x the target, repair, then to the target
DEGENERATE_AREA_RATIO = 1e-14  # of the squared bounding-box diagonal
FACES_PER_MANIFOLD_EDGE = 2  # a closed surface uses every edge exactly twice
TRIANGLE_CORNERS = 3

Simplifier = Callable[[FloatArray, IntArray, float], tuple[ArrayLike, ArrayLike]]
"""simplify(vertices, faces, target_reduction 0-1) -> (vertices, faces) of the reduced mesh."""


@dataclass(frozen=True)
class SolidRebuildSettings:
    """How a broken surface is rebuilt as the boundary of a solid (voxel units). keep_open: keep open surfaces (no
    filled interior within open_distance) as thin walls."""

    resolution: int = SOLID_RESOLUTION
    closing: int = SOLID_CLOSING
    sigma: float = SOLID_SMOOTHING
    keep_open: bool = True
    open_distance: float = SOLID_OPEN_DISTANCE


@dataclass(frozen=True)
class DistanceGrid:
    """Signed distance field [x, y, z] (voxel units, negative inside); voxel (i, j, k) has its centre at
    origin + (index + 0.5) * voxel_size."""

    field: FloatArray
    origin: FloatArray
    voxel_size: float


def list_directed_edges(faces: IntArray) -> IntArray:
    """(3m, 2) directed edges; edge k belongs to face k % m."""
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])


def index_undirected_edges(faces: IntArray) -> tuple[IntArray, IntArray]:
    """(undirected edge id per directed edge, how many faces use each unique undirected edge)."""
    undirected = np.sort(list_directed_edges(faces), axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    return inverse.reshape(-1), counts


def remove_unused_vertices(mesh: Mesh) -> Mesh:
    """The mesh without vertices no face uses (faces re-indexed)."""
    if len(mesh.faces) == 0:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    used, inverse = np.unique(mesh.faces, return_inverse=True)
    return Mesh(mesh.vertices[used], inverse.reshape(mesh.faces.shape).astype(np.int64))


def merge_vertices(mesh: Mesh) -> Mesh:
    """Weld vertices with equal positions (rounded to MERGE_DIGITS decimals)."""
    keys = np.round(mesh.vertices, MERGE_DIGITS) + 0.0
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return remove_unused_vertices(Mesh(mesh.vertices[first], inverse.reshape(-1)[mesh.faces]))


def compute_face_areas(mesh: Mesh) -> FloatArray:
    """Area of every face [m]."""
    corners = mesh.vertices[mesh.faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return np.asarray(0.5 * np.linalg.norm(cross, axis=1), dtype=np.float64)


def remove_degenerate_faces(mesh: Mesh) -> tuple[Mesh, int]:
    """Drop faces with a repeated vertex or (near) zero area. Returns (mesh, faces removed)."""
    faces = mesh.faces
    repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    span = np.ptp(mesh.vertices, axis=0) if len(mesh.vertices) else np.zeros(3)
    tiny = compute_face_areas(mesh) <= DEGENERATE_AREA_RATIO * float(np.dot(span, span))
    keep = ~(repeated | tiny)
    return remove_unused_vertices(Mesh(mesh.vertices, faces[keep])), int((~keep).sum())


def remove_duplicate_faces(mesh: Mesh) -> tuple[Mesh, int]:
    """Drop faces using the same three vertices as an earlier face. Returns (mesh, faces removed)."""
    _, first = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_index=True)
    keep = np.sort(first)
    return remove_unused_vertices(Mesh(mesh.vertices, mesh.faces[keep])), len(mesh.faces) - len(keep)


def label_pieces(faces: IntArray) -> IntArray:
    """Piece label per face; faces are connected when they share an edge."""
    face_count = len(faces)
    if face_count == 0:
        return np.zeros(0, dtype=np.int64)
    edge_ids, counts = index_undirected_edges(faces)
    owners = np.tile(np.arange(face_count), 3)
    size = face_count + len(counts)
    graph = sparse.coo_matrix((np.ones(len(owners)), (owners, face_count + edge_ids)), shape=(size, size))
    _, labels = csgraph.connected_components(graph, directed=False)
    _, compact = np.unique(labels[:face_count], return_inverse=True)
    return np.asarray(compact.reshape(-1), dtype=np.int64)


def count_pieces(faces: IntArray) -> int:
    """Number of edge-connected pieces."""
    labels = label_pieces(faces)
    return int(labels.max()) + 1 if len(labels) else 0


def remove_small_pieces(mesh: Mesh, fraction: float = SMALL_PIECE_FRACTION) -> tuple[Mesh, int, int]:
    """Drop pieces with fewer than fraction * all faces (the largest piece is always kept).
    Returns (mesh, pieces removed, faces removed)."""
    if len(mesh.faces) == 0:
        return mesh, 0, 0
    labels = label_pieces(mesh.faces)
    sizes = np.bincount(labels)
    small = sizes < fraction * len(mesh.faces)
    small[int(np.argmax(sizes))] = False
    keep = ~small[labels]
    cleaned = remove_unused_vertices(Mesh(mesh.vertices, mesh.faces[keep]))
    return cleaned, int(small.sum()), int((~keep).sum())


def needs_solid_rebuild(mesh: Mesh, threshold: float = NONMANIFOLD_REBUILD_FRACTION) -> tuple[bool, int, int]:
    """True when many edges are shared by more than two faces (raw TRELLIS.2 meshes: ~0.4 % of edges).
    Returns (decision, non-manifold edges, boundary edges)."""
    _, counts = index_undirected_edges(mesh.faces)
    nonmanifold, boundary = int((counts > FACES_PER_MANIFOLD_EDGE).sum()), int((counts == 1).sum())
    return nonmanifold > max(NONMANIFOLD_REBUILD_MINIMUM, threshold * len(counts)), nonmanifold, boundary


def sample_triangle_points(mesh: Mesh, spacing: float) -> FloatArray:
    """Points on the surface [k, 3]: vertices, centroids and edge midpoints, plus barycentric samples on triangles
    longer than spacing."""
    corners = mesh.vertices[mesh.faces]
    points = [mesh.vertices, corners.mean(axis=1), (corners + np.roll(corners, -1, axis=1)).reshape(-1, 3) / 2]
    longest = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2).max(axis=1)
    steps = np.ceil(longest / spacing).astype(int)
    for count in np.unique(steps[steps > 1]):
        chosen = corners[steps == count]
        grid = np.array([(i, j) for i in range(count + 1) for j in range(count + 1 - i)], dtype=float) / count
        weights = np.column_stack([1 - grid.sum(axis=1), grid])
        points.append(np.einsum("kc,tcd->tkd", weights, chosen).reshape(-1, 3))
    return np.concatenate(points)


def fill_enclosed_voxels(surface: BoolArray, closing: int) -> BoolArray:
    """Voxels the surface encloses (plus the surface), after closing gaps of up to `closing` voxels."""
    ball = ndimage.generate_binary_structure(3, 1)
    if closing <= 0:  # iterations=0 would mean "repeat until nothing changes" in scipy
        return np.asarray(ndimage.binary_fill_holes(surface), dtype=bool)
    solid = ndimage.binary_fill_holes(ndimage.binary_dilation(surface, ball, iterations=closing))
    return np.asarray(ndimage.binary_erosion(solid, ball, iterations=closing, border_value=0) | surface, dtype=bool)


def compute_signed_distance_grid(mesh: Mesh, settings: SolidRebuildSettings) -> DistanceGrid:
    """Signed distance (voxel units, negative inside, clamped to +-DISTANCE_CLAMP away from the surface):
    1. mark the voxels the surface passes through, close gaps of up to settings.closing voxels, fill what is enclosed;
    2. near the surface, use the true distance to the surface samples, signed by inside / outside;
    settings.keep_open: where the surface is open (no interior within open_distance voxels), keep it as a thin wall."""
    low, high = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    voxel = float((high - low).max()) / settings.resolution
    margin = settings.closing + GRID_MARGIN_BEYOND_CLOSING
    origin = low - margin * voxel
    shape = np.ceil((high - low) / voxel).astype(int) + 2 * margin + 1
    points = sample_triangle_points(mesh, SURFACE_SAMPLE_SPACING * voxel)
    cells = np.clip(np.floor((points - origin) / voxel).astype(int), 0, shape - 1)
    surface = np.zeros(shape, dtype=bool)
    surface[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    solid = fill_enclosed_voxels(surface, settings.closing)
    interior, exterior = solid & ~surface, ~solid
    ball = ndimage.generate_binary_structure(3, 1)
    distance = np.full(shape, DISTANCE_CLAMP, dtype=np.float32)
    band = np.argwhere(ndimage.binary_dilation(surface, ball, iterations=NEAR_SURFACE_BAND))
    distance[tuple(band.T)] = cKDTree(points).query(origin + (band + 0.5) * voxel)[0] / voxel
    nearer_inside = ndimage.distance_transform_edt(~interior) < ndimage.distance_transform_edt(~exterior)
    inside = interior | (surface & nearer_inside)
    field = np.where(inside, -distance, distance)
    if settings.keep_open:
        far_from_interior = ndimage.distance_transform_edt(~interior) > settings.open_distance
        wall = ndimage.binary_dilation(surface & far_from_interior, ball, iterations=1) & ~interior
        field[wall] = -np.maximum(distance[wall], MIN_WALL_DISTANCE)
    return DistanceGrid(field, origin, voxel)


def extract_surface(grid: DistanceGrid, sigma: float = SOLID_SMOOTHING) -> Mesh:
    """Smooth the signed field lightly (Gaussian sigma in voxels) and extract its zero level with marching cubes."""
    measure = importlib.import_module("skimage.measure")  # scikit-image is only needed for broken meshes
    vertices, faces, _, _ = measure.marching_cubes(ndimage.gaussian_filter(grid.field, sigma), level=0.0)
    return Mesh((vertices + 0.5) * grid.voxel_size + grid.origin, faces.astype(np.int64))


def rebuild_as_solid(mesh: Mesh, settings: SolidRebuildSettings | None = None) -> tuple[Mesh, float]:
    """Rebuild a broken surface as the boundary of a solid, keeping the original surface position (signed distance
    grid + light smoothing + marching cubes). Returns (mesh, voxel size in mesh units)."""
    settings = settings or SolidRebuildSettings()
    grid = compute_signed_distance_grid(mesh, settings)
    return extract_surface(grid, settings.sigma), grid.voxel_size


def reduce_faces(mesh: Mesh, max_faces: int, simplify: Simplifier) -> tuple[Mesh, int]:
    """Reduce to at most max_faces faces with simplify, retrying with a slightly lower target when it overshoots.
    Returns (mesh, attempts); does nothing when the mesh is already small enough."""
    attempts = 0
    while len(mesh.faces) > max_faces and attempts < MAX_REDUCE_ATTEMPTS:
        target = max_faces * (REDUCE_TARGET_SHRINK_PER_ATTEMPT**attempts)
        reduction = 1.0 - target / len(mesh.faces)
        vertices, faces = simplify(mesh.vertices, mesh.faces, reduction)
        mesh = remove_unused_vertices(
            Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64).reshape(-1, 3))
        )
        attempts += 1
    return mesh, attempts


def find_boundary_loops(faces: IntArray) -> list[list[int]]:
    """Closed loops of boundary edges (vertex indices), each following its faces' edge direction."""
    edge_ids, counts = index_undirected_edges(faces)
    boundary = list_directed_edges(faces)[counts[edge_ids] == 1]
    outgoing: dict[int, list[int]] = {}
    for start, end in boundary.tolist():
        outgoing.setdefault(start, []).append(end)
    loops = []
    for origin in list(outgoing):
        while outgoing.get(origin):
            loop = [origin]
            current = outgoing[origin].pop()
            while current != origin and outgoing.get(current) and len(loop) <= len(boundary):
                loop.append(current)
                current = outgoing[current].pop()
            if current == origin and len(loop) >= TRIANGLE_CORNERS:
                loops.append(loop)
    return loops


def remove_nonmanifold_faces(mesh: Mesh) -> tuple[Mesh, int, int]:
    """Drop every face on an edge used by more than two faces (decimation can pinch sheets together); fill_holes
    closes the holes this leaves. Returns (mesh, bad edges, faces removed)."""
    if len(mesh.faces) == 0:
        return mesh, 0, 0
    edge_ids, counts = index_undirected_edges(mesh.faces)
    bad_edges = np.nonzero(counts > FACES_PER_MANIFOLD_EDGE)[0]
    if len(bad_edges) == 0:
        return mesh, 0, 0
    owners = np.tile(np.arange(len(mesh.faces)), 3)
    drop = np.unique(owners[np.isin(edge_ids, bad_edges)])
    kept = remove_unused_vertices(Mesh(mesh.vertices, np.delete(mesh.faces, drop, axis=0)))
    return kept, len(bad_edges), len(drop)


def fill_holes(mesh: Mesh) -> tuple[Mesh, int, int]:
    """Close every boundary loop (one triangle, or a fan around the loop centre).
    Returns (mesh, holes filled, faces added)."""
    loops = find_boundary_loops(mesh.faces)
    vertex_blocks, face_blocks = [mesh.vertices], [mesh.faces]
    next_index = len(mesh.vertices)
    for loop in loops:
        ring = np.asarray(loop, dtype=np.int64)
        if len(ring) == TRIANGLE_CORNERS:  # a triangle hole is closed by one face
            face_blocks.append(ring[[2, 1, 0]][None, :])
            continue
        vertex_blocks.append(mesh.vertices[ring].mean(axis=0)[None, :])
        centre = np.full(len(ring), next_index, dtype=np.int64)
        face_blocks.append(np.column_stack([np.roll(ring, -1), ring, centre]))
        next_index += 1
    filled = Mesh(np.concatenate(vertex_blocks), np.concatenate(face_blocks))
    return filled, len(loops), len(filled.faces) - len(mesh.faces)


def repair_and_fill(mesh: Mesh) -> tuple[Mesh, dict[str, int]]:
    """Remove non-manifold faces and fill holes until no non-manifold edge is left (at most MAX_REPAIR_ROUNDS).
    Returns (mesh, counts)."""
    counts = {"nonmanifold_edges_repaired": 0, "nonmanifold_faces_removed": 0, "holes_filled": 0, "hole_faces_added": 0}
    for _ in range(MAX_REPAIR_ROUNDS):
        mesh, edges, removed = remove_nonmanifold_faces(mesh)
        mesh, holes, added = fill_holes(mesh)
        counts["nonmanifold_edges_repaired"] += edges
        counts["nonmanifold_faces_removed"] += removed
        counts["holes_filled"] += holes
        counts["hole_faces_added"] += added
        if edges == 0:
            break
    return mesh, counts


def compute_signed_volumes(mesh: Mesh) -> FloatArray:
    """Signed volume of the tetrahedron (origin, face) for every face; their sum is the enclosed volume."""
    corners = mesh.vertices[mesh.faces]
    return np.asarray(np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])) / 6.0)


def compute_volume(mesh: Mesh) -> float:
    """Enclosed volume (positive for outward-facing faces of a closed mesh)."""
    return float(compute_signed_volumes(mesh).sum())


def pair_manifold_edges(faces: IntArray) -> tuple[IntArray, IntArray, BoolArray]:
    """For edges shared by exactly two faces: (first face, second face, both run the edge in the same direction)."""
    edge_ids, counts = index_undirected_edges(faces)
    directed = list_directed_edges(faces)
    owners = np.tile(np.arange(len(faces)), 3)
    shared = np.nonzero(counts[edge_ids] == FACES_PER_MANIFOLD_EDGE)[0]
    ordered = shared[np.argsort(edge_ids[shared], kind="stable")]
    first, second = ordered[0::2], ordered[1::2]
    same_direction = directed[first, 0] == directed[second, 0]
    return owners[first], owners[second], same_direction


def orient_faces_consistently(faces: IntArray) -> tuple[BoolArray, IntArray]:
    """Faces to flip so neighbours agree on winding. Returns (flip mask, piece label per face)."""
    face_count = len(faces)
    face_a, face_b, same_direction = pair_manifold_edges(faces)
    graph = sparse.coo_matrix((np.ones(len(face_a)), (face_a, face_b)), shape=(face_count, face_count)).tocsr()
    _, labels = csgraph.connected_components(graph, directed=False)
    low, high = np.minimum(face_a, face_b), np.maximum(face_a, face_b)
    pair_keys = low * face_count + high
    order = np.argsort(pair_keys)
    sorted_keys, sorted_same = pair_keys[order], same_direction[order]
    flip = np.zeros(face_count, dtype=bool)
    roots = np.unique(labels, return_index=True)[1]
    for root in roots:
        visit, parents = csgraph.breadth_first_order(graph, int(root), directed=False, return_predecessors=True)
        children = visit[1:]
        parent_of = parents[children]
        keys = np.minimum(children, parent_of) * face_count + np.maximum(children, parent_of)
        relation = sorted_same[np.searchsorted(sorted_keys, keys)]
        for child, parent, same in zip(children.tolist(), parent_of.tolist(), relation.tolist(), strict=True):
            flip[child] = flip[parent] ^ same
    return flip, np.asarray(labels, dtype=np.int64)


def fix_normals(mesh: Mesh) -> tuple[Mesh, int]:
    """Make the winding consistent per piece and outward (positive volume). Returns (mesh, faces flipped)."""
    if len(mesh.faces) == 0:
        return mesh, 0
    flip, labels = orient_faces_consistently(mesh.faces)
    faces = mesh.faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    piece_volume = np.bincount(labels, weights=compute_signed_volumes(Mesh(mesh.vertices, faces)))
    inward = piece_volume[labels] < 0
    faces[inward] = faces[inward][:, ::-1]
    return Mesh(mesh.vertices, faces), int((flip ^ inward).sum())


def check_watertight(faces: IntArray) -> JsonDict:
    """watertight: every edge used by exactly two faces; winding_consistent: and in opposite directions; plus the
    boundary and non-manifold edge counts."""
    if len(faces) == 0:
        return {"watertight": False, "winding_consistent": False, "boundary_edges": 0, "nonmanifold_edges": 0}
    _, counts = index_undirected_edges(faces)
    watertight = bool((counts == FACES_PER_MANIFOLD_EDGE).all())
    directed_unique = len(np.unique(list_directed_edges(faces), axis=0)) == 3 * len(faces)
    return {
        "watertight": watertight,
        "winding_consistent": bool(watertight and directed_unique),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > FACES_PER_MANIFOLD_EDGE).sum()),
    }


def remove_debris(mesh: Mesh, report: JsonDict) -> Mesh:
    """Weld vertices, drop degenerate / duplicate faces and small pieces; counts go into report."""
    report["faces_in"], report["vertices_in"] = len(mesh.faces), len(mesh.vertices)
    mesh = merge_vertices(mesh)
    report["vertices_after_merge"] = len(mesh.vertices)
    mesh, report["degenerate_faces_removed"] = remove_degenerate_faces(mesh)
    mesh, report["duplicate_faces_removed"] = remove_duplicate_faces(mesh)
    mesh, report["small_pieces_removed_first"], report["small_piece_faces_removed_first"] = remove_small_pieces(mesh)
    return mesh


def rebuild_if_broken(mesh: Mesh, settings: SolidRebuildSettings, report: JsonDict) -> tuple[Mesh, bool]:
    """Rebuild the mesh as a solid when needs_solid_rebuild says so (and drop the small pieces that makes); counts go
    into report. Returns (mesh, rebuilt)."""
    rebuild, report["nonmanifold_edges_raw"], report["boundary_edges_raw"] = needs_solid_rebuild(mesh)
    report["solid_rebuilt"], report["solid_voxel_size"] = rebuild, None
    if not rebuild:
        return mesh, False
    mesh, voxel = rebuild_as_solid(mesh, settings)
    mesh, extra_pieces, extra_faces = remove_small_pieces(mesh)
    report["solid_voxel_size"] = round(voxel, 6)
    report["faces_after_rebuild"] = len(mesh.faces)
    report["small_pieces_removed_first"] += extra_pieces
    report["small_piece_faces_removed_first"] += extra_faces
    return mesh, True


def reduce_and_repair(mesh: Mesh, max_faces: int, simplify: Simplifier, *, staged: bool, report: JsonDict) -> Mesh:
    """Reduce to max_faces (staged: first to STAGED_REDUCTION_FACTOR x, repaired in between), drop small pieces,
    repair non-manifold edges and holes; counts go into report."""
    report["faces_before_reduce"] = len(mesh.faces)
    staged_counts: dict[str, int] = {}
    if staged:
        mesh, _ = reduce_faces(mesh, STAGED_REDUCTION_FACTOR * max_faces, simplify)
        mesh, staged_counts = repair_and_fill(mesh)
        report["faces_after_first_stage"] = len(mesh.faces)
    mesh, report["reduce_attempts"] = reduce_faces(mesh, max_faces, simplify)
    report["faces_after_reduce"] = len(mesh.faces)
    mesh, pieces_second, faces_second = remove_small_pieces(mesh)
    report["small_pieces_removed_second"] = pieces_second
    report["small_piece_faces_removed_second"] = faces_second
    report["pieces_removed"] = report["small_pieces_removed_first"] + pieces_second
    mesh, repair_counts = repair_and_fill(mesh)
    for key, value in staged_counts.items():
        repair_counts[key] += value
    report.update(repair_counts)
    return mesh


def clean_mesh(
    mesh: Mesh, max_faces: int, simplify: Simplifier, rebuild_settings: SolidRebuildSettings | None = None
) -> tuple[Mesh, JsonDict]:
    """Run the full clean-up. Returns (closed, outward-facing mesh with at most max_faces faces, report of every
    count)."""
    report: JsonDict = {}
    mesh = remove_debris(mesh, report)
    mesh, rebuilt = rebuild_if_broken(mesh, rebuild_settings or SolidRebuildSettings(), report)
    mesh = reduce_and_repair(mesh, max_faces, simplify, staged=rebuilt, report=report)
    mesh, report["faces_flipped"] = fix_normals(mesh)
    mesh = remove_unused_vertices(mesh)
    report["faces_out"] = len(mesh.faces)
    report["vertices_out"] = len(mesh.vertices)
    report["pieces_out"] = count_pieces(mesh.faces)
    report.update(check_watertight(mesh.faces))
    report["volume"] = round(compute_volume(mesh), 6)
    return mesh, report
