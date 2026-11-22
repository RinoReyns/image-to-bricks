"""Surface colours from the mesh (vertex colours or UV texture) onto the cells, then the whole colour pipeline
(assign_voxel_colors).

Cell arrays are [z, y, x]; mesh space is the scaled, Z-stretched mesh that was voxelised (mm).
"""

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from tqdm import tqdm

from image_to_bricks.colours import trim_snap
from image_to_bricks.colours.hand_edits import apply_paints, paint_front_ovals
from image_to_bricks.colours.palette_matching import (
    LIGHT_GAMMA,
    SAT_BOOST,
    cluster_to_lego,
    enhance_colors,
    map_regions_to_lego,
)
from image_to_bricks.colours.surface_cleanup import (
    SMOOTH_MAJORITY,
    TRIM_MIN_CELLS,
    TrimRings,
    absorb_small_color_regions_3d,
    apply_automatic_colour_cleanup,
    apply_color_overrides,
    denoise_voxel_colors,
    recolour_from_neighbours,
    smooth_lego_colors,
    snap_trim_to_rings,
    trim_ring_mask,
    unify_layer_regions,
)
from image_to_bricks.consts import COLOR_MPL, COLOR_NAMES, STUD_MM
from image_to_bricks.instructions import fast_render
from image_to_bricks.mesh.voxelize import surface_mask
from image_to_bricks.options import BuildOptions, ColourOptions

# --- colour pipeline switches ---------------------------------------------
# Vertex colour used by the colouriser for surfaces no photo reached.
# Voxels close to it are treated as "no data" and get the nearest real colour.
MISSING_TOL = 28  # RGB distance
USE_DENOISE = False  # denoise_voxel_colors (bleeds saturated colours)
USE_UNIFY_LAYERS = False  # unify_layer_regions (one colour per layer ring)
UNIFY_MIN_REGION_CELLS = 20  # unify_layer_regions: smallest region made one colour
NEAREST_TRIANGLE_CANDIDATES = 16  # nearest proxy triangles checked per sampled voxel face (exact closest point)
MAX_PROXY_TRIANGLES = 300_000  # triangles in the k-d tree of triangle centres (every n-th one on bigger meshes)
SAMPLE_BATCH_POINTS = 2000  # sample points per closest-point batch
DEGENERATE_TRIANGLE_AREA = 1e-8  # barycentric: a smaller determinant counts as a degenerate triangle
REGION_COLOUR_THRESHOLD = 35  # map_regions_to_lego: RGB distance that still joins a region (no --color-groups)
REGION_MIN_CELLS = 6  # map_regions_to_lego: smaller regions are merged
UNKNOWN_ID_RGB = (255, 0, 255)  # debug images: magenta for a colour id missing from the palette
SURFACE_DISTANCE_PERCENTILE = 95  # reported sample distance percentile


@dataclass(frozen=True)
class VoxelFrame:
    """Where the cells sit in mesh space: xyz position of the CENTRE of cell [0, 0, 0] and the cell size per axis
    (trimesh puts cell centres on multiples of the pitch)."""

    origin: NDArray[np.float64]
    pitch: NDArray[np.float64]


@dataclass(frozen=True)
class ColourDebug:
    """Where the step-by-step colour images (<model>_dbg*.png) go; directory None = no images."""

    directory: str | None = None
    model_name: str = "model"

    def save(self, grid: NDArray[Any], values: NDArray[Any], step: str, title: str, *, is_ids: bool = False) -> None:
        """<directory>/<model>_<step>.png of the cells coloured by RGB or palette ids (nothing without directory)."""
        if self.directory:
            path = os.path.join(self.directory, f"{self.model_name}_{step}.png")
            save_color_debug(grid, values, path, title, is_ids=is_ids)


def barycentric(
    p: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Barycentric coordinates (u, v, w) of point p in triangle (a, b, c); (1, 0, 0) for a degenerate triangle."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < DEGENERATE_TRIANGLE_AREA:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])


@dataclass(frozen=True)
class _MeshColours:
    """The colour source of a mesh: its UV texture (RGB float, baseColorFactor applied) or its vertex colours."""

    faces: NDArray[Any]
    texture: NDArray[np.float32] | None = None
    uv: NDArray[Any] | None = None
    vertex_colours: NDArray[np.float32] | None = None

    def colour_at(self, triangle: NDArray[Any], bary: NDArray[np.float64]) -> NDArray[np.float32]:
        """RGB at the barycentric point of the triangle (vertex indices): texture pixel or blended vertex colour."""
        if self.texture is not None and self.uv is not None:
            u, v = np.mod(self.uv[triangle].T @ bary, 1.0)
            texture_height, texture_width = self.texture.shape[:2]
            texel: NDArray[np.float32] = self.texture[
                int((1.0 - v) * (texture_height - 1)), int(u * (texture_width - 1))
            ]
            return texel
        if self.vertex_colours is None:
            raise ValueError("mesh colours have neither a texture nor vertex colours")
        vertex_colours = self.vertex_colours
        blended: NDArray[np.float32] = (
            bary[0] * vertex_colours[triangle[0]]
            + bary[1] * vertex_colours[triangle[1]]
            + bary[2] * vertex_colours[triangle[2]]
        )
        return blended


def _texture_colours(faces: NDArray[Any], visual: "trimesh.visual.TextureVisuals") -> _MeshColours:
    """Colour source of a textured mesh: its base colour texture with baseColorFactor applied (0..255 RGB)."""
    material = visual.material
    if not isinstance(material, trimesh.visual.material.PBRMaterial) or material._data.get("baseColorTexture") is None:
        raise ValueError("The GLB material has no baseColorTexture.")
    texture_image = material._data["baseColorTexture"]
    texture = np.asarray(texture_image.convert("RGB"), dtype=np.float32)
    factor = np.asarray(material._data.get("baseColorFactor", [255, 255, 255, 255]), dtype=np.float32)[:3]
    if factor.max() <= 1.0:
        factor *= 255.0
    texture *= factor / 255.0
    return _MeshColours(faces, texture=texture, uv=visual.uv)


def _prepare_visual_data(mesh: trimesh.Trimesh) -> _MeshColours:
    """The colour source of the mesh; ValueError for a material without a texture or an unsupported visual."""
    visual = mesh.visual
    if isinstance(visual, trimesh.visual.TextureVisuals):
        return _texture_colours(mesh.faces, visual)
    if isinstance(visual, trimesh.visual.ColorVisuals) and visual.kind == "vertex":
        return _MeshColours(mesh.faces, vertex_colours=visual.vertex_colors[:, :3].astype(np.float32))
    raise ValueError(f"Unsupported visual type: {None if visual is None else visual.kind}")


def fill_from_nearest(
    values: NDArray[Any], source_mask: NDArray[np.bool_], target_mask: NDArray[np.bool_]
) -> NDArray[Any]:
    """Copy values[source] into every target voxel from its nearest source voxel."""
    if not source_mask.any():
        return values
    _, idx = distance_transform_edt(~source_mask, return_indices=True)
    out = values.copy()
    tz, ty, tx = np.nonzero(target_mask & ~source_mask)
    out[tz, ty, tx] = values[idx[0][tz, ty, tx], idx[1][tz, ty, tx], idx[2][tz, ty, tx]]
    return out


def save_color_debug(
    grid: NDArray[Any], rgb_or_ids: NDArray[Any], path: str, title: str, *, is_ids: bool = False
) -> None:
    """Render voxels coloured either by raw RGB (uint8 [...,3]) or LEGO ids
    (fast PIL renderer: front + back view side by side)."""
    if is_ids:
        rgb = np.full((*grid.shape, 3), UNKNOWN_ID_RGB, np.uint8)
        for colour_id, colour_hex in COLOR_MPL.items():
            rgb[rgb_or_ids == colour_id] = fast_render.hex_to_rgb(colour_hex)
    else:
        rgb = rgb_or_ids
    views = [
        fast_render.iso_render(grid, rgb, fast_render.IsoView(view=view, elev=20, title=f"{title} ({name})"))
        for view, name in ((0, "front"), (2, "back"))
    ]
    fast_render.side_by_side(views).save(path)
    print(f"  saved colour debug: {path}")


@dataclass(frozen=True)
class _SurfaceSampler:
    """Closest-point colour lookup on the mesh, through a k-d tree of (every n-th) triangle centre."""

    mesh: trimesh.Trimesh
    colours: _MeshColours
    proxy_face_ids: NDArray[np.int64]
    tree: cKDTree
    candidate_count: int


def _surface_sampler(mesh: trimesh.Trimesh, colours: _MeshColours) -> _SurfaceSampler:
    """Sampler over at most MAX_PROXY_TRIANGLES triangle centres."""
    stride = max(1, int(np.ceil(len(colours.faces) / MAX_PROXY_TRIANGLES)))
    proxy_face_ids = np.arange(0, len(colours.faces), stride, dtype=np.int64)
    proxy_centres = mesh.vertices[colours.faces[proxy_face_ids]].mean(axis=1)
    # A mesh with fewer triangles than the wanted candidates would make the k-d tree pad its answer with an
    # out-of-range index, so never ask for more candidates than there are proxy triangles.
    candidate_count = min(NEAREST_TRIANGLE_CANDIDATES, len(proxy_face_ids))
    return _SurfaceSampler(mesh, colours, proxy_face_ids, cKDTree(proxy_centres), candidate_count)


def _face_sample_points(
    grid: NDArray[np.bool_], surface: NDArray[np.bool_], frame: VoxelFrame
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """(points, cells): the centre of every exposed face of every surface cell, in mesh space (Z is stretched, so
    every axis uses the cell pitch -- not PLATE_MM), and the cell (z, y, x) each point belongs to."""
    depth_z, depth_y, depth_x = grid.shape
    half = frame.pitch / 2.0
    points: list[tuple[float, float, float]] = []
    point_cells: list[tuple[int, int, int]] = []
    for z, y, x in np.argwhere(surface):
        centre_x, centre_y, centre_z = frame.origin + np.array([x, y, z]) * (2.0 * half)
        x0, x1 = centre_x - half[0], centre_x + half[0]
        y0, y1 = centre_y - half[1], centre_y + half[1]
        z0, z1 = centre_z - half[2], centre_z + half[2]
        mid_x, mid_y, mid_z = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
        faces = [
            ((mid_x, mid_y, z1), z == depth_z - 1 or not grid[z + 1, y, x]),
            ((mid_x, mid_y, z0), z == 0 or not grid[z - 1, y, x]),
            ((x0, mid_y, mid_z), x == 0 or not grid[z, y, x - 1]),
            ((x1, mid_y, mid_z), x == depth_x - 1 or not grid[z, y, x + 1]),
            ((mid_x, y0, mid_z), y == 0 or not grid[z, y - 1, x]),
            ((mid_x, y1, mid_z), y == depth_y - 1 or not grid[z, y + 1, x]),
        ]
        exposed = [point for point, is_exposed in faces if is_exposed]
        points.extend(exposed)
        point_cells.extend([(z, y, x)] * len(exposed))
    return points, point_cells


def _colour_batch(
    sampler: _SurfaceSampler,
    query: NDArray[np.float64],
    cells: list[tuple[int, int, int]],
    colours: NDArray[np.uint8],
    best_distance: NDArray[np.float32],
) -> None:
    """Colour the cells of one batch of sample points from the closest mesh point, where it is closer than the
    cell's best sample so far (colours and best_distance change in place)."""
    count = sampler.candidate_count
    _, proxy_indices = sampler.tree.query(query, k=count, workers=-1)
    candidate_faces = sampler.proxy_face_ids[proxy_indices]
    faces = sampler.colours.faces
    triangles = sampler.mesh.vertices[faces[candidate_faces.reshape(-1)]]
    repeated = np.repeat(query, count, axis=0)
    closest = trimesh.triangles.closest_point(triangles, repeated)
    distances = np.linalg.norm(closest - repeated, axis=1).reshape(-1, count)
    for local_index, choice in enumerate(distances.argmin(axis=1)):
        z, y, x = cells[local_index]
        distance = distances[local_index, choice]
        if distance >= best_distance[z, y, x]:
            continue
        triangle = faces[candidate_faces[local_index, choice]]
        a, b, c = sampler.mesh.vertices[triangle]
        bary = barycentric(closest[local_index * count + choice], a, b, c)
        colours[z, y, x] = sampler.colours.colour_at(triangle, bary)
        best_distance[z, y, x] = distance


def _sample_surface_rgb(
    grid: NDArray[np.bool_], surface: NDArray[np.bool_], sampler: _SurfaceSampler, frame: VoxelFrame
) -> NDArray[np.uint8]:
    """uint8 RGB [z, y, x, 3] of every surface cell: the colour of the mesh point closest to one of its exposed face
    centres (0 elsewhere). Prints the sample distances."""
    points, point_cells = _face_sample_points(grid, surface, frame)
    print(f"Sampling texture on {len(points)} exposed voxel faces...")
    colours = np.zeros((*grid.shape, 3), dtype=np.uint8)
    best_distance = np.full(grid.shape, np.inf, dtype=np.float32)
    for start in tqdm(range(0, len(points), SAMPLE_BATCH_POINTS), desc="Colouring voxels"):
        stop = min(start + SAMPLE_BATCH_POINTS, len(points))
        query = np.asarray(points[start:stop], dtype=np.float64)
        _colour_batch(sampler, query, point_cells[start:stop], colours, best_distance)
    distances = best_distance[surface]
    print(
        f"Surface sample distance to mesh (mm, stretched space): "
        f"median={np.median(distances):.2f}, p95={np.percentile(distances, SURFACE_DISTANCE_PERCENTILE):.2f}, "
        f"max={distances.max():.2f}"
    )
    return colours


def _fill_missing_data(
    colours: NDArray[np.uint8], surface: NDArray[np.bool_], colour: ColourOptions, debug: ColourDebug
) -> NDArray[np.uint8]:
    """--missing-rgb: surface cells near the colouriser's "no data" grey copy the nearest real colour."""
    if colour.missing_rgb is None:
        return colours
    difference = colours.astype(np.float32) - np.asarray(colour.missing_rgb, np.float32)
    missing = surface & (np.linalg.norm(difference, axis=-1) <= MISSING_TOL)
    print(f"Missing-data grey voxels: {int(missing.sum())} / {int(surface.sum())} surface")
    colours = fill_from_nearest(colours, surface & ~missing, surface)
    debug.save(surface, colours, "dbg1b_missing_filled", "1b grey 'no data' filled")
    return colours


def _map_to_palette(
    colours: NDArray[np.uint8], surface: NDArray[np.bool_], colour: ColourOptions, debug: ColourDebug
) -> NDArray[Any]:
    """Palette ids of the surface cells: enhanced RGB, then k-means groups (--color-groups) or region growing.
    All colour processing runs on VISIBLE cells only: interior cells were never sampled (they stayed RGB 0, 0, 0 =
    black) and used to win the majority votes, turning whole layers black."""
    if USE_DENOISE:
        colours = denoise_voxel_colors(colours, surface)
    colours = enhance_colors(colours, surface)
    print("Unique RGB colors:", len(np.unique(colours[surface], axis=0)))
    debug.save(surface, colours, "dbg2_enhanced_rgb", f"2 enhanced (sat x{SAT_BOOST}, gamma {LIGHT_GAMMA})")
    if colour.colour_groups > 0:
        lego = cluster_to_lego(colours, surface, colour.colour_groups, colour.allowed_colour_ids, colour.max_colours)
    else:
        lego = map_regions_to_lego(
            colours,
            surface,
            color_threshold=REGION_COLOUR_THRESHOLD,
            min_region_size=REGION_MIN_CELLS,
            allowed=colour.allowed_colour_ids,
        )
    print(f"LEGO colors after mapping: {len(np.unique(lego[surface]))} -> {list(np.unique(lego[surface]))}")
    debug.save(surface, lego, "dbg3_mapped_ids", "3 mapped to LEGO", is_ids=True)
    return lego


def _trim_rings(
    lego: NDArray[Any], grid: NDArray[np.bool_], surface: NDArray[np.bool_], colour: ColourOptions
) -> tuple[NDArray[Any], TrimRings]:
    """--trim-colors (+ --trim-snap): one full ring per trim feature, its stray cells recoloured from their
    neighbours. Returns (lego, rings)."""
    trim_ids = colour.trim_colour_ids
    trim = trim_ring_mask(lego, grid, trim_ids, colour.trim_min_fraction, TRIM_MIN_CELLS) if trim_ids else {}
    if colour.trim_snap_layers and trim:
        lego, trim = snap_trim_to_rings(lego, grid, surface, trim, colour)
    for trim_id, (ring, stray) in trim.items():
        lego = recolour_from_neighbours(lego, surface, stray, set(trim_ids))
        lego[ring] = trim_id
        print(
            f"Trim ring {COLOR_NAMES.get(trim_id, trim_id)}: {int(ring.sum())} cells in layers "
            f"{sorted(set(np.nonzero(ring.any(axis=(1, 2)))[0].tolist()))}, {int(stray.sum())} stray cells recoloured"
        )
    return lego, trim


def _smooth_and_clean(
    lego: NDArray[Any], surface: NDArray[np.bool_], trim: TrimRings, colour: ColourOptions, debug: ColourDebug
) -> NDArray[Any]:
    """Smoothing, small-region absorbing, the automatic clean-up rules; the trim rings survive all of them."""
    if USE_UNIFY_LAYERS:
        lego = unify_layer_regions(lego, surface, min_region_size=UNIFY_MIN_REGION_CELLS)
    lego = smooth_lego_colors(lego, surface, iters=colour.smooth_iterations, majority=SMOOTH_MAJORITY)
    lego = absorb_small_color_regions_3d(lego, surface, min_region_size=colour.absorb_min_region)
    for trim_id, (ring, _) in trim.items():  # smoothing must not eat the rings
        lego[ring] = trim_id
    lego = apply_automatic_colour_cleanup(lego, surface, colour.trim_colour_ids, colour)
    if colour.trim_snap_layers and trim:
        lego, cleared = trim_snap.clear_strays(lego, surface, trim)
        print(f"Trim snap: {cleared} trim cell(s) spread off the rings by smoothing / clean-up recoloured")
    debug.save(surface, lego, "dbg4_final_ids", "4 after unify/smooth/absorb", is_ids=True)
    return lego


def _finish_colours(
    lego: NDArray[Any], grid: NDArray[np.bool_], surface: NDArray[np.bool_], options: BuildOptions
) -> NDArray[Any]:
    """Hidden cells copy the nearest visible colour (keeps bricks single-coloured through the wall), then the
    hand-made edits: --color-alias, --front-oval, --paint. Prints the final visible colour usage."""
    colour, shape = options.colour, options.shape
    lego = fill_from_nearest(lego, surface, grid.astype(bool))
    if colour.colour_alias:
        lego = apply_color_overrides(lego, grid, colour.colour_alias)
    lego = paint_front_ovals(lego, grid, colour.front_ovals, shape.front, shape.shell_thickness_studs)
    lego = apply_paints(lego, surface, grid, colour.paints)
    final_ids, counts = np.unique(lego[surface], return_counts=True)
    usage = {COLOR_NAMES.get(int(c), c): int(n) for c, n in zip(final_ids, counts, strict=True)}
    print("Final visible colour usage:", usage)
    return lego


def assign_voxel_colors(
    grid: NDArray[np.bool_],
    texture_mesh: trimesh.Trimesh,
    frame: VoxelFrame | None = None,
    debug: ColourDebug | None = None,
    options: BuildOptions | None = None,
) -> NDArray[Any]:
    """Palette colour id [z, y, x] of every filled cell, from the exposed cell faces (UV texture or vertex
    colours). texture_mesh must be the SAME (scaled + Z-stretched) mesh that was voxelised; frame describes that
    grid (default: origin 0, pitch STUD_MM). options: its colour settings and, for --front-oval, the front side and
    wall thickness (default BuildOptions())."""
    options = options if options is not None else BuildOptions()
    frame = frame if frame is not None else VoxelFrame(np.zeros(3), np.full(3, STUD_MM))
    frame = VoxelFrame(np.asarray(frame.origin, dtype=np.float64), np.asarray(frame.pitch, dtype=np.float64))
    debug = debug if debug is not None else ColourDebug()
    colour = options.colour
    mesh_colours = _prepare_visual_data(texture_mesh)
    surface = surface_mask(grid)
    print(f"Surface voxels: {int(surface.sum())} / {int(grid.sum())} filled")
    sampler = _surface_sampler(texture_mesh, mesh_colours)
    colours = _sample_surface_rgb(grid, surface, sampler, frame)
    debug.save(surface, colours, "dbg1_raw_rgb", "1 raw sampled RGB")
    colours = _fill_missing_data(colours, surface, colour, debug)
    lego = _map_to_palette(colours, surface, colour, debug)
    lego, trim = _trim_rings(lego, grid, surface, colour)
    lego = _smooth_and_clean(lego, surface, trim, colour, debug)
    return _finish_colours(lego, grid, surface, options)
