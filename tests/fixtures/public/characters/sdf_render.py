"""Tiny numpy SDF raymarcher that made the test characters' pictures (robot, frog, bear: own work, no third-party
images or models). Renders a scene from front / left / back / right as RGBA PNGs with a transparent background:
orthographic camera on +z looking at -z; the "left" picture is taken from the character's left side.

    python tests/fixtures/public/characters/sdf_render.py tests/fixtures/public/characters/scenes/robot.py OUT 512

Scenes (scenes/*.py) define HALF_EXTENT, CENTER and build(make_primitive) -> list of primitives. The same distance
functions build the true shapes (tools/images_to_model/build_truth.py).
"""

import importlib.util
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatArray = NDArray[Any]  # float32 at run time
Size = float | tuple[float, ...]
FLOAT = np.float32
VIEW_YAW_DEGREES = {"front": 0, "left": -90, "back": 180, "right": 90}
DEFAULT_SIZE_PIXELS = 512
SUPERSAMPLING = 2
MAX_MARCH_STEPS = 160
MARCH_STEP_SHARE = 0.9  # of the distance, per raymarching step
HIT_DISTANCE = 2e-4
FAR_DISTANCE = 6.0
CAMERA_Z = 3.0
NORMAL_STEP = 1e-3
TINY_GRADIENT = 1e-8
TINY_ELLIPSOID_GRADIENT = 1e-6
KEY_LIGHT = (-0.45, 0.65, 0.62)
FILL_LIGHT = (0.6, 0.2, 0.5)
KEY_LIGHT_SHARE = 0.75
FILL_LIGHT_SHARE = 0.25
DIFFUSE_STRENGTH = 0.8
OCCLUSION_STEPS = 5
OCCLUSION_STEP = 0.012
OCCLUSION_FALLOFF = 0.9
OCCLUSION_STRENGTH = 4.0
MIN_OCCLUSION = 0.35
SPECULAR_POWER = 40
SPECULAR_STRENGTH = 0.35
RIM_POWER = 3
RIM_STRENGTH = 0.12
SKY_BASE = 0.35
SKY_FROM_UP = 0.15
GAMMA = 1.15
BYTE_SCALE = 255
COLOUR_SCALE = 255.0
RGBA_CHANNELS = 4


def rotation_matrix(x_degrees: float = 0.0, y_degrees: float = 0.0, z_degrees: float = 0.0) -> FloatArray:
    """Rotation about x, then y, then z (degrees), as float32."""
    rx, ry, rz = map(math.radians, (x_degrees, y_degrees, z_degrees))
    about_x = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    about_y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    about_z = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return np.asarray((about_z @ about_y @ about_x).astype(FLOAT))


def as_pair(radius: Size) -> tuple[float, float]:
    """(start, end) radius of a capsule from one radius or two."""
    return (radius[0], radius[1]) if isinstance(radius, tuple) else (radius, radius)


@dataclass(frozen=True)
class Primitive:
    """One shape of a scene: kind, colour (0-1 float32 RGB), smooth-union blend distance with the shapes before
    it (0 = hard union), world -> local rotation (None = axis-aligned), mirrored about x = 0, and its sizes."""

    kind: str
    colour: FloatArray
    blend: float
    world_to_local: FloatArray | None
    is_mirrored: bool
    sizes: dict[str, Size]

    def to_local(self, points: FloatArray, centre: Size) -> FloatArray:
        """Points relative to centre, in the shape's own axes."""
        local = points - np.array(centre, FLOAT)
        if self.world_to_local is not None:
            local = local @ self.world_to_local.T
        return local

    def distance(self, points: FloatArray) -> FloatArray:
        """Signed distance of every point [n, 3] to the shape (negative inside)."""
        if self.is_mirrored:  # symmetric about x = 0
            points = points.copy()
            points[:, 0] = np.abs(points[:, 0])
        return DISTANCE_FUNCTIONS[self.kind](self, points)


def make_primitive(
    kind: str,
    colour: tuple[int, int, int],
    *,
    blend: float = 0.0,
    rotation: tuple[float, float, float] | None = None,
    mirror: bool = False,
    **sizes: Size,
) -> Primitive:
    """A scene shape; colour 0-255 RGB, rotation in degrees (x, y, z), sizes as the kind's distance function reads
    them."""
    world_to_local = rotation_matrix(*rotation).T if rotation else None
    return Primitive(kind, np.array(colour, FLOAT) / COLOUR_SCALE, blend, world_to_local, mirror, sizes)


def sphere_distance(shape: Primitive, points: FloatArray) -> FloatArray:
    """Sphere: centre, radius."""
    return np.asarray(np.linalg.norm(points - np.array(shape.sizes["centre"], FLOAT), axis=1) - shape.sizes["radius"])


def ellipsoid_distance(shape: Primitive, points: FloatArray) -> FloatArray:
    """Ellipsoid (approximate distance): centre, radii."""
    local = shape.to_local(points, shape.sizes["centre"])
    radii = np.array(shape.sizes["radii"], FLOAT)
    k0 = np.linalg.norm(local / radii, axis=1)
    k1 = np.linalg.norm(local / (radii * radii), axis=1)
    return np.asarray(k0 * (k0 - 1.0) / np.maximum(k1, TINY_ELLIPSOID_GRADIENT))


def capsule_distance(shape: Primitive, points: FloatArray) -> FloatArray:
    """Round cone between start (radius[0]) and end (radius[1]), or a capsule with one radius."""
    start, end = np.array(shape.sizes["start"], FLOAT), np.array(shape.sizes["end"], FLOAT)
    start_radius, end_radius = as_pair(shape.sizes["radius"])
    axis = end - start
    from_start = points - start
    along = np.clip(from_start @ axis / (axis @ axis), 0, 1)
    radius = start_radius + (end_radius - start_radius) * along
    return np.asarray(np.linalg.norm(from_start - along[:, None] * axis, axis=1) - radius)


def rounded_box_distance(shape: Primitive, points: FloatArray) -> FloatArray:
    """Box with rounded edges: centre, half_size, corner_radius."""
    corner = shape.sizes["corner_radius"]
    q = np.abs(shape.to_local(points, shape.sizes["centre"])) - (np.array(shape.sizes["half_size"], FLOAT) - corner)
    return np.asarray(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0) - corner)


def torus_distance(shape: Primitive, points: FloatArray) -> FloatArray:
    """Torus around the local y axis: centre, ring_radius, tube_radius."""
    local = shape.to_local(points, shape.sizes["centre"])
    across = np.sqrt(local[:, 0] ** 2 + local[:, 2] ** 2) - shape.sizes["ring_radius"]
    return np.asarray(np.sqrt(across**2 + local[:, 1] ** 2) - shape.sizes["tube_radius"])


def cylinder_distance(shape: Primitive, points: FloatArray) -> FloatArray:
    """Rounded cylinder along the local y axis: centre, radius, half_height, corner_radius (default 0)."""
    local = shape.to_local(points, shape.sizes["centre"])
    corner = shape.sizes.get("corner_radius", 0)
    radial = np.sqrt(local[:, 0] ** 2 + local[:, 2] ** 2) - shape.sizes["radius"] + corner
    vertical = np.abs(local[:, 1]) - shape.sizes["half_height"] + corner
    outside = np.sqrt(np.maximum(radial, 0) ** 2 + np.maximum(vertical, 0) ** 2)
    return np.asarray(np.minimum(np.maximum(radial, vertical), 0) + outside - corner)


DISTANCE_FUNCTIONS: dict[str, Callable[[Primitive, FloatArray], FloatArray]] = {
    "sphere": sphere_distance,
    "ellipsoid": ellipsoid_distance,
    "capsule": capsule_distance,
    "rbox": rounded_box_distance,
    "torus": torus_distance,
    "cyl": cylinder_distance,
}


def scene_distance(shapes: Sequence[Primitive], points: FloatArray) -> FloatArray:
    """Signed distance of the whole scene (shapes joined in order, smoothly where a shape has a blend)."""
    combined = shapes[0].distance(points)
    for shape in shapes[1:]:
        distance = shape.distance(points)
        if shape.blend > 0:
            weight = np.clip(0.5 + 0.5 * (distance - combined) / shape.blend, 0, 1)
            combined = distance * (1 - weight) + combined * weight - shape.blend * weight * (1 - weight)
        else:
            combined = np.minimum(combined, distance)
    return combined


def scene_colour(shapes: Sequence[Primitive], points: FloatArray) -> FloatArray:
    """RGB (0-1) at every point: the colour of the nearest shape (the first one on ties)."""
    nearest = shapes[0].distance(points)
    colours = np.broadcast_to(shapes[0].colour, (len(points), 3)).copy()
    for shape in shapes[1:]:
        distance = shape.distance(points)
        closer = distance < nearest
        nearest = np.where(closer, distance, nearest)
        colours[closer] = shape.colour
    return colours


def surface_normals(shapes: Sequence[Primitive], points: FloatArray) -> FloatArray:
    """Unit normals [n, 3] from central differences of the scene distance."""
    normals = np.zeros_like(points)
    for axis in range(3):
        step = np.zeros(3, FLOAT)
        step[axis] = NORMAL_STEP
        normals[:, axis] = scene_distance(shapes, points + step) - scene_distance(shapes, points - step)
    return np.asarray(normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), TINY_GRADIENT))


def unit(vector: tuple[float, float, float]) -> FloatArray:
    """The vector scaled to length 1 (float32)."""
    array = np.array(vector, FLOAT)
    return array / np.linalg.norm(array)


@dataclass(frozen=True)
class Camera:
    """Orthographic view: picture size in pixels, half the visible extent, and the (x, y) picture centre."""

    size: int
    half_extent: float
    centre: tuple[float, float]


def march_rays(
    shapes: Sequence[Primitive], origins: FloatArray, rotation: FloatArray
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Sphere tracing along -z from every origin, the scene turned by rotation. Returns (distance travelled, hit)."""
    direction = np.array([0, 0, -1], FLOAT)
    travelled = np.zeros(len(origins), FLOAT)
    hit = np.zeros(len(origins), bool)
    active = np.ones(len(origins), bool)
    for _ in range(MAX_MARCH_STEPS):
        indices = np.nonzero(active)[0]
        if len(indices) == 0:
            break
        distance = scene_distance(shapes, (origins[indices] + travelled[indices, None] * direction) @ rotation)
        travelled[indices] += distance * MARCH_STEP_SHARE
        arrived = distance < HIT_DISTANCE
        hit[indices[arrived]] = True
        active[indices[arrived | (travelled[indices] > FAR_DISTANCE)]] = False
    return travelled, hit


def shade(shapes: Sequence[Primitive], points: FloatArray, rotation: FloatArray) -> FloatArray:
    """Lit RGB (0-1) at surface points (scene coordinates): two lights, sky, ambient occlusion, highlight, rim."""
    colours = scene_colour(shapes, points)
    normals = surface_normals(shapes, points) @ rotation.T  # to world coordinates for lighting
    to_viewer = -np.array([0, 0, -1], FLOAT)
    key, fill = unit(KEY_LIGHT), unit(FILL_LIGHT)
    diffuse = KEY_LIGHT_SHARE * np.clip(normals @ key, 0, 1) + FILL_LIGHT_SHARE * np.clip(normals @ fill, 0, 1)
    local_normals = normals @ rotation
    occlusion: FloatArray = np.ones(len(points), FLOAT)
    for step_index in range(1, OCCLUSION_STEPS + 1):
        probe = OCCLUSION_STEP * step_index
        gap = probe - scene_distance(shapes, points + local_normals * probe)
        occlusion -= gap * (OCCLUSION_FALLOFF / step_index) * OCCLUSION_STRENGTH
    occlusion = np.clip(occlusion, MIN_OCCLUSION, 1)
    half_vector = key + to_viewer
    half_vector /= np.linalg.norm(half_vector)
    specular = np.clip(normals @ half_vector, 0, 1) ** SPECULAR_POWER * SPECULAR_STRENGTH
    rim = (1 - np.clip(normals @ to_viewer, 0, 1)) ** RIM_POWER * RIM_STRENGTH
    sky = SKY_BASE + SKY_FROM_UP * normals[:, 1]
    lit = colours * (sky[:, None] + diffuse[:, None] * DIFFUSE_STRENGTH) * occlusion[:, None]
    return np.asarray(np.clip(lit + specular[:, None] + rim[:, None], 0, 1) ** (1 / GAMMA))


def render(shapes: Sequence[Primitive], yaw_degrees: float, camera: Camera) -> NDArray[np.uint8]:
    """RGBA picture [size, size, 4] of the scene turned by yaw_degrees about y (world = rotation * local)."""
    rotation = rotation_matrix(0, yaw_degrees, 0)
    steps = (np.arange(camera.size, dtype=FLOAT) + 0.5) / camera.size * 2 - 1
    grid_x, grid_y = np.meshgrid(steps * camera.half_extent, -steps * camera.half_extent)
    origins = np.stack(
        [grid_x.ravel() + camera.centre[0], grid_y.ravel() + camera.centre[1], np.full(grid_x.size, CAMERA_Z, FLOAT)],
        1,
    ).astype(FLOAT)
    travelled, hit = march_rays(shapes, origins, rotation)
    picture = np.zeros((len(origins), RGBA_CHANNELS), FLOAT)
    indices = np.nonzero(hit)[0]
    points = (origins[indices] + travelled[indices, None] * np.array([0, 0, -1], FLOAT)) @ rotation
    picture[indices, :3] = shade(shapes, points, rotation)
    picture[indices, 3] = 1
    return (picture.reshape(camera.size, camera.size, RGBA_CHANNELS) * BYTE_SCALE).astype(np.uint8)


def load_scene(scene_file: str) -> ModuleType:
    """Import a scene module from its file."""
    spec = importlib.util.spec_from_file_location("scene", scene_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {scene_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_views(scene_file: str, out_dir: str, size: int = DEFAULT_SIZE_PIXELS) -> None:
    """front.png, left.png, back.png, right.png of a scene (rendered SUPERSAMPLING times larger, then reduced)."""
    scene = load_scene(scene_file)
    shapes = scene.build(make_primitive)
    os.makedirs(out_dir, exist_ok=True)
    camera = Camera(size * SUPERSAMPLING, scene.HALF_EXTENT, scene.CENTER)
    for name, yaw_degrees in VIEW_YAW_DEGREES.items():
        picture = Image.fromarray(render(shapes, yaw_degrees, camera), "RGBA")
        picture.resize((size, size), Image.Resampling.LANCZOS).save(os.path.join(out_dir, name + ".png"))
        print(out_dir, name, flush=True)


if __name__ == "__main__":
    render_views(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SIZE_PIXELS)
