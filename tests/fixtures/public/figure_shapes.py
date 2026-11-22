"""Signed-distance shapes of the three public test figures (robot, frog, bear): our own designs, free to publish.

Each figure is a list of coloured primitives (sphere, ellipsoid, round cone, rounded box, torus, rounded cylinder)
in glTF orientation: Y up, feet at y = 0, face toward +Z, about 2 units tall. make_public_fixtures.py turns them into
vertex-coloured GLB files. Distances are in figure units; a negative distance is inside the primitive.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

Vector = tuple[float, float, float]
Rgb = tuple[int, int, int]


def rotation_matrix_degrees(around_x: float = 0.0, around_y: float = 0.0, around_z: float = 0.0) -> np.ndarray:
    """3x3 rotation, applied X first, then Y, then Z (angles in degrees)."""
    angle_x, angle_y, angle_z = (math.radians(angle) for angle in (around_x, around_y, around_z))
    rotate_x = np.array(
        [[1, 0, 0], [0, math.cos(angle_x), -math.sin(angle_x)], [0, math.sin(angle_x), math.cos(angle_x)]]
    )
    rotate_y = np.array(
        [[math.cos(angle_y), 0, math.sin(angle_y)], [0, 1, 0], [-math.sin(angle_y), 0, math.cos(angle_y)]]
    )
    rotate_z = np.array(
        [[math.cos(angle_z), -math.sin(angle_z), 0], [math.sin(angle_z), math.cos(angle_z), 0], [0, 0, 1]]
    )
    return rotate_z @ rotate_y @ rotate_x


@dataclass
class Primitive:
    """One coloured shape. kind selects the distance function; parameters hold its sizes (see _DISTANCE_BY_KIND).
    mirrored: the shape is repeated on both sides of x = 0 (eyes, ears, legs). blend_radius > 0 melts it smoothly
    into the shapes listed before it (used for the joins between body parts)."""

    kind: str
    rgb: Rgb
    parameters: dict[str, object]
    mirrored: bool = False
    blend_radius: float = 0.0
    rotation_degrees: Vector | None = None
    _world_to_local: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rotation_degrees is not None:
            self._world_to_local = rotation_matrix_degrees(*self.rotation_degrees).T

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from every point (float [n, 3]) to this primitive."""
        if self.mirrored:
            points = points.copy()
            points[:, 0] = np.abs(points[:, 0])
        return _DISTANCE_BY_KIND[self.kind](self, points)

    def to_local(self, points: np.ndarray, centre: Vector) -> np.ndarray:
        """Points relative to centre, in the primitive's own (unrotated) frame."""
        relative = points - np.asarray(centre, float)
        return relative if self._world_to_local is None else relative @ self._world_to_local.T


def _sphere_distance(primitive: Primitive, points: np.ndarray) -> np.ndarray:
    return (
        np.linalg.norm(points - np.asarray(primitive.parameters["centre"], float), axis=1)
        - primitive.parameters["radius"]
    )


def _ellipsoid_distance(primitive: Primitive, points: np.ndarray) -> np.ndarray:
    local = primitive.to_local(points, primitive.parameters["centre"])
    radii = np.asarray(primitive.parameters["radii"], float)
    scaled_length = np.linalg.norm(local / radii, axis=1)
    gradient_length = np.linalg.norm(local / (radii * radii), axis=1)
    return scaled_length * (scaled_length - 1.0) / np.maximum(gradient_length, 1e-9)


def _round_cone_distance(primitive: Primitive, points: np.ndarray) -> np.ndarray:
    """Capsule from start (radius start_radius) to end (radius end_radius)."""
    start, end = np.asarray(primitive.parameters["start"], float), np.asarray(primitive.parameters["end"], float)
    start_radius, end_radius = primitive.parameters["radii"]
    axis = end - start
    along = np.clip((points - start) @ axis / (axis @ axis), 0.0, 1.0)
    nearest_on_axis = start + along[:, None] * axis
    return np.linalg.norm(points - nearest_on_axis, axis=1) - (start_radius + (end_radius - start_radius) * along)


def _rounded_box_distance(primitive: Primitive, points: np.ndarray) -> np.ndarray:
    corner_radius = primitive.parameters["corner_radius"]
    outside = np.abs(primitive.to_local(points, primitive.parameters["centre"])) - (
        np.asarray(primitive.parameters["half_size"], float) - corner_radius
    )
    return np.linalg.norm(np.maximum(outside, 0.0), axis=1) + np.minimum(outside.max(axis=1), 0.0) - corner_radius


def _torus_distance(primitive: Primitive, points: np.ndarray) -> np.ndarray:
    """Ring around the local y axis."""
    local = primitive.to_local(points, primitive.parameters["centre"])
    ring_offset = np.hypot(local[:, 0], local[:, 2]) - primitive.parameters["ring_radius"]
    return np.hypot(ring_offset, local[:, 1]) - primitive.parameters["tube_radius"]


def _rounded_cylinder_distance(primitive: Primitive, points: np.ndarray) -> np.ndarray:
    """Cylinder along the local y axis with rounded edges."""
    local = primitive.to_local(points, primitive.parameters["centre"])
    edge_radius = primitive.parameters.get("edge_radius", 0.0)
    radial = np.hypot(local[:, 0], local[:, 2]) - primitive.parameters["radius"] + edge_radius
    vertical = np.abs(local[:, 1]) - primitive.parameters["half_height"] + edge_radius
    return (
        np.minimum(np.maximum(radial, vertical), 0.0)
        + np.hypot(np.maximum(radial, 0.0), np.maximum(vertical, 0.0))
        - edge_radius
    )


_DISTANCE_BY_KIND: dict[str, Callable[[Primitive, np.ndarray], np.ndarray]] = {
    "sphere": _sphere_distance,
    "ellipsoid": _ellipsoid_distance,
    "round_cone": _round_cone_distance,
    "rounded_box": _rounded_box_distance,
    "torus": _torus_distance,
    "rounded_cylinder": _rounded_cylinder_distance,
}


def figure_distance_and_colour(primitives: list[Primitive], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(signed distance [n], uint8 rgb [n, 3]) of the whole figure: smooth union of the primitives; the colour is
    the one of the nearest primitive."""
    distance = nearest_distance = None
    colours = np.zeros((len(points), 3), np.uint8)
    for primitive in primitives:
        primitive_distance = primitive.distance(points)
        if distance is None:
            distance, nearest_distance = primitive_distance, primitive_distance.copy()
            colours[:] = primitive.rgb
            continue
        is_nearer = primitive_distance < nearest_distance
        nearest_distance = np.where(is_nearer, primitive_distance, nearest_distance)
        colours[is_nearer] = primitive.rgb
        if primitive.blend_radius > 0:
            weight = np.clip(0.5 + 0.5 * (primitive_distance - distance) / primitive.blend_radius, 0.0, 1.0)
            distance = (
                primitive_distance * (1 - weight) + distance * weight - primitive.blend_radius * weight * (1 - weight)
            )
        else:
            distance = np.minimum(distance, primitive_distance)
    return distance, colours


def build_robot() -> list[Primitive]:
    teal, dark, grey, orange, screen, cyan = (
        (64, 176, 170),
        (45, 52, 64),
        (200, 205, 212),
        (242, 140, 50),
        (30, 38, 48),
        (120, 235, 255),
    )
    return [
        Primitive(
            "rounded_cylinder",
            grey,
            dict(centre=(0.22, 0.28, 0), radius=0.13, half_height=0.2, edge_radius=0.04),
            mirrored=True,
        ),
        Primitive(
            "rounded_box",
            dark,
            dict(centre=(0.22, 0.07, 0.05), half_size=(0.18, 0.08, 0.24), corner_radius=0.06),
            mirrored=True,
            blend_radius=0.02,
        ),
        Primitive(
            "rounded_box",
            teal,
            dict(centre=(0, 0.78, 0), half_size=(0.45, 0.35, 0.32), corner_radius=0.14),
            blend_radius=0.03,
        ),
        Primitive("rounded_box", grey, dict(centre=(0, 0.75, 0.29), half_size=(0.24, 0.16, 0.06), corner_radius=0.05)),
        Primitive("sphere", orange, dict(centre=(-0.09, 0.78, 0.34), radius=0.055)),
        Primitive("sphere", cyan, dict(centre=(0.09, 0.78, 0.34), radius=0.055)),
        Primitive("sphere", grey, dict(centre=(0.5, 0.98, 0), radius=0.12), mirrored=True),
        Primitive(
            "round_cone",
            teal,
            dict(start=(0.52, 0.95, 0), end=(0.6, 0.62, 0.05), radii=(0.1, 0.09)),
            mirrored=True,
            blend_radius=0.03,
        ),
        Primitive("sphere", dark, dict(centre=(0.61, 0.52, 0.06), radius=0.12), mirrored=True),
        Primitive("rounded_cylinder", grey, dict(centre=(0, 1.18, 0), radius=0.12, half_height=0.07, edge_radius=0.02)),
        Primitive(
            "rounded_box",
            teal,
            dict(centre=(0, 1.55, 0), half_size=(0.52, 0.36, 0.4), corner_radius=0.17),
            blend_radius=0.02,
        ),
        Primitive("rounded_box", screen, dict(centre=(0, 1.56, 0.33), half_size=(0.4, 0.24, 0.08), corner_radius=0.08)),
        Primitive("ellipsoid", cyan, dict(centre=(0.16, 1.6, 0.405), radii=(0.07, 0.09, 0.02)), mirrored=True),
        Primitive(
            "torus",
            cyan,
            dict(centre=(0, 1.46, 0.405), ring_radius=0.07, tube_radius=0.015),
            rotation_degrees=(90, 0, 0),
        ),
        Primitive(
            "rounded_cylinder",
            orange,
            dict(centre=(0.53, 1.55, 0), radius=0.1, half_height=0.05, edge_radius=0.02),
            mirrored=True,
            rotation_degrees=(0, 0, 90),
        ),
        Primitive("round_cone", grey, dict(start=(0, 1.9, 0), end=(0, 2.1, 0), radii=(0.025, 0.025))),
        Primitive("sphere", orange, dict(centre=(0, 2.12, 0), radius=0.065)),
        Primitive("rounded_box", grey, dict(centre=(0, 0.8, -0.31), half_size=(0.22, 0.2, 0.05), corner_radius=0.04)),
    ]


def build_frog() -> list[Primitive]:
    green, belly, dark, white, pink, bow = (
        (104, 184, 72),
        (232, 240, 190),
        (28, 30, 34),
        (250, 250, 248),
        (240, 140, 150),
        (220, 80, 60),
    )
    smile = [
        Primitive("round_cone", dark, dict(start=start, end=end, radii=(0.017, 0.017)))
        for start, end in zip(_smile_points()[:-1], _smile_points()[1:])
    ]
    return [
        Primitive("ellipsoid", green, dict(centre=(0.28, 0.06, 0.15), radii=(0.2, 0.07, 0.26)), mirrored=True),
        Primitive(
            "round_cone",
            green,
            dict(start=(0.27, 0.1, 0.05), end=(0.3, 0.35, -0.02), radii=(0.11, 0.15)),
            mirrored=True,
            blend_radius=0.05,
        ),
        Primitive("ellipsoid", green, dict(centre=(0, 0.62, 0), radii=(0.48, 0.52, 0.42)), blend_radius=0.08),
        Primitive("ellipsoid", belly, dict(centre=(0, 0.55, 0.13), radii=(0.36, 0.4, 0.31))),
        Primitive(
            "round_cone",
            green,
            dict(start=(0.4, 0.85, 0.05), end=(0.48, 0.5, 0.25), radii=(0.1, 0.08)),
            mirrored=True,
            blend_radius=0.04,
        ),
        Primitive(
            "ellipsoid",
            green,
            dict(centre=(0.48, 0.46, 0.28), radii=(0.1, 0.06, 0.1)),
            mirrored=True,
            blend_radius=0.02,
        ),
        Primitive("ellipsoid", green, dict(centre=(0, 1.25, 0.02), radii=(0.6, 0.4, 0.48)), blend_radius=0.1),
        Primitive("sphere", green, dict(centre=(0.3, 1.55, 0.08), radius=0.2), mirrored=True, blend_radius=0.06),
        Primitive("sphere", white, dict(centre=(0.3, 1.57, 0.14), radius=0.155), mirrored=True),
        Primitive("sphere", dark, dict(centre=(0.3, 1.58, 0.27), radius=0.075), mirrored=True),
        *smile,
        Primitive("ellipsoid", pink, dict(centre=(0.36, 1.2, 0.38), radii=(0.08, 0.05, 0.04)), mirrored=True),
        Primitive(
            "ellipsoid",
            bow,
            dict(centre=(0.11, 0.95, 0.37), radii=(0.11, 0.075, 0.05)),
            mirrored=True,
            rotation_degrees=(0, 0, -12),
        ),
        Primitive("sphere", bow, dict(centre=(0, 0.95, 0.4), radius=0.05)),
    ]


def _smile_points(point_count: int = 9) -> list[Vector]:
    """Points of the frog's smile, a U curve lying just outside the head surface."""
    points = []
    for index in range(point_count):
        position = -1 + 2 * index / (point_count - 1)
        x, y = 0.17 * position, 1.24 + 0.07 * position * position
        z = 0.48 * math.sqrt(max(0.0, 1 - (x / 0.6) ** 2 - ((y - 1.25) / 0.4) ** 2)) + 0.035
        points.append((x, y, z))
    return points


def build_bear() -> list[Primitive]:
    fur, light, dark, scarf, scarf_stripe, white = (
        (176, 120, 78),
        (236, 206, 168),
        (40, 30, 26),
        (60, 110, 200),
        (240, 240, 236),
        (255, 255, 255),
    )
    return [
        Primitive(
            "round_cone", fur, dict(start=(0.2, 0.12, 0.02), end=(0.2, 0.5, 0), radii=(0.17, 0.16)), mirrored=True
        ),
        Primitive("ellipsoid", light, dict(centre=(0.2, 0.1, 0.13), radii=(0.13, 0.12, 0.06)), mirrored=True),
        Primitive("ellipsoid", fur, dict(centre=(0, 0.72, 0), radii=(0.42, 0.45, 0.36)), blend_radius=0.06),
        Primitive("ellipsoid", light, dict(centre=(0, 0.66, 0.17), radii=(0.28, 0.3, 0.22))),
        Primitive(
            "round_cone",
            fur,
            dict(start=(0.36, 0.95, 0), end=(0.5, 0.6, 0.12), radii=(0.12, 0.11)),
            mirrored=True,
            blend_radius=0.05,
        ),
        Primitive("torus", scarf, dict(centre=(0, 1.12, 0), ring_radius=0.28, tube_radius=0.085)),
        Primitive(
            "rounded_box",
            scarf,
            dict(centre=(0.17, 0.9, 0.3), half_size=(0.08, 0.2, 0.04), corner_radius=0.035),
            rotation_degrees=(-10, 0, 10),
        ),
        Primitive(
            "rounded_box",
            scarf_stripe,
            dict(centre=(0.135, 0.71, 0.33), half_size=(0.083, 0.03, 0.042), corner_radius=0.02),
            rotation_degrees=(-10, 0, 10),
        ),
        Primitive("sphere", fur, dict(centre=(0, 1.58, 0), radius=0.5), blend_radius=0.05),
        Primitive("sphere", fur, dict(centre=(0.38, 1.96, -0.05), radius=0.16), mirrored=True, blend_radius=0.04),
        Primitive("sphere", light, dict(centre=(0.39, 1.97, 0.03), radius=0.095), mirrored=True),
        Primitive("ellipsoid", light, dict(centre=(0, 1.46, 0.4), radii=(0.21, 0.15, 0.14)), blend_radius=0.02),
        Primitive("ellipsoid", dark, dict(centre=(0, 1.53, 0.52), radii=(0.08, 0.055, 0.045))),
        Primitive("sphere", dark, dict(centre=(0.19, 1.66, 0.44), radius=0.065), mirrored=True),
        Primitive("sphere", white, dict(centre=(0.17, 1.69, 0.495), radius=0.018), mirrored=True),
        Primitive("sphere", fur, dict(centre=(0, 0.5, -0.36), radius=0.1)),
    ]


FIGURE_BUILDERS: dict[str, Callable[[], list[Primitive]]] = {
    "robot": build_robot,
    "frog": build_frog,
    "bear": build_bear,
}
