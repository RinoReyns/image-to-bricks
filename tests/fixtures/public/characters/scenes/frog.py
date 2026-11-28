"""Frog test character (own work): green body, eyes on top, smile, cheeks and a bow tie."""

import itertools
import math
from collections.abc import Callable

HALF_EXTENT = 1.1
CENTER = (0.0, 0.95)
GREEN = (104, 184, 72)
BELLY = (232, 240, 190)
DARK = (28, 30, 34)
WHITE = (250, 250, 248)
PINK = (240, 140, 150)
BAND = (220, 80, 60)
SMILE_POINTS = 9
SMILE_HALF_WIDTH = 0.17
SMILE_HEIGHT = 1.24
SMILE_CURVE = 0.07
HEAD_CENTRE_Y = 1.25
HEAD_RADII = (0.6, 0.4, 0.48)  # x, y, z of the head ellipsoid the smile lies on
SMILE_LIFT = 0.035  # in front of the head surface


def build_smile_points() -> list[tuple[float, float, float]]:
    """Points of the smile on the head surface, left to right."""
    points = []
    for index in range(SMILE_POINTS):
        t = -1 + 2 * index / (SMILE_POINTS - 1)
        x = SMILE_HALF_WIDTH * t
        y = SMILE_HEIGHT + SMILE_CURVE * t * t
        z = HEAD_RADII[2] * math.sqrt(max(0, 1 - (x / HEAD_RADII[0]) ** 2 - ((y - HEAD_CENTRE_Y) / HEAD_RADII[1]) ** 2))
        points.append((x, y, z + SMILE_LIFT))
    return points


def build(make: Callable[..., object]) -> list[object]:
    """The frog's shapes, in drawing order (make: sdf_render.make_primitive)."""
    smile = build_smile_points()
    return [
        # feet
        make("ellipsoid", GREEN, mirror=True, centre=(0.28, 0.06, 0.15), radii=(0.2, 0.07, 0.26)),
        make(
            "capsule",
            GREEN,
            mirror=True,
            blend=0.05,
            start=(0.27, 0.1, 0.05),
            end=(0.3, 0.35, -0.02),
            radius=(0.11, 0.15),
        ),
        # body
        make("ellipsoid", GREEN, blend=0.08, centre=(0, 0.62, 0), radii=(0.48, 0.52, 0.42)),
        make("ellipsoid", BELLY, centre=(0, 0.55, 0.13), radii=(0.36, 0.4, 0.31)),
        # arms
        make(
            "capsule",
            GREEN,
            mirror=True,
            blend=0.04,
            start=(0.4, 0.85, 0.05),
            end=(0.48, 0.5, 0.25),
            radius=(0.1, 0.08),
        ),
        make("ellipsoid", GREEN, mirror=True, blend=0.02, centre=(0.48, 0.46, 0.28), radii=(0.1, 0.06, 0.1)),
        # head
        make("ellipsoid", GREEN, blend=0.1, centre=(0, 1.25, 0.02), radii=HEAD_RADII),
        # eyes on top
        make("sphere", GREEN, mirror=True, blend=0.06, centre=(0.3, 1.55, 0.08), radius=0.2),
        make("sphere", WHITE, mirror=True, centre=(0.3, 1.57, 0.14), radius=0.155),
        make("sphere", DARK, mirror=True, centre=(0.3, 1.58, 0.27), radius=0.075),
        make("sphere", WHITE, mirror=True, centre=(0.27, 1.62, 0.335), radius=0.022),
        # smile (a chain of capsules on the head surface)
        *(make("capsule", DARK, start=start, end=end, radius=0.017) for start, end in itertools.pairwise(smile)),
        # cheeks, bow tie
        make("ellipsoid", PINK, mirror=True, centre=(0.36, 1.2, 0.38), radii=(0.08, 0.05, 0.04)),
        make(
            "ellipsoid", BAND, mirror=True, rotation=(0, 0, -12), centre=(0.11, 0.95, 0.37), radii=(0.11, 0.075, 0.05)
        ),
        make("sphere", BAND, centre=(0, 0.95, 0.4), radius=0.05),
    ]
