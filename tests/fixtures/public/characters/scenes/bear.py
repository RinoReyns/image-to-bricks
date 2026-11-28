"""Bear test character (own work): brown fur, light muzzle and belly, blue scarf, tail."""

from collections.abc import Callable

HALF_EXTENT = 1.15
CENTER = (0.0, 1.05)
FUR = (176, 120, 78)
LIGHT = (236, 206, 168)
DARK = (40, 30, 26)
SCARF = (60, 110, 200)
SCARF2 = (240, 240, 236)
WHITE = (255, 255, 255)


def build(make: Callable[..., object]) -> list[object]:
    """The bear's shapes, in drawing order (make: sdf_render.make_primitive)."""
    return [
        # legs
        make("capsule", FUR, mirror=True, start=(0.2, 0.12, 0.02), end=(0.2, 0.5, 0), radius=(0.17, 0.16)),
        make("ellipsoid", LIGHT, mirror=True, centre=(0.2, 0.1, 0.13), radii=(0.13, 0.12, 0.06)),
        # body
        make("ellipsoid", FUR, blend=0.06, centre=(0, 0.72, 0), radii=(0.42, 0.45, 0.36)),
        make("ellipsoid", LIGHT, centre=(0, 0.66, 0.17), radii=(0.28, 0.3, 0.22)),
        # arms
        make("capsule", FUR, mirror=True, blend=0.05, start=(0.36, 0.95, 0), end=(0.5, 0.6, 0.12), radius=(0.12, 0.11)),
        # scarf
        make("torus", SCARF, centre=(0, 1.12, 0), ring_radius=0.28, tube_radius=0.085),
        make(
            "rbox",
            SCARF,
            rotation=(-10, 0, 10),
            centre=(0.17, 0.9, 0.3),
            half_size=(0.08, 0.2, 0.04),
            corner_radius=0.035,
        ),
        make(
            "rbox",
            SCARF2,
            rotation=(-10, 0, 10),
            centre=(0.135, 0.71, 0.33),
            half_size=(0.083, 0.03, 0.042),
            corner_radius=0.02,
        ),
        # head
        make("sphere", FUR, blend=0.05, centre=(0, 1.58, 0), radius=0.5),
        make("sphere", FUR, mirror=True, blend=0.04, centre=(0.38, 1.96, -0.05), radius=0.16),
        make("sphere", LIGHT, mirror=True, centre=(0.39, 1.97, 0.03), radius=0.095),
        make("ellipsoid", LIGHT, blend=0.02, centre=(0, 1.46, 0.4), radii=(0.21, 0.15, 0.14)),
        make("ellipsoid", DARK, centre=(0, 1.53, 0.52), radii=(0.08, 0.055, 0.045)),
        make("capsule", DARK, start=(0, 1.47, 0.535), end=(0, 1.41, 0.53), radius=0.012),
        make("sphere", DARK, mirror=True, centre=(0.19, 1.66, 0.44), radius=0.065),
        make("sphere", WHITE, mirror=True, centre=(0.17, 1.69, 0.495), radius=0.018),
        # tail
        make("sphere", FUR, centre=(0, 0.5, -0.36), radius=0.1),
    ]
