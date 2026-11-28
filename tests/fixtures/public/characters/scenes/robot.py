"""Robot test character (own work): teal body and head, screen face with cyan eyes, antenna."""

from collections.abc import Callable

HALF_EXTENT = 1.15
CENTER = (0.0, 1.05)
TEAL = (64, 176, 170)
DARK = (45, 52, 64)
GREY = (200, 205, 212)
ORANGE = (242, 140, 50)
SCREEN = (30, 38, 48)
CYAN = (120, 235, 255)


def build(make: Callable[..., object]) -> list[object]:
    """The robot's shapes, in drawing order (make: sdf_render.make_primitive)."""
    return [
        # legs + feet
        make("cyl", GREY, mirror=True, centre=(0.22, 0.28, 0), radius=0.13, half_height=0.2, corner_radius=0.04),
        make(
            "rbox",
            DARK,
            mirror=True,
            blend=0.02,
            centre=(0.22, 0.07, 0.05),
            half_size=(0.18, 0.08, 0.24),
            corner_radius=0.06,
        ),
        # body
        make("rbox", TEAL, blend=0.03, centre=(0, 0.78, 0), half_size=(0.45, 0.35, 0.32), corner_radius=0.14),
        make("rbox", GREY, centre=(0, 0.75, 0.29), half_size=(0.24, 0.16, 0.06), corner_radius=0.05),  # chest panel
        make("sphere", ORANGE, centre=(-0.09, 0.78, 0.34), radius=0.055),
        make("sphere", CYAN, centre=(0.09, 0.78, 0.34), radius=0.055),
        # arms
        make("sphere", GREY, mirror=True, centre=(0.5, 0.98, 0), radius=0.12),
        make(
            "capsule", TEAL, mirror=True, blend=0.03, start=(0.52, 0.95, 0), end=(0.6, 0.62, 0.05), radius=(0.1, 0.09)
        ),
        make("sphere", DARK, mirror=True, centre=(0.61, 0.52, 0.06), radius=0.12),
        # neck + head
        make("cyl", GREY, centre=(0, 1.18, 0), radius=0.12, half_height=0.07, corner_radius=0.02),
        make("rbox", TEAL, blend=0.02, centre=(0, 1.55, 0), half_size=(0.52, 0.36, 0.4), corner_radius=0.17),
        make("rbox", SCREEN, centre=(0, 1.56, 0.33), half_size=(0.4, 0.24, 0.08), corner_radius=0.08),
        make("ellipsoid", CYAN, centre=(-0.16, 1.6, 0.405), radii=(0.07, 0.09, 0.02)),
        make("ellipsoid", CYAN, centre=(0.16, 1.6, 0.405), radii=(0.07, 0.09, 0.02)),
        make("torus", CYAN, rotation=(90, 0, 0), centre=(0, 1.46, 0.405), ring_radius=0.07, tube_radius=0.015),
        # ear bolts + antenna
        make(
            "cyl",
            ORANGE,
            mirror=True,
            rotation=(0, 0, 90),
            centre=(0.53, 1.55, 0),
            radius=0.1,
            half_height=0.05,
            corner_radius=0.02,
        ),
        make("capsule", GREY, start=(0, 1.9, 0), end=(0, 2.1, 0), radius=0.025),
        make("sphere", ORANGE, centre=(0, 2.12, 0), radius=0.065),
        # back hatch
        make("rbox", GREY, centre=(0, 0.8, -0.31), half_size=(0.22, 0.2, 0.05), corner_radius=0.04),
    ]
