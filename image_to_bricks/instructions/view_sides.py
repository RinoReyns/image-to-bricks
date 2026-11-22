"""Which side of the figure a camera sees, from the --front direction.

Grid convention: grid[z, y, x], z up, right-handed (checked against the source photos, see COLOR_FIX_PLAN.md
step 64). Directions on the ground are (x, y) unit vectors. "Left" and "right" always mean the FIGURE's own
left and right (as the figure stands, facing --front), never the side of the picture.
"""

GroundVector = tuple[int, int]

DEFAULT_FRONT = "-y"  # used when --front is not given (the old fixed assumption of the flat views)
FRONT_VECTORS = {"-y": (0, -1), "+y": (0, 1), "-x": (-1, 0), "+x": (1, 0)}
FIGURE_SIDES = ("front", "back", "left", "right")


def front_or_default(front: str | None) -> str:
    """The --front value ("-y" / "+y" / "-x" / "+x"), or DEFAULT_FRONT when it is None."""
    return front if front in FRONT_VECTORS else DEFAULT_FRONT


def figure_left_vector(front: str) -> GroundVector:
    """Ground direction of the figure's own left hand: up (z) x facing direction."""
    facing_x, facing_y = FRONT_VECTORS[front]
    return -facing_y, facing_x


def side_vector(side: str, front: str) -> GroundVector:
    """Ground direction from the figure towards one of its sides ('front', 'back', 'left', 'right')."""
    facing = FRONT_VECTORS[front]
    left = figure_left_vector(front)
    vectors = {"front": facing, "back": (-facing[0], -facing[1]), "left": left, "right": (-left[0], -left[1])}
    return vectors[side]


def screen_right_vector(camera_side: GroundVector) -> GroundVector:
    """Ground direction that points to the right of the picture for a camera standing at camera_side and
    looking at the figure with z up (looking direction x z)."""
    side_x, side_y = camera_side
    return -side_y, side_x


def diagonal_view_name(camera_side: GroundVector, front: str) -> str:
    """Name of an isometric (diagonal) camera position, e.g. 'front-left' = in front of the figure, on the
    figure's own left. camera_side is (sign x, sign y) of the camera seen from the figure."""
    facing = FRONT_VECTORS[front]
    left = figure_left_vector(front)
    along_front = camera_side[0] * facing[0] + camera_side[1] * facing[1]
    along_left = camera_side[0] * left[0] + camera_side[1] * left[1]
    return f"{'front' if along_front > 0 else 'rear'}-{'left' if along_left > 0 else 'right'}"
