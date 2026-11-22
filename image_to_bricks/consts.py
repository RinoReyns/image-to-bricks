"""Units and the colour palette (the single source of truth for colour ids, names, RGB and LDraw codes)."""

STUD_MM = 8.0  # stud pitch (one cell sideways)
PLATE_MM = 3.2  # plate height (one cell upwards)
BRICK_HEIGHT_PLATES = 3  # a brick is as tall as 3 plates
# Single source of truth for colours: id -> (name, RGB, LDraw colour code).
# RGB values follow LDraw LDConfig. All other tables are derived below so
# they can never disagree again.
PALETTE = {
    1: ("White", (242, 243, 242), 15),
    2: ("Red", (201, 26, 9), 4),
    3: ("Green", (35, 120, 65), 2),
    4: ("Blue", (0, 85, 191), 1),
    5: ("Yellow", (242, 205, 55), 14),
    6: ("Black", (27, 42, 52), 0),
    7: ("Light Bluish Gray", (160, 165, 169), 71),
    8: ("Dark Bluish Gray", (108, 110, 104), 72),
    9: ("Orange", (254, 138, 24), 25),
    10: ("Tan", (228, 205, 158), 19),
    11: ("Reddish Brown", (88, 42, 18), 70),
    12: ("Dark Brown", (53, 33, 0), 308),
    13: ("Dark Tan", (149, 138, 115), 28),
    14: ("Medium Nougat", (170, 125, 85), 84),
    15: ("Nougat", (208, 145, 104), 92),
    16: ("Bright Light Orange", (248, 187, 61), 191),
    17: ("Dark Red", (114, 14, 15), 320),
    18: ("Pearl Gold", (170, 127, 46), 297),
    19: ("Dark Orange", (169, 85, 0), 484),
    20: ("Medium Azure", (54, 174, 191), 322),
    21: ("Dark Azure", (7, 139, 201), 321),
    22: ("Olive Green", (119, 119, 78), 330),
    23: ("Bright Green", (75, 159, 74), 10),
    24: ("Light Nougat", (246, 215, 179), 78),
    25: ("Chrome Silver", (224, 224, 224), 383),
    26: ("Dark Pink", (223, 102, 149), 47),
    27: ("Bright Pink", (228, 173, 200), 29),
    28: ("Sand Green", (160, 188, 172), 378),
}

COLOR_NAMES = {cid: v[0] for cid, v in PALETTE.items()}
BLACK_ID = 6  # palette id of Black
LEGO_RGB = {cid: v[1] for cid, v in PALETTE.items()}
COLOR_MPL = {cid: "#{:02x}{:02x}{:02x}".format(*v[1]) for cid, v in PALETTE.items()}
# LEGO Black's real RGB looks navy on screen; render it as near-black.
COLOR_MPL[BLACK_ID] = "#151515"


def colour_name(colour_id: int | None) -> str:
    """Palette name of a colour id; "Color<id>" for an id outside the palette (reports never fail on it)."""
    if colour_id is None:
        return "ColorNone"
    return COLOR_NAMES.get(colour_id, f"Color{colour_id}")
