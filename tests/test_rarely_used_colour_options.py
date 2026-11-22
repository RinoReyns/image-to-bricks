"""Options used rarely but still offered on the command line: --carve-gap (gap between a cape and the body),
--color-groups 0 (region growing instead of k-means) and --harmonize (one colour per 3-plate band column).
Run: python tests/test_rarely_used_colour_options.py"""

from dataclasses import replace

import numpy as np
import support

from image_to_bricks.colours import cape_carve, palette_matching
from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.options import ColourOptions
from image_to_bricks.packing import layer_packer

COLOUR_ID_BY_NAME = {name: colour_id for colour_id, name in COLOR_NAMES.items()}
BLACK = COLOUR_ID_BY_NAME["Black"]
ORANGE = COLOUR_ID_BY_NAME["Orange"]
RED = COLOUR_ID_BY_NAME["Red"]
BLUE = COLOUR_ID_BY_NAME["Blue"]
LAYER_COUNT, DEPTH, WIDTH = 10, 8, 4
CAPE_SETTINGS = dict(CAPE_THICKNESS=1, CAPE_GAP=1, CAPE_MIN_BODY=2)
WHOLE_HEIGHT_BLACK_CAPE = ColourOptions(
    carve_gap=True, carve_colour_id=BLACK, carve_from_height=0.0, carve_to_height=1.0
)


def make_body_with_cape() -> tuple:
    """Solid block (z, y, x) = (10, 8, 4): orange body whose high-y side is a one-cell black cape fused to it."""
    grid = np.ones((LAYER_COUNT, DEPTH, WIDTH), bool)
    lego = np.full(grid.shape, ORANGE, np.uint8)
    lego[:, -1, :] = BLACK
    return grid, lego


def carve(grid: np.ndarray, lego: np.ndarray) -> tuple:
    with support.module_settings(cape_carve, **CAPE_SETTINGS), support.quiet_stdout():
        return cape_carve.carve_cape_gap(grid, lego, colour=WHOLE_HEIGHT_BLACK_CAPE)


def test_carve_finds_the_cape_side_from_its_colour():
    grid, lego = make_body_with_cape()
    (axis, sign), black_share = cape_carve._back_direction(grid, lego, 0, LAYER_COUNT, BLACK)
    assert (axis, sign) == (1, 1), "the cape is on the high-y side (axis 1, +)"
    assert black_share == 1.0


def test_carve_removes_one_cell_between_cape_and_body():
    grid, lego = make_body_with_cape()
    carved_grid, carved_lego = carve(grid, lego)
    assert carved_grid[:, -1, :].all(), "the cape stays"
    assert not carved_grid[:, -2, :].any(), "the gap behind the cape is removed"
    assert carved_grid[:, :-2, :].all(), "the body stays"
    assert (carved_lego[:, -2, :] == 0).all(), "removed cells have no colour"


def test_carve_recolours_newly_exposed_body_cells_from_the_body():
    grid, lego = make_body_with_cape()
    _, carved_lego = carve(grid, lego)
    assert (carved_lego[:, -3, :] == ORANGE).all(), "the body face behind the gap is not painted black"


def test_carve_keeps_layers_outside_the_height_range():
    grid, lego = make_body_with_cape()
    upper_half = replace(WHOLE_HEIGHT_BLACK_CAPE, carve_from_height=0.5)
    with support.module_settings(cape_carve, **CAPE_SETTINGS), support.quiet_stdout():
        carved_grid, _ = cape_carve.carve_cape_gap(grid, lego, colour=upper_half)
    assert carved_grid[: LAYER_COUNT // 2].all(), "layers below FROM are untouched"
    assert not carved_grid[LAYER_COUNT // 2 :, -2, :].any()


def test_carve_does_not_cut_a_body_that_is_too_thin_behind_the_cape():
    grid, lego = make_body_with_cape()
    thin_grid, thin_lego = grid[:, -3:, :].copy(), lego[:, -3:, :].copy()  # cape + only 2 body cells
    carved_grid, _ = carve(thin_grid, thin_lego)
    assert carved_grid.all(), "CAPE_MIN_BODY is not met, nothing is carved"


def test_region_growing_gives_each_region_one_lego_colour():
    colours = np.zeros((2, 4, 6, 3), np.uint8)
    colours[:, :, :3] = (200, 30, 30)
    colours[:, :, 3:] = (30, 60, 200)
    colours[0, 0, 0] = (205, 35, 28)  # slightly different red inside the red region
    mask = np.ones((2, 4, 6), bool)
    lego = palette_matching.map_regions_to_lego(colours, mask, color_threshold=35, min_region_size=6)
    assert (lego[:, :, :3] == RED).all() and (lego[:, :, 3:] == BLUE).all(), np.unique(lego)


def test_harmonize_gives_a_full_band_column_its_majority_colour():
    grid = np.ones((3, 1, 2), bool)
    colours = np.array([[[RED, RED]], [[BLUE, RED]], [[RED, BLUE]]], np.uint8)
    with support.quiet_stdout():
        harmonised = layer_packer.harmonize_bands(grid, colours)
    assert (harmonised[:, 0, 0] == RED).all(), "top and bottom agree -> their colour wins"
    assert (harmonised[:, 0, 1] == RED).all(), "otherwise the middle layer's colour wins"
    assert colours[1, 0, 0] == BLUE, "the input is not changed"


def test_harmonize_leaves_columns_that_do_not_fill_the_band():
    grid = np.ones((3, 1, 1), bool)
    grid[2, 0, 0] = False
    colours = np.array([[[RED]], [[BLUE]], [[0]]], np.uint8)
    with support.quiet_stdout():
        harmonised = layer_packer.harmonize_bands(grid, colours)
    assert harmonised[0, 0, 0] == RED and harmonised[1, 0, 0] == BLUE


if __name__ == "__main__":
    support.run_tests(globals())
