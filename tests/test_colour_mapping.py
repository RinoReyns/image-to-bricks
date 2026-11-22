"""Colour mapping on voxel colours (no mesh): nearest palette colour, colour boost, k-means groups -> one LEGO colour,
dark-neutral rule, palette limit, majority smoothing, small-region absorb, overrides, barycentric weights,
denoise, per-layer unify, fill from nearest. Run: python tests/test_colour_mapping.py"""

import numpy as np
import support

from image_to_bricks.colours import colour_sampling, palette_matching, surface_cleanup
from image_to_bricks.consts import COLOR_NAMES

COLOUR_ID_BY_NAME = {name: colour_id for colour_id, name in COLOR_NAMES.items()}
RED = COLOUR_ID_BY_NAME["Red"]
DARK_RED = COLOUR_ID_BY_NAME["Dark Red"]
BLUE = COLOUR_ID_BY_NAME["Blue"]
YELLOW = COLOUR_ID_BY_NAME["Yellow"]
BLACK = COLOUR_ID_BY_NAME["Black"]
DARK_BLUISH_GRAY = COLOUR_ID_BY_NAME["Dark Bluish Gray"]

BRIGHT_RED_RGB = (200, 30, 30)
RED_IN_SHADOW_RGB = (150, 20, 20)
BLUE_RGB = (30, 60, 200)
DARK_GREY_RGB = (90, 90, 94)  # nearest palette colour is Dark Bluish Gray, but dark enough for the Black rule


def uniform_colours(shape_zyx: tuple, rgb: tuple) -> np.ndarray:
    """uint8 [z, y, x, 3] filled with one RGB colour."""
    return np.full(shape_zyx + (3,), rgb, np.uint8)


def cluster(
    colours: np.ndarray,
    mask: np.ndarray,
    group_count: int,
    allowed_ids: list[int] | None = None,
    max_colours: int = 8,
) -> np.ndarray:
    with support.quiet_stdout():
        return palette_matching.cluster_to_lego(colours, mask, group_count, allowed_ids, max_colours)


def test_nearest_lego_colour_finds_red_for_a_red_rgb():
    assert palette_matching.nearest_lego_color(np.array([BRIGHT_RED_RGB])) == RED


def test_nearest_lego_colour_only_returns_allowed_colours():
    chosen = palette_matching.nearest_lego_color(np.array([BRIGHT_RED_RGB]), allowed=[BLUE, YELLOW, DARK_RED])
    assert chosen == DARK_RED, COLOR_NAMES[chosen]


def test_nearest_lego_colour_without_a_list_uses_the_whole_palette():
    assert palette_matching.nearest_lego_color(np.array([BRIGHT_RED_RGB]), allowed=None) == RED
    assert palette_matching.nearest_lego_color(np.array([BRIGHT_RED_RGB]), allowed=[]) == RED


def test_enhance_colours_changes_only_masked_cells():
    colours = uniform_colours((1, 1, 2), (120, 90, 90))
    mask = np.array([[[True, False]]])
    enhanced = palette_matching.enhance_colors(colours, mask)
    assert tuple(enhanced[0, 0, 1]) == (120, 90, 90), "unmasked cells keep their colour"
    assert not np.array_equal(enhanced[0, 0, 0], colours[0, 0, 0])


def test_enhance_colours_boosts_saturation():
    colours = uniform_colours((1, 1, 1), (120, 90, 90))
    enhanced = palette_matching.enhance_colors(colours, np.ones((1, 1, 1), bool))
    original_spread = int(colours[0, 0, 0].max()) - int(colours[0, 0, 0].min())
    enhanced_spread = int(enhanced[0, 0, 0].max()) - int(enhanced[0, 0, 0].min())
    assert enhanced_spread > original_spread, (enhanced[0, 0, 0], colours[0, 0, 0])


def test_light_and_shadow_of_one_material_get_one_lego_colour():
    colours = uniform_colours((2, 4, 4), BRIGHT_RED_RGB)
    colours[:, :, 2:] = BLUE_RGB
    colours[0, 0, 0] = RED_IN_SHADOW_RGB
    mask = np.ones((2, 4, 4), bool)
    lego = cluster(colours, mask, group_count=2)
    assert lego[0, 0, 0] == lego[0, 0, 1] == RED, [COLOR_NAMES[colour_id] for colour_id in np.unique(lego)]
    assert lego[0, 0, 3] == BLUE


def test_cluster_colours_only_the_masked_cells():
    colours = uniform_colours((1, 2, 2), BRIGHT_RED_RGB)
    mask = np.array([[[True, True], [False, False]]])
    lego = cluster(colours, mask, group_count=2)
    assert (lego[mask] == RED).all()
    assert (lego[~mask] == 0).all(), "cells outside the mask stay 0 (no colour)"


def test_dark_neutral_group_becomes_black_not_grey():
    assert palette_matching.nearest_lego_color(np.array([DARK_GREY_RGB])) == DARK_BLUISH_GRAY
    lego = cluster(uniform_colours((1, 1, 2), DARK_GREY_RGB), np.ones((1, 1, 2), bool), group_count=1)
    assert lego[0, 0, 0] == BLACK


def test_dark_neutral_stays_grey_when_black_is_not_allowed():
    lego = cluster(
        uniform_colours((1, 1, 2), DARK_GREY_RGB),
        np.ones((1, 1, 2), bool),
        group_count=1,
        allowed_ids=[DARK_BLUISH_GRAY, RED],
    )
    assert lego[0, 0, 0] == DARK_BLUISH_GRAY


def test_palette_limit_keeps_only_the_most_used_colours():
    six_colours = [(200, 30, 30), (30, 60, 200), (250, 220, 40), (40, 160, 60), (240, 240, 240), (30, 30, 30)]
    colours = np.zeros((1, 3, len(six_colours), 3), np.uint8)
    for column, rgb in enumerate(six_colours):
        colours[0, :, column] = rgb
    mask = np.ones(colours.shape[:3], bool)
    lego = cluster(colours, mask, group_count=len(six_colours), max_colours=3)
    assert len(np.unique(lego[mask])) == 3


def test_cluster_is_deterministic():
    random_colours = np.random.default_rng(7).integers(0, 256, (3, 5, 5, 3), dtype=np.uint8)
    mask = np.ones((3, 5, 5), bool)
    assert np.array_equal(cluster(random_colours, mask, 6), cluster(random_colours, mask, 6))


def test_smoothing_recolours_a_lone_cell_surrounded_by_one_colour():
    lego = np.full((1, 3, 3), RED, np.uint8)
    lego[0, 1, 1] = BLUE
    smoothed = surface_cleanup.smooth_lego_colors(lego, np.ones((1, 3, 3), bool), iters=1, majority=4)
    assert smoothed[0, 1, 1] == RED
    assert lego[0, 1, 1] == BLUE, "the input array must not change"


def test_smoothing_keeps_a_cell_without_enough_agreeing_neighbours():
    lego = np.full((1, 3, 3), RED, np.uint8)
    lego[0, 1, 1] = BLUE
    smoothed = surface_cleanup.smooth_lego_colors(lego, np.ones((1, 3, 3), bool), iters=1, majority=5)
    assert smoothed[0, 1, 1] == BLUE, "only 4 in-plane neighbours exist, majority 5 cannot be reached"


def test_absorb_recolours_regions_below_the_minimum_size():
    lego = np.full((1, 4, 4), RED, np.uint8)
    lego[0, 0, 0] = BLUE
    lego[0, 2:4, 2:4] = YELLOW
    absorbed = surface_cleanup.absorb_small_color_regions_3d(lego, np.ones((1, 4, 4), bool), min_region_size=3)
    assert absorbed[0, 0, 0] == RED, "a 1-cell region is absorbed by its neighbours"
    assert (absorbed[0, 2:4, 2:4] == YELLOW).all(), "a 4-cell region is kept"


def test_colour_overrides_recolour_globally_and_inside_a_box():
    lego = np.full((2, 2, 2), RED, np.uint8)
    grid = np.ones((2, 2, 2), bool)
    globally = surface_cleanup.apply_color_overrides(lego, grid, global_overrides={RED: BLUE})
    assert (globally == BLUE).all()
    region = (slice(0, 1), slice(0, 2), slice(0, 2), YELLOW)
    boxed = surface_cleanup.apply_color_overrides(lego, grid, region_overrides=[region])
    assert (boxed[0] == YELLOW).all() and (boxed[1] == RED).all()


def test_barycentric_weights_at_corners_and_centre():
    corner_a, corner_b, corner_c = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    assert np.allclose(colour_sampling.barycentric(corner_a, corner_a, corner_b, corner_c), (1, 0, 0))
    assert np.allclose(colour_sampling.barycentric(corner_b, corner_a, corner_b, corner_c), (0, 1, 0))
    centre = (corner_a + corner_b + corner_c) / 3
    assert np.allclose(colour_sampling.barycentric(centre, corner_a, corner_b, corner_c), (1 / 3, 1 / 3, 1 / 3))


def test_barycentric_of_a_degenerate_triangle_uses_the_first_corner():
    point = np.array([0.0, 0, 0])
    assert np.allclose(colour_sampling.barycentric(point, point, point, point), (1, 0, 0))


def test_denoise_replaces_a_dull_cell_next_to_a_saturated_one():
    colours = uniform_colours((1, 1, 2), BRIGHT_RED_RGB)
    colours[0, 0, 1] = (128, 128, 128)
    denoised = surface_cleanup.denoise_voxel_colors(colours, np.ones((1, 1, 2), bool), iters=1)
    assert tuple(denoised[0, 0, 1]) == BRIGHT_RED_RGB


def test_unify_layer_gives_a_large_region_its_main_colour_and_keeps_small_ones():
    lego = np.full((1, 6, 6), RED, np.uint8)
    lego[0, 0, 0] = BLUE
    grid = np.zeros((1, 6, 6), bool)
    grid[0, :, :] = True
    unified = surface_cleanup.unify_layer_regions(lego, grid, min_region_size=20)
    assert (unified[grid] == RED).all()
    small_island = np.zeros((1, 6, 6), bool)
    small_island[0, 0, :2] = True
    kept = surface_cleanup.unify_layer_regions(lego, small_island, min_region_size=20)
    assert kept[0, 0, 0] == BLUE, "regions below min_region_size keep their colours"


def test_fill_from_nearest_copies_the_closest_source_value():
    values = np.zeros((1, 1, 5), np.uint8)
    values[0, 0, 0], values[0, 0, 4] = RED, BLUE
    source = np.zeros((1, 1, 5), bool)
    source[0, 0, [0, 4]] = True
    filled = colour_sampling.fill_from_nearest(values, source, np.ones((1, 1, 5), bool))
    assert list(filled[0, 0]) == [RED, RED, filled[0, 0, 2], BLUE, BLUE]
    assert filled[0, 0, 2] in (RED, BLUE)


def test_fill_from_nearest_without_sources_changes_nothing():
    values = np.full((1, 1, 3), RED, np.uint8)
    filled = colour_sampling.fill_from_nearest(values, np.zeros((1, 1, 3), bool), np.ones((1, 1, 3), bool))
    assert np.array_equal(filled, values)


if __name__ == "__main__":
    support.run_tests(globals())
