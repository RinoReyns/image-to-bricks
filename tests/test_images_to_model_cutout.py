"""images_to_model step 1: cut-outs keep transparent pictures, the remover's largest region otherwise, and reject
pictures where the character covers too little or too much."""

import numpy as np
import support
from images_to_model_fakes import FakeRemover
from images_to_model_helpers import assert_raises, is_close
from PIL import Image

from image_to_bricks.images_to_model.cutout import (
    CutOutError,
    cut_out_image,
    has_transparent_pixels,
    keep_largest_region,
    measure_coverage,
)


def build_rgba_with_square(size: int = 100, square: int = 40) -> Image.Image:
    pixels = np.zeros((size, size, 4), dtype=np.uint8)
    start = (size - square) // 2
    pixels[start : start + square, start : start + square] = [200, 50, 50, 255]
    return Image.fromarray(pixels, "RGBA")


def test_picture_with_transparent_pixels_is_kept() -> None:
    remover = FakeRemover()
    image = build_rgba_with_square()
    cutout, info = cut_out_image(image, remover)
    assert remover.calls == 0
    assert info.method == "kept" and is_close(info.coverage, 0.16)
    assert np.array_equal(np.asarray(cutout), np.asarray(image))


def test_opaque_rgba_goes_through_the_remover() -> None:
    remover = FakeRemover()
    opaque = Image.new("RGBA", (80, 80), (10, 20, 30, 255))
    assert not has_transparent_pixels(opaque)
    cut_out_image(opaque, remover)
    assert remover.calls == 1


def test_remover_path_keeps_only_the_largest_region() -> None:
    cutout, info = cut_out_image(Image.new("RGB", (100, 100), (90, 90, 90)), FakeRemover())
    alpha = np.asarray(cutout)[..., 3]
    assert info.method == "rembg-u2net" and info.regions_before == 2
    assert alpha[0:3, 0:3].max() == 0 and alpha[50, 50] == 255
    assert 0.2 < info.coverage < 0.35
    assert info.to_json() == {"method": "rembg-u2net", "coverage": info.coverage, "regions_before": 2}


def test_largest_region_uses_8_connectivity() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = mask[1, 1] = mask[2, 2] = True
    mask[4, 4] = True
    kept = keep_largest_region(mask)
    assert kept.sum() == 3 and not kept[4, 4]


def test_too_small_or_too_large_character_is_rejected_with_advice() -> None:
    with assert_raises(CutOutError, "plain background"):
        cut_out_image(build_rgba_with_square(100, 9), FakeRemover())
    with assert_raises(CutOutError, "covers"):
        cut_out_image(build_rgba_with_square(100, 99), FakeRemover())


def test_coverage_counts_alpha_from_the_threshold_up() -> None:
    alpha = np.array([[127, 128], [0, 255]], dtype=np.uint8)
    assert measure_coverage(alpha) == 0.5


if __name__ == "__main__":
    support.run_tests(globals())
