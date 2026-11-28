"""Step 1: cut the character out of each picture (keep pictures that already have transparency)."""

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from PIL import Image
from scipy import ndimage

from image_to_bricks.images_to_model.type_aliases import AnyArray, BoolArray, JsonDict

ALPHA_THRESHOLD = 128  # alpha 0-255: at or above = part of the character
MIN_COVERAGE = 0.01  # share of the picture the character must cover at least
MAX_COVERAGE = 0.95  # ... and at most (more = the background was not removed)
EIGHT_NEIGHBOURS = np.ones((3, 3), dtype=bool)  # pixels touching at a corner belong to the same region


class CutOutError(ValueError):
    """The character covers too little or too much of the picture."""


class BackgroundRemover(Protocol):
    """Background removal model (rembg u2net in production, a fake in tests)."""

    def remove(self, image: Image.Image) -> Image.Image:
        """RGBA picture with the background made transparent."""
        ...


@dataclass(frozen=True)
class CutOutInfo:
    """How a picture was cut out: method "kept" (had transparency) or "rembg-u2net", share of the picture covered by
    the character, and the number of separate regions the remover found (None when kept)."""

    method: str
    coverage: float
    regions_before: int | None

    def to_json(self) -> JsonDict:
        """The info as written to report.json."""
        return asdict(self)


def has_transparent_pixels(image: Image.Image) -> bool:
    """True if any pixel is fully transparent."""
    alpha = np.asarray(image.convert("RGBA"))[..., 3]
    return bool((alpha == 0).any())


def keep_largest_region(mask: BoolArray) -> BoolArray:
    """Only the largest 8-connected True region of a boolean mask [rows, columns]."""
    labels, count = ndimage.label(mask, structure=EIGHT_NEIGHBOURS)
    if count == 0:
        return mask.copy()
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return np.asarray(labels == int(np.argmax(sizes)), dtype=bool)


def measure_coverage(alpha: AnyArray) -> float:
    """Share of pixels (0-1) whose alpha is at least ALPHA_THRESHOLD."""
    return float((alpha >= ALPHA_THRESHOLD).mean())


def remove_background(image: Image.Image, remover: BackgroundRemover) -> tuple[Image.Image, int]:
    """Run the remover and keep only its largest region. Returns (RGBA cut-out, number of regions it found)."""
    removed = np.array(remover.remove(image.convert("RGB")).convert("RGBA"))
    solid = removed[..., 3] >= ALPHA_THRESHOLD
    regions = int(ndimage.label(solid, structure=EIGHT_NEIGHBOURS)[1])
    largest = keep_largest_region(solid)
    removed[..., 3] = np.where(largest, removed[..., 3], 0)
    return Image.fromarray(removed, "RGBA"), regions


def cut_out_image(image: Image.Image, remover: BackgroundRemover) -> tuple[Image.Image, CutOutInfo]:
    """RGBA cut-out of one picture and how it was made. Raises CutOutError when the character covers less than
    MIN_COVERAGE or more than MAX_COVERAGE of the picture."""
    if has_transparent_pixels(image):
        cutout, method, regions = image.convert("RGBA"), "kept", None
    else:
        (cutout, regions), method = remove_background(image, remover), "rembg-u2net"
    coverage = measure_coverage(np.asarray(cutout)[..., 3])
    if coverage < MIN_COVERAGE or coverage > MAX_COVERAGE:
        raise CutOutError(
            f"the character covers {coverage:.2%} of the picture (allowed {MIN_COVERAGE:.0%}-{MAX_COVERAGE:.0%}): "
            "use a picture with a plain background, or one with a transparent background"
        )
    return cutout, CutOutInfo(method, round(coverage, 5), regions)
