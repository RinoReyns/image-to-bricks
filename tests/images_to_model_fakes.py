"""Fakes for the engines images_to_model's pipeline gets injected (no GPU, no model weights)."""

import json
import struct

import numpy as np
from images_to_model_helpers import build_icosphere
from numpy.typing import ArrayLike
from PIL import Image

from image_to_bricks.images_to_model.engines import Engines
from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.type_aliases import FloatArray, IntArray, JsonDict

GIB = 1024**3
STRAY_ALPHA = 200  # a 3 x 3 blob in the corner the cut-out must drop
SOFT_ALPHA = 100  # one pixel below the alpha threshold


class FakeRemover:
    """Returns the picture with an alpha mask: a big disc plus a small blob."""

    def __init__(self, disc_radius_share: float = 0.3) -> None:
        self.calls = 0
        self.disc_radius_share = disc_radius_share

    def remove(self, image: Image.Image) -> Image.Image:
        """The picture with the disc opaque."""
        self.calls += 1
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        rows, cols = np.mgrid[0:height, 0:width]
        radius = self.disc_radius_share * min(height, width)
        disc = (rows - height / 2) ** 2 + (cols - width / 2) ** 2 < radius**2
        alpha = np.where(disc, 255, 0).astype(np.uint8)
        alpha[0:3, 0:3] = STRAY_ALPHA
        alpha[height // 2, 1] = SOFT_ALPHA
        return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


class FakeShapeGenerator:
    """A stretched icosphere (1280 faces) instead of TRELLIS.2; records its calls."""

    view_names: tuple[str, ...] = ("front", "left", "back", "right")

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def generate(self, images: dict[str, Image.Image], seed: int) -> Mesh:
        """An ellipsoid 2 units tall."""
        self.calls.append((sorted(images), seed))
        sphere = build_icosphere(3, 0.5)
        return Mesh(sphere.vertices * np.array([1.0, 2.0, 0.8]) + 0.3, sphere.faces)


class FakeFrontOnlyShapeGenerator(FakeShapeGenerator):
    """Like TRELLIS.2: uses the front picture only."""

    view_names: tuple[str, ...] = ("front",)


class FakeColouredModel:
    """A coloured model that writes the smallest valid GLB."""

    def __init__(self, mesh: Mesh) -> None:
        self.vertices = mesh.vertices.copy()
        self.faces = mesh.faces.copy()

    def export(self, path: str) -> None:
        """Write a minimal GLB."""
        with open(path, "wb") as handle:
            handle.write(build_minimal_glb())


class FakePainter:
    """Records the picture sizes it got; returns a FakeColouredModel."""

    def __init__(self) -> None:
        self.image_sizes: list[tuple[int, int]] = []
        self.result: FakeColouredModel | None = None
        self.last_info: JsonDict | None = {"front_direction": "+z"}

    def paint(self, mesh: Mesh, views: dict[str, Image.Image]) -> FakeColouredModel:
        """The mesh, uncoloured."""
        self.image_sizes = [image.size for image in views.values()]
        self.result = FakeColouredModel(mesh)
        return self.result


class FakeRescalingPainter(FakePainter):
    """Returns the mesh re-centred into a unit box (a painter that moves the mesh)."""

    def paint(self, mesh: Mesh, views: dict[str, Image.Image]) -> FakeColouredModel:
        """The mesh moved and scaled."""
        self.image_sizes = [image.size for image in views.values()]
        low, high = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
        moved = Mesh((mesh.vertices - (low + high) / 2) / (high - low).max(), mesh.faces.copy())
        self.result = FakeColouredModel(moved)
        return self.result


class FakeProbe:
    """Always reports the same peak memory."""

    def __init__(self, peak_bytes: int = 3 * GIB) -> None:
        self.resets = 0
        self.peak_bytes = peak_bytes

    def reset(self) -> None:
        """Count resets."""
        self.resets += 1

    def read_peak_bytes(self) -> int:
        """The fixed peak."""
        return self.peak_bytes


def simplify_by_dropping_faces(vertices: FloatArray, faces: IntArray, reduction: float) -> tuple[ArrayLike, ArrayLike]:
    """Fake reducer: keeps the first (1 - reduction) share of faces, plus one extra."""
    keep = int(len(faces) * (1.0 - reduction)) + 1
    return vertices, faces[:keep]


def simplify_to_coarser_ellipsoid(
    vertices: FloatArray, faces: IntArray, reduction: float
) -> tuple[ArrayLike, ArrayLike]:
    """Fake reducer that stays closed: a coarser icosphere stretched to the input's bounds."""
    target = len(faces) * (1.0 - reduction)
    level = 0
    while 20 * 4 ** (level + 1) <= target:
        level += 1
    sphere = build_icosphere(level, 1.0)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    return sphere.vertices * (high - low) / 2 + (high + low) / 2, sphere.faces


def refuse_to_simplify(vertices: FloatArray, faces: IntArray, reduction: float) -> tuple[ArrayLike, ArrayLike]:
    """A reducer that must not be called."""
    raise AssertionError(f"simplify must not be called ({len(vertices)} vertices, {len(faces)} faces, {reduction})")


def build_fake_engines(painter: FakePainter | None = None) -> Engines:
    """Engines made of fakes (the real ProjectionPainter unless a painter is given)."""
    engines = Engines(FakeRemover(), FakeShapeGenerator(), simplify_to_coarser_ellipsoid, FakeProbe())
    if painter is not None:
        engines.painter = painter
    return engines


def build_minimal_glb(binary: bytes = b"\x01\x02\x03\x04") -> bytes:
    """Smallest valid GLB: a JSON chunk with an asset and a 4-byte binary chunk."""
    document = json.dumps({"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}]}).encode()
    document += b" " * ((4 - len(document) % 4) % 4)
    json_chunk = struct.pack("<I4s", len(document), b"JSON") + document
    body = json_chunk + struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body
