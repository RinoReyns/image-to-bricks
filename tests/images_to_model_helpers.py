"""Shared helpers of the images_to_model tests: numpy-only test meshes (same geometry as trimesh.creation.icosphere),
synthetic pictures rendered from them, and small assertion helpers (the tests run without pytest)."""

import contextlib
import math
import os
from collections.abc import Iterator

import numpy as np
import support
from PIL import Image

from image_to_bricks.images_to_model.mesh import Mesh, combine_meshes
from image_to_bricks.images_to_model.type_aliases import FloatArray
from image_to_bricks.images_to_model.verification.metrics import render_vertex_colours

GOLDEN_RATIO = (1.0 + 5.0**0.5) / 2.0
ICOSAHEDRON_VERTICES = np.array(
    [
        [-1, GOLDEN_RATIO, 0], [1, GOLDEN_RATIO, 0], [-1, -GOLDEN_RATIO, 0], [1, -GOLDEN_RATIO, 0],
        [0, -1, GOLDEN_RATIO], [0, 1, GOLDEN_RATIO], [0, -1, -GOLDEN_RATIO], [0, 1, -GOLDEN_RATIO],
        [GOLDEN_RATIO, 0, -1], [GOLDEN_RATIO, 0, 1], [-GOLDEN_RATIO, 0, -1], [-GOLDEN_RATIO, 0, 1],
    ],
    dtype=np.float64,
)  # fmt: skip
ICOSAHEDRON_FACES = np.array(
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
        [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
        [8, 6, 7], [9, 8, 1],
    ],
    dtype=np.int64,
)  # fmt: skip
CUBE_VERTICES = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64
)
CUBE_FACES = np.array(
    [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5],
     [0, 4, 7], [0, 7, 3]],
    dtype=np.int64,
)  # fmt: skip
INPUT_VIEW_SIZE = 200


def subdivide_mesh(mesh: Mesh) -> Mesh:
    """Every triangle split into 4 (edge midpoints added)."""
    faces = mesh.faces
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    midpoints = mesh.vertices[unique].mean(axis=1)
    middle = inverse.reshape(3, -1).T + len(mesh.vertices)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    ab, bc, ca = middle[:, 0], middle[:, 1], middle[:, 2]
    new_faces = np.concatenate(
        [
            np.column_stack([a, ab, ca]),
            np.column_stack([ab, b, bc]),
            np.column_stack([ca, bc, c]),
            np.column_stack([ab, bc, ca]),
        ]
    )
    return Mesh(np.concatenate([mesh.vertices, midpoints]), new_faces)


def project_to_sphere(vertices: FloatArray, radius: float) -> FloatArray:
    """Vertices pushed onto a sphere around the origin."""
    return np.asarray(vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius)


def build_icosphere(subdivisions: int, radius: float) -> Mesh:
    """Icosphere with 20 * 4^subdivisions faces."""
    mesh = Mesh(project_to_sphere(ICOSAHEDRON_VERTICES, radius), ICOSAHEDRON_FACES.copy())
    for _ in range(subdivisions):
        mesh = subdivide_mesh(mesh)
        mesh = Mesh(project_to_sphere(mesh.vertices, radius), mesh.faces)
    return mesh


def build_check_b_mesh() -> Mesh:
    """Icosphere(4, r=1) minus face 0, plus face 5 again, plus icosphere(0, r=0.05) at x=2."""
    big = build_icosphere(4, 1.0)
    faces = np.concatenate([np.delete(big.faces, 0, axis=0), big.faces[5:6]])
    small = build_icosphere(0, 0.05)
    moved = Mesh(small.vertices + np.array([2.0, 0.0, 0.0]), small.faces)
    return combine_meshes([Mesh(big.vertices, faces), moved])


def build_cube() -> Mesh:
    """Unit cube, outward faces."""
    return Mesh(CUBE_VERTICES.copy(), CUBE_FACES.copy())


def split_vertices_per_face(mesh: Mesh) -> Mesh:
    """Give every face its own three vertices (like a GLB with UV seams everywhere)."""
    return Mesh(mesh.vertices[mesh.faces].reshape(-1, 3), np.arange(3 * len(mesh.faces)).reshape(-1, 3))


def build_box(width: float, height: float, depth: float) -> Mesh:
    """Box standing on y = 0, centred in x and z."""
    cube = build_cube()
    vertices = (cube.vertices - 0.5) * np.array([width, height, depth]) + np.array([0, height / 2, 0])
    return Mesh(vertices, cube.faces)


def colour_by_height(mesh: Mesh) -> FloatArray:
    """Red at the top to green at the bottom."""
    heights = (mesh.vertices[:, 1] - mesh.vertices[:, 1].min()) / np.ptp(mesh.vertices[:, 1])
    return np.column_stack([heights, 1 - heights, np.full(len(heights), 0.5)])


def build_asymmetric_figure() -> tuple[Mesh, FloatArray]:
    """Tall box with a block sticking out of its +z face, coloured by height, the block red."""
    body = build_box(0.5, 1.0, 0.3)
    nose = build_box(0.2, 0.2, 0.3)
    nose = Mesh(nose.vertices + np.array([0, 0.5, 0.25]), nose.faces)
    mesh = Mesh(np.vstack([body.vertices, nose.vertices]), np.vstack([body.faces, nose.faces + 8]))
    colours = colour_by_height(mesh)
    colours[8:] = [1.0, 0.0, 0.0]
    return mesh, colours


def render_input_view(mesh: Mesh, colours: FloatArray, direction: str) -> Image.Image:
    """RGBA picture (transparent background) of a coloured mesh seen from direction."""
    rgb, mask = render_vertex_colours(mesh, colours, direction, INPUT_VIEW_SIZE)
    rgba = np.dstack([(rgb * 255).astype(np.uint8), (mask * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def render_four_views(mesh: Mesh, colours: FloatArray) -> dict[str, Image.Image]:
    """front / left / back / right pictures of a mesh facing +z."""
    directions = {"front": "+z", "left": "+x", "back": "-z", "right": "-x"}
    return {name: render_input_view(mesh, colours, direction) for name, direction in directions.items()}


def is_close(actual: float, expected: float, *, rel: float = 1e-6, abs_tol: float = 1e-12) -> bool:
    """actual equals expected up to a relative or an absolute tolerance."""
    return math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=abs_tol)


@contextlib.contextmanager
def assert_raises(expected_error: type[BaseException], message_part: str = "") -> Iterator[None]:
    """The block must raise expected_error whose text contains message_part."""
    try:
        yield
    except expected_error as error:
        assert message_part in str(error), f"{expected_error.__name__} without {message_part!r}: {error}"
        return
    raise AssertionError(f"expected {expected_error.__name__}")


def require_scikit_image() -> None:
    """Skip the test without scikit-image (marching cubes)."""
    try:
        import skimage.measure  # noqa: F401, PLC0415
    except ImportError as error:
        raise support.SkipTest("scikit-image is not installed (pip install -r requirements-dev.txt)") from error


@contextlib.contextmanager
def environment_variable(name: str, value: str) -> Iterator[None]:
    """Set an environment variable for the block, restore it afterwards."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            del os.environ[name]
        else:
            os.environ[name] = previous
