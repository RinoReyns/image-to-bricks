"""Build the true shapes of the test characters from their SDF scenes (marching cubes), the reference of the
shape-accuracy metrics in compare_models.py.

    python -m tools.images_to_model.build_truth tests/fixtures/public/characters --out repro_out/truth

Writes <out>/<name>.glb (y up, front = +z, unlit base colours as vertex colours): exactly the shapes the pictures
were rendered from.
"""

import argparse
import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import numpy as np
import trimesh

from image_to_bricks.images_to_model.clean import check_watertight, compute_volume, count_pieces
from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.type_aliases import AnyArray

DEFAULT_RESOLUTION = 192  # grid points along each axis of the scene cube
SAMPLES_PER_CHUNK = 2_000_000
COLOUR_SCALE = 255


class SceneRenderer(Protocol):
    """The parts of tests/fixtures/public/characters/sdf_render.py used here: make_primitive(...) -> shape,
    scene_distance(shapes, points) -> signed distances, scene_colour(shapes, points) -> RGB 0-1."""

    @property
    def make_primitive(self) -> Callable[..., object]:
        """Shape factory."""
        ...

    @property
    def scene_distance(self) -> Callable[..., AnyArray]:
        """Signed distance of every point."""
        ...

    @property
    def scene_colour(self) -> Callable[..., AnyArray]:
        """RGB 0-1 of every point."""
        ...


class Scene(Protocol):
    """A scene module: visible half extent, picture centre (x, y) and its shapes."""

    HALF_EXTENT: float
    CENTER: tuple[float, float]

    def build(self, make: Callable[..., object]) -> list[object]:
        """The shapes, in drawing order."""
        ...


def load_module(path: Path, name: str) -> ModuleType:
    """Import a module from its file. Raises ImportError when it cannot be loaded."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sample_distances(
    renderer: SceneRenderer, shapes: Sequence[object], bounds: tuple[AnyArray, AnyArray], resolution: int
) -> AnyArray:
    """Scene distance on a resolution^3 grid [x, y, z] between bounds (low, high corner), float32."""
    low, high = bounds
    axes = [np.linspace(low[i], high[i], resolution, dtype=np.float32) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    distances = np.empty(len(grid), dtype=np.float32)
    for start in range(0, len(grid), SAMPLES_PER_CHUNK):
        distances[start : start + SAMPLES_PER_CHUNK] = renderer.scene_distance(
            shapes, grid[start : start + SAMPLES_PER_CHUNK]
        )
    return distances.reshape(resolution, resolution, resolution)


def build_truth_mesh(renderer: SceneRenderer, scene: Scene, resolution: int) -> tuple[AnyArray, AnyArray, AnyArray]:
    """Marching cubes of the scene's distance field. Returns (vertices [n, 3], faces [m, 3], colours 0-255 [n, 3])."""
    measure = importlib.import_module("skimage.measure")
    shapes = scene.build(renderer.make_primitive)
    half, centre = scene.HALF_EXTENT, scene.CENTER
    low = np.array([-half, centre[1] - half, -half], dtype=np.float32)
    high = np.array([half, centre[1] + half, half], dtype=np.float32)
    distances = sample_distances(renderer, shapes, (low, high), resolution)
    spacing = tuple((high - low) / (resolution - 1))
    vertices, faces, _, _ = measure.marching_cubes(distances, level=0.0, spacing=spacing)
    vertices = vertices + low
    colours = renderer.scene_colour(shapes, vertices.astype(np.float32))
    return vertices, faces, (np.clip(colours, 0, 1) * COLOUR_SCALE).astype(np.uint8)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with sdf_render.py and scenes/*.py")
    parser.add_argument("--out", type=Path, required=True, help="folder for the <name>.glb true shapes")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    return parser.parse_args(argv)


def build_all(argv: Sequence[str] | None = None) -> int:
    """Build every scene's true shape."""
    arguments = parse_args(argv)
    renderer = cast(SceneRenderer, load_module(arguments.folder / "sdf_render.py", "sdf_render"))
    arguments.out.mkdir(parents=True, exist_ok=True)
    for scene_path in sorted((arguments.folder / "scenes").glob("*.py")):
        scene = cast(Scene, load_module(scene_path, f"scene_{scene_path.stem}"))
        vertices, faces, colours = build_truth_mesh(renderer, scene, arguments.resolution)
        mesh = Mesh(np.asarray(vertices, dtype=np.float64), faces.astype(np.int64))
        target = arguments.out / f"{scene_path.stem}.glb"
        trimesh.Trimesh(vertices, faces, vertex_colors=colours, process=False).export(target)
        print(
            f"{scene_path.stem}: {len(faces)} faces, pieces {count_pieces(mesh.faces)}, "
            f"watertight {check_watertight(mesh.faces)['watertight']}, volume {compute_volume(mesh):.4f}, "
            f"height {np.ptp(vertices[:, 1]):.3f} -> {target}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(build_all())
