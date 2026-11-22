"""Write robot.glb, frog.glb and bear.glb (vertex-coloured, glTF Y-up) from figure_shapes.py.

Only needed when a figure shape changes; the GLB files are committed. Needs scikit-image (marching cubes) on top of
the project's requirements. Run from the project folder:
    python tests/fixtures/public/make_public_fixtures.py
After regenerating, update the golden files: python tests/test_golden_builds.py --update-golden
"""

import os
import sys

import numpy as np
import trimesh
from skimage.measure import marching_cubes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figure_shapes import FIGURE_BUILDERS, figure_distance_and_colour

SAMPLE_SPACING = 0.025  # figure units between distance samples (the figures are about 2 units tall)
BOUNDING_PADDING = 0.1  # empty margin around the figure in the sample grid
FIGURE_BOUNDS = ((-0.95, -0.05, -0.75), (0.95, 2.3, 0.75))  # (min xyz, max xyz) that holds every figure


def sample_grid_points() -> tuple:
    """(points [n, 3], grid shape) of a regular grid over FIGURE_BOUNDS plus BOUNDING_PADDING."""
    lower = np.asarray(FIGURE_BOUNDS[0]) - BOUNDING_PADDING
    upper = np.asarray(FIGURE_BOUNDS[1]) + BOUNDING_PADDING
    axes = [np.arange(low, high + SAMPLE_SPACING, SAMPLE_SPACING) for low, high in zip(lower, upper)]
    mesh_x, mesh_y, mesh_z = np.meshgrid(*axes, indexing="ij")
    points = np.stack([mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()], axis=1)
    return points, mesh_x.shape, lower


def mesh_figure(name: str) -> trimesh.Trimesh:
    """Marching-cubes surface of the figure with the colour of the nearest primitive on every vertex."""
    primitives = FIGURE_BUILDERS[name]()
    points, grid_shape, lower = sample_grid_points()
    distance, _ = figure_distance_and_colour(primitives, points)
    vertices, faces, _, _ = marching_cubes(distance.reshape(grid_shape), level=0.0, spacing=(SAMPLE_SPACING,) * 3)
    vertices += lower
    _, vertex_colours = figure_distance_and_colour(primitives, vertices)
    opaque_colours = np.hstack([vertex_colours, np.full((len(vertex_colours), 1), 255, np.uint8)])
    # Colours go in with the vertices: processing merges duplicate vertices and must keep the colours in step.
    return trimesh.Trimesh(vertices=vertices, faces=faces[:, ::-1], vertex_colors=opaque_colours, process=True)


def main() -> None:
    for name in FIGURE_BUILDERS:
        glb_path = os.path.join(HERE, f"{name}.glb")
        figure_mesh = mesh_figure(name)
        figure_mesh.export(glb_path)
        print(
            f"{glb_path}: {len(figure_mesh.faces)} triangles, watertight={figure_mesh.is_watertight}, "
            f"{os.path.getsize(glb_path) / 1e6:.2f} MB"
        )


if __name__ == "__main__":
    main()
