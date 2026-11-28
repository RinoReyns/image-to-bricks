"""Plain triangle mesh plus a binary PLY writer (no trimesh needed)."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from image_to_bricks.images_to_model.type_aliases import FloatArray, IntArray


@dataclass
class Mesh:
    """Triangle mesh: vertices float [n, 3] (any length unit, y up), faces int [m, 3] (vertex indices)."""

    vertices: FloatArray
    faces: IntArray


class HasVerticesAndFaces(Protocol):
    """Anything with vertex and face arrays, e.g. a trimesh.Trimesh."""

    @property
    def vertices(self) -> ArrayLike:
        """Vertex positions [n, 3]."""
        ...

    @property
    def faces(self) -> ArrayLike:
        """Vertex indices of each triangle [m, 3]."""
        ...


def convert_to_mesh(source: HasVerticesAndFaces) -> Mesh:
    """Copy the vertices and faces of source (e.g. a trimesh.Trimesh) into a Mesh (float64 / int64)."""
    vertices = np.asarray(source.vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(source.faces, dtype=np.int64).reshape(-1, 3)
    return Mesh(vertices.copy(), faces.copy())


def combine_meshes(meshes: Sequence[Mesh]) -> Mesh:
    """Stack several meshes into one, re-indexing the faces of each mesh after the vertices before it."""
    vertex_blocks, face_blocks, offset = [], [], 0
    for mesh in meshes:
        vertex_blocks.append(mesh.vertices)
        face_blocks.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    return Mesh(np.concatenate(vertex_blocks), np.concatenate(face_blocks))


def write_ply(mesh: Mesh, path: Path) -> None:
    """Write mesh as a binary little-endian PLY (float32 positions; opens in Blender, MeshLab, trimesh). Creates the
    folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(mesh.vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(mesh.faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    face_records = np.zeros(len(mesh.faces), dtype=[("n", "u1"), ("idx", "<i4", (3,))])
    face_records["n"] = 3
    face_records["idx"] = mesh.faces
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(mesh.vertices.astype("<f4").tobytes())
        handle.write(face_records.tobytes())
