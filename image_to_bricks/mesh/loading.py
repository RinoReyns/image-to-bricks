"""Loading the GLB and putting it into the build's coordinate system and scale."""

import numpy as np
import trimesh

from image_to_bricks.consts import PLATE_MM, STUD_MM


def load_mesh(glb_path: str) -> trimesh.Trimesh:
    """Load the first mesh of a GLB / mesh file with its scene-node transform applied (texture kept for the colour
    sampling). Raises ValueError for a file without a triangle mesh."""
    print("Loading mesh...")
    scene = trimesh.load(glb_path, process=False)
    # GLB meshes are commonly stored with their Blender-to-glTF orientation
    # on a scene node.  Selecting scene.geometry directly discards that node
    # transform and can mirror or rotate the voxelized model.  Apply it while
    # retaining the geometry's texture visual for later color sampling.
    if isinstance(scene, trimesh.Scene):
        geometry_nodes = list(scene.graph.nodes_geometry)
        if not geometry_nodes:
            raise ValueError("The GLB scene contains no mesh geometry.")
        node_transform, geometry_name = scene.graph.get(geometry_nodes[0])
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            raise ValueError(f"{glb_path}: the first geometry of the scene is not a triangle mesh.")
        mesh: trimesh.Trimesh = geometry.copy()
        mesh.apply_transform(np.asarray(node_transform, dtype=np.float64))
        return mesh
    if not isinstance(scene, trimesh.Trimesh):
        raise ValueError(f"{glb_path}: the file holds no triangle mesh.")
    single_mesh: trimesh.Trimesh = scene.copy()
    return single_mesh


def rotate_mesh(
    mesh: trimesh.Trimesh, rotate_x: float = 0.0, rotate_y: float = 0.0, rotate_z: float = 0.0
) -> trimesh.Trimesh:
    """Rotate a mesh around its centroid in X, then Y, then Z order.

    Angles are degrees in the post-import Blender-aligned coordinate frame.
    Keeping the pivot at the centroid prevents rotation from moving the model
    away from the origin before scale_mesh() places it on the LEGO grid.
    """
    print(f"Rotating mesh: X={rotate_x:g}°, Y={rotate_y:g}°, Z={rotate_z:g}°")
    mesh = mesh.copy()
    pivot = mesh.centroid
    for angle_degrees, axis in (
        (rotate_x, (1.0, 0.0, 0.0)),
        (rotate_y, (0.0, 1.0, 0.0)),
        (rotate_z, (0.0, 0.0, 1.0)),
    ):
        if angle_degrees:
            transform = trimesh.transformations.rotation_matrix(np.deg2rad(angle_degrees), axis, point=pivot)
            mesh.apply_transform(transform)
    return mesh


def scale_mesh(mesh: trimesh.Trimesh, target_studs_xy: int) -> trimesh.Trimesh:
    """Uniformly scale so the model's largest horizontal (x or y) extent
    equals target_studs_xy studs (target_studs_xy * STUD_MM millimeters).
    Also pre-stretches Z by STUD_MM/PLATE_MM so that a single isotropic
    voxelization pass (pitch = STUD_MM) produces correctly proportioned
    plate-height layers in Z."""
    print(f"Scaling to {target_studs_xy} studs wide...")
    mesh = mesh.copy()
    extents = mesh.extents  # [x, y, z] in mesh's native units
    xy_extent = max(extents[0], extents[1])
    if xy_extent <= 0:
        raise ValueError("Degenerate mesh: zero horizontal extent.")

    target_mm = target_studs_xy * STUD_MM
    uniform_scale = target_mm / xy_extent

    z_stretch = STUD_MM / PLATE_MM  # anisotropic Z pre-stretch
    transform = np.diag([uniform_scale, uniform_scale, uniform_scale * z_stretch, 1.0])
    mesh.apply_transform(transform)
    mesh.apply_translation(-mesh.bounds[0])  # sit on origin
    return mesh
