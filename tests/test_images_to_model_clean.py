"""images_to_model step 3: mesh clean-up counts, welding, debris, reduction, hole filling, normals, the solid
rebuild of broken surfaces and the full clean_mesh report."""

import numpy as np
import support
from images_to_model_fakes import refuse_to_simplify, simplify_by_dropping_faces
from images_to_model_helpers import (
    build_check_b_mesh,
    build_cube,
    build_icosphere,
    is_close,
    require_scikit_image,
    split_vertices_per_face,
)

from image_to_bricks.images_to_model.clean import (
    SolidRebuildSettings,
    check_watertight,
    clean_mesh,
    compute_volume,
    count_pieces,
    fill_holes,
    fix_normals,
    merge_vertices,
    needs_solid_rebuild,
    rebuild_as_solid,
    reduce_faces,
    remove_degenerate_faces,
    remove_duplicate_faces,
    remove_nonmanifold_faces,
    remove_small_pieces,
    repair_and_fill,
    sample_triangle_points,
)
from image_to_bricks.images_to_model.mesh import Mesh, combine_meshes


def test_check_b_counts_and_volume() -> None:
    _, report = clean_mesh(build_check_b_mesh(), 40000, refuse_to_simplify)
    assert report["faces_in"] == 5140 and report["faces_out"] == 5120
    assert report["duplicate_faces_removed"] == 1 and report["pieces_removed"] == 1
    assert report["holes_filled"] == 1 and report["hole_faces_added"] == 1
    assert report["watertight"] and report["winding_consistent"] and report["pieces_out"] == 1
    assert is_close(report["volume"], 4.1797, abs_tol=5e-4)


def test_merge_vertices_welds_a_split_cube() -> None:
    merged = merge_vertices(split_vertices_per_face(build_cube()))
    assert len(merged.vertices) == 8 and check_watertight(merged.faces)["watertight"]


def test_degenerate_faces_repeated_and_flat_are_dropped() -> None:
    cube = build_cube()
    vertices = np.vstack([cube.vertices, [[2, 0, 0], [3, 0, 0], [4, 0, 0]]])
    faces = np.vstack([cube.faces, [[0, 0, 1]], [[8, 9, 10]]])
    cleaned, removed = remove_degenerate_faces(Mesh(vertices, faces))
    assert removed == 2 and len(cleaned.faces) == 12 and len(cleaned.vertices) == 8


def test_duplicate_faces_are_found_in_any_vertex_order() -> None:
    cube = build_cube()
    faces = np.vstack([cube.faces, cube.faces[3][::-1], cube.faces[3]])
    cleaned, removed = remove_duplicate_faces(Mesh(cube.vertices, faces))
    assert removed == 2 and len(cleaned.faces) == 12


def test_small_pieces_go_and_the_largest_always_stays() -> None:
    big = build_icosphere(2, 1.0)  # 320 faces
    tiny = Mesh(build_icosphere(0, 0.1).vertices + 5, build_icosphere(0, 0.1).faces)  # 20 faces
    kept, pieces, faces = remove_small_pieces(combine_meshes([big, tiny]), 0.1)
    assert (pieces, faces, len(kept.faces)) == (1, 20, 320)
    alone, pieces, _ = remove_small_pieces(tiny, 0.5)
    assert pieces == 0 and len(alone.faces) == 20


def test_reduction_is_skipped_when_small_enough() -> None:
    mesh, attempts = reduce_faces(build_cube(), 12, refuse_to_simplify)
    assert attempts == 0 and len(mesh.faces) == 12


def test_reduction_retries_until_under_the_limit() -> None:
    mesh, attempts = reduce_faces(build_icosphere(3, 1.0), 100, simplify_by_dropping_faces)  # 1280 faces
    assert len(mesh.faces) <= 100 and attempts >= 2


def test_a_quad_hole_is_closed_with_a_fan() -> None:
    cube = build_cube()
    filled, holes, added = fill_holes(Mesh(cube.vertices, cube.faces[2:]))  # bottom face (2 triangles) removed
    assert (holes, added) == (1, 4)
    assert check_watertight(filled.faces)["winding_consistent"] and is_close(compute_volume(filled), 1.0)


def test_flipped_faces_and_inside_out_meshes_are_turned_outward() -> None:
    sphere = build_icosphere(2, 1.0)
    faces = sphere.faces.copy()
    faces[::7] = faces[::7][:, ::-1]
    fixed, flipped = fix_normals(Mesh(sphere.vertices, faces))
    assert flipped == len(faces[::7]) and check_watertight(fixed.faces)["winding_consistent"]
    assert compute_volume(fixed) > 4.0
    fixed, flipped = fix_normals(Mesh(sphere.vertices, sphere.faces[:, ::-1].copy()))
    assert flipped == len(sphere.faces) and compute_volume(fixed) > 4.0


def build_pinched_sphere() -> Mesh:
    """Two icospheres welded along one edge: that edge is used by four faces."""
    first, second = build_icosphere(2, 1.0), build_icosphere(2, 1.0)
    a, b = first.faces[0][:2]
    shift = 2.0 * (first.vertices[a] + first.vertices[b]) / 2.0
    moved = second.vertices.copy() * -1.0 + shift  # point reflection maps edge (a, b) onto itself
    return merge_vertices(combine_meshes([first, Mesh(moved, second.faces)]))


def test_repair_removes_faces_on_a_nonmanifold_edge_and_closes_the_gap() -> None:
    pinched = build_pinched_sphere()
    assert check_watertight(pinched.faces)["nonmanifold_edges"] == 1
    repaired, counts = repair_and_fill(pinched)
    assert counts["nonmanifold_edges_repaired"] == 1 and counts["nonmanifold_faces_removed"] == 4
    report = check_watertight(fix_normals(repaired)[0].faces)
    assert report["watertight"] and report["winding_consistent"] and report["nonmanifold_edges"] == 0


def test_a_clean_mesh_has_nothing_nonmanifold_to_remove() -> None:
    same, edges, removed = remove_nonmanifold_faces(build_cube())
    assert (edges, removed, len(same.faces)) == (0, 0, 12)


def test_pieces_and_watertight_report() -> None:
    two = combine_meshes([build_cube(), Mesh(build_cube().vertices + 3, build_cube().faces)])
    assert count_pieces(two.faces) == 2
    report = check_watertight(build_cube().faces[1:])
    assert not report["watertight"] and report["boundary_edges"] == 3


def test_clean_mesh_reports_every_count_with_reduction() -> None:
    _, report = clean_mesh(build_icosphere(3, 1.0), 300, simplify_by_dropping_faces)
    expected_keys = {
        "faces_in", "vertices_in", "vertices_after_merge", "degenerate_faces_removed", "duplicate_faces_removed",
        "small_pieces_removed_first", "small_piece_faces_removed_first", "faces_before_reduce", "reduce_attempts",
        "faces_after_reduce", "small_pieces_removed_second", "small_piece_faces_removed_second", "pieces_removed",
        "holes_filled", "hole_faces_added", "faces_flipped", "solid_rebuilt", "nonmanifold_edges_raw",
        "nonmanifold_edges_repaired", "nonmanifold_faces_removed", "faces_out", "vertices_out", "pieces_out",
        "watertight", "winding_consistent", "boundary_edges", "nonmanifold_edges", "volume",
    }  # fmt: skip
    assert expected_keys <= set(report)
    assert report["faces_before_reduce"] == 1280 and report["faces_after_reduce"] <= 300


def build_sheeted_sphere() -> Mesh:
    """Sphere with 200 inward fins: each fin is a new triangle on an existing edge, so that edge is used 3 times."""
    sphere = build_icosphere(3, 1.0)
    edges = sphere.faces[:200, :2]
    tips = sphere.vertices[sphere.faces[:200]].mean(axis=1) * 0.8
    tip_ids = np.arange(len(sphere.vertices), len(sphere.vertices) + 200)
    return Mesh(np.vstack([sphere.vertices, tips]), np.vstack([sphere.faces, np.column_stack([edges, tip_ids])]))


def test_solid_rebuild_only_for_many_nonmanifold_edges() -> None:
    decision, nonmanifold, _ = needs_solid_rebuild(build_sheeted_sphere())
    assert decision and nonmanifold > 100
    assert not needs_solid_rebuild(build_pinched_sphere())[0]  # 1 bad edge: repaired, not rebuilt
    assert not needs_solid_rebuild(build_check_b_mesh())[0]


def test_rebuild_as_solid_closes_a_holey_sphere() -> None:
    require_scikit_image()
    sphere = build_icosphere(4, 1.0)
    holey = Mesh(sphere.vertices, np.delete(sphere.faces, np.arange(0, len(sphere.faces), 17), axis=0))
    assert not check_watertight(holey.faces)["watertight"]
    solid, voxel = rebuild_as_solid(holey, SolidRebuildSettings(resolution=96))
    solid, _ = fix_normals(solid)
    assert check_watertight(solid.faces)["winding_consistent"] and count_pieces(solid.faces) == 1
    assert is_close(compute_volume(solid), 4.18, rel=0.05) and is_close(voxel, 2.0 / 96, rel=0.05)


def test_surface_samples_cover_large_triangles() -> None:
    big = Mesh(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))
    points = sample_triangle_points(big, 0.1)
    assert len(points) > 50 and points.min() >= -1e-9 and points.sum(axis=1).max() <= 1 + 1e-9


def test_clean_mesh_rebuilds_a_broken_mesh_and_reports_it() -> None:
    require_scikit_image()
    _, report = clean_mesh(build_sheeted_sphere(), 10**7, refuse_to_simplify)
    assert report["solid_rebuilt"] and report["nonmanifold_edges_raw"] > 100
    assert report["watertight"] and report["pieces_out"] == 1
    _, report_b = clean_mesh(build_check_b_mesh(), 40000, refuse_to_simplify)
    assert report_b["solid_rebuilt"] is False and report_b["faces_out"] == 5120


def build_open_tube() -> Mesh:
    """Closed box body with an open-ended tube (a 'leg') hanging below it, like TRELLIS.2's open limbs."""
    box = build_cube()
    body = Mesh(box.vertices * np.array([1.0, 0.6, 1.0]) + np.array([0, 0.4, 0]), box.faces)
    angles = np.linspace(0, 2 * np.pi, 25)[:-1]
    ring = np.column_stack([0.15 * np.cos(angles) + 0.5, np.zeros(24), 0.15 * np.sin(angles) + 0.5])
    tube_vertices = np.vstack([ring, ring + np.array([0, 0.38, 0])])
    i = np.arange(24)
    j = (i + 1) % 24
    tube_faces = np.vstack([np.column_stack([i, j, i + 24]), np.column_stack([j, j + 24, i + 24])])
    return combine_meshes([body, Mesh(tube_vertices, tube_faces)])


def test_rebuild_keeps_open_parts_only_with_keep_open() -> None:
    require_scikit_image()
    mesh = build_open_tube()
    kept, _ = rebuild_as_solid(mesh, SolidRebuildSettings(resolution=64, keep_open=True))
    dropped, _ = rebuild_as_solid(mesh, SolidRebuildSettings(resolution=64, keep_open=False))
    assert kept.vertices[:, 1].min() < 0.05  # the tube reaches down to y=0
    assert dropped.vertices[:, 1].min() > 0.3  # without keep_open only the body survives
    assert check_watertight(fix_normals(kept)[0].faces)["watertight"]


if __name__ == "__main__":
    support.run_tests(globals())
