"""Verification scores of one model: a perfect copy scores near perfect, a worse shape scores worse; markdown
table and contact sheet."""

import numpy as np
import support
from images_to_model_helpers import build_asymmetric_figure, render_input_view
from PIL import Image

from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.type_aliases import FloatArray
from image_to_bricks.images_to_model.verification.comparison import (
    ScoringInput,
    build_contact_sheet,
    build_markdown,
    format_cell,
    place_on_floor,
    score_output,
)


def build_case() -> tuple[Mesh, FloatArray, dict[str, Image.Image]]:
    mesh, colours = build_asymmetric_figure()
    views = {
        "front": render_input_view(mesh, colours, "+z"),
        "left": render_input_view(mesh, colours, "+x"),
        "back": render_input_view(mesh, colours, "-z"),
    }
    return place_on_floor(mesh), colours, views


def test_a_perfect_copy_scores_near_perfect_and_fills_the_report_fields() -> None:
    mesh, colours, views = build_case()
    report = {
        "engine": "fake",
        "shape_views": ["front"],
        "seconds": {"total": 3.0},
        "peak_vram_gib": {"max": 1.5},
        "clean": {"faces_out": 24, "watertight": True, "pieces_out": 2, "faces_in": 30},
    }
    row, renders = score_output(ScoringInput("demo", mesh, colours, mesh, report), views, mesh, "truth")
    assert row["seen_silhouette_iou"] > 0.9 and row["unseen_silhouette_iou"] > 0.9
    assert row["chamfer_l1"] < 0.02 and row["volume_iou"] > 0.9
    assert row["shape_views"] == "front" and row["reference"] == "truth"
    assert set(renders) == {"+z", "+x", "-z", "-x"}
    markdown = build_markdown([row, dict(row, engine="other", volume_iou=0.5)], ["note"])
    assert "## demo" in markdown and "| fake |" in markdown and "Mean over cases" in markdown
    sheet = build_contact_sheet("demo", views, renders, row["assignment"], size=64)
    assert sheet.size == (64 * 3 + 110, (64 + 18) * 2)


def test_a_worse_shape_scores_worse() -> None:
    mesh, colours, views = build_case()
    squashed = Mesh(mesh.vertices * np.array([1.6, 1.0, 1.0]), mesh.faces)
    report = {"shape_views": ["front", "left", "back"]}
    good, _ = score_output(ScoringInput("demo", mesh, colours, mesh, report), views, mesh, "truth")
    bad, _ = score_output(ScoringInput("demo", squashed, colours, squashed, report), views, mesh, "truth")
    assert bad["seen_silhouette_iou"] < good["seen_silhouette_iou"] - 0.1
    assert bad["volume_iou"] < good["volume_iou"] - 0.2 and bad["chamfer_l1"] > good["chamfer_l1"]
    assert good["unseen_silhouette_iou"] is None


def test_table_cells() -> None:
    assert [format_cell(value) for value in (None, True, False, 0.12345, 123.456, 7)] == [
        "-", "yes", "**no**", "0.123", "123.5", "7",
    ]  # fmt: skip


if __name__ == "__main__":
    support.run_tests(globals())
