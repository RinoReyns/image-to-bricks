"""Score a model against the input pictures and a reference shape; markdown table and contact sheet of the scores.

Pure numpy / Pillow: loading GLB / PLY files happens in tools/images_to_model/compare_models.py.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.place import apply_placement, compute_placement
from image_to_bricks.images_to_model.type_aliases import BoolArray, FloatArray, JsonDict
from image_to_bricks.images_to_model.verification.metrics import (
    VIEW_SIZE,
    align_to_reference,
    compare_surfaces,
    compare_with_views,
    compute_lego_volume_iou,
    compute_volume_iou,
    normalise_view,
    summarise_views,
)
from image_to_bricks.images_to_model.view_geometry import split_rgba

TABLE_COLUMNS = (
    ("engine", "Engine"),
    ("shape_views", "Views used"),
    ("solid_rebuilt", "Rebuilt"),
    ("faces_out", "Faces"),
    ("watertight", "Watertight"),
    ("pieces_out", "Pieces"),
    ("seen_silhouette_iou", "Sil. IoU seen"),
    ("unseen_silhouette_iou", "Sil. IoU unseen"),
    ("seen_psnr", "PSNR seen"),
    ("unseen_psnr", "PSNR unseen"),
    ("seen_ssim", "SSIM seen"),
    ("unseen_ssim", "SSIM unseen"),
    ("reference", "Shape vs"),
    ("chamfer_l1", "Chamfer"),
    ("fscore_0.01", "F@1%"),
    ("fscore_0.02", "F@2%"),
    ("normal_consistency", "Normals"),
    ("volume_iou", "Vol. IoU"),
    ("lego_volume_iou", "LEGO IoU"),
    ("seconds_total", "Time s"),
    ("peak_vram_gib", "VRAM GiB"),
)
FIRST_SCORE_COLUMN = 6  # the mean table starts at the first score column
SMALL_NUMBER_LIMIT = 10  # below: 3 decimals, above: 1
EMPTY_CELL = "-"
SHEET_LABEL_HEIGHT = 18  # pixels above every tile
SHEET_LABEL_WIDTH = 110  # pixels left of the first tile
SHEET_TEXT_MARGIN = 4


@dataclass(frozen=True)
class ScoringInput:
    """One model of one case: the coloured mesh (texture as vertex colours [n, 3]), the cleaned watertight mesh used
    for geometry, and its run report."""

    name: str
    coloured: Mesh
    colours: FloatArray
    solid: Mesh
    report: JsonDict


def place_on_floor(mesh: Mesh, height: float = 1.0) -> Mesh:
    """The mesh on the floor, centred in x / z, `height` tall."""
    return Mesh(apply_placement(mesh.vertices, *compute_placement(mesh.vertices, height)), mesh.faces)


def score_geometry(mesh: Mesh, reference: Mesh) -> JsonDict:
    """Align (24 axis rotations), then Chamfer / F-score / normal consistency / volume IoU / LEGO-grid IoU."""
    aligned, rotation, _ = align_to_reference(mesh, reference)
    scores = compare_surfaces(aligned, reference)
    scores["volume_iou"] = compute_volume_iou(aligned, reference)
    scores["lego_volume_iou"] = compute_lego_volume_iou(aligned, reference)
    scores["alignment_rotation"] = rotation.astype(int).tolist()
    return scores


def describe_run(case: ScoringInput) -> JsonDict:
    """The table fields that come from the run report (mesh hygiene, time, memory)."""
    report = case.report
    clean = report.get("clean", {})
    return {
        "case": case.name,
        "engine": report.get("engine", "?"),
        "solid_rebuilt": clean.get("solid_rebuilt", False),
        "faces_out": clean.get("faces_out"),
        "watertight": clean.get("watertight"),
        "pieces_out": clean.get("pieces_out"),
        "raw_faces": clean.get("faces_in"),
        "pieces_removed": clean.get("pieces_removed"),
        "nonmanifold_edges_repaired": clean.get("nonmanifold_edges_repaired"),
        "holes_filled": clean.get("holes_filled"),
        "hole_faces_added": clean.get("hole_faces_added"),
        "seconds_total": report.get("seconds", {}).get("total"),
        "seconds_shape": report.get("seconds", {}).get("shape"),
        "seconds_texture": report.get("seconds", {}).get("texture"),
        "peak_vram_gib": report.get("peak_vram_gib", {}).get("max"),
    }


def score_output(
    case: ScoringInput, views: dict[str, Image.Image], reference: Mesh | None = None, reference_label: str | None = None
) -> tuple[JsonDict, dict[str, tuple[FloatArray, BoolArray]]]:
    """Every metric of one model on one case: views against the pictures (seen / unseen), geometry against the
    reference when given. Returns (table row, renders by direction)."""
    matched = compare_with_views(case.coloured, case.colours, views)
    seen = tuple(case.report.get("shape_views", views))
    row = describe_run(case)
    row.update({"shape_views": "+".join(seen), "assignment": matched.assignment, "views": matched.views})
    row.update(summarise_views(matched.views, seen))
    if reference is not None:
        row["reference"] = reference_label
        row.update(score_geometry(case.solid, reference))
    return row, matched.renders


def format_cell(value: object) -> str:
    """A table cell: dash for missing, yes / **no** for flags, 3 or 1 decimals for numbers."""
    if value is None:
        return EMPTY_CELL
    if isinstance(value, bool):
        return "yes" if value else "**no**"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < SMALL_NUMBER_LIMIT else f"{value:.1f}"
    return str(value)


def is_number(value: object) -> bool:
    """True for ints and floats, not for flags."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def build_case_table(case: str, rows: Sequence[JsonDict]) -> list[str]:
    """Markdown table of every model of one case."""
    lines = [f"## {case}", "", "| " + " | ".join(title for _, title in TABLE_COLUMNS) + " |"]
    lines.append("|" + "---|" * len(TABLE_COLUMNS))
    for row in (r for r in rows if r["case"] == case):
        lines.append("| " + " | ".join(format_cell(row.get(key)) for key, _ in TABLE_COLUMNS) + " |")
    return [*lines, ""]


def build_mean_table(rows: Sequence[JsonDict]) -> list[str]:
    """Markdown table of the mean score of every engine over the cases."""
    score_columns = [(key, title) for key, title in TABLE_COLUMNS[FIRST_SCORE_COLUMN:] if key != "reference"]
    lines = ["## Mean over cases", "", "| Engine | " + " | ".join(title for _, title in score_columns) + " |"]
    lines.append("|" + "---|" * (len(TABLE_COLUMNS) - FIRST_SCORE_COLUMN))
    for engine in dict.fromkeys(row["engine"] for row in rows):
        cells = []
        for key, _ in score_columns:
            values = [r[key] for r in rows if r["engine"] == engine and is_number(r.get(key))]
            cells.append(format_cell(float(np.mean(values))) if values else EMPTY_CELL)
        lines.append(f"| {engine} | " + " | ".join(cells) + " |")
    return lines


def build_markdown(rows: Sequence[JsonDict], notes: Sequence[str]) -> str:
    """report.md: the notes, one table per case, the mean table."""
    lines = ["# Model comparison", "", *(f"- {note}" for note in notes), ""]
    for case in dict.fromkeys(row["case"] for row in rows):
        lines += build_case_table(case, rows)
    return "\n".join(lines + build_mean_table(rows)) + "\n"


def convert_to_tile(rgb: FloatArray, mask: BoolArray, size: int) -> Image.Image:
    """A size x size tile of a view on white."""
    normal_rgb, normal_mask = normalise_view(rgb, mask, size)
    canvas = np.where(normal_mask[..., None], normal_rgb, 1.0)
    return Image.fromarray((canvas * 255).astype(np.uint8))


def build_contact_sheet(
    case: str,
    views: dict[str, Image.Image],
    renders: dict[str, tuple[FloatArray, BoolArray]],
    assignment: dict[str, str],
    size: int = VIEW_SIZE,
) -> Image.Image:
    """Two rows: the input pictures, then the model rendered from the matching directions."""
    names = list(views)
    sheet = Image.new("RGB", (size * len(names) + SHEET_LABEL_WIDTH, (size + SHEET_LABEL_HEIGHT) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, label in enumerate(("input", "model")):
        top = row_index * (size + SHEET_LABEL_HEIGHT)
        draw.text((6, top + size // 2), label, fill="black")
        for column, view in enumerate(names):
            picture = split_rgba(views[view]) if label == "input" else renders[assignment[view]]
            left = SHEET_LABEL_WIDTH + column * size
            sheet.paste(convert_to_tile(*picture, size), (left, top + SHEET_LABEL_HEIGHT))
            draw.text((left + SHEET_TEXT_MARGIN, top + 2), f"{case} {view}", fill="black")
    return sheet
