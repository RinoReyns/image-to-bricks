"""Score the models made by run_cases.py against the input pictures and the true shapes.

    python -m tools.images_to_model.compare_models --runs repro_out/runs --case robot \
        --reference robot=repro_out/truth/robot.glb --out repro_out/compare

Per case: silhouette IoU / PSNR / SSIM against each picture (split into pictures the shape model was given and
pictures it never saw), shape against the true shape, mesh hygiene, time and GPU memory. Writes <out>/report.md,
report.json and one contact sheet PNG per case.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from image_to_bricks.images_to_model.mesh import Mesh, convert_to_mesh
from image_to_bricks.images_to_model.type_aliases import FloatArray, JsonDict
from image_to_bricks.images_to_model.verification.comparison import (
    ScoringInput,
    build_contact_sheet,
    build_markdown,
    place_on_floor,
    score_output,
)
from image_to_bricks.images_to_model.views import VIEW_NAMES
from tools.images_to_model.run_cases import parse_pairs

REPORTED_SCORES = (
    "seen_silhouette_iou",
    "unseen_silhouette_iou",
    "seen_psnr",
    "unseen_psnr",
    "chamfer_l1",
    "fscore_0.02",
    "volume_iou",
)
NOTES = (
    "Silhouette IoU / PSNR / SSIM: each input cut-out vs the coloured model rendered (orthographic, vertex colours) "
    "from the matching side, both cropped to the character and scaled to the same height.",
    "'seen' = pictures the shape model got as input, 'unseen' = sides it had to invent.",
    "Shape metrics (cleaned meshes, aligned over 24 axis rotations, lengths in units of the figure height): against "
    "the true shape.",
)
COLOUR_SCALE = 255.0


def load_trimesh(path: Path) -> trimesh.Trimesh:
    """The file's geometry as one triangle mesh. Raises ValueError when it holds none."""
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{path} holds no triangle mesh")
    return loaded


def load_coloured_mesh(path: Path) -> tuple[Mesh, FloatArray]:
    """GLB -> (Mesh, per-vertex RGB 0-1): the stored vertex colours, or a texture sampled at the vertices."""
    loaded = load_trimesh(path)
    visual = loaded.visual
    if visual is None:
        raise ValueError(f"{path} has no colours")
    if hasattr(visual, "to_color"):
        visual = visual.to_color()
    colours = np.asarray(visual.vertex_colors, dtype=np.float64)[:, :3] / COLOUR_SCALE
    return convert_to_mesh(loaded), colours


def load_plain_mesh(path: Path) -> Mesh:
    """Any trimesh-readable mesh file as a Mesh."""
    return convert_to_mesh(load_trimesh(path))


def load_cutouts(work: Path) -> dict[str, Image.Image]:
    """The cut-outs a run saved, by view name."""
    views = {}
    for name in VIEW_NAMES:
        path = work / f"cutout_{name}.png"
        if path.is_file():
            with Image.open(path) as image:
                views[name] = image.convert("RGBA")
    return views


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="output folder of run_cases.py")
    parser.add_argument("--case", action="append", required=True, help="case name as given to run_cases.py")
    parser.add_argument("--reference", action="append", required=True, help="CASE=MESH true shape, repeatable")
    parser.add_argument("--out", type=Path, default=Path("compare_out"))
    return parser.parse_args(argv)


def score_case(case: str, arguments: argparse.Namespace, references: dict[str, str]) -> JsonDict:
    """Score one case, write its contact sheet, return its table row."""
    work = arguments.runs / f"{case}_work"
    views = load_cutouts(work)
    coloured, colours = load_coloured_mesh(arguments.runs / f"{case}.glb")
    solid = load_plain_mesh(work / "cleaned_placed.ply")
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    reference = place_on_floor(load_plain_mesh(Path(references[case])))
    row, renders = score_output(ScoringInput(case, coloured, colours, solid, report), views, reference, "truth")
    build_contact_sheet(case, views, renders, row["assignment"]).save(arguments.out / f"{case}_views.png")
    print(f"[{case}] " + ", ".join(f"{key}={row.get(key)}" for key in REPORTED_SCORES), flush=True)
    return row


def compare_cases(argv: Sequence[str] | None = None) -> int:
    """Score every case; exit code 1 if any could not be scored."""
    arguments = parse_args(argv)
    references = parse_pairs(arguments.reference)
    arguments.out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for case in arguments.case:
        try:
            rows.append(score_case(case, arguments, references))
        except (OSError, KeyError, ValueError) as error:
            failures.append(f"{case}: {type(error).__name__}: {error}")
            print("FAILED", failures[-1], flush=True)
    notes = [*NOTES, *(f"FAILED {item}" for item in failures)]
    (arguments.out / "report.md").write_text(build_markdown(rows, notes), encoding="utf-8")
    report = json.dumps({"rows": rows, "failures": failures}, indent=2, default=str)
    (arguments.out / "report.json").write_text(report, encoding="utf-8")
    print(f"\nreport: {arguments.out / 'report.md'}  ({len(failures)} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(compare_cases())
