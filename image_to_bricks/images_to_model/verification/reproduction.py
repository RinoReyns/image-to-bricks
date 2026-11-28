"""The numbers that define "our results" for the model step, and their comparison with the saved ones
(lock/images_to_model/expected_results.json)."""

import hashlib
import json
from pathlib import Path

from image_to_bricks.images_to_model.type_aliases import JsonDict

# metric -> allowed absolute difference
TOLERANCES = {
    "seen_silhouette_iou": 0.005,
    "unseen_silhouette_iou": 0.005,
    "seen_psnr": 0.25,
    "unseen_psnr": 0.25,
    "seen_ssim": 0.005,
    "unseen_ssim": 0.005,
    "chamfer_l1": 0.001,
    "fscore_0.02": 0.005,
    "normal_consistency": 0.005,
    "volume_iou": 0.005,
    "lego_volume_iou": 0.01,
}
FACE_TOLERANCE = 0.01  # relative
EXACT = ("watertight", "pieces", "usage", "restricted_modules_loaded", "rmbg_disabled")


def hash_bytes(path: Path) -> str:
    """sha256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_case(case: str, run_dir: Path, row: JsonDict, summary: JsonDict) -> JsonDict:
    """Saved numbers of one case: the metrics of its comparison row, mesh checks and licence status from its
    report.json, the run summary's restricted-module checks, the geometry hash and the time."""
    work = Path(run_dir) / f"{case}_work"
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    result = {name: row.get(name) for name in TOLERANCES}
    result.update(
        {
            "faces": report["clean"]["faces_out"],
            "watertight": report["clean"]["watertight"],
            "pieces": report["clean"]["pieces_out"],
            "usage": report.get("package", {}).get("usage"),
            "restricted_modules_loaded": summary.get("restricted_modules_loaded"),
            "rmbg_disabled": summary.get("rmbg_disabled"),
            "geometry_sha256": hash_bytes(work / "cleaned_placed.ply"),
            "seconds": report["seconds"].get("total"),
        }
    )
    return result


def collect_results(run_dir: Path, compare_report: JsonDict, summary: JsonDict) -> JsonDict:
    """Saved numbers of every case in the comparison report."""
    return {row["case"]: collect_case(row["case"], run_dir, row, summary) for row in compare_report["rows"]}


def compare_value(name: str, expected: object, actual: object) -> tuple[bool, str]:
    """(within tolerance, text) for one number or flag."""
    if name in TOLERANCES:
        if not isinstance(expected, int | float) or not isinstance(actual, int | float):
            return expected == actual, f"{name}: expected {expected}, got {actual}"
        difference = abs(float(actual) - float(expected))
        text = f"{name}: expected {expected:.4f}, got {actual:.4f} (diff {difference:.4f}, allowed {TOLERANCES[name]})"
        return difference <= TOLERANCES[name], text
    if name == "faces" and isinstance(expected, int) and isinstance(actual, int):
        difference = abs(actual - expected) / max(expected, 1)
        return difference <= FACE_TOLERANCE, f"faces: expected {expected}, got {actual}"
    return expected == actual, f"{name}: expected {expected}, got {actual}"


def compare_results(expected: JsonDict, actual: JsonDict) -> tuple[bool, list[str]]:
    """(all within tolerance, one PASS / FAIL / INFO line per check). The geometry hash and the time are reported,
    not judged."""
    lines, all_ok = [], True
    for case in sorted(expected):
        if case not in actual:
            lines.append(f"FAIL [{case}] missing from this run")
            all_ok = False
            continue
        for name in [*TOLERANCES, "faces", *EXACT]:
            ok, text = compare_value(name, expected[case].get(name), actual[case].get(name))
            all_ok = all_ok and ok
            lines.append(f"{'PASS' if ok else 'FAIL'} [{case}] {text}")
        identical = expected[case].get("geometry_sha256") == actual[case].get("geometry_sha256")
        lines.append(
            f"INFO [{case}] geometry bit-identical: {'yes' if identical else 'no'}; "
            f"time {actual[case].get('seconds')} s (was {expected[case].get('seconds')} s)"
        )
    return all_ok, lines
