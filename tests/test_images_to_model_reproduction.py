"""The saved results of the model step: the same numbers pass, small drift passes, large drift, a missing case or a
changed licence status fails."""

import json
from pathlib import Path

import support

from image_to_bricks.images_to_model.type_aliases import JsonDict
from image_to_bricks.images_to_model.verification.reproduction import collect_results, compare_results

METRICS = {
    "seen_silhouette_iou": 0.94,
    "unseen_silhouette_iou": 0.84,
    "seen_psnr": 24.9,
    "unseen_psnr": 20.0,
    "seen_ssim": 0.9,
    "unseen_ssim": 0.8,
    "chamfer_l1": 0.009,
    "fscore_0.02": 0.919,
    "normal_consistency": 0.9,
    "volume_iou": 0.8,
    "lego_volume_iou": 0.7,
}


def write_run(run_dir: Path, case: str, faces: int) -> None:
    work = run_dir / f"{case}_work"
    work.mkdir(parents=True)
    report = {
        "clean": {"faces_out": faces, "watertight": True, "pieces_out": 1},
        "package": {"usage": "permissive-with-attribution"},
        "seconds": {"total": 40.0},
    }
    (work / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (work / "cleaned_placed.ply").write_bytes(b"ply" + str(faces).encode())


def collect(name: str, faces: int, metrics: JsonDict) -> JsonDict:
    run_dir = Path(support.make_temp_dir(f"reproduction_{name}"))
    write_run(run_dir, "robot", faces)
    summary = {"restricted_modules_loaded": [], "rmbg_disabled": True}
    return collect_results(run_dir, {"rows": [dict(metrics, case="robot")]}, summary)


def test_the_same_numbers_pass() -> None:
    ok, lines = compare_results(collect("a", 39998, METRICS), collect("b", 39998, METRICS))
    assert ok and any("bit-identical: yes" in line for line in lines)


def test_small_drift_passes_large_drift_fails() -> None:
    expected = collect("a", 39998, METRICS)
    ok, lines = compare_results(expected, collect("b", 39990, dict(METRICS, seen_psnr=25.0)))
    assert ok and any("bit-identical: no" in line for line in lines)
    ok, lines = compare_results(expected, collect("c", 39998, dict(METRICS, **{"fscore_0.02": 0.90})))
    assert not ok and any(line.startswith("FAIL [robot] fscore_0.02") for line in lines)


def test_a_missing_case_or_a_changed_licence_fails() -> None:
    expected = collect("a", 39998, METRICS)
    assert not compare_results(expected, {})[0]
    changed = collect("b", 39998, METRICS)
    changed["robot"]["usage"] = "evaluation-only"
    assert not compare_results(expected, changed)[0]


def test_the_saved_results_cover_the_three_characters() -> None:
    path = Path(support.PROJECT_DIR) / "lock" / "images_to_model" / "expected_results.json"
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(expected) == ["bear", "frog", "robot"]
    assert all(case["usage"] == "permissive-with-attribution" for case in expected.values())


if __name__ == "__main__":
    support.run_tests(globals())
