"""Run the model step over several cases, loading the models once (used by scripts/reproduce_images_to_model.sh).

    python -m tools.images_to_model.run_cases --out repro_out/runs --case robot=tests/fixtures/public/characters/robot

Writes <out>/<case>.glb, <out>/<case>_work/ (cut-outs, meshes, report.json) and <out>/run_summary.json.
"""

import argparse
import json
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

from image_to_bricks.images_to_model.engines import Engines
from image_to_bricks.images_to_model.pipeline import DEFAULT_SEED, ModelOutput, ModelSettings, run_pipeline
from image_to_bricks.images_to_model.provenance import ENGINE_NAME, list_restricted_modules
from image_to_bricks.images_to_model.type_aliases import JsonDict
from image_to_bricks.images_to_model.views import find_views_in_folder

LOAD_TIME_DIGITS = 1
DEFAULT_HEIGHT_METRES = 1.0


def parse_pairs(items: Sequence[str]) -> dict[str, str]:
    """NAME=VALUE items -> {name: value}. Raises SystemExit for an item without both parts."""
    pairs = {}
    for item in items:
        name, _, value = item.partition("=")
        if not name or not value:
            raise SystemExit(f"expected NAME=PATH, got {item!r}")
        pairs[name] = value
    return pairs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", required=True, help="NAME=PICTURES_FOLDER, repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT_METRES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def run_case(name: str, pictures_folder: str, engines: Engines, arguments: argparse.Namespace) -> JsonDict:
    """One case; a failure is recorded (with the traceback printed) and the GPU memory freed, so later cases run."""
    from image_to_bricks.images_to_model.trellis_engine import release_gpu_memory  # noqa: PLC0415

    output = ModelOutput(arguments.out / f"{name}.glb", arguments.out / f"{name}_work")
    try:
        report = run_pipeline(
            find_views_in_folder(Path(pictures_folder)),
            output,
            ModelSettings(arguments.height, seed=arguments.seed),
            engines,
        )
    except (RuntimeError, ValueError, OSError) as error:  # recorded; the next case still runs
        traceback.print_exc()
        release_gpu_memory()
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    print(
        f"[{ENGINE_NAME}] {name}: {report['seconds']['total']} s, peak {report['peak_vram_gib']['max']} GiB, "
        f"faces {report['clean']['faces_out']}, watertight {report['clean']['watertight']}",
        flush=True,
    )
    return {
        "ok": True,
        "seconds": report["seconds"],
        "peak_vram_gib": report["peak_vram_gib"],
        "package": report["package"],
    }


def run_cases(argv: Sequence[str] | None = None) -> int:
    """Run every case; exit code 1 if any failed."""
    from image_to_bricks.images_to_model.trellis_engine import check_rmbg_disabled, load_engines  # noqa: PLC0415

    arguments = parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engines = load_engines()
    summary: JsonDict = {"engine": ENGINE_NAME, "load_seconds": round(time.perf_counter() - started, LOAD_TIME_DIGITS)}
    print(f"[{ENGINE_NAME}] models loaded in {summary['load_seconds']} s", flush=True)
    summary["cases"] = {
        name: run_case(name, folder, engines, arguments) for name, folder in parse_pairs(arguments.case).items()
    }
    summary["restricted_modules_loaded"] = list_restricted_modules()
    summary["rmbg_disabled"] = check_rmbg_disabled()
    print(
        f"[{ENGINE_NAME}] restricted modules loaded: {summary['restricted_modules_loaded'] or 'none'}; "
        f"RMBG-2.0 disabled: {summary['rmbg_disabled']}",
        flush=True,
    )
    (arguments.out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failures = sum(not case["ok"] for case in summary["cases"].values())
    print(f"[{ENGINE_NAME}] {failures} failed case(s); summary in {arguments.out / 'run_summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run_cases())
