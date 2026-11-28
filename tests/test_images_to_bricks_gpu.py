"""End to end on the GPU: the robot's four pictures -> TRELLIS.2 model -> LEGO build with instructions, through
images_to_bricks.py in a fresh process. Skipped unless the GPU setup is complete (scripts/setup_images_to_model.sh
writes .venv/IMAGES_TO_MODEL_OK and the model cache .models/). Run: python tests/test_images_to_bricks_gpu.py
(about 3 minutes; the numbers of the model step are checked by scripts/reproduce_images_to_model.sh)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import support

from image_to_bricks.instructions.outputs_handler import load_model

SETUP_MARKER = "IMAGES_TO_MODEL_OK"  # written by scripts/setup_images_to_model.sh into the venv
ROBOT_PICTURES = Path(support.PUBLIC_FIXTURES_DIR) / "characters" / "robot"
MAX_FACES = 40000
RUN_TIMEOUT_SECONDS = 1800
GRID_FRONT_BY_MODEL_DIRECTION = {"+z": "-y", "-z": "+y", "+x": "+x", "-x": "-x"}  # y-up model stood up by --rotate-x 90


def require_gpu_setup() -> None:
    """Skip unless this Python is the venv of a finished GPU setup and the model cache exists."""
    marker = Path(sys.prefix) / SETUP_MARKER
    if not marker.is_file():
        raise support.SkipTest(f"no GPU setup in this venv ({marker} missing: bash scripts/setup_images_to_model.sh)")
    if not (Path(support.PROJECT_DIR) / ".models").is_dir():
        raise support.SkipTest("no model cache .models/ (bash scripts/setup_images_to_model.sh)")


def run_images_to_bricks(outdir: Path) -> subprocess.CompletedProcess[str]:
    """images_to_bricks.py on the robot pictures, 12 studs wide on a 2-plate stand."""
    command = [
        sys.executable,
        os.path.join(support.PROJECT_DIR, "images_to_bricks.py"),
        "--views",
        str(ROBOT_PICTURES),
        "--outdir",
        str(outdir),
        "--",
        "--target-studs",
        "12",
        "--base",
        "2",
        "--save-model",
    ]
    return subprocess.run(
        command,
        cwd=support.PROJECT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=RUN_TIMEOUT_SECONDS,
        check=False,
    )


def test_robot_pictures_become_a_licensed_model_and_a_brick_build() -> None:
    require_gpu_setup()
    outdir = Path(support.make_temp_dir("gpu_images_to_bricks")) / "out"
    support.assert_cli_succeeded(run_images_to_bricks(outdir))
    report = json.loads((outdir / "model" / "robot_work" / "report.json").read_text(encoding="utf-8"))
    assert report["engine"] == "trellis-proj" and report["shape_views"] == ["front"]
    assert report["clean"]["watertight"] and report["clean"]["pieces_out"] == 1
    assert 0 < report["clean"]["faces_out"] <= MAX_FACES
    assert report["package"]["usage"] == "permissive-with-attribution"
    notice = (outdir / "model" / "robot_package" / "NOTICE.md").read_text(encoding="utf-8")
    assert (
        "Built with DINOv3" in notice
        and "Restricted (NVIDIA research-only) modules loaded during the run: none" in notice
    )
    bricks = outdir / "bricks"
    for suffix in ("inventory.csv", "instructions.pdf", "stability.csv"):
        assert (bricks / f"robot_{suffix}").is_file(), suffix
    model = load_model(str(bricks / "robot_model.pkl"))
    layers, depth, width = model.grid.shape
    assert layers > max(depth, width), f"the robot stands upright (grid {model.grid.shape})"
    direction = report["painter"]["front_direction"]
    assert model.front == GRID_FRONT_BY_MODEL_DIRECTION[direction], (
        f"face found at {direction}, bricks front {model.front}"
    )


if __name__ == "__main__":
    support.run_tests(globals())
