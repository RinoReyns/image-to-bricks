"""images_to_bricks.py: pictures -> model -> LEGO build in one command (fake shape engine, real colour projection).

Also its options, the face direction handed to model_to_bricks, and a real TRELLIS.2 model of the robot built
upright. Needs the real trimesh. Run: python tests/test_images_to_bricks.py   (under a minute)
"""

import contextlib
import hashlib
import io
import os
import re
from pathlib import Path

import numpy as np
import support
from images_to_model_fakes import FakeShapeGenerator, build_fake_engines
from images_to_model_helpers import assert_raises
from PIL import Image

from image_to_bricks.images_to_bricks import (
    brick_front_from_report,
    build_brick_arguments,
    run,
    split_arguments,
)
from image_to_bricks.images_to_model.engines import Engines
from image_to_bricks.instructions.outputs_handler import load_model

CHARACTERS_DIR = Path(support.PUBLIC_FIXTURES_DIR) / "characters"
TRELLIS_ROBOT = CHARACTERS_DIR / "robot_trellis" / "robot.glb"


def write_character_pictures(folder: Path) -> Path:
    """hero/ with a front picture (red top half, blue bottom half on a transparent background) and a back one."""
    views = folder / "hero"
    views.mkdir(parents=True)
    pixels = np.zeros((128, 128, 4), dtype=np.uint8)
    pixels[16:64, 40:88] = [200, 30, 30, 255]
    pixels[64:112, 40:88] = [30, 60, 200, 255]
    Image.fromarray(pixels, "RGBA").save(views / "front.png")
    Image.fromarray(pixels, "RGBA").save(views / "back.png")
    return views


def load_fake_engines() -> Engines:
    """Fake engines with the real colour projection; the shape is a tall ellipsoid."""
    engines = build_fake_engines()
    engines.shape = FakeShapeGenerator()
    return engines


def test_arguments_after_the_separator_go_to_model_to_bricks() -> None:
    assert split_arguments(["--views", "v", "--", "--target-studs", "8"]) == (["--views", "v"], ["--target-studs", "8"])
    assert split_arguments(["--views", "v"]) == (["--views", "v"], [])


def test_the_face_direction_of_the_model_becomes_the_bricks_front() -> None:
    assert brick_front_from_report({"painter": {"front_direction": "+z"}}) == "down"
    assert brick_front_from_report({"painter": {"front_direction": "-x"}}) == "left"
    assert brick_front_from_report({}) is None


def test_the_users_brick_options_come_last_so_they_win() -> None:
    arguments = build_brick_arguments(Path("m/x.glb"), Path("o/bricks"), "down", ["--rotate-x", "0", "--front", "up"])
    assert arguments[0] == str(Path("m/x.glb")) and arguments[-4:] == ["--rotate-x", "0", "--front", "up"]
    assert arguments[arguments.index("--rotate-x") + 1] == "90" and "--pdf" in arguments
    assert "--front" not in build_brick_arguments(Path("x.glb"), Path("o"), None, [])


def test_an_outdir_after_the_separator_or_a_used_outdir_is_rejected() -> None:
    folder = Path(support.make_temp_dir("e2e_rejects"))
    views = write_character_pictures(folder)
    with contextlib.redirect_stderr(io.StringIO()), assert_raises(SystemExit):
        run(["--views", str(views), "--outdir", str(folder / "a"), "--", "--outdir", "x"], load_fake_engines)
    used = folder / "used"
    used.mkdir()
    (used / "old.txt").write_text("keep me", encoding="utf-8")
    with contextlib.redirect_stderr(io.StringIO()), assert_raises(SystemExit):
        run(["--views", str(views), "--outdir", str(used)], load_fake_engines)
    assert (used / "old.txt").read_text(encoding="utf-8") == "keep me"


def test_pictures_become_a_model_and_a_brick_build_with_instructions() -> None:
    folder = Path(support.make_temp_dir("e2e_images_to_bricks"))
    views = write_character_pictures(folder)
    outdir = folder / "out"
    brick_options = ["--target-studs", "6", "--colors", "Red,Blue,White,Black", "--no-log", "--save-model"]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = run(
            ["--views", str(views), "--outdir", str(outdir), "--max-faces", "2000", "--", *brick_options],
            load_fake_engines,
        )
    assert code == 0
    model_dir, bricks_dir = outdir / "model", outdir / "bricks"
    assert (model_dir / "hero.glb").is_file() and (model_dir / "hero_package" / "NOTICE.md").is_file()
    for suffix in ("inventory.csv", "instructions.pdf", "model.pkl"):
        assert (bricks_dir / f"hero_{suffix}").is_file(), suffix
    output = printed.getvalue()
    assert "Step 1/2" in output and "Step 2/2" in output and "face looks" in output
    model = load_model(str(bricks_dir / "hero_model.pkl"))
    layers, depth, width = model.grid.shape
    assert layers > 2 * max(depth, width), "the ellipsoid (2 units tall) stands upright"


def test_the_trellis_robot_model_builds_upright() -> None:
    outdir = os.path.join(support.make_temp_dir("e2e_trellis_robot"), "out")
    arguments = [str(TRELLIS_ROBOT), "--rotate-x", "90", "--front", "down", "--target-studs", "10", "--save-model"]
    support.assert_cli_succeeded(support.run_cli([*arguments, "--outdir", outdir]))
    model = load_model(os.path.join(outdir, "robot_model.pkl"))
    layers, depth, width = model.grid.shape
    assert layers > max(depth, width), f"taller than wide (grid {model.grid.shape}): the y-up GLB stands upright"


def test_the_character_pictures_are_the_ones_the_trellis_robot_was_made_from() -> None:
    notice = (TRELLIS_ROBOT.parent / "NOTICE.md").read_text(encoding="utf-8")
    for view, digest in re.findall(r"- (\w+)\.png sha256 ([0-9a-f]{64})", notice):
        picture = CHARACTERS_DIR / "robot" / f"{view}.png"
        assert hashlib.sha256(picture.read_bytes()).hexdigest() == digest, view
    assert "Built with DINOv3" in notice


if __name__ == "__main__":
    support.run_tests(globals())
