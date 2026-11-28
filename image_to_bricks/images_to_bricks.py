"""Command line of images_to_bricks.py: pictures of a character -> 3D model -> LEGO build with instructions.

    python images_to_bricks.py --views DIR --outdir OUT [--height METRES] [-- model_to_bricks options]
    python images_to_bricks.py --front F.png [--left L.png --back B.png --right R.png] --outdir OUT [-- options]

Step 1, images_to_model (GPU): OUT/model/NAME.glb with its licence package (NAME_package/, NOTICE.md).
Step 2, model_to_bricks: OUT/bricks/ with the parts list, layer maps and the instruction PDF.
Everything after "--" goes to model_to_bricks.py unchanged (e.g. -- --target-studs 16 --base 2 --stock my.csv).
The model is stored y-up and the colour projection finds where its face looks, so --rotate-x 90, --front and --pdf
are set for you (the same options after "--" win). Each step also runs on its own: images_to_model.py,
model_to_bricks.py.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from image_to_bricks import cli as bricks_cli
from image_to_bricks.images_to_model.cli import (
    EngineLoader,
    add_model_arguments,
    add_picture_arguments,
    choose_pictures,
    load_production_engines,
    parse_positive_number,
)
from image_to_bricks.images_to_model.cutout import CutOutError
from image_to_bricks.images_to_model.pipeline import ModelOutput, ModelSettings, run_pipeline
from image_to_bricks.images_to_model.trellis_engine import release_gpu_memory
from image_to_bricks.images_to_model.type_aliases import JsonDict

BRICK_OPTIONS_SEPARATOR = "--"
MODEL_FOLDER = "model"
BRICKS_FOLDER = "bricks"
DEFAULT_HEIGHT_METRES = 0.3  # GLB size only; the build size is --target-studs
Y_UP_TO_Z_UP_DEGREES = "90"  # --rotate-x that stands a y-up GLB upright on the brick grid (z up)
# where the face looks after --rotate-x 90 (y-up -> z-up turns +z into -y), by the projection's front direction
BRICK_FRONT_BY_MODEL_DIRECTION = {"+z": "down", "-z": "up", "+x": "right", "-x": "left"}
OPTIONS_SET_BY_THIS_COMMAND = ("--outdir",)
EXIT_BAD_INPUT = 2


def split_arguments(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """(arguments of this command, arguments for model_to_bricks.py after the first "--")."""
    arguments = list(argv)
    if BRICK_OPTIONS_SEPARATOR not in arguments:
        return arguments, []
    separator = arguments.index(BRICK_OPTIONS_SEPARATOR)
    return arguments[:separator], arguments[separator + 1 :]


def build_argument_parser() -> argparse.ArgumentParser:
    """Options of images_to_bricks.py (the model_to_bricks options come after "--")."""
    parser = argparse.ArgumentParser(
        prog="images_to_bricks.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_picture_arguments(parser)
    parser.add_argument(
        "--outdir", type=Path, required=True, help="new output folder: model/ and bricks/ are written into it"
    )
    parser.add_argument(
        "--height",
        type=parse_positive_number,
        default=DEFAULT_HEIGHT_METRES,
        help="height of the GLB in metres (default %(default)s; the build size is --target-studs after --)",
    )
    add_model_arguments(parser)
    return parser


def check_brick_options(brick_options: Sequence[str], parser: argparse.ArgumentParser) -> None:
    """parser.error for a model_to_bricks option this command sets itself (the input model, --outdir)."""
    for option in brick_options:
        if option.split("=", 1)[0] in OPTIONS_SET_BY_THIS_COMMAND:
            parser.error(f"{option} is set by images_to_bricks.py (use its --outdir); remove it after '--'")


def check_new_outdir(outdir: Path, parser: argparse.ArgumentParser) -> None:
    """parser.error when outdir already holds files (every run gets a new folder, nothing is overwritten)."""
    if outdir.exists() and any(outdir.iterdir()):
        parser.error(f"--outdir {outdir} is not empty: use a new folder for every run")


def brick_front_from_report(report: JsonDict) -> str | None:
    """--front for model_to_bricks from the colour projection's front direction (None when it is not known)."""
    direction = report.get("painter", {}).get("front_direction")
    return BRICK_FRONT_BY_MODEL_DIRECTION.get(direction) if isinstance(direction, str) else None


def build_brick_arguments(glb_path: Path, bricks_outdir: Path, front: str | None, extra: Sequence[str]) -> list[str]:
    """model_to_bricks arguments: the model, upright, face direction, PDF, output folder, then the user's options
    (later options win in argparse, so the user's --rotate-x / --front replace ours)."""
    arguments = [str(glb_path), "--rotate-x", Y_UP_TO_Z_UP_DEGREES, "--pdf", "--outdir", str(bricks_outdir)]
    if front is not None:
        arguments += ["--front", front]
    return arguments + list(extra)


def run(argv: Sequence[str] | None = None, engine_loader: EngineLoader = load_production_engines) -> int:
    """Both steps. Returns the exit code (EXIT_BAD_INPUT for unusable pictures)."""
    own_arguments, brick_options = split_arguments(sys.argv[1:] if argv is None else argv)
    parser = build_argument_parser()
    arguments = parser.parse_args(own_arguments)
    pictures = choose_pictures(arguments, parser)
    check_brick_options(brick_options, parser)
    check_new_outdir(arguments.outdir, parser)
    name = pictures.character_name
    glb_path = arguments.outdir / MODEL_FOLDER / f"{name}.glb"
    print(f"Step 1/2: 3D model from {len(pictures.files)} picture(s) -> {glb_path}", flush=True)
    settings = ModelSettings(arguments.height, arguments.max_faces, arguments.seed)
    try:
        report = run_pipeline(pictures, ModelOutput(glb_path), settings, engine_loader())
    except (CutOutError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT
    release_gpu_memory()
    front = brick_front_from_report(report)
    print(f"Model: {glb_path} (licence: {report['package']['usage']}, see {report['package']['folder']}/NOTICE.md)")
    bricks_outdir = arguments.outdir / BRICKS_FOLDER
    print(f"Step 2/2: LEGO build -> {bricks_outdir} (face looks {front or 'unknown'})", flush=True)
    bricks_cli.main(build_brick_arguments(glb_path, bricks_outdir, front, brick_options))
    print(f"Instructions: {bricks_outdir / (name + '_instructions.pdf')}")
    return 0


def main() -> None:
    """Command-line entry point."""
    sys.exit(run())
