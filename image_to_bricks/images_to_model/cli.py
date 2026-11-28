"""Command line of images_to_model.py: pictures of a character -> coloured, cleaned, scaled GLB + licence package.

    python images_to_model.py --views DIR --output X.glb --height METRES
    python images_to_model.py --front F.png [--left L.png --back B.png --right R.png] --output X.glb --height METRES

Needs the GPU setup (scripts/setup_images_to_model.sh). Prints the run report as JSON.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from image_to_bricks.images_to_model.cutout import CutOutError
from image_to_bricks.images_to_model.engines import Engines
from image_to_bricks.images_to_model.pipeline import (
    DEFAULT_MAX_FACES,
    DEFAULT_SEED,
    TIME_DIGITS,
    ModelOutput,
    ModelSettings,
    run_pipeline,
)
from image_to_bricks.images_to_model.views import (
    FRONT_VIEW,
    VIEW_NAMES,
    ViewPictures,
    collect_view_files,
    find_views_in_folder,
)

EngineLoader = Callable[[], Engines]
EXIT_BAD_INPUT = 2


def parse_positive_number(text: str) -> float:
    """argparse type: a number > 0."""
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from error
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def add_picture_arguments(parser: argparse.ArgumentParser) -> None:
    """--views DIR, or the pictures one by one (--front required, --left / --back / --right optional)."""
    group = parser.add_argument_group(
        "pictures", "a folder with front.png (+ left.png, back.png, right.png), or the files one by one"
    )
    group.add_argument("--views", type=Path, help="folder with front.png and optionally left.png, back.png, right.png")
    for name in VIEW_NAMES:
        group.add_argument(
            f"--{name}", type=Path, help=f"{name} picture (RGBA with transparent background, or a plain background)"
        )


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Face limit and seed of the model step."""
    parser.add_argument(
        "--max-faces",
        type=int,
        default=DEFAULT_MAX_FACES,
        help="faces of the cleaned model (default %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed of the shape model (default %(default)s)")


def choose_pictures(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> ViewPictures:
    """The pictures from --views or from --front / --left / --back / --right (parser.error when both or neither
    are given, or a file is missing)."""
    single_files = {name: getattr(arguments, name) for name in VIEW_NAMES}
    has_single_files = any(path is not None for path in single_files.values())
    if (arguments.views is None) == (not has_single_files):
        parser.error(f"give either --views DIR or --{FRONT_VIEW} FILE (with optional --left / --back / --right)")
    try:
        if arguments.views is not None:
            return find_views_in_folder(arguments.views)
        return collect_view_files(single_files)
    except FileNotFoundError as error:
        parser.error(str(error))


def build_argument_parser() -> argparse.ArgumentParser:
    """Options of images_to_model.py."""
    parser = argparse.ArgumentParser(
        prog="images_to_model.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_picture_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="output .glb")
    parser.add_argument("--height", type=parse_positive_number, required=True, help="model height in metres")
    parser.add_argument("--work", type=Path, default=None, help="work folder (default: <output>_work)")
    add_model_arguments(parser)
    return parser


def load_production_engines() -> Engines:
    """TRELLIS.2 and friends (imported here, so this module loads without the GPU setup)."""
    from image_to_bricks.images_to_model.trellis_engine import load_engines  # noqa: PLC0415

    return load_engines()


def run_cli(argv: Sequence[str] | None = None, engine_loader: EngineLoader = load_production_engines) -> int:
    """Make the model; print the report as JSON. Returns the exit code (EXIT_BAD_INPUT for unusable pictures)."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    pictures = choose_pictures(arguments, parser)
    started = time.perf_counter()
    engines = engine_loader()
    load_seconds = round(time.perf_counter() - started, TIME_DIGITS)
    output = ModelOutput(arguments.output, arguments.work)
    settings = ModelSettings(arguments.height, arguments.max_faces, arguments.seed)
    try:
        report = run_pipeline(pictures, output, settings, engines)
    except (CutOutError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT
    report["seconds"]["load_models"] = load_seconds
    print(json.dumps(report, indent=2))
    return 0


def main() -> None:
    """Command-line entry point."""
    sys.exit(run_cli())
