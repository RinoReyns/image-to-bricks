"""Shared helpers for the function-style tests (test files that need the real trimesh, the CLI or fixtures).

Every test file that imports this module can be run on its own (`python tests/test_x.py`) or through
`python tests/run_all.py`. The older script-style tests do not use this module: they stub trimesh / tqdm themselves.
"""

import contextlib
import dataclasses
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import traceback
import types
from collections.abc import Callable, Iterator, Sequence

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
FIXTURES_DIR = os.path.join(TESTS_DIR, "fixtures")
PUBLIC_FIXTURES_DIR = os.path.join(FIXTURES_DIR, "public")
LOCAL_FIXTURES_DIR = os.path.join(FIXTURES_DIR, "local")
CLI_SCRIPT = os.path.join(PROJECT_DIR, "model_to_bricks.py")
CLI_TIMEOUT_SECONDS = 1800
EXHAUSTIVE_FLAG = "--exhaustive"
EXHAUSTIVE_ENVIRONMENT_VARIABLE = "BRICKS_EXHAUSTIVE_TESTS"  # set by `run_all.py --exhaustive` for every test file
SHARD_ENVIRONMENT_VARIABLE = "BRICKS_TEST_SHARD"  # "index/count", set by run_all.py to split a long sweep over cores

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def stub_mesh_libraries() -> None:
    """Replace trimesh and tqdm by empty stand-ins (tqdm passes its iterable through), for tests of code that imports
    them but does not need them. Call before importing image_to_bricks."""
    for library_name in ("trimesh", "tqdm"):
        stub = types.ModuleType(library_name)
        stub.tqdm = lambda iterable=None, **_options: iterable  # type: ignore[attr-defined]
        sys.modules[library_name] = stub
    sys.modules["trimesh"].Trimesh = object  # type: ignore[attr-defined]


def is_exhaustive() -> bool:
    """True when the slow, exhaustive variants of the sweep tests should run (`--exhaustive` on the command line or
    the BRICKS_EXHAUSTIVE_TESTS=1 environment variable); by default they check a fixed representative sample."""
    return EXHAUSTIVE_FLAG in sys.argv or os.environ.get(EXHAUSTIVE_ENVIRONMENT_VARIABLE) == "1"


def select_shard[T](items: Sequence[T]) -> list[T]:
    """The part of a sweep this process checks: every count-th item from index on when BRICKS_TEST_SHARD="index/count"
    is set (run_all.py starts count processes, together they check every item once), else all items."""
    shard_text = os.environ.get(SHARD_ENVIRONMENT_VARIABLE, "")
    if not shard_text:
        return list(items)
    index_text, count_text = shard_text.split("/")
    return list(items[int(index_text) :: int(count_text)])


class SkipTest(Exception):
    """Raised by a test that cannot run here (missing local fixture); the runner reports it as SKIP, not as a pass."""


@contextlib.contextmanager
def module_settings(module, **overrides) -> Iterator[None]:
    """Temporarily set module-level settings (e.g. colour_sampling.MISSING_TOL) and restore them afterwards, so one test
    cannot change the behaviour of the next one. Unknown names raise AttributeError (typos fail loudly)."""
    previous = {name: getattr(module, name) for name in overrides}
    for name, value in overrides.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


@contextlib.contextmanager
def quiet_stdout() -> Iterator[None]:
    """Silence the pipeline's progress prints inside a test (failures still show their assertion message)."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def make_temp_dir(prefix: str) -> str:
    """New empty folder for one test's files (left on disk so a failing run can be inspected)."""
    return tempfile.mkdtemp(prefix=f"bricks_{prefix}_")


def run_cli(arguments: Sequence[str], timeout_seconds: int = CLI_TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    """Run `python model_to_bricks.py ARGUMENTS` in a fresh process from the project folder (the CLI sets module
    globals, so every run needs its own process). Returns the finished process with stdout + stderr as text."""
    return subprocess.run(
        [sys.executable, CLI_SCRIPT, *arguments],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )


def assert_cli_succeeded(process: subprocess.CompletedProcess) -> None:
    """Fail with the end of the CLI output when the run did not exit with 0."""
    assert process.returncode == 0, (
        f"CLI exited with {process.returncode}\n--- stdout (end) ---\n"
        f"{process.stdout[-3000:]}\n--- stderr (end) ---\n{process.stderr[-3000:]}"
    )


def fingerprint_model(model) -> str:
    """sha256 of everything that defines a build: grid shape + cells, base layers, base colour, front, accents and
    every piece (all PlacedBrick fields, order-independent). Equal fingerprints = identical builds."""
    digest = hashlib.sha256()
    grid = np.ascontiguousarray(model.grid.astype(bool))
    digest.update(repr(grid.shape).encode())
    digest.update(grid.tobytes())
    digest.update(repr((model.base_layers, model.base_color, model.front)).encode())
    digest.update(repr(sorted(map(repr, model.accents))).encode())
    pieces = sorted(dataclasses.astuple(piece) for piece in model.bricks)
    digest.update(repr(pieces).encode())
    return digest.hexdigest()


def load_image(path: str):
    """The image at path, fully read and with its file CLOSED. Windows cannot delete a file that is still open, so a
    lazy Image.open inside a TemporaryDirectory makes the folder clean-up fail there (Linux does not notice)."""
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        return image.copy()


def read_text_lines(path: str) -> list[str]:
    """Lines of a text file without line endings (CRLF and LF compare equal: Windows and Linux runs match)."""
    with open(path, encoding="utf-8") as text_file:
        return text_file.read().splitlines()


def cells_covered_by_pieces(model) -> np.ndarray:
    """int [z, y, x]: how many pieces cover each cell (1 everywhere a cell is filled = exact cover)."""
    from image_to_bricks.model import piece_cells

    cover_count = np.zeros(model.grid.shape, np.int32)
    for piece in model.bricks:
        for stud_x, stud_y in piece_cells(piece):
            cover_count[piece.z : piece.z + piece.h_plates, stud_y, stud_x] += 1
    return cover_count


def assert_pieces_cover_grid_exactly(model) -> None:
    """Every filled cell is covered by exactly one piece and no piece covers an empty cell."""
    cover_count = cells_covered_by_pieces(model)
    overlapping_cells = int((cover_count > 1).sum())
    uncovered_cells = int((model.grid & (cover_count == 0)).sum())
    pieces_outside_grid = int((~model.grid & (cover_count > 0)).sum())
    assert overlapping_cells == 0, f"{overlapping_cells} cells covered by more than one piece"
    assert uncovered_cells == 0, f"{uncovered_cells} filled cells without a piece"
    assert pieces_outside_grid == 0, f"{pieces_outside_grid} empty cells covered by a piece"


def run_tests(namespace: dict[str, object]) -> None:
    """Minimal runner: call every `test_*` function of the calling module (in file order), print ok / SKIP / FAIL per
    test and exit with 1 if any test failed. An optional first command-line argument runs only the tests whose name
    contains it (`python tests/test_x.py rotate`); arguments starting with "--" are flags for the test module, not
    filters. Also works under pytest, which collects the same functions."""
    name_filters = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    only: str | None = name_filters[0] if name_filters else None
    tests: list[Callable[[], None]] = [
        value
        for name, value in namespace.items()
        if name.startswith("test_") and callable(value) and (only is None or only in name)
    ]
    failed = skipped = 0
    for test in tests:
        started = time.perf_counter()
        try:
            test()
        except SkipTest as reason:
            skipped += 1
            print(f"SKIP {test.__name__}: {reason}")
            continue
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}\n{traceback.format_exc()}")
            continue
        print(f"ok   {test.__name__} ({time.perf_counter() - started:.1f} s)")
    passed = len(tests) - failed - skipped
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
