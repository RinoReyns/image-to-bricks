"""Golden regression builds: fixed commands on fixed models must give EXACTLY the frozen result (every piece, the
inventory and the stock report). This is the safety net for refactoring: any change in output fails here.

Cases come from tests/fixtures/public/golden_cases.json (our own figures, committed) and
tests/fixtures/local/golden_cases.json (third-party reference models, never committed; skipped when missing).

    python tests/test_golden_builds.py                   check every case
    python tests/test_golden_builds.py robot             only cases whose test name contains "robot"
    python tests/test_golden_builds.py --update-golden   re-freeze the expected files (only after an intended change)

Results depend slightly on the numeric libraries (numpy / scipy / OpenCV versions change k-means colour groups on big
models), so every case also freezes the library versions it was built with. A case frozen with other versions of
RESULT_DEFINING_PACKAGES is SKIPPED with a message. The Python version is recorded but not compared: with the same
pinned libraries, CI checks the goldens on every Python version it runs.

Needs the real trimesh.
"""

import importlib.metadata
import json
import os
import platform
import shutil
import sys
from collections.abc import Callable

import support

from image_to_bricks.instructions.outputs_handler import load_model

UPDATE_GOLDEN_FLAG = "--update-golden"
CASES_FILE_NAME = "golden_cases.json"
EXPECTED_OUTPUTS = {  # expected-file suffix -> build output suffix
    "expected_inventory.csv": "inventory.csv",
    "expected_stock_report.csv": "stock_report.csv",
}
FINGERPRINT_SUFFIX = "expected_fingerprint.txt"
ENVIRONMENT_SUFFIX = "expected_environment.json"
RESULT_DEFINING_PACKAGES = ("numpy", "scipy", "opencv-python", "trimesh", "pillow")


def describe_environment() -> dict[str, str]:
    """Python minor version + the versions of the packages that can change a build's numbers."""
    environment = {"python": ".".join(platform.python_version_tuple()[:2])}
    for package in RESULT_DEFINING_PACKAGES:
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = "not installed"
    return environment


def frozen_environment(fixtures_dir: str, case: dict) -> dict[str, str]:
    path = expected_path(fixtures_dir, case, ENVIRONMENT_SUFFIX)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as environment_file:
        return json.load(environment_file)


def load_cases(fixtures_dir: str) -> list[dict]:
    """Cases of one fixtures folder ([] when the folder or its cases file is missing)."""
    cases_path = os.path.join(fixtures_dir, CASES_FILE_NAME)
    if not os.path.exists(cases_path):
        return []
    with open(cases_path, encoding="utf-8") as cases_file:
        return json.load(cases_file)["cases"]


def resolve_arguments(case: dict, fixtures_dir: str, outdir: str) -> list[str]:
    """CLI arguments of a case: the model path first, {fixtures} / {outdir} placeholders filled in."""
    filled = [
        argument.replace("{fixtures}", fixtures_dir).replace("{outdir}", outdir) for argument in case["arguments"]
    ]
    return [os.path.join(fixtures_dir, case["model"]), *filled]


def expected_path(fixtures_dir: str, case: dict, suffix: str) -> str:
    return os.path.join(fixtures_dir, f"{case['name']}.{suffix}")


def build_output_path(outdir: str, case: dict, suffix: str) -> str:
    model_name = os.path.splitext(case["model"])[0]
    return os.path.join(outdir, f"{model_name}_{suffix}")


def run_case(case: dict, fixtures_dir: str) -> str:
    """Build the case into a new temp folder; returns that folder."""
    outdir = os.path.join(support.make_temp_dir(f"golden_{case['name']}"), "out")
    process = support.run_cli(resolve_arguments(case, fixtures_dir, outdir))
    support.assert_cli_succeeded(process)
    return outdir


def freeze_case(case: dict, fixtures_dir: str, outdir: str) -> None:
    """Write the expected files of a case from a fresh build (--update-golden)."""
    with open(expected_path(fixtures_dir, case, ENVIRONMENT_SUFFIX), "w", encoding="utf-8") as environment_file:
        json.dump(describe_environment(), environment_file, indent=2)
    with open(expected_path(fixtures_dir, case, FINGERPRINT_SUFFIX), "w", encoding="utf-8") as fingerprint_file:
        fingerprint_file.write(support.fingerprint_model(load_model(build_output_path(outdir, case, "model.pkl"))))
    for expected_suffix, output_suffix in EXPECTED_OUTPUTS.items():
        output = build_output_path(outdir, case, output_suffix)
        if os.path.exists(output):
            shutil.copyfile(output, expected_path(fixtures_dir, case, expected_suffix))
    print(f"  froze {case['name']} into {fixtures_dir}")


def compare_case(case: dict, fixtures_dir: str, outdir: str) -> None:
    """Fail on the first difference between a fresh build and the frozen files."""
    fingerprint_path = expected_path(fixtures_dir, case, FINGERPRINT_SUFFIX)
    for expected_suffix, output_suffix in EXPECTED_OUTPUTS.items():
        expected_file = expected_path(fixtures_dir, case, expected_suffix)
        if not os.path.exists(expected_file):
            continue
        expected_lines = support.read_text_lines(expected_file)
        actual_lines = support.read_text_lines(build_output_path(outdir, case, output_suffix))
        assert actual_lines == expected_lines, describe_first_difference(output_suffix, expected_lines, actual_lines)
    with open(fingerprint_path, encoding="utf-8") as fingerprint_file:
        expected_fingerprint = fingerprint_file.read().strip()
    actual_fingerprint = support.fingerprint_model(load_model(build_output_path(outdir, case, "model.pkl")))
    assert actual_fingerprint == expected_fingerprint, (
        f"{case['name']}: same piece counts but the pieces moved (fingerprint differs)"
    )


def describe_first_difference(file_label: str, expected_lines: list[str], actual_lines: list[str]) -> str:
    for line_number, (expected, actual) in enumerate(zip(expected_lines, actual_lines), start=1):
        if expected != actual:
            return f"{file_label} line {line_number}:\n  expected: {expected}\n  actual:   {actual}"
    return f"{file_label}: {len(expected_lines)} expected lines, {len(actual_lines)} actual lines"


def skip_if_frozen_elsewhere(case: dict, fixtures_dir: str) -> None:
    """Skip (with the version difference) when the case was frozen with other versions of the result-defining
    libraries than these (the Python version may differ)."""
    frozen = frozen_environment(fixtures_dir, case)
    if not frozen:
        raise AssertionError(f"no frozen result for {case['name']}: run with {UPDATE_GOLDEN_FLAG} once")
    current = describe_environment()
    differences = [
        f"{name} {frozen.get(name)} -> {current.get(name)}"
        for name in RESULT_DEFINING_PACKAGES
        if frozen.get(name) != current.get(name)
    ]
    if differences:
        raise support.SkipTest(
            f"frozen in another environment ({', '.join(differences)}); "
            f"freeze on this machine with {UPDATE_GOLDEN_FLAG}"
        )


def make_case_test(case: dict, fixtures_dir: str, is_local: bool) -> Callable[[], None]:
    """Test function for one case; local cases skip when their model is not on this computer."""

    def test_case() -> None:
        if is_local and not os.path.exists(os.path.join(fixtures_dir, case["model"])):
            raise support.SkipTest(f"local model {case['model']} not in {fixtures_dir}")
        updating = UPDATE_GOLDEN_FLAG in sys.argv
        if not updating:
            skip_if_frozen_elsewhere(case, fixtures_dir)
        outdir = run_case(case, fixtures_dir)
        if updating:
            freeze_case(case, fixtures_dir, outdir)
        else:
            compare_case(case, fixtures_dir, outdir)

    test_case.__name__ = ("test_local_golden_" if is_local else "test_golden_") + case["name"]
    return test_case


def collect_case_tests() -> dict[str, Callable[[], None]]:
    tests = {}
    for fixtures_dir, is_local in ((support.PUBLIC_FIXTURES_DIR, False), (support.LOCAL_FIXTURES_DIR, True)):
        for case in load_cases(fixtures_dir):
            case_test = make_case_test(case, fixtures_dir, is_local)
            tests[case_test.__name__] = case_test
    return tests


def test_public_cases_exist():
    assert load_cases(support.PUBLIC_FIXTURES_DIR), "tests/fixtures/public/golden_cases.json has no cases"


globals().update(collect_case_tests())

if __name__ == "__main__":
    support.run_tests(globals())
