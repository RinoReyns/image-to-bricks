"""The "not clean yet" lists in pyproject.toml only ever shrink.

pyproject.toml lists, per file, the ruff rules it still breaks and, per module, whether mypy still finds errors in it.
These tests fail when a listed rule or module is already clean, so the list is updated in the same commit that
cleans the code. They are skipped when ruff / mypy are not installed (pip install -r requirements-dev.txt).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import support

PYPROJECT_PATH = os.path.join(support.PROJECT_DIR, "pyproject.toml")
RUFF_RATCHET_ENTRY = re.compile(r'^"([^"*]+\.py)" = \[(.*?)\]', re.MULTILINE | re.DOTALL)
MYPY_RATCHET_BLOCK = re.compile(
    r"\[\[tool\.mypy\.overrides\]\]\n# Not clean yet.*?module = \[(.*?)\]\nignore_errors = true\n", re.DOTALL
)
MYPY_ERROR_LINE = re.compile(r"^([\w/\\.]+\.py):\d+: error", re.MULTILINE)
NO_DROPPED_ENTRIES = "lint.per-file-ignores = {}"
TOOL_TIMEOUT_SECONDS = 900


def read_pyproject() -> str:
    with open(PYPROJECT_PATH, encoding="utf-8") as pyproject:
        return pyproject.read()


def find_tool(name: str) -> list[str]:
    """Command that runs ruff / mypy from this Python's environment (or from PATH); skip the test if neither has it."""
    executable = shutil.which(name, path=os.path.dirname(sys.executable)) or shutil.which(name)
    if executable is None:
        raise support.SkipTest(f"{name} is not installed (pip install -r requirements-dev.txt)")
    return [executable]


def module_name(file_path: str) -> str:
    """mypy's module name for a checked file: tests and fixtures are top-level modules (mypy_path), the rest dotted."""
    relative_path = os.path.relpath(file_path, support.PROJECT_DIR).replace("\\", "/")[: -len(".py")]
    if relative_path.startswith("tests/"):
        return relative_path.rsplit("/", 1)[-1]
    return relative_path.replace("/", ".")


def test_every_listed_ruff_rule_is_still_broken() -> None:
    listed_codes_by_file = {
        file_path: set(re.findall(r'"([A-Z]+[0-9]+)"', codes))
        for file_path, codes in RUFF_RATCHET_ENTRY.findall(read_pyproject())
    }
    assert listed_codes_by_file, "no per-file 'not clean yet' entries found in pyproject.toml"
    process = subprocess.run(
        [*find_tool("ruff"), "check", "--output-format", "json", "--exit-zero", "--config", NO_DROPPED_ENTRIES, "."],
        cwd=support.PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=TOOL_TIMEOUT_SECONDS,
        check=True,
    )
    broken_codes_by_file: dict[str, set[str]] = {}
    for finding in json.loads(process.stdout):
        file_path = os.path.relpath(finding["filename"], support.PROJECT_DIR).replace("\\", "/")
        broken_codes_by_file.setdefault(file_path, set()).add(finding["code"])
    stale = {
        file_path: sorted(codes - broken_codes_by_file.get(file_path, set()))
        for file_path, codes in listed_codes_by_file.items()
    }
    stale = {file_path: codes for file_path, codes in stale.items() if codes}
    assert not stale, f"already clean, delete from pyproject.toml [tool.ruff.lint.per-file-ignores]: {stale}"


def test_every_listed_mypy_module_still_has_errors() -> None:
    pyproject_text = read_pyproject()
    ratchet_block = MYPY_RATCHET_BLOCK.search(pyproject_text)
    assert ratchet_block, "no mypy 'not clean yet' override found in pyproject.toml"
    listed_modules = set(re.findall(r'"([\w.]+)"', ratchet_block.group(1)))
    config_dir = support.make_temp_dir("mypy_ratchet_")
    config_path = os.path.join(config_dir, "pyproject.toml")
    with open(config_path, "w", encoding="utf-8") as config:
        config.write(pyproject_text.replace(ratchet_block.group(0), ""))
    process = subprocess.run(
        [
            *find_tool("mypy"),
            "--config-file",
            config_path,
            "--cache-dir",
            os.path.join(tempfile.gettempdir(), "mypy_ratchet_cache"),
        ],
        cwd=support.PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=TOOL_TIMEOUT_SECONDS,
        check=False,
    )
    if "site-packages" in process.stdout and "[syntax]" in process.stdout:
        raise support.SkipTest("the installed library stubs need a newer python_version than pyproject.toml's")
    modules_with_errors = {module_name(file_path) for file_path in MYPY_ERROR_LINE.findall(process.stdout)}
    stale = sorted(listed_modules - modules_with_errors)
    assert not stale, f"no type errors any more, delete from the mypy 'not clean yet' list in pyproject.toml: {stale}"


def test_modules_checked_strictly_by_mypy_do_not_use_bare_ndarray() -> None:
    """A bare np.ndarray annotation passes mypy with numpy >= 2.3 but fails with the pinned numpy 2.2: every module
    that is not on the mypy "not clean yet" list must also be clean of ruff TID251 (the ban on bare ndarray)."""
    pyproject_text = read_pyproject()
    ratchet_block = MYPY_RATCHET_BLOCK.search(pyproject_text)
    assert ratchet_block, "no mypy 'not clean yet' override found in pyproject.toml"
    listed_modules = set(re.findall(r'"([\w.]+)"', ratchet_block.group(1)))
    allowed_bare_ndarray = [
        file_path
        for file_path, codes in RUFF_RATCHET_ENTRY.findall(pyproject_text)
        if "TID251" in codes and module_name(os.path.join(support.PROJECT_DIR, file_path)) not in listed_modules
    ]
    assert not allowed_bare_ndarray, f"type-clean modules still allowed bare np.ndarray: {allowed_bare_ndarray}"


if __name__ == "__main__":
    support.run_tests(globals())
