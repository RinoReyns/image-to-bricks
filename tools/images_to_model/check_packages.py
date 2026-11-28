"""Check that the running Python environment holds exactly the packages of the lock files, and none of the NVIDIA
research-only packages.

    python -m tools.images_to_model.check_packages requirements.txt requirements-dev.txt \
        lock/images_to_model/requirements-gpu.txt lock/images_to_model/torch.txt lock/images_to_model/builds.txt
"""

import argparse
import importlib.metadata
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path

from image_to_bricks.images_to_model.verification.lockfiles import (
    compare_packages,
    list_installed_packages,
    read_expected_packages,
)

FORBIDDEN_MODULES = ("nvdiffrast", "nvdiffrec_render", "renderutils")
INSTALLER_PACKAGES = ("pip",)  # the venv's own installer, not part of any lock


def check_packages(argv: Sequence[str] | None = None) -> int:
    """Print every difference; exit code 1 when there is any."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lock_files", nargs="+", type=Path, help="requirement / lock files the environment must match")
    arguments = parser.parse_args(argv)
    expected = read_expected_packages(arguments.lock_files)
    installed = list_installed_packages(importlib.metadata.distributions())
    for name in INSTALLER_PACKAGES:
        installed.pop(name, None)
    problems = compare_packages(expected, installed)
    problems += [
        f"research-only module installed: {name}" for name in FORBIDDEN_MODULES if importlib.util.find_spec(name)
    ]
    for problem in problems:
        print("PROBLEM", problem)
    print(f"packages: {len(installed)} installed, {len(expected)} locked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(check_packages())
