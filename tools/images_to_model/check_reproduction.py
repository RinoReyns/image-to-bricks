"""Compare a reproduce run with the saved results, or save a run as the new baseline.

python -m tools.images_to_model.check_reproduction --runs repro_out/runs --compare repro_out/compare/report.json
... --write-expected      (only when we agree that this run is the new baseline)
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from image_to_bricks.images_to_model.verification.reproduction import collect_results, compare_results

DEFAULT_EXPECTED = Path("lock/images_to_model/expected_results.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--compare", type=Path, required=True)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--write-expected", action="store_true")
    return parser.parse_args(argv)


def check_reproduction(argv: Sequence[str] | None = None) -> int:
    """PASS / FAIL per number; exit code 1 when any check fails."""
    arguments = parse_args(argv)
    summary = json.loads((arguments.runs / "run_summary.json").read_text(encoding="utf-8"))
    compare_report = json.loads(arguments.compare.read_text(encoding="utf-8"))
    actual = collect_results(arguments.runs, compare_report, summary)
    (arguments.runs.parent / "results.json").write_text(json.dumps(actual, indent=2), encoding="utf-8")
    if arguments.write_expected:
        arguments.expected.parent.mkdir(parents=True, exist_ok=True)
        arguments.expected.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
        print(f"saved {len(actual)} cases as the expected results in {arguments.expected}")
        return 0
    ok, lines = compare_results(json.loads(arguments.expected.read_text(encoding="utf-8")), actual)
    print("\n".join(lines))
    failed = sum(line.startswith("FAIL") for line in lines)
    print(f"\nREPRODUCTION {'PASSED' if ok else 'FAILED'}: {failed} check(s) failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check_reproduction())
