"""Run every test file in tests/ (each in its own process, like `python tests/test_x.py`) and print one summary.

    python tests/run_all.py                all tests, sweeps on a fixed sample, one file per CPU core at a time
    python tests/run_all.py --fast         also skip the golden builds and the CLI end-to-end runs
    python tests/run_all.py --exhaustive   sweep tests check every configuration; the base sweep is split over --jobs
    python tests/run_all.py --jobs 4       run 4 test processes at a time (default: number of CPU cores)
    python tests/run_all.py golden         only files whose name contains "golden"

Exit code 1 if any file failed. Skipped golden cases (local model missing, frozen in another environment) are listed
with their reason, so a skip is never mistaken for a pass.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
# Started first, so the long ones overlap with the many short ones (the order of the rest does not matter).
SLOWEST_TEST_FILES = (
    "test_golden_builds.py",
    "test_pipeline_end_to_end.py",
    "test_cli_options.py",
    "test_base_bricks.py",
    "test_cleanup.py",
    "test_base_holds_together.py",
    "test_l_pieces.py",
)
FAST_MODE_SKIPPED_FILES = {"test_golden_builds.py", "test_pipeline_end_to_end.py"}  # real CLI builds, ~1 min each
EXHAUSTIVE_ENVIRONMENT_VARIABLE = "BRICKS_EXHAUSTIVE_TESTS"  # same name as in support.py (read by the test files)
SHARD_ENVIRONMENT_VARIABLE = "BRICKS_TEST_SHARD"  # same name as in support.py (support.select_shard)
# Exhaustive sweeps that check independent cases: with --exhaustive each one is split into --jobs processes.
SHARDED_WHEN_EXHAUSTIVE_FILES = {"test_base_holds_together.py"}
TEST_FILE_TIMEOUT_SECONDS = 1800
STILL_RUNNING_NOTE_SECONDS = 30  # quiet longer than this -> print which files are still running
SUMMARY_LINE = re.compile(r"^(\d+) passed, (\d+) failed, (\d+) skipped$", re.MULTILINE)
SKIP_LINE = re.compile(r"^SKIP .*$", re.MULTILINE)


@dataclass(frozen=True)
class TestJob:
    """One test process: a whole test file, or one shard of a split sweep (shard_text "index/count")."""

    path: str
    shard_text: str = ""

    @property
    def name(self) -> str:
        file_name = os.path.basename(self.path)
        return f"{file_name} [{self.shard_text}]" if self.shard_text else file_name


@dataclass
class FileResult:
    file_name: str
    exit_code: int
    seconds: float
    output: str

    @property
    def is_passed(self) -> bool:
        return self.exit_code == 0

    @property
    def skip_lines(self) -> list[str]:
        return SKIP_LINE.findall(self.output)

    @property
    def counts_text(self) -> str:
        """'12 passed, 0 failed, 1 skipped' for function-style files; '' for the older script-style ones."""
        match = SUMMARY_LINE.search(self.output)
        return match.group(0) if match else ""


def find_test_files(name_filter: str, is_fast_mode: bool) -> list[str]:
    """Test files to run, the slowest first (so they overlap with the short ones)."""
    paths = sorted(glob.glob(os.path.join(TESTS_DIR, "test_*.py")))
    selected = [
        path
        for path in paths
        if (not name_filter or name_filter in os.path.basename(path))
        and not (is_fast_mode and os.path.basename(path) in FAST_MODE_SKIPPED_FILES)
    ]

    def start_order(path: str) -> int:
        file_name = os.path.basename(path)
        return SLOWEST_TEST_FILES.index(file_name) if file_name in SLOWEST_TEST_FILES else len(SLOWEST_TEST_FILES)

    return sorted(selected, key=start_order)


def build_jobs(paths: list[str], is_exhaustive: bool, job_count: int) -> list[TestJob]:
    """One job per file; with --exhaustive the sweeps in SHARDED_WHEN_EXHAUSTIVE_FILES become job_count jobs each."""
    jobs: list[TestJob] = []
    for path in paths:
        if is_exhaustive and job_count > 1 and os.path.basename(path) in SHARDED_WHEN_EXHAUSTIVE_FILES:
            jobs.extend(TestJob(path, f"{index}/{job_count}") for index in range(job_count))
        else:
            jobs.append(TestJob(path))
    return sorted(jobs, key=lambda job: not job.shard_text)  # the split sweep is the longest work: start it first


def run_test_job(job: TestJob, is_exhaustive: bool, running_files: set[str]) -> FileResult:
    """Run one test file (or one shard of it) in its own process; while it runs its name is in running_files."""
    file_name = job.name
    environment = dict(os.environ)
    if is_exhaustive:
        environment[EXHAUSTIVE_ENVIRONMENT_VARIABLE] = "1"
    if job.shard_text:
        environment[SHARD_ENVIRONMENT_VARIABLE] = job.shard_text
    running_files.add(file_name)
    started = time.perf_counter()
    try:
        process = subprocess.run(
            [sys.executable, job.path],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TEST_FILE_TIMEOUT_SECONDS,
            env=environment,
        )
        exit_code, output = process.returncode, process.stdout + process.stderr
    except subprocess.TimeoutExpired as timeout:
        exit_code, output = 1, f"TIMEOUT after {timeout.timeout} s"
    finally:
        running_files.discard(file_name)
    return FileResult(file_name, exit_code, time.perf_counter() - started, output)


def print_result(result: FileResult) -> None:
    status = "ok  " if result.is_passed else "FAIL"
    print(f"{status} {result.file_name:34s} {result.seconds:6.1f} s  {result.counts_text}", flush=True)
    for skip_line in result.skip_lines:
        print(f"       {skip_line}")
    if not result.is_passed:
        print("       " + "\n       ".join(result.output.strip().splitlines()[-25:]))


def run_jobs_printing_as_they_finish(jobs: list[TestJob], job_count: int, is_exhaustive: bool) -> list[FileResult]:
    """Run the files, print each result as soon as it is done (a slow file does not hold back the others' lines)
    and, when nothing finishes for STILL_RUNNING_NOTE_SECONDS, which files are running and how many still wait."""
    results = []
    running_files: set[str] = set()
    with ThreadPoolExecutor(max_workers=job_count) as pool:
        pending = {pool.submit(run_test_job, job, is_exhaustive, running_files): job for job in jobs}
        while pending:
            done, _ = wait(pending, timeout=STILL_RUNNING_NOTE_SECONDS, return_when=FIRST_COMPLETED)
            if not done:
                waiting_count = len(pending) - len(running_files)
                print(
                    f"     ... running: {', '.join(sorted(running_files))}"
                    + (f" ({waiting_count} more waiting)" if waiting_count > 0 else ""),
                    flush=True,
                )
                continue
            for future in done:
                del pending[future]
                result = future.result()
                print_result(result)
                results.append(result)
    return sorted(results, key=lambda result: result.file_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every test file and print one summary.")
    parser.add_argument("name_filter", nargs="?", default="", help="only test files whose name contains this")
    parser.add_argument(
        "--fast", action="store_true", help=f"skip the real CLI builds: {', '.join(sorted(FAST_MODE_SKIPPED_FILES))}"
    )
    parser.add_argument(
        "--exhaustive", action="store_true", help="sweep tests check every configuration instead of a fixed sample"
    )
    default_job_count = os.cpu_count() or 1
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_job_count,
        help=f"test files run at the same time (default {default_job_count})",
    )
    arguments = parser.parse_args()
    paths = find_test_files(arguments.name_filter, arguments.fast)
    if not paths:
        sys.exit(f"no test files match {arguments.name_filter!r}")
    started = time.perf_counter()
    job_count = max(1, arguments.jobs)
    jobs = build_jobs(paths, arguments.exhaustive, job_count)
    results = run_jobs_printing_as_they_finish(jobs, job_count, arguments.exhaustive)
    failed = [result.file_name for result in results if not result.is_passed]
    skipped_cases = sum(len(result.skip_lines) for result in results)
    print(
        f"\n{len(results) - len(failed)} of {len(results)} test processes passed, {skipped_cases} case(s) skipped, "
        f"{time.perf_counter() - started:.0f} s" + (f"\nFAILED: {', '.join(failed)}" if failed else "")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
