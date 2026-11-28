"""Put the exact model files of lock/images_to_model/models.txt into the project's model cache, then verify every byte.

    python -m tools.images_to_model.prepare_models --lock lock/images_to_model/models.txt --cache .models

Hugging Face repos go to <cache>/hf/hub at the pinned revision; refs/main is pointed at that revision so the code
(which asks for "main") gets exactly it when run offline (HF_HUB_OFFLINE=1). Files already in --seed-hub (your normal
cache) are hard-linked instead of downloaded; every file is then checked against its content hash. u2net.onnx goes to
<cache>/u2net and is checked against its sha256.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from image_to_bricks.images_to_model.type_aliases import JsonDict
from image_to_bricks.images_to_model.verification.lockfiles import (
    build_patterns,
    check_references,
    check_snapshot,
    find_pipeline_references,
    hash_sha256,
    read_model_lock,
)

MAIN_REPO = "microsoft/TRELLIS.2-4B"  # its pipeline.json names the other repos
FROM_PIPELINE = "from-pipeline"
INCOMPLETE_SUFFIX = ".incomplete"
SHORT_REVISION_LENGTH = 12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lock", type=Path, default=Path("lock/images_to_model/models.txt"))
    parser.add_argument("--cache", type=Path, default=Path(".models"))
    parser.add_argument("--seed-hub", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--seed-u2net", type=Path, default=Path.home() / ".u2net")
    return parser.parse_args(argv)


def read_token() -> str | None:
    """HF_TOKEN, else the token saved by 'hf auth login' in the normal cache (the project cache has none)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = Path(os.environ.get("HF_TOKEN_PATH", Path.home() / ".cache/huggingface/token"))
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def repo_folder(hub: Path, repo: str) -> Path:
    """Cache folder of a repo."""
    return Path(hub) / ("models--" + repo.replace("/", "--"))


def link_seed_blobs(seed_hub: Path, hub: Path, repo: str) -> int:
    """Hard-link blobs already downloaded in the normal cache (no copy, no download). Returns how many."""
    source, target = repo_folder(seed_hub, repo) / "blobs", repo_folder(hub, repo) / "blobs"
    if not source.is_dir():
        return 0
    target.mkdir(parents=True, exist_ok=True)
    linked = 0
    for blob in source.iterdir():
        destination = target / blob.name
        if destination.exists() or blob.name.endswith(INCOMPLETE_SUFFIX):
            continue
        try:
            os.link(blob, destination)
        except OSError:  # another file system: the download fetches it instead
            continue
        linked += 1
    return linked


def fetch_repo(hub: Path, entry: JsonDict, patterns: list[str] | None, token: str | None) -> Path:
    """Download (or complete) one pinned repo and point its refs/main at the pinned revision. Returns the snapshot
    folder. Raises SystemExit with the steps to take when the repo is gated and access is missing."""
    huggingface_hub = importlib.import_module("huggingface_hub")
    hub_errors = importlib.import_module("huggingface_hub.errors")
    try:
        path = huggingface_hub.snapshot_download(
            entry["repo"], revision=entry["revision"], cache_dir=str(hub), allow_patterns=patterns, token=token
        )
    except hub_errors.GatedRepoError as error:
        raise SystemExit(
            f"{entry['repo']} is gated: accept its licence on huggingface.co/{entry['repo']}, then run "
            "'uvx --from huggingface_hub hf auth login' and this script again"
        ) from error
    refs = repo_folder(hub, entry["repo"]) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(entry["revision"], encoding="utf-8")
    return Path(path)


def list_snapshot_files(snapshot: Path) -> list[str]:
    """Files of a snapshot, relative to it."""
    return [str(item.relative_to(snapshot)) for item in snapshot.rglob("*") if item.is_file()]


def prepare_u2net(folder: Path, seed: Path, entry: JsonDict) -> str:
    """Copy (from seed, when it matches) or download a pinned weights file and check its sha256. Returns its path.
    Raises SystemExit (and removes the file) when the hash does not match."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / entry["name"]
    seed_file = seed / entry["name"]
    if not target.is_file() and seed_file.is_file() and hash_sha256(seed_file) == entry["sha256"]:
        shutil.copy2(seed_file, target)
    if not target.is_file():
        print(f"downloading {entry['url']}", flush=True)
        urllib.request.urlretrieve(entry["url"], target)  # a fixed https URL from the lock file
    digest = hash_sha256(target)
    if digest != entry["sha256"]:
        target.unlink()
        raise SystemExit(f"{entry['name']}: sha256 {digest} != locked {entry['sha256']} (file removed)")
    return str(target)


def order_entries(lock: JsonDict) -> list[JsonDict]:
    """TRELLIS.2 first (its pipeline.json says which files of the other repos are needed)."""
    entries: list[JsonDict] = lock["hf"]
    return [entry for entry in entries if entry["repo"] == MAIN_REPO] + [
        entry for entry in entries if entry["repo"] != MAIN_REPO
    ]


def prepare_models(argv: Sequence[str] | None = None) -> int:
    """Fetch and check every pinned file; exit code 1 when anything differs from the lock."""
    arguments = parse_args(argv)
    lock = read_model_lock(arguments.lock)
    cache = arguments.cache.resolve()
    hub = cache / "hf" / "hub"
    token = read_token()
    references: dict[str, list[str]] = {}
    problems: list[str] = []
    for entry in order_entries(lock):
        patterns = None
        if entry["files"] == FROM_PIPELINE:
            if entry["repo"] not in references:
                problems.append(f"{entry['repo']}: pinned, but pipeline.json does not refer to it")
                continue
            patterns = build_patterns(references[entry["repo"]])
        linked = link_seed_blobs(arguments.seed_hub, hub, entry["repo"])
        snapshot = fetch_repo(hub, entry, patterns, token)
        revision = entry["revision"][:SHORT_REVISION_LENGTH]
        file_count = len(list_snapshot_files(snapshot))
        print(f"{entry['repo']} @ {revision}: {file_count} files ({linked} linked from your cache)", flush=True)
        if entry["repo"] == MAIN_REPO:
            pipeline = json.loads((snapshot / "pipeline.json").read_text(encoding="utf-8"))
            references = find_pipeline_references(pipeline, list_snapshot_files(snapshot))
            print(f"  pipeline.json refers to: {references}", flush=True)
            problems += check_references(references, lock)
        bad = check_snapshot(snapshot)
        problems += [f"{entry['repo']}: {name} does not match its content hash" for name in bad]
        print(f"  content hashes: {'all OK' if not bad else f'{len(bad)} BAD'}", flush=True)
    for entry in lock["files"]:
        print(f"{entry['name']}: OK at {prepare_u2net(cache / 'u2net', arguments.seed_u2net, entry)}", flush=True)
    for problem in problems:
        print("PROBLEM", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(prepare_models())
