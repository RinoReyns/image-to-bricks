"""Read the lock files of the GPU setup (lock/images_to_model/) and check an installed environment and the model
cache against them."""

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from image_to_bricks.images_to_model.type_aliases import JsonDict

HASH_CHUNK_BYTES = 16 * 1024 * 1024
SHA256_HEX_LENGTH = 64  # Hugging Face blobs of large files are named by their sha256
GIT_SHA1_HEX_LENGTH = 40  # ... of small files by their git blob sha1
MIN_REPO_PATH_PARTS = 2  # "owner/name"
LOCAL_BUILD = "@ local"


class HashDigest(Protocol):
    """A hashlib digest."""

    def update(self, data: bytes, /) -> None:
        """Add data."""
        ...

    def hexdigest(self) -> str:
        """The digest as hex text."""
        ...


class PackageMetadata(Protocol):
    """Metadata fields of a distribution."""

    def __getitem__(self, key: str, /) -> str: ...


class Distribution(Protocol):
    """The part of an importlib.metadata.Distribution the checks use."""

    @property
    def metadata(self) -> PackageMetadata:
        """Package metadata ("Name", ...)."""
        ...

    @property
    def version(self) -> str:
        """Installed version."""
        ...

    def read_text(self, filename: str) -> str | None:
        """Text of a metadata file, None when missing."""
        ...


def read_lock_rows(path: Path) -> list[list[str]]:
    """Non-empty, non-comment lines of a lock file, split on whitespace (inline "# ..." comments removed)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.split(" #", 1)[0].strip()
        if text and not text.startswith("#"):
            rows.append(text.split())
    return rows


def normalize_name(name: str) -> str:
    """Distribution name as pip compares it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_package_line(line: str) -> tuple[str, str]:
    """'name==1.0' -> (name, '1.0'); 'name @ <commit>' / 'name @ local' -> (name, '@ <commit>' / '@ local')."""
    if " @ " in line:
        name, _, spec = line.partition(" @ ")
        return normalize_name(name.strip()), "@ " + spec.strip()
    name, _, version = line.partition("==")
    return normalize_name(name.strip()), version.strip()


def read_expected_packages(paths: Sequence[Path]) -> dict[str, str]:
    """{name: version or '@ ...'} from requirement / lock files. Raises ValueError when two files pin one package to
    different versions."""
    expected: dict[str, str] = {}
    for path in paths:
        for row in read_lock_rows(path):
            name, spec = parse_package_line(" ".join(row))
            if expected.get(name, spec) != spec:
                raise ValueError(f"{path}: {name} {spec}, another lock file says {expected[name]}")
            expected[name] = spec
    return expected


def describe_distribution(name: str, version: str, direct_url: JsonDict | None) -> tuple[str, str]:
    """How an installed package is listed in the lock files; direct_url: its parsed direct_url.json or None."""
    if not direct_url:
        return normalize_name(name), version
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    return normalize_name(name), f"@ {commit}" if commit else LOCAL_BUILD


def list_installed_packages(distributions: Iterable[Distribution]) -> dict[str, str]:
    """{name: version or '@ ...'} of the installed distributions (importlib.metadata.distributions())."""
    installed = {}
    for distribution in distributions:
        text = distribution.read_text("direct_url.json")
        name, spec = describe_distribution(
            distribution.metadata["Name"], distribution.version, json.loads(text) if text else None
        )
        installed[name] = spec
    return installed


def compare_packages(expected: dict[str, str], installed: dict[str, str]) -> list[str]:
    """Problems as text lines; empty = the environment is exactly the locked one."""
    problems = [f"missing: {name} {spec}" for name, spec in sorted(expected.items()) if name not in installed]
    problems += [f"not in lock: {name} {spec}" for name, spec in sorted(installed.items()) if name not in expected]
    problems += [
        f"different: {name} is {installed[name]}, lock says {spec}"
        for name, spec in sorted(expected.items())
        if name in installed and installed[name] != spec
    ]
    return problems


def read_model_lock(path: Path) -> JsonDict:
    """{'hf': [{repo, revision, files}], 'files': [{name, sha256, url}], 'unused': {repo: reason}}. Raises ValueError
    for an unknown row kind."""
    lock: JsonDict = {"hf": [], "files": [], "unused": {}}
    for row in read_lock_rows(path):
        if row[0] == "hf":
            lock["hf"].append({"repo": row[1], "revision": row[2], "files": row[3]})
        elif row[0] == "file":
            lock["files"].append({"name": row[1], "sha256": row[2], "url": row[3]})
        elif row[0] == "unused":
            lock["unused"][row[1]] = " ".join(row[2:])
        else:
            raise ValueError(f"unknown model lock row: {' '.join(row)}")
    return lock


def collect_strings(value: object) -> list[str]:
    """Every string inside nested dicts / lists."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [text for item in value for text in collect_strings(item)]
    return []


def is_local_reference(text: str, local_files: Sequence[str]) -> bool:
    """True when text names a file (or file prefix) inside the snapshot itself."""
    return any(name == text or name.startswith((text + ".", text + "/")) for name in local_files)


def find_pipeline_references(pipeline: JsonDict, local_files: Sequence[str]) -> dict[str, list[str]]:
    """Other Hugging Face repos a TRELLIS.2 pipeline.json points to: {repo: [path prefix, ...]} ('' = whole repo).
    Strings that name a file (or file prefix) inside the snapshot itself are local, not references."""
    references: dict[str, list[str]] = {}
    for text in collect_strings(pipeline):
        parts = text.split("/")
        if len(parts) < MIN_REPO_PATH_PARTS or " " in text or text.startswith((".", "/")) or not all(parts):
            continue
        if is_local_reference(text, local_files):
            continue
        prefixes = references.setdefault("/".join(parts[:MIN_REPO_PATH_PARTS]), [])
        prefix = "/".join(parts[MIN_REPO_PATH_PARTS:])
        if prefix not in prefixes:
            prefixes.append(prefix)
    return references


def build_patterns(prefixes: Sequence[str]) -> list[str] | None:
    """allow_patterns for snapshot_download; None = the whole repo."""
    if "" in prefixes:
        return None
    return [pattern for prefix in prefixes for pattern in (f"{prefix}.*", f"{prefix}/*")]


def check_references(references: dict[str, list[str]], lock: JsonDict) -> list[str]:
    """Problems: repos pipeline.json refers to that the lock neither pins nor lists as unused."""
    pinned = {entry["repo"] for entry in lock["hf"]}
    return [
        f"pipeline.json refers to {repo} ({', '.join(prefixes) or 'whole repo'}), which is not pinned in the lock"
        for repo, prefixes in sorted(references.items())
        if repo not in pinned and repo not in lock["unused"]
    ]


def hash_stream(path: Path, digest: HashDigest) -> str:
    """Feed a file to digest in chunks; returns the hex digest."""
    with open(path, "rb") as handle:
        block = handle.read(HASH_CHUNK_BYTES)
        while block:
            digest.update(block)
            block = handle.read(HASH_CHUNK_BYTES)
    return digest.hexdigest()


def hash_sha256(path: Path) -> str:
    """sha256 of a file."""
    return hash_stream(path, hashlib.sha256())


def hash_git_blob(path: Path) -> str:
    """git blob sha1 of a file."""
    digest = hashlib.sha1()  # git's blob id, not a security hash
    digest.update(f"blob {Path(path).stat().st_size}\0".encode())
    return hash_stream(path, digest)


def check_blob(path: Path) -> bool:
    """A Hugging Face cache blob is named after its content: sha256 (large files) or git blob sha1 (small files)."""
    name = Path(path).name
    if len(name) == SHA256_HEX_LENGTH:
        return hash_sha256(path) == name
    if len(name) == GIT_SHA1_HEX_LENGTH:
        return hash_git_blob(path) == name
    return False


def check_snapshot(snapshot: Path) -> list[str]:
    """Files of a snapshot whose content does not match their blob name."""
    return [
        str(item.relative_to(snapshot))
        for item in sorted(Path(snapshot).rglob("*"))
        if item.is_file() and not check_blob(item.resolve())
    ]
