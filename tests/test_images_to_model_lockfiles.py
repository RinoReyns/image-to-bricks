"""Lock files of the GPU setup: installed packages against the locks, model lock rows, pipeline.json references,
Hugging Face blob hashes, and the shipped lock files agreeing with requirements.txt."""

import hashlib
import json
from pathlib import Path

import support
from images_to_model_helpers import assert_raises

from image_to_bricks.images_to_model.type_aliases import JsonDict
from image_to_bricks.images_to_model.verification.lockfiles import (
    build_patterns,
    check_blob,
    check_references,
    check_snapshot,
    compare_packages,
    find_pipeline_references,
    list_installed_packages,
    read_expected_packages,
    read_model_lock,
)

LOCK_DIR = Path(support.PROJECT_DIR) / "lock" / "images_to_model"


class FakeDistribution:
    """An installed distribution: name, version and its direct_url.json (None for a plain PyPI install)."""

    def __init__(self, name: str, version: str, direct_url: JsonDict | None = None) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        """Text of a metadata file."""
        return json.dumps(self.direct_url) if filename == "direct_url.json" and self.direct_url else None


def test_installed_packages_match_the_locks() -> None:
    folder = Path(support.make_temp_dir("lockfiles_packages"))
    (folder / "a.txt").write_text("# comment\nnumpy==2.2.6\n    # via something\ncumesh @ abc123\n", encoding="utf-8")
    (folder / "b.txt").write_text(
        "o-voxel @ local  # built from .deps\npyyaml==6.0.4\nnumpy==2.2.6\n", encoding="utf-8"
    )
    installed = list_installed_packages(
        [
            FakeDistribution("numpy", "2.2.6"),
            FakeDistribution(
                "CuMesh", "0.0.1", {"url": "file:///x", "vcs_info": {"vcs": "git", "commit_id": "abc123"}}
            ),
            FakeDistribution("o_voxel", "0.1", {"url": "file:///y", "dir_info": {}}),
            FakeDistribution("PyYAML", "6.0.4"),
        ]
    )
    assert compare_packages(read_expected_packages([folder / "a.txt", folder / "b.txt"]), installed) == []


def test_two_locks_pinning_different_versions_are_an_error() -> None:
    folder = Path(support.make_temp_dir("lockfiles_conflict"))
    (folder / "a.txt").write_text("numpy==2.2.6\n", encoding="utf-8")
    (folder / "b.txt").write_text("numpy==2.4.6\n", encoding="utf-8")
    with assert_raises(ValueError, "numpy"):
        read_expected_packages([folder / "a.txt", folder / "b.txt"])


def test_package_differences_are_listed() -> None:
    problems = compare_packages({"numpy": "2.4.6", "torch": "2.7.1+cu128"}, {"numpy": "2.4.5", "nvdiffrast": "@ local"})
    assert problems == [
        "missing: torch 2.7.1+cu128",
        "not in lock: nvdiffrast @ local",
        "different: numpy is 2.4.5, lock says 2.4.6",
    ]


def test_model_lock_and_pipeline_references() -> None:
    lock_file = Path(support.make_temp_dir("lockfiles_models")) / "models.txt"
    lock_file.write_text(
        "hf microsoft/TRELLIS.2-4B aaa *\nhf microsoft/TRELLIS-image-large bbb from-pipeline\n"
        "file u2net.onnx ccc https://x/u2net.onnx\nunused briaai/RMBG-2.0 not used\n",
        encoding="utf-8",
    )
    lock = read_model_lock(lock_file)
    assert lock["hf"][1] == {"repo": "microsoft/TRELLIS-image-large", "revision": "bbb", "files": "from-pipeline"}
    assert lock["unused"] == {"briaai/RMBG-2.0": "not used"}
    pipeline = {
        "name": "Trellis2ImageTo3DPipeline",
        "args": {
            "models": {"flow": "ckpts/ss_flow_img", "decoder": "microsoft/TRELLIS-image-large/ckpts/ss_dec_16l8"},
            "image_cond_model": {"name": "DinoV3", "args": {"model_name": "facebook/dinov3-vitl16"}},
            "rembg_model": {"args": {"model_name": "briaai/RMBG-2.0"}},
            "pipeline_type": "1024_cascade",
        },
    }
    local = ["pipeline.json", "ckpts/ss_flow_img.json", "ckpts/ss_flow_img.safetensors"]
    references = find_pipeline_references(pipeline, local)
    assert references == {
        "microsoft/TRELLIS-image-large": ["ckpts/ss_dec_16l8"],
        "facebook/dinov3-vitl16": [""],
        "briaai/RMBG-2.0": [""],
    }
    assert build_patterns(["ckpts/ss_dec_16l8"]) == ["ckpts/ss_dec_16l8.*", "ckpts/ss_dec_16l8/*"]
    assert build_patterns([""]) is None
    assert check_references(references, lock) == [
        "pipeline.json refers to facebook/dinov3-vitl16 (whole repo), which is not pinned in the lock"
    ]


def test_blob_hashes() -> None:
    folder = Path(support.make_temp_dir("lockfiles_blobs"))
    content = b"weights"
    large = folder / hashlib.sha256(content).hexdigest()
    large.write_bytes(content)
    small = folder / hashlib.sha1(b"blob 7\0" + content).hexdigest()  # git's blob id
    small.write_bytes(content)
    assert check_blob(large) and check_blob(small)
    (folder / ("0" * 64)).write_bytes(content)
    assert not check_blob(folder / ("0" * 64))
    snapshot = folder / "snap"
    snapshot.mkdir()
    (snapshot / "ok.bin").write_bytes(content)
    try:
        (snapshot / "ok.bin").unlink()
        (snapshot / "ok.bin").symlink_to(large)
        (snapshot / "bad.bin").symlink_to(folder / ("0" * 64))
    except OSError as error:  # Windows without developer mode cannot make symlinks
        raise support.SkipTest(f"symlinks not available: {error}") from error
    assert check_snapshot(snapshot) == ["bad.bin"]


def test_the_shipped_lock_files_agree_with_each_other() -> None:
    project = Path(support.PROJECT_DIR)
    files = [
        project / "requirements.txt",
        project / "requirements-dev.txt",
        LOCK_DIR / "requirements-gpu.txt",
        LOCK_DIR / "torch.txt",
        LOCK_DIR / "builds.txt",
    ]
    expected = read_expected_packages(files)  # raises when two files pin a package differently
    assert expected["torch"].endswith("+cu128") and expected["cumesh"] == "@ local"
    assert "nvdiffrast" not in expected and "opencv-python-headless" not in expected
    assert read_model_lock(LOCK_DIR / "models.txt")["hf"][0]["repo"] == "microsoft/TRELLIS.2-4B"


if __name__ == "__main__":
    support.run_tests(globals())
