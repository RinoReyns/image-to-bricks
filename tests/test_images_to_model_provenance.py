"""Licence package of every model: usage status from the strictest component, restricted code found at run time,
input rights, GLB metadata, PROVENANCE.json, NOTICE.md and the zip."""

import json
import os
import struct
import zipfile
from collections.abc import Sequence
from pathlib import Path

import support
from images_to_model_fakes import build_minimal_glb
from images_to_model_helpers import assert_raises, environment_variable
from PIL import Image

from image_to_bricks.images_to_model.provenance import (
    build_provenance,
    decide_usage,
    describe_inputs,
    embed_glb_metadata,
    format_notice,
    hash_weights_file,
    list_components,
    list_restricted_modules,
    package_output,
    read_git_commit,
    read_package_version,
)
from image_to_bricks.images_to_model.type_aliases import JsonDict
from image_to_bricks.images_to_model.views import find_views_in_folder


def read_glb(path: Path) -> tuple[tuple[bytes, int, int], JsonDict, bytes]:
    data = path.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return struct.unpack("<4sII", data[:12]), json.loads(data[20 : 20 + length]), data[20 + length :]


def fake_runtime(components: Sequence[JsonDict], restricted: Sequence[str]) -> JsonDict:
    return {"versions": {c["key"]: "test" for c in components}, "restricted_modules_loaded": list(restricted)}


def write_views(folder: Path, source: JsonDict | None) -> Path:
    folder.mkdir(parents=True)
    Image.new("RGBA", (8, 8)).save(folder / "front.png")
    if source is not None:
        (folder / "SOURCE.json").write_text(json.dumps(source), encoding="utf-8")
    return folder


def test_the_engine_is_permissive_with_the_dinov3_credit() -> None:
    usage = decide_usage(list_components([]), {"licence": "own work"})
    assert usage["status"] == "permissive-with-attribution" and usage["attributions"] == ["Built with DINOv3"]
    assert usage["input_rights_known"]


def test_restricted_code_found_at_run_time_makes_it_evaluation_only() -> None:
    components = list_components(["nvdiffrast.torch", "cumesh._cubvh"])
    keys = [component["key"] for component in components]
    assert "nvdiffrast" in keys and "cubvh" in keys
    assert decide_usage(components, {"licence": "own work"})["status"] == "evaluation-only"


def test_unknown_input_rights_are_flagged() -> None:
    usage = decide_usage(list_components([]), {"licence": "unknown"})
    assert not usage["input_rights_known"] and "rights unknown" in usage["summary"]


def test_restricted_modules_are_found_by_name() -> None:
    names = ["numpy", "nvdiffrast.torch", "cumesh._cubvh", "rembg.sessions.bria_rmbg"]
    assert list_restricted_modules(names) == ["cumesh._cubvh", "nvdiffrast.torch"]


def test_glb_metadata_is_embedded_and_the_binary_chunk_kept() -> None:
    folder = Path(support.make_temp_dir("provenance_glb"))
    path = folder / "m.glb"
    path.write_bytes(build_minimal_glb(b"ABCDEFGH"))
    embed_glb_metadata(path, "gen", "copy", {"provenance": {"engine": "x"}})
    (magic, version, length), document, rest = read_glb(path)
    assert magic == b"glTF" and version == 2 and length == len(path.read_bytes())
    assert document["asset"]["generator"] == "gen" and document["asset"]["extras"]["provenance"]["engine"] == "x"
    assert rest.endswith(b"ABCDEFGH") and len(path.read_bytes()) % 4 == 0
    (folder / "bad.glb").write_bytes(b"nope" + b"\0" * 16)
    with assert_raises(ValueError, "not a GLB"):
        embed_glb_metadata(folder / "bad.glb", "g", "c", {})


def test_package_has_folder_zip_and_notice() -> None:
    folder = Path(support.make_temp_dir("provenance_package"))
    source = {"source": "own drawing", "author": "me", "licence": "own work"}
    views = find_views_in_folder(write_views(folder / "views", source))
    output = folder / "robot.glb"
    output.write_bytes(build_minimal_glb())
    report = {
        "output": str(output),
        "engine": "trellis-proj",
        "height": 1.0,
        "seed": 1,
        "max_faces": 40000,
        "shape_views": ["front"],
        "texture_views": ["front"],
        "clean": {"faces_out": 10, "watertight": True, "pieces_out": 1},
    }
    result = package_output(report, views, [], runtime_collector=fake_runtime)
    package = folder / "robot_package"
    assert result["usage"] == "permissive-with-attribution"
    provenance = json.loads((package / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["inputs"]["source"]["licence"] == "own work" and "front" in provenance["inputs"]["sha256"]
    assert provenance["checks"] == {"faces": 10, "watertight": True, "pieces": 1}
    notice = (package / "NOTICE.md").read_text(encoding="utf-8")
    assert "Built with DINOv3" in notice and "trellis2" in notice and "own drawing" in notice
    assert "image-to-bricks" in notice and "GPL-3.0-or-later" in notice
    usage = read_glb(package / "robot.glb")[1]["asset"]["extras"]["provenance"]["usage"]
    assert usage == "permissive-with-attribution"
    with zipfile.ZipFile(folder / "robot_package.zip") as bundle:
        assert sorted(bundle.namelist()) == [
            "robot_package/NOTICE.md",
            "robot_package/PROVENANCE.json",
            "robot_package/robot.glb",
        ]


def test_notice_without_source_says_the_rights_are_unknown() -> None:
    views = find_views_in_folder(write_views(Path(support.make_temp_dir("provenance_nosource")) / "v", None))
    provenance = build_provenance(
        {"output": "x.glb", "engine": "trellis-proj"}, list_components([]), describe_inputs(views), fake_runtime([], [])
    )
    assert "licence: unknown" in format_notice(provenance)


def test_package_version_tries_the_alternatives() -> None:
    assert read_package_version(["no-such-package-xyz", "numpy"]) != ""
    assert read_package_version(["no-such-package-xyz"]) == ""


def test_weights_file_hash() -> None:
    folder = support.make_temp_dir("provenance_weights")
    with environment_variable("U2NET_HOME", folder):
        assert hash_weights_file("u2net.onnx") == ""
        Path(folder, "u2net.onnx").write_bytes(b"abc")
        assert hash_weights_file("u2net.onnx") == "sha256 ba7816bf8f01cfea"


def test_git_commit_outside_a_checkout_is_empty() -> None:
    assert read_git_commit(Path(support.make_temp_dir("provenance_git"))) == ""
    if os.path.isdir(os.path.join(support.PROJECT_DIR, ".git")):
        assert read_git_commit(Path(support.PROJECT_DIR)).startswith("git ")


if __name__ == "__main__":
    support.run_tests(globals())
