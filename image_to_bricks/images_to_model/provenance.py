"""Provenance and licence package for every generated model.

Next to X.glb every run writes X_package/ (X.glb, PROVENANCE.json, NOTICE.md) and X_package.zip, and embeds a short
provenance note in the GLB itself (asset.generator / asset.copyright / asset.extras).

The licence facts below are a reading of each project's licence text (checked 2026-11-21), not legal advice.
"""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from image_to_bricks.images_to_model.type_aliases import JsonDict
from image_to_bricks.images_to_model.views import ViewPictures

ENGINE_NAME = "trellis-proj"  # TRELLIS.2 shape + colours projected from the pictures
PROJECT_URL = "https://github.com/RinoReyns/image-to-bricks"
PROJECT_LICENCE = "GPL-3.0-or-later"
SOURCE_FILE = "SOURCE.json"
UNKNOWN = "unknown"
EVALUATION_ONLY = "evaluation-only"
PERMISSIVE = "permissive-with-attribution"
GIT_TIMEOUT_SECONDS = 10
SHORT_HASH_LENGTH = 12
WEIGHTS_HASH_LENGTH = 16
GLB_MAGIC = b"glTF"
GLB_HEADER = struct.Struct("<4sII")  # magic, version, total length
GLB_CHUNK_HEADER = struct.Struct("<I4s")  # chunk length, chunk type
GLB_ALIGNMENT = 4  # bytes


@dataclass(frozen=True)
class Component:
    """One model or library a run uses: what it does, its licence and where its version is read from (installed
    package, weights file, Hugging Face repo or git checkout of a Python package). restriction: "none" or
    "evaluation-only"; condition / attribution: what the licence asks for."""

    role: str
    licence: str
    url: str
    restriction: str = "none"
    package: tuple[str, ...] = ()
    weights_file: str | None = None
    hf_repo: str | None = None
    git_module: str | None = None
    condition: str | None = None
    attribution: str | None = None

    def to_json(self, key: str) -> JsonDict:
        """The component as listed in PROVENANCE.json (fields that are not set are left out)."""
        package: str | list[str] | None = None
        if self.package:
            package = self.package[0] if len(self.package) == 1 else list(self.package)
        fields = {
            "role": self.role,
            "licence": self.licence,
            "url": self.url,
            "restriction": self.restriction,
            "package": package,
            "weights_file": self.weights_file,
            "hf_repo": self.hf_repo,
            "git_module": self.git_module,
            "condition": self.condition,
            "attribution": self.attribution,
        }
        return dict({name: value for name, value in fields.items() if value is not None}, key=key)


NVIDIA_RESEARCH_ONLY = "Non-commercial: research or evaluation purposes only."
COMPONENTS = {
    "image-to-bricks": Component(
        "pipeline (cut-out, clean, colour projection, place, packaging)",
        PROJECT_LICENCE,
        PROJECT_URL,
        git_module="image_to_bricks",
    ),
    "rembg": Component("background removal library", "MIT", "https://github.com/danielgatis/rembg", package=("rembg",)),
    "u2net": Component(
        "background removal weights (u2net.onnx)",
        "Apache-2.0",
        "https://github.com/xuebinqin/U-2-Net",
        weights_file="u2net.onnx",
    ),
    "onnxruntime": Component(
        "runs the u2net weights",
        "MIT",
        "https://github.com/microsoft/onnxruntime",
        package=("onnxruntime", "onnxruntime-gpu"),
    ),
    "trimesh": Component("mesh I/O", "MIT", "https://github.com/mikedh/trimesh", package=("trimesh",)),
    "fast-simplification": Component(
        "mesh reduction", "MIT", "https://github.com/pyvista/fast-simplification", package=("fast-simplification",)
    ),
    "numpy": Component("arrays", "BSD-3-Clause", "https://github.com/numpy/numpy", package=("numpy",)),
    "scipy": Component(
        "distance fields, filters", "BSD-3-Clause", "https://github.com/scipy/scipy", package=("scipy",)
    ),
    "scikit-image": Component(
        "marching cubes, image regions",
        "BSD-3-Clause",
        "https://github.com/scikit-image/scikit-image",
        package=("scikit-image",),
    ),
    "pillow": Component("image I/O", "MIT-CMU (HPND)", "https://github.com/python-pillow/Pillow", package=("pillow",)),
    "pytorch": Component("GPU runtime", "BSD-3-Clause", "https://github.com/pytorch/pytorch", package=("torch",)),
    "trellis2": Component(
        "shape model (microsoft/TRELLIS.2-4B) + trellis2 code + O-Voxel",
        "MIT",
        "https://github.com/microsoft/TRELLIS.2",
        hf_repo="microsoft/TRELLIS.2-4B",
        git_module="trellis2",
    ),
    "trellis-image-large": Component(
        "sparse-structure decoder used by TRELLIS.2 (microsoft/TRELLIS-image-large)",
        "MIT",
        "https://huggingface.co/microsoft/TRELLIS-image-large",
        hf_repo="microsoft/TRELLIS-image-large",
    ),
    "dinov3": Component(
        "image encoder inside TRELLIS.2 (facebook/dinov3-vitl16-pretrain-lvd1689m)",
        "DINOv3 License (Meta, custom, worldwide)",
        "https://ai.meta.com/resources/models-and-libraries/dinov3-license",
        hf_repo="facebook/dinov3-vitl16-pretrain-lvd1689m",
        condition='Show "Built with DINOv3" wherever results are shown; follow the acceptable use policy.',
        attribution="Built with DINOv3",
    ),
    "cumesh": Component(
        "mesh hole filling (CuMesh core, without its cubvh part)",
        "MIT",
        "https://github.com/JeffreyXiang/CuMesh",
        package=("cumesh",),
    ),
    "flexgemm": Component(
        "sparse convolution kernels", "MIT", "https://github.com/JeffreyXiang/FlexGEMM", package=("flex_gemm",)
    ),
    "xformers": Component(
        "attention kernels", "BSD-3-Clause", "https://github.com/facebookresearch/xformers", package=("xformers",)
    ),
    "utils3d": Component("3D utilities", "MIT", "https://github.com/EasternJournalist/utils3d", package=("utils3d",)),
    "projection-colours": Component(
        "colours projected from the input pictures (image-to-bricks, numpy)", PROJECT_LICENCE, PROJECT_URL
    ),
    # NVIDIA research-only code: never installed or loaded by this engine; listed so a run that finds it loaded
    # (list_restricted_modules) is marked evaluation-only.
    "nvdiffrast": Component(
        "rasteriser (TRELLIS.2 texturing, not used)",
        "NVIDIA Source Code License",
        "https://github.com/NVlabs/nvdiffrast",
        EVALUATION_ONLY,
        package=("nvdiffrast",),
        condition=NVIDIA_RESEARCH_ONLY,
    ),
    "nvdiffrec": Component(
        "material / render utilities (TRELLIS.2 texturing, not used)",
        "NVIDIA Source Code License",
        "https://github.com/NVlabs/nvdiffrec",
        EVALUATION_ONLY,
        condition=NVIDIA_RESEARCH_ONLY,
    ),
    "cubvh": Component(
        "BVH code bundled in CuMesh (contains NVIDIA instant-ngp code)",
        "MIT + NVIDIA Source Code License (instant-ngp parts)",
        "https://github.com/ashawkey/cubvh",
        EVALUATION_ONLY,
        condition="instant-ngp parts: non-commercial, research or evaluation only.",
    ),
}
ENGINE_COMPONENTS = (
    "image-to-bricks", "rembg", "u2net", "onnxruntime", "trimesh", "fast-simplification", "numpy", "scipy",
    "scikit-image", "pillow", "pytorch", "trellis2", "trellis-image-large", "dinov3", "cumesh", "flexgemm",
    "xformers", "utils3d", "projection-colours",
)  # fmt: skip
# part of a loaded module's name -> component (NVIDIA research / evaluation-only code)
RESTRICTED_MODULE_COMPONENT = {
    "nvdiffrast": "nvdiffrast", "nvdiffrec": "nvdiffrec", "renderutils": "nvdiffrec", "_cubvh": "cubvh",
    "cumesh.bvh": "cubvh",
}  # fmt: skip

RuntimeCollector = Callable[[Sequence[JsonDict], Sequence[str]], JsonDict]


def list_restricted_modules(module_names: Iterable[str] | None = None) -> list[str]:
    """Loaded modules (default: sys.modules) whose licence restricts use to research / evaluation (NVIDIA Source
    Code License). RMBG-2.0 is not listed: rembg and TRELLIS.2 load MIT wrapper code for it, but its weights are only
    fetched when that model runs, which trellis_engine.UnusedBackgroundRemover prevents."""
    names = sys.modules if module_names is None else module_names
    markers = [marker.lower() for marker in RESTRICTED_MODULE_COMPONENT]
    return sorted(name for name in names if any(marker in name.lower() for marker in markers))


def hash_file(path: Path) -> str:
    """sha256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_input_source(folder: Path) -> JsonDict:
    """SOURCE.json of the pictures folder: {"source", "author", "licence", "notes"}; missing -> licence unknown."""
    path = Path(folder) / SOURCE_FILE
    if not path.is_file():
        return {"licence": UNKNOWN, "source": str(folder), "notes": f"no {SOURCE_FILE} in the pictures folder"}
    source: JsonDict = json.loads(path.read_text(encoding="utf-8"))
    return source


def describe_inputs(views: ViewPictures) -> JsonDict:
    """Pictures folder, sha256 of every picture by view name, and the pictures' source / licence."""
    pictures = {name: hash_file(path) for name, path in views.files.items()}
    return {"folder": str(views.source_folder), "sha256": pictures, "source": read_input_source(views.source_folder)}


def list_components(restricted_loaded: Sequence[str]) -> list[JsonDict]:
    """Components of the engine; NVIDIA research-only code found loaded at run time is added too."""
    keys = list(ENGINE_COMPONENTS)
    for module in restricted_loaded:
        for marker, key in RESTRICTED_MODULE_COMPONENT.items():
            if marker in module and key not in keys:
                keys.append(key)
    return [COMPONENTS[key].to_json(key) for key in keys]


def decide_usage(components: Sequence[JsonDict], input_source: JsonDict) -> JsonDict:
    """What the model may be used for, from the strictest component and the input pictures' licence."""
    restrictions = {component["restriction"] for component in components}
    conditions = [f"{c['key']}: {c['condition']}" for c in components if c.get("condition")]
    if EVALUATION_ONLY in restrictions:
        status = EVALUATION_ONLY
        summary = "Evaluation only: NVIDIA research/evaluation-only code was used. Do not publish or use commercially."
    else:
        status = PERMISSIVE
        summary = "Permissive licences only. Keep NOTICE.md with the model and show the required credits."
    attributions = [c["attribution"] for c in components if c.get("attribution")]
    licence = str(input_source.get("licence", UNKNOWN)).strip().lower()
    input_rights_known = licence not in ("", UNKNOWN)
    if not input_rights_known:
        summary += " Input pictures: rights unknown, check before publishing."
    return {
        "status": status,
        "input_rights_known": input_rights_known,
        "summary": summary,
        "attributions": attributions,
        "conditions": conditions,
    }


def build_provenance(report: JsonDict, components: Sequence[JsonDict], inputs: JsonDict, runtime: JsonDict) -> JsonDict:
    """The PROVENANCE.json document of a run (report: run_pipeline's report)."""
    clean = report.get("clean", {})
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_file": Path(report["output"]).name,
        "engine": report.get("engine"),
        "settings": {key: report.get(key) for key in ("height", "seed", "max_faces", "shape_views", "texture_views")},
        "inputs": inputs,
        "components": list(components),
        "runtime": runtime,
        "usage": decide_usage(components, inputs["source"]),
        "checks": {
            "faces": clean.get("faces_out"),
            "watertight": clean.get("watertight"),
            "pieces": clean.get("pieces_out"),
        },
    }


def format_usage_lines(usage: JsonDict) -> list[str]:
    """NOTICE.md lines: usage status, required credits, conditions."""
    lines = ["## What you may do with this model", "", f"**{usage['status']}** — {usage['summary']}", ""]
    if usage["attributions"]:
        lines += ["Required credit wherever the model or images of it are shown:", ""]
        lines += [f"- {text}" for text in usage["attributions"]] + [""]
    if usage["conditions"]:
        lines += ["Conditions:", ""] + [f"- {text}" for text in usage["conditions"]] + [""]
    return lines


def format_input_lines(inputs: JsonDict) -> list[str]:
    """NOTICE.md lines: where the pictures come from and their checksums."""
    source = inputs["source"]
    lines = [
        "## Input pictures",
        "",
        f"- source: {source.get('source', UNKNOWN)}",
        f"- author: {source.get('author', UNKNOWN)}",
        f"- licence: {source.get('licence', UNKNOWN)}",
    ]
    if source.get("notes"):
        lines.append(f"- notes: {source['notes']}")
    return lines + [f"- {name}.png sha256 {digest}" for name, digest in inputs["sha256"].items()] + [""]


def format_notice(provenance: JsonDict) -> str:
    """NOTICE.md: what the model may be used for, required credits, inputs, every component with its version."""
    lines = [
        f"# {provenance['model_file']}",
        "",
        f"Generated {provenance['created']} by image-to-bricks (images_to_model), engine `{provenance['engine']}`.",
        "",
    ]
    lines += format_usage_lines(provenance["usage"]) + format_input_lines(provenance["inputs"])
    lines += [
        "## Models and libraries used",
        "",
        "| Component | Role | Licence | Version / revision | Link |",
        "|---|---|---|---|---|",
    ]
    versions = provenance["runtime"].get("versions", {})
    for component in provenance["components"]:
        version = versions.get(component["key"], "")
        lines.append(
            f"| {component['key']} | {component['role']} | {component['licence']} | {version} | {component['url']} |"
        )
    restricted = provenance["runtime"].get("restricted_modules_loaded") or "none"
    lines += [
        "",
        f"Restricted (NVIDIA research-only) modules loaded during the run: {restricted}.",
        "",
        "This notice records licences as read from each project; it is not legal advice.",
        "",
    ]
    return "\n".join(lines)


def read_package_version(names: Sequence[str]) -> str:
    """Installed version of the first of these distributions that is installed ("" if none is)."""
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return ""


def hash_weights_file(name: str) -> str:
    """Short sha256 of a rembg weights file (in U2NET_HOME, default ~/.u2net); "" when it is not there."""
    path = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net")).expanduser() / name
    return f"sha256 {hash_file(path)[:WEIGHTS_HASH_LENGTH]}" if path.is_file() else ""


def read_hf_revision(repo_id: str) -> str:
    """Revisions of a Hugging Face repo in the model cache ("" without huggingface_hub, a cache or the repo)."""
    try:
        huggingface_hub = importlib.import_module("huggingface_hub")  # only installed with the GPU setup
        hub_errors = importlib.import_module("huggingface_hub.errors")
    except ImportError:
        return ""
    try:
        repos = huggingface_hub.scan_cache_dir().repos
    except (hub_errors.CacheNotFound, OSError):
        return ""
    for repo in repos:
        if repo.repo_id == repo_id:
            commits = sorted(revision.commit_hash[:SHORT_HASH_LENGTH] for revision in repo.revisions)
            return "hf revision " + ", ".join(commits)
    return ""


def read_git_commit(folder: Path) -> str:
    """Short commit of the git checkout that contains folder ("" when it is not one, or git is missing)."""
    try:
        process = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", f"--short={SHORT_HASH_LENGTH}", "HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    commit = process.stdout.strip()
    return f"git {commit}" if process.returncode == 0 and commit else ""


def read_module_commit(module_name: str) -> str:
    """Git commit of the checkout a Python package is imported from ("" when not found)."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.submodule_search_locations:
        return ""
    return read_git_commit(Path(next(iter(spec.submodule_search_locations))).parent)


def describe_component_version(component: JsonDict) -> str:
    """Installed version, weights hash, Hugging Face revision and git commit of a component (those that apply)."""
    found = []
    if component.get("package"):
        package = component["package"]
        found.append(read_package_version([package] if isinstance(package, str) else package))
    if component.get("weights_file"):
        found.append(hash_weights_file(component["weights_file"]))
    if component.get("hf_repo"):
        found.append(read_hf_revision(component["hf_repo"]))
    if component.get("git_module"):
        found.append(read_module_commit(component["git_module"]))
    return ", ".join(item for item in found if item)


def collect_runtime(components: Sequence[JsonDict], restricted_loaded: Sequence[str]) -> JsonDict:
    """Version of every component as found at run time, and the restricted modules that were loaded."""
    versions = {component["key"]: describe_component_version(component) for component in components}
    return {"versions": versions, "restricted_modules_loaded": list(restricted_loaded)}


def embed_glb_metadata(path: Path, generator: str, copyright_text: str, extras: JsonDict) -> None:
    """Write asset.generator / asset.copyright / asset.extras into a GLB's JSON chunk (binary chunk untouched).
    Raises ValueError for a file that is not a GLB."""
    data = Path(path).read_bytes()
    magic, version, _ = GLB_HEADER.unpack(data[: GLB_HEADER.size])
    if magic != GLB_MAGIC:
        raise ValueError(f"{path} is not a GLB file")
    json_start = GLB_HEADER.size + GLB_CHUNK_HEADER.size
    json_length, json_type = GLB_CHUNK_HEADER.unpack(data[GLB_HEADER.size : json_start])
    document = json.loads(data[json_start : json_start + json_length])
    rest = data[json_start + json_length :]
    asset = document.setdefault("asset", {"version": "2.0"})
    asset["generator"] = generator
    asset["copyright"] = copyright_text
    asset.setdefault("extras", {}).update(extras)
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((GLB_ALIGNMENT - len(encoded) % GLB_ALIGNMENT) % GLB_ALIGNMENT)
    body = GLB_CHUNK_HEADER.pack(len(encoded), json_type) + encoded + rest
    Path(path).write_bytes(GLB_HEADER.pack(magic, version, GLB_HEADER.size + len(body)) + body)


def write_package(output: Path, provenance: JsonDict) -> JsonDict:
    """Embed the note in the GLB, then write X_package/ (GLB, PROVENANCE.json, NOTICE.md) and X_package.zip.
    Returns {"folder", "zip", "usage"}."""
    output = Path(output)
    usage = provenance["usage"]
    credits = "; ".join(usage["attributions"])
    note = {
        "engine": provenance["engine"],
        "usage": usage["status"],
        "input_licence": provenance["inputs"]["source"].get("licence", UNKNOWN),
        "components": [component["key"] for component in provenance["components"]],
        "attributions": usage["attributions"],
        "created": provenance["created"],
    }
    copyright_text = f"{usage['status']}: see NOTICE.md" + (f". {credits}" if credits else "")
    embed_glb_metadata(output, f"image-to-bricks ({provenance['engine']})", copyright_text, {"provenance": note})
    folder = output.with_name(output.stem + "_package")
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, folder / output.name)
    (folder / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (folder / "NOTICE.md").write_text(format_notice(provenance), encoding="utf-8")
    archive = output.with_name(output.stem + "_package.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(folder.iterdir()):
            bundle.write(item, f"{folder.name}/{item.name}")
    return {"folder": str(folder), "zip": str(archive), "usage": usage["status"]}


def package_output(
    report: JsonDict,
    views: ViewPictures,
    restricted_loaded: Sequence[str],
    runtime_collector: RuntimeCollector = collect_runtime,
) -> JsonDict:
    """Licence package of a finished run (report: run_pipeline's report). Returns {"folder", "zip", "usage"}."""
    components = list_components(restricted_loaded)
    inputs = describe_inputs(views)
    provenance = build_provenance(report, components, inputs, runtime_collector(components, restricted_loaded))
    return write_package(Path(report["output"]), provenance)
