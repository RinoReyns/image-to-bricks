"""The production engines: TRELLIS.2 shape model, rembg u2net background removal, fast-simplification and a torch
GPU memory probe. The GPU libraries are imported only when load_engines runs (scripts/setup_images_to_model.sh
installs them), so the rest of the package and its tests work without them.

Licence notes for this engine:
- TRELLIS.2 code and weights: MIT.
- Image encoder facebook/dinov3-vitl16-pretrain-lvd1689m: DINOv3 License (custom; needs a "Built with DINOv3" credit
  and accepting the licence on Hugging Face).
- RMBG-2.0 background removal (CC BY-NC) is NOT loaded: it is replaced by UnusedBackgroundRemover, because our own
  u2net cut-out always hands TRELLIS.2 an RGBA picture with transparent pixels.
- nvdiffrast / nvdiffrec (NVIDIA Source Code License, research only) are neither installed nor loaded: only the shape
  part of TRELLIS.2 runs, and the package __init__ files that would import them are skipped.
"""

import gc
import importlib
import importlib.util
import os
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, Protocol, Self

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image

from image_to_bricks.images_to_model.engines import Engines
from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.type_aliases import AnyArray, FloatArray, IntArray
from image_to_bricks.images_to_model.views import FRONT_VIEW

TRELLIS_REPO = "microsoft/TRELLIS.2-4B"
PIPELINE_TYPE = "1024_cascade"
REMBG_MODEL = "u2net"
BACKEND_ENVIRONMENT = {
    "ATTN_BACKEND": "xformers",
    "SPARSE_ATTN_BACKEND": "xformers",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_CACHE_FOLDER = ".models"  # made by scripts/setup_images_to_model.sh: pinned model files, used offline
OFFLINE_ENVIRONMENT = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
HOPPER_MAJOR_CAPABILITY = 9  # sm_90, the only GPUs xformers' FlashAttention-3 kernels run on
UNREACHABLE_CAPABILITY = (99, 0)
LEAN_PACKAGES = ("trellis2", "o_voxel", "cumesh")
OPAQUE_ALPHA = 255


class UnusedBackgroundRemover:
    """Stands in for TRELLIS.2's RMBG-2.0 loader so those weights are never downloaded or run."""

    def __init__(self, *_arguments: object, **options: object) -> None:
        self.model_name = options.get("model_name")

    def to(self, *_arguments: object, **_options: object) -> Self:
        """No device to move to."""
        return self

    def cuda(self) -> Self:
        """No device to move to."""
        return self

    def cpu(self) -> Self:
        """No device to move to."""
        return self

    def __call__(self, *_arguments: object) -> NoReturn:
        """Always fails: TRELLIS.2 must get pictures that are already cut out."""
        raise RuntimeError("TRELLIS.2 background removal is disabled: pass an RGBA cut-out with transparency")


def use_project_model_cache(project_dir: Path = PROJECT_DIR) -> bool:
    """When <project>/.models exists, read the models from there and offline (only the pinned files). Must run before
    huggingface_hub is imported; variables already set are kept. Returns True if the cache was found."""
    cache = Path(project_dir) / MODEL_CACHE_FOLDER
    if not cache.is_dir():
        return False
    os.environ.setdefault("HF_HOME", str(cache / "hf"))
    os.environ.setdefault("U2NET_HOME", str(cache / "u2net"))
    for name, value in OFFLINE_ENVIRONMENT.items():
        os.environ.setdefault(name, value)
    return True


def limit_flash3_to_hopper(flash3_module: object, capability: tuple[int, int]) -> bool:
    """xformers 0.0.31 dispatches its FlashAttention-3 kernels (Hopper only, sm_90) on any GPU >= 9.0, which fails on
    Blackwell (sm_120) with 'flash_fwd_launch_template.h: invalid argument'. Raising their minimum capability makes
    xformers use FlashAttention-2 / CUTLASS instead. Returns True if it changed anything."""
    if capability[0] == HOPPER_MAJOR_CAPABILITY:
        return False
    changed = False
    for name in ("FwOp", "BwOp"):
        operator = getattr(flash3_module, name, None)
        if operator is not None:
            operator.CUDA_MINIMUM_COMPUTE_CAPABILITY = UNREACHABLE_CAPABILITY
            changed = True
    return changed


def disable_hopper_only_attention() -> bool:
    """limit_flash3_to_hopper for the first GPU. Returns True if it changed anything."""
    torch = importlib.import_module("torch")
    flash3 = importlib.import_module("xformers.ops.fmha.flash3")
    major, minor = torch.cuda.get_device_capability(0)
    return limit_flash3_to_hopper(flash3, (int(major), int(minor)))


def register_package_without_init(name: str) -> bool:
    """Make `name` importable as a package without running its __init__.py. trellis2/__init__.py imports
    `renderers`, which loads nvdiffrast even when only the shape pipeline is used; registered this way, Python loads
    only the submodules that are actually imported. Returns True if it registered the package. Raises ImportError
    when the package is not installed."""
    if name in sys.modules:
        return False
    spec = importlib.util.find_spec(name)
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError(f"package {name} not found")
    package = types.ModuleType(name)
    package.__path__ = list(spec.submodule_search_locations)
    package.__file__ = spec.origin
    package.__spec__ = spec
    sys.modules[name] = package
    return True


def load_without_restricted_modules() -> None:
    """Shape-only TRELLIS.2 without NVIDIA research-only code: skip the __init__ of trellis2 (imports renderers ->
    nvdiffrast), o_voxel (imports postprocess -> nvdiffrast, cumesh) and cumesh (imports bvh -> cubvh, which contains
    instant-ngp code under the NVIDIA Source Code License). Only the MIT parts the shape path calls are loaded:
    o_voxel.convert (mesh extraction) and cumesh.cumesh.CuMesh (fill_holes)."""
    for name in LEAN_PACKAGES:
        register_package_without_init(name)
    cumesh_core = importlib.import_module("cumesh.cumesh")
    vars(sys.modules["cumesh"])["CuMesh"] = cumesh_core.CuMesh


def check_rmbg_disabled() -> bool:
    """True if TRELLIS.2's RMBG-2.0 loader is replaced by UnusedBackgroundRemover (or TRELLIS.2 is not loaded)."""
    module = sys.modules.get("trellis2.pipelines.rembg")
    return module is None or getattr(module, "BiRefNet", None) is UnusedBackgroundRemover


def prepare_trellis_environment() -> None:
    """Pick the attention backends (before trellis2 is imported), load trellis2 without its research-only parts,
    keep xformers off the Hopper-only kernels and disable RMBG-2.0."""
    for name, value in BACKEND_ENVIRONMENT.items():
        os.environ.setdefault(name, value)
    load_without_restricted_modules()
    disable_hopper_only_attention()
    trellis_rembg = importlib.import_module("trellis2.pipelines.rembg")
    vars(trellis_rembg)["BiRefNet"] = UnusedBackgroundRemover


def convert_z_up_to_y_up(vertices: AnyArray) -> FloatArray:
    """TRELLIS.2 works z-up; GLB and our placement step are y-up (the same swap as o_voxel.to_glb). Returns a copy."""
    converted = np.asarray(vertices, dtype=np.float64).copy()
    converted[:, 1] = vertices[:, 2]
    converted[:, 2] = -vertices[:, 1]
    return converted


def require_transparency(image: Image.Image) -> Image.Image:
    """The picture as RGBA. Raises ValueError when no pixel is transparent (TRELLIS.2 would need RMBG-2.0)."""
    alpha = np.asarray(image.convert("RGBA"))[..., 3]
    if not (alpha < OPAQUE_ALPHA).any():
        raise ValueError("TRELLIS.2 needs an RGBA cut-out with transparent pixels (RMBG-2.0 is disabled)")
    return image.convert("RGBA")


class TensorLike(Protocol):
    """The part of a torch tensor the adapter uses."""

    def detach(self) -> "TensorLike":
        """Tensor without gradient history."""
        ...

    def cpu(self) -> "TensorLike":
        """Tensor in main memory."""
        ...

    def numpy(self) -> AnyArray:
        """Values as a numpy array."""
        ...


class TrellisMesh(Protocol):
    """A mesh TRELLIS.2 returns (torch tensors, z up)."""

    vertices: TensorLike
    faces: TensorLike


class TrellisImageTo3D(Protocol):
    """The part of TRELLIS.2's Trellis2ImageTo3DPipeline the adapter uses."""

    def run(self, image: Image.Image, *, seed: int, pipeline_type: str) -> Sequence[TrellisMesh]:
        """Meshes for one RGBA picture."""
        ...


class TrellisShapeGenerator:
    """TRELLIS.2-4B shape from the front picture (the model takes one picture)."""

    view_names: tuple[str, ...] = (FRONT_VIEW,)

    def __init__(self, pipeline: TrellisImageTo3D, pipeline_type: str = PIPELINE_TYPE) -> None:
        self.pipeline = pipeline
        self.pipeline_type = pipeline_type

    @classmethod
    def load(cls) -> Self:
        """Load TRELLIS.2-4B onto the GPU (shape part only)."""
        prepare_trellis_environment()
        trellis_pipelines = importlib.import_module("trellis2.pipelines.trellis2_image_to_3d")
        pipeline = trellis_pipelines.Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS_REPO)
        pipeline.cuda()
        return cls(pipeline)

    def generate(self, images: dict[str, Image.Image], seed: int) -> Mesh:
        """Raw mesh (y up) from the front cut-out."""
        front = require_transparency(images[FRONT_VIEW])
        result = self.pipeline.run(front, seed=seed, pipeline_type=self.pipeline_type)[0]
        vertices = result.vertices.detach().cpu().numpy()
        faces = result.faces.detach().cpu().numpy().astype(np.int64)
        return Mesh(convert_z_up_to_y_up(vertices), faces)


class RembgRemover:
    """rembg background removal with the u2net model (MIT code, Apache-2.0 weights)."""

    def __init__(self, model_name: str = REMBG_MODEL) -> None:
        self.rembg = importlib.import_module("rembg")
        self.session = self.rembg.new_session(model_name)

    def remove(self, image: Image.Image) -> Image.Image:
        """RGBA picture with the background made transparent."""
        removed = self.rembg.remove(image, session=self.session)
        if not isinstance(removed, Image.Image):
            raise TypeError(f"rembg returned {type(removed).__name__} for a PIL picture")
        return removed


class TorchMemoryProbe:
    """Peak GPU memory through torch.cuda."""

    def __init__(self) -> None:
        self.torch = importlib.import_module("torch")

    def reset(self) -> None:
        """Start measuring a new step."""
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def read_peak_bytes(self) -> int:
        """Peak bytes allocated since the last reset."""
        self.torch.cuda.synchronize()
        return int(self.torch.cuda.max_memory_allocated())


def simplify_with_fast_simplification(
    vertices: FloatArray, faces: IntArray, reduction: float
) -> tuple[ArrayLike, ArrayLike]:
    """Quadric decimation removing `reduction` (0-0.999) of the faces. Returns (vertices, faces)."""
    fast_simplification = importlib.import_module("fast_simplification")
    reduction = float(np.clip(reduction, 0.0, 0.999))
    reduced_vertices, reduced_faces = fast_simplification.simplify(
        np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32), target_reduction=reduction
    )
    return reduced_vertices, reduced_faces


def release_gpu_memory() -> None:
    """Free cached GPU memory (after the models are no longer needed, or after a failed case)."""
    gc.collect()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    torch.cuda.empty_cache()


def load_engines() -> Engines:
    """The production engines (loads TRELLIS.2 onto the GPU, about a minute). Uses the project's model cache when
    scripts/setup_images_to_model.sh made one."""
    use_project_model_cache()
    return Engines(
        remover=RembgRemover(),
        shape=TrellisShapeGenerator.load(),
        simplify=simplify_with_fast_simplification,
        probe=TorchMemoryProbe(),
    )
