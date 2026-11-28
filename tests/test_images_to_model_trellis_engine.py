"""The TRELLIS.2 adapter without a GPU: axis swap, RMBG-2.0 stand-in, transparency check, Blackwell attention fix,
loading a package without its __init__, the project model cache, and the pipeline with a front-only shape model."""

import os
import sys
import types
from pathlib import Path

import numpy as np
import support
from images_to_model_fakes import (
    FakeFrontOnlyShapeGenerator,
    FakeProbe,
    FakeRemover,
    FakeRescalingPainter,
    simplify_to_coarser_ellipsoid,
)
from images_to_model_helpers import assert_raises, is_close
from PIL import Image
from test_images_to_model_pipeline import write_views

from image_to_bricks.images_to_model.engines import Engines
from image_to_bricks.images_to_model.pipeline import ModelOutput, ModelSettings, run_pipeline
from image_to_bricks.images_to_model.provenance import list_restricted_modules
from image_to_bricks.images_to_model.trellis_engine import (
    OFFLINE_ENVIRONMENT,
    UNREACHABLE_CAPABILITY,
    UnusedBackgroundRemover,
    check_rmbg_disabled,
    convert_z_up_to_y_up,
    limit_flash3_to_hopper,
    register_package_without_init,
    require_transparency,
    use_project_model_cache,
)
from image_to_bricks.images_to_model.views import find_views_in_folder

CACHE_VARIABLES = ("HF_HOME", "U2NET_HOME", *OFFLINE_ENVIRONMENT)


def test_z_up_becomes_y_up_like_o_voxel() -> None:
    vertices = np.array([[1.0, 2.0, 3.0]])
    assert convert_z_up_to_y_up(vertices).tolist() == [[1.0, 3.0, -2.0]]
    assert vertices.tolist() == [[1.0, 2.0, 3.0]]


def test_the_rmbg_stand_in_never_runs() -> None:
    remover = UnusedBackgroundRemover(model_name="briaai/RMBG-2.0")
    assert remover.model_name == "briaai/RMBG-2.0"
    assert remover.to("cuda") is remover and remover.cpu() is remover and remover.cuda() is remover
    assert check_rmbg_disabled()  # TRELLIS.2 is not loaded in this process
    with assert_raises(RuntimeError, "disabled"):
        remover(Image.new("RGB", (4, 4)))


def test_trellis_needs_a_transparent_cut_out() -> None:
    with assert_raises(ValueError, "transparent"):
        require_transparency(Image.new("RGBA", (4, 4), (1, 2, 3, 255)))
    assert require_transparency(Image.new("RGBA", (4, 4), (1, 2, 3, 0))).mode == "RGBA"


class FakeOperator:
    """An xformers attention operator class."""

    CUDA_MINIMUM_COMPUTE_CAPABILITY = (9, 0)


def test_flash3_is_blocked_on_blackwell_but_not_on_hopper() -> None:
    hopper = types.SimpleNamespace(FwOp=type("FwOp", (FakeOperator,), {}))
    assert not limit_flash3_to_hopper(hopper, (9, 0))
    assert hopper.FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY == (9, 0)
    blackwell = types.SimpleNamespace(FwOp=type("FwOp", (FakeOperator,), {}), BwOp=type("BwOp", (FakeOperator,), {}))
    assert limit_flash3_to_hopper(blackwell, (12, 0))
    assert blackwell.FwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY == UNREACHABLE_CAPABILITY
    assert blackwell.BwOp.CUDA_MINIMUM_COMPUTE_CAPABILITY == UNREACHABLE_CAPABILITY


def test_a_package_is_registered_without_running_its_init() -> None:
    folder = Path(support.make_temp_dir("lean_package"))
    package = folder / "fakepkg_lean"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('__init__ must not run')\n", encoding="utf-8")
    (package / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (package / "sub" / "leaf.py").write_text("VALUE = 7\n", encoding="utf-8")
    sys.path.insert(0, str(folder))
    try:
        assert register_package_without_init("fakepkg_lean")
        assert __import__("fakepkg_lean.sub.leaf", fromlist=["VALUE"]).VALUE == 7
        assert not register_package_without_init("fakepkg_lean")  # already registered
        with assert_raises(ImportError, "not found"):
            register_package_without_init("no_such_package_xyz")
    finally:
        sys.path.remove(str(folder))
        for name in [name for name in sys.modules if name.startswith("fakepkg_lean")]:
            del sys.modules[name]


def test_restricted_modules_loaded_in_this_process_are_found() -> None:
    sys.modules["nvdiffrast.torch"] = types.ModuleType("nvdiffrast.torch")
    try:
        assert "nvdiffrast.torch" in list_restricted_modules() and "numpy" not in list_restricted_modules()
    finally:
        del sys.modules["nvdiffrast.torch"]


def test_the_project_model_cache_is_used_offline_when_it_exists() -> None:
    saved = {name: os.environ.pop(name, None) for name in CACHE_VARIABLES}
    try:
        project = Path(support.make_temp_dir("model_cache"))
        assert not use_project_model_cache(project) and "HF_HOME" not in os.environ
        (project / ".models").mkdir()
        assert use_project_model_cache(project)
        assert os.environ["HF_HOME"] == str(project / ".models" / "hf") and os.environ["HF_HUB_OFFLINE"] == "1"
    finally:
        for name, value in saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value


def test_pipeline_places_a_moved_model_and_reports_the_front_only_shape() -> None:
    folder = Path(support.make_temp_dir("trellis_pipeline"))
    engines = Engines(FakeRemover(), FakeFrontOnlyShapeGenerator(), simplify_to_coarser_ellipsoid, FakeProbe())
    painter = FakeRescalingPainter()
    engines.painter = painter
    report = run_pipeline(
        find_views_in_folder(write_views(folder)), ModelOutput(folder / "t.glb"), ModelSettings(1.5), engines
    )
    assert painter.result is not None
    placed = painter.result.vertices
    assert is_close(placed[:, 1].min(), 0.0, abs_tol=1e-9) and is_close(placed[:, 1].max(), 1.5)
    assert is_close(float(np.ptp(placed[:, 0])), report["measurements"]["width"], rel=1e-3)
    assert report["engine"] == "trellis-proj" and report["shape_views"] == ["front"]
    assert report["texture_views"] == ["front", "back", "right"]


if __name__ == "__main__":
    support.run_tests(globals())
