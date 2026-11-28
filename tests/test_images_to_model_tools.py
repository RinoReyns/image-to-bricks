"""The scripts of scripts/reproduce_images_to_model.sh without a GPU: the true shape of an SDF scene, scoring a run
made with fake engines against a true shape, saving and checking the results, and the package check."""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import support
import trimesh
from images_to_model_fakes import build_fake_engines
from images_to_model_helpers import build_icosphere, is_close, require_scikit_image
from test_images_to_model_pipeline import write_views

from image_to_bricks.images_to_model.clean import check_watertight, compute_volume
from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.pipeline import ModelOutput, ModelSettings, run_pipeline
from image_to_bricks.images_to_model.type_aliases import AnyArray
from image_to_bricks.images_to_model.views import find_views_in_folder
from tools.images_to_model import check_packages, check_reproduction, compare_models
from tools.images_to_model.build_truth import build_truth_mesh

CHARACTERS_DIR = Path(support.PUBLIC_FIXTURES_DIR) / "characters"


class FakePrimitive:
    """A scene shape that only has a colour."""

    def __init__(self, colour: tuple[int, int, int]) -> None:
        self.colour = np.array(colour, np.float32) / 255.0


class FakeRenderer:
    """Like sdf_render: one sphere of radius 0.5 at y = 1."""

    @staticmethod
    def make_primitive(colour: tuple[int, int, int]) -> FakePrimitive:
        """A shape with a colour."""
        return FakePrimitive(colour)

    @staticmethod
    def scene_distance(_shapes: list[FakePrimitive], points: AnyArray) -> AnyArray:
        """Distance to the sphere."""
        return np.asarray(np.linalg.norm(points - np.array([0, 1, 0], np.float32), axis=1) - 0.5)

    @staticmethod
    def scene_colour(shapes: list[FakePrimitive], points: AnyArray) -> AnyArray:
        """The first shape's colour everywhere."""
        return np.broadcast_to(shapes[0].colour, (len(points), 3)).copy()


class FakeScene:
    """A scene of one red shape."""

    HALF_EXTENT = 0.8
    CENTER = (0.0, 1.0)

    @staticmethod
    def build(make: object) -> list[object]:
        """One red shape."""
        assert callable(make)
        return [make((255, 0, 0))]


def test_the_true_shape_of_a_sphere_scene_is_closed_outward_and_coloured() -> None:
    require_scikit_image()
    vertices, faces, colours = build_truth_mesh(FakeRenderer(), FakeScene(), 64)
    mesh = Mesh(np.asarray(vertices, dtype=np.float64), faces.astype(np.int64))
    assert check_watertight(mesh.faces)["winding_consistent"]
    assert is_close(compute_volume(mesh), 4 / 3 * np.pi * 0.125, rel=0.03)
    assert is_close(vertices[:, 1].min(), 0.5, abs_tol=0.02) and (colours == [255, 0, 0]).all()


def test_the_character_scenes_build_with_the_real_renderer() -> None:
    from tools.images_to_model.build_truth import load_module  # noqa: PLC0415

    renderer = load_module(CHARACTERS_DIR / "sdf_render.py", "sdf_render")
    for scene_name in ("robot", "frog", "bear"):
        scene = load_module(CHARACTERS_DIR / "scenes" / f"{scene_name}.py", f"scene_{scene_name}")
        shapes = scene.build(renderer.make_primitive)
        centre = np.array([[0.0, scene.CENTER[1], 0.0]], np.float32)
        assert renderer.scene_distance(shapes, centre)[0] < 0, f"{scene_name}: its centre is inside the character"


def make_run(folder: Path) -> Path:
    """A run folder like run_cases.py writes, made with fake engines (case 'demo')."""
    runs = folder / "runs"
    pictures = find_views_in_folder(write_views(folder))
    output = ModelOutput(runs / "demo.glb", runs / "demo_work")
    run_pipeline(pictures, output, ModelSettings(1.0, 500), build_fake_engines())
    summary = {"engine": "trellis-proj", "restricted_modules_loaded": [], "rmbg_disabled": True, "cases": {}}
    (runs / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return runs


def test_a_fake_run_is_scored_saved_and_checked() -> None:
    folder = Path(support.make_temp_dir("tools_flow"))
    runs = make_run(folder)
    truth = folder / "truth.glb"
    sphere = build_icosphere(3, 0.5)
    trimesh.Trimesh(sphere.vertices * np.array([1.0, 2.0, 0.8]), sphere.faces).export(truth)
    compare = folder / "compare"
    with contextlib.redirect_stdout(io.StringIO()):
        code = compare_models.compare_cases(
            ["--runs", str(runs), "--case", "demo", "--reference", f"demo={truth}", "--out", str(compare)]
        )
    assert code == 0 and (compare / "report.md").is_file() and (compare / "demo_views.png").is_file()
    row = json.loads((compare / "report.json").read_text(encoding="utf-8"))["rows"][0]
    assert row["reference"] == "truth" and row["volume_iou"] > 0.5
    expected = folder / "expected.json"
    common = ["--runs", str(runs), "--compare", str(compare / "report.json"), "--expected", str(expected)]
    with contextlib.redirect_stdout(io.StringIO()):
        assert check_reproduction.check_reproduction([*common, "--write-expected"]) == 0
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert check_reproduction.check_reproduction(common) == 0
    assert "REPRODUCTION PASSED" in printed.getvalue() and "bit-identical: yes" in printed.getvalue()


def test_the_package_check_lists_what_differs_from_a_lock() -> None:
    lock = Path(support.make_temp_dir("tools_packages")) / "lock.txt"
    lock.write_text("numpy==0.0.1\nno-such-package-xyz==1.0\n", encoding="utf-8")
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert check_packages.check_packages([str(lock)]) == 1
    assert "different: numpy" in printed.getvalue() and "missing: no-such-package-xyz 1.0" in printed.getvalue()


def test_compare_models_reads_the_vertex_colours_of_a_run() -> None:
    folder = Path(support.make_temp_dir("tools_load"))
    painter_run = make_run(folder)
    mesh, colours = compare_models.load_coloured_mesh(painter_run / "demo.glb")
    assert len(mesh.faces) > 0 and colours.shape == (len(mesh.vertices), 3)


if __name__ == "__main__":
    support.run_tests(globals())
