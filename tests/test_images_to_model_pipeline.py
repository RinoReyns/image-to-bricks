"""images_to_model end to end with fake engines: pictures (folder or single files) -> cut-outs, meshes, report,
GLB, licence package; the command line and its errors; the PLY writer."""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import support
from images_to_model_fakes import FakePainter, FakeProbe, build_fake_engines
from images_to_model_helpers import assert_raises, is_close
from PIL import Image

from image_to_bricks.images_to_model.cli import build_argument_parser, run_cli
from image_to_bricks.images_to_model.cutout import CutOutError
from image_to_bricks.images_to_model.engines import Engines
from image_to_bricks.images_to_model.mesh import Mesh, write_ply
from image_to_bricks.images_to_model.pipeline import ModelOutput, ModelSettings, run_pipeline
from image_to_bricks.images_to_model.views import collect_view_files, find_views_in_folder

WORK_FILES = (
    "cutout_front.png",
    "cutout_back.png",
    "cutout_right.png",
    "generated.ply",
    "cleaned.ply",
    "cleaned_placed.ply",
    "report.json",
)


def save_view(folder: Path, name: str, size: int, *, transparent: bool) -> None:
    if transparent:
        pixels = np.zeros((size, size, 4), dtype=np.uint8)
        pixels[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
        Image.fromarray(pixels, "RGBA").save(folder / f"{name}.png")
    else:
        Image.new("RGB", (size, size), (120, 130, 140)).save(folder / f"{name}.png")


def write_views(folder: Path) -> Path:
    """views/ with a transparent front (64 px), an opaque back (48 px) and a transparent right (40 px)."""
    views = folder / "views"
    views.mkdir(parents=True)
    save_view(views, "front", 64, transparent=True)
    save_view(views, "back", 48, transparent=False)
    save_view(views, "right", 40, transparent=True)
    return views


def test_a_folder_without_a_front_picture_is_rejected() -> None:
    with assert_raises(FileNotFoundError, "front.png"):
        find_views_in_folder(Path(support.make_temp_dir("no_front")))


def test_single_picture_files_are_collected_in_view_order() -> None:
    views = write_views(Path(support.make_temp_dir("single_files")))
    pictures = collect_view_files({"right": views / "right.png", "front": views / "front.png", "left": None})
    assert list(pictures.files) == ["front", "right"] and pictures.source_folder == views
    assert pictures.character_name == "views"
    with assert_raises(FileNotFoundError, "front picture is required"):
        collect_view_files({"back": views / "back.png"})
    with assert_raises(FileNotFoundError, "not found"):
        collect_view_files({"front": views / "missing.png"})
    with assert_raises(ValueError, "unknown view names"):
        collect_view_files({"front": views / "front.png", "top": views / "back.png"})


def test_a_renamed_front_picture_names_the_character() -> None:
    views = write_views(Path(support.make_temp_dir("renamed")))
    (views / "front.png").rename(views / "hero.png")
    assert collect_view_files({"front": views / "hero.png"}).character_name == "hero"


def test_the_pipeline_with_fakes_writes_every_file_and_reports_every_step() -> None:
    folder = Path(support.make_temp_dir("pipeline_fakes"))
    painter, probe = FakePainter(), FakeProbe()
    engines = build_fake_engines(painter)
    engines.probe = probe
    output = folder / "out" / "hero.glb"
    report = run_pipeline(
        find_views_in_folder(write_views(folder)), ModelOutput(output), ModelSettings(1.75, 500), engines
    )
    work = folder / "out" / "hero_work"
    for name in WORK_FILES:
        assert (work / name).is_file(), name
    assert output.read_bytes()[:4] == b"glTF"
    assert report["package"]["usage"] == "permissive-with-attribution"
    assert (folder / "out" / "hero_package" / "NOTICE.md").is_file() and (folder / "out" / "hero_package.zip").is_file()
    assert report["shape_views"] == ["front", "back", "right"]
    assert painter.image_sizes == [(64, 64), (48, 48), (40, 40)]  # front, back, right
    assert report["texture_views"] == ["front", "back", "right"] and report["painter"] == {"front_direction": "+z"}
    assert report["cutouts"]["back"]["method"] == "rembg-u2net" and report["cutouts"]["front"]["method"] == "kept"
    assert report["clean"]["faces_in"] == 1280 and report["clean"]["faces_after_reduce"] <= 500
    assert report["clean"]["watertight"]
    assert is_close(report["measurements"]["height"], 1.75) and is_close(report["measurements"]["min_y"], 0.0)
    assert painter.result is not None
    placed = painter.result.vertices
    assert is_close(placed[:, 1].min(), 0.0, abs_tol=1e-9) and is_close(placed[:, 1].max(), 1.75)
    assert report["peak_vram_gib"] == {"shape": 3.0, "texture": 3.0, "max": 3.0} and probe.resets == 2
    assert set(report["seconds"]) == {"cutout", "shape", "clean", "texture", "place_export", "total"}
    assert json.loads((work / "report.json").read_text(encoding="utf-8"))["clean"] == report["clean"]


def test_the_real_colour_projection_runs_in_the_pipeline() -> None:
    folder = Path(support.make_temp_dir("pipeline_projection"))
    report = run_pipeline(
        find_views_in_folder(write_views(folder)),
        ModelOutput(folder / "p.glb"),
        ModelSettings(1.0, 500),
        build_fake_engines(),
    )
    assert report["painter"]["views"] == ["back", "front", "right"] and report["painter"]["front_direction"] in {
        "+z", "+x", "-z", "-x",
    }  # fmt: skip
    assert (folder / "p.glb").stat().st_size > 1000


def test_the_pipeline_stops_on_a_bad_cut_out() -> None:
    folder = Path(support.make_temp_dir("pipeline_bad_cutout"))
    views = folder / "views"
    views.mkdir()
    Image.new("RGBA", (50, 50), (0, 0, 0, 0)).save(views / "front.png")
    with assert_raises(CutOutError):
        run_pipeline(
            find_views_in_folder(views), ModelOutput(folder / "x.glb"), ModelSettings(1.0), build_fake_engines()
        )


def test_the_work_folder_defaults_to_next_to_the_glb() -> None:
    assert ModelOutput(Path("a/b/x.glb")).work == Path("a/b/x_work")
    assert ModelOutput(Path("a/b/x.glb"), Path("w")).work == Path("w")


def fake_engine_loader() -> Engines:
    return build_fake_engines(FakePainter())


def test_command_line_options_and_errors() -> None:
    folder = Path(support.make_temp_dir("cli"))
    views = write_views(folder)
    arguments = build_argument_parser().parse_args(["--views", str(views), "--output", "o.glb", "--height", "2"])
    assert (arguments.height, arguments.max_faces, arguments.seed, arguments.work) == (2.0, 40000, 12345, None)
    with contextlib.redirect_stderr(io.StringIO()), assert_raises(SystemExit):
        build_argument_parser().parse_args(["--views", "v", "--output", "o.glb", "--height", "-1"])
    output = folder / "cli.glb"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = run_cli(["--views", str(views), "--output", str(output), "--height", "1"], fake_engine_loader)
    assert code == 0 and output.is_file() and "load_models" in json.loads(printed.getvalue())["seconds"]
    single = ["--front", str(views / "front.png"), "--back", str(views / "back.png"), "--output", str(folder / "s.glb")]
    with contextlib.redirect_stdout(io.StringIO()):
        assert run_cli([*single, "--height", "1"], fake_engine_loader) == 0
    for bad in (["--output", "o.glb", "--height", "1"], ["--views", str(views), "--front", "f.png", "--output", "o.glb",
                "--height", "1"], ["--views", str(folder), "--output", "o.glb", "--height", "1"]):  # fmt: skip
        with contextlib.redirect_stderr(io.StringIO()), assert_raises(SystemExit):
            run_cli(bad, fake_engine_loader)


def test_ply_header_and_size() -> None:
    path = Path(support.make_temp_dir("ply")) / "m.ply"
    write_ply(Mesh(np.zeros((4, 3)), np.array([[0, 1, 2], [0, 2, 3]])), path)
    data = path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    assert b"element vertex 4" in data[:header_end] and b"element face 2" in data[:header_end]
    assert len(data) - header_end == 4 * 12 + 2 * 13


if __name__ == "__main__":
    support.run_tests(globals())
