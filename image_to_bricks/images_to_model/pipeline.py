"""Pictures -> coloured, cleaned, scaled GLB with its licence package. Every model is passed in through Engines.

Steps: cut out (cutout.py) -> shape (TRELLIS.2, front picture) -> clean (clean.py) -> colour from all pictures
(projection.py) -> place and scale (place.py) -> licence package (provenance.py). Intermediate files and report.json go
to the work folder.
"""

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from image_to_bricks.images_to_model.clean import clean_mesh
from image_to_bricks.images_to_model.cutout import cut_out_image
from image_to_bricks.images_to_model.engines import ColouredModel, Engines
from image_to_bricks.images_to_model.mesh import Mesh, write_ply
from image_to_bricks.images_to_model.place import apply_placement, compute_placement, measure_mesh
from image_to_bricks.images_to_model.provenance import ENGINE_NAME, list_restricted_modules, package_output
from image_to_bricks.images_to_model.type_aliases import JsonDict
from image_to_bricks.images_to_model.views import TEXTURE_ORDER, ViewPictures, load_view_images

DEFAULT_MAX_FACES = 40000
DEFAULT_SEED = 12345
BYTES_PER_GIB = 1024**3
TIME_DIGITS = 2  # seconds in the report
MEMORY_DIGITS = 2  # GiB in the report
PLACEMENT_DIGITS = 6
WORK_FOLDER_SUFFIX = "_work"


@dataclass(frozen=True)
class ModelSettings:
    """height: of the placed model, in GLB units (metres); max_faces: of the cleaned mesh; seed: of the shape model."""

    height: float
    max_faces: int = DEFAULT_MAX_FACES
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class ModelOutput:
    """Where the model goes: the GLB (its licence package is written next to it) and the work folder for cut-outs,
    intermediate meshes and report.json (default: <glb name>_work next to the GLB)."""

    glb_path: Path
    work_folder: Path | None = None

    @property
    def work(self) -> Path:
        """The work folder."""
        if self.work_folder is not None:
            return Path(self.work_folder)
        glb_path = Path(self.glb_path)
        return glb_path.with_name(glb_path.stem + WORK_FOLDER_SUFFIX)


@contextmanager
def timed_step(report: JsonDict, step: str) -> Iterator[None]:
    """Record how long the block takes as report["seconds"][step]."""
    started = time.perf_counter()
    yield
    report["seconds"][step] = round(time.perf_counter() - started, TIME_DIGITS)


def record_peak_memory(report: JsonDict, step: str, engines: Engines) -> None:
    """Peak GPU memory of the step that just ran, in GiB, as report["peak_vram_gib"][step]."""
    report["peak_vram_gib"][step] = round(engines.probe.read_peak_bytes() / BYTES_PER_GIB, MEMORY_DIGITS)


def cut_out_views(
    images: dict[str, Image.Image], engines: Engines, work: Path, report: JsonDict
) -> dict[str, Image.Image]:
    """RGBA cut-out of every picture (saved as cutout_<view>.png); how each was made goes into report["cutouts"]."""
    cutouts, report["cutouts"] = {}, {}
    for name, image in images.items():
        cutouts[name], info = cut_out_image(image, engines.remover)
        report["cutouts"][name] = info.to_json()
        cutouts[name].save(work / f"cutout_{name}.png")
    return cutouts


def generate_shape(cutouts: dict[str, Image.Image], engines: Engines, seed: int, report: JsonDict) -> Mesh:
    """Raw mesh from the shape model; the pictures it used go into report["shape_views"]."""
    engines.probe.reset()
    report["shape_views"] = [name for name in cutouts if name in engines.shape.view_names]
    generated = engines.shape.generate(cutouts, seed)
    record_peak_memory(report, "shape", engines)
    return generated


def colour_mesh(cleaned: Mesh, cutouts: dict[str, Image.Image], engines: Engines, report: JsonDict) -> ColouredModel:
    """The cleaned mesh coloured from all pictures (summed in TEXTURE_ORDER); the painter's info goes into
    report["painter"]."""
    engines.probe.reset()
    texture_views = {name: cutouts[name] for name in TEXTURE_ORDER if name in cutouts}
    coloured = engines.painter.paint(cleaned, texture_views)
    if engines.painter.last_info:
        report["painter"] = engines.painter.last_info
    record_peak_memory(report, "texture", engines)
    report["texture_views"] = list(texture_views)
    return coloured


def place_and_export(
    cleaned: Mesh, coloured: ColouredModel, height: float, output: ModelOutput, report: JsonDict
) -> None:
    """Put both meshes on the floor, centred, `height` tall; write cleaned_placed.ply and the GLB."""
    scale, offset = compute_placement(cleaned.vertices, height)
    placed_clean = Mesh(apply_placement(cleaned.vertices, scale, offset), cleaned.faces)
    write_ply(placed_clean, output.work / "cleaned_placed.ply")
    coloured_vertices = np.asarray(coloured.vertices, dtype=np.float64)
    # placed from its own bounds, so a painter that re-centres or re-scales the mesh still ends up in the same place
    coloured.vertices = apply_placement(coloured_vertices, *compute_placement(coloured_vertices, height))
    glb_path = Path(output.glb_path)
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    coloured.export(str(glb_path))
    report["placement"] = {
        "scale": round(scale, PLACEMENT_DIGITS),
        "offset": [round(float(v), PLACEMENT_DIGITS) for v in offset],
    }
    report["measurements"] = measure_mesh(placed_clean)


def start_report(views: ViewPictures, output: ModelOutput, settings: ModelSettings) -> JsonDict:
    """The report's fixed fields: inputs, outputs, engine and settings."""
    return {
        "views_dir": str(views.source_folder),
        "views": {name: str(path) for name, path in views.files.items()},
        "output": str(output.glb_path),
        "work_dir": str(output.work),
        "engine": ENGINE_NAME,
        "height": settings.height,
        "seed": settings.seed,
        "max_faces": settings.max_faces,
        "seconds": {},
        "peak_vram_gib": {},
    }


def run_pipeline(views: ViewPictures, output: ModelOutput, settings: ModelSettings, engines: Engines) -> JsonDict:
    """Make the model from the pictures. Returns the report (also written to <work>/report.json): counts of every
    step, times, peak GPU memory, placement, measurements and the licence package. Raises cutout.CutOutError for a
    picture whose background cannot be removed."""
    output.work.mkdir(parents=True, exist_ok=True)
    report = start_report(views, output, settings)
    started = time.perf_counter()
    with timed_step(report, "cutout"):
        cutouts = cut_out_views(load_view_images(views), engines, output.work, report)
    with timed_step(report, "shape"):
        generated = generate_shape(cutouts, engines, settings.seed, report)
    write_ply(generated, output.work / "generated.ply")
    with timed_step(report, "clean"):
        cleaned, report["clean"] = clean_mesh(generated, settings.max_faces, engines.simplify)
    write_ply(cleaned, output.work / "cleaned.ply")
    with timed_step(report, "texture"):
        coloured = colour_mesh(cleaned, cutouts, engines, report)
    with timed_step(report, "place_export"):
        place_and_export(cleaned, coloured, settings.height, output, report)
    report["package"] = package_output(report, views, list_restricted_modules())
    report["seconds"]["total"] = round(time.perf_counter() - started, TIME_DIGITS)
    report["peak_vram_gib"]["max"] = max(report["peak_vram_gib"].values())
    (output.work / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
