"""What run_pipeline needs from the outside: background removal, the shape model, the face reducer and a GPU memory
probe. Production engines come from trellis_engine.load_engines; tests pass fakes."""

from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from image_to_bricks.images_to_model.clean import Simplifier
from image_to_bricks.images_to_model.cutout import BackgroundRemover
from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.projection import ProjectionPainter
from image_to_bricks.images_to_model.type_aliases import FloatArray, JsonDict


class ShapeGenerator(Protocol):
    """Image-to-3D shape model. view_names: the pictures it uses (TRELLIS.2: front only)."""

    view_names: tuple[str, ...]

    def generate(self, images: dict[str, Image.Image], seed: int) -> Mesh:
        """Raw mesh (y up) from the RGBA cut-outs by view name."""
        ...


class ColouredModel(Protocol):
    """A coloured mesh that can be written as a GLB."""

    vertices: FloatArray

    def export(self, path: str) -> None:
        """Write the model to path (GLB)."""
        ...


class Painter(Protocol):
    """Colours a cleaned mesh from the named cut-outs; last_info: what the last run found (or None)."""

    last_info: JsonDict | None

    def paint(self, mesh: Mesh, views: dict[str, Image.Image]) -> ColouredModel:
        """The coloured mesh."""
        ...


class MemoryProbe(Protocol):
    """Peak GPU memory of a step."""

    def reset(self) -> None:
        """Start measuring a new step."""
        ...

    def read_peak_bytes(self) -> int:
        """Peak bytes allocated since the last reset."""
        ...


@dataclass
class Engines:
    """Everything model-related run_pipeline uses, injected so the pipeline runs with fakes in tests."""

    remover: BackgroundRemover
    shape: ShapeGenerator
    simplify: Simplifier
    probe: MemoryProbe
    painter: Painter = field(default_factory=ProjectionPainter)
