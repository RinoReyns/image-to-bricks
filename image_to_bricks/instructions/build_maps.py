"""Top-down layer maps of a build_order.Build (the main model or a sub-assembly): only what is on the model while
that layer is built. Pieces of a sub-assembly are not drawn before it is placed; on the layer where it is placed as
a unit, its pieces of that layer are drawn faded with its name and colour (they come with it); the layer below is
hatched with what is really there."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from PIL import Image

from image_to_bricks.analysis.build_order import Build
from image_to_bricks.instructions import fast_render as fr
from image_to_bricks.instructions.instruction_parts import colour_rgb
from image_to_bricks.model import LegoModel, PlacedBrick, plate_origin


def pieces_by_covered_layer(pieces: Iterable[PlacedBrick]) -> dict[int, list[PlacedBrick]]:
    """{z: pieces covering plate layer z} (a brick covers 3 layers)."""
    covering: dict[int, list[PlacedBrick]] = {}
    for piece in pieces:
        for layer in range(piece.z, piece.z + piece.h_plates):
            covering.setdefault(layer, []).append(piece)
    return covering


@dataclass(frozen=True)
class BuildMapContext:
    """What every map of one build needs: the model (axis origin, FRONT arrow), the build, its pieces by covered
    layer (pieces_by_covered_layer of build.pieces), whether the layer below is hatched, whether the legend is
    drawn under the map (with the layer below only) and the label shown for a sub-assembly instead of its name
    ({name: label}, e.g. "3a" for a small part joined inside a step)."""

    model: LegoModel
    build: Build
    covering: dict[int, list[PlacedBrick]]
    show_layer_below: bool
    show_legend: bool = True
    labels: dict[str, str] = field(default_factory=dict)


def build_map_context(
    model: LegoModel,
    build: Build,
    *,
    show_layer_below: bool,
    show_legend: bool = True,
    labels: dict[str, str] | None = None,
) -> BuildMapContext:
    """BuildMapContext of a build."""
    return BuildMapContext(
        model, build, pieces_by_covered_layer(build.pieces), show_layer_below, show_legend, labels or {}
    )


def subassembly_marks_on_layer(
    build: Build, pieces: Iterable[PlacedBrick], layer: int, labels: dict[str, str] | None = None
) -> dict[int, tuple[str, tuple[int, int, int]]]:
    """{id(piece): (label, outline rgb)} for the pieces that come with a sub-assembly placed on `layer` (label: its
    entry in labels, else its name)."""
    labels = labels or {}
    marks: dict[int, tuple[str, tuple[int, int, int]]] = {}
    for piece in pieces:
        subassembly = build.subassembly_placed_on(piece, layer)
        if subassembly is not None:
            marks[id(piece)] = (
                labels.get(subassembly.name, subassembly.name),
                fr.subassembly_outline_colour(subassembly.colour_index),
            )
    return marks


def pieces_drawn_on_layer(context: BuildMapContext, layer: int) -> list[PlacedBrick]:
    """Pieces covering `layer` that are on the model while it is built (not in a sub-assembly placed later)."""
    return [
        piece for piece in context.covering.get(layer, []) if not context.build.is_waiting_for_attachment(piece, layer)
    ]


def pieces_hatched_below(context: BuildMapContext, layer: int) -> list[PlacedBrick] | None:
    """Pieces covering the layer below that are on the model when `layer` is built, or None when the layer below
    is not shown."""
    if not context.show_layer_below:
        return None
    return [
        piece for piece in context.covering.get(layer - 1, []) if context.build.is_in_place_before_layer(piece, layer)
    ]


def shown_region(context: BuildMapContext, layer: int, margin_studs: int) -> tuple[int, int, int, int] | None:
    """Grid region (x0, x1, y0, y1; ends exclusive) around what the map of `layer` shows (its pieces and the layer
    below) plus margin_studs, clipped to the grid; None if it shows nothing."""
    studs = fr.studs_of_pieces(pieces_drawn_on_layer(context, layer) + (pieces_hatched_below(context, layer) or []))
    if not studs:
        return None
    _, depth, width = context.model.grid.shape
    xs, ys = [stud[0] for stud in studs], [stud[1] for stud in studs]
    return (
        max(min(xs) - margin_studs, 0),
        min(max(xs) + 1 + margin_studs, width),
        max(min(ys) - margin_studs, 0),
        min(max(ys) + 1 + margin_studs, depth),
    )


def draw_build_layer_map(
    context: BuildMapContext, layer: int, region: tuple[int, int, int, int], title: str
) -> Image.Image:
    """Map of `layer` of the build in region (x0, x1, y0, y1; grid coordinates, ends exclusive)."""
    build = context.build
    x0, x1, y0, y1 = region
    drawn = pieces_drawn_on_layer(context, layer)
    plate_x0, plate_y0 = plate_origin(context.model.grid, context.model.base_layers)
    marks = fr.MapMarks(
        new_z=layer,
        floating=build.floating,
        front=getattr(context.model, "front", None),
        below_bricks=pieces_hatched_below(context, layer),
        subassembly_marks=subassembly_marks_on_layer(build, drawn, layer, context.labels),
        show_legend=context.show_legend,
    )
    region_shown = fr.MapRegion(x0, y0, x1 - x0, y1 - y0, plate_x0=plate_x0, plate_y0=plate_y0)
    return fr.layer_map(drawn, region_shown, title, colour_rgb, marks)
