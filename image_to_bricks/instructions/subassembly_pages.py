"""Sub-assembly pages of the instruction PDF (build_order.SubAssembly: a part that hangs in the air while the model
is built bottom-up, with the pieces that clamp it). Each one is built on pages placed right before the page of the
layer it is placed on:
- a big one (more steps or pieces than build_order.COMPACT_MAX_STEPS / COMPACT_MAX_PIECES, or with its own
  sub-assemblies, e.g. a tail) gets one page per step: its own parts table, a 3D view of it built so far and a
  top-down map with only its pieces;
- the small ones placed on the same layer share one page: per sub-assembly its name (in its map colour), its parts
  table and its steps as small maps side by side.
Sub-assemblies inside a sub-assembly get the same pages, right before its step they are placed on. Also: the notes
on the pages where sub-assemblies are placed, and the build-plan summary line.

Layers are model z indices; page texts use the figure layer names (instruction_parts.layer_name)."""

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from reportlab.lib.styles import StyleSheet1
from reportlab.lib.units import inch
from reportlab.platypus import Flowable, KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import Image as RLImage

from image_to_bricks.analysis.build_order import Build, SubAssembly
from image_to_bricks.instructions import fast_render as fr
from image_to_bricks.instructions.build_maps import build_map_context, draw_build_layer_map
from image_to_bricks.instructions.instruction_parts import (
    PAGE_CONTENT_WIDTH_INCH,
    LayerPage,
    cells_coloured_by_pieces,
    cells_covered_by_pieces,
    hold_notes,
    layer_name,
    layer_page_flowables,
    parts_table,
    piece_numbers_by_cell,
)
from image_to_bricks.model import LegoModel, PlacedBrick

SUBASSEMBLY_MAP_MARGIN_STUDS = 2  # empty studs around the sub-assembly's footprint on its maps
SUBASSEMBLY_MAP_MIN_SPAN_STUDS = 9  # its maps are at least this wide / deep (room for the title and FRONT arrow)
SUBASSEMBLY_VIEW_CELL_PX = 14  # cell size of the 3D view (as the cumulative views of the main build)
EARLIER_STEPS_FADE = 0.55  # 3D view: earlier steps faded this far toward white (as the main build)
COMPACT_MAP_HEIGHT_INCH = 2.3  # height of one step map on a shared page of small sub-assemblies
COMPACT_MAP_GAP_INCH = 0.1
INLINE_MAP_HEIGHT_INCH = 1.5  # height of one step map of a small part shown on its parent's step page
INLINE_MAX_HEIGHT_ON_STEP_PAGE_INCH = 3.2  # pre-steps taller than this get their own page(s) before the step page
LAYOUT_HEIGHT_INCH = 10  # height offered to a flowable when measuring it (more than any block needs)
STEP_LETTERS = "abcdefghijklmnopqrstuvwxyz"  # pre-step letters: 3a, 3b, ...

type Region = tuple[int, int, int, int]  # (x0, x1, y0, y1) in grid studs, ends exclusive


@dataclass(frozen=True)
class SubAssemblyImages:
    """Image files of one sub-assembly, by step layer (z): top-down map, and the 3D view (big ones only)."""

    map_path_by_layer: dict[int, str]
    view_path_by_layer: dict[int, str]


def colour_hex(subassembly: SubAssembly) -> str:
    """'#rrggbb' of the sub-assembly's map colour (for page texts)."""
    red, green, blue = fr.subassembly_outline_colour(subassembly.colour_index)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _span_with_minimum(low: int, high: int, minimum: int, limit: int) -> tuple[int, int]:
    """[low, high) widened evenly to at least `minimum` cells, kept inside [0, limit)."""
    missing = max(0, minimum - (high - low))
    low, high = low - missing // 2, high + missing - missing // 2
    shift = max(0, -low) - max(0, high - limit)
    return max(low + shift, 0), min(high + shift, limit)


def footprint_region(
    pieces: Sequence[PlacedBrick], grid_shape: tuple[int, ...], margin_studs: int, min_span_studs: int = 0
) -> Region:
    """Grid region (x0, x1, y0, y1; ends exclusive) around the studs of the pieces plus margin_studs, at least
    min_span_studs wide and deep, clipped to the grid."""
    _, depth, width = grid_shape
    studs = fr.studs_of_pieces(pieces)
    xs, ys = [stud[0] for stud in studs], [stud[1] for stud in studs]
    x0, x1 = _span_with_minimum(min(xs) - margin_studs, max(xs) + 1 + margin_studs, min_span_studs, width)
    y0, y1 = _span_with_minimum(min(ys) - margin_studs, max(ys) + 1 + margin_studs, min_span_studs, depth)
    return x0, x1, y0, y1


def is_inline(subassembly: SubAssembly) -> bool:
    """A small sub-assembly inside a sub-assembly: shown as a pre-step ("3a") on its parent's step page."""
    return "." in subassembly.name and subassembly.is_compact


def _step_letter(index: int) -> str:
    """Letter of the pre-step `index` (0 = a)."""
    return STEP_LETTERS[index]


def inline_step_labels(plan: Build) -> dict[str, str]:
    """{name: "3a"} for every inline sub-assembly: its parent's step number + a letter in the order of that step
    (the parent's own pieces of the step come after them, with the next letter)."""
    labels: dict[str, str] = {}
    for parent in plan.all_subassemblies():
        steps = parent.step_layers
        for layer in steps:
            inline = [child for child in parent.build.subassemblies if child.join_layer == layer and is_inline(child)]
            for index, child in enumerate(inline):
                labels[child.name] = f"{steps.index(layer) + 1}{_step_letter(index)}"
    return labels


def step_title(subassembly: SubAssembly, layer: int) -> str:
    """'S1 step 3/14' (map title; short so it leaves room for the FRONT arrow on narrow maps)."""
    steps = subassembly.step_layers
    return f"{subassembly.name} step {steps.index(layer) + 1}/{len(steps)}"


def _render_step_view(model: LegoModel, subassembly: SubAssembly, layer: int, region: Region) -> Image.Image:
    """3D view of the sub-assembly built up to `layer` (its own sub-assemblies only once placed): earlier steps
    faded, this layer in colour; the frame is the whole sub-assembly so all steps have the same scale. No title
    (the page caption names the step)."""
    x0, x1, y0, y1 = region
    build = subassembly.build
    colours = cells_coloured_by_pieces(build.pieces, model.grid.shape)
    all_cells = cells_covered_by_pieces(build.pieces, model.grid.shape)
    not_placed = [child.pieces for child in build.subassemblies if child.join_layer > layer]
    built_cells = all_cells.copy()
    for child_pieces in not_placed:
        built_cells &= ~cells_covered_by_pieces(child_pieces, model.grid.shape)
    faded = (colours.astype(np.float32) * (1 - EARLIER_STEPS_FADE) + 255 * EARLIER_STEPS_FADE).astype(np.uint8)
    lowest = build.lowest_layer
    view_colours = faded[lowest : layer + 1].copy()
    view_colours[-1] = colours[layer]
    piece_numbers = piece_numbers_by_cell(build.pieces, model.grid.shape)
    return fr.iso_render(
        built_cells[lowest : layer + 1, y0:y1, x0:x1],
        view_colours[:, y0:y1, x0:x1],
        fr.IsoView(
            px=SUBASSEMBLY_VIEW_CELL_PX,
            frame_grid_zyx=all_cells[lowest : build.top_layer + 1, y0:y1, x0:x1],
            piece_ids_zyx=piece_numbers[lowest : layer + 1, y0:y1, x0:x1],
        ),
    )


def _write_subassembly_images(
    model: LegoModel, subassembly: SubAssembly, outdir: str, labels: dict[str, str], *, show_layer_below: bool
) -> SubAssemblyImages:
    """Map (and for a big one the 3D view) of every step of one sub-assembly (labels: inline_step_labels)."""
    region = footprint_region(
        subassembly.pieces, model.grid.shape, SUBASSEMBLY_MAP_MARGIN_STUDS, SUBASSEMBLY_MAP_MIN_SPAN_STUDS
    )
    context = build_map_context(
        model,
        subassembly.build,
        show_layer_below=show_layer_below,
        show_legend=not subassembly.is_compact,
        labels=labels,
    )
    steps = subassembly.step_layers
    map_paths: dict[int, str] = {}
    view_paths: dict[int, str] = {}
    for step_number, layer in enumerate(steps, start=1):
        title = (
            f"{labels[subassembly.name]}: {step_number}/{len(steps)}"
            if subassembly.name in labels
            else step_title(subassembly, layer)
        )
        map_paths[layer] = os.path.join(outdir, f"{subassembly.name}_step_{step_number:02d}.png")
        draw_build_layer_map(context, layer, region, title).save(map_paths[layer])
        if not subassembly.is_compact:
            view_paths[layer] = os.path.join(outdir, f"{subassembly.name}_view_{step_number:02d}.png")
            _render_step_view(model, subassembly, layer, region).save(view_paths[layer])
    return SubAssemblyImages(map_paths, view_paths)


def generate_subassembly_images(
    model: LegoModel, plan: Build, outdir: str, *, show_layer_below: bool = True
) -> dict[str, SubAssemblyImages]:
    """Write the step maps (<S>_step_NN.png) and, for big sub-assemblies, the 3D views (<S>_view_NN.png) of every
    sub-assembly, nested ones included, to outdir. Returns {sub-assembly name: SubAssemblyImages}."""
    subassemblies = plan.all_subassemblies()
    if not subassemblies:
        return {}
    os.makedirs(outdir, exist_ok=True)
    labels = inline_step_labels(plan)
    return {
        subassembly.name: _write_subassembly_images(
            model, subassembly, outdir, labels, show_layer_below=show_layer_below
        )
        for subassembly in subassemblies
    }


def _short_layer_span(model: LegoModel, pieces: Sequence[PlacedBrick]) -> str:
    """'layers 8-21' (figure layer numbers) of the pieces."""
    lowest = min(piece.z for piece in pieces)
    top = max(piece.z + piece.h_plates - 1 for piece in pieces)
    return f"layers {layer_name(model, lowest).split()[-1]}-{layer_name(model, top).split()[-1]}"


def _name_html(subassembly: SubAssembly) -> str:
    """The sub-assembly's name, bold in its map colour (page text markup)."""
    return f"<font color='{colour_hex(subassembly)}'><b>{subassembly.name}</b></font>"


def _inline_place_note(placed_inline: list[SubAssembly], labels: dict[str, str], styles: StyleSheet1) -> list[Flowable]:
    """'3b: put 3a in its place ...' (after the pre-step blocks of a step page)."""
    names = [labels[subassembly.name] for subassembly in placed_inline]
    step_label = names[0][:-1] + _step_letter(len(names))
    joined = ", ".join(f"<b>{name}</b>" for name in names)
    return [
        Paragraph(
            f"<b>{step_label}:</b> put {joined} in {'its' if len(names) == 1 else 'their'} place (faded, marked "
            f"{joined} on the map below: {'its' if len(names) == 1 else 'their'} top piece(s) press onto the pieces of "
            f"the step before), then add this step's other pieces.",
            styles["Normal"],
        ),
        Spacer(1, 4),
    ]


def place_note(build: Build, layer: int, styles: StyleSheet1, labels: dict[str, str] | None = None) -> list[Flowable]:
    """Note on a step page where sub-assemblies are placed as units: lower each one into place (or slide it in when
    something above is in the way) so its top pieces (faded, with its name on the map) press onto this layer, then
    place the other pieces of the layer. Inline ones (in labels) get the short "3b: put 3a in its place" line."""
    labels = labels or {}
    placed_all = [subassembly for subassembly in build.subassemblies if subassembly.join_layer == layer]
    placed_inline = [subassembly for subassembly in placed_all if subassembly.name in labels]
    placed = [subassembly for subassembly in placed_all if subassembly.name not in labels]
    notes = _inline_place_note(placed_inline, labels, styles) if placed_inline else []
    if not placed:
        return notes
    names = ", ".join(_name_html(subassembly) for subassembly in placed)
    blocked = [subassembly for subassembly in placed if subassembly.is_blocked_from_above]
    text = (
        f"<b>First place sub-assembl{'y' if len(placed) == 1 else 'ies'}</b> {names} (built on the page(s) "
        f"before): lower {'it' if len(placed) == 1 else 'each one'} into place from above, so the pieces "
        f"marked with {'its name' if len(placed) == 1 else 'their names'} on the map (faded, they come with "
        f"it) press onto this layer and carry the hanging part. Then place this layer's other pieces."
    )
    if blocked:
        text += " " + " ".join(
            f"{_name_html(subassembly)}: pieces above it are in the way, slide it in from the side."
            for subassembly in blocked
        )
    return [*notes, Paragraph(text, styles["Normal"]), Spacer(1, 4)]


def _intro_note(model: LegoModel, subassembly: SubAssembly, styles: StyleSheet1) -> list[Flowable]:
    """First-step note of a big sub-assembly."""
    return [
        Paragraph(
            f"Sub-assembly {_name_html(subassembly)}: {len(subassembly.pieces)} pieces "
            f"({_short_layer_span(model, subassembly.pieces)}) that hang in the air while the model is built from the "
            f"bottom, with the pieces that clamp it on top. Build it on its own from its lowest layer up (start on the "
            f"table), then place it as one unit on {layer_name(model, subassembly.join_layer)}. Only its pieces are on "
            f"these maps; the axis numbers are those of the main maps.",
            styles["Normal"],
        ),
        Spacer(1, 4),
    ]


def _finished_note(model: LegoModel, subassembly: SubAssembly, styles: StyleSheet1) -> list[Flowable]:
    """Last-step note of a big sub-assembly."""
    return [
        Paragraph(
            f"{_name_html(subassembly)} <b>is complete.</b> Keep it aside; it is placed on "
            f"{layer_name(model, subassembly.join_layer)}.",
            styles["Normal"],
        ),
        Spacer(1, 4),
    ]


@dataclass(frozen=True)
class PageInputs:
    """What the sub-assembly pages need: the model, the images of every sub-assembly, the Own / Buy flags
    (is_owned, or None), the paragraph styles and the labels of the inline sub-assemblies (inline_step_labels)."""

    model: LegoModel
    images: dict[str, SubAssemblyImages]
    is_owned: dict[int, bool] | None
    styles: StyleSheet1
    inline_labels: dict[str, str] = field(default_factory=dict)


def _big_subassembly_pages(inputs: PageInputs, subassembly: SubAssembly) -> list[Flowable]:
    """All pages of a big sub-assembly (its own sub-assemblies' pages before the step they are placed on), each
    followed by a page break."""
    model, styles = inputs.model, inputs.styles
    images = inputs.images[subassembly.name]
    story: list[Flowable] = []
    steps = subassembly.step_layers
    for step_number, layer in enumerate(steps, start=1):
        story += pages_before_layer(inputs, subassembly.build, layer, subassembly)
        heading = f"Sub-assembly {subassembly.name} - step {step_number} / {len(steps)} ({layer_name(model, layer)})"
        pre_steps = _pre_step_flowables(inputs, subassembly, layer)
        notes = _intro_note(model, subassembly, styles) if step_number == 1 else []
        if _height_inch(pre_steps) > INLINE_MAX_HEIGHT_ON_STEP_PAGE_INCH:
            story += _pre_step_pages(inputs, subassembly, layer, heading, notes + pre_steps)
            notes = []
        else:
            notes += pre_steps
        notes += place_note(subassembly.build, layer, styles, inputs.inline_labels)
        step_pieces = subassembly.build.pieces_placed_directly(layer)
        notes += hold_notes(model, step_pieces, subassembly.build.floating, styles)
        if step_number == len(steps):
            notes += _finished_note(model, subassembly, styles)
        step_page = LayerPage(
            heading,
            step_pieces,
            images.view_path_by_layer[layer],
            f"3D view of {subassembly.name} so far (this step in colour):",
            notes,
            images.map_path_by_layer[layer],
        )
        story += layer_page_flowables(step_page, inputs.is_owned, styles)
        story.append(PageBreak())
    return story


def _compact_block(inputs: PageInputs, subassembly: SubAssembly) -> KeepTogether:
    """One small sub-assembly on a shared page: name + what it is, its parts table, its steps as maps in a row."""
    model, styles = inputs.model, inputs.styles
    steps = subassembly.step_layers
    title = Paragraph(
        f"{_name_html(subassembly)}: {len(subassembly.pieces)} pieces, {len(steps)} step(s) "
        f"({_short_layer_span(model, subassembly.pieces)}). Build it from the bottom up, one map after the other.",
        styles["Normal"],
    )
    return KeepTogether(
        [
            title,
            Spacer(1, 3),
            parts_table(list(subassembly.pieces), font_size=8, is_owned=inputs.is_owned),
            Spacer(1, 4),
            _step_map_row(inputs, subassembly, COMPACT_MAP_HEIGHT_INCH),
            Spacer(1, 10),
        ]
    )


def _step_map_row(inputs: PageInputs, subassembly: SubAssembly, map_height_inch: float) -> Table:
    """The sub-assembly's step maps side by side over the page width, each at most map_height_inch high."""
    steps = subassembly.step_layers
    map_width_inch = (PAGE_CONTENT_WIDTH_INCH - COMPACT_MAP_GAP_INCH * (len(steps) - 1)) / len(steps)
    maps = [
        RLImage(
            inputs.images[subassembly.name].map_path_by_layer[layer],
            width=map_width_inch * inch,
            height=map_height_inch * inch,
            kind="proportional",
        )
        for layer in steps
    ]
    map_row = Table([maps], colWidths=[(map_width_inch + COMPACT_MAP_GAP_INCH) * inch] * len(steps))
    map_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return map_row


def _inline_block(inputs: PageInputs, subassembly: SubAssembly) -> list[Flowable]:
    """A small part joined inside a step of a big sub-assembly, shown on that step page: "3a: first connect these
    N pieces", its parts table and its steps as small maps in a row."""
    steps = subassembly.step_layers
    label = inputs.inline_labels[subassembly.name]
    title = Paragraph(
        f"<font color='{colour_hex(subassembly)}'><b>{label}:</b></font> first connect these "
        f"{len(subassembly.pieces)} pieces on their own, from the bottom up ("
        + ", then ".join(f"map {label}: {number}/{len(steps)}" for number in range(1, len(steps) + 1))
        + "): they hang in the air "
        "until this step joins them.",
        inputs.styles["Normal"],
    )
    return [
        title,
        Spacer(1, 3),
        parts_table(list(subassembly.pieces), font_size=8, is_owned=inputs.is_owned),
        Spacer(1, 3),
        _step_map_row(inputs, subassembly, INLINE_MAP_HEIGHT_INCH),
        Spacer(1, 6),
    ]


def _pre_step_flowables(inputs: PageInputs, parent: SubAssembly, layer: int) -> list[Flowable]:
    """The pre-step blocks (3a, 3b ...) of the small parts joined on the parent's step `layer`."""
    flowables: list[Flowable] = []
    for child in parent.build.subassemblies:
        if child.join_layer == layer and child.name in inputs.inline_labels:
            flowables += _inline_block(inputs, child)
    return flowables


def _height_inch(flowables: Sequence[Flowable]) -> float:
    """Height of the flowables laid out on the page width."""
    page_width = PAGE_CONTENT_WIDTH_INCH * inch
    return float(sum(flowable.wrap(page_width, LAYOUT_HEIGHT_INCH * inch)[1] for flowable in flowables) / inch)


def _pre_step_pages(
    inputs: PageInputs, parent: SubAssembly, layer: int, step_heading: str, flowables: list[Flowable]
) -> list[Flowable]:
    """Page(s) with the pre-steps of a step that has too many to share its page, followed by a page break."""
    labels = [
        inputs.inline_labels[child.name]
        for child in parent.build.subassemblies
        if child.join_layer == layer and child.name in inputs.inline_labels
    ]
    span = labels[0] if len(labels) == 1 else f"{labels[0]}-{labels[-1]}"
    return [
        Paragraph(f"{step_heading}: pre-steps {span}", inputs.styles["Heading2"]),
        Paragraph(
            "Small parts that hang in the air until this step joins them: connect each one first. The "
            "next page puts them in place.",
            inputs.styles["Normal"],
        ),
        Spacer(1, 6),
        *flowables,
        PageBreak(),
    ]


def _compact_page(inputs: PageInputs, subassemblies: list[SubAssembly], target: str) -> list[Flowable]:
    """Shared page of the small sub-assemblies placed on the same step (target: 'Layer 69' or 'S4 step 3 (Layer
    10)'), followed by a page break."""
    styles = inputs.styles
    names = ", ".join(subassembly.name for subassembly in subassemblies)
    plural = len(subassemblies) > 1
    heading = f"Sub-assembl{'ies' if plural else 'y'} {names} (for {target})"
    intro = Paragraph(
        f"Small part{'s' if plural else ''} that hang{'' if plural else 's'} in the air while building: build "
        f"{'each one' if plural else 'it'} here, with the piece(s) that clamp it on top, then place "
        f"{'them' if plural else 'it'} on {target}. The axis numbers are those of the main maps.",
        styles["Normal"],
    )
    blocks = [_compact_block(inputs, subassembly) for subassembly in subassemblies]
    return [Paragraph(heading, styles["Heading2"]), intro, Spacer(1, 6), *blocks, PageBreak()]


def pages_before_layer(
    inputs: PageInputs, build: Build, layer: int, parent: SubAssembly | None = None
) -> list[Flowable]:
    """Pages of the sub-assemblies of `build` placed on `layer` (of the main model, or of the sub-assembly
    `parent`): the shared page of the small ones, then the pages of each big one (with page breaks)."""
    placed = [
        subassembly
        for subassembly in build.subassemblies
        if subassembly.join_layer == layer and subassembly.name not in inputs.inline_labels
    ]
    compact = [subassembly for subassembly in placed if subassembly.is_compact]
    target = layer_name(inputs.model, layer)
    if parent is not None:
        target = f"{parent.name} step {parent.step_layers.index(layer) + 1} ({target})"
    story = _compact_page(inputs, compact, target) if compact else []
    for subassembly in placed:
        if not subassembly.is_compact:
            story += _big_subassembly_pages(inputs, subassembly)
    return story


def describe_build_plan(model: LegoModel, plan: Build) -> str:
    """'Build plan: 38 sub-assemblies for the parts that hang while building (S4 63 pieces, layers 8-21, placed on
    Layer 21; ... ; 25 small ones on shared pages; 17 more inside them)' (console and cover page)."""
    if not plan.subassemblies:
        return "Build plan: nothing hangs in the air, no sub-assemblies"
    big = [subassembly for subassembly in plan.subassemblies if not subassembly.is_compact]
    parts = [
        f"{subassembly.name} {len(subassembly.pieces)} pieces ({_short_layer_span(model, subassembly.pieces)}, "
        f"placed on {layer_name(model, subassembly.join_layer)})"
        for subassembly in big
    ]
    small_count = len(plan.subassemblies) - len(big)
    nested_count = len(plan.all_subassemblies()) - len(plan.subassemblies)
    parts.append(f"{small_count} small one(s) on shared pages")
    if nested_count:
        parts.append(f"{nested_count} small part(s) joined inside them as pre-steps (3a, 3b ...)")
    held = f"; {len(plan.floating)} piece(s) to hold in place" if plan.floating else ""
    return (
        f"Build plan: {len(plan.subassemblies)} sub-assemblies for the parts that hang while building ("
        + "; ".join(parts)
        + ")"
        + held
    )
