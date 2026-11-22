"""Writing a finished build: inventory CSV, layer maps, isometric views, step renders and the instruction PDF."""

import os
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer
from reportlab.platypus import Image as RLImage

from image_to_bricks.analysis import build_order
from image_to_bricks.consts import PLATE_MM, STUD_MM
from image_to_bricks.instructions import build_maps, owned_layers, subassembly_pages
from image_to_bricks.instructions import fast_render as fr
from image_to_bricks.instructions.instruction_parts import (
    LayerPage,
    cells_coloured_by_pieces,
    cells_covered_by_pieces,
    hold_notes,
    layer_label,
    layer_name,
    layer_page_flowables,
    parts_table,
    piece_numbers_by_cell,
)
from image_to_bricks.instructions.owned_layers import LayerOwnership
from image_to_bricks.instructions.view_sides import diagonal_view_name, front_or_default
from image_to_bricks.model import LegoModel, PlacedBrick, export_inventory_csv, plate_origin
from image_to_bricks.options import OutputOptions
from image_to_bricks.stock import Stock


def model_colors(model: LegoModel) -> NDArray[np.uint8]:
    """uint8 [z,y,x,3] colour of every cell, from the placed bricks."""
    return cells_coloured_by_pieces(model.bricks, model.grid.shape)


def save_model(model: LegoModel, path: str) -> None:
    """Pickle the model (the --save-model file); prints the path."""
    print(path)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str) -> LegoModel:
    """The model pickled by save_model. Raises TypeError if the file holds something else."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, LegoModel):
        raise TypeError(f"{path} does not hold a LegoModel (got {type(model).__name__})")
    return model


def unsupported_pieces(model: LegoModel) -> dict[int, int]:
    """Pieces with nothing directly under them (no piece in the layer below touches their footprint), built
    bottom-up they float until a piece above clamps them. Returns {id(piece): layer that holds it (z index of
    the lowest piece above), or -1 if nothing is above}. Base layer 0 rests on the ground
    (build_order.find_unsupported_pieces)."""
    return build_order.find_unsupported_pieces(model.bricks)


MAP_MARGIN_STUDS = 2  # empty studs around what a layer map shows


def map_region(
    model: LegoModel,
    z: int,
    *,
    show_layer_below: bool,
    margin_studs: int = MAP_MARGIN_STUDS,
    plan: build_order.Build | None = None,
) -> tuple[int, int, int, int]:
    """Grid region (x0, x1, y0, y1; ends exclusive) of the layer map of layer z: the pieces the map shows (with
    plan: not the sub-assemblies placed later), plus the layer below when it is shown, plus margin_studs, clipped to
    the grid. A layer that shows nothing falls back to its grid footprint."""
    context = build_maps.build_map_context(
        model, plan or build_order.empty_build(model.bricks), show_layer_below=show_layer_below
    )
    region = build_maps.shown_region(context, z, margin_studs)
    if region is not None:
        return region
    _, depth, width = model.grid.shape
    ys, xs = np.nonzero(model.grid[z])
    return (
        max(int(xs.min()) - margin_studs, 0),
        min(int(xs.max()) + 1 + margin_studs, width),
        max(int(ys.min()) - margin_studs, 0),
        min(int(ys.max()) + 1 + margin_studs, depth),
    )


def generate_build_layers(
    model: LegoModel, outdir: str, *, show_layer_below: bool = True, plan: build_order.Build | None = None
) -> list[str]:
    """Top-down map per plate layer (fast PIL), see build_maps.draw_build_layer_map: tall bricks that started on a
    lower layer are shown faded; new pieces have a bold outline + size label. show_layer_below: the pieces of the
    layer below are drawn hatched where this layer leaves them uncovered (+ a legend row). plan (build_order.Build,
    default: build_order.plan_build of the model): sub-assembly pieces are not drawn before the layer they are
    placed on, and drawn faded with the sub-assembly's name on it."""
    print("Generating build layer images...")
    plan = plan or build_order.plan_build(model.bricks)
    os.makedirs(outdir, exist_ok=True)
    context = build_maps.build_map_context(model, plan, show_layer_below=show_layer_below)
    image_paths = []
    for z in range(model.grid.shape[0]):
        if not model.grid[z].any():
            continue
        region = build_maps.shown_region(context, z, MAP_MARGIN_STUDS) or map_region(
            model, z, show_layer_below=show_layer_below
        )
        img = build_maps.draw_build_layer_map(context, z, region, layer_label(model, z))
        img_path = os.path.join(outdir, f"layer_{z + 1:03d}.png")
        img.save(img_path)
        image_paths.append(img_path)
    return image_paths


class VoxelAxes(Protocol):
    """The part of matplotlib's Axes3D (mpl_toolkits, untyped) that draw_voxel_scene uses."""

    def voxels(
        self, filled: NDArray[np.bool_], /, *, facecolors: NDArray[Any], edgecolor: str, linewidth: float
    ) -> object:
        """Draw filled voxels."""
        ...

    def set_box_aspect(self, aspect: tuple[float, float, float]) -> None:
        """Axis length ratios."""
        ...

    def view_init(self, elev: float, azim: float) -> None:
        """Camera angles in degrees."""
        ...

    def set_axis_off(self) -> None:
        """Hide the axes."""
        ...


def draw_voxel_scene(
    ax: VoxelAxes,
    filled_xyz: NDArray[np.bool_],
    facecolors_xyz: NDArray[Any],
    depth_z: int | None = None,
    view_angles: tuple[float, float] = (90, 0),
) -> None:
    """Voxels [x, y, z] in true LEGO proportions (with depth_z), seen from view_angles (elevation, azimuth)."""
    elev, azim = view_angles
    ax.voxels(filled_xyz, facecolors=facecolors_xyz, edgecolor="black", linewidth=0.15)
    w, h, _ = filled_xyz.shape
    # true physical proportions: x/y are stud-pitch (8mm), z is plate-height (3.2mm)
    if depth_z:
        ax.set_box_aspect((w, h, max(1, depth_z) * (PLATE_MM / STUD_MM)))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def isometric_view_names(front: str | None) -> list[tuple[str, str, int]]:
    """(file name part, title, iso_render view) for the four isometric views, named by the camera position
    relative to the figure: 'front-left' = in front of the figure, on the figure's own left (--front)."""
    names = []
    for view, camera_side in sorted(fr.ISO_CAMERA_SIDES.items()):
        name = diagonal_view_name(camera_side, front_or_default(front))
        names.append((name.replace("-", "_"), name.capitalize(), view))
    return names


def generate_isometric_view_set(model: LegoModel, outdir: str, model_name: str) -> str:
    """Save four isometric angles and a 2x2 overview for instructions."""
    print("Generating isometric_preview...")
    os.makedirs(outdir, exist_ok=True)
    cols = model_colors(model)
    piece_numbers = piece_numbers_by_cell(model.bricks, model.grid.shape)
    images = []
    for filename, title, view in isometric_view_names(getattr(model, "front", None)):
        img = fr.iso_render(model.grid, cols, fr.IsoView(view=view, title=title, piece_ids_zyx=piece_numbers))
        img.save(os.path.join(outdir, f"{model_name}_isometric_{filename}.png"))
        images.append(img)
    overview_path = os.path.join(outdir, f"{model_name}_isometric_overview.png")
    fr.grid_2x2(images).save(overview_path)
    return overview_path


def _cells_not_attached_after_layers(model: LegoModel, plan: build_order.Build) -> list[tuple[int, NDArray[np.bool_]]]:
    """[(join layer, bool [z, y, x] cells of the sub-assembly)]: those cells are not on the model before the end
    of the join layer's page."""
    return [
        (subassembly.join_layer, cells_covered_by_pieces(subassembly.pieces, model.grid.shape))
        for subassembly in plan.subassemblies
    ]


def generate_cumulative_layer_renders(
    model: LegoModel, outdir: str, plan: build_order.Build | None = None
) -> list[str]:
    """For each plate layer N: isometric view of everything built through N.
    Earlier layers are faded, layer N is shown in full colour. plan: sub-assemblies are left out until their join
    layer."""
    print("Generating cumulative isometric instruction views...")
    os.makedirs(outdir, exist_ok=True)
    pending_subassemblies = _cells_not_attached_after_layers(model, plan) if plan else []
    depth_z = model.grid.shape[0]
    cols = model_colors(model)
    piece_numbers = piece_numbers_by_cell(model.bricks, model.grid.shape)
    faded = (cols.astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)
    ys, xs = np.nonzero(model.grid.any(axis=0))
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    image_paths = []
    for z in range(depth_z):
        if not model.grid[z].any():
            continue
        c = faded[: z + 1].copy()
        c[z] = cols[z]
        built_cells = model.grid[: z + 1].copy()
        for join_layer, subassembly_cells in pending_subassemblies:
            if z < join_layer:
                built_cells &= ~subassembly_cells[: z + 1]
        # frame = what is built so far (constant scale, no empty space above)
        img = fr.iso_render(
            built_cells[:, y0:y1, x0:x1],
            c[:, y0:y1, x0:x1],
            fr.IsoView(
                px=14,
                title=layer_label(model, z),
                frame_grid_zyx=built_cells[:, y0:y1, x0:x1],
                piece_ids_zyx=piece_numbers[: z + 1, y0:y1, x0:x1],
            ),
        )
        img_path = os.path.join(outdir, f"step_{z + 1:03d}.png")
        img.save(img_path)
        image_paths.append(img_path)
    return image_paths


def _ownership_paragraph(record: LayerOwnership, styles: StyleSheet1, *, is_buildable: bool) -> Paragraph:
    """Layer-page line: owned / to buy, and whether the layer can be built now from owned pieces."""
    if record.is_complete:
        status = "<font color='#007a00'><b>All pieces owned</b></font>" + (
            " - you can build this layer now." if is_buildable else " (but a layer below still needs pieces to buy)."
        )
    else:
        status = (
            f"<font color='#cc0000'><b>To buy: {record.pieces_to_buy} piece(s)</b></font> (column Buy in the table)."
        )
    return Paragraph(
        f"From your stock: {record.owned_pieces} of {record.owned_pieces + record.pieces_to_buy} "
        f"pieces owned. {status}",
        styles["Normal"],
    )


@dataclass(frozen=True)
class LayerPageContext:
    """What every layer page needs besides its own pieces: plan = build_order.Build (sub-assemblies, pieces to hold),
    ownership = {layer: owned_layers record of the pieces placed directly on that page}, buildable_layers = layers
    buildable bottom-up from owned pieces, is_owned = {id(piece): owned} (Own / Buy columns) or None."""

    plan: build_order.Build
    ownership: dict[int, LayerOwnership]
    buildable_layers: int
    is_owned: dict[int, bool] | None = None


def _layer_notes(
    model: LegoModel,
    layer_index: int,
    layer_bricks: list[PlacedBrick],
    context: LayerPageContext,
    styles: StyleSheet1,
) -> list[Flowable]:
    """Flowables between a layer page's parts row and its map: the "from your stock" line, sub-assemblies placed on
    this layer, special parts and pieces with nothing under them. Returned as a list so the page can measure their
    height."""
    notes: list[Flowable] = []
    if layer_index in context.ownership:
        notes.append(
            _ownership_paragraph(
                context.ownership[layer_index], styles, is_buildable=layer_index < context.buildable_layers
            )
        )
        notes.append(Spacer(1, 4))
    notes += subassembly_pages.place_note(context.plan, layer_index, styles)
    plate_x0, plate_y0 = plate_origin(model.grid, model.base_layers)
    for accent_z, accent_y, accent_x, text in getattr(model, "accents", []):
        if accent_z == layer_index:
            notes.append(
                Paragraph(
                    f"<b>* Special part on this layer:</b> {text}, at stud x={accent_x - plate_x0}, "
                    f"y={accent_y - plate_y0} (axis numbers of the map: 0, 0 = corner of the base "
                    f"plate).",
                    styles["Normal"],
                )
            )
            notes.append(Spacer(1, 4))
    notes += hold_notes(model, layer_bricks, context.plan.floating, styles)
    return notes


@dataclass(frozen=True)
class InstructionImages:
    """Image files of the build steps: layer_maps[z] and step_views[z] (3D view of the build so far) per layer,
    subassemblies = {name: subassembly_pages.SubAssemblyImages}."""

    layer_maps: list[str]
    step_views: list[str]
    subassemblies: dict[str, "subassembly_pages.SubAssemblyImages"]


def _skipped_layers_note(model: LegoModel, skipped_layers: list[int], styles: StyleSheet1) -> list[Flowable]:
    """Note on the first page after layers that got no page (nothing to place on them)."""
    if not skipped_layers:
        return []
    first, last = (layer_name(model, layer) for layer in (skipped_layers[0], skipped_layers[-1]))
    is_single = len(skipped_layers) == 1
    span = first if is_single else f"Layers {first.split()[-1]}-{last.split()[-1]}"
    return [
        Paragraph(
            f"<b>{span}: nothing new to place</b> (the bricks from below reach up through "
            f"{'it' if is_single else 'them'}), so {'it has' if is_single else 'they have'} no page.",
            styles["Normal"],
        ),
        Spacer(1, 4),
    ]


def build_step_pages(
    model: LegoModel, images: InstructionImages, context: LayerPageContext, styles: StyleSheet1
) -> list[Flowable]:
    """Flowables of the build steps: one page per layer with something to do (pieces placed directly on it or a
    sub-assembly placed on it), with the pages of those sub-assemblies right before it. Layers with nothing to do
    get no page; the next page says so."""
    plan = context.plan
    page_inputs = subassembly_pages.PageInputs(
        model, images.subassemblies, context.is_owned, styles, subassembly_pages.inline_step_labels(plan)
    )
    layers_with_a_step = set(plan.step_layers)
    pages: list[list[Flowable]] = []
    skipped_layers: list[int] = []
    for layer, map_path in enumerate(images.layer_maps):
        if layer not in layers_with_a_step:
            skipped_layers.append(layer)
            continue
        page = subassembly_pages.pages_before_layer(page_inputs, plan, layer)
        layer_pieces = plan.pieces_placed_directly(layer)
        notes = _skipped_layers_note(model, skipped_layers, styles) + _layer_notes(
            model, layer, layer_pieces, context, styles
        )
        skipped_layers = []
        layer_page = LayerPage(
            layer_name(model, layer),
            layer_pieces,
            images.step_views[layer] if layer < len(images.step_views) else None,
            "3D view (current layer in colour):",
            notes,
            map_path,
        )
        page += layer_page_flowables(layer_page, context.is_owned, styles)
        pages.append(page)
    story: list[Flowable] = []
    for page_number, page in enumerate(pages):
        story += page + ([PageBreak()] if page_number != len(pages) - 1 else [])
    return story


@dataclass(frozen=True)
class InstructionPdf:
    """What the PDF shows besides the model and its layer maps: model_name (title), cover_image (isometric
    overview), iso_step_images (3D view per layer), is_owned (owned_layers.allocate_owned_pieces, with --stock: Own /
    Buy columns and a per-layer "from your stock" line), plan (default build_order.plan_build of the model) and
    subassembly_images (default: written to layers/subassemblies next to the PDF)."""

    model_name: str = "model"
    cover_image: str | None = None
    iso_step_images: Sequence[str] = field(default_factory=list)
    is_owned: dict[int, bool] | None = None
    plan: build_order.Build | None = None
    subassembly_images: dict[str, "subassembly_pages.SubAssemblyImages"] | None = None


def _cover_page(
    model: LegoModel, details: InstructionPdf, plan: build_order.Build, layer_count: int, styles: StyleSheet1
) -> list[Flowable]:
    """Title, piece counts, special parts, the sub-assembly summary, stock reach and the cover picture."""
    story: list[Flowable] = [
        Paragraph(f"LEGO Build Instructions: {details.model_name}", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Total bricks/plates: {len(model.bricks)}", styles["Normal"]),
    ]
    if model.base_layers:
        base_count = sum(1 for b in model.bricks if b.z < model.base_layers)
        story.append(
            Paragraph(
                f"Base: {model.base_layers} plate layer(s), {base_count} pieces; figure: "
                f"{len(model.bricks) - base_count} pieces",
                styles["Normal"],
            )
        )
    for _, _, _, text in getattr(model, "accents", []):
        story.append(
            Paragraph(f"<b>Special part:</b> {text} (marked with a star on its layer page).", styles["Normal"])
        )
    floating = unsupported_pieces(model)
    if floating:
        story.append(
            Paragraph(
                f"<font color='#cc0000'>{len(floating)} piece(s) have nothing under them</font> while building from "
                f"the bottom (tail, brim, edges). Every part that hangs is a sub-assembly: built on its own page(s) "
                f"right before the layer it is placed on, together with the pieces that clamp it, then placed as one "
                f"unit (its pieces are marked with its name on that layer's map). "
                f"{subassembly_pages.describe_build_plan(model, plan)}.",
                styles["Normal"],
            )
        )
    story.append(Paragraph(f"Total layers: {layer_count}", styles["Normal"]))
    if details.is_owned is not None:
        story.append(_stock_reach_paragraph(model, details.is_owned, styles))
    if details.cover_image:
        story.append(Spacer(1, 12))
        story.append(RLImage(details.cover_image, width=5.5 * inch, height=5.5 * inch, kind="proportional"))
    story.append(PageBreak())
    return story


def _stock_reach_paragraph(model: LegoModel, is_owned: dict[int, bool], styles: StyleSheet1) -> Paragraph:
    """Cover line: pieces owned / to buy and how far up the owned pieces alone reach."""
    buildable_layers = owned_layers.count_complete_layers_from_bottom(
        owned_layers.ownership_by_layer(model.bricks, is_owned)
    )
    owned_total = sum(1 for b in model.bricks if is_owned.get(id(b)))
    reach = (
        f"up to and including <b>{layer_label(model, buildable_layers - 1).split(' / ')[0]}</b>"
        if buildable_layers
        else "<b>no complete layer yet</b>"
    )
    return Paragraph(
        f"From your stock: {owned_total} of {len(model.bricks)} pieces owned, {len(model.bricks) - owned_total} "
        f"to buy. With owned pieces only you can build {reach} (owned pieces are used from the bottom up; "
        f"columns Own / Buy on every page).",
        styles["Normal"],
    )


def _parts_list_page(model: LegoModel, is_owned: dict[int, bool] | None, styles: StyleSheet1) -> list[Flowable]:
    """Parts list: the base (build first) and the figure, each as a table."""
    base_layers = model.base_layers
    base_bricks = [b for b in model.bricks if b.z < base_layers]
    figure_bricks = [b for b in model.bricks if b.z >= base_layers]
    story: list[Flowable] = [Paragraph("Parts List", styles["Heading2"])]
    if base_bricks:
        story.append(
            Paragraph(
                f"Base ({base_layers} plate layer(s), {len(base_bricks)} pieces) - build first", styles["Heading3"]
            )
        )
        story.append(parts_table(base_bricks, font_size=9, is_owned=is_owned))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Figure ({len(figure_bricks)} pieces)", styles["Heading3"]))
    story.append(parts_table(figure_bricks, font_size=9, is_owned=is_owned))
    story.append(PageBreak())
    return story


def generate_instruction_pdf(
    model: LegoModel, layer_images: list[str], path: str, details: InstructionPdf | None = None
) -> None:
    """Instruction PDF: cover, parts list and one page per layer with something to do (see InstructionPdf for
    the optional parts). The pages of each sub-assembly are inserted right before the page of the layer it is
    attached on."""
    print("Generating instruction PDF...")
    details = details if details is not None else InstructionPdf()
    plan = details.plan or build_order.plan_build(model.bricks)
    subassembly_images = details.subassembly_images
    if subassembly_images is None:
        subassembly_images = subassembly_pages.generate_subassembly_images(
            model, plan, os.path.join(os.path.dirname(path), "layers", "subassemblies")
        )
    styles = getSampleStyleSheet()
    ownership: dict[int, LayerOwnership] = {}
    buildable_layers = 0
    is_owned = details.is_owned
    if is_owned is not None:
        layer_records = owned_layers.ownership_by_layer(model.bricks, is_owned)
        buildable_layers = owned_layers.count_complete_layers_from_bottom(layer_records)
        ownership = {
            record.layer: record for record in owned_layers.ownership_by_layer(plan.pieces_placed_directly(), is_owned)
        }
    doc = SimpleDocTemplate(path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = _cover_page(model, details, plan, len(layer_images), styles)
    story += _parts_list_page(model, is_owned, styles)
    page_context = LayerPageContext(plan, ownership, buildable_layers, is_owned)
    images = InstructionImages(layer_images, list(details.iso_step_images), subassembly_images)
    story += build_step_pages(model, images, page_context, styles)
    doc.build(story)
    print(f"Instruction PDF saved: {path}")


def _report_ownership(model: LegoModel, stock: Stock, outdir: str, model_name: str) -> dict[int, bool]:
    """With a stock: hands out owned pieces bottom-up, writes <name>_owned_layers.csv, prints the summary line.
    Returns {id(piece): owned}."""
    is_owned = owned_layers.allocate_owned_pieces(model.bricks, stock)
    records = owned_layers.ownership_by_layer(model.bricks, is_owned)
    buildable = owned_layers.count_complete_layers_from_bottom(records)
    path = os.path.join(outdir, f"{model_name}_owned_layers.csv")
    owned_layers.write_ownership_csv(records, path, lambda z: layer_label(model, z).split(" / ")[0])
    reach = layer_label(model, records[buildable - 1].layer).split(" / ")[0] if buildable else "no complete layer"
    print(
        f"Owned layers: {sum(is_owned.values())} of {len(model.bricks)} pieces owned; with owned pieces only "
        f"you can build up to: {reach}; complete layers anywhere: {sum(r.is_complete for r in records)} "
        f"of {len(records)}"
    )
    print(f"  -> {path}")
    return is_owned


def save_outputs(
    model: LegoModel,
    model_name: str,
    outdir: str,
    stock: Stock | None = None,
    options: OutputOptions | None = None,
) -> None:
    """Inventory CSV, layer maps, isometric views, step renders and (options.make_pdf) the instruction PDF. With
    stock the PDF and <name>_owned_layers.csv say, layer by layer, which pieces are owned. Parts that hang while
    building get sub-assembly pages (build_order.plan_build, subassembly_pages)."""
    options = options if options is not None else OutputOptions()
    show_layer_below = options.show_layer_below
    is_owned = _report_ownership(model, stock, outdir, model_name) if stock is not None else None
    csv_path = os.path.join(outdir, f"{model_name}_inventory.csv")
    layers_dir = os.path.join(outdir, "layers")
    export_inventory_csv(model, csv_path)
    floating = unsupported_pieces(model)
    plan = build_order.plan_build(model.bricks)
    if floating:
        print(f"Pieces with nothing under them while building from the bottom: {len(floating)}")
        print(subassembly_pages.describe_build_plan(model, plan))
    layer_images = generate_build_layers(model, layers_dir, show_layer_below=show_layer_below, plan=plan)
    subassembly_images = subassembly_pages.generate_subassembly_images(
        model, plan, os.path.join(layers_dir, "subassemblies"), show_layer_below=show_layer_below
    )
    isometric_overview_path = generate_isometric_view_set(model, layers_dir, model_name)
    iso_step_images = generate_cumulative_layer_renders(model, os.path.join(layers_dir, "cumulative"), plan)
    if options.make_pdf:
        pdf_path = os.path.join(outdir, f"{model_name}_instructions.pdf")
        details = InstructionPdf(
            model_name, isometric_overview_path, iso_step_images, is_owned, plan, subassembly_images
        )
        generate_instruction_pdf(model, layer_images, pdf_path, details)
        print(f"  -> {pdf_path}")
    print("Done.")
    print(f"  -> {csv_path}")
    print(f"  -> {layers_dir}/ ({len(layer_images)} images)")
