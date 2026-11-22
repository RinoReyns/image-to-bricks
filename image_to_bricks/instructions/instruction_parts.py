"""Shared pieces of the instruction pages (layer pages, sub-assembly pages): piece colours, layer names, piece
descriptions and the parts table."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import StyleSheet1
from reportlab.lib.units import inch
from reportlab.platypus import Flowable, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import Image as RLImage

from image_to_bricks.consts import BRICK_HEIGHT_PLATES, COLOR_MPL, COLOR_NAMES
from image_to_bricks.instructions import fast_render as fr
from image_to_bricks.model import LegoModel, PlacedBrick, piece_mask

UNKNOWN_COLOUR_HEX = "#4169e1"  # royal blue for a colour id missing from the palette
_RGB_CACHE: dict[int, tuple[int, int, int]] = {}

type PartsKey = tuple[int, int, int, int, bool]  # (height plates, long side, short side, colour, is L piece)


def colour_rgb(color_id: int) -> tuple[int, int, int]:
    """(r, g, b) 0..255 of a palette colour id (royal blue for an unknown id), cached."""
    if color_id not in _RGB_CACHE:
        _RGB_CACHE[color_id] = fr.hex_to_rgb(COLOR_MPL.get(color_id, UNKNOWN_COLOUR_HEX))
    return _RGB_CACHE[color_id]


def cells_coloured_by_pieces(pieces: Sequence[PlacedBrick], grid_shape: tuple[int, ...]) -> NDArray[np.uint8]:
    """uint8 [z, y, x, 3] colour of every cell covered by the pieces (L pieces: not the notch cell), 0 elsewhere."""
    colours = np.zeros((*grid_shape, 3), np.uint8)
    for piece in pieces:
        piece_cells_zyx = colours[
            piece.z : piece.z + piece.h_plates, piece.y : piece.y + piece.d, piece.x : piece.x + piece.w
        ]
        if piece.notch >= 0:
            piece_cells_zyx[:, piece_mask(piece)] = colour_rgb(piece.color)
        else:
            piece_cells_zyx[:] = colour_rgb(piece.color)
    return colours


def cells_covered_by_pieces(pieces: Sequence[PlacedBrick], grid_shape: tuple[int, ...]) -> NDArray[np.bool_]:
    """bool [z, y, x]: True on every cell covered by the pieces (L pieces: not the notch cell)."""
    cells = np.zeros(tuple(grid_shape), bool)
    for piece in pieces:
        cells[piece.z : piece.z + piece.h_plates, piece.y : piece.y + piece.d, piece.x : piece.x + piece.w] |= (
            piece_mask(piece)[np.newaxis]
        )
    return cells


def piece_numbers_by_cell(pieces: Sequence[PlacedBrick], grid_shape: tuple[int, ...]) -> NDArray[np.int32]:
    """int32 [z, y, x]: 1-based index of the piece covering each cell (L pieces: not the notch cell), 0 elsewhere
    (for fast_render.iso_render(piece_ids_zyx=...): lines only between pieces)."""
    numbers = np.zeros(tuple(grid_shape), np.int32)
    for number, piece in enumerate(pieces, start=1):
        piece_cells_zyx = numbers[
            piece.z : piece.z + piece.h_plates, piece.y : piece.y + piece.d, piece.x : piece.x + piece.w
        ]
        piece_cells_zyx[:, piece_mask(piece)] = number
    return numbers


def layer_label(model: LegoModel, z: int) -> str:
    """'Base layer 1 / 2' for the stand, then 'Layer 1 / n' counted from the figure's first layer."""
    base_layers = model.base_layers
    if z < base_layers:
        return f"Base layer {z + 1} / {base_layers}"
    return f"Layer {z - base_layers + 1} / {model.grid.shape[0] - base_layers}"


def layer_name(model: LegoModel, z: int) -> str:
    """'Layer 10' / 'Base layer 2' (the page heading, without ' / total')."""
    return layer_label(model, z).split(" / ")[0]


OWNED_TABLE_COLUMN_WIDTHS = [0.3 * inch, 1.15 * inch, 0.75 * inch, 0.95 * inch, 0.4 * inch, 0.4 * inch, 0.4 * inch]
PLAIN_TABLE_COLUMN_WIDTHS = [0.3 * inch, 1.6 * inch, 0.8 * inch, 1.0 * inch, 0.6 * inch]
TO_BUY_TEXT_COLOUR = rl_colors.Color(0.8, 0, 0)


def _piece_kind(height_plates: int, *, is_l: bool) -> str:
    """'Plate', 'Brick', 'Corner plate' or 'Corner brick'."""
    is_brick = height_plates == BRICK_HEIGHT_PLATES
    if is_l:
        return "Corner brick" if is_brick else "Corner plate"
    return "Brick" if is_brick else "Plate"


def parts_table(bricks: Sequence[PlacedBrick], font_size: float = 9, is_owned: dict[int, bool] | None = None) -> Table:
    """Parts table grouped by piece, size AND colour, with a colour swatch. With is_owned ({id(piece): owned},
    from owned_layers.allocate_owned_pieces) two more columns say how many come from the user's stock and how many
    must be bought (in red)."""
    counts: dict[PartsKey, int] = {}
    owned_counts: dict[PartsKey, int] = {}
    for b in bricks:
        w, d = sorted((b.w, b.d), reverse=True)
        key = (b.h_plates, w, d, b.color, b.notch >= 0)
        counts[key] = counts.get(key, 0) + 1
        if is_owned is not None and is_owned.get(id(b)):
            owned_counts[key] = owned_counts.get(key, 0) + 1
    header = ["", "Colour", "Piece", "Size (studs)", "Qty"] + (["Own", "Buy"] if is_owned is not None else [])
    data = [header]
    row_styles: list[tuple[object, ...]] = []
    for row, ((h_plates, w, d, color, is_l), qty) in enumerate(
        sorted(counts.items(), key=lambda kv: (COLOR_NAMES.get(kv[0][3], ""), -kv[0][0], kv[0][4], kv[0][1], kv[0][2])),
        start=1,
    ):
        cells = [
            "",
            COLOR_NAMES.get(color, f"Color{color}"),
            _piece_kind(h_plates, is_l=is_l),
            f"{w} x {d} (L, 3 studs)" if is_l else f"{w} x {d}",
            str(qty),
        ]
        if is_owned is not None:
            owned = owned_counts.get((h_plates, w, d, color, is_l), 0)
            cells += [str(owned), str(qty - owned) if qty > owned else "-"]
            if qty > owned:
                row_styles.append(("TEXTCOLOR", (6, row), (6, row), TO_BUY_TEXT_COLOUR))
        data.append(cells)
        r, g, bl = colour_rgb(color)
        row_styles.append(("BACKGROUND", (0, row), (0, row), rl_colors.Color(r / 255, g / 255, bl / 255)))
    widths = OWNED_TABLE_COLUMN_WIDTHS if is_owned is not None else PLAIN_TABLE_COLUMN_WIDTHS
    tbl = Table(data, colWidths=widths)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ALIGN", (4, 1), (-1, -1), "CENTER"),
                *row_styles,
            ]
        )
    )
    return tbl


HOLD_TEXT_COLOUR = "#cc0000"


def hold_notes(
    model: LegoModel, layer_pieces: Sequence[PlacedBrick], floating: dict[int, int], styles: StyleSheet1
) -> list[Flowable]:
    """Note for the pieces of the page with nothing under them (floating = {id(piece): clamping layer or -1}): hold
    them in place until the layer that clamps them."""
    hanging = [(piece, floating[id(piece)]) for piece in layer_pieces if id(piece) in floating]
    if not hanging:
        return []
    base_layers = model.base_layers
    clamp_layers = sorted(
        {(clamp - base_layers + 1) if clamp >= 0 else None for _, clamp in hanging},
        key=lambda layer_number: (layer_number is None, layer_number),
    )
    held = ", ".join(
        f"layer {layer_number}" if layer_number is not None else "nothing above" for layer_number in clamp_layers
    )
    return [
        Paragraph(
            f"<font color='{HOLD_TEXT_COLOUR}'><b>! {len(hanging)} piece(s) on this layer have nothing under them</b>"
            f"</font> (red outline and ! on the map). Hold them in place; they are clamped by {held}.",
            styles["Normal"],
        ),
        Spacer(1, 4),
    ]


PAGE_CONTENT_WIDTH_INCH = 7.5  # letter page minus 0.5 inch margins
PARTS_COLUMN_WIDTH_INCH = 4.4  # top row of a layer page: parts table (left) ...
VIEW_COLUMN_WIDTH_INCH = 3.1  # ... and the 3D view (right)
VIEW_IMAGE_SIZE_INCH = (3.0, 4.0)
VIEW_MIN_HEIGHT_INCH = 2.0  # the 3D view shrinks down to this when the notes leave the map too little room
MAP_HEIGHT_RANGE_INCH = (2.5, 6.8)  # the map takes the space left on the page, within these bounds
MAP_PAGE_RESERVE_PTS = 6 + 30 + 18 + 12  # heading, map label and spacing on a layer page
LAYOUT_HEIGHT_INCH = 10  # height offered to a flowable when measuring it (more than any block needs)
PAGE_MARGINS_PTS = 72  # top + bottom margin of the page frame


@dataclass(frozen=True)
class LayerPage:
    """What one layer page (or sub-assembly step page) shows."""

    heading: str
    pieces: Sequence[PlacedBrick]  # placed on this page (parts table)
    view_image: str | None  # 3D view file (None = no view)
    view_caption: str
    notes: list[Flowable]  # between the top row and the map
    map_image: str  # top-down map file


def _parts_and_view_row(
    page: LayerPage, view_height_inch: float, is_owned: dict[int, bool] | None, styles: StyleSheet1
) -> Table:
    """Top row of a layer page: parts table (left) + 3D view (right)."""
    left: list[Flowable] = [Paragraph("Required parts:", styles["Normal"])]
    if page.pieces:
        left.append(parts_table(page.pieces, font_size=8, is_owned=is_owned))
    right: list[Flowable] = []
    if page.view_image:
        right = [
            Paragraph(page.view_caption, styles["Normal"]),
            RLImage(
                page.view_image,
                width=VIEW_IMAGE_SIZE_INCH[0] * inch,
                height=view_height_inch * inch,
                kind="proportional",
            ),
        ]
    top = Table([[left, right]], colWidths=[PARTS_COLUMN_WIDTH_INCH * inch, VIEW_COLUMN_WIDTH_INCH * inch])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (1, 0), (1, 0), "CENTER")]))
    return top


def _space_left_for_map_inch(top: Table, notes: Sequence[Flowable]) -> float:
    """Height left on the page for the map under the top row and the notes."""
    page_width = PAGE_CONTENT_WIDTH_INCH * inch
    _, top_height_pts = top.wrap(page_width, LAYOUT_HEIGHT_INCH * inch)
    notes_height_pts = sum(note.wrap(page_width, LAYOUT_HEIGHT_INCH * inch)[1] for note in notes)
    frame_height_pts = letter[1] - PAGE_MARGINS_PTS
    return float((frame_height_pts - top_height_pts - notes_height_pts - MAP_PAGE_RESERVE_PTS) / inch)


def layer_page_flowables(page: LayerPage, is_owned: dict[int, bool] | None, styles: StyleSheet1) -> list[Flowable]:
    """One layer page (no page break): heading, parts table (left) + 3D view (right), the notes, then the
    top-down map sized to the space left on the page (the 3D view shrinks when long notes leave the map less than
    its minimum height, so the map stays on the page)."""
    view_height_inch = VIEW_IMAGE_SIZE_INCH[1]
    top = _parts_and_view_row(page, view_height_inch, is_owned, styles)
    space_inch = _space_left_for_map_inch(top, page.notes)
    if space_inch < MAP_HEIGHT_RANGE_INCH[0] and page.view_image:
        view_height_inch = max(VIEW_MIN_HEIGHT_INCH, view_height_inch - (MAP_HEIGHT_RANGE_INCH[0] - space_inch))
        top = _parts_and_view_row(page, view_height_inch, is_owned, styles)
        space_inch = _space_left_for_map_inch(top, page.notes)
    map_height_inch = max(MAP_HEIGHT_RANGE_INCH[0], min(MAP_HEIGHT_RANGE_INCH[1], space_inch))
    return [
        Paragraph(page.heading, styles["Heading2"]),
        top,
        Spacer(1, 6),
        *page.notes,
        Paragraph("Top-down placement diagram:", styles["Normal"]),
        RLImage(
            page.map_image, width=PAGE_CONTENT_WIDTH_INCH * inch, height=map_height_inch * inch, kind="proportional"
        ),
    ]
