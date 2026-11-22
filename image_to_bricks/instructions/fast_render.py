"""Fast PIL-based renderers (replace matplotlib ax.voxels, which is very slow).

iso_render : orthographic isometric view of a voxel model, painter's algorithm,
             only visible faces are drawn.
layer_map  : top-down 2D map of one plate layer with brick outlines + sizes.
"""

from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from image_to_bricks.consts import BRICK_HEIGHT_PLATES, PLATE_MM, STUD_MM
from image_to_bricks.model import PlacedBrick, piece_cells

type RGB = tuple[int, int, int]
type Font = ImageFont.FreeTypeFont | ImageFont.ImageFont
type Point = tuple[float, float]
type ColourOf = Callable[[int], RGB]
type LegendEntry = tuple[str, str]  # (swatch style, text)

WHITE: RGB = (255, 255, 255)
_FONT_CACHE: dict[int, Font] = {}


def font(size: int) -> Font:
    """Scalable font (Pillow >= 10.1 default font, else DejaVu/Arial, else bitmap)."""
    if size not in _FONT_CACHE:
        found: Font | None = None
        for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
            try:
                found = ImageFont.truetype(name, size)
                break
            except OSError:
                pass
        if found is None:
            try:
                found = ImageFont.load_default(size=size)
            except TypeError:
                found = ImageFont.load_default()
        _FONT_CACHE[size] = found
    return _FONT_CACHE[size]


Z_SCALE = PLATE_MM / STUD_MM  # one plate is 0.4 of a stud
_SHADE = {"top": 1.0, "left": 0.80, "right": 0.62}
_FACES = {
    # name: (neighbour offset in xyz, 4 corner offsets in xyz voxel units)
    "left": ((-1, 0, 0), ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))),
    "right": ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    "top": ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
}
MIN_OUTLINE_PX = 6  # iso_render: no outlines below this cell size (they would cover the colours)
ISO_MARGIN_PX = 10  # iso_render: margin around the model
ISO_TITLE_ROW_PX = 34  # iso_render: height of the title row
DARK_TEXT_LUMINANCE = 110  # labels on fills lighter than this are black, else white


def hex_to_rgb(hex_colour: str) -> RGB:
    """'#rrggbb' -> (r, g, b) 0..255."""
    digits = hex_colour.lstrip("#")
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


type Vector = NDArray[np.float64]


def _camera(elev_deg: float) -> tuple[Vector, Vector, Vector]:
    """(forward = camera -> scene, right, up) unit vectors of the isometric camera at this elevation."""
    elevation = np.radians(elev_deg)
    forward = np.array(
        [np.cos(elevation) * np.sqrt(0.5), np.cos(elevation) * np.sqrt(0.5), -np.sin(elevation)]
    )  # camera -> scene
    right = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0.0])
    up = np.cross(right, forward)
    return forward, right, up


def _to_xyz(array: NDArray[Any]) -> NDArray[Any]:
    """[z, y, x, ...] -> [x, y, z, ...]."""
    return np.transpose(array, (2, 1, 0, *tuple(range(3, array.ndim))))


# Where the camera of iso_render(view=...) stands, seen from the model: (sign x, sign y) on the ground.
# Checked by tests/test_views.py (a block with one colour per side).
ISO_CAMERA_SIDES = {0: (1, -1), 1: (1, 1), 2: (-1, 1), 3: (-1, -1)}


def _face_edge_directions(corners: Sequence[tuple[int, int, int]]) -> NDArray[np.int_]:
    """(4, 3) int: for edge k of a face (corner k to k+1), the step to the cell across that edge in the face's
    plane."""
    corner_array = np.array(corners, float)
    centre = corner_array.mean(axis=0)
    midpoints = (corner_array + np.roll(corner_array, -1, axis=0)) / 2
    return np.asarray(np.sign(midpoints - centre), dtype=np.int_)


def _hidden_face_edges(
    cell_indexes: NDArray[np.int_],
    face_offset: tuple[int, int, int],
    edge_directions: NDArray[np.int_],
    padded_cells: NDArray[np.bool_],
    padded_piece_ids: NDArray[Any],
) -> NDArray[np.bool_]:
    """(N, 4) bool: True where the edge of a visible face is inside one piece (the cell across the edge has the
    same face visible and belongs to the same piece), so no line is drawn there. Padded arrays: 1 cell of border."""
    hidden = np.zeros((len(cell_indexes), 4), bool)
    own_ids = padded_piece_ids[tuple((cell_indexes + 1).T)]
    for edge, direction in enumerate(edge_directions):
        across = cell_indexes + 1 + direction
        across_open = across + np.array(face_offset)
        hidden[:, edge] = (
            padded_cells[tuple(across.T)]
            & ~padded_cells[tuple(across_open.T)]
            & (padded_piece_ids[tuple(across.T)] == own_ids)
            & (own_ids > 0)
        )
    return hidden


@dataclass(frozen=True)
class IsoView:
    """How iso_render draws: view 0..3 rotates the model 90 deg about the vertical axis (ISO_CAMERA_SIDES says
    where the camera stands for each; name it with view_sides.diagonal_view_name), px = pixels per stud, elev =
    camera elevation in degrees, title row (None = none), frame_grid_zyx fixes the image frame (pass the full model
    so cumulative steps line up), outline colour (None = no lines), background, piece_ids_zyx: int [z,y,x] piece
    number per cell (0 = none): lines are drawn only between different pieces, so a brick looks like one brick,
    not 3 plates (None: every cell outlined)."""

    view: int = 0
    px: int = 12
    elev: float = 30.0
    title: str | None = None
    frame_grid_zyx: NDArray[Any] | None = None
    outline: RGB | None = (30, 30, 30)
    background: RGB = WHITE
    piece_ids_zyx: NDArray[Any] | None = None


@dataclass
class _Faces:
    """Visible cell faces collected for painting: screen polygons (N, 4, 2), fills (N, 3), depths (N,), hidden
    edges (N, 4)."""

    polygons: list[NDArray[Any]] = field(default_factory=list)
    fills: list[NDArray[np.uint8]] = field(default_factory=list)
    depths: list[NDArray[Any]] = field(default_factory=list)
    hidden_edges: list[NDArray[np.bool_]] = field(default_factory=list)


def _paint_faces(draw: ImageDraw.ImageDraw, faces: _Faces, iso: IsoView, *, has_piece_ids: bool) -> None:
    """Painter's algorithm, far -> near; outlines only from MIN_OUTLINE_PX, only on edges between pieces when
    piece ids are known."""
    polygons, fills = np.concatenate(faces.polygons), np.concatenate(faces.fills)
    depths, hidden_edges = np.concatenate(faces.depths), np.concatenate(faces.hidden_edges)
    draw_outline = iso.px >= MIN_OUTLINE_PX
    for i in np.argsort(-depths, kind="stable"):  # far -> near
        corners_px = [tuple(p) for p in polygons[i]]
        fill = tuple(int(c) for c in fills[i])
        if not has_piece_ids:
            draw.polygon(corners_px, fill=fill, outline=iso.outline if draw_outline else None)
            continue
        draw.polygon(corners_px, fill=fill)
        if draw_outline:
            for edge in range(4):
                if not hidden_edges[i, edge]:
                    draw.line([corners_px[edge], corners_px[(edge + 1) % 4]], fill=iso.outline)


def iso_render(grid_zyx: NDArray[Any], colors_zyx: NDArray[Any], iso: IsoView | None = None) -> Image.Image:
    """grid_zyx: bool [z,y,x]; colors_zyx: uint8 [z,y,x,3]. See IsoView for the view, size, frame and outlines."""
    iso = iso if iso is not None else IsoView()
    turns = (3 - iso.view) % 4  # match the old matplotlib azimuths (-60, 30, 120, 210)
    cells = np.rot90(_to_xyz(grid_zyx.astype(bool)), turns, axes=(0, 1))
    colours = np.rot90(_to_xyz(colors_zyx), turns, axes=(0, 1))
    frame = (
        cells if iso.frame_grid_zyx is None else np.rot90(_to_xyz(iso.frame_grid_zyx.astype(bool)), turns, axes=(0, 1))
    )
    piece_ids = None if iso.piece_ids_zyx is None else np.rot90(_to_xyz(iso.piece_ids_zyx), turns, axes=(0, 1))
    forward, right, up = _camera(iso.elev)
    zs = np.array([1.0, 1.0, Z_SCALE])
    px = iso.px

    # frame from the 8 bbox corners of the frame grid
    nx, ny, nz = frame.shape
    corners = np.array([[x, y, z] for x in (0, nx) for y in (0, ny) for z in (0, nz)], float) * zs
    sx, sy = corners @ right, -(corners @ up)
    margin, title_h = ISO_MARGIN_PX, (ISO_TITLE_ROW_PX if iso.title else 0)
    width_px = int((sx.max() - sx.min()) * px) + 2 * margin
    height_px = int((sy.max() - sy.min()) * px) + 2 * margin + title_h
    ox, oy = -sx.min() * px + margin, -sy.min() * px + margin + title_h

    faces = _Faces()
    pad = np.pad(cells, 1, constant_values=False)
    padded_ids = None if piece_ids is None else np.pad(piece_ids, 1, constant_values=0)
    for name, (off, cs) in _FACES.items():
        neighbours = pad[
            1 + off[0] : 1 + off[0] + cells.shape[0],
            1 + off[1] : 1 + off[1] + cells.shape[1],
            1 + off[2] : 1 + off[2] + cells.shape[2],
        ]
        idx = np.argwhere(cells & ~neighbours)
        if len(idx) == 0:
            continue
        pts = (idx[:, None, :] + np.array(cs)[None]) * zs  # (N,4,3)
        faces.polygons.append(np.stack([pts @ right * px + ox, -(pts @ up) * px + oy], -1))  # (N,4,2)
        faces.fills.append(
            (colours[idx[:, 0], idx[:, 1], idx[:, 2]].astype(float) * _SHADE[name]).clip(0, 255).astype(np.uint8)
        )
        faces.depths.append(((idx + 0.5) * zs) @ forward)
        faces.hidden_edges.append(
            np.zeros((len(idx), 4), bool)
            if padded_ids is None
            else _hidden_face_edges(idx, off, _face_edge_directions(cs), pad, padded_ids)
        )
    img = Image.new("RGB", (max(width_px, 1), max(height_px, 1)), iso.background)
    draw = ImageDraw.Draw(img)
    if faces.polygons:
        _paint_faces(draw, faces, iso, has_piece_ids=piece_ids is not None)
    if iso.title:
        draw.text((margin, 4), iso.title, fill=(0, 0, 0), font=font(24))
    return img


def side_by_side(images: Sequence[Image.Image], background: RGB = WHITE) -> Image.Image:
    """The images in one row, top-aligned."""
    width_px = sum(i.width for i in images)
    height_px = max(i.height for i in images)
    out = Image.new("RGB", (width_px, height_px), background)
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def grid_2x2(images: Sequence[Image.Image], background: RGB = WHITE) -> Image.Image:
    """The first 4 images in a 2 x 2 grid of equal cells."""
    w = max(i.width for i in images)
    h = max(i.height for i in images)
    out = Image.new("RGB", (2 * w, 2 * h), background)
    for k, im in enumerate(images[:4]):
        out.paste(im, ((k % 2) * w, (k // 2) * h))
    return out


def _text_color(rgb: RGB) -> RGB:
    """Black on light fills, white on dark ones."""
    return (0, 0, 0) if (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) > DARK_TEXT_LUMINANCE else WHITE


def _l_outline(px0: float, py0: float, px1: float, py1: float, notch: int) -> list[Point]:
    """Outline (pixel points) of an L piece: the box minus the quadrant `notch` (0 = low x / low y,
    1 = high x / low y, 2 = low x / high y, 3 = high x / high y; y points UP in the map, DOWN in pixels)."""
    xm, ym = (px0 + px1) / 2, (py0 + py1) / 2
    # pixel corners clockwise: TL (low x, high y), TR (high x, high y), BR (high x, low y), BL (low x, low y)
    corners: list[Point] = [(px0, py0), (px1, py0), (px1, py1), (px0, py1)]
    cut = {2: 0, 3: 1, 1: 2, 0: 3}[notch]  # which pixel corner is the notch
    before = [(px0, ym), (xm, py0), (px1, ym), (xm, py1)]
    after = [(xm, py0), (px1, ym), (xm, py1), (px0, ym)]
    out: list[Point] = []
    for i, c in enumerate(corners):
        if i == cut:
            out += [before[i], (xm, ym), after[i]]
        else:
            out.append(c)
    return out


MAP_TITLE_ROW_PX = 40  # layer map: height of the title row
MAP_MARGIN_PX = 34  # layer map: margin around the stud grid (axis numbers live here)


HOLD_MARK = ((214, 0, 0), "!")  # piece with nothing under it: hold it in place
SUBASSEMBLY_OUTLINE_COLOURS = (
    (128, 0, 170),
    (0, 120, 120),
    (200, 0, 110),
    (30, 80, 200),
    (150, 85, 0),
    (0, 120, 40),
)  # outline + name of a sub-assembly placed as a unit (one per neighbour)
SUBASSEMBLY_OUTLINE_WIDTH_PX = 3
SUBASSEMBLY_FADE = 0.7  # its fill is faded this far toward white (built on its own pages)


def subassembly_outline_colour(colour_index: int) -> tuple[int, int, int]:
    """Outline / name colour of a sub-assembly (build_order.SubAssembly.colour_index)."""
    return SUBASSEMBLY_OUTLINE_COLOURS[colour_index % len(SUBASSEMBLY_OUTLINE_COLOURS)]


def hanging_mark(piece_id: int, floating: Collection[int]) -> tuple[RGB, str] | None:
    """(outline colour, symbol) for a piece that hangs in the air, None for a piece that rests on something."""
    return HOLD_MARK if piece_id in floating else None


@dataclass(frozen=True)
class MapRegion:
    """Grid region a layer map shows: x0..x0+width_studs, y0..y0+depth_studs (grid coordinates), cell = pixels
    per stud, and the base plate's corner (plate_x0, plate_y0): axis numbers (every 5 studs) count from it, so they
    can be negative where the figure overhangs the plate."""

    x0: int
    y0: int
    width_studs: int
    depth_studs: int
    cell: int = 34
    plate_x0: int = 0
    plate_y0: int = 0


@dataclass(frozen=True)
class MapMarks:
    """What a layer map marks besides the pieces. new_z: pieces that started on a lower layer (b.z < new_z) are drawn
    faded and unlabelled so new pieces stand out. floating: ids of pieces with nothing under them (red outline +
    "!"). front: "-y"/"+y"/"-x"/"+x" draws an arrow marked FRONT (the side the figure's face looks to) in the title
    row. below_bricks: the pieces of the layer below; their studs that this layer leaves uncovered are drawn
    hatched (no fill, dashed outline) under the map and a legend row is added at the bottom (None = no layer below
    and no legend). subassembly_marks: {id(piece): ("S1", outline rgb)} for pieces of a sub-assembly placed as a
    unit on this page (faded fill, outline + name in its colour, no size label). show_legend: False leaves the
    legend out even with below_bricks (small maps next to each other)."""

    new_z: int | None = None
    floating: Collection[int] = ()
    front: str | None = None
    below_bricks: Sequence[PlacedBrick] | None = None
    subassembly_marks: dict[int, tuple[str, RGB]] = field(default_factory=dict)
    show_legend: bool = True

    def is_continuing(self, piece: PlacedBrick) -> bool:
        """True for a tall brick that started on a lower layer."""
        return self.new_z is not None and piece.z < self.new_z


FRONT_ARROW_STEPS = {"-y": (0, 1), "+y": (0, -1), "-x": (-1, 0), "+x": (1, 0)}  # pixel step per FRONT side
FRONT_COLOUR: RGB = (200, 0, 0)
GRID_MAJOR_EVERY_STUDS = 5  # axis numbers and darker grid lines every 5 studs
GRID_MAJOR_COLOUR: RGB = (185, 185, 185)
GRID_MINOR_COLOUR: RGB = (230, 230, 230)
AXIS_NUMBER_COLOUR: RGB = (90, 90, 90)
OUTLINE_NEW: tuple[RGB, int] = ((0, 0, 0), 3)  # outline colour and width of a piece placed now
OUTLINE_CONTINUING: tuple[RGB, int] = ((130, 130, 130), 1)  # ... of a brick continuing from below
HOLD_OUTLINE_WIDTH_PX = 4


def _draw_front_arrow(draw: ImageDraw.ImageDraw, image_width: int, front: str) -> None:
    """Arrow marked FRONT in the title row, top right (y points up in the map)."""
    dx, dy = FRONT_ARROW_STEPS[front]
    cx, cy = image_width - MAP_MARGIN_PX - 70, 22
    draw.line([(cx - 12 * dx, cy - 12 * dy), (cx + 12 * dx, cy + 12 * dy)], fill=FRONT_COLOUR, width=4)
    tip = (cx + 12 * dx, cy + 12 * dy)
    draw.polygon(
        [
            tip,
            (tip[0] - 8 * dx - 6 * dy, tip[1] - 8 * dy + 6 * dx),
            (tip[0] - 8 * dx + 6 * dy, tip[1] - 8 * dy - 6 * dx),
        ],
        fill=FRONT_COLOUR,
    )
    draw.text((cx + 22, 8), "FRONT", fill=FRONT_COLOUR, font=font(22))


def _draw_stud_grid(draw: ImageDraw.ImageDraw, frame: "MapFrame", region: MapRegion) -> None:
    """Grid lines per stud (darker every 5) and the axis numbers, counted from the plate corner."""
    gx0, gy0, cell = frame.gx0, frame.gy0, region.cell
    w, h = region.width_studs, region.depth_studs
    small = font(max(10, cell // 3))
    for i in range(w + 1):
        major = (region.x0 + i - region.plate_x0) % GRID_MAJOR_EVERY_STUDS == 0
        draw.line(
            [(gx0 + i * cell, gy0), (gx0 + i * cell, gy0 + h * cell)],
            fill=GRID_MAJOR_COLOUR if major else GRID_MINOR_COLOUR,
        )
        if major and i < w:
            number = str(region.x0 + i - region.plate_x0)
            draw.text((gx0 + i * cell + 2, gy0 + h * cell + 4), number, fill=AXIS_NUMBER_COLOUR, font=small)
    for j in range(h + 1):
        major = (region.y0 + j - region.plate_y0) % GRID_MAJOR_EVERY_STUDS == 0
        yy = gy0 + (h - j) * cell
        draw.line([(gx0, yy), (gx0 + w * cell, yy)], fill=GRID_MAJOR_COLOUR if major else GRID_MINOR_COLOUR)
        if major and j < h:
            draw.text((4, yy - cell + 4), str(region.y0 + j - region.plate_y0), fill=AXIS_NUMBER_COLOUR, font=small)


def _height_suffix(piece: PlacedBrick) -> str:
    """'B' for a brick (3 plates high), '' for a plate."""
    return "B" if piece.h_plates == BRICK_HEIGHT_PLATES else ""


@dataclass(frozen=True)
class _PieceDrawer:
    """Draws the pieces of one layer map."""

    draw: ImageDraw.ImageDraw
    frame: "MapFrame"
    label_font: Font
    marks: MapMarks

    def draw_piece(self, piece: PlacedBrick, rgb: RGB) -> None:
        """Sub-assembly piece, L piece or rectangle, with its size label and hold mark."""
        if id(piece) in self.marks.subassembly_marks:
            subassembly_mark = self.marks.subassembly_marks[id(piece)]
            outline = self.frame.piece_outline(piece)
            draw_subassembly_piece(self.draw, outline, rgb, subassembly_mark, self.label_font)
            return
        is_continuing = self.marks.is_continuing(piece)
        if is_continuing:  # continuing tall brick: fade toward white
            rgb = faded_toward_white(rgb, CONTINUING_FADE)
        frame, cell = self.frame, self.frame.cell
        px0 = frame.gx0 + (piece.x - frame.x0) * cell
        py0 = frame.gy0 + (frame.depth_studs - (piece.y - frame.y0) - piece.d) * cell  # y up, like the old maps
        box = (px0, py0, px0 + piece.w * cell, py0 + piece.d * cell)
        mark = None if is_continuing else hanging_mark(id(piece), self.marks.floating)
        if getattr(piece, "notch", -1) >= 0:  # L piece (corner plate / brick, 3 studs)
            self._draw_l_piece(piece, box, rgb, mark)
        else:
            self._draw_rectangle(piece, box, rgb, mark)

    def _outline_style(self, piece: PlacedBrick) -> tuple[RGB, int]:
        return OUTLINE_CONTINUING if self.marks.is_continuing(piece) else OUTLINE_NEW

    def _draw_hold_symbol(self, px0: int, py0: int, mark: tuple[RGB, str]) -> None:
        self.draw.text((px0 + 4, py0 + 1), mark[1], fill=mark[0], font=font(max(12, int(self.frame.cell * 0.5))))

    def _draw_l_piece(
        self, piece: PlacedBrick, box: tuple[int, int, int, int], rgb: RGB, mark: tuple[RGB, str] | None
    ) -> None:
        """L outline, its label centred on the 3 studs."""
        px0, py0, px1, py1 = box
        cell = self.frame.cell
        pts = _l_outline(px0, py0, px1, py1, piece.notch)
        outline_colour, outline_width = self._outline_style(piece)
        self.draw.polygon(pts, fill=rgb)
        self.draw.line([*pts, pts[0]], fill=outline_colour, width=outline_width, joint="curve")
        if mark:
            self.draw.line([*pts, pts[0]], fill=mark[0], width=HOLD_OUTLINE_WIDTH_PX, joint="curve")
            self._draw_hold_symbol(px0, py0, mark)
        if self.marks.is_continuing(piece):
            return
        label = "L" + _height_suffix(piece)
        tb = self.draw.textbbox((0, 0), label, font=self.label_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        # centre of the 3 cells = box centre pushed away from the notch by a quarter cell
        sx = 1 if (piece.notch & 1) == 0 else -1
        sy = -1 if (piece.notch >> 1) == 0 else 1  # pixel y points down, notch bit 1 = high y = top
        cx, cy = (px0 + px1) / 2 + sx * cell * 0.25, (py0 + py1) / 2 + sy * cell * 0.25
        self.draw.text((cx - tw / 2, cy - th / 2 - tb[1]), label, fill=_text_color(rgb), font=self.label_font)

    def _draw_rectangle(
        self, piece: PlacedBrick, box: tuple[int, int, int, int], rgb: RGB, mark: tuple[RGB, str] | None
    ) -> None:
        """Rectangle with its size label ('2x4', 'B' for bricks; length only when that does not fit)."""
        px0, py0, px1, py1 = box
        cell = self.frame.cell
        outline_colour, outline_width = self._outline_style(piece)
        self.draw.rectangle([px0 + 1, py0 + 1, px1 - 1, py1 - 1], fill=rgb, outline=outline_colour, width=outline_width)
        if mark:
            self.draw.rectangle([px0 + 1, py0 + 1, px1 - 1, py1 - 1], outline=mark[0], width=HOLD_OUTLINE_WIDTH_PX)
            self._draw_hold_symbol(px0, py0, mark)
        if self.marks.is_continuing(piece):
            return
        label = f"{piece.w}x{piece.d}" + _height_suffix(piece)
        tb = self.draw.textbbox((0, 0), label, font=self.label_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        if tw > piece.w * cell - 4:  # 1-wide pieces: length only ("" for 1x1)
            length = max(piece.w, piece.d)
            label = (str(length) if length > 1 else "") + _height_suffix(piece)
            tb = self.draw.textbbox((0, 0), label, font=self.label_font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        self.draw.text(
            ((px0 + px1 - tw) / 2, (py0 + py1 - th) / 2 - tb[1]), label, fill=_text_color(rgb), font=self.label_font
        )


def layer_map(
    bricks: Sequence[PlacedBrick],
    region: MapRegion,
    title: str,
    color_of: ColourOf,
    marks: MapMarks | None = None,
) -> Image.Image:
    """Top-down map of the region (see MapRegion), title on top. bricks: the pieces to draw; color_of(id) ->
    (r,g,b). See MapMarks for faded / hold / sub-assembly pieces, the FRONT arrow and the layer below."""
    marks = marks if marks is not None else MapMarks()
    cell, h, w = region.cell, region.depth_studs, region.width_studs
    title_h, m = MAP_TITLE_ROW_PX, MAP_MARGIN_PX
    image_width = w * cell + 2 * m
    has_hold_marks = any(
        id(piece) in marks.floating and id(piece) not in marks.subassembly_marks and not marks.is_continuing(piece)
        for piece in bricks
    )
    entries = legend_entries(has_hold_marks=has_hold_marks, has_subassembly_pieces=bool(marks.subassembly_marks))
    has_legend = marks.below_bricks is not None and marks.show_legend
    legend_h = legend_height(image_width - 2 * m, cell, entries) if has_legend else 0
    img = Image.new("RGB", (image_width, h * cell + 2 * m + title_h + legend_h), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((m, 6), title, fill=(0, 0, 0), font=font(26))
    frame = layer_map_frame(h, w, cell, region.x0, region.y0)
    if marks.front in FRONT_ARROW_STEPS:
        _draw_front_arrow(draw, img.width, marks.front)
    _draw_stud_grid(draw, frame, region)
    if marks.below_bricks is not None:
        img = draw_layer_below(img, marks.below_bricks, bricks, frame, color_of)
        if has_legend:
            draw_map_legend(img, (m, frame.gy0 + h * cell + m), cell, image_width - 2 * m, entries)
        draw = ImageDraw.Draw(img)
    drawer = _PieceDrawer(draw, frame, font(max(12, int(cell * 0.5))), marks)
    for piece in bricks:
        drawer.draw_piece(piece, color_of(piece.color))
    return img


# ---------------------------------------------------------------------------------------------------------------
# Layer below (ghost) on the layer maps
# ---------------------------------------------------------------------------------------------------------------
LEGEND_ROW_PX = 30  # extra image height for the legend row under the axis numbers
GHOST_HATCH_SPACING_PX = 9  # distance between the diagonal hatch lines of the layer below
GHOST_HATCH_WIDTH_PX = 2  # thickness of one hatch line
GHOST_OUTLINE_WIDTH_PX = 2  # dashed outline thickness of a layer-below piece
GHOST_DASH_PX = (6, 4)  # dashed outline of a piece of the layer below: dash length, gap length
GHOST_LIGHT_LUMINANCE = 200  # ghost colours lighter than this are darkened, so white / yellow show on white
GHOST_LIGHT_DARKEN = 0.75  # factor for those light ghost colours
GHOST_CLEAR_PAD_PX = 2  # ghost is also cleared this far around covered studs (no dash bits at the edges)
CONTINUING_FADE = 0.6  # continuing tall bricks are faded this far toward white (as drawn above)


@dataclass(frozen=True)
class MapFrame:
    """Where the stud grid sits on a layer map: pixel corner of the map's low-x / HIGH-y stud (gx0, gy0), cell
    size in pixels, the grid region shown (x0, y0, width_studs, depth_studs). y points up in the map."""

    gx0: int
    gy0: int
    cell: int
    x0: int
    y0: int
    width_studs: int
    depth_studs: int

    def cell_box(self, grid_x: int, grid_y: int) -> tuple[int, int, int, int]:
        """Pixel box (left, top, right, bottom; right / bottom exclusive) of one stud cell."""
        left = self.gx0 + (grid_x - self.x0) * self.cell
        top = self.gy0 + (self.depth_studs - (grid_y - self.y0) - 1) * self.cell
        return left, top, left + self.cell, top + self.cell

    def piece_outline(self, piece: PlacedBrick) -> list[Point]:
        """Pixel outline points of a piece (rectangle or L)."""
        left = self.gx0 + (piece.x - self.x0) * self.cell
        top = self.gy0 + (self.depth_studs - (piece.y - self.y0) - piece.d) * self.cell
        right, bottom = left + piece.w * self.cell, top + piece.d * self.cell
        if getattr(piece, "notch", -1) >= 0:
            return _l_outline(left, top, right, bottom, piece.notch)
        return [(left, top), (right, top), (right, bottom), (left, bottom)]


def layer_map_frame(depth_studs: int, width_studs: int, cell: int, x0: int, y0: int) -> MapFrame:
    """MapFrame of a layer map drawn by layer_map with these arguments."""
    return MapFrame(MAP_MARGIN_PX, MAP_MARGIN_PX + MAP_TITLE_ROW_PX, cell, x0, y0, width_studs, depth_studs)


def studs_of_pieces(pieces: Iterable[PlacedBrick]) -> set[tuple[int, int]]:
    """{(grid_x, grid_y)} studs covered by the pieces (the 3 studs of an L piece)."""
    studs: set[tuple[int, int]] = set()
    for piece in pieces:
        studs.update(piece_cells(piece))
    return studs


def _draw_dashed_closed_line(
    draw: ImageDraw.ImageDraw, points: Sequence[Point], fill: RGB | tuple[int, int, int, int], width: int
) -> None:
    """Closed polyline through points, drawn as dashes of GHOST_DASH_PX (PIL has no dashed lines)."""
    dash_px, gap_px = GHOST_DASH_PX
    for start, end in zip(points, [*points[1:], *points[:1]], strict=True):
        length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        if length == 0:
            continue
        unit_x, unit_y = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        position = 0.0
        while position < length:
            stop = min(position + dash_px, length)
            draw.line(
                [
                    (start[0] + unit_x * position, start[1] + unit_y * position),
                    (start[0] + unit_x * stop, start[1] + unit_y * stop),
                ],
                fill=fill,
                width=width,
            )
            position = stop + gap_px


def ghost_colour(rgb: RGB) -> RGB:
    """Colour of the hatch / dashed outline of a layer-below piece: its own colour, darkened when it is so light
    that it would vanish on the white map."""
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    red, green, blue = rgb
    if luminance <= GHOST_LIGHT_LUMINANCE:
        return int(red), int(green), int(blue)
    return int(red * GHOST_LIGHT_DARKEN), int(green * GHOST_LIGHT_DARKEN), int(blue * GHOST_LIGHT_DARKEN)


def _hatch_mask(size: tuple[int, int]) -> NDArray[np.bool_]:
    """bool [height, width]: True on the diagonal hatch lines of the whole image."""
    width_px, height_px = size
    rows, cols = np.mgrid[0:height_px, 0:width_px]
    return np.asarray((rows + cols) % GHOST_HATCH_SPACING_PX < GHOST_HATCH_WIDTH_PX, dtype=np.bool_)


def draw_layer_below(
    img: Image.Image,
    below_pieces: Sequence[PlacedBrick],
    current_pieces: Iterable[PlacedBrick],
    frame: MapFrame,
    color_of: ColourOf,
) -> Image.Image:
    """Draw the pieces of the layer below as a ghost: diagonal hatching + dashed outline in the piece's colour,
    no fill, no label, only on studs the current layer leaves uncovered (covered studs are hidden under the new
    pieces anyway). Returns the new image; the grid lines stay visible between the hatch lines."""
    hatch = _hatch_mask(img.size)
    overlay_pixels = np.zeros((img.size[1], img.size[0], 4), np.uint8)
    for piece in below_pieces:
        piece_area = Image.new("L", img.size, 0)
        ImageDraw.Draw(piece_area).polygon(frame.piece_outline(piece), fill=255)
        overlay_pixels[(np.asarray(piece_area) > 0) & hatch] = (*ghost_colour(color_of(piece.color)), 255)
    overlay = Image.fromarray(overlay_pixels, "RGBA")
    overlay_draw = ImageDraw.Draw(overlay)
    for piece in below_pieces:
        _draw_dashed_closed_line(
            overlay_draw,
            frame.piece_outline(piece),
            (*ghost_colour(color_of(piece.color)), 255),
            GHOST_OUTLINE_WIDTH_PX,
        )
    for grid_x, grid_y in studs_of_pieces(current_pieces):
        left, top, right, bottom = frame.cell_box(grid_x, grid_y)
        overlay_draw.rectangle(
            [
                left - GHOST_CLEAR_PAD_PX,
                top - GHOST_CLEAR_PAD_PX,
                right - 1 + GHOST_CLEAR_PAD_PX,
                bottom - 1 + GHOST_CLEAR_PAD_PX,
            ],
            fill=(0, 0, 0, 0),
        )
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


LEGEND_ENTRIES = (("solid", "place now"), ("faded", "brick from below continues"), ("hatched", "layer below"))
HOLD_LEGEND_ENTRY = ("hold", "nothing under it: hold it")
SUBASSEMBLY_LEGEND_ENTRY = ("subassembly", "comes with a sub-assembly (built before)")


def legend_entries(*, has_hold_marks: bool = False, has_subassembly_pieces: bool = False) -> tuple[LegendEntry, ...]:
    """Legend entries of a map: the hold and sub-assembly entries only on maps that show such pieces."""
    return (
        LEGEND_ENTRIES
        + ((HOLD_LEGEND_ENTRY,) if has_hold_marks else ())
        + ((SUBASSEMBLY_LEGEND_ENTRY,) if has_subassembly_pieces else ())
    )


def faded_toward_white(rgb: RGB, fade: float) -> RGB:
    """Colour moved `fade` (0..1) of the way to white."""
    red, green, blue = rgb
    return int(red + (255 - red) * fade), int(green + (255 - green) * fade), int(blue + (255 - blue) * fade)


def draw_subassembly_piece(
    draw: ImageDraw.ImageDraw, outline_points: list[Point], rgb: RGB, mark: tuple[str, RGB], label_font: Font
) -> None:
    """A piece that comes with a sub-assembly placed as a unit: faded fill, outline and the sub-assembly's name in
    its colour; mark = (name, outline colour)."""
    name, outline_rgb = mark
    draw.polygon(outline_points, fill=faded_toward_white(rgb, SUBASSEMBLY_FADE))
    draw.line([*outline_points, outline_points[0]], fill=outline_rgb, width=SUBASSEMBLY_OUTLINE_WIDTH_PX, joint="curve")
    xs, ys = [point[0] for point in outline_points], [point[1] for point in outline_points]
    text_box = draw.textbbox((0, 0), name, font=label_font)
    text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
    draw.text(
        ((min(xs) + max(xs) - text_width) / 2, (min(ys) + max(ys) - text_height) / 2 - text_box[1]),
        name,
        fill=outline_rgb,
        font=label_font,
    )


def _legend_positions(
    available_width_px: int, cell: int, entries: tuple[LegendEntry, ...] = LEGEND_ENTRIES
) -> list[tuple[str, str, int, int]]:
    """[(style, text, left offset, row)]: legend entries laid out left to right, wrapped to new rows when the map
    is narrow."""
    swatch_px, legend_font = max(14, cell // 2), font(max(11, cell // 3))
    positions: list[tuple[str, str, int, int]] = []
    left, row = 0, 0
    for style, text in entries:
        entry_width = swatch_px + 6 + int(legend_font.getlength(text))
        if left and left + entry_width > available_width_px:
            left, row = 0, row + 1
        positions.append((style, text, left, row))
        left += entry_width + 24
    return positions


def legend_height(available_width_px: int, cell: int, entries: tuple[LegendEntry, ...] = LEGEND_ENTRIES) -> int:
    """Pixel height of the legend for a map whose grid is available_width_px wide."""
    return LEGEND_ROW_PX * (1 + max(row for *_, row in _legend_positions(available_width_px, cell, entries)))


def _draw_legend_swatch(img: Image.Image, draw: ImageDraw.ImageDraw, box: list[int], style: str) -> None:
    """One legend swatch in the style of the map (see LEGEND_ENTRIES)."""
    example_colour: RGB = (40, 40, 40)
    if style == "solid":
        draw.rectangle(box, fill=example_colour, outline=(0, 0, 0), width=3)
    elif style == "faded":
        faded = faded_toward_white(example_colour, CONTINUING_FADE)
        draw.rectangle(box, fill=faded, outline=(130, 130, 130), width=1)
    elif style == "hatched":
        size = box[2] - box[0]
        swatch = np.full((size, size, 3), 255, np.uint8)
        swatch[_hatch_mask((size, size))] = example_colour
        img.paste(Image.fromarray(swatch, "RGB"), (box[0], box[1]))
        outline: list[Point] = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
        _draw_dashed_closed_line(draw, outline, example_colour, 2)
    elif style == "subassembly":
        square: list[Point] = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
        mark = ("S", SUBASSEMBLY_OUTLINE_COLOURS[0])
        draw_subassembly_piece(draw, square, example_colour, mark, font(max(11, (box[3] - box[1]) - 4)))
    else:
        colour, symbol = HOLD_MARK
        draw.rectangle(box, fill=example_colour, outline=colour, width=3)
        draw.text((box[0] + 4, box[1]), symbol, fill=(255, 255, 255), font=font(max(11, (box[3] - box[1]) - 2)))


def draw_map_legend(
    img: Image.Image,
    top_left: tuple[int, int],
    cell: int,
    available_width_px: int,
    entries: tuple[LegendEntry, ...] = LEGEND_ENTRIES,
) -> None:
    """Legend (drawn into img), wrapped to the map width: solid = place now, faded = brick continues from below,
    hatched = layer below (+ red ! = hold it, coloured outline = comes with a sub-assembly, on maps that have them)."""
    draw = ImageDraw.Draw(img)
    swatch_px, legend_font = max(14, cell // 2), font(max(11, cell // 3))
    for style, text, left_offset, row in _legend_positions(available_width_px, cell, entries):
        left, top = top_left[0] + left_offset, top_left[1] + row * LEGEND_ROW_PX
        _draw_legend_swatch(img, draw, [left, top, left + swatch_px, top + swatch_px], style)
        draw.text((left + swatch_px + 6, top), text, fill=(60, 60, 60), font=legend_font)
