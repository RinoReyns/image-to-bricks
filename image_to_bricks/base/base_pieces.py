"""The stand as pieces: hand-built layout, layered packer, plate / brick packer, re-tiling of loose pieces,
spare-colour fill."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.analysis import stability
from image_to_bricks.analysis.stability import analyse_graph, build_contacts
from image_to_bricks.base import base_fill, base_layout, base_packer
from image_to_bricks.consts import BRICK_HEIGHT_PLATES, COLOR_NAMES
from image_to_bricks.model import LegoModel, PlacedBrick
from image_to_bricks.options import BaseOptions
from image_to_bricks.packing.layer_packer import (
    LayerBelow,
    PackedRect,
    PackStyle,
    StockUse,
    brick_band,
    pack_layer_staggered,
)
from image_to_bricks.stock import PLATE_HEIGHT_PLATES, Stock

type Footprint = tuple[int, int, int, int]  # (x, y, width, depth) in studs

LAYERS_PER_BRICK_BAND = BRICK_HEIGHT_PLATES  # plate layers of one band packed as bricks
BOND_PHASE_PERIOD = 4  # brick-bond phases of the stand repeat every 4 studs (2 per plate layer)
MAX_RECONNECT_ROUNDS = 30  # re-tiling rounds for loose base plates (each round fixes at most one)
MAX_RETILE_NEIGHBOURS = 4  # largest touching plates tried as re-tiling partner of a loose plate


class BasePiece(NamedTuple):
    """One stand piece in the model grid; colour None = it covers only hidden cells (packed without stock)."""

    x: int
    y: int
    z: int
    w: int
    d: int
    colour: int | None
    height_plates: int


@dataclass(frozen=True)
class StandCells:
    """What the stand is packed into (model grid): grid [z, y, x] bool cells, colours [z, y, x] (WILD = any),
    owner [z, y, x] (index into figure_rects of the piece covering a cell, -1 = none), figure_rects [(x, y, w, d)]
    of the figure's pieces, figure_bricks (the figure standing on the stand) and layer_count (base layers)."""

    grid: NDArray[np.bool_]
    colours: NDArray[Any]
    owner: NDArray[Any]
    figure_rects: Sequence[Footprint]
    figure_bricks: Sequence[PlacedBrick]
    layer_count: int

    def find_loose(self, pieces: Sequence[BasePiece]) -> list[int]:
        """Indices of the stand pieces that are not connected to the rest (see loose_base_pieces)."""
        base = [PlacedBrick(p.x, p.y, p.z, p.w, p.d, p.height_plates, 0) for p in pieces]
        return loose_base_pieces(self.grid, self.figure_bricks, base)


def loose_base_pieces(
    grid: NDArray[np.bool_], figure_bricks: Sequence[PlacedBrick], base: Sequence[PlacedBrick]
) -> list[int]:
    """Indices (into `base`) of base pieces not in the largest stud-connected group of base pieces, counting the
    figure standing on them (so plates joined only through the feet still count as connected)."""
    figure_count = len(figure_bricks)
    component, _, _ = analyse_graph(build_contacts(LegoModel(grid=grid, bricks=list(figure_bricks) + list(base))))
    size_by_component: dict[int, int] = {}
    for i in range(len(base)):
        size_by_component[component[figure_count + i]] = size_by_component.get(component[figure_count + i], 0) + 1
    main = max(size_by_component, key=lambda c: size_by_component[c])
    return [i for i in range(len(base)) if component[figure_count + i] != main]


def _touching(first: BasePiece, second: BasePiece) -> bool:
    """Two pieces in the same layer share part of an edge."""
    if first.x + first.w == second.x or second.x + second.w == first.x:
        return min(first.y + first.d, second.y + second.d) > max(first.y, second.y)
    if first.y + first.d == second.y or second.y + second.d == first.y:
        return min(first.x + first.w, second.x + second.w) > max(first.x, second.x)
    return False


class _RetileVariant(NamedTuple):
    """One way to re-tile a region: flipped scan, columns first (transposed), brick-bond phase."""

    is_flipped: bool
    is_transposed: bool
    phase: tuple[int, int] | None


RETILE_VARIANTS = tuple(
    _RetileVariant(is_flipped, is_transposed, phase)
    for is_transposed in (False, True)
    for is_flipped in (False, True)
    for phase in (None, (0, 0), (2, 2), (0, 2), (2, 0))
)


def _retile(
    region: NDArray[np.bool_], colours: NDArray[Any], variant: _RetileVariant, base_colour: int, stock: Stock | None
) -> list[PackedRect]:
    """Re-pack the cells of `region` (bool [h,w]); returns [(x, y, w, d, colour)]. Cells of any colour take
    base_colour. stock: a fork whose used counts already include everything outside `region`."""
    style = PackStyle(is_flipped=variant.is_flipped, phase=variant.phase)
    stock_use = StockUse(stock, fallback=base_colour) if stock is not None else None
    if variant.is_transposed:
        packed = pack_layer_staggered(region.T.copy(), colours.T, style=style, stock_use=stock_use)
        return [(y, x, d, w, c) for (x, y, w, d, c) in packed]
    return pack_layer_staggered(region.copy(), colours, style=style, stock_use=stock_use)


def place_hand_built_base(
    grid: NDArray[np.bool_], base_layers: int, base: BaseOptions, stock: Stock | None = None
) -> tuple[list[PlacedBrick], NDArray[Any]]:
    """--base-layout: places the hand-built base where the figure tips least (base_layout.py), prints where and
    which seams are not tied. Returns (base plates in the model grid, grid whose base layers hold the layout)."""
    layers = base_layout.layer_count(base.layout)
    if layers != base_layers:
        raise SystemExit(
            f"--base-layout {base.layout_path} has {layers} layer(s) but --base is {base_layers}: use --base {layers}"
        )

    def tip_angle(trial_grid: NDArray[np.bool_]) -> float:
        report = stability.stability_check(LegoModel(grid=trial_grid, bricks=[], base_layers=base_layers))
        return float(report["tip_angle"])

    placement = base_layout.choose_placement(grid, base_layers, base.layout, tip_angle)
    if placement is None:
        width, depth = base_layout.layout_size(base.layout)
        raise SystemExit(
            f"--base-layout {base.layout_path} ({width} x {depth} studs) does not fit under the figure "
            f"(grid {grid.shape[2]} x {grid.shape[1]}, the whole first figure layer must stand on "
            f"top-layer plates): raise --base-margin (now {base.margin_studs}) or check the layout"
        )
    if stock is not None:
        base_layout.register_special_pieces(stock, base.layout)
    plates = base_layout.place_layout(base.layout, placement.grid_x, placement.grid_y)
    feet_y, feet_x = np.nonzero(grid[base_layers])
    width, depth = base_layout.layout_size(base.layout)
    print(
        f"Base layout: {base.layout_path}: {len(plates)} plates in {layers} layers, {width} x {depth} studs, used "
        f"exactly; the first figure layer stands on layout x {feet_x.min() - placement.grid_x}-"
        f"{feet_x.max() - placement.grid_x}, y {feet_y.min() - placement.grid_y}-{feet_y.max() - placement.grid_y}; "
        f"tip angle {placement.tip_angle_degrees:.1f} deg"
    )
    problems = base_layout.untied_seams(base.layout)
    for problem in problems:
        print(f"  WARNING base layout: {problem}")
    print(
        "  base layout seams: "
        + (
            "every seam tied by the layer below"
            if not problems
            else f"{len(problems)} place(s) not tied (see WARNING lines)"
        )
    )
    return plates, base_layout.base_grid(grid, plates, base_layers)


def pack_layered_base_with_report(
    cells: StandCells, base: BaseOptions, stock: Stock | None = None
) -> tuple[list[PlacedBrick], NDArray[Any]]:
    """Base from big plates (--base-hidden-layers, see base_packer): prints the summary line and WARNINGs for
    loose base plates and untied seams. Returns (base plates, not yet taken from stock; new grid whose base
    layers hold only the cells the plates cover - sparse layers leave holes)."""
    if base.layered is None:
        raise ValueError("pack_layered_base_with_report needs base.layered (--base-hidden-layers)")
    layer_count = cells.layer_count
    base_cells = base_packer.BaseCells(cells.grid, cells.colours, cells.owner[layer_count], layer_count)
    pieces, reports = base_packer.pack_layered_base(base_cells, stock, base.layered, base.colour_id)
    print("  " + base_packer.summary(pieces, reports, base.layered.hidden_layers))
    open_ties = sum(r.open_requirements for r in reports)
    if open_ties:
        print(
            f"  WARNING: {open_ties} seam tie(s) / feet could not be placed in the sparse layer(s); "
            f"try a smaller --base-tie-every or --base-sparse-layers 0"
        )
    grid = base_packer.base_cells(cells.grid, pieces, layer_count)
    loose = loose_base_pieces(grid, cells.figure_bricks, pieces)
    if loose:
        print(
            f"  WARNING: {len(loose)} base piece(s) are not connected to the rest; "
            f"try another --base-max-plate or --base-hidden-layers"
        )
    return pieces, grid


class _PackVariant(NamedTuple):
    """One way to pack the stand: thin-strip guard, layer parity shift, brick-bond phase, columns first."""

    avoids_thin_strips: bool
    shift: int
    uses_bond_phase: bool
    is_transposed: bool


PACK_VARIANTS = tuple(
    _PackVariant(avoids_thin_strips, shift, uses_bond_phase, is_transposed)
    for is_transposed in (False, True)
    for uses_bond_phase in (True, False)
    for avoids_thin_strips in (True, False)
    for shift in (0, 1)
)
SINGLE_LAYER_VARIANT = _PackVariant(avoids_thin_strips=False, shift=0, uses_bond_phase=False, is_transposed=False)


@dataclass
class _VariantPacking:
    """The stand packed in one _PackVariant. A transposed variant packs with x and y swapped (scan columns first);
    `pieces` are always in the model grid. rects / owner grow with every placed piece (index = piece in rects)."""

    grid: NDArray[np.bool_]
    colours: NDArray[Any]
    owner: NDArray[Any]
    rects: list[Footprint]
    variant: _PackVariant
    base_colour: int
    stock: Stock | None
    pieces: list[BasePiece] = field(default_factory=list)

    @classmethod
    def start(
        cls, cells: StandCells, variant: _PackVariant, base_colour: int, stock: Stock | None
    ) -> "_VariantPacking":
        """Empty packing of the variant; stock is forked (None without --stock)."""
        fork = stock.fork() if stock is not None else None
        if variant.is_transposed:
            return cls(
                cells.grid.transpose(0, 2, 1),
                cells.colours.transpose(0, 2, 1),
                cells.owner.transpose(0, 2, 1).copy(),
                [(y, x, d, w) for (x, y, w, d) in cells.figure_rects],
                variant,
                base_colour,
                fork,
            )
        return cls(cells.grid, cells.colours, cells.owner.copy(), list(cells.figure_rects), variant, base_colour, fork)

    def style(self, flip_index: int, phase_layer: int) -> PackStyle:
        """Scan direction alternates with flip_index; the bond phase follows phase_layer."""
        shift = self.variant.shift
        phase = ((2 * (phase_layer + shift)) % BOND_PHASE_PERIOD,) * 2 if self.variant.uses_bond_phase else None
        return PackStyle(bool((flip_index + shift) % 2), self.variant.avoids_thin_strips, phase)

    def add(self, rect: PackedRect, z: int, height_plates: int) -> None:
        """Record a packed piece (packing coordinates) in rects / owner and, in the model grid, in pieces."""
        x, y, w, d, colour = rect
        self.rects.append((x, y, w, d))
        self.owner[z : z + height_plates, y : y + d, x : x + w] = len(self.rects) - 1
        if self.variant.is_transposed:
            self.pieces.append(BasePiece(y, x, z, d, w, colour, height_plates))
        else:
            self.pieces.append(BasePiece(x, y, z, w, d, colour, height_plates))

    def pack_plates(self, z: int, mask: NDArray[np.bool_]) -> None:
        """Plates over the mask cells of layer z, bridging the pieces of layer z + 1."""
        stock_use = StockUse(self.stock, fallback=self.base_colour) if self.stock is not None else None
        below = LayerBelow(self.owner[z + 1], self.rects)
        for rect in pack_layer_staggered(mask, self.colours[z], below, self.style(z, z), stock_use):
            self.add(rect, z, PLATE_HEIGHT_PLATES)

    def pack_brick_band(self, band: int) -> None:
        """Band of 3 layers: bricks over the cells that agree in colour in all 3 layers, plates for the rest."""
        z0 = LAYERS_PER_BRICK_BAND * band
        band_layers = range(z0, z0 + LAYERS_PER_BRICK_BAND)
        same, band_colours = brick_band(self.grid, self.colours, z0)
        left = {z: self.grid[z].copy() for z in band_layers}
        stock_use = (
            StockUse(self.stock, BRICK_HEIGHT_PLATES, self.base_colour, "exhausted") if self.stock is not None else None
        )
        below = LayerBelow(self.owner[z0 + LAYERS_PER_BRICK_BAND], self.rects)
        for rect in pack_layer_staggered(same.copy(), band_colours, below, self.style(band, z0), stock_use):
            self.add(rect, z0, BRICK_HEIGHT_PLATES)
            x, y, w, d, _ = rect
            for layer_cells in left.values():
                layer_cells[y : y + d, x : x + w] = False
        for z in reversed(band_layers):
            self.pack_plates(z, left[z])

    def pack(self, layer_count: int, brick_bands: int) -> None:
        """Plate layers from the top down to the brick bands, then the bands, top band first."""
        for z in range(layer_count - 1, LAYERS_PER_BRICK_BAND * brick_bands - 1, -1):
            self.pack_plates(z, self.grid[z].copy())
        for band in range(brick_bands - 1, -1, -1):
            self.pack_brick_band(band)


def _stock_without(
    stock: Stock | None, pieces: Sequence[BasePiece], excluded: tuple[int, int], base_colour: int
) -> Stock | None:
    """Stock fork whose ledger holds every piece except the excluded two (None without stock)."""
    if stock is None:
        return None
    fork = stock.fork()
    for index, piece in enumerate(pieces):
        if index not in excluded:
            colour = piece.colour if piece.colour is not None else base_colour
            fork.take(piece.height_plates, piece.w, piece.d, colour)
    return fork


@dataclass(frozen=True)
class _Reconnection:
    """Re-tiling of loose stand plates together with a touching plate."""

    cells: StandCells
    base_colour: int
    stock: Stock | None

    def retile_pair(self, pieces: list[BasePiece], pair: tuple[int, int], loose_count: int) -> list[BasePiece] | None:
        """Pieces with the pair re-tiled by the first RETILE_VARIANTS entry that leaves fewer loose pieces."""
        region = np.zeros(self.cells.grid.shape[1:], bool)
        for index in pair:
            piece = pieces[index]
            region[piece.y : piece.y + piece.d, piece.x : piece.x + piece.w] = True
        z = pieces[pair[0]].z
        fork = _stock_without(self.stock, pieces, pair, self.base_colour)
        kept = [piece for index, piece in enumerate(pieces) if index not in pair]
        for variant in RETILE_VARIANTS:
            retiled = _retile(
                region, self.cells.colours[z], variant, self.base_colour, fork.fork() if fork is not None else None
            )
            candidate = kept + [BasePiece(x, y, z, w, d, c, PLATE_HEIGHT_PLATES) for (x, y, w, d, c) in retiled]
            if len(self.cells.find_loose(candidate)) < loose_count:
                return candidate
        return None

    def fix_one(self, pieces: list[BasePiece], loose_plates: Sequence[int]) -> list[BasePiece] | None:
        """First re-tiling (loose plate, one of its largest touching plates) that leaves fewer loose pieces."""
        loose_count = len(self.cells.find_loose(pieces))
        for loose_index in loose_plates:
            loose_piece = pieces[loose_index]
            neighbours = sorted(
                (
                    j
                    for j, piece in enumerate(pieces)
                    if j != loose_index
                    and piece.z == loose_piece.z
                    and piece.height_plates == PLATE_HEIGHT_PLATES
                    and _touching(loose_piece, piece)
                ),
                key=lambda j: -pieces[j].w * pieces[j].d,
            )
            for neighbour in neighbours[:MAX_RETILE_NEIGHBOURS]:
                fixed = self.retile_pair(pieces, (loose_index, neighbour), loose_count)
                if fixed is not None:
                    return fixed
        return None

    def run(self, pieces: list[BasePiece]) -> list[BasePiece]:
        """Up to MAX_RECONNECT_ROUNDS rounds, each fixing one loose plate; stops when none is left or none helps."""
        for _ in range(MAX_RECONNECT_ROUNDS):
            loose_plates = [i for i in self.cells.find_loose(pieces) if pieces[i].height_plates == PLATE_HEIGHT_PLATES]
            if not loose_plates:
                break
            fixed = self.fix_one(pieces, loose_plates)
            if fixed is None:
                break
            pieces = fixed
        return pieces


def pack_base_with_plates(
    cells: StandCells, base_colour: int, brick_bands: int = 0, stock: Stock | None = None
) -> list[BasePiece]:
    """Pack the cells.layer_count base layers (top first, bridging the figure's bottom pieces).
    brick_bands: the lowest bands of 3 layers are packed as 3-plate BRICKS first (cells that agree in
    colour in all 3 layers; the rest of the band as plates); the layers above stay plates and tie
    the bricks together (needs layer_count >= 3 * brick_bands + 1).
    Tries a few variants (scan parity/direction, thin-strip guard, brick-bond phase) and
    keeps the one with the fewest loose base pieces (not connected to the main assembly,
    counting the figure standing on it), then the fewest bought, then the fewest pieces. Remaining loose plates
    are fixed by re-tiling the area around them."""
    layer_count = cells.layer_count
    brick_bands = max(0, min(brick_bands, (layer_count - 1) // LAYERS_PER_BRICK_BAND))
    is_single_layer = layer_count <= 1
    best: tuple[tuple[int, int, int], list[BasePiece]] | None = None
    for variant in (SINGLE_LAYER_VARIANT,) if is_single_layer else PACK_VARIANTS:
        packing = _VariantPacking.start(cells, variant, base_colour, stock)
        packing.pack(layer_count, brick_bands)
        loose_count = 0 if is_single_layer else len(cells.find_loose(packing.pieces))
        bought = packing.stock.need_total() if packing.stock is not None else 0
        key = (loose_count, bought, len(packing.pieces))
        if best is None or key < best[0]:
            best = (key, packing.pieces)
    if best is None:
        raise ValueError("no stand packing variant")
    (loose_count, _, _), pieces = best
    if is_single_layer:
        print("  base of 1 plate: only the plates under the figure are held; use --base 2 or more")
        return pieces
    pieces = _Reconnection(cells, base_colour, stock).run(pieces)
    loose_count = len(cells.find_loose(pieces))
    if loose_count:
        print(
            f"  WARNING: {loose_count} base piece(s) are not connected to the rest; "
            f"try a different --base-margin / --base-shape"
        )
    return pieces


def fill_base_from_spare_pieces(model: LegoModel, base_layers: int, base: BaseOptions, stock: Stock) -> None:
    """--base-fill-colors: base pieces short in the stock are re-tiled with spare owned pieces (base_fill.py); one
    console line per changed layer, then a summary."""
    hidden_layers = base.layered.hidden_layers if base.layered is not None else base_layers - 1
    options = base_fill.BaseFillOptions(base.fill_colour_ids, hidden_layers, base_packer.BASE_BUY_SIZES)

    def count_loose(bricks: list[PlacedBrick]) -> int:
        figure = [b for b in bricks if b.z >= base_layers]
        return len(loose_base_pieces(model.grid, figure, [b for b in bricks if b.z < base_layers]))

    refills = base_fill.fill_base_from_spare(model.bricks, base_layers, stock, options, count_loose)
    for refill in refills:
        colours = Counter(COLOR_NAMES.get(p.color, p.color) for p in refill.added)
        print(
            f"  base layer {refill.layer + 1}: {len(refill.removed)} short piece(s) -> {len(refill.added)} "
            f"({', '.join(f'{name} {count}' for name, count in sorted(colours.items()))}); "
            f"pieces to buy {refill.to_buy_before} -> {refill.to_buy_after}"
        )
    saved = sum(r.to_buy_before - r.to_buy_after for r in refills)
    print(
        f"Base fill: {saved} fewer piece(s) to buy, base re-tiled with spare owned pieces in {len(refills)} "
        f"layer(s) (visible: {', '.join(COLOR_NAMES.get(c, str(c)) for c in base.fill_colour_ids)}; hidden: any)"
    )
