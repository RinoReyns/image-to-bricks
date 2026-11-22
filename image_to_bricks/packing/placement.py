"""Cells -> pieces for the whole model (optimize_brick_placement) and the passes after it."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.base import base_layout
from image_to_bricks.base.base_pieces import (
    StandCells,
    pack_base_with_plates,
    pack_layered_base_with_report,
    place_hand_built_base,
)
from image_to_bricks.consts import BRICK_HEIGHT_PLATES
from image_to_bricks.model import LegoModel, PlacedBrick
from image_to_bricks.options import BaseOptions, Box, BuildOptions, RepairOptions
from image_to_bricks.packing import anchor_floating, l_pieces, merge_plates, tie_repair
from image_to_bricks.packing.assembly import SIZES
from image_to_bricks.packing.layer_packer import (
    BRICK_BAND_PHASE_COUNT,
    WILD,
    LayerBelow,
    PackedRect,
    PackStyle,
    StockUse,
    brick_band,
    harmonize_bands,
    pack_layer_staggered,
)
from image_to_bricks.packing.retile_setting import RetileSetting, StabilityScore
from image_to_bricks.stock import PLATE_HEIGHT_PLATES, Stock

ANCHOR_SPLITS_BRICKS_MODE = 2  # --anchor-floating 2: bricks may be split to anchor a floating plate


@dataclass(frozen=True)
class PackingCells:
    """The cells handed to the packer, all [z, y, x]: grid (filled cells), colours (colour id per cell), visible
    (cells seen from outside, None = unknown), fill (cells hollowing removed: pieces may grow into them; None = no
    growth) and base_layers (plate layers of the stand at the bottom)."""

    grid: NDArray[np.bool_]
    colours: NDArray[Any]
    visible: NDArray[np.bool_] | None = None
    fill: NDArray[np.bool_] | None = None
    base_layers: int = 0


@dataclass(frozen=True)
class PassContext:
    """What the passes after packing work on: the model grid [z, y, x], the visible cells (None = unknown), the
    stand's layers, the smallest hanging group that counts as weak (RepairOptions.tie_min_hanging_pieces) and the
    stock (None without --stock)."""

    grid: NDArray[np.bool_]
    visible: NDArray[np.bool_] | None
    base_layers: int
    hang_min: int
    stock: Stock | None = None

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """(layers, depth, width) of the grid."""
        layer_count, depth, width = self.grid.shape
        return layer_count, depth, width

    def stability_score(self, trial_bricks: Sequence[PlacedBrick]) -> StabilityScore:
        """l_pieces.stability_score of the trial pieces in this grid."""
        return l_pieces.stability_score(trial_bricks, self.grid, self.base_layers, self.hang_min)

    def standard_sizes(self) -> list[tuple[int, int]]:
        """assembly.SIZES plus the larger sizes in the stock."""
        return SIZES if self.stock is None else self.stock.sizes_with(SIZES)


def _anchor_floating(bricks: Sequence[PlacedBrick], context: PassContext, mode: int) -> list[PlacedBrick]:
    """--anchor-floating: floating plates get re-tiled with a supported same-colour neighbour (anchor_floating.py);
    mode 1 = plates, 2 = also split bricks. Prints one line, returns the new pieces."""
    tallest_neighbour = BRICK_HEIGHT_PLATES if mode >= ANCHOR_SPLITS_BRICKS_MODE else PLATE_HEIGHT_PLATES
    count_before = len(bricks)
    setting = RetileSetting(context.grid_shape, context.standard_sizes(), context.stability_score, context.stock)
    new_bricks, result = anchor_floating.anchor_floating_pieces(list(bricks), setting, tallest_neighbour)
    print(
        f"Anchor floating: pieces with nothing under them {result.floating_before} -> {result.floating_after} "
        f"({result.retilings} re-tiling(s){', bricks split where needed' if tallest_neighbour > 1 else ''}); "
        f"pieces {count_before} -> {len(new_bricks)}"
    )
    return new_bricks


def _merge_plates(bricks: Sequence[PlacedBrick], context: PassContext) -> list[PlacedBrick]:
    """--merge-plates: groups of 2-3 neighbouring plates that fill a standard plate become one (merge_plates.py);
    the base is left alone. Prints one line; returns the new pieces."""
    sizes = sorted(set(context.standard_sizes()) | set(anchor_floating.ANCHOR_EXTRA_SIZES))
    single_studs_before = sum(1 for b in bricks if b.w * b.d == 1)
    setting = RetileSetting(context.grid_shape, sizes, context.stability_score, context.stock)
    new_bricks, result = merge_plates.merge_plates(
        list(bricks), setting, context.visible, first_layer=context.base_layers
    )
    print(
        f"Merge plates: {result.merges} group(s) of neighbouring plates merged into one plate; pieces "
        f"{result.pieces_before} -> {result.pieces_after}, 1x1 {single_studs_before} -> "
        f"{sum(1 for b in new_bricks if b.w * b.d == 1)}"
    )
    return new_bricks


def _figure_box_to_grid(box: Box, base_layers: int, margin: int) -> l_pieces.Box:
    """Box (slice z, slice y, slice x) in the figure's own grid -> l_pieces.Box in the grid with base plates and
    base margin (the shift --accent and --retile-box use); margin: studs of base around the figure (0 = no base)."""
    z, y, x = box
    return l_pieces.Box(
        z.start + base_layers,
        z.stop + base_layers,
        y.start + margin,
        y.stop + margin,
        x.start + margin,
        x.stop + margin,
    )


def _swap_in_l_pieces(
    bricks: list[PlacedBrick], context: PassContext, margin: int, skip_boxes: Sequence[Box]
) -> list[PlacedBrick]:
    """Last pipeline step (--use-l-pieces): swap small pieces for owned L pieces; see l_pieces.py. margin: studs
    of base around the figure; skip_boxes: figure boxes whose pieces are never swapped (--l-skip-box)."""
    if context.stock is None:
        print("L pieces: --use-l-pieces needs --stock (the L pieces come from your stock file), skipped")
        return bricks
    options = l_pieces.LOptions(
        base_layers=context.base_layers,
        hang_min=context.hang_min,
        skip=tuple(_figure_box_to_grid(box, context.base_layers, margin) for box in skip_boxes),
    )
    new_bricks, _ = l_pieces.use_l_pieces(bricks, context.grid, context.visible, context.stock, options)
    return new_bricks


@dataclass
class _Packer:
    """Pieces placed so far: bricks (model grid), rects (x, y, w, d) and owner [z, y, x] (index of the piece in
    each cell, -1 = none), plus the joint counts printed at the end. voxel_colours: the cells' own colours (pieces
    over hidden-only cells take them); colours: the packing colours (WILD = any)."""

    grid: NDArray[np.bool_]
    voxel_colours: NDArray[Any]
    colours: NDArray[Any]
    base_layers: int
    stock: Stock | None
    bricks: list[PlacedBrick] = field(default_factory=list)
    rects: list[tuple[int, int, int, int]] = field(default_factory=list)
    owner: NDArray[np.int32] = field(init=False)
    brick_count: int = 0
    plate_count: int = 0
    bridged: int = 0
    stacked: int = 0

    def __post_init__(self) -> None:
        self.owner = np.full(self.grid.shape, -1, np.int32)

    def place(self, rect: PackedRect, z: int, height_plates: int) -> None:
        """Place a piece (colour None = only hidden cells: the cell's own nearest-surface colour), commit it to the
        stock ledger (the packer only took from a fork) and count the joints it bridges / stacks on."""
        x, y, w, d, colour = rect
        if z > 0:
            ids = np.unique(self.owner[z - 1, y : y + d, x : x + w])
            ids = ids[ids >= 0]
            self.bridged += len(ids) > 1
            self.stacked += len(ids) == 1 and self.rects[ids[0]] == (x, y, w, d)
        if colour is None:
            colour = int(self.voxel_colours[z, y, x])
        if self.stock is not None:
            self.stock.take(height_plates, w, d, colour)
        index = len(self.bricks)
        self.bricks.append(PlacedBrick(x, y, z, w, d, h_plates=height_plates, color=colour))
        self.rects.append((x, y, w, d))
        self.owner[z : z + height_plates, y : y + d, x : x + w] = index

    def pack(
        self,
        mask: NDArray[np.bool_],
        colours: NDArray[Any],
        z: int,
        *,
        height_plates: int = PLATE_HEIGHT_PLATES,
        buy: bool | Literal["exhausted"] = True,
    ) -> list[PackedRect]:
        """pack_layer_staggered of the mask cells at layer z over the pieces of layer z - 1, scan direction
        alternating per layer; with stock from a fresh fork (see StockUse for buy)."""
        below = LayerBelow(self.owner[z - 1], self.rects) if z > 0 else None
        stock_use = (
            StockUse(self.stock.fork(), height_plates, self.voxel_colours[z], buy) if self.stock is not None else None
        )
        style = PackStyle(is_flipped=bool((z - self.base_layers) % 2))
        return pack_layer_staggered(mask, colours, below, style, stock_use)

    def pack_plates(self, mask: NDArray[np.bool_], z: int) -> None:
        """Plates over the mask cells of layer z."""
        for rect in self.pack(mask, self.colours[z], z):
            self.place(rect, z, PLATE_HEIGHT_PLATES)
            self.plate_count += 1

    def pack_figure_bands(self) -> None:
        """The figure in fixed 3-plate bands from z = base_layers: bricks where all 3 layers agree (with stock,
        bought only for colours with no unused owned piece left, else the owned plates go first), plates for the
        rest of each layer."""
        layer_count = self.grid.shape[0]
        for z in range(self.base_layers, layer_count, BRICK_HEIGHT_PLATES):
            layers = list(range(z, min(z + BRICK_HEIGHT_PLATES, layer_count)))
            remaining = {k: self.grid[k].copy() for k in layers}
            if len(layers) == BRICK_HEIGHT_PLATES:
                same, band_colours = brick_band(self.grid, self.colours, z)
                for rect in self.pack(same, band_colours, z, height_plates=BRICK_HEIGHT_PLATES, buy="exhausted"):
                    self.place(rect, z, BRICK_HEIGHT_PLATES)
                    x, y, w, d, _ = rect
                    for layer_cells in remaining.values():
                        layer_cells[y : y + d, x : x + w] = False
                    self.brick_count += 1
            for k in layers:
                self.pack_plates(remaining[k], k)

    def pack_bricks_all_phases_then_plates(self) -> None:
        """--prefer-bricks packing order: 3-plate bricks first over the WHOLE figure, in BRICK_BAND_PHASE_COUNT
        passes with the bands shifted by 0, 1, 2 plates (a brick fits wherever 3 consecutive free layers agree in
        colour, not only on the fixed bands); bricks are bought even when owned plates of that colour are left.
        Then plates for what is left, layer by layer from the bottom."""
        layer_count = self.grid.shape[0]
        free = self.grid.copy()
        free[: self.base_layers] = False
        for phase in range(BRICK_BAND_PHASE_COUNT):
            for z in range(self.base_layers + phase, layer_count - 2, BRICK_HEIGHT_PLATES):
                same, band_colours = brick_band(free, self.colours, z)
                for rect in self.pack(same, band_colours, z, height_plates=BRICK_HEIGHT_PLATES, buy=True):
                    self.place(rect, z, BRICK_HEIGHT_PLATES)
                    x, y, w, d, _ = rect
                    free[z : z + BRICK_HEIGHT_PLATES, y : y + d, x : x + w] = False
                    self.brick_count += 1
        for z in range(self.base_layers, layer_count):
            self.pack_plates(free[z], z)

    def place_pieces(self, pieces: Sequence[PlacedBrick]) -> None:
        """Place finished pieces (the stand) as they are."""
        for b in pieces:
            self.place((b.x, b.y, b.w, b.d, b.color), b.z, b.h_plates)


def _packing_colours(
    cells: PackingCells, voxel_colours: NDArray[Any], options: BuildOptions, stock: Stock | None
) -> NDArray[Any]:
    """uint8 copy of the colours with WILD on hidden cells (options.packing.hidden_any_colour) and, with stock, on
    hidden base cells (options.base.hidden_any_colour); prints how many."""
    grid, visible, base_layers = cells.grid, cells.visible, cells.base_layers
    colours = voxel_colours.astype(np.uint8).copy()
    if options.packing.hidden_any_colour and visible is not None:
        hidden = grid & ~visible
        colours[hidden] = WILD
        print(f"  hidden cells (any colour): {int(hidden.sum())} of {int(grid.sum())}")
    if stock is not None and base_layers and visible is not None:
        hidden_base = grid[:base_layers] & ~visible[:base_layers]  # base cells nobody sees
        is_any_colour = options.base.hidden_any_colour
        colours[:base_layers][hidden_base] = WILD if is_any_colour else voxel_colours[:base_layers][hidden_base]
        if is_any_colour:
            print(f"  base cells hidden (any colour): {int(hidden_base.sum())} of {int(grid[:base_layers].sum())}")
    return colours


def _pack_stand(packer: _Packer, base: BaseOptions) -> tuple[NDArray[np.bool_], list[PlacedBrick], int]:
    """The stand, top layer first ("below" = the pieces above it, symmetric bridging): the hand-built layout, the
    layered packer or the plate / brick packer. Returns (grid after the stand, hand-built plates, stand pieces)."""
    grid, base_layers = packer.grid, packer.base_layers
    count_before = len(packer.bricks)
    layout_plates: list[PlacedBrick] = []
    if not base_layers:
        return grid, layout_plates, 0
    if base.layout:
        layout_plates, grid = place_hand_built_base(grid, base_layers, base, packer.stock)
        packer.place_pieces(layout_plates)
    elif base.layered is not None:
        cells = StandCells(grid, packer.colours, packer.owner, packer.rects, packer.bricks, base_layers)
        base_plates, grid = pack_layered_base_with_report(cells, base, packer.stock)
        packer.place_pieces(base_plates)
    else:
        cells = StandCells(grid, packer.colours, packer.owner, packer.rects, packer.bricks, base_layers)
        for x, y, z, w, d, colour, height_plates in pack_base_with_plates(
            cells, base.colour_id, base.brick_bands, packer.stock
        ):
            packer.place((x, y, w, d, colour), z, height_plates)
    return grid, layout_plates, len(packer.bricks) - count_before


def _print_packing_summary(packer: _Packer, figure_total: int, base_count: int, *, prefers_bricks: bool) -> None:
    """The two console lines on pieces and joints after packing."""
    total = figure_total + base_count
    print(
        f"  bricks: {packer.brick_count}, plates: {packer.plate_count}, total: {figure_total}"
        + (" (prefer bricks: bricks first in 3 band shifts, bought freely)" if prefers_bricks else "")
        + (f" (+ {base_count} base pieces = {total})" if packer.base_layers else "")
    )
    print(
        f"  bridging joints below: {packer.bridged} ({100 * packer.bridged / max(total, 1):.0f}%), "
        f"stacked exactly on identical piece: {packer.stacked}"
    )


def _repair_ties(
    bricks: list[PlacedBrick], context: PassContext, fill: NDArray[np.bool_] | None, repair: RepairOptions, margin: int
) -> tuple[list[PlacedBrick], NDArray[np.bool_]]:
    """tie_repair.repair with the repair options (ties and / or clean-up); returns (pieces, grid)."""
    retile_box = (
        tuple(_figure_box_to_grid(repair.retile_box, context.base_layers, margin))
        if repair.retile_box is not None
        else None
    )
    target = tie_repair.RepairTarget(context.grid, context.visible, fill, context.base_layers, retile_box)
    new_bricks, grid = tie_repair.repair(target, bricks, repair, context.stock)
    print(f"  after tie repair: {len(new_bricks)} pieces")
    return new_bricks, grid


def optimize_brick_placement(
    cells: PackingCells, options: BuildOptions | None = None, stock: Stock | None = None
) -> LegoModel:
    """Band-based packing. The model is cut into 3-plate bands (z = 0-2, 3-5, ...).
    In each band, every cell that is filled in all 3 layers with a compatible
    colour is packed into 3-plate BRICKS (largest first); everything left over
    is packed per layer into PLATES. With options.packing.hidden_any_colour the
    cells that are not visible match any colour.
    The lowest cells.base_layers layers are the stand. The figure is packed first
    (its bands start at z = base_layers, exactly as without a base); the base
    plates are packed afterwards, top layer first, so they bridge the joints
    between the figure's bottom pieces and tie the feet together. Then the repair passes run (tie repair, anchor
    floating, merge plates, L pieces). options: the build settings (default: all off)."""
    print("Optimizing brick placement (bricks first, plates for the rest)...")
    options = options if options is not None else BuildOptions()
    base, packing, repair = options.base, options.packing, options.repair
    base_layers = cells.base_layers
    margin = base.margin_studs if base.plates > 0 else 0
    voxel_colours = harmonize_bands(cells.grid, cells.colours) if packing.harmonize_bricks else cells.colours
    colours = _packing_colours(cells, voxel_colours, options, stock)
    packer = _Packer(cells.grid, voxel_colours, colours, base_layers, stock)
    if packing.prefer_bricks:
        packer.pack_bricks_all_phases_then_plates()
    else:
        packer.pack_figure_bands()
    figure_total = packer.brick_count + packer.plate_count
    grid, layout_plates, base_count = _pack_stand(packer, base)
    _print_packing_summary(packer, figure_total, base_count, prefers_bricks=packing.prefer_bricks)
    bricks = packer.bricks
    context = PassContext(grid, cells.visible, base_layers, repair.tie_min_hanging_pieces, stock)
    if repair.tie_repair or repair.cleanup_small:
        bricks, grid = _repair_ties(bricks, context, cells.fill, repair, margin)
        context = PassContext(grid, cells.visible, base_layers, repair.tie_min_hanging_pieces, stock)
    if packing.anchor_floating:
        bricks = _anchor_floating(bricks, context, packing.anchor_floating)
    if packing.merge_plates:
        bricks = _merge_plates(bricks, context)
    if repair.use_l_pieces:
        bricks = _swap_in_l_pieces(bricks, context, margin, repair.l_skip_boxes)
    if layout_plates:  # repair steps may have re-tiled the base: put the hand-built one back
        bricks = base_layout.reimpose(bricks, layout_plates, base_layers, stock)
        grid = base_layout.base_grid(grid, layout_plates, base_layers)
    return LegoModel(
        grid=grid,
        bricks=bricks,
        base_layers=base_layers,
        base_color=base.colour_id if base_layers else None,
        base_margin=margin if base_layers else 0,
    )
