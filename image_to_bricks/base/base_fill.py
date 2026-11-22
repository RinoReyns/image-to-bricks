"""Fill the base with spare owned plates (--base-fill-colors, needs --stock).

A base layer with pieces that are short in the stock (they would have to be bought) is tiled again with SPARE owned
plates of the fill colours (visible layers) or of any colour (hidden base layers), owned plates of each cell's own
colour, and bought standard sizes of the cell's own colour last. Two regions (only the short pieces, the whole
layer) and two greedy tilers (cell by cell, largest pieces first) are tried; the one that buys fewest wins.
Spare = owned minus what the whole finished build (figure and base) uses, so the figure never loses a piece it
could be built from. A layer's new tiling is kept only if it buys fewer pieces and leaves no more loose base pieces
than before (the caller's check).

Pieces: model.PlacedBrick (model grid); stock: stock.Stock (its running ledger follows every change).
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from image_to_bricks.model import PlacedBrick
from image_to_bricks.stock import PLATE_HEIGHT_PLATES, PieceKey, Stock, brick_key, make_key

type Cell = tuple[int, int]  # (y, x)
type SizeCandidate = tuple[int, int, tuple[int, int]]  # (area, colour, (long, short))
type LooseCounter = Callable[[list[PlacedBrick]], int]


@dataclass(frozen=True)
class BaseFillOptions:
    """fill_colours: colours allowed on visible base layers; hidden_layers: lowest base layers nobody sees (any
    spare colour there); buy_sizes: (long, short) plate sizes that may be bought for cells nothing owned fits."""

    fill_colours: Sequence[int]
    hidden_layers: int
    buy_sizes: Sequence[tuple[int, int]]


@dataclass(frozen=True)
class LayerRefill:
    """Result of one base layer: pieces before / after and pieces to buy before / after."""

    layer: int
    removed: list[PlacedBrick]
    added: list[PlacedBrick]
    to_buy_before: int
    to_buy_after: int


def count_to_buy(stock: Stock, used: Counter[PieceKey]) -> int:
    """Pieces that are not covered by owned ones."""
    return sum(max(0, count - stock.have.get(key, 0)) for key, count in used.items())


def _short_pieces(layer_pieces: Sequence[PlacedBrick], stock: Stock, used: Counter[PieceKey]) -> list[PlacedBrick]:
    """Rectangular plates of the layer whose kind is used more often than owned (some must be bought)."""
    short = []
    for piece in layer_pieces:
        key = brick_key(piece)
        if key is not None and piece.h_plates == PLATE_HEIGHT_PLATES and used[key] > stock.have.get(key, 0):
            short.append(piece)
    return short


def _spare_sizes(stock: Stock, used: Counter[PieceKey], colours: Sequence[int]) -> list[SizeCandidate]:
    """(area, colour, (long, short)) of spare owned 1-plate pieces of the given colours, largest first."""
    sizes = {
        (key[1] * key[2], key[3], (key[1], key[2]))
        for key in stock.have
        if key[0] == PLATE_HEIGHT_PLATES and key[3] in colours and stock.have[key] - used.get(key, 0) > 0
    }
    order = {colour: rank for rank, colour in enumerate(colours)}
    return sorted(sizes, key=lambda item: (-item[0], order[item[1]], item[2]))


def _bought_sizes(colours: Sequence[int], buy_sizes: Sequence[tuple[int, int]]) -> list[SizeCandidate]:
    """(area, colour, (w, d)) of every size that may be bought (1x1 always) in every colour, largest first."""
    return sorted(
        {(w * d, colour, (w, d)) for colour in colours for w, d in [*list(buy_sizes), (1, 1)]},
        key=lambda item: (-item[0], item[1]),
    )


@dataclass
class _Tiling:
    """One exact-cover attempt of region {cell: original colour} in layer z; used = ledger counts (changed)."""

    region: dict[Cell, int]
    z: int
    used: Counter[PieceKey]
    covered: set[Cell] = field(default_factory=set)

    def fits(self, y: int, x: int, size: tuple[int, int], cell_colour: int | None) -> bool:
        """True if a piece of size (width, depth) with its corner at (y, x) lies on free region cells (all of
        cell_colour if given)."""
        width, depth = size
        cells = [(y + j, x + i) for j in range(depth) for i in range(width)]
        return all(
            c in self.region and c not in self.covered and (cell_colour is None or self.region[c] == cell_colour)
            for c in cells
        )

    def place(self, y: int, x: int, size: tuple[int, int], colour: int) -> PlacedBrick:
        """Mark the piece's cells covered and its kind used; returns the plate."""
        width, depth = size
        self.covered.update((y + j, x + i) for j in range(depth) for i in range(width))
        self.used[make_key(PLATE_HEIGHT_PLATES, width, depth, colour)] += 1
        return PlacedBrick(x, y, self.z, width, depth, PLATE_HEIGHT_PLATES, colour)

    def is_used_up(self, owned_from: Stock | None, size: tuple[int, int], colour: int) -> bool:
        """True if owned_from is given and no spare piece of this kind is left."""
        if owned_from is None:
            return False
        key = make_key(PLATE_HEIGHT_PLATES, size[0], size[1], colour)
        return owned_from.have.get(key, 0) - self.used.get(key, 0) <= 0

    def place_largest_first(
        self, candidates: Sequence[SizeCandidate], owned_from: Stock | None, *, keeps_cell_colour: bool
    ) -> list[PlacedBrick]:
        """Places candidate pieces, largest first, wherever they fit; owned candidates (owned_from given) only while
        a spare one is left, bought ones (owned_from None) without limit."""
        pieces = []
        for _, colour, (long_side, short_side) in candidates:
            for y, x in sorted(self.region):
                for size in sorted({(long_side, short_side), (short_side, long_side)}):
                    if self.is_used_up(owned_from, size, colour):
                        break
                    if keeps_cell_colour and self.region.get((y, x)) != colour:
                        break
                    if self.fits(y, x, size, colour if keeps_cell_colour else None):
                        pieces.append(self.place(y, x, size, colour))
                        break
        return pieces

    def place_at(
        self, cell: Cell, candidates: Sequence[SizeCandidate], owned_from: Stock | None, *, keeps_cell_colour: bool
    ) -> PlacedBrick | None:
        """First candidate that fits with its corner at cell (owned ones only while spare); marks it used."""
        y, x = cell
        for _, colour, (long_side, short_side) in candidates:
            for size in sorted({(long_side, short_side), (short_side, long_side)}):
                if self.is_used_up(owned_from, size, colour):
                    break
                if self.fits(y, x, size, colour if keeps_cell_colour else None):
                    return self.place(y, x, size, colour)
        return None


type Tiler = Callable[[_Tiling, Stock, Sequence[int], Sequence[tuple[int, int]]], list[PlacedBrick]]


def _tile_largest_first(
    tiling: _Tiling, stock: Stock, spare_colours: Sequence[int], buy_sizes: Sequence[tuple[int, int]]
) -> list[PlacedBrick]:
    """Exact cover of the region, largest pieces first in three rounds: owned pieces of the cells' own colours,
    spare pieces of the fill colours, then bought sizes of the cells' own colours (1x1 last)."""
    own_colours = sorted(set(tiling.region.values()))
    pieces = tiling.place_largest_first(_spare_sizes(stock, tiling.used, own_colours), stock, keeps_cell_colour=True)
    pieces += tiling.place_largest_first(
        _spare_sizes(stock, tiling.used, spare_colours), stock, keeps_cell_colour=False
    )
    pieces += tiling.place_largest_first(_bought_sizes(own_colours, buy_sizes), None, keeps_cell_colour=True)
    return pieces


def _tile_cell_by_cell(
    tiling: _Tiling, stock: Stock, spare_colours: Sequence[int], buy_sizes: Sequence[tuple[int, int]]
) -> list[PlacedBrick]:
    """Exact cover of the region, scanning cells in order: at each free cell the largest spare piece of the fill
    colours, else the largest owned / bought piece of the cell's own colour."""
    pieces = []
    for cell in sorted(tiling.region):
        if cell in tiling.covered:
            continue
        own_colour = [tiling.region[cell]]
        rounds: tuple[tuple[list[SizeCandidate], Stock | None, bool], ...] = (
            (_spare_sizes(stock, tiling.used, spare_colours), stock, False),
            (_spare_sizes(stock, tiling.used, own_colour), stock, True),
            (_bought_sizes(own_colour, buy_sizes), None, True),
        )
        for candidates, owned_from, keeps_cell_colour in rounds:
            placed = tiling.place_at(cell, candidates, owned_from, keeps_cell_colour=keeps_cell_colour)
            if placed is not None:
                pieces.append(placed)
                break
    return pieces


TILERS: tuple[Tiler, ...] = (_tile_cell_by_cell, _tile_largest_first)


@dataclass(frozen=True)
class _LayerFill:
    """One base layer being refilled: its layer z, the stock, the ledger counts of the whole build and options."""

    z: int
    stock: Stock
    used: Counter[PieceKey]
    options: BaseFillOptions

    def trial(self, replaced: Sequence[PlacedBrick], tiler: Tiler) -> tuple[list[PlacedBrick], Counter[PieceKey]]:
        """(new pieces, ledger counts after) when `replaced` is tiled again by `tiler`; `used` is not changed."""
        trial_used = Counter(self.used)
        region: dict[Cell, int] = {}
        for piece in replaced:
            key = brick_key(piece)
            if key is not None:
                trial_used[key] -= 1
            region.update({(piece.y + j, piece.x + i): piece.color for j in range(piece.d) for i in range(piece.w)})
        if self.z >= self.options.hidden_layers:
            spare_colours = list(self.options.fill_colours)
        else:
            spare_colours = sorted({key[3] for key in self.stock.have})
        tiling = _Tiling(region, self.z, trial_used)
        return tiler(tiling, self.stock, spare_colours, self.options.buy_sizes), trial_used


def _apply_refill(
    bricks: list[PlacedBrick], stock: Stock, replaced: Sequence[PlacedBrick], added: Sequence[PlacedBrick]
) -> None:
    """Swap the replaced pieces for the added ones in `bricks` and in the stock ledger."""
    for piece in replaced:
        bricks.remove(piece)
        stock.give_brick(piece)
    for piece in added:
        bricks.append(piece)
        stock.take_brick(piece)


def refill_base_layer(
    bricks: list[PlacedBrick], z: int, stock: Stock, options: BaseFillOptions, count_loose: LooseCounter
) -> LayerRefill | None:
    """Base layer z with short pieces is tiled again (in place in `bricks`, ledger updated). Two regions are
    tried, only the short pieces and the whole layer (so owned pieces can move too); the one that buys fewest is
    kept (each with both tilers in TILERS) if it buys fewer than now and count_loose(bricks) does not grow.
    Returns what changed or None."""
    used = Counter(key for key in (brick_key(b) for b in bricks) if key is not None)
    layer_pieces = [b for b in bricks if b.z == z and b.notch < 0 and b.h_plates == PLATE_HEIGHT_PLATES]
    short = _short_pieces(layer_pieces, stock, used)
    if not short:
        return None
    layer_fill = _LayerFill(z, stock, used, options)
    to_buy_before, loose_before = count_to_buy(stock, used), count_loose(bricks)
    best: tuple[list[PlacedBrick], list[PlacedBrick], int] | None = None
    for replaced, tiler in [(region, tiler) for region in (short, layer_pieces) for tiler in TILERS]:
        added, trial_used = layer_fill.trial(replaced, tiler)
        to_buy_after = count_to_buy(stock, trial_used)
        trial = [b for b in bricks if all(b is not r for r in replaced)] + added
        if (
            to_buy_after < to_buy_before
            and count_loose(trial) <= loose_before
            and (best is None or to_buy_after < best[2])
        ):
            best = (replaced, added, to_buy_after)
    if best is None:
        return None
    replaced, added, to_buy_after = best
    _apply_refill(bricks, stock, replaced, added)
    return LayerRefill(z, replaced, added, to_buy_before, to_buy_after)


def fill_base_from_spare(
    bricks: list[PlacedBrick],
    base_layers: int,
    stock: Stock,
    options: BaseFillOptions,
    count_loose: LooseCounter,
) -> list[LayerRefill]:
    """refill_base_layer for every base layer, top first; returns the layers that changed."""
    results = [refill_base_layer(bricks, z, stock, options, count_loose) for z in range(base_layers - 1, -1, -1)]
    return [r for r in results if r is not None]
