"""Layered base (--base-hidden-layers): big plates first, hidden layers with offset seams.

The pattern of the LEGO Architecture sets (e.g. 21065): the base is a few big plates in stacked plate
layers. Only the top layer(s) show the zone colours (grass / pond / path); the layers under them are
hidden and may be any colour, so the biggest owned spare plates go there. Each layer is laid out so
that its pieces span as many joints of the layer above as possible (like brick bonding), which keeps
a base of few big plates rigid.

Coordinates: everything is in the model grid (`grid[z, y, x]`, base layers are z < n_base_layers).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.base import base_ties
from image_to_bricks.model import PlacedBrick
from image_to_bricks.stock import PLATE_HEIGHT_PLATES, Stock

WILD = 255  # colour code of a cell that may take any colour (same value as layer_packer.WILD)

# Plate sizes (long side, short side) that may be BOUGHT for the base: the standard sizes with a row in
# my_bricks_catalog/price_list.csv. Owned plates of any size are used as well.
BASE_BUY_SIZES: tuple[tuple[int, int], ...] = (
    (8, 2),
    (4, 4),
    (6, 1),
    (4, 2),
    (3, 2),
    (4, 1),
    (2, 2),
    (3, 1),
    (2, 1),
    (1, 1),
)
BASE_MAX_PLATE: tuple[int, int] = (16, 16)  # largest plate (long side, short side) the base may use
# Share of the layer above's joints that a layer should span (0..1); layouts below it are only used
# when no scan order reaches it.
BASE_MIN_BRIDGED = 0.75
# Score bonus (in studs of area) per extra piece of the layer above that one plate spans: a plate that ties
# pieces together beats a slightly bigger one that sits on a single piece (seams move between layers).
BASE_SPAN_BONUS = 3.0

type Cell = tuple[int, int]  # (y, x)
type Rectangle = tuple[int, int, int, int]  # (x, y, width, depth) in studs


@dataclass(frozen=True)
class LayeredBaseOptions:
    """hidden_layers: number of plate layers at the bottom of the base that nobody sees (any colour).
    sparse_layers: the lowest of those layers that are sparse tie layers (base_ties), 0..hidden_layers.
    tie_every: studs of seam per tie in a sparse layer.
    max_plate: largest plate (long, short) in studs. buy_sizes: plate sizes that may be bought.
    min_bridged: preferred share of the joints of the layer above each full layer must span."""

    hidden_layers: int
    sparse_layers: int = 0
    tie_every: int = base_ties.BASE_TIE_EVERY
    max_plate: tuple[int, int] = BASE_MAX_PLATE
    buy_sizes: tuple[tuple[int, int], ...] = BASE_BUY_SIZES
    min_bridged: float = BASE_MIN_BRIDGED


@dataclass(frozen=True)
class BaseCells:
    """What the layered base is packed into: grid [z, y, x] bool (base outline per layer in grid[:layer_count]),
    colours [z, y, x] (colour per cell, WILD = any; the hidden layers are made WILD by the packer) and figure_ids
    [y, x] (piece index of the figure's lowest layer, -1 = none; the top base layer spans its joints)."""

    grid: NDArray[np.bool_]
    colours: NDArray[Any]
    figure_ids: NDArray[Any]
    layer_count: int


@dataclass
class LayerReport:
    """What one base layer got: z, pieces, and either the share of joints spanned (full layer) or the seam /
    foot requirements met (sparse layer)."""

    z: int
    pieces: int
    bridged: float | None = None
    requirements: int = 0
    open_requirements: int = 0

    @property
    def is_sparse(self) -> bool:
        """True for a sparse tie layer (no joint share)."""
        return self.bridged is None


@dataclass
class _Layout:
    """One packed layer: its plates, the stock fork after it, its joint share and the pieces it buys."""

    pieces: list[PlacedBrick]
    stock: Stock | None  # stock.Stock fork after the layer (None without --stock)
    bridged: float
    bought: int = field(default=0)


def parse_plate_size(text: str) -> tuple[int, int]:
    """'8x8' / '12X6' -> (long side, short side). Raises ValueError with the expected format."""
    try:
        first, second = (int(v) for v in text.lower().split("x"))
    except ValueError as error:
        raise ValueError(f"plate size must look like 8x8 (WxD in studs), got {text!r}") from error
    if first < 1 or second < 1:
        raise ValueError(f"plate size must be at least 1x1, got {text!r}")
    return max(first, second), min(first, second)


def _fits_max(w: int, d: int, max_plate: tuple[int, int]) -> bool:
    """True if a w x d plate is not bigger than max_plate (long, short)."""
    return max(w, d) <= max_plate[0] and min(w, d) <= max_plate[1]


def candidate_sizes(stock: Stock | None, options: LayeredBaseOptions) -> list[tuple[int, int]]:
    """Every (w, d) the packer may try, both orientations, largest area first (then longest side)."""
    sizes = set()
    for long_side, short_side in options.buy_sizes:
        sizes |= {(long_side, short_side), (short_side, long_side)}
    if stock is not None:
        owned_plates = [(w, d) for (h, w, d, _c) in stock.have if h == PLATE_HEIGHT_PLATES]
        sizes |= set(owned_plates) | {(d, w) for w, d in owned_plates}
    fitting = [size for size in sizes if _fits_max(*size, options.max_plate)]
    return sorted(fitting, key=lambda size: (-size[0] * size[1], -max(size), -size[0]))


def scan_orders(mask: NDArray[np.bool_]) -> list[list[Cell]]:
    """The cells of `mask` [h, w] in 8 scan orders (4 start corners x rows / columns first), as (y, x)."""
    depth, width = mask.shape
    orders = []
    for y_step, x_step, is_rows_first in product((1, -1), (1, -1), (True, False)):
        ys = list(range(depth))[::y_step]
        xs = list(range(width))[::x_step]
        cells = [(y, x) for y in ys for x in xs] if is_rows_first else [(y, x) for x in xs for y in ys]
        orders.append([(y, x) for (y, x) in cells if mask[y, x]])
    return orders


def _rect_colour(colours: NDArray[Any], rect: Rectangle) -> tuple[bool, int | None]:
    """(ok, colour): ok if the visible cells under the rectangle agree; colour None = all cells are WILD."""
    x, y, w, d = rect
    cells = colours[y : y + d, x : x + w]
    visible = cells[cells != WILD]
    if len(visible) and not (visible == visible[0]).all():
        return False, None
    return True, (int(visible[0]) if len(visible) else None)


def _resolve(colour: int | None, w: int, d: int, stock: Stock | None, fallback: int) -> tuple[int, bool]:
    """(colour, owned) of a plate: a WILD-only plate takes the owned colour with most left, else `fallback`."""
    if colour is None:
        colour = stock.any_owned_colour(PLATE_HEIGHT_PLATES, w, d) if stock is not None else None
        colour = fallback if colour is None else colour
    return colour, stock is not None and stock.avail(PLATE_HEIGHT_PLATES, w, d, colour)


def _anchors(x: int, y: int, w: int, d: int) -> tuple[tuple[int, int], ...]:
    """Positions (x0, y0) of a w x d rectangle that has cell (x, y) as one of its corners."""
    return ((x, y), (x - w + 1, y), (x, y - d + 1), (x - w + 1, y - d + 1))


def _span_score(upper_ids: NDArray[Any], rect: Rectangle) -> float:
    """Area of a w x d plate plus BASE_SPAN_BONUS per extra piece of the layer above it spans."""
    x0, y0, w, d = rect
    ids = np.unique(upper_ids[y0 : y0 + d, x0 : x0 + w])
    return w * d + BASE_SPAN_BONUS * max(0, int((ids >= 0).sum()) - 1)


@dataclass(frozen=True)
class _PlateLayerJob:
    """One full base layer to cover: mask [h, w] bool, colours [h, w] (WILD = any), upper_ids [h, w] (piece index
    of the layer above, -1 = none; plates that span its joints score higher), the layer z, the options and the
    colour of bought plates over WILD-only cells."""

    mask: NDArray[np.bool_]
    colours: NDArray[Any]
    upper_ids: NDArray[Any]
    z: int
    options: LayeredBaseOptions
    fallback: int

    def piece_at(
        self, free: NDArray[np.bool_], cell: Cell, sizes: Sequence[tuple[int, int]], stock: Stock | None
    ) -> tuple[Rectangle, int] | None:
        """Best plate covering cell (y, x) as a corner: highest `_span_score`, owned beats bought at equal score.
        Returns (rectangle, colour) or None if nothing fits."""
        y, x = cell
        depth, width = free.shape
        buy_sizes = set(self.options.buy_sizes)
        best: tuple[Rectangle, int] | None = None
        best_key: tuple[float, bool, int] | None = None
        for w, d in sizes:
            for x0, y0 in _anchors(x, y, w, d):
                if x0 < 0 or y0 < 0 or x0 + w > width or y0 + d > depth or not free[y0 : y0 + d, x0 : x0 + w].all():
                    continue
                rect = (x0, y0, w, d)
                is_one_colour, colour = _rect_colour(self.colours, rect)
                if not is_one_colour:
                    continue
                resolved_colour, is_owned = _resolve(colour, w, d, stock, self.fallback)
                if not is_owned and (max(w, d), min(w, d)) not in buy_sizes:
                    continue
                key = (_span_score(self.upper_ids, rect), is_owned, w * d)
                if best_key is None or key > best_key:
                    best, best_key = (rect, resolved_colour), key
        return best

    def pack(self, order: Sequence[Cell], stock: Stock | None) -> list[PlacedBrick]:
        """Cover the mask exactly with plates, scanning cells in `order`; owned plates are taken from `stock` (a
        FORK). Raises RuntimeError if a cell cannot be covered (only possible when 1x1 cannot be bought)."""
        free = self.mask.copy()
        sizes = candidate_sizes(stock, self.options)
        pieces = []
        for y, x in order:
            if not free[y, x]:
                continue
            found = self.piece_at(free, (y, x), sizes, stock)
            if found is None:
                raise RuntimeError(f"base layer {self.z}: no plate fits cell x={x}, y={y}; add 1x1 to the buy sizes")
            (x0, y0, w, d), colour = found
            free[y0 : y0 + d, x0 : x0 + w] = False
            if stock is not None:
                stock.take(PLATE_HEIGHT_PLATES, w, d, colour)
            pieces.append(PlacedBrick(x0, y0, self.z, w, d, h_plates=PLATE_HEIGHT_PLATES, color=colour))
        return pieces


def piece_ids(pieces: Sequence[PlacedBrick], shape: tuple[int, ...]) -> NDArray[np.int32]:
    """int [h, w]: index of the piece covering each cell (-1 = none)."""
    ids = np.full(shape, -1, np.int32)
    for i, b in enumerate(pieces):
        ids[b.y : b.y + b.d, b.x : b.x + b.w] = i
    return ids


def bridged_fraction(upper_ids: NDArray[Any], lower_ids: NDArray[Any]) -> float:
    """Share of the joints of the upper layer (edges between two different upper pieces, over cells the lower
    layer covers) that lie inside ONE lower piece. 1.0 if the upper layer has no such joint."""
    total = bridged = 0
    for first, second in (
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
    ):
        upper_first, upper_second = upper_ids[first], upper_ids[second]
        lower_first, lower_second = lower_ids[first], lower_ids[second]
        joint = (
            (upper_first >= 0)
            & (upper_second >= 0)
            & (upper_first != upper_second)
            & (lower_first >= 0)
            & (lower_second >= 0)
        )
        total += int(joint.sum())
        bridged += int((joint & (lower_first == lower_second)).sum())
    return bridged / total if total else 1.0


def _layout_key(layout: _Layout, min_bridged: float) -> tuple[bool, int, int, float]:
    """Smaller is better: enough joints spanned, then fewer bought, fewer pieces, more joints spanned."""
    return (layout.bridged < min_bridged, layout.bought, len(layout.pieces), -layout.bridged)


def best_layer(job: _PlateLayerJob, stock: Stock | None) -> _Layout:
    """Pack one base layer in every scan order and keep the best layout (see `_layout_key`)."""
    best: _Layout | None = None
    min_bridged = job.options.min_bridged
    for order in scan_orders(job.mask):
        fork = stock.fork() if stock is not None else None
        pieces = job.pack(order, fork)
        bought = 0 if fork is None or stock is None else fork.need_total() - stock.need_total()
        layout = _Layout(pieces, fork, bridged_fraction(job.upper_ids, piece_ids(pieces, job.mask.shape)), bought)
        if best is None or _layout_key(layout, min_bridged) < _layout_key(best, min_bridged):
            best = layout
    if best is None:
        raise ValueError(f"base layer {job.z} has no scan order")
    return best


def _validate(options: LayeredBaseOptions, layer_count: int) -> None:
    """ValueError if the hidden / sparse layer counts or tie_every do not fit a base of layer_count layers."""
    if not 0 <= options.hidden_layers < layer_count:
        raise ValueError(
            f"hidden layers must be 0..{layer_count - 1} (at least one visible layer), got {options.hidden_layers}"
        )
    if not 0 <= options.sparse_layers <= options.hidden_layers:
        raise ValueError(
            f"sparse layers must be 0..{options.hidden_layers} (only hidden layers can be sparse), "
            f"got {options.sparse_layers}"
        )
    if options.tie_every < 1:
        raise ValueError(f"tie_every must be at least 1, got {options.tie_every}")


@dataclass(frozen=True)
class _BasePacking:
    """The cells, options and fallback colour of one layered base being packed."""

    cells: BaseCells
    options: LayeredBaseOptions
    fallback: int

    def full_layer(
        self, z: int, upper: NDArray[Any], stock: Stock | None
    ) -> tuple[list[PlacedBrick], Stock | None, LayerReport]:
        """A full plate layer (hidden layers take any colour): (pieces, stock after, report)."""
        colours = self.cells.colours
        layer_colours = (
            colours[z] if z >= self.options.hidden_layers else np.full(colours[z].shape, WILD, colours.dtype)
        )
        job = _PlateLayerJob(self.cells.grid[z], layer_colours, upper, z, self.options, self.fallback)
        layout = best_layer(job, stock)
        return layout.pieces, layout.stock, LayerReport(z, len(layout.pieces), bridged=layout.bridged)

    def sparse_layer(
        self, z: int, upper: NDArray[Any], stock: Stock | None
    ) -> tuple[list[PlacedBrick], Stock | None, LayerReport]:
        """A sparse tie layer (base_ties): (pieces, stock after, report)."""
        supply = base_ties.TieSupply(stock, frozenset(self.options.buy_sizes), self.fallback)
        rules = base_ties.TieRules(candidate_sizes(stock, self.options), self.options.tie_every)
        pieces, requirement_count, open_reqs = base_ties.pack_tie_layer(self.cells.grid[z], upper, z, supply, rules)
        report = LayerReport(z, len(pieces), requirements=requirement_count, open_requirements=len(open_reqs))
        return pieces, stock, report


def pack_layered_base(
    cells: BaseCells, stock: Stock | None, options: LayeredBaseOptions, fallback: int
) -> tuple[list[PlacedBrick], list[LayerReport]]:
    """Pack the cells.layer_count base plate layers, top first (see BaseCells). The lowest options.sparse_layers
    layers are sparse tie layers (base_ties): they do NOT cover the whole outline. stock: the real stock.Stock (not
    changed: the caller takes the returned pieces from it) or None. fallback: colour of bought plates over WILD-only
    cells. Returns (pieces, one LayerReport per layer, top first)."""
    _validate(options, cells.layer_count)
    packing = _BasePacking(cells, options, fallback)
    work = stock.fork() if stock is not None else None
    upper = cells.figure_ids
    pieces: list[PlacedBrick] = []
    reports: list[LayerReport] = []
    for z in range(cells.layer_count - 1, -1, -1):
        if z < options.sparse_layers:
            layer, work, report = packing.sparse_layer(z, upper, work)
        else:
            layer, work, report = packing.full_layer(z, upper, work)
        pieces += layer
        reports.append(report)
        upper = piece_ids(layer, cells.grid[z].shape)
    return pieces, reports


def base_cells(grid: NDArray[np.bool_], pieces: Sequence[PlacedBrick], n_layers: int) -> NDArray[np.bool_]:
    """Copy of `grid` [z, y, x] whose base layers hold exactly the cells the base pieces cover (sparse layers
    leave holes)."""
    out = grid.copy()
    out[:n_layers] = False
    for b in pieces:
        out[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] = True
    return out


def _report_text(r: LayerReport) -> str:
    """'12 (joints 80%)' for a full layer, '5 sparse (7/9 seam ties + feet)' for a sparse one."""
    if r.bridged is not None:
        return f"{r.pieces} (joints {100 * r.bridged:.0f}%)"
    met = r.requirements - r.open_requirements
    return f"{r.pieces} sparse ({met}/{r.requirements} seam ties + feet)"


def summary(pieces: Sequence[PlacedBrick], reports: Sequence[LayerReport], hidden_layers: int) -> str:
    """One console line: pieces per layer (top first) with joints spanned (full layers) or ties met (sparse)."""
    layers = ", ".join(_report_text(r) for r in reports)
    sparse = sum(r.is_sparse for r in reports)
    return (
        f"Layered base: {len(pieces)} plates in {len(reports)} layers ({hidden_layers} hidden, any colour"
        + (f", {sparse} sparse" if sparse else "")
        + f"); top first: {layers}"
    )
