"""Packing one layer of cells into plates / bricks (staggered over the joints below), and the band colour
harmonising.

Layer arrays are [y, x] in studs; rectangles are (x, y, width, depth) in studs of the packed layer.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.consts import BRICK_HEIGHT_PLATES
from image_to_bricks.stock import Stock

STAGGER = True  # prefer pieces that bridge joints of the layer below
STAGGER_BONUS = 1.5  # score bonus per extra piece bridged below
STACK_PENALTY = 2.0  # score penalty for sitting exactly on an identical piece
STACK_PENALTY_PER_STUD = 0.5  # ...plus this per stud (stops big identical stacks)
WILD = 255  # colour code for hidden cells: matches any colour
BRICK_BAND_PHASE_COUNT = 3  # --prefer-bricks: band shifts tried for bricks (a brick is 3 plates tall)
BOND_PERIOD_STUDS = 4  # brick bond: piece edges land on x % 4 == phase x (and the same for y)
THIN_STRIP_STUDS = 1  # a free run of exactly this many studs beyond a piece is a thin strip (--avoid-thin)

_SIZES = [(4, 4), (2, 4), (4, 2), (2, 3), (3, 2), (2, 2), (1, 4), (4, 1), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1)]
_STANDARD_SIZES = set(_SIZES)  # sizes that can be bought

type Rect = tuple[int, int, int, int]  # (x, y, width, depth) in studs
type PackedRect = tuple[int, int, int, int, int | None]  # (x, y, width, depth, colour or None = hidden only)
type CandidateKey = tuple[bool, int]  # (not owned, tier): the smallest wins
type Candidate = tuple[float, Rect, int | None]  # (score, rectangle, colour)

# tiers of a candidate: passes the guards, off the brick-bond phase, leaves a thin strip (last resort)
TIER_OK, TIER_OFF_PHASE, TIER_THIN_STRIP = 0, 1, 2


@dataclass(frozen=True)
class LayerBelow:
    """The packed layer under the one being packed."""

    piece_ids: NDArray[Any]  # int [y, x]: index of the piece under each cell (-1 = none)
    rects: Sequence[Rect]  # rects[i] = footprint of piece i


@dataclass(frozen=True)
class PackStyle:
    """How the packer scans and which shapes it avoids.

    is_flipped: scan from the opposite corner so joints shift between layers.
    avoids_thin_strips: reject sizes that would leave a free run of exactly 1 stud beyond the piece (in scan
    direction, along its anchor row / column) - such 1-wide edge strips end up as stacks of strips that only touch
    each other (used for the base).
    phase (px, py): brick bond - piece edges (in scan direction) must land on x % 4 == px / y % 4 == py (or at the
    end of a free run), so layers with different phases interlock in both directions.
    Both guards fall back to the best plain fit when nothing passes."""

    is_flipped: bool = False
    avoids_thin_strips: bool = False
    phase: tuple[int, int] | None = None


@dataclass(frozen=True)
class StockUse:
    """Packing from a stock (normally a fork): owned pieces are preferred. At every cell the largest scoring size
    that is still in stock (height_plates, colour of the piece) wins; only if none fits, a piece is "bought" (a
    standard size up to 4x4, any count). buy=True allows that, buy=False never buys (cells may stay unpacked),
    buy="exhausted" buys only for colours with no unused owned piece left. Every chosen piece is taken from
    `stock`. fallback: colour (int or [y, x] array) for rectangles that cover only hidden cells."""

    stock: Stock
    height_plates: int = 1
    fallback: int | NDArray[Any] | None = None
    buy: bool | Literal["exhausted"] = True


def _free_run(remaining: NDArray[np.bool_], x: int, y: int, step_x: int, step_y: int) -> int:
    """Number of consecutive free cells from (x, y) in direction (step_x, step_y), counted up to
    THIN_STRIP_STUDS + 1."""
    depth, width = remaining.shape
    count = 0
    while count <= THIN_STRIP_STUDS and 0 <= x < width and 0 <= y < depth and remaining[y, x]:
        count += 1
        x += step_x
        y += step_y
    return count


@dataclass
class _LayerPacking:
    """State of one greedy packing pass over a layer."""

    remaining: NDArray[np.bool_]  # [y, x] cells still to cover
    colours: NDArray[Any]  # [y, x] colour id per cell, WILD = hidden (any colour)
    below: LayerBelow | None
    style: PackStyle
    stock_use: StockUse | None
    sizes: list[tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.sizes = _SIZES if self.stock_use is None else self.stock_use.stock.sizes_with(_SIZES)

    def scan_order(self) -> list[tuple[int, int]]:
        """(y, x) anchor cells: visible cells first, so they can grow into the hidden (any-colour) cells behind
        them; then the leftover hidden cells."""
        depth, width = self.remaining.shape
        ys = list(range(depth - 1, -1, -1)) if self.style.is_flipped else list(range(depth))
        xs = list(range(width - 1, -1, -1)) if self.style.is_flipped else list(range(width))
        visible = [(y, x) for y in ys for x in xs if self.colours[y, x] != WILD]
        return visible + [(y, x) for y in ys for x in xs if self.colours[y, x] == WILD]

    def rect_at(self, x: int, y: int, size: tuple[int, int]) -> Rect | None:
        """The rectangle of `size` anchored at (x, y) (towards the scan direction) if it fits on free cells."""
        width, depth = size
        x0, y0 = (x - width + 1, y - depth + 1) if self.style.is_flipped else (x, y)
        layer_depth, layer_width = self.remaining.shape
        if x0 < 0 or y0 < 0 or x0 + width > layer_width or y0 + depth > layer_depth:
            return None
        if not self.remaining[y0 : y0 + depth, x0 : x0 + width].all():
            return None
        return x0, y0, width, depth

    def visible_colour(self, rect: Rect) -> tuple[bool, int | None]:
        """(fits, colour): the one colour of the rectangle's visible cells (None = hidden cells only); fits is
        False if the visible cells have more than one colour."""
        x0, y0, width, depth = rect
        cells = self.colours[y0 : y0 + depth, x0 : x0 + width]
        visible = cells[cells != WILD]  # hidden cells (WILD) match any colour
        if len(visible) and not (visible == visible[0]).all():
            return False, None
        return True, int(visible[0]) if len(visible) else None

    def stock_colour(self, rect: Rect, colour: int | None) -> tuple[bool, bool, int | None]:
        """(allowed, not owned, colour) of the rectangle under the stock rules (always allowed without stock)."""
        if self.stock_use is None:
            return True, False, colour
        stock, height = self.stock_use.stock, self.stock_use.height_plates
        x0, y0, width, depth = rect
        fallback = self.stock_use.fallback
        if colour is None:  # only hidden cells: any colour will do - an owned one if possible
            colour = stock.any_owned_colour(height, width, depth)
            if colour is None and fallback is not None:
                colour = int(fallback) if isinstance(fallback, int) else int(fallback[y0, x0])
        is_not_owned = colour is None or not stock.avail(height, width, depth, colour)
        buy = self.stock_use.buy
        can_buy = (width, depth) in _STANDARD_SIZES and (
            buy is True or (buy == "exhausted" and (colour is None or not stock.colour_has_stock(colour)))
        )
        return not is_not_owned or can_buy, is_not_owned, colour

    def score(self, rect: Rect) -> float:
        """Area, plus STAGGER_BONUS per extra piece bridged below, minus the stack penalty when exactly on top of
        an identical footprint."""
        x0, y0, width, depth = rect
        score = float(width * depth)
        if STAGGER and self.below is not None:
            ids = np.unique(self.below.piece_ids[y0 : y0 + depth, x0 : x0 + width])
            ids = ids[ids >= 0]
            if len(ids) > 1:
                score += STAGGER_BONUS * (len(ids) - 1)
            elif len(ids) == 1 and self.below.rects[ids[0]] == rect:
                score -= STACK_PENALTY + STACK_PENALTY_PER_STUD * width * depth
        return score

    def tier(self, rect: Rect, x: int, y: int) -> int:
        """TIER_THIN_STRIP if the rectangle leaves a thin strip, TIER_OFF_PHASE if it misses the brick bond, else
        TIER_OK (x, y: the anchor cell)."""
        x0, y0, width, depth = rect
        is_flipped, remaining = self.style.is_flipped, self.remaining
        step = -1 if is_flipped else 1
        if self.style.avoids_thin_strips and (
            _free_run(remaining, x0 - 1 if is_flipped else x0 + width, y, step, 0) == THIN_STRIP_STUDS
            or _free_run(remaining, x, y0 - 1 if is_flipped else y0 + depth, 0, step) == THIN_STRIP_STUDS
        ):
            return TIER_THIN_STRIP
        phase = self.style.phase
        if phase is None:
            return TIER_OK
        edge_x, edge_y = (x0, y0) if is_flipped else (x0 + width, y0 + depth)
        beyond_x = edge_x - 1 if is_flipped else edge_x  # first cell beyond the piece, in scan direction
        beyond_y = edge_y - 1 if is_flipped else edge_y
        layer_depth, layer_width = remaining.shape
        is_on_phase_x = (
            edge_x % BOND_PERIOD_STUDS == phase[0] or not (0 <= beyond_x < layer_width) or not remaining[y, beyond_x]
        )
        is_on_phase_y = (
            edge_y % BOND_PERIOD_STUDS == phase[1] or not (0 <= beyond_y < layer_depth) or not remaining[beyond_y, x]
        )
        return TIER_OK if is_on_phase_x and is_on_phase_y else TIER_OFF_PHASE

    def best_at(self, x: int, y: int) -> Candidate | None:
        """The best rectangle anchored at (x, y): owned beats bought, then the lowest tier, then the highest score
        (the first size wins ties); None if nothing is allowed."""
        candidates: dict[CandidateKey, Candidate] = {}
        for size in self.sizes:
            rect = self.rect_at(x, y, size)
            if rect is None:
                continue
            fits, colour = self.visible_colour(rect)
            if not fits:
                continue
            is_allowed, is_not_owned, colour = self.stock_colour(rect, colour)
            if not is_allowed:
                continue
            score = self.score(rect)
            key = (is_not_owned, self.tier(rect, x, y))
            if key not in candidates or score > candidates[key][0]:
                candidates[key] = (score, rect, colour)
        return candidates[min(candidates)] if candidates else None


def pack_layer_staggered(
    layer: NDArray[np.bool_],
    colors: NDArray[Any],
    below: LayerBelow | None = None,
    style: PackStyle | None = None,
    stock_use: StockUse | None = None,
) -> list[PackedRect]:
    """Greedy packing of the cells of `layer` (bool [y, x]; colors [y, x], WILD = hidden): at each anchor cell
    it scores every fitting size (see _LayerPacking.score) and keeps the best (see PackStyle and StockUse).
    Returns (x, y, w, d, colour) - colour None if the piece covers only hidden cells (with stock: the resolved
    colour). Without stock every cell is covered ((1, 1) always fits)."""
    packing = _LayerPacking(layer.copy(), colors, below, style or PackStyle(), stock_use)
    packed: list[PackedRect] = []
    for y, x in packing.scan_order():
        if not packing.remaining[y, x]:
            continue
        best = packing.best_at(x, y)
        if best is None:
            continue  # only with stock and buy=False: leave the cell to the caller
        _, (x0, y0, width, depth), colour = best
        if stock_use is not None and colour is not None:
            stock_use.stock.take(stock_use.height_plates, width, depth, colour)
        packed.append((x0, y0, width, depth, colour))
        packing.remaining[y0 : y0 + depth, x0 : x0 + width] = False
    return packed


def _colours_agree(first: NDArray[Any], second: NDArray[Any]) -> NDArray[np.bool_]:
    """Cells where two colour layers may share one piece (equal, or one of them WILD)."""
    return np.asarray((first == WILD) | (second == WILD) | (first == second), dtype=np.bool_)


def brick_band(cells: NDArray[np.bool_], colours: NDArray[Any], z: int) -> tuple[NDArray[np.bool_], NDArray[Any]]:
    """Layers z..z+2 as one brick band: (cells filled in all 3 layers whose colours agree (WILD = any), the
    band's colour per cell = the first visible colour from the bottom). cells / colours are [z, y, x]."""
    bottom, middle, top = colours[z], colours[z + 1], colours[z + 2]
    same = (
        cells[z]
        & cells[z + 1]
        & cells[z + 2]
        & _colours_agree(bottom, middle)
        & _colours_agree(middle, top)
        & _colours_agree(bottom, top)
    )
    band_colours = np.where(bottom != WILD, bottom, np.where(middle != WILD, middle, top))
    return same, band_colours


def harmonize_bands(grid: NDArray[np.bool_], colors: NDArray[Any]) -> NDArray[Any]:
    """Within each 3-plate band, cells filled in all 3 layers take the majority
    colour of the three (ties -> middle layer), so they can become one brick."""
    colors = colors.copy()
    layer_count = grid.shape[0]
    changed = 0
    for z in range(0, layer_count - (BRICK_HEIGHT_PLATES - 1), BRICK_HEIGHT_PLATES):
        full = grid[z] & grid[z + 1] & grid[z + 2]
        bottom, middle, top = colors[z], colors[z + 1], colors[z + 2]
        # bottom == top -> bottom; otherwise the middle wins (bottom == middle or middle == top or all differ)
        majority = np.where(bottom == top, bottom, middle)
        differs = full & ~((bottom == middle) & (middle == top))
        changed += int(differs.sum())
        for offset in range(BRICK_HEIGHT_PLATES):
            colors[z + offset][differs] = majority[differs]
    print(f"Band colour harmonisation: {changed} columns unified")
    return colors
