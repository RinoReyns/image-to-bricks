"""Tie repair: make the finished brick model hold together.

LEGO pieces only bond when stacked (studs), never sideways. A sub-assembly that only
touches the rest sideways - a head next to the body, a floating 1x1 - hangs by
nothing or by a single stud. This module finds such places with the stud-contact graph
(see stability.py) and fixes them by re-tiling a small area: two same-layer neighbours
A (hanging side) and B (the rest) are replaced by one piece N that covers cells of both
plus a re-tiling of the remainder. N then sits on / under pieces of both sides.

Colours: N takes one colour; visible cells of the other colour change (at most
`max_recolor` per tie, hidden cells are free). Every change is accepted only if the
global score (loose pieces, big weak joints) improves, otherwise it is rolled back.
"""

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.model import PlacedBrick
from image_to_bricks.options import RepairOptions
from image_to_bricks.packing import cleanup
from image_to_bricks.packing.assembly import (
    Assembly,
    find_issues,
    piece_sizes,
    score,
    to_plates,
)
from image_to_bricks.packing.joint_retile import RetileBox, RetileJob, run_retile
from image_to_bricks.packing.layer_packer import WILD, StockUse, pack_layer_staggered
from image_to_bricks.stock import PLATE_HEIGHT_PLATES, Stock

MAX_TRIES = 60  # candidate replacements tried per issue
MAX_FIXES = 1000  # safety cap on accepted fixes
TIE_OPTIONS_PER_PAIR = 6  # best tie pieces kept per neighbour pair
GROW_OPTIONS_PER_PAIR = 3  # best grown rectangles kept per piece pair
GROWTH_REACH_STUDS = 3  # a piece above / below within this many studs sideways may be reached by growing
PLATES_PER_BRICK = 3
FIRST_PIECE, SECOND_PIECE = 1, 2  # owner codes of the two neighbours in a tie region
TOP_ISSUES_SHOWN = 5

type PieceRect = tuple[int, int, int, int, int]  # (x, y, w, d, colour)
type Rect = tuple[int, int, int, int]  # (x, y, w, d)
type Freed = tuple[int, int, int, int]  # (height in plates, w, d, colour) of a piece given back to the stock
type TieOption = tuple[int, int, PieceRect]  # (recoloured cells, -area, piece)
type Stats = dict[str, int]


def _need_after(asm: Assembly, height: int, rect: PieceRect, freed: Sequence[Freed]) -> int:
    """0 if piece rect = (x, y, w, d, colour) of this height is still owned once the pieces `freed`
    [(h, w, d, colour)] go back to the stock, else 1 (it would have to be bought)."""
    if asm.stock is None:
        return 0
    fk = asm.stock.fork()
    for f in freed:
        fk.give(*f)
    return 0 if fk.avail(height, rect[2], rect[3], rect[4]) else 1


def _split_need(asm: Assembly, piece: PlacedBrick) -> int:
    """A brick that gets split into 3 plates needs 3 plates of its size and colour: 1 if they are not owned."""
    if asm.stock is None or piece.h_plates == PLATE_HEIGHT_PLATES:
        return 0
    return 0 if asm.stock.free(PLATE_HEIGHT_PLATES, piece.w, piece.d, piece.color) >= PLATES_PER_BRICK else 1


@dataclass(frozen=True)
class _Problem:
    """One issue being fixed: the hanging side (pieces to tie to the rest) and what counts as the rest, from the
    contact analysis the issue came from."""

    asm: Assembly
    hanging: set[int]
    kind: str
    root: int
    comp: list[int]
    main: int

    def in_rest(self, j: int) -> bool:
        """True if piece j is on the side the hanging pieces should be tied to."""
        if self.kind == "loose":
            return True
        if self.kind == "footing":
            return j != self.root
        return self.comp[j] == self.main and j not in self.hanging


def _hanging_side(asm: Assembly, root: int, partner: int) -> set[int]:
    """Pieces reachable from root without crossing the root - partner joint."""
    seen, stack = {root}, [root]
    while stack:
        a = stack.pop()
        for n in asm.adj[a]:
            if (a == root and n == partner) or n in seen:
                continue
            seen.add(n)
            stack.append(n)
    return seen


def _pairs(problem: _Problem) -> dict[tuple[int, int, int], int]:
    """Lateral neighbours (a hanging, b in the rest) at layer z, by shared edge length: {(a, b, z): n}."""
    asm = problem.asm
    shared: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    ny, nx = asm.ny, asm.nx
    for a in problem.hanging:
        piece = asm.piece(a)
        sides = (
            [(piece.x - 1, y) for y in range(piece.y, piece.y + piece.d)]
            + [(piece.x + piece.w, y) for y in range(piece.y, piece.y + piece.d)]
            + [(x, piece.y - 1) for x in range(piece.x, piece.x + piece.w)]
            + [(x, piece.y + piece.d) for x in range(piece.x, piece.x + piece.w)]
        )
        for z in range(piece.z, piece.z + piece.h_plates):
            for x, y in sides:
                if 0 <= x < nx and 0 <= y < ny:
                    j = int(asm.owner[z, y, x])
                    if j >= 0 and j not in problem.hanging and problem.in_rest(j):
                        shared[(a, j, z)] += 1
    return shared


class _Region(NamedTuple):
    """Union box of two neighbours: corner, size, which piece owns each cell (1 / 2), cell colours, visibility."""

    x0: int
    y0: int
    h: int
    w: int
    reg: NDArray[np.int8]
    col: NDArray[np.int_]
    vis: NDArray[np.bool_]


def _options(
    asm: Assembly, first: PlacedBrick, second: PlacedBrick, visible: NDArray[np.bool_] | None, max_recolor: int
) -> tuple[list[TieOption], _Region]:
    """Pieces N covering cells of both pieces (same layer, same height) inside their union box:
    [(recoloured cells, -area, (x, y, w, d, colour))] sorted, and the region."""
    x0, y0 = min(first.x, second.x), min(first.y, second.y)
    x1, y1 = max(first.x + first.w, second.x + second.w), max(first.y + first.d, second.y + second.d)
    h, w = y1 - y0, x1 - x0
    reg = np.zeros((h, w), np.int8)
    reg[first.y - y0 : first.y - y0 + first.d, first.x - x0 : first.x - x0 + first.w] = FIRST_PIECE
    reg[second.y - y0 : second.y - y0 + second.d, second.x - x0 : second.x - x0 + second.w] = SECOND_PIECE
    col = np.where(reg == FIRST_PIECE, first.color, second.color)
    vis: NDArray[np.bool_]
    if visible is None:
        vis = np.ones((h, w), bool)
    else:
        vis = np.asarray(visible[first.z : first.z + first.h_plates, y0:y1, x0:x1].any(axis=0), dtype=np.bool_)
    opts: list[TieOption] = []
    for bw, bd in piece_sizes(asm):
        for yy in range(h - bd + 1):
            for xx in range(w - bw + 1):
                sub = reg[yy : yy + bd, xx : xx + bw]
                if (sub == 0).any() or not ((sub == FIRST_PIECE).any() and (sub == SECOND_PIECE).any()):
                    continue
                sv = vis[yy : yy + bd, xx : xx + bw]
                sc = col[yy : yy + bd, xx : xx + bw]
                colours = np.unique(sc[sv]) if sv.any() else np.array([first.color])
                best = min((int((sv & (sc != c)).sum()), -int(c == first.color), int(c)) for c in colours)
                if best[0] <= max_recolor:
                    opts.append((best[0], -bw * bd, (x0 + xx, y0 + yy, bw, bd, best[2])))
    opts.sort()
    return opts, _Region(x0, y0, h, w, reg, col, vis)


def _gain(problem: _Problem, rect: PieceRect, z: int, height: int, pair: tuple[int, int]) -> tuple[int, int]:
    """Studs a piece at rect would share with the hanging pieces and with the other side (above/below its
    footprint). A split brick's sibling plates count for its own side."""
    asm = problem.asm
    a, b = pair
    x, y, w, d = rect[:4]
    n1 = n2 = 0
    first, second = asm.piece(a), asm.piece(b)
    for zz in (z + height, z - 1):
        if not 0 <= zz < asm.nz:
            continue
        reg = asm.owner[zz, y : y + d, x : x + w]
        ids, cnt = np.unique(reg[reg >= 0], return_counts=True)
        for j, c in zip(ids.tolist(), cnt.tolist(), strict=True):
            if j == a:
                n1 += c if first.h_plates > 1 and first.z <= zz < first.z + first.h_plates else 0
            elif j == b:
                n2 += c if second.h_plates > 1 and second.z <= zz < second.z + second.h_plates else 0
            elif j in problem.hanging:
                n1 += c
            elif problem.in_rest(j):
                n2 += c
    return n1, n2


type Ints = int | NDArray[np.int64]


def _span_overlap(lo: Ints, length: Ints, t_lo: int, t_len: int) -> NDArray[np.int64]:
    """Overlap along one axis of [lo, lo + length) with [t_lo, t_lo + t_len); lo / length may be numpy arrays."""
    return np.asarray(np.maximum(np.minimum(lo + length, t_lo + t_len) - np.maximum(lo, t_lo), 0))


type GrowShapes = tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]
_GROW_SHAPES: dict[tuple[int, int, tuple[tuple[int, int], ...]], GrowShapes | None] = {}


def _grow_shapes(pw: int, pd: int, sizes: Sequence[tuple[int, int]]) -> GrowShapes | None:
    """All rectangles that contain a pw x pd footprint at its own corner offsets, as int arrays (w, d, ox, oy):
    the rectangle (x + ox, y + oy, w, d) contains the footprint at (x, y). One row per size in `sizes` order
    (duplicates kept) and offset; the footprint's own size is left out. Cached per (pw, pd, sizes)."""
    key = (pw, pd, tuple(sizes))
    if key not in _GROW_SHAPES:
        rows = [
            (w, d, ox, oy)
            for (w, d) in sizes
            if (w, d) != (pw, pd)
            for ox in range(pw - w, 1)
            for oy in range(pd - d, 1)
        ]
        if rows:
            w, d, ox, oy = (np.array(c, np.int64).reshape(-1) for c in zip(*rows, strict=True))
            _GROW_SHAPES[key] = (w, d, ox, oy)
        else:
            _GROW_SHAPES[key] = None
    return _GROW_SHAPES[key]


def _grow_options(asm: Assembly, piece: PlacedBrick, target: PlacedBrick) -> list[tuple[int, int, Rect]]:
    """Rectangles R >= footprint of piece (same layer/height) whose extra cells are all free interior
    cells and that overlap target (a piece right above/below the piece's layers) more than the piece does.
    Returns [(added_cells, -gain, (x, y, w, d))], sorted."""
    assert asm.grid is not None and asm.fill is not None, "growth needs the grid and the fill cells"
    shapes = _grow_shapes(piece.w, piece.d, piece_sizes(asm))
    if shapes is None:
        return []
    w, d, ox, oy = shapes
    x0, y0 = piece.x + ox, piece.y + oy
    inside = (x0 >= 0) & (y0 >= 0) & (x0 + w <= asm.nx) & (y0 + d <= asm.ny)
    w, d, x0, y0 = w[inside], d[inside], x0[inside], y0[inside]
    cur = int(_span_overlap(piece.x, piece.w, target.x, target.w) * _span_overlap(piece.y, piece.d, target.y, target.d))
    gain = _span_overlap(x0, w, target.x, target.w) * _span_overlap(y0, d, target.y, target.d) - cur
    zs = slice(piece.z, piece.z + piece.h_plates)
    usable = ((~asm.grid[zs] & asm.fill[zs]) | (asm.owner[zs] == asm.owner[piece.z, piece.y, piece.x])).all(axis=0)
    # summed-area table: a rectangle is usable iff its sum of usable cells equals its area
    sat = np.zeros((asm.ny + 1, asm.nx + 1), np.int64)
    sat[1:, 1:] = usable.cumsum(0).cumsum(1)
    filled = sat[y0 + d, x0 + w] - sat[y0, x0 + w] - sat[y0 + d, x0] + sat[y0, x0]
    keep = np.nonzero((gain > 0) & (filled == w * d))[0]
    out = [
        (int(w[k] * d[k]) - piece.w * piece.d, -int(gain[k]), (int(x0[k]), int(y0[k]), int(w[k]), int(d[k])))
        for k in keep
    ]
    out.sort()
    return out


def _growth_pairs(problem: _Problem, reach: int = GROWTH_REACH_STUDS) -> set[tuple[int, int]]:
    """(P, T) piece pairs on different sides, T directly above/below P's layers and within `reach`
    studs sideways: {(a, b)} with a hanging, b on the other side."""
    asm = problem.asm
    pairs = set()
    for a in problem.hanging:
        lower = asm.piece(a)
        for zz in (lower.z - 1, lower.z + lower.h_plates):
            if not 0 <= zz < asm.nz:
                continue
            reg = asm.owner[
                zz,
                max(0, lower.y - reach) : lower.y + lower.d + reach,
                max(0, lower.x - reach) : lower.x + lower.w + reach,
            ]
            for j in np.unique(reg[reg >= 0]).tolist():
                if j in problem.hanging or not problem.in_rest(j):
                    continue
                other = asm.piece(j)
                if (zz < lower.z and other.z + other.h_plates == lower.z) or (
                    zz >= lower.z + lower.h_plates and other.z == lower.z + lower.h_plates
                ):
                    pairs.add((a, j))
    return pairs


def _replace(asm: Assembly, a: int, b: int, option: PieceRect, region: _Region) -> list[int]:
    """Swap pieces a and b for the option piece + a tiling of the remainder; returns the new piece ids."""
    x0, y0, _, _, reg, col, vis = region
    first = asm.piece(a)
    z, hp = first.z, first.h_plates
    nx_, ny_, nw, nd, nc = option
    rem = reg > 0
    rem[ny_ - y0 : ny_ - y0 + nd, nx_ - x0 : nx_ - x0 + nw] = False
    colors = np.where(vis, col, WILD).astype(np.uint8)
    asm.remove(a)
    asm.remove(b)
    ids = [asm.add(PlacedBrick(nx_, ny_, z, nw, nd, hp, nc))]
    if rem.any():
        stock_use = StockUse(asm.stock.fork(), hp, first.color) if asm.stock is not None else None
        for x, y, bw, bd, c in pack_layer_staggered(rem, colors, stock_use=stock_use):
            ids.append(asm.add(PlacedBrick(x0 + x, y0 + y, z, bw, bd, hp, first.color if c is None else c)))
    return ids


class _Candidate(NamedTuple):
    """A repair to try: rank (sorted ascending), kind "tie" (pieces first + second at layer z become the piece
    rect in colour + a re-tiling of region, recolouring rank[1] cells) or "grow" (piece first grows to rect)."""

    rank: tuple[bool, int, int, int, int, int, int]
    kind: str
    first: int
    second: int
    z: int
    rect: Rect
    colour: int
    region: _Region | None


def _as_plate(piece: PlacedBrick, z: int) -> PlacedBrick:
    """The piece itself if it is a plate, else its plate at layer z."""
    if piece.h_plates == PLATE_HEIGHT_PLATES:
        return piece
    return PlacedBrick(piece.x, piece.y, z, piece.w, piece.d, PLATE_HEIGHT_PLATES, piece.color)


@dataclass(frozen=True)
class _TieJob:
    """One tie-repair phase: visible cells (None = all), visible cells a tie may recolour, whether pieces may
    grow into the hollow interior, smallest hanging group that counts."""

    asm: Assembly
    visible: NDArray[np.bool_] | None
    recolor: int
    hang_min: int
    grow: bool = field(default=True, kw_only=True)

    def tie_candidates(self, problem: _Problem) -> list[_Candidate]:
        """One piece spanning a hanging piece and a neighbour of the rest (up to TIE_OPTIONS_PER_PAIR per pair)."""
        asm = self.asm
        cands = []
        for (a, b, z), edge in _pairs(problem).items():
            first, second = asm.piece(a), asm.piece(b)
            if first.h_plates != second.h_plates:  # brick next to a plate: look at the plate layer
                pa, pb = _as_plate(first, z), _as_plate(second, z)
            else:
                pa, pb = first, second
            opts, region = _options(asm, pa, pb, self.visible, self.recolor)
            kept = 0
            for cost, _, rect in opts:
                n1, n2 = _gain(problem, rect, z, pa.h_plates, (a, b))
                if n2 == 0 or (n1 == 0 and len(problem.hanging) > 1):
                    continue  # N would not touch the other side / would drop its own links
                freed = (
                    [
                        (first.h_plates, first.w, first.d, first.color),
                        (second.h_plates, second.w, second.d, second.color),
                    ]
                    if first.h_plates == second.h_plates
                    else []
                )
                need = _need_after(asm, pa.h_plates, rect, freed) + _split_need(asm, first) + _split_need(asm, second)
                rank = (problem.comp[b] != problem.main, cost, need, 0, 0, -min(n1, n2), -edge)
                cands.append(_Candidate(rank, "tie", a, b, z, rect[:4], rect[4], region))
                kept += 1
                if kept == TIE_OPTIONS_PER_PAIR:
                    break
        return cands

    def grow_candidates(self, problem: _Problem) -> list[_Candidate]:
        """A piece of either side grown into the hollow interior to widen its contact with the other side."""
        asm = self.asm
        cands = []
        for a, b in _growth_pairs(problem):
            for grown, target, grown_id in ((asm.piece(a), asm.piece(b), a), (asm.piece(b), asm.piece(a), b)):
                for added, ngain, rect in _grow_options(asm, grown, target)[:GROW_OPTIONS_PER_PAIR]:
                    need = _need_after(
                        asm,
                        grown.h_plates,
                        (rect[0], rect[1], rect[2], rect[3], grown.color),
                        [(grown.h_plates, grown.w, grown.d, grown.color)],
                    )
                    rank = (problem.comp[b] != problem.main, 0, need, 1, added, ngain, 0)
                    cands.append(_Candidate(rank, "grow", grown_id, -1, -1, rect, -1, None))
        return cands

    def apply(self, candidate: _Candidate) -> int:
        """Make the change (the caller logs / rolls back); returns the visible cells recoloured."""
        asm = self.asm
        if candidate.kind == "grow":
            asm.grow(candidate.first, candidate.rect)
            return 0
        assert candidate.region is not None, "a tie candidate has its region"
        a = to_plates(asm, candidate.first, candidate.z)
        b = to_plates(asm, candidate.second, candidate.z)
        _replace(asm, a, b, (*candidate.rect, candidate.colour), candidate.region)
        return candidate.rank[1]

    def fix_one(self, problem: _Problem, base: tuple[int, ...]) -> tuple[int, str] | None:
        """Try candidate repairs for one problem; keeps the first that improves the score.
        Returns (visible cells recoloured, kind), or None if nothing worked."""
        asm = self.asm
        cands = self.tie_candidates(problem)
        if self.grow and asm.fill is not None:  # widen the contact by growing a piece into the hollow interior
            cands += self.grow_candidates(problem)
        cands.sort(key=lambda c: c.rank)
        owned0 = asm.stock.owned_used() if asm.stock is not None else 0
        # with stock: first only repairs that keep every owned piece in use (e.g. don't split an owned
        # brick into bought plates); the structure matters more, so any improving repair is the fallback
        for strict in (True, False) if asm.stock is not None else (False,):
            for candidate in cands[:MAX_TRIES]:
                asm.log = []
                cost = self.apply(candidate)
                keeps_owned = asm.stock is None or asm.stock.owned_used() >= owned0
                if score(asm, self.hang_min) < base and not (strict and not keeps_owned):
                    asm.log = None
                    return cost, candidate.kind
                asm.rollback()
        return None

    def run_phase(self, stats: Stats) -> None:
        """Fix the biggest issue first until none can be fixed (MAX_FIXES at most); counts in stats."""
        asm = self.asm
        failed: set[tuple[str, int, int, int]] = set()

        def failure_key(kind: str, root: int) -> tuple[str, int, int, int]:
            piece = asm.piece(root)
            return (kind, piece.x, piece.y, piece.z)

        for _ in range(MAX_FIXES):
            comp, main, issues = find_issues(asm, self.hang_min)
            issues = [t for t in issues if failure_key(t.kind, t.root) not in failed]
            if not issues:
                break
            issues.sort(key=lambda t: -t.size)
            issue = issues[0]
            if issue.kind == "loose":
                hanging = set(issue.extra)
            elif issue.kind == "footing":
                hanging = {issue.root}
            else:
                root, partner = issue.extra
                hanging = _hanging_side(asm, root, partner)
            problem = _Problem(asm, hanging, issue.kind, issue.root, comp, main)
            res = self.fix_one(problem, score(asm, self.hang_min))
            if res is None:
                failed.add(failure_key(issue.kind, issue.root))
            else:
                stats[res[1]] += 1
                stats["recolored"] += res[0]
                failed.clear()  # the assembly changed: earlier failures may work now


def _remerge(asm: Assembly) -> int:
    """Stock mode: three identical plates stacked on each other (same footprint and colour) become one
    brick again - the stud contacts to the outside are the same - when that uses more owned pieces or
    leaves fewer pieces to buy (e.g. owned bricks that tie repair had split into plates). Returns the merges."""
    stock = asm.stock
    assert stock is not None, "re-merging needs the stock"
    merged = 0
    for i in range(len(asm.bricks)):
        a = asm.bricks[i]
        if a is None or a.h_plates != PLATE_HEIGHT_PLATES or a.z + 2 >= asm.nz:
            continue
        j, k = int(asm.owner[a.z + 1, a.y, a.x]), int(asm.owner[a.z + 2, a.y, a.x])
        if j < 0 or k < 0:
            continue
        b, c = asm.piece(j), asm.piece(k)
        if any((p.h_plates, p.x, p.y, p.w, p.d, p.color) != (1, a.x, a.y, a.w, a.d, a.color) for p in (b, c)):
            continue
        before = (stock.owned_used(), -stock.need_total())
        asm.log = []
        for t in (i, j, k):
            asm.remove(t)
        asm.add(PlacedBrick(a.x, a.y, a.z, a.w, a.d, PLATES_PER_BRICK, a.color))
        if (stock.owned_used(), -stock.need_total()) > before:
            asm.log = None
            merged += 1
        else:
            asm.rollback()
    return merged


class _Progress:
    """Console progress of tie repair (one line per phase with its time and the fixes so far), so a long run
    does not look stuck. Silent when verbose is False."""

    def __init__(self, stats: Stats, *, verbose: bool) -> None:
        self.verbose, self.stats, self.t = verbose, stats, 0.0

    def start(self, message: str) -> None:
        """Print the start line and start the clock."""
        self.t = time.time()
        if self.verbose:
            print(message, flush=True)

    def done(self, phase: str) -> None:
        """One line: the phase, its time and the fixes so far."""
        now = time.time()
        if self.verbose:
            s = self.stats
            print(
                f"  ... {phase}: done in {now - self.t:.0f} s (so far {s['tie']} ties, {s['grow']} grown, "
                f"{s['retile']} re-tilings)",
                flush=True,
            )
        self.t = now


@dataclass(frozen=True)
class RepairTarget:
    """Where to repair: the model grid [z, y, x] bool, visible cells (None = all count as visible), fill = cells
    hollowing removed (pieces may grow into them; None = no growth), the base layers at the bottom and the retile
    box (z0, z1, y0, y1, x0, x1) in this grid: only joints whose root piece starts inside are re-tiled (None = all)."""

    grid: NDArray[np.bool_]
    visible: NDArray[np.bool_] | None = None
    fill: NDArray[np.bool_] | None = None
    base_layers: int = 0
    retile_box: RetileBox | None = None


def _drop_loose_groups(asm: Assembly, drop_max: int) -> list[PlacedBrick]:
    """Remove the loose groups of at most drop_max pieces; returns the removed pieces."""
    dropped: list[PlacedBrick] = []
    if drop_max <= 0:
        return dropped
    comp, _, main, _ = asm.analyse()
    for mem in asm.loose_groups(comp, main).values():
        if len(mem) <= drop_max:
            for i in mem:
                dropped.append(asm.piece(i))
                asm.remove(i)
    return dropped


@dataclass(frozen=True)
class _RepairReport:
    """What repair prints at the end."""

    stats: Stats
    dropped: list[PlacedBrick]
    merged: int
    before: tuple[int, ...]
    after: tuple[int, ...]


def _print_report(asm: Assembly, report: _RepairReport, options: RepairOptions, base_layers: int) -> None:
    """Summary lines and the VERDICT."""
    stats, dropped, before, after = report.stats, report.dropped, report.before, report.after
    hang_min = options.tie_min_hanging_pieces
    print(
        f"Tie repair: {stats['tie']} ties (re-tiling), {stats['grow']} pieces grown into the hollow interior, "
        f"{stats['recolored']} visible cell(s) recoloured, {len(dropped)} floating piece(s) removed"
    )
    if options.retile and options.tie_repair:
        print(f"  re-tiling around weak joints: {stats['retile']} change(s) kept")
    for b in dropped:
        print(f"  removed floating piece: layer {b.z + 1}, x {b.x}, y {b.y}, {b.w}x{b.d}, colour {b.color}")
    if asm.stock is not None:
        print(f"  stock: {report.merged} stack(s) of 3 plates merged into a brick")
    print(
        f"  loose pieces {before[0]} -> {after[0]}, single-stud joints hanging >= {hang_min} pieces "
        f"{before[2]} -> {after[2]}"
    )
    if base_layers:
        print(f"  single-stud footings on the base {before[4]} -> {after[4]}")
    if after[0] == 0 and after[2] == 0 and after[4] == 0:
        print(
            "  VERDICT: structurally sound (one connected assembly, no large single-stud joints"
            + (", no single-stud footings)" if base_layers else ")")
        )
        return
    _, _, issues = find_issues(asm, hang_min)
    print(
        f"  VERDICT: {after[0]} loose piece(s), {after[2]} large single-stud joint(s)"
        + (f", {after[4]} single-stud footing(s)" if base_layers else "")
        + " left; biggest:"
    )
    for issue in sorted(issues, key=lambda t: -t.size)[:TOP_ISSUES_SHOWN]:
        b = asm.piece(issue.root)
        print(f"    {issue.kind}: {issue.size} pieces at layer {b.z + 1}, x {b.x}, y {b.y}")


def _run_tie_phases(asm: Assembly, target: RepairTarget, options: RepairOptions, progress: _Progress) -> None:
    """Ties without recolouring + growth, ties that recolour, then re-tiling around the joints left."""
    hang_min, max_recolor = options.tie_min_hanging_pieces, options.tie_max_recoloured_cells
    stats = progress.stats
    _TieJob(asm, target.visible, 0, hang_min, grow=True).run_phase(stats)
    progress.done("ties without recolouring + growth")
    if max_recolor > 0:
        _TieJob(asm, target.visible, max_recolor, hang_min, grow=True).run_phase(stats)
        progress.done(f"ties recolouring up to {max_recolor} visible cell(s)")
    if options.retile:
        drop_max = min(options.retile_drop_max_pieces, options.tie_drop_max_pieces)
        job = RetileJob(asm, target.visible, max_recolor, hang_min, drop_max)
        run_retile(job, target.retile_box, stats, max(1, options.retile_reach_studs))
        progress.done("re-tiling around weak joints")


DEFAULT_REPAIR = RepairOptions()


def repair(
    target: RepairTarget,
    bricks: Sequence[PlacedBrick],
    options: RepairOptions = DEFAULT_REPAIR,
    stock: Stock | None = None,
    *,
    verbose: bool = True,
) -> tuple[list[PlacedBrick], NDArray[np.bool_]]:
    """Returns (bricks, grid). With options.tie_repair, fixes in order of visibility: (1) re-tilings that change no
    colour and growth of pieces into the hollowed-out interior (target.fill), (2) re-tilings that recolour up to
    tie_max_recoloured_cells visible cells, (3) with options.retile, re-tiling around the single-stud joints that
    are left (see joint_retile; a re-tiling may leave retile_drop_max_pieces floating), then (4) removal of loose
    groups of at most tie_drop_max_pieces pieces that touch nothing (they would float in the air). With
    cleanup_small, the lossless 1x1 clean-up (cleanup.cleanup) runs next; with a stock, stacks of 3 plates
    become bricks again where that uses owned pieces. stock: the ledger, already counting `bricks`."""
    hang_min = options.tie_min_hanging_pieces
    grid = target.grid
    asm = Assembly(grid.shape, bricks, grid, target.fill, target.base_layers)
    asm.stock = stock  # ledger already counts `bricks`; from here every add / remove / rollback updates it
    before = score(asm, hang_min)
    stats: Stats = {"tie": 0, "grow": 0, "recolored": 0, "retile": 0}
    progress = _Progress(stats, verbose=verbose)
    progress.start(
        f"Tie repair started: {len(bricks)} pieces, {before[0]} loose, {before[2]} single-stud joint(s) "
        f"hanging >= {hang_min} pieces" + (f", {before[4]} single-stud footing(s)" if target.base_layers else "")
    )
    if options.tie_repair:
        _run_tie_phases(asm, target, options, progress)
    dropped = _drop_loose_groups(asm, options.tie_drop_max_pieces if options.tie_repair else 0)
    if options.cleanup_small:
        cleanup.cleanup(
            asm,
            hang_min,
            verbose=verbose,
            extra_stacks=options.cleanup_extra_stacks,
            extra_seams=options.cleanup_extra_seams,
        )
        progress.done("clean-up of 1x1 pieces")
    merged = _remerge(asm) if asm.stock is not None else 0
    out_grid = asm.grid if asm.grid is not None else grid.copy()
    for b in dropped:
        out_grid[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] = False
    after = score(asm, hang_min)
    out = [b for b in asm.bricks if b is not None]
    if verbose:
        _print_report(asm, _RepairReport(stats, dropped, merged, before, after), options, target.base_layers)
    return out, out_grid
