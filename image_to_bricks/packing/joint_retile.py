"""Re-tiling around the weak joints the pair-wise ties could not fix (--retile, part of tie repair).

All pieces of one plate layer in a small window around the joint are re-tiled with any mix of sizes (so a 2-wide
plate can span the two sides); random tilings are sampled, coloured by their cells' majority colour within the
recolour budget, and the one with the best global score is kept."""

import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.model import PlacedBrick
from image_to_bricks.packing.assembly import (
    Assembly,
    Issue,
    find_issues,
    piece_sizes,
    score_drop,
    to_plates,
)
from image_to_bricks.stock import PLATE_HEIGHT_PLATES

RETILE_MAX_CELLS = 30  # largest region (cells of one layer) that is re-tiled
RETILE_SAMPLES = 400  # random tilings drawn per region
RETILE_EVAL = 150  # best-ranked tilings scored per region
MAX_FIXES = 1000  # safety cap on accepted re-tilings
TILING_DRAWS_PER_RESULT = 4  # draws allowed per wanted distinct tiling
SIZE_WEIGHT_POWER = 1.5  # a bigger piece is drawn with weight area ** this

type Stud = tuple[int, int]  # (x, y)
type Rect = tuple[int, int, int, int]  # (x, y, w, d)
type ColouredRect = tuple[int, int, int, int, int]  # (x, y, w, d, colour)
type RetileBox = tuple[int, ...]  # (z0, z1, y0, y1, x0, x1), ends exclusive


def _fitting_rects(
    cells: set[Stud], sizes: Sequence[tuple[int, int]]
) -> dict[Stud, list[tuple[int, int, tuple[Stud, ...]]]]:
    """{(x, y): [(w, d, cells of the w x d rectangle at (x, y))]} for the sizes that lie inside `cells`, in
    `sizes` order. Computed once per region; a tiling draw then only checks the rectangles against what it has
    already covered."""
    fits: dict[Stud, list[tuple[int, int, tuple[Stud, ...]]]] = {}
    for x, y in cells:
        fits[(x, y)] = []
        for w, d in sizes:
            cs = tuple((xx, yy) for yy in range(y, y + d) for xx in range(x, x + w))
            if all(c in cells for c in cs):
                fits[(x, y)].append((w, d, cs))
    return fits


def sample_tilings(cells: set[Stud], sizes: Sequence[tuple[int, int]], n: int, seed: int = 7) -> list[list[Rect]]:
    """Up to n different tilings of `cells` {(x, y)} by rectangles of `sizes`: [[(x, y, w, d)]].
    The first is the greedy 'largest first' tiling, the rest are random (bigger pieces more likely)."""
    rng = random.Random(seed)
    order = sorted(cells, key=lambda p: (p[1], p[0]))
    fits = _fitting_rects(cells, sizes)
    seen: set[frozenset[Rect]] = set()
    out: list[list[Rect]] = []
    for draw in range(n * TILING_DRAWS_PER_RESULT):
        if len(out) >= n:
            break
        covered: set[Stud] = set()
        rects: list[Rect] = []
        for x, y in order:
            if (x, y) in covered:
                continue
            opts = [(w, d, cs) for (w, d, cs) in fits[(x, y)] if covered.isdisjoint(cs)]
            if draw == 0:
                w, d, cs = max(opts, key=lambda o: o[0] * o[1])
            else:
                w, d, cs = rng.choices(opts, weights=[(o[0] * o[1]) ** SIZE_WEIGHT_POWER for o in opts])[0]
            rects.append((x, y, w, d))
            covered.update(cs)
        key = frozenset(rects)
        if key not in seen:
            seen.add(key)
            out.append(rects)
    return out


def colour_tiling(
    rects: Iterable[Rect], col: dict[Stud, int], vis: dict[Stud, bool], budget: int
) -> tuple[list[ColouredRect], int] | None:
    """Colour each rectangle with the majority colour of its visible cells (hidden cells are free).
    Returns ([(x, y, w, d, colour)], recoloured visible cells) or None if over budget."""
    out: list[ColouredRect] = []
    cost = 0
    for x, y, w, d in rects:
        cs = [(xx, yy) for yy in range(y, y + d) for xx in range(x, x + w)]
        vc = [col[c] for c in cs if vis[c]]
        pool = vc if vc else [col[c] for c in cs]
        best, n = max(Counter(pool).items(), key=lambda t: (t[1], t[0] == col[(x, y)], -t[0]))
        cost += len(vc) - n if vc else 0
        if cost > budget:
            return None
        out.append((x, y, w, d, best))
    return out, cost


def _region_ids(asm: Assembly, z: int, centre: Stud, reach: int) -> set[int]:
    """Pieces with a cell at layer z within `reach` studs of centre."""
    cx, cy = centre
    ids = set()
    for y in range(max(0, cy - reach), min(asm.ny, cy + reach + 1)):
        for x in range(max(0, cx - reach), min(asm.nx, cx + reach + 1)):
            o = int(asm.owner[z, y, x])
            if o >= 0:
                ids.add(o)
    return ids


def _apply_tiling(asm: Assembly, ids: Iterable[int], z: int, tiling: Iterable[ColouredRect]) -> None:
    """Replace the pieces `ids` (bricks are split into plates first) by the tiling at plate layer z."""
    for i in sorted(ids):
        piece = asm.bricks[i]
        if piece is not None and piece.h_plates > PLATE_HEIGHT_PLATES:
            to_plates(asm, i, z)
    cells = {(x, y) for (x0, y0, w, d, _) in tiling for y in range(y0, y0 + d) for x in range(x0, x0 + w)}
    for x, y in cells:
        o = int(asm.owner[z, y, x])
        if o >= 0:
            asm.remove(o)
    for x, y, w, d, c in tiling:
        asm.add(PlacedBrick(x, y, z, w, d, PLATE_HEIGHT_PLATES, c))


type RetileKey = tuple[tuple[int, ...], tuple[int, ...], int, int]  # (score, stock, recoloured, pieces)


@dataclass(frozen=True)
class RetileResult:
    """Best re-tiling found: its ranking key, layer, tiling, the pieces it replaces and the recoloured cells."""

    key: RetileKey
    z: int
    tiling: list[ColouredRect]
    ids: set[int]
    cost: int


@dataclass(frozen=True)
class RetileJob:
    """One re-tiling pass: the assembly, visible cells (None = all), recolour budget per re-tiling, smallest
    hanging group that counts and how many pieces a re-tiling may leave floating."""

    asm: Assembly
    visible: NDArray[np.bool_] | None
    budget: int
    hang_min: int
    drop_max: int

    def try_retile(self, z: int, centre: Stud, reach: int, base: tuple[int, ...]) -> RetileResult | None:
        """Best re-tiling of the pieces around stud centre (x, y) at plate layer z that improves `base`, or None."""
        asm = self.asm
        ids = _region_ids(asm, z, centre, reach)
        cells: dict[Stud, int] = {}
        for i in ids:
            b = asm.piece(i)
            for y in range(b.y, b.y + b.d):
                for x in range(b.x, b.x + b.w):
                    cells[(x, y)] = b.color
        if not cells or len(cells) > RETILE_MAX_CELLS:
            return None
        visible = self.visible
        vis = {c: True if visible is None else bool(visible[z, c[1], c[0]]) for c in cells}
        cands = []
        for rects in sample_tilings(set(cells), piece_sizes(asm), RETILE_SAMPLES):
            r = colour_tiling(rects, cells, vis, self.budget)
            if r is not None:
                cands.append((r[1], len(rects), r[0]))
        cands.sort(key=lambda t: t[:2])
        best: RetileResult | None = None
        for cost, piece_count, tiling in cands[:RETILE_EVAL]:
            asm.log = []
            _apply_tiling(asm, ids, z, tiling)
            sc = score_drop(asm, self.hang_min, self.drop_max)
            st = (asm.stock.need_total(), -asm.stock.owned_used()) if asm.stock is not None else ()
            asm.rollback()
            if sc < base:
                key = (sc, st, cost, piece_count)
                if best is None or key < best.key:
                    best = RetileResult(key, z, tiling, ids, cost)
        return best


def _in_box(b: PlacedBrick, box: RetileBox | None) -> bool:
    """True if the piece starts inside the box (or there is no box)."""
    if box is None:
        return True
    z0, z1, y0, y1, x0, x1 = box
    return z0 <= b.z < z1 and y0 <= b.y < y1 and x0 <= b.x < x1


def _joint_layers_and_centre(asm: Assembly, issue: Issue) -> tuple[int, int, Stud]:
    """(lower layer, upper layer, stud) where the joint sits: a weak joint's contact (first overlapping stud) or a
    single-stud footing (the foot layer and the top base layer under it)."""
    if issue.kind == "weak":
        root, partner = issue.extra
        r, p = asm.piece(root), asm.piece(partner)
        lo, hi = (r, p) if r.z < p.z else (p, r)
        ox = range(max(r.x, p.x), min(r.x + r.w, p.x + p.w))
        oy = range(max(r.y, p.y), min(r.y + r.d, p.y + p.d))
        return lo.z + lo.h_plates - 1, hi.z, (ox[0], oy[0]) if ox and oy else (r.x, r.y)
    r = asm.piece(issue.root)
    return r.z - 1, r.z, (r.x, r.y)


def _failure_key(asm: Assembly, issue: Issue) -> tuple[int, int, int]:
    """Where an issue's joint piece is (its retry is skipped until the assembly changes)."""
    r = asm.piece(issue.root)
    return (r.x, r.y, r.z)


def run_retile(job: RetileJob, box: RetileBox | None, stats: dict[str, int], reach_studs: int) -> None:
    """Weak joints and single-stud footings the pair-wise ties could not fix: re-tile ALL pieces of one layer
    around the joint, trying the layer above and below the contact and growing windows (1..reach_studs); the
    tiling with the best global score is kept (box = only joints whose root is inside). Counts in stats."""
    asm = job.asm
    failed: set[tuple[int, int, int]] = set()
    for _ in range(MAX_FIXES):
        _, _, issues = find_issues(asm, job.hang_min)
        issues = [
            t
            for t in issues
            if t.kind in ("weak", "footing") and _in_box(asm.piece(t.root), box) and _failure_key(asm, t) not in failed
        ]
        if not issues:
            break
        issues.sort(key=lambda t: -t.size)
        issue = issues[0]
        z_low, z_high, centre = _joint_layers_and_centre(asm, issue)
        base = score_drop(asm, job.hang_min, job.drop_max)
        done = None
        for reach in range(1, reach_studs + 1):  # window around the joint, 1..N studs (tried in this order)
            res = [r for r in (job.try_retile(z, centre, reach, base) for z in (z_high, z_low)) if r is not None]
            if res:
                done = min(res, key=lambda r: r.key)
                break
        if done is None:
            failed.add(_failure_key(asm, issue))
            continue
        asm.log = None
        _apply_tiling(asm, done.ids, done.z, done.tiling)
        stats["retile"] += 1
        stats["recolored"] += done.cost
        failed.clear()
