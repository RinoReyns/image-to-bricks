"""The editable piece model used by tie repair and the clean-up: pieces, owner grid, stud-contact graph with cheap
add / remove / rollback, and the structural score and issues (loose groups, single-stud joints, footings)."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from image_to_bricks.analysis.stability import Bridge, analyse_graph
from image_to_bricks.model import PlacedBrick
from image_to_bricks.stock import PLATE_HEIGHT_PLATES, Stock

SIZES = [(4, 4), (2, 4), (4, 2), (2, 3), (3, 2), (2, 2), (1, 4), (4, 1), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1)]

type Cells = tuple[slice, slice, slice]  # (z, y, x) slices of a piece's cells


class Assembly:
    """Pieces + owner grid + stud-contact graph with cheap replace / rollback."""

    def __init__(
        self,
        shape: tuple[int, ...],
        bricks: Iterable[PlacedBrick],
        grid: NDArray[np.bool_] | None = None,
        fill: NDArray[np.bool_] | None = None,
        base_layers: int = 0,
    ) -> None:
        """Assembly of `bricks` in a grid of `shape` (z, y, x); grid / fill are copied / kept for grow."""
        self.nb = base_layers
        self.nz, self.ny, self.nx = shape
        self.owner: NDArray[np.int32] = np.full(shape, -1, np.int32)
        self.grid = None if grid is None else grid.copy()
        self.fill = fill  # cells that may be added (hollowed-out interior), or None
        self.stock: Stock | None = None  # ledger kept in step with the pieces (set after loading)
        self.bricks: list[PlacedBrick | None] = []  # None = removed (indices stay stable)
        self.adj: list[dict[int, int]] = []
        self.log: list[tuple[Any, ...]] | None = None  # changes since a checkpoint (None = not recording)
        for b in bricks:
            self.add(b)

    def piece(self, i: int) -> PlacedBrick:
        """Piece i, which must not be removed."""
        b = self.bricks[i]
        assert b is not None, f"piece {i} was removed"
        return b

    def _cells(self, b: PlacedBrick) -> Cells:
        return (slice(b.z, b.z + b.h_plates), slice(b.y, b.y + b.d), slice(b.x, b.x + b.w))

    def _link(self, i: int) -> None:
        b = self.piece(i)
        for z in (b.z + b.h_plates, b.z - 1):
            if 0 <= z < self.nz:
                reg = self.owner[z, b.y : b.y + b.d, b.x : b.x + b.w]
                ids, cnt = np.unique(reg[reg >= 0], return_counts=True)
                for j, c in zip(ids.tolist(), cnt.tolist(), strict=True):
                    self.adj[i][j] = c
                    self.adj[j][i] = c

    def add(self, b: PlacedBrick) -> int:
        """Place piece b (stock taken, change logged); returns its index."""
        i = len(self.bricks)
        self.bricks.append(b)
        self.adj.append({})
        self.owner[self._cells(b)] = i
        self._link(i)
        if self.stock is not None:
            self.stock.take_brick(b)
        if self.log is not None:
            self.log.append(("add", i))
        return i

    def _unlink(self, i: int) -> None:
        for j in list(self.adj[i]):
            del self.adj[j][i]
        self.adj[i] = {}

    def remove(self, i: int) -> None:
        """Remove piece i (its slot becomes None, stock given back, change logged)."""
        b = self.piece(i)
        self._unlink(i)
        self.owner[self._cells(b)] = -1
        self.bricks[i] = None
        if self.stock is not None:
            self.stock.give_brick(b)
        if self.log is not None:
            self.log.append(("remove", i, b))

    def grow(self, i: int, rect: tuple[int, int, int, int]) -> int:
        """Replace piece i by the bigger rectangle (x0, y0, w, d) (same layer/height/colour); the extra cells
        come out of the hollowed-out interior (must be free fill cells). Returns the new piece's index."""
        assert self.grid is not None, "grow needs the grid"
        x0, y0, w, d = rect
        b = self.piece(i)
        sub = (slice(b.z, b.z + b.h_plates), slice(y0, y0 + d), slice(x0, x0 + w))
        new = ~self.grid[sub]
        self.remove(i)
        self.grid[sub] = True
        if self.log is not None:
            self.log.append(("grid", sub, new))
        return self.add(PlacedBrick(x0, y0, b.z, w, d, b.h_plates, b.color))

    def rollback(self) -> None:
        """Undo every change logged since `log` was set to [] and stop logging."""
        assert self.log is not None, "rollback needs a checkpoint (log = [])"
        for e in reversed(self.log):
            if e[0] == "grid":
                assert self.grid is not None
                self.grid[e[1]][e[2]] = False
            elif e[0] == "add":
                i = e[1]
                self._unlink(i)
                added = self.piece(i)
                self.owner[self._cells(added)] = -1
                if self.stock is not None:
                    self.stock.give_brick(added)
                self.bricks.pop()
                self.adj.pop()
            else:
                _, i, b = e
                self.bricks[i] = b
                self.owner[self._cells(b)] = i
                if self.stock is not None:
                    self.stock.take_brick(b)
                self._link(i)
        self.log = None

    def analyse(self) -> tuple[list[int], dict[int, int], int, list[Bridge]]:
        """(component per piece, {live component: pieces}, main component, bridges) of the contact graph."""
        comp, sizes, bridges = analyse_graph(self.adj)
        alive = {c: n for c, n in sizes.items() if self.bricks[c] is not None}
        main = max(alive, key=lambda c: alive[c])
        return comp, alive, main, bridges

    def loose_groups(self, comp: list[int], main: int) -> dict[int, list[int]]:
        """{component: its live pieces} for every component except the main one (first piece order)."""
        groups: defaultdict[int, list[int]] = defaultdict(list)
        for i, b in enumerate(self.bricks):
            if b is not None and comp[i] != main:
                groups[comp[i]].append(i)
        return groups


def piece_sizes(asm: Assembly) -> list[tuple[int, int]]:
    """Piece footprints to consider: the standard ones, plus the larger sizes in the stock."""
    return SIZES if asm.stock is None else asm.stock.sizes_with(SIZES)


def is_footing(asm: Assembly, i: int) -> bool:
    """First figure layer piece that stands on the base with a single stud."""
    b = asm.bricks[i]
    return (
        asm.nb > 0
        and b is not None
        and b.z == asm.nb
        and sum(w for j, w in asm.adj[i].items() if asm.piece(j).z < asm.nb) == 1
    )


type Analysis = tuple[list[int], dict[int, int], int, list[Bridge]]  # Assembly.analyse()


def _weak_joint_sizes(asm: Assembly, analysis: Analysis, hang_min: int) -> list[int]:
    """Pieces on the smaller side of every single-stud joint of the main assembly that hangs >= hang_min."""
    comp, alive, main, bridges = analysis
    return [
        min(su, alive[main] - su)
        for p, u, su in bridges
        if comp[p] == main and asm.adj[p][u] <= 1 and min(su, alive[main] - su) >= hang_min
    ]


def _footing_count(asm: Assembly) -> int:
    """Single-stud footings on the base (0 without a base)."""
    return sum(1 for i in range(len(asm.bricks)) if is_footing(asm, i)) if asm.nb else 0


def score(asm: Assembly, hang_min: int) -> tuple[int, int, int, int, int]:
    """Lower is better: (loose pieces, loose groups, single-stud hangs of >= hang_min pieces, their pieces, feet)."""
    analysis = asm.analyse()
    _, alive, main, _ = analysis
    loose = sum(n for c, n in alive.items() if c != main)
    big = _weak_joint_sizes(asm, analysis, hang_min)
    return (loose, len(alive) - 1, len(big), sum(big), _footing_count(asm))


def score_drop(asm: Assembly, hang_min: int, drop_max: int) -> tuple[int, int, int, int, int, int]:
    """Like score, but loose groups of <= drop_max pieces do not count as loose (they are removed at the end);
    their pieces are the last term, so tilings without such orphans win."""
    analysis = asm.analyse()
    _, alive, main, _ = analysis
    groups = [n for c, n in alive.items() if c != main]
    loose = [n for n in groups if n > drop_max]
    big = _weak_joint_sizes(asm, analysis, hang_min)
    return (sum(loose), len(loose), len(big), sum(big), _footing_count(asm), sum(groups) - sum(loose))


class Issue(NamedTuple):
    """One structural problem: size (pieces affected), kind ("loose" / "weak" / "footing"), root piece and
    extra: the loose group's pieces, the weak joint's (root, partner) or the footing's (piece,)."""

    size: int
    kind: str
    root: int
    extra: frozenset[int] | tuple[int, ...]


def find_issues(asm: Assembly, hang_min: int) -> tuple[list[int], int, list[Issue]]:
    """(component per piece, main component, issues): loose groups, single-stud joints hanging >= hang_min pieces
    (root = first piece of the smaller side) and single-stud footings."""
    comp, alive, main, bridges = asm.analyse()
    out = [Issue(len(mem), "loose", mem[0], frozenset(mem)) for mem in asm.loose_groups(comp, main).values()]
    total = alive[main]
    for p, u, su in bridges:
        if comp[p] != main or asm.adj[p][u] > 1:
            continue
        root, partner, hang = (u, p, su) if su <= total - su else (p, u, total - su)
        if hang >= hang_min:
            out.append(Issue(hang, "weak", root, (root, partner)))
    if asm.nb:
        out += [Issue(1, "footing", i, (i,)) for i in range(len(asm.bricks)) if is_footing(asm, i)]
    return comp, main, out


def to_plates(asm: Assembly, i: int, z: int) -> int:
    """Piece i as a single plate at layer z: a 3-plate brick is split into 3 stacked plates."""
    b = asm.piece(i)
    if b.h_plates == PLATE_HEIGHT_PLATES:
        return i
    asm.remove(i)
    ids = {
        zz: asm.add(PlacedBrick(b.x, b.y, zz, b.w, b.d, PLATE_HEIGHT_PLATES, b.color))
        for zz in range(b.z, b.z + b.h_plates)
    }
    return ids[z]
