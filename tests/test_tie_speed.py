"""Tie repair speed-ups: the fast _grow_options / _sample_tilings / Stock.sizes_with give exactly the results of the
plain (slow) definitions, and repair() prints one progress line per phase."""

import contextlib
import io
import random
import sys

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.model import PlacedBrick
from image_to_bricks.packing import assembly, joint_retile
from image_to_bricks.packing import tie_repair as tr
from image_to_bricks.stock import Stock

failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def slow_grow_options(asm, P, T):
    """The plain definition: every size and position, checked cell by cell."""

    def overlap(x, y, w, d):
        return max(0, min(x + w, T.x + T.w) - max(x, T.x)) * max(0, min(y + d, T.y + T.d) - max(y, T.y))

    zs = slice(P.z, P.z + P.h_plates)
    free, own = ~asm.grid[zs] & asm.fill[zs], asm.owner[zs]
    cur, out = overlap(P.x, P.y, P.w, P.d), []
    for w, d in assembly.piece_sizes(asm):
        for x0 in range(max(0, P.x + P.w - w), min(P.x, asm.nx - w) + 1):
            for y0 in range(max(0, P.y + P.d - d), min(P.y, asm.ny - d) + 1):
                gain = overlap(x0, y0, w, d) - cur
                if (w, d) == (P.w, P.d) or gain <= 0:
                    continue
                if (free[:, y0 : y0 + d, x0 : x0 + w] | (own[:, y0 : y0 + d, x0 : x0 + w] == own[0, P.y, P.x])).all():
                    out.append((w * d - P.w * P.d, -gain, (x0, y0, w, d)))
    return sorted(out)


def slow_sample_tilings(cells, sizes, n, seed=7):
    """The plain definition of the sampler (same random draws)."""
    rng = random.Random(seed)
    order = sorted(cells, key=lambda p: (p[1], p[0]))
    seen, out = set(), []
    for t in range(n * 4):
        if len(out) >= n:
            break
        covered, rects = set(), []
        for x, y in order:
            if (x, y) in covered:
                continue
            opts = [
                (w, d)
                for (w, d) in sizes
                if all(
                    (xx, yy) in cells and (xx, yy) not in covered for yy in range(y, y + d) for xx in range(x, x + w)
                )
            ]
            w, d = (
                max(opts, key=lambda o: o[0] * o[1])
                if t == 0
                else rng.choices(opts, weights=[(o[0] * o[1]) ** 1.5 for o in opts])[0]
            )
            rects.append((x, y, w, d))
            covered.update((xx, yy) for yy in range(y, y + d) for xx in range(x, x + w))
        if frozenset(rects) not in seen:
            seen.add(frozenset(rects))
            out.append(rects)
    return out


def random_assembly(rng, k):
    nz, ny, nx = 4, rng.randint(6, 12), rng.randint(6, 12)
    grid, bricks = np.zeros((nz, ny, nx), bool), []
    for z in range(nz):
        for _ in range(rng.randint(2, 6)):
            w, d = rng.randint(1, 3), rng.randint(1, 3)
            x, y = rng.randint(0, nx - w), rng.randint(0, ny - d)
            if not grid[z, y : y + d, x : x + w].any():
                grid[z, y : y + d, x : x + w] = True
                bricks.append(PlacedBrick(x, y, z, w, d, 1, 9))
    fill = ~grid & (np.random.RandomState(k).rand(*grid.shape) < 0.6)
    return assembly.Assembly(grid.shape, bricks, grid, fill), bricks


def test_grow_options_match_plain_definition():
    rng, cases, bad = random.Random(3), 0, 0
    for k in range(120):
        asm, bricks = random_assembly(rng, k)
        for P in bricks:
            for T in bricks:
                if abs(T.z - P.z) == 1:
                    cases += 1
                    bad += tr._grow_options(asm, P, T) != slow_grow_options(asm, P, T)
    check(f"_grow_options == plain definition ({cases} cases)", bad == 0, bad)
    P, T = bricks[0], bricks[-1]
    opts = tr._grow_options(asm, P, T)
    check(
        "_grow_options returns plain ints (pickles stay numpy-free)",
        all(type(v) is int for o in opts for v in (o[0], o[1]) + o[2]),
    )


def test_sampler_matches_plain_definition():
    rng, bad = random.Random(1), 0
    for k in range(80):
        cells = set()
        while len(cells) < rng.randint(4, 30):
            cells.add((rng.randint(0, 7), rng.randint(0, 7)))
        sizes = assembly.SIZES + ([(8, 4), (4, 8), (6, 1), (1, 6)] if k % 2 else [])
        bad += joint_retile.sample_tilings(cells, sizes, 40, seed=k) != slow_sample_tilings(cells, sizes, 40, seed=k)
    check("_sample_tilings == plain definition (same random tilings)", bad == 0, bad)


def test_sizes_with_cache():
    stock = Stock({(1, 8, 4, 3): 1})
    first = stock.sizes_with(assembly.SIZES)
    first.append((99, 99))  # callers may change the returned list
    check("sizes_with returns a fresh list each call", (99, 99) not in stock.sizes_with(assembly.SIZES))
    stock.have[(1, 12, 6, 3)] = 1  # a new owned size shows up
    check("sizes_with sees a new owned size", (12, 6) in stock.sizes_with(assembly.SIZES))
    check("forks share the cached sizes", stock.fork().sizes_with(assembly.SIZES) == stock.sizes_with(assembly.SIZES))


def test_progress_lines():
    rng = random.Random(9)
    asm, bricks = random_assembly(rng, 9)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        tr.repair(tr.RepairTarget(asm.grid, fill=asm.fill), bricks)
    log = out.getvalue()
    check("repair prints a start line", "Tie repair started:" in log, log[:200])
    check("repair prints one line per phase", log.count("  ... ") == 3, log)
    with contextlib.redirect_stdout(io.StringIO()) as quiet:
        tr.repair(tr.RepairTarget(asm.grid, fill=asm.fill), bricks, verbose=False)
    check("no progress lines when verbose is off", quiet.getvalue() == "", quiet.getvalue()[:200])


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
    sys.exit(1 if failed else 0)
