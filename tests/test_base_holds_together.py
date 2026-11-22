"""The generated base holds the figure together: for every base shape, radius, thickness, margin and figure
footprint, the pieces cover the grid exactly, the whole build is ONE connected assembly and no base piece is loose.

By default a fixed sample of the configurations runs (the extremes of every dimension + a seeded random pick, the
same on every run); `--exhaustive` (or `python tests/run_all.py --exhaustive`) runs all of them.
Synthetic data, no mesh needed. Run: python tests/test_base_holds_together.py [--exhaustive]
"""

import contextlib
import io
import itertools
import os
import random
from dataclasses import dataclass
from unittest import mock

import support

support.stub_mesh_libraries()  # the packing code needs neither trimesh nor tqdm; keeps this test light

import numpy as np

from image_to_bricks.analysis import stability
from image_to_bricks.base import stand
from image_to_bricks.options import BaseOptions, BuildOptions, RepairOptions
from image_to_bricks.packing import placement

FIGURE_LAYERS = 3  # a plain block as the figure
GRID_LAYERS = 6
FIGURE_COLOUR_ID = 9
SHAPE_RADII = (("rect", 3), ("rounded", 2), ("rounded", 3), ("rounded", 5))
BASE_PLATE_COUNTS = (2, 3)
FIGURE_WIDTHS_STUDS = tuple(range(2, 22))
FIGURE_DEPTHS_STUDS = (3, 4, 5, 7, 8)
BASE_MARGINS_STUDS = (1, 2, 3, 4)
SAMPLE_SIZE = 160
SAMPLE_SEED = 20261121


@dataclass(frozen=True)
class BaseCase:
    shape: str
    radius: int
    base_plates: int
    figure_width_studs: int
    figure_depth_studs: int
    margin_studs: int


def all_cases() -> list[BaseCase]:
    return [
        BaseCase(shape, radius, plates, width, depth, margin)
        for (shape, radius), plates, width, depth, margin in itertools.product(
            SHAPE_RADII, BASE_PLATE_COUNTS, FIGURE_WIDTHS_STUDS, FIGURE_DEPTHS_STUDS, BASE_MARGINS_STUDS
        )
    ]


def is_extreme(case: BaseCase) -> bool:
    """Smallest / largest figure in both directions: where the base geometry is most likely to go wrong."""
    return case.figure_width_studs in (min(FIGURE_WIDTHS_STUDS), max(FIGURE_WIDTHS_STUDS)) and (
        case.figure_depth_studs in (min(FIGURE_DEPTHS_STUDS), max(FIGURE_DEPTHS_STUDS))
    )


def cases_to_run() -> list[BaseCase]:
    cases = all_cases()
    if support.is_exhaustive():
        return cases
    extremes = [case for case in cases if is_extreme(case)]
    others = [case for case in cases if not is_extreme(case)]
    return extremes + random.Random(SAMPLE_SEED).sample(others, SAMPLE_SIZE - len(extremes))


def build(case: BaseCase):
    """Base + packing for one case (tie repair off: this checks the base design itself)."""
    grid = np.zeros((GRID_LAYERS, case.figure_depth_studs, case.figure_width_studs), bool)
    grid[0:FIGURE_LAYERS] = True
    colours = np.full(grid.shape, FIGURE_COLOUR_ID, np.uint8)
    base = BaseOptions(
        plates=case.base_plates, shape=case.shape, margin_studs=case.margin_studs, corner_radius_studs=case.radius
    )
    with contextlib.redirect_stdout(io.StringIO()) as log:
        grid_with_base, colours_with_base, visible, _ = stand.add_base(grid, colours, grid, base)
        model = placement.optimize_brick_placement(
            placement.PackingCells(grid_with_base, colours_with_base, visible=visible, base_layers=case.base_plates),
            options=BuildOptions(base=base, repair=RepairOptions(tie_repair=False)),
        )
    return model, log.getvalue()


def find_problems(case: BaseCase) -> list[str]:
    model, log = build(case)
    problems = []
    covered = support.cells_covered_by_pieces(model)
    if not np.array_equal(covered, model.grid.astype(int)):
        problems.append("pieces do not cover the grid exactly")
    components, sizes, _ = stability.analyse_graph(stability.build_contacts(model))
    if len(sizes) > 1:
        problems.append(f"{len(sizes)} separate assemblies")
    main_assembly = max(sizes, key=lambda component: sizes[component])
    loose_base_pieces = sum(
        1
        for index, piece in enumerate(model.bricks)
        if piece.z < case.base_plates and components[index] != main_assembly
    )
    if loose_base_pieces:
        problems.append(f"{loose_base_pieces} loose base piece(s)")
    if "WARNING" in log:
        problems.append("a WARNING was printed")
    return problems


def test_every_base_holds_the_figure_together():
    cases = support.select_shard(cases_to_run())  # run_all.py may split the exhaustive sweep over several processes
    failures = [(case, problems) for case in cases if (problems := find_problems(case))]
    print(f"  {len(cases)} base configurations checked{' (exhaustive)' if support.is_exhaustive() else ''}")
    assert not failures, "\n".join(f"{case}: {', '.join(problems)}" for case, problems in failures[:20])


def test_sample_contains_every_value_of_every_dimension():
    cases = cases_to_run()
    assert {(case.shape, case.radius) for case in cases} == set(SHAPE_RADII)
    assert {case.base_plates for case in cases} == set(BASE_PLATE_COUNTS)
    assert {case.margin_studs for case in cases} == set(BASE_MARGINS_STUDS)
    assert {case.figure_depth_studs for case in cases} == set(FIGURE_DEPTHS_STUDS)
    assert len(cases) == (len(all_cases()) if support.is_exhaustive() else SAMPLE_SIZE)


def test_shards_together_check_every_case_exactly_once():
    cases = all_cases()
    checked = []
    for shard_index in range(3):
        with mock.patch.dict(os.environ, {support.SHARD_ENVIRONMENT_VARIABLE: f"{shard_index}/3"}):
            checked.extend(support.select_shard(cases))
    assert sorted(map(repr, checked)) == sorted(map(repr, cases))


if __name__ == "__main__":
    support.run_tests(globals())
