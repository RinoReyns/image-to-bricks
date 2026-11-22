"""--base-bricks: the lowest band(s) of the base are 3-plate bricks, plates on top tie them together."""

import contextlib
import io

import support

support.stub_mesh_libraries()
import numpy as np

from image_to_bricks.analysis import stability
from image_to_bricks.base import stand
from image_to_bricks.options import BaseOptions, BuildOptions, RepairOptions
from image_to_bricks.packing import placement

bad = []
tot = 0
nbricks = 0
for shape in ("rect", "rounded"):
    for nb, nbr in ((4, 1), (5, 1), (7, 2)):
        for fw in range(2, 22, 4):
            for fd in (3, 8):
                for margin in (2, 4):
                    g = np.zeros((9, fd, fw), bool)
                    g[0:3] = True
                    c = np.full(g.shape, 9, np.uint8)
                    base = BaseOptions(
                        plates=nb, shape=shape, margin_studs=margin, corner_radius_studs=3, brick_bands=nbr
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        g2, c2, v2, mask = stand.add_base(g, c, g, base)
                        m = placement.optimize_brick_placement(
                            placement.PackingCells(g2, c2, visible=v2, base_layers=nb),
                            options=BuildOptions(base=base, repair=RepairOptions(tie_repair=False)),
                        )
                    tot += 1
                    cov = np.zeros(g2.shape, int)
                    for b in m.bricks:
                        cov[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] += 1
                    assert (cov == g2.astype(int)).all(), "coverage"
                    br = [b for b in m.bricks if b.z < nb and b.h_plates == 3]
                    nbricks += len(br)
                    assert br, "no base bricks"
                    assert all(b.z % 3 == 0 and b.z < 3 * nbr for b in br)
                    comp, sizes, _ = stability.analyse_graph(stability.build_contacts(m))
                    main = max(sizes, key=lambda component: sizes[component])
                    if any(comp[i] != main for i, b in enumerate(m.bricks) if b.z < nb):
                        bad.append((shape, nb, nbr, fw, fd, margin))
print("configs", tot, "base bricks", nbricks, "with loose base pieces", len(bad), bad[:10])
assert not bad
print("test_base_bricks: all passed")
