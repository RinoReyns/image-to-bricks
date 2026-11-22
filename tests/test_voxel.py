"""--voxel-supersample: true volume instead of the fattened surface voxelisation; thin parts stay connected.
Analytic shapes (two legs with a gap, a thin tail, a thin blade) voxelised like trimesh does (every cell that
contains a point of the solid), no mesh needed."""

from typing import Any, cast

import support

support.stub_mesh_libraries()
import numpy as np
from scipy.ndimage import label

from image_to_bricks.mesh import voxelize
from image_to_bricks.options import VoxelOptions


def inside(P):
    x, y, z = P.T
    legA = (x > 2.2) & (x < 6.9) & (y > 1.3) & (y < 5.9) & (z > 0) & (z < 12)
    legB = (x > 9.4) & (x < 14.1) & (y > 1.3) & (y < 5.9) & (z > 0) & (z < 12)
    body = (x > 2.2) & (x < 14.1) & (y > 1.3) & (y < 5.9) & (z >= 12) & (z < 20)
    a = np.array([14, 3.4, 10.0])
    b = np.array([22, 3.4, 15.0])
    d = b - a
    t = np.clip(((P - a) @ d) / (d @ d), 0, 1)
    tail = np.linalg.norm(P - (a + t[:, None] * d), axis=1) < 0.45
    blade = (x > 1.0) & (x < 1.25) & (y > 2.5) & (y < 3.3) & (z > 0.5) & (z < 10.5)
    return legA | legB | body | tail | blade


h = 0.15
X, Y, Z = np.meshgrid(np.arange(-1, 25, h), np.arange(0, 8, h), np.arange(-1, 22, h), indexing="ij")
P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
PTS = P[inside(P)]
true_vol = len(PTS) * h**3


def conservative(pitch):
    idx = np.rint(PTS / pitch).astype(int)
    mn = idx.min(0)
    m = np.zeros(idx.max(0) - mn + 1, bool)
    m[tuple((idx - mn).T)] = True
    return m, mn * pitch


def stats(m, org):
    w = np.argwhere(m) + org

    def at(x, y, z):
        i = np.rint(np.array([x, y, z]) - org).astype(int)
        return bool((i >= 0).all() and (i < m.shape).all() and m[tuple(i)])

    face = np.zeros((3, 3, 3), bool)
    face[1, 1, :] = face[1, :, 1] = face[:, 1, 1] = True
    return dict(
        gap=sum(not at(x, 3.4, 6.0) for x in (7, 8, 9, 10)),
        tail=int((w[:, 0] > 15.2).sum()),
        blade=int(((w[:, 0] < 1.9) & (w[:, 2] < 11)).sum()),
        total=int(m.sum()),
        pieces=label(m, structure=face)[1],
    )


m1, o1 = conservative(1.0)
s1 = stats(m1, o1)
mf, of = conservative(1 / 3)
mc, oc = voxelize.downsample_voxels(mf, of, 1 / 3, 1.0, VoxelOptions(threshold=0.4, thin_threshold=0.2))
s3 = stats(mc, oc)
print("true volume", round(true_vol), "| now", s1, "\n supersample 3 (0.4 / thin 0.2)", s3)
assert s1["total"] > 1.5 * true_vol, "the plain surface voxelisation is expected to inflate"
assert abs(s3["total"] - true_vol) < 0.15 * true_vol, "supersampling should give about the true volume"
assert s3["gap"] > s1["gap"], "gap between the legs must get wider"
assert s3["pieces"] == s1["pieces"] == 1, "thin parts must stay connected"
assert s3["blade"] >= 5 and s3["tail"] >= 8, "thin parts must not vanish"
assert np.allclose(oc, np.rint(oc)), "coarse cells must sit on multiples of the pitch"


# voxelize_mesh with a fake mesh: same result as calling the function directly, grid is [z, y, x]
class FakeVox:
    def __init__(self, pitch):
        m, o = conservative(pitch)
        self.matrix = m
        self.transform = np.eye(4)
        self.transform[:3, 3] = o

    def fill(self):
        return self


class FakeMesh:
    def voxelized(self, pitch):
        return FakeVox(pitch)


import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    grid, origin, pitch = voxelize.voxelize_mesh(
        cast(Any, FakeMesh()), 1.0, VoxelOptions(supersample=3, threshold=0.4, thin_threshold=0.2)
    )
assert grid.shape == mc.shape[::-1] and (grid == np.transpose(mc, (2, 1, 0))).all()
assert np.allclose(origin, oc) and np.allclose(pitch, 1.0)
with contextlib.redirect_stdout(io.StringIO()):
    g1, _, _ = voxelize.voxelize_mesh(cast(Any, FakeMesh()), 1.0, VoxelOptions(supersample=1))
assert (g1 == np.transpose(m1, (2, 1, 0))).all(), "supersample 1 must keep the old behaviour"
# box-limited thin threshold: a blade 1 fine cell wide (0.04 of a coarse cell at N=5) is lost at 0.05, kept (and
# connected to the body) inside the box at 0.02, and speckles outside the box stay out
N = 5
fine = np.zeros((60, 60, 120), bool)
fine[10:50, 10:50, 40:120] = True  # body
fine[25, 30, 0:40] = True  # blade below the body, 1 fine cell wide
fine[57, 57, 60] = True  # a hair far from anything, outside the box
a, _ = voxelize.downsample_voxels(fine, np.zeros(3), 1.0, float(N), VoxelOptions(threshold=0.4, thin_threshold=0.05))
b, _ = voxelize.downsample_voxels(
    fine,
    np.zeros(3),
    1.0,
    float(N),
    VoxelOptions(threshold=0.4, thin_threshold=0.05, box=(slice(0, 9), slice(0, 12), slice(0, 12)), box_threshold=0.02),
)
blade_cells = lambda m: int(m[3, 4, 0:8].sum())  # grid starts at the body (coarse cell 2)
assert blade_cells(a) == 0 and blade_cells(b) >= 6, (blade_cells(a), blade_cells(b))
lab, n = label(b)
assert n == 1, "blade must stay attached to the body"
assert b.shape[0] <= 9 + 1 and not b[9:, 9:, :].any(), "the hair (outside the box) stays out"
print("voxel box: blade cells", blade_cells(a), "->", blade_cells(b))
print("OK")
