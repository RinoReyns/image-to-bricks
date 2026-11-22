"""Stability report for a finished LegoModel (measure only, never changes the build).

1. Pieces are connected when one sits on the other (shared studs). Reported:
   - disconnected parts (pieces that would fall off / separate completely),
   - weak joints: a sub-assembly held to the rest by <= WEAK_STUDS studs
     (e.g. a sword tip or ear hanging on one stud), via bridge detection,
   - single-stud footings: pieces of the figure's first layer that sit on the
     base with only one stud (needs --base).
2. Centre of mass (hollow model, every filled cell = same mass, so a 3-plate
   brick weighs 3x a plate) against the support area (convex hull of the
   bottom layer = the base): distance to the nearest edge, tip angle, and a
   suggested extra base margin.
"""

import csv
import math
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from image_to_bricks.consts import COLOR_NAMES, PLATE_MM, STUD_MM
from image_to_bricks.model import LegoModel, piece_mask, plate_origin

WEAK_STUDS = 1  # a joint with <= this many studs counts as weak
MIN_TIP_ANGLE = 15.0  # degrees the model may be tilted about the base edge before it tips
CELL_CENTRE = 0.5  # a cell's centre, in cells from its corner
CSV_HEADER = ["kind", "where", "x", "y", "studs", "pieces_affected", "colour", "plate_x", "plate_y"]


type Contacts = list[dict[int, int]]  # contacts[i] = {j: studs shared} for pieces stacked on each other
type Bridge = tuple[int, int, int]  # (piece, piece hanging on it, pieces on the hanging side)
type Point3 = tuple[float, float, float]  # (x, y, z) in mm
# (kind, layer name, x, y, studs, pieces affected, colour name or id) in grid studs
type StabilityRow = tuple[str, str, int, int, int, int, str | int]
type WeakJoint = tuple[int, int, int]  # (pieces hanging, studs holding them, index of the piece they hang from)


def build_contacts(model: LegoModel) -> Contacts:
    """contacts[i] = {j: studs shared} for pieces stacked directly on each other."""
    bricks = model.bricks
    layer_count = model.grid.shape[0]
    owner = np.full(model.grid.shape, -1, np.int32)
    for i, b in enumerate(bricks):
        if b.notch >= 0:  # L piece: only its 3 cells
            owner[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w][:, piece_mask(b)] = i
        else:
            owner[b.z : b.z + b.h_plates, b.y : b.y + b.d, b.x : b.x + b.w] = i
    contacts: Contacts = [{} for _ in bricks]
    for i, b in enumerate(bricks):
        layer_above = b.z + b.h_plates
        if layer_above >= layer_count:
            continue
        owners_above = owner[layer_above, b.y : b.y + b.d, b.x : b.x + b.w]
        if b.notch >= 0:
            owners_above = owners_above[piece_mask(b)]
        ids, counts = np.unique(owners_above[owners_above >= 0], return_counts=True)
        for j, studs in zip(ids.tolist(), counts.tolist(), strict=True):
            contacts[i][j] = contacts[i].get(j, 0) + studs
            contacts[j][i] = contacts[j].get(i, 0) + studs
    return contacts


def analyse_graph(adj: Contacts) -> tuple[list[int], dict[int, int], list[Bridge]]:
    """Connected parts and bridges of the contact graph (iterative Tarjan).
    Returns (component of every piece, {component: pieces}, bridges as (piece, child, pieces on the child's side))."""
    piece_count = len(adj)
    discovery = [-1] * piece_count
    low_link = [0] * piece_count
    subtree_size = [1] * piece_count
    component = [-1] * piece_count
    bridges: list[Bridge] = []
    time = 0
    for start in range(piece_count):
        if discovery[start] != -1:
            continue
        discovery[start] = low_link[start] = time
        time += 1
        component[start] = start
        stack = [(start, -1, iter(adj[start]))]
        while stack:
            node, parent, neighbours = stack[-1]
            for neighbour in neighbours:
                if neighbour == parent:
                    continue
                if discovery[neighbour] == -1:
                    discovery[neighbour] = low_link[neighbour] = time
                    time += 1
                    component[neighbour] = start
                    stack.append((neighbour, node, iter(adj[neighbour])))
                    break
                low_link[node] = min(low_link[node], discovery[neighbour])
            else:
                stack.pop()
                if parent != -1:
                    low_link[parent] = min(low_link[parent], low_link[node])
                    subtree_size[parent] += subtree_size[node]
                    if low_link[node] > discovery[parent]:
                        bridges.append((parent, node, subtree_size[node]))
    component_sizes: dict[int, int] = {}
    for component_id in component:
        component_sizes[component_id] = component_sizes.get(component_id, 0) + 1
    return component, component_sizes, bridges


def _centre_of_cells_mm(grid: NDArray[np.bool_], from_layer: int) -> Point3 | None:
    """(x, y, z) mean of the filled cells in layers >= from_layer, in mm; None if there are none."""
    cells = np.argwhere(grid & (np.arange(grid.shape[0])[:, None, None] >= from_layer))
    if len(cells) == 0:
        return None
    z = (cells[:, 0] + CELL_CENTRE) * PLATE_MM
    y = (cells[:, 1] + CELL_CENTRE) * STUD_MM
    x = (cells[:, 2] + CELL_CENTRE) * STUD_MM
    return float(x.mean()), float(y.mean()), float(z.mean())


def centre_of_mass(model: LegoModel) -> tuple[Point3 | None, Point3 | None]:
    """(x, y, z) of the centre of mass in mm (z up from the bottom of the base),
    for the whole model and for the figure alone (None without a base)."""
    base_layers = model.base_layers
    figure = _centre_of_cells_mm(model.grid, base_layers) if base_layers else None
    return _centre_of_cells_mm(model.grid, 0), figure


def support_hull(model: LegoModel) -> ConvexHull | None:
    """Convex hull (scipy ConvexHull) of the bottom layer's footprint, in mm; None if it has no area."""
    ys, xs = np.nonzero(model.grid[0])
    corners = [
        ((x + step_x) * STUD_MM, (y + step_y) * STUD_MM)
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
        for step_y in (0, 1)
        for step_x in (0, 1)
    ]
    try:
        return ConvexHull(np.array(corners))
    except Exception:  # qhull raises several error types for an empty or flat footprint: no hull then
        return None


def _direction(normal: NDArray[np.float64]) -> str:
    """Readable side of the footprint an outward edge normal (x, y) points to."""
    normal_x, normal_y = normal
    if abs(normal_x) >= abs(normal_y):
        return "high x (right in the layer maps)" if normal_x > 0 else "low x (left in the layer maps)"
    return "high y (up in the layer maps)" if normal_y > 0 else "low y (down in the layer maps)"


def stability_check(model: LegoModel, min_tip_angle: float | None = None) -> dict[str, Any]:
    """Dict with the centre-of-mass result."""
    min_tip_angle = MIN_TIP_ANGLE if min_tip_angle is None else min_tip_angle
    total, figure = centre_of_mass(model)
    hull = support_hull(model)
    result: dict[str, Any] = {"com_mm": total, "figure_com_mm": figure, "min_tip_angle": min_tip_angle}
    if total is None or hull is None:
        result["ok"] = None
        return result
    x, y, z = total
    # hull.equations: n.p + offset <= 0 inside, n normalised
    distances = -(hull.equations[:, 0] * x + hull.equations[:, 1] * y + hull.equations[:, 2])
    nearest_edge = int(np.argmin(distances))
    distance = float(distances[nearest_edge])
    result.update(dist_mm=distance, height_mm=z, direction=_direction(hull.equations[nearest_edge, :2]))
    if distance > 0:
        result["tip_angle"] = math.degrees(math.atan2(distance, z))
    else:
        result["tip_angle"] = -math.degrees(math.atan2(-distance, z))
    needed_distance = z * math.tan(math.radians(min_tip_angle))
    result["extra_margin_studs"] = max(0.0, (needed_distance - distance) / STUD_MM)
    result["ok"] = distance > 0 and result["tip_angle"] >= min_tip_angle
    return result


def _label(model: LegoModel, z: int) -> str:
    """'base 2' or 'layer 5' for model layer z."""
    base_layers = model.base_layers
    return f"base {z + 1}" if z < base_layers else f"layer {z - base_layers + 1}"


def _row(model: LegoModel, kind: str, index: int, studs: int, pieces_affected: int) -> StabilityRow:
    """Report row about piece `index`."""
    b = model.bricks[index]
    return kind, _label(model, b.z), b.x, b.y, studs, pieces_affected, COLOR_NAMES.get(b.color, b.color)


def _weak_joints(
    contacts: Contacts, component: list[int], component_sizes: dict[int, int], bridges: list[Bridge]
) -> list[WeakJoint]:
    """Bridges held by <= WEAK_STUDS studs, the smaller side counted as hanging; most pieces hanging first."""
    weak: list[WeakJoint] = []
    for parent, child, child_side in bridges:
        studs = contacts[parent][child]
        if studs > WEAK_STUDS:
            continue
        total = component_sizes[component[parent]]
        root, hanging = (child, child_side) if child_side <= total - child_side else (parent, total - child_side)
        weak.append((hanging, studs, root))
    weak.sort(key=lambda joint: -joint[0])
    return weak


def _single_stud_footings(model: LegoModel, contacts: Contacts) -> list[int]:
    """Indices of the first-figure-layer pieces that touch the base with exactly one stud."""
    base_layers, bricks = model.base_layers, model.bricks
    if not base_layers:
        return []
    return [
        i
        for i, b in enumerate(bricks)
        if b.z == base_layers and sum(studs for j, studs in contacts[i].items() if bricks[j].z < base_layers) == 1
    ]


def _write_csv(path: str, rows: list[StabilityRow], model: LegoModel) -> None:
    """<model>_stability.csv: the rows plus x / y counted from the base plate's corner (the map axes)."""
    origin_x, origin_y = plate_origin(model.grid, model.base_layers)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows([(*row, row[2] - origin_x, row[3] - origin_y) for row in rows])


def _print_connections(
    model: LegoModel, contacts: Contacts, parts: list[tuple[int, int]], weak: list[WeakJoint], footings: list[int]
) -> None:
    """Console lines about pieces, disconnected parts, weak joints and single-stud footings."""
    print("Stability report:")
    print(f"  pieces: {len(model.bricks)}, stud contacts: {sum(len(c) for c in contacts) // 2}")
    if parts:
        print(
            f"  WARNING: {len(parts)} disconnected part(s), {sum(n for _, n in parts)} pieces would fall off (see CSV)"
        )
    else:
        print("  all pieces form one connected assembly")
    print(
        f"  weak joints (<= {WEAK_STUDS} stud): {len(weak)}"
        + (f", largest hangs {weak[0][0]} piece(s) on {weak[0][1]} stud(s)" if weak else "")
    )
    if model.base_layers:
        print(f"  single-stud footings on the base (first figure layer): {len(footings)}")


def _print_centre_of_mass(model: LegoModel, check: dict[str, Any]) -> None:
    """Console lines about the centre of mass, the tip angle and what to change if the model would tip."""
    centre = check["com_mm"]
    if centre is None or check.get("ok") is None:
        print("  centre of mass: could not be computed")
        return
    x, y, z = centre
    print(f"  centre of mass: x {x / STUD_MM:.1f}, y {y / STUD_MM:.1f} studs, {z / STUD_MM:.1f} studs high")
    if check["dist_mm"] <= 0:
        print(
            f"  WARNING: centre of mass is OUTSIDE the support area "
            f"({-check['dist_mm'] / STUD_MM:.1f} studs past the edge, toward {check['direction']}): "
            f"the model will tip over."
        )
    else:
        print(
            f"  centre of mass is {check['dist_mm'] / STUD_MM:.1f} studs inside the support edge "
            f"(nearest: {check['direction']}); tip angle {check['tip_angle']:.0f} deg "
            f"(minimum {check['min_tip_angle']:.0f})"
        )
    if check["ok"]:
        return
    if model.base_layers:
        print(
            f"  WARNING: unstable. Try --base-margin +{math.ceil(check['extra_margin_studs'])} "
            f"studs or more, or a rectangle that also covers what sticks out toward "
            f"{check['direction']} (--base-from-layers 0)."
        )
    else:
        print(
            f"  WARNING: unstable without a base. Add --base with a margin of at least "
            f"{math.ceil(check['extra_margin_studs'])} studs."
        )


def report(
    model: LegoModel,
    outdir: str | None = None,
    model_name: str = "model",
    min_tip_angle: float | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Print the stability report (verbose) and write <model_name>_stability.csv into outdir (if given).
    Returns the counts, the centre-of-mass check and the rows."""
    contacts = build_contacts(model)
    component, component_sizes, bridges = analyse_graph(contacts)
    main = max(component_sizes, key=lambda component_id: component_sizes[component_id])
    parts = [(component_id, n) for component_id, n in component_sizes.items() if component_id != main]
    rows: list[StabilityRow] = []
    for component_id, pieces in sorted(parts, key=lambda part: -part[1]):
        first = next(i for i in range(len(model.bricks)) if component[i] == component_id)
        rows.append(_row(model, "disconnected", first, 0, pieces))
    weak = _weak_joints(contacts, component, component_sizes, bridges)
    rows += [_row(model, "weak_joint", root, studs, hanging) for hanging, studs, root in weak]
    footings = _single_stud_footings(model, contacts)
    rows += [_row(model, "single_stud_footing", i, 1, 1) for i in footings]
    check = stability_check(model, min_tip_angle)
    if outdir:
        _write_csv(os.path.join(outdir, f"{model_name}_stability.csv"), rows, model)
    if verbose:
        _print_connections(model, contacts, parts, weak, footings)
        _print_centre_of_mass(model, check)
    return {
        "components": len(component_sizes),
        "disconnected_pieces": sum(n for _, n in parts),
        "weak_joints": len(weak),
        "single_stud_footings": len(footings),
        "stability": check,
        "rows": rows,
    }
