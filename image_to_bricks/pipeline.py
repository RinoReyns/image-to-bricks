"""The whole build: GLB -> cells -> colours -> pieces (build_model).

Arrays are [z, y, x] in model cells (x, y in studs, z in plate layers).
"""

import os
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

from image_to_bricks.analysis.debug_images import (
    verify_brick_placement,
    visualize_voxel_grid,
)
from image_to_bricks.base.base_pieces import fill_base_from_spare_pieces
from image_to_bricks.base.stand import add_base, choose_base_margin
from image_to_bricks.colours.cape_carve import carve_cape_gap
from image_to_bricks.colours.colour_sampling import ColourDebug, VoxelFrame, assign_voxel_colors
from image_to_bricks.consts import BLACK_ID
from image_to_bricks.mesh.loading import load_mesh, rotate_mesh, scale_mesh
from image_to_bricks.mesh.shape_edits import (
    add_prop,
    fix_gaps_and_pinholes,
    thicken_sheets,
    thicken_thin,
    widen_joints,
)
from image_to_bricks.mesh.voxelize import (
    surface_mask,
    voxelize_mesh,
    voxels_to_lego_units,
)
from image_to_bricks.model import LegoModel
from image_to_bricks.options import BaseOptions, BuildOptions, ShapeOptions
from image_to_bricks.packing.accents import apply_accents
from image_to_bricks.packing.placement import PackingCells, optimize_brick_placement
from image_to_bricks.stock import Stock

# Save step-by-step colour diagnostic renders (<model>_dbg*.png) into outdir.
COLOR_DEBUG = True


@dataclass(frozen=True)
class BuildInput:
    """What to build: the mesh file, how it is turned and sized, and whether the figure is hollow."""

    glb_path: str
    target_studs: int  # size of the figure in studs (see mesh.loading.scale_mesh)
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)  # turns about x, y, z
    is_hollow: bool = False

    @property
    def model_name(self) -> str:
        """The mesh file name without folder and extension (prefix of every output file)."""
        return os.path.splitext(os.path.basename(self.glb_path))[0]


def _load_turned_scaled_mesh(build_input: BuildInput) -> trimesh.Trimesh:
    """The mesh, turned by build_input.rotation_degrees and scaled to build_input.target_studs."""
    mesh = load_mesh(build_input.glb_path)
    if any(build_input.rotation_degrees):
        mesh = rotate_mesh(mesh, *build_input.rotation_degrees)
    return scale_mesh(mesh, build_input.target_studs)


def _thicken(grid: NDArray[np.bool_], shape: ShapeOptions) -> NDArray[np.bool_]:
    """--thicken-thin (+ joint widening inside --thicken-box) and --thicken-sheet-box; prints what was added."""
    if shape.thicken_thin:
        grid, added_cells = thicken_thin(grid, shape.thicken_box)
        print(f"Thicken thin parts: +{added_cells} cells (1-wide parts made 2 wide)")
        if shape.thicken_box is not None:
            grid, added_cells = widen_joints(grid, shape.thicken_box)
            print(f"Widen joints: +{added_cells} cells (single-stud joints between layers made 2 studs)")
    for sheet_box in shape.thicken_sheet_boxes:
        grid, added_cells = thicken_sheets(grid, sheet_box, shape.sheet_direction)
        print(
            f"Thicken sheets: +{added_cells} cells (1-plate-thick sheets in {sheet_box} got a second plate "
            f"{'above' if shape.sheet_direction > 0 else 'below'})"
        )
    return grid


def _edit_shape(grid: NDArray[np.bool_], shape: ShapeOptions) -> tuple[NDArray[np.bool_], NDArray[np.bool_] | None]:
    """Thickening and the --prop-box post; returns (grid, cells of the post or None)."""
    grid = _thicken(grid, shape)
    if shape.prop_box is None:
        return grid, None
    grid, prop_mask = add_prop(grid, shape.prop_box, shape.prop_size_studs)
    print(
        f"Prop (post under a hanging part): +{int(prop_mask.sum())} cells"
        + ("" if prop_mask.any() else " - nothing hangs inside --prop-box")
    )
    return grid, prop_mask


def _hollow_figure(
    grid: NDArray[np.bool_], colours: NDArray[Any], shape: ShapeOptions, *, is_hollow: bool
) -> PackingCells:
    """Model cells of the coloured figure: visibility is taken before hollowing."""
    visible = surface_mask(grid)
    solid_grid = grid.astype(bool)
    grid = voxels_to_lego_units(grid, hollow=is_hollow, shell_thickness_studs=shape.shell_thickness_studs)
    return PackingCells(grid, colours, visible & grid, solid_grid & ~grid)


def _add_stand(cells: PackingCells, base: BaseOptions) -> tuple[PackingCells, BaseOptions]:
    """The figure on its stand (the grid grows by the base layers and margin); base gets the automatic margin
    when asked for."""
    if cells.visible is None or cells.fill is None:
        raise ValueError("_add_stand needs the visible and fill cells of the figure")
    _, figure_depth_y, figure_depth_x = cells.grid.shape
    if base.is_margin_auto:
        base = replace(base, margin_studs=choose_base_margin(cells.grid, base))
    grid, colours, visible, _ = add_base(cells.grid, cells.colours, cells.visible, base)
    margin = base.margin_studs
    fill = np.zeros(grid.shape, bool)
    fill[base.plates :, margin : margin + figure_depth_y, margin : margin + figure_depth_x] = cells.fill
    return PackingCells(grid, colours, visible, fill, base.plates), base


def _finish_model(model: LegoModel, options: BuildOptions, stock: Stock | None, outdir: str, model_name: str) -> None:
    """Steps after packing: accents, base fill from spare pieces, placement check (--debug, images in outdir)."""
    model.front = options.shape.front
    print(f"Placed {len(model.bricks)} bricks/plates.")
    if options.colour.accents:
        apply_accents(model, options.colour.accents, stock)
    base = options.base
    if base.fill_colour_ids and stock is not None and model.base_layers and not base.layout:
        fill_base_from_spare_pieces(model, model.base_layers, base, stock)
    if options.output.debug and not verify_brick_placement(model, outdir, model_name):
        print(
            "WARNING: Brick placement differs from the input voxel grid. "
            "Inspect the placement diagnostic PNGs before exporting."
        )


def build_model(
    build_input: BuildInput, outdir: str, options: BuildOptions | None = None, stock: Stock | None = None
) -> LegoModel:
    """The whole pipeline: GLB -> cells -> colours -> pieces (options: the build settings, default all off).
    Debug and colour diagnostic images go to outdir; the stock ledger follows every placed piece."""
    options = options if options is not None else BuildOptions()
    model_name, shape = build_input.model_name, options.shape
    debug_dir = outdir if COLOR_DEBUG else None
    mesh = _load_turned_scaled_mesh(build_input)
    grid, voxel_origin, voxel_pitch = voxelize_mesh(mesh, voxel=options.voxel)
    grid, prop_mask = _edit_shape(grid, shape)
    if options.output.debug:
        visualize_voxel_grid(grid, outdir, model_name)
    # Colour the SOLID grid first (exposure tests see true outside faces), then carve the cape gap, then hollow.
    colours = assign_voxel_colors(
        grid, mesh, VoxelFrame(voxel_origin, voxel_pitch), ColourDebug(debug_dir, model_name), options
    )
    if prop_mask is not None and prop_mask.any():
        colours[prop_mask] = shape.prop_colour_id if shape.prop_colour_id is not None else BLACK_ID
    if options.colour.carve_gap:
        grid, colours = carve_cape_gap(grid, colours, debug_dir=debug_dir, model_name=model_name, colour=options.colour)
    grid, colours = fix_gaps_and_pinholes(grid, colours, shape)
    cells = _hollow_figure(grid, colours, shape, is_hollow=build_input.is_hollow)
    base = options.base
    if base.plates > 0:
        cells, base = _add_stand(cells, base)
    options = replace(options, base=base)
    model = optimize_brick_placement(cells, options, stock)
    _finish_model(model, options, stock, outdir, model_name)
    return model
