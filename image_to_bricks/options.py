"""Every setting of a build, in typed groups with their defaults (the single place for them).

The command line turns its arguments into one BuildOptions (cli.options_from_arguments); the pipeline reads its
settings from there. Coordinates of boxes are in the figure's own grid at its real size: (slice z, slice y, slice x),
ends exclusive. Colours are palette ids (consts.COLOR_NAMES).
"""

import math
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from image_to_bricks.analysis.stability import MIN_TIP_ANGLE
from image_to_bricks.base.base_layout import LayoutPiece
from image_to_bricks.base.base_packer import LayeredBaseOptions
from image_to_bricks.colours.fur_ovals import FrontOval
from image_to_bricks.consts import BLACK_ID

type Box = tuple[slice, slice, slice]

DARK_BLUISH_GRAY_ID = 8


class Accent(NamedTuple):
    """--accent: a piece of one colour placed inside box (chrome buckle, eye)."""

    box: Box
    colour_id: int
    width_studs: int
    depth_studs: int
    note: str
    only_colour_id: int | None  # only cells of this colour take the accent; None = any colour


class Paint(NamedTuple):
    """--paint / --paint-deep: cells of box take colour_id (deep: also the cells behind the surface)."""

    box: Box
    colour_id: int
    is_deep: bool


class BaseZone(NamedTuple):
    """--base-zone: a pond or path of colour_id on the base, reaching the given side."""

    colour_id: int
    kind: str  # "pond" | "path"
    side: str  # "left" | "right" | "up" | "down"
    size: float


@dataclass(frozen=True)
class VoxelOptions:
    """How the mesh becomes cells."""

    supersample: int = 1  # odd N > 1: voxelise at N x resolution, keep a cell if enough of it is inside
    threshold: float = 0.4  # share of the N^3 sub-cells that must be filled
    thin_threshold: float | None = 0.2  # the same for thin parts (no kept cell beside them); None = no special rule
    box: Box | None = None  # region with its own threshold (a thin blade)
    box_threshold: float | None = None  # threshold inside box


@dataclass(frozen=True)
class ColourOptions:
    """How surface colours become palette colours, and the hand-made colour edits."""

    allowed_colour_ids: list[int] | None = None  # --colors; None = the whole palette
    colour_alias: dict[int, int] = field(default_factory=dict)  # --color-alias: matched colour -> used colour
    max_colours: int = 8  # keep the N most-used colours; 0 = no limit
    colour_groups: int = 16  # k-means clusters of surface colours, one palette colour each; 0 = off
    smooth_iterations: int = 2  # speckle clean-up passes
    absorb_min_region: int = 3  # regions smaller than this are absorbed by their neighbours
    missing_rgb: tuple[int, int, int] | None = None  # the colouriser's "no data" grey, filled from neighbours
    trim_colour_ids: list[int] = field(default_factory=list)  # thin trim colours completed to full rings
    trim_min_fraction: float = 0.15  # share of a ring the trim must already cover to be completed
    trim_snap_layers: int = 0  # ring-less trim snaps to the nearest trim ring within N layers; 0 = off
    trim_snap_min_area_ratio: float = 0.65  # snapped part area within [r, 1/r] of the ringed part; 0 = off
    despeckle_max_cells: int = 0  # lone colour patches up to N cells per layer take the neighbours' colour; 0 = off
    neutral_enclaves: bool = False  # light grey / white regions enclosed by colour take that colour
    dark_enclave_max_cells: int = 0  # small regions inside a Black part become Black; 0 = off
    front_ovals: list[FrontOval] = field(default_factory=list)  # oval fur bands on the front
    paints: list[Paint] = field(default_factory=list)  # hand-painted details
    accents: list[Accent] = field(default_factory=list)
    carve_gap: bool = False  # separate a layer (cape) from the body by a gap
    carve_colour_id: int = BLACK_ID
    carve_from_height: float = 0.22  # share of the figure height where the carving starts...
    carve_to_height: float = 0.48  # ...and ends


@dataclass(frozen=True)
class ShapeOptions:
    """Edits of the cell grid before packing."""

    front: str | None = None  # "-y" / "+y" / "-x" / "+x": the side the figure's face looks to
    shell_thickness_studs: int = 1  # hollow wall thickness (sideways); tops and bottoms stay 1 plate
    thicken_thin: bool = False  # widen 1-stud-wide parts to 2 so they hold together
    thicken_box: Box | None = None  # where to thicken; None = everywhere
    thicken_sheet_boxes: list[Box] = field(default_factory=list)  # 1-plate sheets (hat brim) get a second plate
    sheet_direction: int = 1  # +1 = the second plate goes above the sheet, -1 = below
    prop_box: Box | None = None  # a post under the lowest hanging cell inside it
    prop_size_studs: int = 2
    prop_colour_id: int | None = None  # None = Black
    drop_boxes: list[Box] = field(default_factory=list)  # filled cells removed
    fill_pinholes_max_cells: int = 0  # see-through holes up to N cells closed; 0 = off
    open_gaps: bool = False  # single-cell spurs that close a gap between two parts are removed
    join_corners_max_cells: int = 0  # corner-only steps in thin parts get a cell between; 0 = off
    ground_overhangs_max_plates: int = 0  # columns starting up to N plates in the air reach down; 0 = off
    ground_min_column_plates: int = 3  # ...if the column above is at least this tall


@dataclass(frozen=True)
class PackingOptions:
    """How cells become pieces."""

    hidden_any_colour: bool = True  # pieces may span hidden cells of any colour
    prefer_bricks: bool = False  # bricks first over the whole figure
    harmonize_bricks: bool = False  # merge colour differences inside each 3-layer band
    merge_plates: bool = False  # neighbouring plates that fill a standard plate exactly are merged
    anchor_floating: int = 0  # floating plates re-tiled with a supported neighbour: 0 off, 1 plates, 2 also bricks


@dataclass(frozen=True)
class RepairOptions:
    """Passes after packing that make the model hold together and use fewer small pieces."""

    tie_repair: bool = True  # re-tile around loose / single-stud joints
    tie_max_recoloured_cells: int = 4  # visible cells a tie may recolour (0 = never change colours)
    tie_min_hanging_pieces: int = 5  # fix weak joints that hang at least this many pieces on <= 1 stud
    tie_drop_max_pieces: int = 6  # untieable floating groups of at most this many pieces are removed (0 = keep)
    retile: bool = True  # after the ties: re-tile around the single-stud joints that are left
    retile_box: Box | None = None  # only joints rooted inside; None = all
    retile_reach_studs: int = 3  # window around a weak joint that may be re-tiled
    retile_drop_max_pieces: int = 1  # pieces a re-tiling may leave floating (and remove) per joint
    cleanup_small: bool = False  # lossless re-tiling around 1x1 pieces
    cleanup_extra_stacks: int = 0  # extra exact stacks the clean-up may create per group
    cleanup_extra_seams: int = 0  # extra aligned seams the clean-up may create per group
    use_l_pieces: bool = False  # owned L pieces replace small pieces
    l_skip_boxes: list[Box] = field(default_factory=list)  # pieces touching these are never swapped for an L


@dataclass(frozen=True)
class BaseOptions:
    """The stand under the figure."""

    plates: int = 0  # plates in the stand (0 = no base)
    brick_bands: int = 0  # lowest N bands of 3 layers packed as bricks (needs plates >= 3N + 1)
    margin_studs: int = 3  # base around the footprint (when not automatic)
    is_margin_auto: bool = True  # smallest margin that reaches min_tip_angle_degrees
    min_tip_angle_degrees: float = MIN_TIP_ANGLE
    shape: str = "rect"  # "rect" | "rounded"
    corner_radius_studs: int = 3  # for the rounded shape
    colour_id: int = DARK_BLUISH_GRAY_ID
    footprint_layers: int = 3  # footprint = the lowest N figure layers (0 = the whole figure)
    zones: list[BaseZone] = field(default_factory=list)
    zone_seed: int = 1  # shapes of the ponds / paths
    hidden_any_colour: bool = True  # with a stock: base cells nobody sees may be any colour
    layered: LayeredBaseOptions | None = None  # the layered base packer; None = the plate / brick packer
    layout: list[LayoutPiece] = field(default_factory=list)  # hand-built base plates used exactly
    layout_path: str = ""  # the layout file, for the console line
    fill_colour_ids: list[int] = field(default_factory=list)  # short base pieces take a spare owned colour


@dataclass(frozen=True)
class OutputOptions:
    """What is written besides the instructions."""

    write_views: bool = True  # flat front / back / left / right views
    debug: bool = False  # raw voxel image + placement check with diagnostic images
    make_pdf: bool = False  # instruction PDF (--pdf)
    show_layer_below: bool = True  # layer maps also show the layer below, hatched (--no-layer-below turns it off)


@dataclass(frozen=True)
class BuildOptions:
    """All settings of one build."""

    voxel: VoxelOptions = field(default_factory=VoxelOptions)
    colour: ColourOptions = field(default_factory=ColourOptions)
    shape: ShapeOptions = field(default_factory=ShapeOptions)
    packing: PackingOptions = field(default_factory=PackingOptions)
    repair: RepairOptions = field(default_factory=RepairOptions)
    base: BaseOptions = field(default_factory=BaseOptions)
    output: OutputOptions = field(default_factory=OutputOptions)


def _scale_box(box: Box, factor: float) -> Box:
    """Box with every end multiplied by factor, rounded outward (at least one cell wide)."""
    z_range, y_range, x_range = (
        slice(math.floor(part.start * factor), max(math.ceil(part.stop * factor), math.floor(part.start * factor) + 1))
        for part in box
    )
    return z_range, y_range, x_range


def _scale_optional_box(box: Box | None, factor: float) -> Box | None:
    return None if box is None else _scale_box(box, factor)


def scale_boxes(options: BuildOptions, factor: float) -> BuildOptions:
    """--boxes-for-studs: the options with every thicken / sheet / retile / prop / voxel / accent box scaled by factor
    (boxes written for one figure size, used on another)."""
    shape, colour = options.shape, options.colour
    return replace(
        options,
        voxel=replace(options.voxel, box=_scale_optional_box(options.voxel.box, factor)),
        shape=replace(
            shape,
            thicken_box=_scale_optional_box(shape.thicken_box, factor),
            thicken_sheet_boxes=[_scale_box(box, factor) for box in shape.thicken_sheet_boxes],
            prop_box=_scale_optional_box(shape.prop_box, factor),
        ),
        repair=replace(options.repair, retile_box=_scale_optional_box(options.repair.retile_box, factor)),
        colour=replace(
            colour, accents=[accent._replace(box=_scale_box(accent.box, factor)) for accent in colour.accents]
        ),
    )
