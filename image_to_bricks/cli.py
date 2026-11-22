"""Command line of model_to_bricks.py: arguments -> BuildOptions, the run log, and the whole run."""

import argparse
import os
import sys
from typing import TextIO, cast

from image_to_bricks.analysis import stability
from image_to_bricks.base import base_layout, base_packer, base_ties
from image_to_bricks.colours import fur_ovals
from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.instructions import elevation
from image_to_bricks.instructions.outputs_handler import save_model, save_outputs
from image_to_bricks.model import LegoModel
from image_to_bricks.options import (
    Accent,
    BaseOptions,
    BaseZone,
    Box,
    BuildOptions,
    ColourOptions,
    OutputOptions,
    PackingOptions,
    Paint,
    RepairOptions,
    ShapeOptions,
    VoxelOptions,
    scale_boxes,
)
from image_to_bricks.pipeline import BuildInput, build_model
from image_to_bricks.stock import Stock

DEFAULTS = BuildOptions()  # argparse shows and uses these defaults
RANGE_ENDS = 2  # a box range is START:STOP


def _color_id(name: str) -> int:
    """Palette id of a colour name (case-insensitive); exits listing the palette on an unknown name."""
    lookup = {v.lower(): k for k, v in COLOR_NAMES.items()}
    key = name.strip().lower()
    if key not in lookup:
        raise SystemExit(f"Unknown colour '{name}'. Available: {', '.join(COLOR_NAMES.values())}")
    return lookup[key]


# TODO:
# Clean up code
# Per page - bricks list, different angle


def _ratio_from_zero_to_one(text: str) -> float:
    """argparse type: a number in [0, 1]."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number from 0 to 1, got {text!r}") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"expected a number from 0 to 1, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    """argparse type: an integer >= 0."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a whole number, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def _plate_size(text: str) -> tuple[int, int]:
    """argparse type for --base-max-plate: '8x8' -> (8, 8) (long side, short side)."""
    try:
        return base_packer.parse_plate_size(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from None


def _layered_base_options(args: argparse.Namespace) -> base_packer.LayeredBaseOptions | None:
    """base_packer.LayeredBaseOptions from --base-hidden-layers / --base-max-plate, or None (old base packer).
    Exits with a message on combinations that cannot work."""
    if args.base_hidden_layers <= 0:
        if args.base_max_plate is not None or args.base_sparse_layers:
            raise SystemExit(
                "--base-max-plate / --base-sparse-layers need --base-hidden-layers "
                "(they only apply to the layered base)"
            )
        return None
    if not 0 <= args.base_sparse_layers <= args.base_hidden_layers:
        raise SystemExit(
            f"--base-sparse-layers must be 0..{args.base_hidden_layers} (only hidden layers can be "
            f"sparse), got {args.base_sparse_layers}"
        )
    if args.base_tie_every < 1:
        raise SystemExit(f"--base-tie-every must be at least 1, got {args.base_tie_every}")
    if args.base < args.base_hidden_layers + 1:
        raise SystemExit(
            f"--base-hidden-layers {args.base_hidden_layers} needs --base of at least "
            f"{args.base_hidden_layers + 1} (at least one visible layer on top)"
        )
    if args.base_bricks:
        raise SystemExit("--base-hidden-layers and --base-bricks cannot be combined (the layered base is plates only)")
    return base_packer.LayeredBaseOptions(
        hidden_layers=args.base_hidden_layers,
        sparse_layers=args.base_sparse_layers,
        tie_every=args.base_tie_every,
        max_plate=args.base_max_plate or base_packer.BASE_MAX_PLATE,
    )


def parse_front_oval(spec: str) -> fur_ovals.FrontOval:
    """'Z0:Z1,H0:H1=COLOUR,COLOUR,...' -> fur_ovals.FrontOval (exits with a clear message on a bad value)."""
    try:
        box_text, colours_text = spec.split("=", 1)
        (z0, z1), (h0, h1) = [tuple(int(v) for v in part.split(":")) for part in box_text.split(",")]
        colour_ids = tuple(_color_id(name.strip()) for name in colours_text.split(",") if name.strip())
    except ValueError:
        raise SystemExit(
            f'--front-oval must look like "24:58,6:16=Yellow,Orange,Dark Orange" (Z0:Z1,H0:H1=COLOURS), got {spec!r}'
        ) from None
    if z1 <= z0 or h1 <= h0 or not colour_ids:
        raise SystemExit(f"--front-oval {spec!r}: the box must not be empty and at least one colour is needed")
    return fur_ovals.FrontOval(z0, z1, h0, h1, colour_ids)


def parse_box(spec: str, option: str) -> Box:
    """'Z0:Z1,Y0:Y1,X0:X1' -> (slice z, slice y, slice x); exits with a message naming the option if it is malformed."""
    try:
        z, y, x = [tuple(int(v) for v in part.split(":")) for part in spec.split(",")]
        if not (len(z) == len(y) == len(x) == RANGE_ENDS):
            raise ValueError(spec)
    except ValueError:
        raise SystemExit(f"{option} must look like 10:34,12:25,12:25 (Z0:Z1,Y0:Y1,X0:X1), got '{spec}'") from None
    return slice(*z), slice(*y), slice(*x)


def _box_from_text(text: str) -> Box:
    """'Z0:Z1,Y0:Y1,X0:X1' -> (slice z, slice y, slice x); raises ValueError when malformed."""
    z_range, y_range, x_range = [tuple(int(value) for value in part.split(":")) for part in text.split(",")]
    return slice(*z_range), slice(*y_range), slice(*x_range)


def _box_option(text: str, message: str) -> Box:
    """A box option, or exit with message (which shows the expected form) when it is malformed."""
    try:
        return _box_from_text(text)
    except ValueError:
        raise SystemExit(message) from None


def _paints(args: argparse.Namespace) -> list[Paint]:
    paints = []
    for option, specs, is_deep in (("--paint", args.paint, False), ("--paint-deep", args.paint_deep, True)):
        for spec in specs:
            box_text, colour_name = spec.rsplit("=", 1) if "=" in spec else (spec, "")
            box = _box_option(box_text, f"{option} must look like 74:75,4:5,11:12=Black (Z0:Z1,Y0:Y1,X0:X1=COLOUR)")
            if not colour_name:
                raise SystemExit(f"{option} must look like 74:75,4:5,11:12=Black (Z0:Z1,Y0:Y1,X0:X1=COLOUR)")
            paints.append(Paint(box, _color_id(colour_name), is_deep))
    return paints


def _base_zones(specs: list[str]) -> list[BaseZone]:
    zones = []
    for spec in specs:
        try:
            colour_name, rest = spec.split("=")
            kind, side, size = rest.split(":")
            if kind not in ("pond", "path") or side not in ("left", "right", "up", "down"):
                raise ValueError(spec)
            zones.append(BaseZone(_color_id(colour_name.strip()), kind, side, float(size)))
        except ValueError:
            raise SystemExit(
                '--base-zone must look like "Dark Azure=pond:right:18" or "Reddish Brown=path:down:3"'
            ) from None
    return zones


def _accents(specs: list[str]) -> list[Accent]:
    accents = []
    for spec in specs:
        try:
            box_text, rest = spec.split("=", 1)
            fields = rest.split(":")
            size, _, only_colour_name = fields[1].partition("@")
            width, depth = size.lower().split("x")
            accents.append(
                Accent(
                    _box_from_text(box_text),
                    _color_id(fields[0].strip()),
                    int(width),
                    int(depth),
                    ":".join(fields[2:]).strip(),
                    _color_id(only_colour_name.strip()) if only_colour_name.strip() else None,
                )
            )
        except (ValueError, IndexError):
            raise SystemExit('--accent must look like "44:48,10:13,17:22=Chrome Silver:1x2:note"') from None
    return accents


def _rgb(text: str, option: str) -> tuple[int, int, int]:
    """'R,G,B' -> (r, g, b); exits naming the option if it is not 3 whole numbers."""
    try:
        red, green, blue = (int(value) for value in text.split(","))
    except ValueError:
        raise SystemExit(f"{option} must look like 128,128,128 (R,G,B), got {text!r}") from None
    return red, green, blue


def _colour_ids(names_text: str | None) -> list[int]:
    """'Black, White' -> palette ids (exits on an unknown colour); empty for None / ''."""
    return [_color_id(name) for name in names_text.split(",")] if names_text else []


def _voxel_options(args: argparse.Namespace) -> VoxelOptions:
    if args.voxel_supersample > 1 and args.voxel_supersample % 2 == 0:
        raise SystemExit("--voxel-supersample must be odd (3, 5): cell edges then line up with the stud grid")
    box = (
        _box_option(args.voxel_box, "--voxel-box must look like 4:40,0:20,12:25 (Z0:Z1,Y0:Y1,X0:X1)")
        if args.voxel_box
        else None
    )
    return VoxelOptions(
        supersample=max(1, args.voxel_supersample),
        threshold=args.voxel_threshold,
        thin_threshold=args.voxel_thin_threshold,
        box=box,
        box_threshold=args.voxel_box_threshold if box else None,
    )


def _colour_options(args: argparse.Namespace) -> ColourOptions:
    defaults = ColourOptions()
    alias = (
        {
            _color_id(a.strip()): _color_id(b.strip())
            for a, b in (pair.split("=") for pair in args.color_alias.split(","))
        }
        if args.color_alias
        else {}
    )
    return ColourOptions(
        allowed_colour_ids=_colour_ids(args.colors) or None,
        colour_alias=alias,
        max_colours=args.max_colors,
        colour_groups=args.color_groups,
        smooth_iterations=args.smooth_iters,
        absorb_min_region=args.absorb_min_region,
        missing_rgb=_rgb(args.missing_rgb, "--missing-rgb") if args.missing_rgb else None,
        trim_colour_ids=_colour_ids(args.trim_colors),
        trim_min_fraction=args.trim_min_fraction,
        trim_snap_layers=args.trim_snap,
        trim_snap_min_area_ratio=args.trim_snap_similarity,
        despeckle_max_cells=args.despeckle,
        neutral_enclaves=args.neutral_enclaves,
        dark_enclave_max_cells=args.dark_enclaves,
        front_ovals=[parse_front_oval(spec) for spec in args.front_oval],
        paints=_paints(args),
        accents=_accents(args.accent),
        carve_gap=bool(args.carve_gap),
        carve_colour_id=_color_id(args.carve_color) if args.carve_gap else defaults.carve_colour_id,
        carve_from_height=args.carve_gap[0] if args.carve_gap else defaults.carve_from_height,
        carve_to_height=args.carve_gap[1] if args.carve_gap else defaults.carve_to_height,
    )


def _shape_options(args: argparse.Namespace) -> ShapeOptions:
    defaults = ShapeOptions()
    prop_box = (
        _box_option(args.prop_box, "--prop-box must look like 8:20,18:27,18:27 (Z0:Z1,Y0:Y1,X0:X1)")
        if args.prop_box
        else None
    )
    thicken_box = (
        _box_option(args.thicken_box, "--thicken-box must look like 10:24,16:26,16:26 (Z0:Z1,Y0:Y1,X0:X1)")
        if args.thicken_box
        else None
    )
    sheet_message = "--thicken-sheet-box must look like 70:82,0:20,0:20 (Z0:Z1,Y0:Y1,X0:X1)"
    return ShapeOptions(
        front={"down": "-y", "up": "+y", "left": "-x", "right": "+x"}.get(args.front),
        shell_thickness_studs=args.shell,
        thicken_thin=thicken_box is not None,
        thicken_box=thicken_box,
        thicken_sheet_boxes=[_box_option(spec, sheet_message) for spec in args.thicken_sheet_box],
        sheet_direction=1 if args.sheet_dir == "up" else -1,
        prop_box=prop_box,
        prop_size_studs=args.prop_size if prop_box else defaults.prop_size_studs,
        prop_colour_id=_color_id(args.prop_color) if prop_box and args.prop_color else None,
        drop_boxes=[parse_box(spec, "--drop-box") for spec in args.drop_box],
        fill_pinholes_max_cells=args.fill_pinholes,
        open_gaps=args.open_gaps,
        join_corners_max_cells=args.join_corners,
        ground_overhangs_max_plates=args.ground_overhangs,
        ground_min_column_plates=args.ground_min_column,
    )


def _repair_options(args: argparse.Namespace) -> RepairOptions:
    return RepairOptions(
        tie_repair=not args.no_ties,
        tie_max_recoloured_cells=max(0, args.tie_recolor),
        tie_min_hanging_pieces=max(1, args.tie_min_hang),
        tie_drop_max_pieces=max(0, args.tie_drop),
        retile=not args.no_retile,
        retile_box=parse_box(args.retile_box, "--retile-box") if args.retile_box else None,
        retile_reach_studs=max(1, args.retile_reach),
        retile_drop_max_pieces=max(0, args.retile_drop),
        cleanup_small=args.cleanup_small,
        cleanup_extra_stacks=max(0, args.cleanup_stacks),
        cleanup_extra_seams=max(0, args.cleanup_seams),
        use_l_pieces=args.use_l_pieces,
        l_skip_boxes=[parse_box(spec, "--l-skip-box") for spec in args.l_skip_box],
    )


def _base_options(args: argparse.Namespace) -> BaseOptions:
    defaults = BaseOptions()
    plates, brick_bands = max(0, args.base), max(0, args.base_bricks)
    if brick_bands and plates < 3 * brick_bands + 1:
        raise SystemExit(
            f"--base-bricks {args.base_bricks} needs --base of at least {3 * args.base_bricks + 1} "
            f"(bricks of 3 plates plus at least one plate on top to tie them together)"
        )
    layered = _layered_base_options(args)
    layout = []
    if args.base_layout:
        try:
            layout = base_layout.load_layout(args.base_layout, _color_id)
        except (OSError, ValueError) as error:
            raise SystemExit(f"--base-layout: {error}") from None
    if args.base_fill_colors and not args.stock:
        raise SystemExit("--base-fill-colors needs --stock (it uses spare owned pieces)")
    is_margin_auto = str(args.base_margin).lower() == "auto"
    return BaseOptions(
        plates=plates,
        brick_bands=brick_bands,
        margin_studs=defaults.margin_studs if is_margin_auto else max(0, int(args.base_margin)),
        is_margin_auto=is_margin_auto,
        min_tip_angle_degrees=args.min_tip_angle,
        shape=args.base_shape,
        corner_radius_studs=max(0, args.base_radius),
        colour_id=_color_id(args.base_color),
        footprint_layers=max(0, args.base_from_layers),
        zones=_base_zones(args.base_zone),
        zone_seed=args.base_seed,
        hidden_any_colour=not args.base_strict,
        layered=layered,
        layout=layout,
        layout_path=args.base_layout or "",
        fill_colour_ids=_colour_ids(args.base_fill_colors),
    )


def options_from_arguments(args: argparse.Namespace) -> BuildOptions:
    """All build settings from the parsed command line; exits with a clear message on a bad value."""
    return BuildOptions(
        voxel=_voxel_options(args),
        colour=_colour_options(args),
        shape=_shape_options(args),
        packing=PackingOptions(
            hidden_any_colour=not args.strict_hidden,
            prefer_bricks=args.prefer_bricks,
            harmonize_bricks=args.harmonize,
            merge_plates=args.merge_plates,
            anchor_floating=args.anchor_floating,
        ),
        repair=_repair_options(args),
        base=_base_options(args),
        output=OutputOptions(
            write_views=not args.no_views,
            debug=args.debug,
            make_pdf=args.pdf,
            show_layer_below=not args.no_layer_below,
        ),
    )


def load_stock(stock_path: str | None, base: BaseOptions) -> Stock | None:
    """The owned pieces from --stock (None without it), with a console summary."""
    if not stock_path:
        return None
    stock = Stock.from_csv(stock_path)
    print(f"Stock: {sum(stock.have.values())} pieces in {len(stock.have)} type/size/colour groups from {stock_path}")
    if base.plates and not stock.owns_colour(base.colour_id):
        print(
            f"  note: no {COLOR_NAMES[base.colour_id]} in stock, so the whole base will be on the "
            f"buy list; pick an owned colour with --base-color"
        )
    return stock


def write_stock_report(model: LegoModel, outdir: str, model_name: str, stock: Stock) -> None:
    """have / used / still need, counted from the final pieces (not from the running ledger stock)."""
    final = stock.recount(model.bricks)
    if +final.used != +stock.used:
        print("WARNING: the running stock ledger differs from the final pieces; the report uses the final pieces")
    report = os.path.join(outdir, f"{model_name}_stock_report.csv")
    buy = os.path.join(outdir, f"{model_name}_buy_list.csv")
    final.write_report(report, buy)
    print(final.summary())
    print(f"  -> {report}\n  -> {buy}")


class _Tee:
    """Write everything to the console AND to a log file (flushed at once, so the console stays live)."""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        """Write to both, flushed."""
        self.stream.write(text)
        self.stream.flush()
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self) -> None:
        """Flush both."""
        self.stream.flush()
        self.log.flush()

    def isatty(self) -> bool:
        """As the console stream (progress bars check it)."""
        return self.stream.isatty()

    def __getattr__(self, name: str) -> object:
        return getattr(self.stream, name)


def start_log(outdir: str, name: str = "run_log.txt") -> TextIO:
    """Copy console output (stdout + stderr, incl. progress bars) to <outdir>/<name>; first line = the command.
    Returns the log file (open for the rest of the run)."""
    os.makedirs(outdir, exist_ok=True)
    # stays open for the whole run: every console line is copied into it
    log = open(os.path.join(outdir, name), "w", encoding="utf-8", buffering=1)  # noqa: SIM115
    log.write("python " + " ".join(f'"{a}"' if " " in a else a for a in sys.argv) + "\n\n")
    # _Tee duck-types a text stream (unknown attributes go to the console stream)
    sys.stdout = cast(TextIO, _Tee(sys.stdout, log))
    sys.stderr = cast(TextIO, _Tee(sys.stderr, log))
    return log


def run_pipeline(
    build_input: BuildInput, outdir: str, options: BuildOptions, stock: Stock | None = None, *, keep_model: bool = False
) -> None:
    """Build the model and write everything: stability report, inventory, maps, views, PDF (options.output), the
    elevation views and, with stock, the stock report. keep_model: also pickle the model (--save-model)."""
    os.makedirs(outdir, exist_ok=True)
    model_name = build_input.model_name
    model = build_model(build_input, outdir, options=options, stock=stock)
    if keep_model:
        save_model(model, os.path.join(outdir, f"{model_name}_model.pkl"))
    stability.report(model, outdir, model_name, options.base.min_tip_angle_degrees)
    save_outputs(model, model_name, outdir, stock, options.output)
    if options.output.write_views:
        for path in elevation.save_elevations(model, outdir, model_name, model.base_margin):
            print(f"  -> {path}")
    if stock is not None:
        write_stock_report(model, outdir, model_name, stock)


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Input model, size, rotation and voxelisation."""
    parser.add_argument("input", help="Path to Untitled.glb")
    parser.add_argument("--outdir", default="lego_output", help="Output directory")
    parser.add_argument("--target-studs", type=int, default=32, help="Target horizontal size in studs (default: 32)")
    parser.add_argument("--hollow", action="store_true", help="Hollow out fully interior cells to save pieces")
    parser.add_argument(
        "--rotate-x", type=float, default=0.0, help="Rotate the imported model around X by this many degrees"
    )
    parser.add_argument(
        "--rotate-y", type=float, default=0.0, help="Rotate the imported model around Y by this many degrees"
    )
    parser.add_argument(
        "--rotate-z", type=float, default=0.0, help="Rotate the imported model around Z by this many degrees"
    )
    parser.add_argument(
        "--voxel-box",
        default=None,
        metavar="Z0:Z1,Y0:Y1,X0:X1",
        help="Region (figure grid, ends exclusive) where --voxel-box-threshold replaces the thin "
        "threshold, to keep a very thin part such as a rapier blade (~0.04 of a cell) "
        "without speckles from whiskers/hair elsewhere. Needs --voxel-supersample.",
    )
    parser.add_argument(
        "--voxel-box-threshold",
        type=float,
        default=0.02,
        help="Thin threshold inside --voxel-box (default %(default)s)",
    )
    parser.add_argument(
        "--voxel-supersample",
        type=int,
        default=1,
        metavar="N",
        help="Odd N (3 recommended): voxelise at N x resolution and keep a cell only if enough of it "
        "is inside. The default surface voxelisation fattens every part by about half a stud "
        "per side (legs fuse, thin tail / sword blobs); N=3 gives the true volume, "
        "roughly 40%% fewer cells. Slower (about N^2 more surface points). Default 1 = off",
    )
    parser.add_argument(
        "--voxel-threshold",
        type=float,
        default=DEFAULTS.voxel.threshold,
        help="--voxel-supersample: fraction of a cell that must be inside to keep it "
        "(0.5 = half; lower = fatter, default %(default)s)",
    )
    parser.add_argument(
        "--voxel-thin-threshold",
        type=float,
        default=DEFAULTS.voxel.thin_threshold,
        help="--voxel-supersample: same for thin parts (tail, blade, brim edge) that would "
        "otherwise break up; the result is reconnected through them (default %(default)s)",
    )
    parser.add_argument(
        "--boxes-for-studs",
        type=int,
        default=None,
        metavar="N",
        help="The boxes (--thicken-box, --accent, --retile-box, --prop-box, --voxel-box) were written "
        "for --target-studs N; they are scaled to the real --target-studs, so the same command "
        "works for a smaller / bigger figure (box ends are rounded outward)",
    )


def _add_colour_arguments(parser: argparse.ArgumentParser) -> None:
    """Palette, colour clean-up, accents and paint."""
    parser.add_argument(
        "--colors",
        default=None,
        help='Comma-separated LEGO colour names to allow, e.g. "Orange,Black,Dark Red". Default: whole palette.',
    )
    parser.add_argument(
        "--color-alias",
        default=None,
        help='Classify with one colour, build with another: "Dark Red=Red,Orange=Dark Orange". '
        "The first colour must be in --colors; the second is what is built (e.g. one you own).",
    )
    parser.add_argument(
        "--trim-colors",
        default=None,
        help="Thin trim colours: each trim feature (hat band, one boot cuff) becomes ONE full ring in "
        'the layer where it has most cells; its other cells are recoloured from the neighbours, e.g. "Dark Red". '
        "Applied before --color-alias.",
    )
    parser.add_argument(
        "--trim-min-fraction",
        type=float,
        default=DEFAULTS.colour.trim_min_fraction,
        help="Share of the ring the trim cluster must already cover to complete it (default %(default)s)",
    )
    parser.add_argument(
        "--accent",
        action="append",
        default=[],
        metavar="Z0:Z1,Y0:Y1,X0:X1=COLOUR:WxD[@ONLYCOLOUR][:note]",
        help="Swap the highest WxD plate inside the box for a special part, e.g. "
        '"44:48,10:13,17:22=Chrome Silver:1x2:P121 plate with ball". It is listed on the cover and '
        "on its step page. Repeatable. Box coordinates as --thicken-box.",
    )
    parser.add_argument(
        "--prop-box",
        default=None,
        metavar="Z0:Z1,Y0:Y1,X0:X1",
        help="Add a post under the lowest hanging cell inside this box (e.g. a tail tip) down to "
        "the ground, so the part can be built bottom-up. Same coordinates as --thicken-box.",
    )
    parser.add_argument("--prop-size", type=int, default=2, help="Post is N x N studs (default %(default)s)")
    parser.add_argument("--prop-color", default=None, help="Colour of the post (default Black)")
    parser.add_argument(
        "--color-groups",
        type=int,
        default=DEFAULTS.colour.colour_groups,
        help="Number of colour groups (k-means clusters) the surface colours are split into before "
        "each group gets one LEGO colour; more groups keep small details such as a nose "
        "(0 = off, default %(default)s)",
    )
    parser.add_argument(
        "--smooth-iters",
        type=int,
        default=DEFAULTS.colour.smooth_iterations,
        help="Speckle clean-up passes (majority vote of the 6 neighbours); 0 = off, keeps 1-2 cell "
        "details such as eyes and nose (default %(default)s)",
    )
    parser.add_argument(
        "--absorb-min-region",
        type=int,
        default=DEFAULTS.colour.absorb_min_region,
        help="Colour regions smaller than this many cells are absorbed by their neighbours; "
        "1 = off (default %(default)s)",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=DEFAULTS.colour.max_colours,
        help="Keep only the N most-used colours (0 = no limit, default %(default)s)",
    )
    parser.add_argument(
        "--missing-rgb",
        default=None,
        help='RGB of "no data" grey in the input mesh to fill from neighbours, e.g. 192,196,205',
    )
    parser.add_argument(
        "--fill-pinholes",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="Close see-through holes of up to N cells in thin parts (a 1-cell hat brim that lost a "
        "cell when voxelised); bigger gaps (between legs, under an arm) stay. 0 = off; 3 suits an 18-stud figure",
    )
    parser.add_argument(
        "--trim-snap",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="--trim-colors: trim that did not become a ring of its own (a second boot cuff, tilted or "
        "higher up) gets a full ring in the nearest trim ring layer within N layers, on the part "
        "right below / above it; every other trim cell takes its neighbours' colour, so each ring "
        "is exactly one layer. 0 = off; 12 suits an 18-stud figure",
    )
    parser.add_argument(
        "--trim-snap-similarity",
        type=_ratio_from_zero_to_one,
        default=DEFAULTS.colour.trim_snap_min_area_ratio,
        metavar="R",
        help="--trim-snap: a snapped ring only goes on a part whose cross-section area is within "
        "[R, 1/R] of a part that already has a ring in that layer (a second boot, not the "
        "smaller tail beside it); the trim must also sit on the same body. 0 = no size check. "
        "Default %(default)s",
    )
    parser.add_argument(
        "--open-gaps",
        action="store_true",
        help="Remove single stray cells that stick into a gap and touch the part on the other side "
        "only at a corner (e.g. a heel cell between two boots): no neighbour above or below, one "
        "side neighbour, and the only link between two parts of its layer. Runs before "
        "--fill-pinholes",
    )
    parser.add_argument(
        "--despeckle",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="Colour clean-up: lone patches of up to N surface cells per layer (a colour none of their "
        "neighbours share) take the neighbours' most common colour. Paints, accents and trim "
        "colours (--trim-colors) stay. 0 = off",
    )
    parser.add_argument(
        "--dark-enclaves",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="Colour clean-up: a region of up to N surface cells of another colour inside a Black part "
        "(border mostly Black, e.g. feather colour on a hat crown) becomes Black. Paints, accents "
        "and trim colours stay. 0 = off; 5 suits an 18-stud figure",
    )
    parser.add_argument(
        "--join-corners",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="Thin parts that run diagonally (a hat flap) touch only at corners: fill one cell between "
        "two cells that touch only at a corner when the smaller of their parts in that layer has "
        "<= N cells and they are one body nearby (two boots touching at a corner stay apart). "
        "Runs after --open-gaps / --fill-pinholes. 0 = off; 3 suits an 18-stud figure",
    )
    parser.add_argument(
        "--neutral-enclaves",
        action="store_true",
        help="Colour clean-up: a light grey / white region whose border is (almost) all colourful "
        "(e.g. a highlight on orange fur matched to Light Bluish Gray) takes the border colours",
    )
    parser.add_argument(
        "--paint",
        action="append",
        default=[],
        metavar="Z0:Z1,Y0:Y1,X0:X1=COLOUR",
        help="Repeatable. Paint every visible cell in the box (figure grid = report coordinates minus "
        "the base height and the base margin; ends exclusive; real figure size, NOT scaled by "
        "--boxes-for-studs) with COLOUR, e.g. eyes and a nose. Applied after the colour "
        "clean-up, so 1-cell details survive. Later paints cover earlier ones",
    )
    parser.add_argument(
        "--front-oval",
        action="append",
        default=[],
        metavar="Z0:Z1,H0:H1=COLOUR,COLOUR,...",
        help="Repeatable. Regular fur pattern on the front: concentric oval bands inside the box (front "
        "view numbers: z, then the horizontal axis = x when the face looks down/up in the maps), "
        "first colour in the middle, last at the rim, e.g. 24:58,6:16=Yellow,Bright Light Orange,"
        "Orange,Dark Orange. Only cells that already have one of these colours change (belt, "
        "cape stay); the wall behind each front cell gets the same colour. Real figure size like "
        "--paint; applied before --paint.",
    )
    parser.add_argument(
        "--paint-deep",
        action="append",
        default=[],
        metavar="Z0:Z1,Y0:Y1,X0:X1=COLOUR",
        help="Like --paint, but also the hidden wall cells inside the box (every filled cell), so a "
        "piece that sits in the wall (e.g. a 1x2 brick as a nose, one stud hidden) is one "
        "colour all through. Painted in the order --paint first, then --paint-deep",
    )


def _add_shape_arguments(parser: argparse.ArgumentParser) -> None:
    """Shape changes: thickening, carving, shell, grounding, dropped parts."""
    parser.add_argument(
        "--thicken-box",
        default=None,
        metavar="Z0:Z1,Y0:Y1,X0:X1",
        help="Make the 1-stud-wide parts inside this box sturdier (a plus-shaped cross-section, "
        "so layers overlap by 2+ studs instead of hanging by one stud), e.g. a thin tail. "
        "Coordinates are the figure's own grid (layer, y, x), ends exclusive; a whole "
        "figure has many 1-wide edges, so always give a box.",
    )
    parser.add_argument(
        "--carve-gap",
        type=float,
        nargs=2,
        metavar=("FROM", "TO"),
        default=None,
        help="Separate a fused cape/coat between these height fractions (0=bottom, 1=top)",
    )
    parser.add_argument(
        "--carve-color", default="Black", help="Colour of the layer to separate with --carve-gap (default Black)"
    )
    parser.add_argument(
        "--shell",
        type=int,
        default=1,
        help="With --hollow: wall thickness in studs (default 1; 2 = sturdier, bigger bricks)",
    )
    parser.add_argument(
        "--ground-overhangs",
        type=_non_negative_int,
        default=0,
        metavar="G",
        help="A wall column (boot, leg) whose lowest cell starts up to G plates above the cell under "
        "it is extended down to it, so it no longer floats while building (it can then be a "
        "brick from the bottom). Only columns at least --ground-min-column plates tall; hat "
        "brims, tail tips and boot folds (bigger gaps) stay. 0 = off; 2 suits an 18-stud figure",
    )
    parser.add_argument(
        "--ground-min-column",
        type=_non_negative_int,
        default=DEFAULTS.shape.ground_min_column_plates,
        metavar="R",
        help="--ground-overhangs: smallest column height (plates) that is extended (default %(default)s)",
    )
    parser.add_argument(
        "--drop-box",
        action="append",
        default=[],
        metavar="Z0:Z1,Y0:Y1,X0:X1",
        help="Repeatable. Leave the filled cells in this box out of the figure (figure grid, ends "
        "exclusive, real figure size like --paint; read the numbers off the flat views), e.g. a "
        "one-cell bump that would hang in the air. Applied after all other shape fixes.",
    )
    parser.add_argument(
        "--thicken-sheet-box",
        action="append",
        default=[],
        metavar="Z0:Z1,Y0:Y1,X0:X1",
        help="Repeatable. Inside the box (figure grid, ends exclusive, like --thicken-box) every "
        "one-plate-thick sheet (hat brim, cape edge: nothing above or below) gets a second plate, "
        "so its pieces overlap and no longer float (tie repair would otherwise remove them)",
    )
    parser.add_argument(
        "--sheet-dir",
        choices=["up", "down"],
        default="up",
        help="--thicken-sheet-box: add the second plate above (default) or below the sheet",
    )


def _add_packing_arguments(parser: argparse.ArgumentParser) -> None:
    """Packing, tie repair, re-tiling, clean-up and L pieces."""
    parser.add_argument(
        "--strict-hidden",
        action="store_true",
        help="Hidden (never visible) cells keep their colour instead of matching any colour",
    )
    parser.add_argument(
        "--harmonize",
        action="store_true",
        help="Unify colours inside each 3-plate band so more 3-plate bricks fit (fewer pieces, thin details may shift)",
    )
    parser.add_argument(
        "--prefer-bricks",
        action="store_true",
        help="Use as many 3-plate bricks as possible and as few plates: bricks are placed first over "
        "the whole figure, wherever 3 layers in a row agree in colour (not only on fixed 3-layer "
        "bands), and are bought even when owned plates of that colour are left",
    )
    parser.add_argument(
        "--merge-plates",
        action="store_true",
        help="After all other steps: 2-3 neighbouring plates that fill a standard plate exactly "
        "(two 1x1 -> 1x2, a 1x2 + two 2x1 -> 2x3, ...) become one plate, if no visible colour "
        "changes (hidden cells may take the neighbours' colour), nothing more must be bought "
        "and the stability checks are not worse",
    )
    parser.add_argument(
        "--anchor-floating",
        type=int,
        choices=(0, 1, 2),
        default=0,
        metavar="LEVEL",
        help="Fewer pieces that hang in the air while building: a plate with nothing under it is "
        "re-tiled with a same-colour neighbour in its layer that rests on the layer below, so "
        "it gets a stud on that layer. 1 = plate neighbours only (few extra pieces), 2 = also "
        "split a neighbouring brick into plates (more pieces, more anchored). Cells, colours "
        "and strength checks unchanged. 0 = off",
    )
    parser.add_argument(
        "--cleanup-small",
        action="store_true",
        help="Lossless clean-up after packing: re-tile same-colour groups around 1x1 pieces to cut "
        "the number of 1x1 / total pieces. No colour changes; a change is kept only if the wall "
        "is not weaker (see --cleanup-stacks / --cleanup-seams), the structure is not worse and, "
        "with --stock, nothing more has to be bought",
    )
    parser.add_argument(
        "--cleanup-stacks",
        type=int,
        default=0,
        help="--cleanup-small: extra exact stacks (a piece on an identical piece) allowed per group "
        "(default 0 = never; 1-2 = fewer 1x1 pieces, straighter columns)",
    )
    parser.add_argument(
        "--cleanup-seams",
        type=int,
        default=0,
        help="--cleanup-small: extra aligned joint segments allowed per group (default 0)",
    )
    parser.add_argument(
        "--use-l-pieces",
        action="store_true",
        help="Last pass (needs --stock): swap small pieces that cover exactly an L-shaped area for one of "
        "your owned L pieces (corner plate / brick 2x2, 3 studs). Visible cells keep their colour, "
        "hidden cells take any owned colour; kept only if the structure does not get worse",
    )
    parser.add_argument(
        "--l-skip-box",
        action="append",
        default=[],
        metavar="Z0:Z1,Y0:Y1,X0:X1",
        help="Repeatable. --use-l-pieces never swaps pieces that touch this box (figure grid, ends "
        "exclusive, real figure size like --paint), e.g. to keep the eyes as 2x1 plates",
    )
    parser.add_argument(
        "--no-ties", action="store_true", help="Skip the tie repair that makes loose / single-stud parts hold together"
    )
    parser.add_argument(
        "--tie-recolor",
        type=int,
        default=DEFAULTS.repair.tie_max_recoloured_cells,
        help="Tie repair: max visible cells recoloured per tie, 0 = never (default %(default)s)",
    )
    parser.add_argument(
        "--tie-min-hang",
        type=int,
        default=DEFAULTS.repair.tie_min_hanging_pieces,
        help="Tie repair: fix single-stud joints that hang at least this many pieces (default %(default)s)",
    )
    parser.add_argument(
        "--tie-drop",
        type=int,
        default=DEFAULTS.repair.tie_drop_max_pieces,
        help="Tie repair: remove floating groups of at most this many pieces that "
        "cannot be tied to anything, 0 = keep them (default %(default)s)",
    )
    parser.add_argument(
        "--no-retile",
        action="store_true",
        help="Skip the re-tiling around single-stud joints that the ties could not fix",
    )
    parser.add_argument(
        "--retile-box",
        default=None,
        metavar="Z0:Z1,Y0:Y1,X0:X1",
        help="Re-tile only joints whose root piece starts inside this box (figure grid, ends "
        "exclusive, same coordinates as --thicken-box); default = every weak joint",
    )
    parser.add_argument(
        "--retile-reach",
        type=int,
        default=DEFAULTS.repair.retile_reach_studs,
        help="Re-tiling: largest window (studs around the joint) whose pieces are re-tiled (default %(default)s)",
    )
    parser.add_argument(
        "--retile-drop",
        type=int,
        default=DEFAULTS.repair.retile_drop_max_pieces,
        help="Re-tiling: pieces it may leave floating, and so remove, per joint; 0 = never remove "
        "anything (default %(default)s; capped by --tie-drop)",
    )


def _add_base_arguments(parser: argparse.ArgumentParser) -> None:
    """The stand (base plates) and the owned-piece stock."""
    parser.add_argument(
        "--base",
        type=int,
        nargs="?",
        const=2,
        default=0,
        metavar="PLATES",
        help="Build the figure on a stand of N plates (default 2 when given without a number)",
    )
    parser.add_argument(
        "--base-bricks",
        type=int,
        default=0,
        metavar="N",
        help="Build the lowest N bands of 3 base layers from 3-plate bricks (the plates above tie "
        "them together, so --base must be at least 3N+1, e.g. --base 4 --base-bricks 1)",
    )
    parser.add_argument(
        "--base-hidden-layers",
        type=int,
        default=0,
        metavar="N",
        help="Layered base (big plates first, like the LEGO Architecture sets): the lowest N base "
        "layers are hidden plates of any (owned) colour, the layers above show the base colour "
        "and zones; each layer spans the joints of the one above. --base must be at least N+1, "
        "e.g. --base 3 --base-hidden-layers 2. Default 0 = the plate / brick base packer",
    )
    parser.add_argument(
        "--base-sparse-layers",
        type=int,
        default=0,
        metavar="N",
        help="With --base-hidden-layers: the lowest N hidden layers are sparse tie layers (like the "
        "LEGO Sagrada Familia base): pieces only across the seams of the layer above, at least "
        "one every --base-tie-every studs, plus feet at the corners of the bottom layer. "
        "Default 0 = full layers",
    )
    parser.add_argument(
        "--base-tie-every",
        type=int,
        default=base_ties.BASE_TIE_EVERY,
        metavar="STUDS",
        help="--base-sparse-layers: studs of seam per tie (default %(default)s; smaller = stiffer)",
    )
    parser.add_argument(
        "--base-max-plate",
        type=_plate_size,
        default=None,
        metavar="WxD",
        help="With --base-hidden-layers: largest plate the base may use, owned or bought "
        "(default 16x16; bought plates are the sizes in base_packer.BASE_BUY_SIZES)",
    )
    parser.add_argument(
        "--base-margin",
        default="auto",
        help='Studs of base around the footprint, or "auto" (default): the smallest margin '
        "that keeps the tip angle at or above --min-tip-angle",
    )
    parser.add_argument(
        "--base-shape",
        choices=["rect", "rounded"],
        default=DEFAULTS.base.shape,
        help="Base outline (default %(default)s)",
    )
    parser.add_argument(
        "--base-radius",
        type=int,
        default=DEFAULTS.base.corner_radius_studs,
        help="Corner radius in studs for --base-shape rounded (default %(default)s)",
    )
    parser.add_argument(
        "--base-color",
        default=COLOR_NAMES[DEFAULTS.base.colour_id],
        help='LEGO colour of the base (default "%(default)s")',
    )
    parser.add_argument(
        "--base-layout",
        default="",
        metavar="FILE",
        help="Use a hand-built base exactly: CSV with layer (1 = bottom), x, y, width, depth, color "
        "(+ optional shape for a special piece, e.g. a technic plate), x / y = studs from the "
        "base corner as on the PDF base maps. Placed where the figure tips least; seams the "
        "layer below does not tie are listed. --base must equal the number of layers",
    )
    parser.add_argument(
        "--base-fill-colors",
        default="",
        metavar="COLOURS",
        help="With --stock: base pieces whose colour you do not own enough of take a SPARE owned "
        'piece of the same size in the first of these colours that has one, e.g. "Black,Blue,'
        'Reddish Brown"; spare = owned minus what the whole build uses, so no other layer '
        "loses a piece. Empty = off",
    )
    parser.add_argument(
        "--base-from-layers",
        type=int,
        default=DEFAULTS.base.footprint_layers,
        help="Base footprint = projection of the lowest N figure layers (0 = whole figure, default %(default)s)",
    )
    parser.add_argument(
        "--stock",
        default=None,
        metavar="CSV",
        help="Bricks I own (same columns as the inventory CSV, so an earlier build's "
        "<name>_inventory.csv can be used). Owned pieces are used first, a size that runs out "
        "falls back to smaller owned pieces, anything else is bought. --colors stays a hard "
        "requirement (colours you do not own are built and listed as 'not owned'). "
        "Writes <name>_stock_report.csv (have / used / still need) and <name>_buy_list.csv",
    )
    parser.add_argument(
        "--base-strict",
        action="store_true",
        help="With --stock: the whole base keeps --base-color (default: base cells nobody sees - "
        "inside the rim and under the figure - may use any owned colour)",
    )
    parser.add_argument(
        "--min-tip-angle",
        type=float,
        default=DEFAULTS.base.min_tip_angle_degrees,
        help="Stability check: the model may tilt this many degrees about the base edge "
        "before it tips (default %(default)s)",
    )
    parser.add_argument(
        "--base-zone",
        action="append",
        default=[],
        metavar="COLOUR=pond:SIDE:PERCENT | COLOUR=path:SIDE:WIDTH",
        help="Paint a zone on the visible base (repeatable, later ones cover earlier ones; the rest "
        "stays --base-color). pond: an irregular patch covering PERCENT %% of the base near SIDE "
        "(left|right|up|down on the layer maps). path: a wavy band WIDTH studs wide from the "
        "middle of the figure out to SIDE. E.g. --base-color Green --base-zone "
        '"Dark Azure=pond:right:18" --base-zone "Reddish Brown=path:down:3"',
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULTS.base.zone_seed,
        help="Random seed for the shape of the base zones (default %(default)s)",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """What is written: PDF, log, views, debug images, saved model, FRONT arrow."""
    parser.add_argument("--pdf", action="store_true", help="Also generate an instruction PDF")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not copy the console output to <outdir>/run_log.txt (default: it is copied)",
    )
    parser.add_argument(
        "--no-layer-below",
        action="store_true",
        help="Layer maps (PNG + PDF) without the layer below (default: the pieces of the layer below "
        "are drawn hatched where the new layer leaves them uncovered, with a legend row)",
    )
    parser.add_argument(
        "--no-views",
        action="store_true",
        help="Do not write the flat front / back / left / right views (<name>_view_*.png, axes in the "
        "figure grid used by --paint / --thicken-box / --accent)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Extra diagnostics for developers: <name>_voxelized.png (raw voxel grid) and a check that "
        "the placed pieces cover the voxel grid exactly (<name>_placed_bricks.png, "
        "_missing_from_bricks.png, _extra_from_bricks.png; a WARNING if they differ)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Also save the placed-brick model (<name>_model.pkl) for later analysis",
    )
    parser.add_argument(
        "--front",
        choices=["down", "up", "left", "right"],
        default=None,
        help="Where the figure's face looks to on the layer maps (top-down, base plate at the bottom "
        "of the page): marked FRONT on every layer map and used by the {front_stud} "
        "placeholder in an --accent note",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """The command-line options (defaults from options.BuildOptions), grouped by topic."""
    parser = argparse.ArgumentParser(description="Convert an STL mesh into a LEGO brick model.")
    _add_input_arguments(parser)
    _add_colour_arguments(parser)
    _add_shape_arguments(parser)
    _add_packing_arguments(parser)
    _add_base_arguments(parser)
    _add_output_arguments(parser)
    return parser


def main() -> None:
    """Build the LEGO model and its instructions from the command line."""
    args = build_argument_parser().parse_args()
    if not args.no_log:
        start_log(args.outdir)
    options = options_from_arguments(args)
    stock = load_stock(args.stock, options.base)
    if args.boxes_for_studs and args.boxes_for_studs != args.target_studs:
        factor = args.target_studs / args.boxes_for_studs
        options = scale_boxes(options, factor)
        print(
            f"Boxes scaled x{factor:.3f} (written for {args.boxes_for_studs} studs, building {args.target_studs}): "
            f"thicken {options.shape.thicken_box}, accents {[accent.box for accent in options.colour.accents]}"
        )

    build_input = BuildInput(
        args.input, args.target_studs, (args.rotate_x, args.rotate_y, args.rotate_z), is_hollow=args.hollow
    )
    run_pipeline(build_input, args.outdir, options, stock, keep_model=args.save_model)


if __name__ == "__main__":
    main()
