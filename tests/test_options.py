"""BuildOptions (options.py): the command line becomes one typed options object with the right values and
defaults, and --boxes-for-studs scales its boxes."""

import operator

import support

support.stub_mesh_libraries()

from image_to_bricks import cli
from image_to_bricks.colours.fur_ovals import FrontOval
from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.options import Accent, BaseZone, BuildOptions, Paint, scale_boxes


def colour_id(name: str) -> int:
    return next(colour for colour, colour_name in COLOR_NAMES.items() if colour_name == name)


BLACK, WHITE = colour_id("Black"), colour_id("White")


def options_from_command_line(*arguments: str) -> BuildOptions:
    return cli.options_from_arguments(cli.build_argument_parser().parse_args(["model.glb", *arguments]))


def test_command_line_defaults_are_the_option_defaults() -> None:
    options = options_from_command_line()
    defaults = BuildOptions()
    assert options.voxel == defaults.voxel
    assert options.colour == defaults.colour
    assert options.repair == defaults.repair
    assert options.packing == defaults.packing
    assert options.output == defaults.output
    assert options.base == defaults.base
    assert options.shape == defaults.shape


def test_arguments_become_typed_options() -> None:
    options = options_from_command_line(
        "--colors", "Black,White",
        "--paint", "1:2,3:4,5:6=Black",
        "--paint-deep", "7:8,1:2,1:2=White",
        "--accent", "44:48,10:13,17:22=White:1x2@Black:buckle",
        "--base", "4",
        "--base-zone", "Dark Azure=pond:right:18",
        "--base-margin", "5",
        "--no-ties",
        "--retile-box", "0:10,0:5,0:5",
        "--cleanup-stacks", "2",
        "--front", "down",
        "--thicken-box", "10:24,16:26,16:26",
    )  # fmt: skip
    assert options.colour.allowed_colour_ids == [BLACK, WHITE]
    assert options.colour.paints == [
        Paint((slice(1, 2), slice(3, 4), slice(5, 6)), BLACK, is_deep=False),
        Paint((slice(7, 8), slice(1, 2), slice(1, 2)), WHITE, is_deep=True),
    ]
    assert options.colour.accents == [
        Accent((slice(44, 48), slice(10, 13), slice(17, 22)), WHITE, 1, 2, "buckle", BLACK)
    ]
    assert options.base.plates == 4
    assert options.base.zones == [BaseZone(colour_id("Dark Azure"), "pond", "right", 18.0)]
    assert options.base.margin_studs == 5
    assert not options.base.is_margin_auto
    assert not options.repair.tie_repair
    assert options.repair.retile_box == (slice(0, 10), slice(0, 5), slice(0, 5))
    assert options.repair.cleanup_extra_stacks == 2
    assert options.shape.front == "-y"
    assert options.shape.thicken_thin
    assert options.shape.thicken_box == (slice(10, 24), slice(16, 26), slice(16, 26))


OPTION_REACHES_FIELD = (  # command-line arguments -> the BuildOptions field (group.name) and its value
    (("--anchor-floating", "2"), "packing.anchor_floating", 2),
    (("--smooth-iters", "5"), "colour.smooth_iterations", 5),
    (("--absorb-min-region", "7"), "colour.absorb_min_region", 7),
    (("--drop-box", "1:2,3:4,5:6"), "shape.drop_boxes", [(slice(1, 2), slice(3, 4), slice(5, 6))]),
    (("--join-corners", "3"), "shape.join_corners_max_cells", 3),
    (("--dark-enclaves", "4"), "colour.dark_enclave_max_cells", 4),
    (("--merge-plates",), "packing.merge_plates", True),
    (("--prefer-bricks",), "packing.prefer_bricks", True),
    (("--paint", "1:2,1:2,1:2=Black"), "colour.paints", [Paint((slice(1, 2), slice(1, 2), slice(1, 2)), BLACK, False)]),
    (("--front-oval", "24:58,6:16=Black,White"), "colour.front_ovals", [FrontOval(24, 58, 6, 16, (BLACK, WHITE))]),
    (("--voxel-supersample", "3"), "voxel.supersample", 3),
    (("--no-views",), "output.write_views", False),
    (("--strict-hidden",), "packing.hidden_any_colour", False),
    (("--cleanup-seams", "3"), "repair.cleanup_extra_seams", 3),
    (("--retile-reach", "0"), "repair.retile_reach_studs", 1),
    (("--no-retile",), "repair.retile", False),
    (("--use-l-pieces",), "repair.use_l_pieces", True),
)


def test_every_listed_option_reaches_its_field() -> None:
    failures = []
    for arguments, field_path, expected in OPTION_REACHES_FIELD:
        value = operator.attrgetter(field_path)(options_from_command_line(*arguments))
        if value != expected:
            failures.append(f"{' '.join(arguments)}: {field_path} = {value!r}")
    assert not failures, failures


def test_boxes_scale_with_the_figure_size() -> None:
    options = scale_boxes(
        options_from_command_line(
            "--thicken-box", "10:34,12:25,12:25",
            "--accent", "36:46,5:14,15:23=Black:1x2:note",
            "--drop-box", "1:2,1:2,1:2",
        ),
        20 / 24,
    )  # fmt: skip
    assert options.shape.thicken_box == (slice(8, 29), slice(10, 21), slice(10, 21))
    assert options.colour.accents[0].box == (slice(30, 39), slice(4, 12), slice(12, 20))
    assert options.colour.accents[0][1:] == (BLACK, 1, 2, "note", None)
    assert options.repair.retile_box is None
    assert options.shape.drop_boxes == [(slice(1, 2), slice(1, 2), slice(1, 2))], "drop boxes are not scaled"
    tiny = scale_boxes(options_from_command_line("--thicken-box", "0:1,0:1,0:1"), 0.5)
    assert tiny.shape.thicken_box == (slice(0, 1), slice(0, 1), slice(0, 1)), "a box stays at least one cell wide"


def test_bad_accent_exits_with_the_expected_form() -> None:
    try:
        options_from_command_line("--accent", "1:2,1:2=Black")
    except SystemExit as exit_message:
        assert "--accent must look like" in str(exit_message)
    else:
        raise AssertionError("a malformed --accent must exit")


if __name__ == "__main__":
    support.run_tests(globals())
