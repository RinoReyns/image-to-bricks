"""Command-line layer of cli.py: argparse value types, option parsers, the clear error messages for bad
or conflicting options (checked in a real CLI process) and --help. Run: python tests/test_cli_options.py"""

import argparse
import types

import support

from image_to_bricks import cli
from image_to_bricks.consts import COLOR_NAMES

COLOUR_ID_BY_NAME = {name: colour_id for colour_id, name in COLOR_NAMES.items()}
MISSING_INPUT_FILE = "does_not_exist.glb"


def assert_raises(expected_error: type, call, *arguments, message_part: str = ""):
    """Call and check that it raises expected_error whose text contains message_part."""
    try:
        call(*arguments)
    except expected_error as error:
        assert message_part in str(error), f"{message_part!r} not in {str(error)!r}"
        return
    raise AssertionError(f"{call.__name__}{arguments} did not raise {expected_error.__name__}")


def run_cli_expecting_error(*arguments: str) -> str:
    """Run the CLI with options that must be rejected before any work starts; returns stdout + stderr."""
    process = support.run_cli(
        [MISSING_INPUT_FILE, "--outdir", support.make_temp_dir("cli"), *arguments], timeout_seconds=120
    )
    assert process.returncode != 0, f"options {arguments} were accepted:\n{process.stdout[-2000:]}"
    return process.stdout + process.stderr


def test_colour_names_are_case_insensitive_and_trimmed():
    assert cli._color_id("  dark red ") == COLOUR_ID_BY_NAME["Dark Red"]


def test_unknown_colour_name_lists_the_available_colours():
    assert_raises(SystemExit, cli._color_id, "Purple Rain", message_part="Available: White")


def test_ratio_type_accepts_zero_to_one_only():
    assert cli._ratio_from_zero_to_one("0.25") == 0.25
    assert_raises(argparse.ArgumentTypeError, cli._ratio_from_zero_to_one, "1.5", message_part="0 to 1")
    assert_raises(argparse.ArgumentTypeError, cli._ratio_from_zero_to_one, "abc", message_part="'abc'")


def test_non_negative_int_type_rejects_negatives_and_text():
    assert cli._non_negative_int("3") == 3
    assert_raises(argparse.ArgumentTypeError, cli._non_negative_int, "-1", message_part="0 or more")
    assert_raises(argparse.ArgumentTypeError, cli._non_negative_int, "2.5", message_part="whole number")


def test_plate_size_type_parses_and_rejects():
    assert cli._plate_size("8x8") == (8, 8)
    assert_raises(argparse.ArgumentTypeError, cli._plate_size, "big")


def test_parse_box_returns_slices_in_z_y_x_order():
    assert cli.parse_box("1:2,3:4,5:6", "--x") == (slice(1, 2), slice(3, 4), slice(5, 6))


def test_parse_box_names_the_option_in_its_error():
    assert_raises(SystemExit, cli.parse_box, "1:2,3:4", "--thicken-box", message_part="--thicken-box")
    assert_raises(SystemExit, cli.parse_box, "1:2:3,3:4,5:6", "--drop-box", message_part="--drop-box")


def test_parse_front_oval_reads_box_and_colours():
    oval = cli.parse_front_oval("24:58,6:16=Yellow, Orange")
    assert (oval.z0, oval.z1, oval.h0, oval.h1) == (24, 58, 6, 16)
    assert oval.colour_ids == (COLOUR_ID_BY_NAME["Yellow"], COLOUR_ID_BY_NAME["Orange"])


def test_parse_front_oval_rejects_bad_and_empty_specs():
    assert_raises(SystemExit, cli.parse_front_oval, "24:58=Yellow", message_part="must look like")
    assert_raises(SystemExit, cli.parse_front_oval, "58:24,6:16=Yellow", message_part="not be empty")
    assert_raises(SystemExit, cli.parse_front_oval, "24:58,6:16=", message_part="not be empty")


def layered_base_arguments(**overrides) -> types.SimpleNamespace:
    """The base options _layered_base_options reads, with the CLI defaults unless overridden."""
    values = dict(
        base=3, base_hidden_layers=2, base_sparse_layers=0, base_tie_every=1, base_max_plate=None, base_bricks=0
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_layered_base_is_off_without_hidden_layers():
    assert cli._layered_base_options(layered_base_arguments(base_hidden_layers=0)) is None


def test_layered_base_rejects_combinations_that_cannot_work():
    check = cli._layered_base_options
    assert_raises(
        SystemExit,
        check,
        layered_base_arguments(base_hidden_layers=0, base_sparse_layers=1),
        message_part="need --base-hidden-layers",
    )
    assert_raises(SystemExit, check, layered_base_arguments(base_sparse_layers=3), message_part="must be 0..2")
    assert_raises(SystemExit, check, layered_base_arguments(base_tie_every=0), message_part="at least 1")
    assert_raises(SystemExit, check, layered_base_arguments(base=2), message_part="--base of at least 3")
    assert_raises(SystemExit, check, layered_base_arguments(base_bricks=1), message_part="cannot be combined")


def test_layered_base_options_carry_the_values():
    options = cli._layered_base_options(layered_base_arguments(base_sparse_layers=1, base_tie_every=2))
    assert (options.hidden_layers, options.sparse_layers, options.tie_every) == (2, 1, 2)


def test_cli_help_lists_the_main_options():
    process = support.run_cli(["--help"], timeout_seconds=120)
    support.assert_cli_succeeded(process)
    for option in ("--target-studs", "--colors", "--stock", "--pdf", "--front"):
        assert option in process.stdout, option


def test_cli_rejects_an_unknown_colour_before_loading_the_mesh():
    assert "Unknown colour 'Purple Rain'" in run_cli_expecting_error("--colors", "Red,Purple Rain")


def test_cli_rejects_an_even_supersample():
    assert "--voxel-supersample must be odd" in run_cli_expecting_error("--voxel-supersample", "4")


def test_cli_rejects_a_malformed_paint_box():
    assert "must look like" in run_cli_expecting_error("--paint", "1:2,3:4=Red")


def test_cli_rejects_base_bricks_on_a_too_thin_base():
    assert "--base-bricks 1 needs --base of at least 4" in run_cli_expecting_error("--base", "2", "--base-bricks", "1")


def test_cli_rejects_base_fill_colours_without_stock():
    assert "--base-fill-colors needs --stock" in run_cli_expecting_error("--base", "2", "--base-fill-colors", "Green")


def test_cli_rejects_an_invalid_front_side():
    assert "invalid choice" in run_cli_expecting_error("--front", "sideways")


def test_cli_rejects_a_missing_input_file_with_an_error():
    output = run_cli_expecting_error()
    assert MISSING_INPUT_FILE in output, output[-1000:]


if __name__ == "__main__":
    support.run_tests(globals())
