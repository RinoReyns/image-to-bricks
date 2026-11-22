"""stock.Stock beyond packing: CSV loading rules (blank / total rows, unknown colours, specials, L pieces, missing
height, bad height), status per row, report + buy list files, printed summary, recount, give back, fork isolation.
Run: python tests/test_stock_ledger.py"""

import csv
import os

import support

from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.model import L_SHAPE, PlacedBrick
from image_to_bricks.stock import Stock

COLOUR_ID_BY_NAME = {name: colour_id for colour_id, name in COLOR_NAMES.items()}
RED = COLOUR_ID_BY_NAME["Red"]
BLUE = COLOUR_ID_BY_NAME["Blue"]
PLATE_HEIGHT = 1
BRICK_HEIGHT = 3
STOCK_HEADER = ["piece_type", "width_studs", "depth_studs", "height_plates", "color_name", "quantity", "shape"]


def write_stock_csv(rows: list) -> str:
    """Stock CSV with STOCK_HEADER columns in a new temp folder; returns its path."""
    path = os.path.join(support.make_temp_dir("stock"), "stock.csv")
    with open(path, "w", newline="", encoding="utf-8") as stock_file:
        writer = csv.writer(stock_file)
        writer.writerow(STOCK_HEADER)
        writer.writerows(rows)
    return path


def load_stock(rows: list) -> Stock:
    with support.quiet_stdout():
        return Stock.from_csv(write_stock_csv(rows))


def plate(width: int, depth: int, colour: int, notch: int = -1) -> PlacedBrick:
    return PlacedBrick(x=0, y=0, z=0, w=width, d=depth, h_plates=PLATE_HEIGHT, color=colour, notch=notch)


def test_loading_skips_blank_and_total_rows():
    stock = load_stock(
        [["Plate", 2, 1, 1, "Red", 4, ""], ["", "", "", "", "", "", ""], ["TOTAL PIECES", "", "", "", "", 4, ""]]
    )
    assert sum(stock.have.values()) == 4


def test_loading_ignores_colours_outside_the_palette():
    stock = load_stock([["Plate", 2, 1, 1, "Trans-Purple Sparkle", 9, ""], ["Plate", 2, 1, 1, "Red", 1, ""]])
    assert sum(stock.have.values()) == 1


def test_loading_treats_long_and_short_side_as_one_piece():
    stock = load_stock([["Plate", 1, 4, 1, "Red", 2, ""], ["Plate", 4, 1, 1, "Red", 3, ""]])
    assert stock.free(PLATE_HEIGHT, 4, 1, RED) == 5 == stock.free(PLATE_HEIGHT, 1, 4, RED)


def test_missing_height_comes_from_the_piece_type():
    stock = load_stock([["Brick", 2, 2, "", "Blue", 1, ""], ["Plate", 2, 2, "", "Blue", 1, ""]])
    assert stock.free(BRICK_HEIGHT, 2, 2, BLUE) == 1 and stock.free(PLATE_HEIGHT, 2, 2, BLUE) == 1


def test_a_height_other_than_plate_or_brick_is_an_error_with_the_line_number():
    path = write_stock_csv([["Plate", 2, 2, 2, "Blue", 1, ""]])
    try:
        with support.quiet_stdout():
            Stock.from_csv(path)
    except ValueError as error:
        assert "line 2" in str(error) and "1 or 3" in str(error)
        return
    raise AssertionError("height 2 must be rejected")


def test_special_pieces_are_kept_aside_and_l_pieces_counted():
    stock = load_stock(
        [["Plate (special)", 2, 2, 1, "Red", 3, L_SHAPE], ["Plate (special)", 4, 4, 1, "Red", 1, "wedge plate 4x4"]]
    )
    assert not stock.have, "special pieces are never packed as rectangles"
    assert sum(stock.specials.values()) == 4
    assert stock.l_free(PLATE_HEIGHT, RED) == 3


def test_row_status_tells_ok_short_not_owned_and_unused():
    stock = load_stock(
        [["Plate", 1, 1, 1, "Red", 2, ""], ["Plate", 2, 1, 1, "Red", 1, ""], ["Plate", 3, 1, 1, "Red", 5, ""]]
    )
    stock.take(PLATE_HEIGHT, 1, 1, RED, 2)
    stock.take(PLATE_HEIGHT, 2, 1, RED, 3)
    stock.take(PLATE_HEIGHT, 4, 1, BLUE, 1)
    status_by_size = {(row[1], row[2], row[4]): row[9] for row in stock.rows()}
    assert status_by_size[(1, 1, "Red")] == "ok"
    assert status_by_size[(2, 1, "Red")] == "short"
    assert status_by_size[(4, 1, "Blue")] == "not owned"
    assert status_by_size[(3, 1, "Red")] == "unused"


def test_report_and_buy_list_files_hold_the_missing_pieces():
    stock = load_stock([["Plate", 2, 1, 1, "Red", 1, ""]])
    stock.take(PLATE_HEIGHT, 2, 1, RED, 3)
    folder = support.make_temp_dir("report")
    report_path, buy_path = os.path.join(folder, "report.csv"), os.path.join(folder, "buy.csv")
    stock.write_report(report_path, buy_path)
    with open(buy_path, newline="") as buy_file:
        buy_rows = [row for row in csv.reader(buy_file) if row]
    assert buy_rows[1][:6] == ["Plate", "2", "1", "1", "Red", "2"], buy_rows
    assert buy_rows[-1][0] == "TOTAL PIECES" and buy_rows[-1][5] == "2", buy_rows[-1]
    with open(report_path, newline="") as report_file:
        report_total = [row for row in csv.reader(report_file) if row][-1]
    assert report_total[5:9] == ["1", "3", "2", "0"], report_total


def test_summary_counts_have_used_and_need_per_colour():
    stock = load_stock([["Plate", 2, 1, 1, "Red", 4, ""], ["Plate (special)", 2, 2, 1, "Red", 1, L_SHAPE]])
    stock.take(PLATE_HEIGHT, 2, 1, RED, 6)
    stock.take_brick(plate(2, 2, RED, notch=0))
    stock.take(PLATE_HEIGHT, 1, 1, BLUE, 2)
    summary_lines = stock.summary().splitlines()
    red_line = next(line for line in summary_lines if line.strip().startswith("Red"))
    blue_line = next(line for line in summary_lines if line.strip().startswith("Blue"))
    assert "have    5" in red_line and "used from stock    5" in red_line and "need    2" in red_line, red_line
    assert "NOT OWNED" in blue_line, blue_line
    assert "build = 9 pieces" in summary_lines[-1], summary_lines[-1]


def test_give_brick_returns_rectangles_and_l_pieces():
    stock = load_stock([["Plate", 2, 1, 1, "Red", 1, ""], ["Plate (special)", 2, 2, 1, "Red", 1, L_SHAPE]])
    rectangle, corner = plate(2, 1, RED), plate(2, 2, RED, notch=1)
    stock.take_brick(rectangle)
    stock.take_brick(corner)
    assert stock.free(PLATE_HEIGHT, 2, 1, RED) == 0 and stock.l_free(PLATE_HEIGHT, RED) == 0
    stock.give_brick(rectangle)
    stock.give_brick(corner)
    assert stock.free(PLATE_HEIGHT, 2, 1, RED) == 1 and stock.l_free(PLATE_HEIGHT, RED) == 1
    assert not stock.used and not stock.l_used, "returned pieces leave no zero entries behind"


def test_recount_counts_exactly_the_given_pieces():
    stock = load_stock([["Plate", 2, 1, 1, "Red", 1, ""]])
    stock.take(PLATE_HEIGHT, 2, 1, RED, 9)
    recounted = stock.recount([plate(2, 1, RED), plate(1, 2, RED), plate(2, 2, RED, notch=0)])
    assert recounted.free(PLATE_HEIGHT, 2, 1, RED) == -1
    assert recounted.l_used[(PLATE_HEIGHT, RED)] == 1
    assert stock.free(PLATE_HEIGHT, 2, 1, RED) == -8, "the original ledger is untouched"


def test_fork_changes_do_not_reach_the_original():
    stock = load_stock([["Plate", 2, 1, 1, "Red", 2, ""]])
    trial = stock.fork()
    trial.take(PLATE_HEIGHT, 2, 1, RED, 2)
    assert stock.free(PLATE_HEIGHT, 2, 1, RED) == 2 and trial.free(PLATE_HEIGHT, 2, 1, RED) == 0


def test_owned_used_and_need_total_split_the_used_pieces():
    stock = load_stock([["Plate", 2, 1, 1, "Red", 2, ""]])
    stock.take(PLATE_HEIGHT, 2, 1, RED, 5)
    assert stock.owned_used() == 2 and stock.need_total() == 3


if __name__ == "__main__":
    support.run_tests(globals())
