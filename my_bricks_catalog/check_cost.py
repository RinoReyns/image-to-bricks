#!/usr/bin/env python3
"""What the pieces I am missing for a build cost to buy, in PLN.

    python check_cost.py <build folder or *_inventory.csv>
    python check_cost.py lego_output_step32_accent
    python check_cost.py lego_output_step32_accent --profile compatible
    python check_cost.py lego_output_step32_accent --factor 0.5      # e.g. other brand = half the price

What it does
  need = pieces the build uses (the build's *_inventory.csv)  -  pieces I own (catalog_stock.csv)
  Only the shortfall is priced. Owned pieces are matched per type + size + colour name.
  Hand-added and photographed pieces both count (catalog_stock.csv has both).
  Irregular ("special") pieces in the stock are ignored, EXCEPT the L-shaped corner pieces
  (corner plate / corner brick 2x2, 3 studs): the build can use them (--use-l-pieces), so they are
  counted and priced like the rest, under the names "Corner plate (L)" / "Corner brick (L)".

Prices come from price_list.csv (PLN per piece):
  piece_type,width_studs,depth_studs,color_name,price_lego_shop,price_compatible,note
  * color_name empty = same price for every colour; fill it to give one colour its own price.
  * --profile lego_shop (default) uses price_lego_shop, --profile compatible uses price_compatible.
    If the chosen cell is empty the script falls back to price_lego_shop and tells you.
  * --factor multiplies every price (shipping, discount, other brand ...).
  * A size with no price row is priced with a rough fallback (--fallback-per-stud, PLN per stud,
    default 0.15 for plates and 0.25 for bricks) and is listed as UNPRICED GUESS. Add a row to
    price_list.csv to make it exact.
Output: a summary on screen and <build>_cost.csv next to the inventory (one line per missing piece).
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

HERE = os.path.dirname(os.path.abspath(__file__))
PLATE_NAMES = ("Plate", "Corner plate (L)")  # priced per stud with the PLATE fallback
UNPRICED = "UNPRICED GUESS"
BIGGEST_TYPES_SHOWN = 8

type SizeKey = tuple[str, int, int]  # (piece name, long side, short side) in studs
type PieceKey = tuple[str, int, int, int, str]  # SizeKey + (height in plates, colour name)
type PriceKey = tuple[str, int, int, str]  # SizeKey + (colour name, "" = every colour)
type Row = Mapping[str, str | None]


def l_type(row: Row) -> str | None:
    """Name "Corner plate (L)" / "Corner brick (L)" for an L-shaped corner piece row, else None.
    Inventory and stock rows mark it with "(L, 3 studs)" or "corner 2x2 (L" in the shape column."""
    shape = (row.get("shape") or "").lower()
    if "(l" not in shape or "corner" not in shape:
        return None
    return "Corner plate (L)" if (row.get("piece_type") or "").startswith("Plate") else "Corner brick (L)"


def piece_name(row: Row) -> str | None:
    """Name used for pricing: Plate / Brick / the L names; None for every other piece."""
    name = (row.get("piece_type") or "").strip()
    if name in ("Plate", "Brick"):
        return name
    return l_type(row)


def size_key(piece_type: str, width: str | int, depth: str | int) -> SizeKey:
    """(name, long side, short side): the same piece in either orientation gets one key."""
    width, depth = int(width), int(depth)
    return (piece_type.strip(), max(width, depth), min(width, depth))


def _piece_key(name: str, row: Row) -> PieceKey:
    """Key of an inventory / stock row whose piece name is known."""
    return (
        *size_key(name, row["width_studs"] or "", row["depth_studs"] or ""),
        int(row["height_plates"] or ""),
        (row["color_name"] or "").strip(),
    )


def find_inventory(path: str) -> str:
    """The *_inventory.csv in a build folder (first by name), or path itself if it is a file."""
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "*_inventory.csv")))
        if not found:
            sys.exit(f"No *_inventory.csv in {path}")
        return found[0]
    return path


def read_inventory(path: str) -> defaultdict[PieceKey, int]:
    """Pieces the build uses, per key (special pieces and the TOTAL line are skipped)."""
    used: defaultdict[PieceKey, int] = defaultdict(int)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = piece_name(row)
            if name is None:
                continue  # skips the TOTAL PIECES line and other special pieces
            used[_piece_key(name, row)] += int(row["quantity"])
    return used


def read_stock(path: str) -> defaultdict[PieceKey, int]:
    """Pieces I own, per key; a missing file means 'I own nothing' (with a WARNING)."""
    have: defaultdict[PieceKey, int] = defaultdict(int)
    if not os.path.exists(path):
        print(f"WARNING: stock file not found ({path}): treating it as 'I own nothing'")
        return have
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = piece_name(row)
            quantity = (row.get("quantity") or "").strip()
            if name is None or not quantity.lstrip("-").isdigit():
                continue
            have[_piece_key(name, row)] += int(quantity)
    return have


@dataclass
class PriceTable:
    """Prices per piece (PLN) and their note, per PriceKey; fallback_sizes: sizes whose compatible price was
    empty, so the lego_shop price was used."""

    prices: dict[PriceKey, float] = field(default_factory=dict)
    notes: dict[PriceKey, str] = field(default_factory=dict)
    fallback_sizes: set[SizeKey] = field(default_factory=set)

    def find(self, size: SizeKey, colour: str) -> tuple[float, str] | None:
        """(price, note) of the colour's own row, else of the every-colour row, else None."""
        for key in ((*size, colour), (*size, "")):
            if key in self.prices:
                return self.prices[key], self.notes[key]
        return None


def read_prices(path: str, profile: str) -> PriceTable:
    """price_list.csv for the profile (lego_shop / compatible; an empty compatible cell falls back to
    lego_shop). Decimal commas are accepted."""
    table = PriceTable()
    column = "price_compatible" if profile == "compatible" else "price_lego_shop"
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if not (row.get("piece_type") or "").strip():
                continue
            size = size_key(row["piece_type"] or "", row["width_studs"] or "", row["depth_studs"] or "")
            key = (*size, (row.get("color_name") or "").strip())
            value = (row.get(column) or "").strip()
            note = row.get("note", "") or ""
            if not value and profile == "compatible":
                value = (row.get("price_lego_shop") or "").strip()
                if value:
                    table.fallback_sizes.add(size)
                    note = "no compatible price, using lego_shop price; " + note
            if value:
                table.prices[key] = float(value.replace(",", "."))
                table.notes[key] = note
    return table


class CostLine(NamedTuple):
    """One missing piece kind (a row of the _cost.csv)."""

    piece_type: str
    long_side: int
    short_side: int
    height_plates: int
    colour: str
    used: int
    have: int
    need: int
    unit_pln: float
    cost_pln: float
    price_source: str


COST_HEADER = [
    "piece_type",
    "width_studs",
    "depth_studs",
    "height_plates",
    "color_name",
    "used",
    "have",
    "need",
    "unit_pln",
    "cost_pln",
    "price_source",
]


@dataclass(frozen=True)
class Pricing:
    """How to price: the price table, the multiplier for every price and the PLN per stud for sizes with no
    price row (plate, brick)."""

    table: PriceTable
    factor: float
    fallback_per_stud: tuple[float, float]


@dataclass
class CostReport:
    """Every missing piece priced, with totals by colour and by size."""

    lines: list[CostLine] = field(default_factory=list)
    total_need: int = 0
    total_cost: float = 0.0
    guess_cost: float = 0.0
    guess_pieces: int = 0
    by_colour: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_size: defaultdict[SizeKey, float] = field(default_factory=lambda: defaultdict(float))
    guessed: set[SizeKey] = field(default_factory=set)
    compatible_fallbacks: set[SizeKey] = field(default_factory=set)  # lego_shop price used

    def add(self, key: PieceKey, counts: tuple[int, int, int], pricing: Pricing) -> None:
        """Price `need` pieces of `key`; counts = (used, have, need)."""
        name, long_side, short_side, _height, colour = key
        used, have, need = counts
        size = (name, long_side, short_side)
        found = pricing.table.find(size, colour)
        if found is not None:
            unit, source = found
        else:
            per_stud = pricing.fallback_per_stud[0 if name in PLATE_NAMES else 1]
            unit, source = per_stud * long_side * short_side, UNPRICED
            self.guessed.add(size)
            self.guess_pieces += need
        unit *= pricing.factor
        cost = unit * need
        self.total_need += need
        self.total_cost += cost
        if source == UNPRICED:
            self.guess_cost += cost
        self.by_colour[colour] += cost
        self.by_size[size] += cost
        line = CostLine(*key, used, have, need, round(unit, 3), round(cost, 2), source)
        self.lines.append(line)


def price_missing(used: Mapping[PieceKey, int], have: Mapping[PieceKey, int], pricing: Pricing) -> CostReport:
    """Price every piece the build uses more often than I own it (sorted by colour, name, area, long side)."""
    report = CostReport()
    for key in sorted(used, key=lambda k: (k[4], k[0], k[1] * k[2], k[1])):
        need = max(0, used[key] - have.get(key, 0))
        if need:
            report.add(key, (used[key], have.get(key, 0), need), pricing)
    return report


def write_cost_csv(path: str, report: CostReport) -> None:
    """One line per missing piece kind plus a TOTAL line."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COST_HEADER)
        writer.writerows(report.lines)
        writer.writerow([])
        writer.writerow(["TOTAL", "", "", "", "", "", "", report.total_need, "", round(report.total_cost, 2), ""])


def print_summary(inventory_path: str, counts: tuple[int, int], report: CostReport, args: argparse.Namespace) -> None:
    """The screen summary; counts = (pieces used, of them owned)."""
    used_total, have_used = counts
    print(f"Build:  {os.path.basename(inventory_path)}")
    print(f"Uses {used_total} pieces; {have_used} of them I already own; {report.total_need} to buy.")
    print(f"Prices: profile={args.profile}, factor={args.factor:.2f} (PLN, shipping NOT included)")
    print(f"\nTOTAL TO BUY: {report.total_cost:.2f} PLN\n")
    print("By colour:")
    for colour, cost in sorted(report.by_colour.items(), key=lambda item: -item[1]):
        print(f"  {colour:<24} {cost:9.2f} PLN")
    print("\nBiggest part types:")
    for (name, long_side, short_side), cost in sorted(report.by_size.items(), key=lambda item: -item[1])[
        :BIGGEST_TYPES_SHOWN
    ]:
        print(f"  {name.ljust(5)} {long_side}x{short_side}{' ' * 12} {cost:9.2f} PLN")
    if report.guessed:
        sizes = ", ".join(f"{name} {long_side}x{short_side}" for name, long_side, short_side in sorted(report.guessed))
        print(
            f"\n{UNPRICED}: {report.guess_pieces} pieces ({report.guess_cost:.2f} PLN of the total) have no row in "
            f"price_list.csv; add rows for: {sizes}"
        )
    estimates = sum(1 for line in report.lines if "ESTIMATE" in str(line.price_source))
    if estimates:
        print(f"Note: {estimates} lines use ESTIMATE prices from price_list.csv (no real price found yet).")
    for name, long_side, short_side in sorted(report.compatible_fallbacks):
        print(f"Note: no compatible price for {name} {long_side}x{short_side}: used the lego_shop price.")


def parse_args() -> argparse.Namespace:
    """Command line (see the module doc)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("build", help="build folder or *_inventory.csv")
    parser.add_argument("--stock", default=os.path.join(HERE, "catalog_stock.csv"))
    parser.add_argument("--prices", default=os.path.join(HERE, "price_list.csv"))
    parser.add_argument("--profile", choices=["lego_shop", "compatible"], default="lego_shop")
    parser.add_argument("--factor", type=float, default=1.0, help="multiply all prices")
    parser.add_argument("--fallback-per-stud", type=float, nargs=2, default=[0.15, 0.25], metavar=("PLATE", "BRICK"))
    return parser.parse_args()


def main() -> None:
    """Price the missing pieces of a build; writes <build>_cost.csv and prints a summary."""
    args = parse_args()
    inventory_path = find_inventory(args.build)
    used, have = read_inventory(inventory_path), read_stock(args.stock)
    table = read_prices(args.prices, args.profile)
    plate_per_stud, brick_per_stud = args.fallback_per_stud
    report = price_missing(used, have, Pricing(table, args.factor, (plate_per_stud, brick_per_stud)))
    out = os.path.splitext(inventory_path)[0].replace("_inventory", "") + "_cost.csv"
    write_cost_csv(out, report)
    report.compatible_fallbacks = table.fallback_sizes
    counts = (sum(used.values()), sum(min(used[k], have.get(k, 0)) for k in used))
    print_summary(inventory_path, counts, report, args)
    print(f"\nDetails: {out}")


if __name__ == "__main__":
    main()
