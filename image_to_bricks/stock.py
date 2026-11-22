"""Stock of owned bricks/plates (--stock) and the have / used / still-need report.

A stock file has the same columns as the build inventory CSV (`<name>_inventory.csv`), so the
inventory of an earlier build can be fed back in:
    piece_type, width_studs, depth_studs, height_plates, color_name, quantity   (+ optional section)
Rows without a size (blank rows, "TOTAL PIECES") are skipped.

Stock is a ledger: `used` counts the pieces placed so far per (height, long side, short side,
colour). Using more than owned is allowed (that is the buy list); the packers just prefer owned
pieces. `fork()` gives a throw-away copy so a packer can try a layout without touching the real
counts.
"""

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

from image_to_bricks.consts import BRICK_HEIGHT_PLATES, COLOR_NAMES, colour_name
from image_to_bricks.model import PlacedBrick

type PieceKey = tuple[int, int, int, int]  # (height in plates, long side, short side, colour id)
type SpecialKey = tuple[int, int, int, int, str]  # (height in plates, width, depth, colour id, shape)
type LKey = tuple[int, int]  # L piece: (height in plates, colour id)
# (piece_type, width, depth, height, colour name, have, used, still need, spare, status)
type ReportRow = tuple[str, int, int, int, str, int, int, int, int, str]

PLATE_HEIGHT_PLATES = 1
L_PIECE_FOOTPRINT_STUDS = (2, 2)  # an L piece is a 2x2 corner piece with one stud missing
SPECIALS_LISTED_IN_LOG = 8  # special pieces named in the console line (the rest as "...")
REPORT_COLUMNS = [
    "piece_type",
    "width_studs",
    "depth_studs",
    "height_plates",
    "color_name",
    "have",
    "used",
    "still_need",
    "spare",
    "status",
]
BUY_LIST_COLUMNS = ["piece_type", "width_studs", "depth_studs", "height_plates", "color_name", "quantity", "status"]

_COLOUR_ID_BY_LOWER_NAME = {name.lower(): colour_id for colour_id, name in COLOR_NAMES.items()}


def make_key(h: int, w: int, d: int, color: int) -> PieceKey:
    """Canonical piece key: long side first (an 8x4 plate is the same piece as 4x8)."""
    w, d = (w, d) if w >= d else (d, w)
    return (int(h), int(w), int(d), int(color))


def brick_key(b: PlacedBrick) -> PieceKey | None:
    """Key of a rectangular piece; None for an L piece (those have their own ledger, see Stock.l_used)."""
    if getattr(b, "notch", -1) >= 0:
        return None
    return make_key(b.h_plates, b.w, b.d, b.color)


def l_key(b: PlacedBrick) -> LKey:
    """Key of an L piece in Stock.l_used: (height in plates, colour id)."""
    return (int(b.h_plates), int(b.color))


def _is_l_shape(shape: str | None) -> bool:
    return "l, 3 studs" in (shape or "").lower()


def _quantity(row: Mapping[str, str | None]) -> int:
    return int(float(row.get("quantity") or 0))


def _stock_status(need: int, have: int, used: int) -> str:
    """'not owned' / 'short' / 'ok' / 'unused' for one report row."""
    if need and not have:
        return "not owned"
    if need:
        return "short"
    if used:
        return "ok"
    return "unused"


def _special_key(row: Mapping[str, str | None]) -> SpecialKey:
    height_text = row.get("height_plates")
    return (
        int(height_text or 0) if (height_text or "").strip() else PLATE_HEIGHT_PLATES,
        int(row["width_studs"] or 0),
        int(row["depth_studs"] or 0),
        _COLOUR_ID_BY_LOWER_NAME[(row.get("color_name") or "").strip().lower()],
        (row.get("shape") or "").strip(),
    )


def _row_height_plates(row: Mapping[str, str | None], path: str, line_number: int) -> int:
    """height_plates of a stock row (a missing height: 3 for a "Brick", else 1); exits on another height."""
    height_text = row.get("height_plates")
    if (height_text or "").strip():
        height = int(height_text or 0)
    else:
        is_brick = (row.get("piece_type") or "").lower() == "brick"
        height = BRICK_HEIGHT_PLATES if is_brick else PLATE_HEIGHT_PLATES
    if height not in (PLATE_HEIGHT_PLATES, BRICK_HEIGHT_PLATES):
        raise ValueError(f"{path} line {line_number}: height_plates must be 1 or 3, got {height}")
    return height


class Stock:
    """Owned pieces (have), the pieces placed so far (used) and the owned special / L pieces."""

    def __init__(
        self,
        have: Mapping[PieceKey, int] | None = None,
        used: Mapping[PieceKey, int] | None = None,
        specials: Mapping[SpecialKey, int] | None = None,
    ) -> None:
        # non-rectangular pieces: {(h, w, d, colour, shape): qty}, never packed
        self.specials: Counter[SpecialKey] = Counter(specials or {})
        self.have: Counter[PieceKey] = Counter(have or {})
        self.used: Counter[PieceKey] = Counter(used or {})
        # owned L pieces (corner plate / brick 2x2, 3 studs): {(height, colour): qty}; placed only by --use-l-pieces
        self.l_have: Counter[LKey] = Counter()
        for (height, width, depth, colour, shape), count in self.specials.items():
            if _is_l_shape(shape) and (width, depth) == L_PIECE_FOOTPRINT_STUDS:
                self.l_have[(height, colour)] += count
        self.l_used: Counter[LKey] = Counter()
        self._by_color: defaultdict[int, list[PieceKey]] = defaultdict(list)
        self._by_size: defaultdict[tuple[int, int, int], list[PieceKey]] = defaultdict(list)
        for key in self.have:
            self._by_color[key[3]].append(key)
            self._by_size[key[:3]].append(key)
        self._sizes_cache: dict[tuple[int, tuple[tuple[int, int], ...]], list[tuple[int, int]]] = {}

    # --- loading --------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: str) -> "Stock":
        """Read a stock CSV (columns of the inventory CSV); prints the special pieces kept aside and the colours
        that are not in the palette."""
        have: Counter[PieceKey] = Counter()
        skipped: Counter[str] = Counter()
        specials: Counter[SpecialKey] = Counter()
        with open(path, newline="", encoding="utf-8-sig") as stock_file:
            for line_number, row in enumerate(csv.DictReader(stock_file), start=2):
                if not (row.get("width_studs") or "").strip():
                    continue  # blank line / TOTAL PIECES
                colour_name = (row.get("color_name") or "").strip()
                is_known_colour = colour_name.lower() in _COLOUR_ID_BY_LOWER_NAME
                if "special" in (row.get("piece_type") or "").lower():
                    # wedges, slopes, round / Technic plates: not rectangular, the packer must not build with them
                    if is_known_colour:
                        specials[_special_key(row)] += _quantity(row)
                    else:
                        skipped[f"{colour_name} (special)"] += _quantity(row)
                    continue
                if not is_known_colour:
                    # not a colour the model can use: ignore the row, but say so (typos show up here too)
                    skipped[colour_name] += _quantity(row)
                    continue
                height = _row_height_plates(row, path, line_number)
                quantity = _quantity(row)
                if quantity > 0:
                    colour = _COLOUR_ID_BY_LOWER_NAME[colour_name.lower()]
                    have[make_key(height, int(row["width_studs"]), int(row["depth_studs"]), colour)] += quantity
        _print_loading_notes(specials, skipped)
        return cls(have, specials=specials)

    # --- queries --------------------------------------------------------------
    def free(self, h: int, w: int, d: int, color: int) -> int:
        """Owned minus used pieces of this kind (negative: that many must be bought)."""
        key = make_key(h, w, d, color)
        return self.have.get(key, 0) - self.used.get(key, 0)

    def avail(self, h: int, w: int, d: int, color: int) -> bool:
        """True if an owned piece of this kind is still unused."""
        return self.free(h, w, d, color) > 0

    def colour_has_stock(self, color: int) -> bool:
        """True if at least one owned piece of this colour is still unused."""
        return any(self.have[key] - self.used.get(key, 0) > 0 for key in self._by_color.get(color, ()))

    def owns_colour(self, color: int) -> bool:
        """True if the stock has any piece of this colour (used or not)."""
        return color in self._by_color

    def any_owned_colour(self, h: int, w: int, d: int) -> int | None:
        """Colour of an unused owned piece of this size and height (the colour with most left), or None.
        For pieces that cover only hidden cells, where any colour will do."""
        best_colour, best_free = None, 0
        for key in self._by_size.get(make_key(h, w, d, 0)[:3], ()):
            free_count = self.have[key] - self.used.get(key, 0)
            if free_count > best_free:
                best_colour, best_free = key[3], free_count
        return best_colour

    def sizes(self) -> set[tuple[int, int]]:
        """Every (w, d) footprint in the stock, both orientations."""
        footprints = set()
        for _, w, d, _ in self.have:
            footprints.add((w, d))
            footprints.add((d, w))
        return footprints

    def sizes_with(self, standard: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        """`standard` (list of (w, d), largest first) plus the stock sizes it lacks, largest area first
        (ties keep the standard order). Cached: tie repair asks for it ~100k times per run; only the KEYS of
        `have` matter, so the cache is keyed on their number (a new key always changes it)."""
        standard_sizes = tuple(standard)
        cache_key = (len(self.have), standard_sizes)
        if cache_key not in self._sizes_cache:
            extra = sorted(self.sizes() - set(standard_sizes), key=lambda size: (-size[0] * size[1], -size[0]))
            self._sizes_cache[cache_key] = sorted(list(standard_sizes) + extra, key=lambda size: -size[0] * size[1])
        return list(self._sizes_cache[cache_key])

    # --- ledger ---------------------------------------------------------------
    def take(self, h: int, w: int, d: int, color: int, n: int = 1) -> None:
        """Count n pieces of this kind as used."""
        self.used[make_key(h, w, d, color)] += n

    def give(self, h: int, w: int, d: int, color: int, n: int = 1) -> None:
        """Count n pieces of this kind as no longer used."""
        key = make_key(h, w, d, color)
        self.used[key] -= n
        if self.used[key] == 0:
            del self.used[key]

    def take_brick(self, b: PlacedBrick) -> None:
        """Count a placed piece (rectangle or L) as used."""
        if b.notch >= 0:
            self.l_used[l_key(b)] += 1
        else:
            self.take(b.h_plates, b.w, b.d, b.color)

    def give_brick(self, b: PlacedBrick) -> None:
        """Count a removed piece (rectangle or L) as no longer used."""
        if b.notch >= 0:
            key = l_key(b)
            self.l_used[key] -= 1
            if self.l_used[key] == 0:
                del self.l_used[key]
        else:
            self.give(b.h_plates, b.w, b.d, b.color)

    # --- L pieces ---------------------------------------------------------------
    def l_free(self, h: int, color: int) -> int:
        """Owned minus used L pieces of this height and colour."""
        return self.l_have.get((h, color), 0) - self.l_used.get((h, color), 0)

    def l_any_colour(self, h: int) -> int | None:
        """Colour of the owned L piece of this height with most pieces left (None if none is left)."""
        best_colour, best_free = None, 0
        for (height, colour), count in self.l_have.items():
            free_count = count - self.l_used.get((height, colour), 0)
            if height == h and free_count > best_free:
                best_colour, best_free = colour, free_count
        return best_colour

    def fork(self) -> "Stock":
        """Independent copy of the used counts (same owned pieces)."""
        copy = Stock.__new__(Stock)
        copy.have, copy.used, copy._by_color, copy._by_size = (
            self.have,
            Counter(self.used),
            self._by_color,
            self._by_size,
        )
        copy.specials = self.specials
        copy._sizes_cache = self._sizes_cache  # same `have`, same sizes
        copy.l_have, copy.l_used = self.l_have, Counter(self.l_used)
        return copy

    def recount(self, bricks: Iterable[PlacedBrick]) -> "Stock":
        """Fresh ledger for exactly these pieces."""
        pieces = list(bricks)
        fresh = self.fork()
        fresh.used = Counter(key for key in (brick_key(b) for b in pieces) if key is not None)
        fresh.l_used = Counter(l_key(b) for b in pieces if b.notch >= 0)
        return fresh

    def owned_used(self) -> int:
        """How many owned pieces are in use (used, capped at what is owned per type/size/colour)."""
        return sum(min(self.have.get(key, 0), n) for key, n in self.used.items() if n > 0) + sum(
            min(self.l_have.get(key, 0), n) for key, n in self.l_used.items() if n > 0
        )

    def need_total(self) -> int:
        """Pieces to buy: used beyond what is owned, summed over every kind (L pieces included)."""
        return sum(max(0, n - self.have.get(key, 0)) for key, n in self.used.items()) + sum(
            max(0, n - self.l_have.get(key, 0)) for key, n in self.l_used.items()
        )

    # --- report ---------------------------------------------------------------
    def rows(self) -> list[ReportRow]:
        """[(piece_type, w, d, h, colour_name, have, used, need, spare, status)] sorted for reading."""
        rows: list[ReportRow] = []
        for key in set(self.have) | set(self.used):
            height, width, depth, colour = key
            have, used = self.have.get(key, 0), self.used.get(key, 0)
            piece_type = "Brick" if height == BRICK_HEIGHT_PLATES else "Plate"
            rows.append(_report_row(piece_type, (width, depth, height), colour, have, used))
        for l_piece in set(self.l_have) | set(self.l_used):
            height, colour = l_piece
            have, used = self.l_have.get(l_piece, 0), self.l_used.get(l_piece, 0)
            piece_type = "Corner brick (L)" if height == BRICK_HEIGHT_PLATES else "Corner plate (L)"
            rows.append(_report_row(piece_type, (*L_PIECE_FOOTPRINT_STUDS, height), colour, have, used))
        rows.sort(key=lambda row: (row[4], -row[3], -row[1] * row[2], -row[1], row[2]))
        return rows

    def write_report(self, report_path: str, buy_path: str) -> None:
        """Write the have / used / need report and the buy list (pieces still needed) as CSV files."""
        rows = self.rows()
        with open(report_path, "w", newline="") as report_file:
            writer = csv.writer(report_file)
            writer.writerow(REPORT_COLUMNS)
            writer.writerows(rows)
            writer.writerow([])
            totals = [sum(int(row[column]) for row in rows) for column in (5, 6, 7, 8)]
            writer.writerow(["TOTAL PIECES", "", "", "", "", *totals, ""])
        with open(buy_path, "w", newline="") as buy_file:
            writer = csv.writer(buy_file)
            writer.writerow(BUY_LIST_COLUMNS)
            for row in rows:
                still_need = row[7]
                if still_need:
                    writer.writerow([row[0], row[1], row[2], row[3], row[4], still_need, row[9]])
            writer.writerow([])
            writer.writerow(["TOTAL PIECES", "", "", "", "", sum(row[7] for row in rows), ""])

    def summary(self) -> str:
        """Printable have / used / need per colour."""
        per_colour: defaultdict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # have, used, owned_used, need
        counts = [(key[3], self.have.get(key, 0), self.used.get(key, 0)) for key in set(self.have) | set(self.used)]
        counts += [
            (colour, self.l_have.get((height, colour), 0), self.l_used.get((height, colour), 0))
            for height, colour in set(self.l_have) | set(self.l_used)
        ]
        for colour, have, used in counts:
            totals = per_colour[colour]
            totals[0] += have
            totals[1] += used
            totals[2] += min(have, used)
            totals[3] += max(0, used - have)
        lines = ["Stock report (pieces): have / used from stock / still need to buy"]
        for colour in sorted(per_colour, key=lambda colour: -per_colour[colour][1]):
            have, _, owned_used, need = per_colour[colour]
            tag = "  NOT OWNED" if have == 0 else ""
            lines.append(
                f"  {colour_name(colour):20s} have {have:4d}   used from stock {owned_used:4d}   need {need:4d}{tag}"
            )
        total = [sum(totals[index] for totals in per_colour.values()) for index in range(4)]
        lines.append(
            f"  {'TOTAL':20s} have {total[0]:4d}   used from stock {total[2]:4d}   need {total[3]:4d}   "
            f"(build = {total[1]} pieces; {total[0] - total[2]} owned pieces left over)"
        )
        return "\n".join(lines)


def _report_row(piece_type: str, size: tuple[int, int, int], colour: int, have: int, used: int) -> ReportRow:
    """One report row; size = (width, depth, height in plates); need = used beyond have, spare = have beyond used."""
    need, spare = max(0, used - have), max(0, have - used)
    status = _stock_status(need, have, used)
    width, depth, height = size
    return (
        piece_type,
        width,
        depth,
        height,
        colour_name(colour),
        have,
        used,
        need,
        spare,
        status,
    )


def _print_loading_notes(specials: Counter[SpecialKey], skipped: Counter[str]) -> None:
    """Console lines about the special pieces kept aside and the rows of unknown colours."""
    if specials:
        named = ", ".join(
            f"{count}x {key[4] or 'special'}" for key, count in list(specials.items())[:SPECIALS_LISTED_IN_LOG]
        )
        print(
            f"Stock: {sum(specials.values())} special (non-rectangular) pieces kept aside, not packed "
            f"(used only by --accent; the L-shaped ones by --use-l-pieces): "
            + named
            + (" ..." if len(specials) > SPECIALS_LISTED_IN_LOG else "")
        )
    if skipped:
        print(
            "Stock: ignored colours not in the palette (cannot be used by the model): "
            + ", ".join(f"{name} ({count} pcs)" for name, count in skipped.items())
        )
