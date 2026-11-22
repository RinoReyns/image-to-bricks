"""--accent: a placed plate swapped for a special part (chrome buckle, eye)."""

from collections.abc import Callable, Sequence

from image_to_bricks.consts import COLOR_NAMES
from image_to_bricks.model import LegoModel, PlacedBrick, front_stud, plate_origin
from image_to_bricks.options import Accent, Box
from image_to_bricks.packing import accent_merge
from image_to_bricks.stock import Stock, make_key

FRONT_STUD_PLACEHOLDER = "{front_stud}"  # in an accent note: replaced by the stud nearest the front

type CellTest = Callable[[int, int, int], bool]  # (z, y, x) in the model grid -> True / False


def _record_accent_in_stock(stock: Stock, brick: PlacedBrick, accent_colour: int) -> None:
    """Moves the accent plate in the running ledger from its old colour to the accent colour (so the ledger keeps
    matching the final pieces) and counts a matching special piece of the stock file as owned."""
    stock.give(brick.h_plates, brick.w, brick.d, brick.color)
    stock.take(brick.h_plates, brick.w, brick.d, accent_colour)
    size = sorted((brick.w, brick.d))
    owned = sum(
        count
        for special, count in stock.specials.items()
        if special[3] == accent_colour and sorted(special[1:3]) == size
    )
    key = make_key(brick.h_plates, brick.w, brick.d, accent_colour)
    if owned and stock.have.get(key, 0) < owned:
        stock.have[key] = owned


def _box_test(model: LegoModel, box: Box) -> CellTest:
    """is_in_box(z, y, x): the model-grid cell lies in `box` (figure grid: without the base layers and margin)."""
    margin = model.base_margin

    def is_in_box(z: int, y: int, x: int) -> bool:
        figure_z, figure_y, figure_x = z - model.base_layers, y - margin, x - margin
        return bool(
            box[0].start <= figure_z < box[0].stop
            and box[1].start <= figure_y < box[1].stop
            and box[2].start <= figure_x < box[2].stop
        )

    return is_in_box


def _highest_plate_in_box(model: LegoModel, accent: Accent, is_in_box: CellTest) -> PlacedBrick | None:
    """The plate of the accent's size (and only_colour, if given) in the box: highest first, then lowest y, x."""
    wanted_size = sorted((accent.width_studs, accent.depth_studs))
    candidates = [
        brick
        for brick in model.bricks
        if brick.h_plates == 1
        and sorted((brick.w, brick.d)) == wanted_size
        and (accent.only_colour_id is None or brick.color == accent.only_colour_id)
        and is_in_box(brick.z, brick.y, brick.x)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda brick: (-brick.z, brick.y, brick.x, id(brick)))


def _merge_accent_plate(
    model: LegoModel, accent: Accent, is_in_box: CellTest, stock: Stock | None
) -> PlacedBrick | None:
    """No W x D plate of the wanted colour in the accent box: merge smaller plates of that colour that fill a
    W x D footprint exactly (accent_merge.py). Returns the merged plate or None (also when no colour is given)."""
    if accent.only_colour_id is None:
        return None
    width, depth = accent.width_studs, accent.depth_studs
    merge = accent_merge.find_mergeable_footprint(model.bricks, width, depth, accent.only_colour_id, is_in_box)
    if merge is None:
        return None
    plate = accent_merge.apply_merge(model.bricks, merge, stock)
    colour_name = COLOR_NAMES.get(accent.only_colour_id, accent.only_colour_id)
    print(f"Accent: merged {len(merge.replaced)} {colour_name} plates into one {width}x{depth} plate for the accent")
    return plate


def _note_with_front_stud(note: str, plate: PlacedBrick, model: LegoModel) -> str:
    """The note with {front_stud} replaced by the plate's stud nearest the front (base plate coordinates)."""
    if FRONT_STUD_PLACEHOLDER not in note:
        return note
    if not model.front:
        return note.replace(FRONT_STUD_PLACEHOLDER, "stud nearest the front (give --front)")
    plate_x0, plate_y0 = plate_origin(model.grid, model.base_layers)
    (front_x, front_y), lies_across = front_stud(plate, model.front)
    stud_text = f"stud nearest the front (x={front_x - plate_x0}, y={front_y - plate_y0})" + (
        " - or its neighbour, the plate lies across the front" if lies_across else ""
    )
    return note.replace(FRONT_STUD_PLACEHOLDER, stud_text)


def apply_accents(model: LegoModel, accents: Sequence[Accent], stock: Stock | None = None) -> LegoModel:
    """--accent: in each box (figure grid, ends exclusive, the same coordinates as --thicken-box) the highest plate
    of the given size is swapped for the special part: it only changes colour in the model (e.g. Chrome Silver),
    is listed on the cover and on its step page, and is counted as owned if a matching special piece is in the
    stock file. Nothing else moves. Uses model.base_layers, model.base_margin and model.front; changes the model
    (and the stock ledger) in place and returns it."""
    margin = model.base_margin
    plate_x0, plate_y0 = plate_origin(model.grid, model.base_layers)
    for accent in accents:
        width, depth, colour = accent.width_studs, accent.depth_studs, accent.colour_id
        is_in_box = _box_test(model, accent.box)
        plate = _highest_plate_in_box(model, accent, is_in_box) or _merge_accent_plate(model, accent, is_in_box, stock)
        if plate is None:
            print(
                f"Accent {COLOR_NAMES.get(colour, colour)} plate {width}x{depth}: no plate of that size in the box"
                " - nothing changed"
            )
            continue
        old_colour_name = COLOR_NAMES.get(plate.color, plate.color)
        if stock is not None:
            _record_accent_in_stock(stock, plate, colour)
        plate.color = colour
        note = _note_with_front_stud(accent.note, plate, model)
        text = f"{COLOR_NAMES.get(colour, colour)} plate {width}x{depth}" + (f" ({note})" if note else "")
        model.accents.append((plate.z, plate.y, plate.x, text))
        print(
            f"Accent: layer {plate.z - model.base_layers + 1}, stud ({plate.x - margin}, {plate.y - margin}) in the "
            f"figure grid, ({plate.x - plate_x0}, {plate.y - plate_y0}) from the base plate corner: "
            f"{old_colour_name} plate {width}x{depth} -> {text}"
        )
    return model
