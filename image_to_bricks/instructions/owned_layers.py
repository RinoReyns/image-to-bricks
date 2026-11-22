"""Which pieces of the build come from the user's own stock, layer by layer (--stock).

Owned pieces are handed out in build order (lowest layer first, then y, x), so the answer is "how far can I build
with what I have": every piece is either OWNED (an unused owned piece of the same size and colour is left when its
layer is built) or TO BUY. A layer is complete when it needs nothing to buy; the build can go on from the bottom as
long as every layer below is complete too.

Pieces: utils.PlacedBrick (rectangles and L pieces); stock: stock.Stock (have / l_have are read, never changed).
"""

import csv
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from image_to_bricks.model import PlacedBrick
from image_to_bricks.stock import LKey, PieceKey, Stock, brick_key, l_key, make_key

type ToBuyKey = PieceKey | tuple[str, int, int]  # (h, w, d, colour) or ("L", h, colour)


@dataclass
class LayerOwnership:
    """Owned / to-buy counts of one plate layer (model z)."""

    layer: int
    owned_pieces: int = 0
    pieces_to_buy: int = 0
    to_buy_by_piece: Counter[ToBuyKey] = field(default_factory=Counter)

    @property
    def is_complete(self) -> bool:
        """True if the layer needs no piece to buy."""
        return self.pieces_to_buy == 0


def to_buy_key(piece: PlacedBrick) -> ToBuyKey:
    """Key of a piece in LayerOwnership.to_buy_by_piece: (h, w, d, colour), or ("L", h, colour) for an L piece."""
    if piece.notch >= 0:
        return ("L", *l_key(piece))
    return make_key(piece.h_plates, piece.w, piece.d, piece.color)


def allocate_owned_pieces(bricks: Iterable[PlacedBrick], stock: Stock) -> dict[int, bool]:
    """{id(piece): True if it is built from an owned piece} for every piece, owned pieces handed out in build
    order (see the module docstring)."""
    handed_out: Counter[PieceKey | None] = Counter()
    l_handed_out: Counter[LKey] = Counter()
    is_owned: dict[int, bool] = {}
    for piece in sorted(bricks, key=lambda b: (b.z, b.y, b.x)):
        if piece.notch >= 0:
            l_piece = l_key(piece)
            is_owned[id(piece)] = l_handed_out[l_piece] < stock.l_have.get(l_piece, 0)
            l_handed_out[l_piece] += 1
        else:
            key = brick_key(piece)
            is_owned[id(piece)] = key is not None and handed_out[key] < stock.have.get(key, 0)
            handed_out[key] += 1
    return is_owned


def ownership_by_layer(bricks: Iterable[PlacedBrick], is_owned: dict[int, bool]) -> list[LayerOwnership]:
    """One LayerOwnership per plate layer that starts a piece, lowest first."""
    layers: dict[int, LayerOwnership] = {}
    for piece in bricks:
        record = layers.setdefault(piece.z, LayerOwnership(piece.z))
        if is_owned[id(piece)]:
            record.owned_pieces += 1
        else:
            record.pieces_to_buy += 1
            record.to_buy_by_piece[to_buy_key(piece)] += 1
    return [layers[z] for z in sorted(layers)]


def count_complete_layers_from_bottom(layers: list[LayerOwnership]) -> int:
    """How many layers, from the bottom, can be built one after the other with owned pieces only."""
    complete = 0
    for record in layers:
        if not record.is_complete:
            break
        complete += 1
    return complete


def write_ownership_csv(layers: list[LayerOwnership], path: str, layer_name: Callable[[int], str]) -> None:
    """<name>_owned_layers.csv: layer, pieces, owned, to buy, complete, buildable from the bottom.
    layer_name(z) -> the layer's name as the PDF prints it."""
    buildable = count_complete_layers_from_bottom(layers)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "pieces", "owned", "to_buy", "complete", "buildable_from_bottom"])
        for index, record in enumerate(layers):
            writer.writerow(
                [
                    layer_name(record.layer),
                    record.owned_pieces + record.pieces_to_buy,
                    record.owned_pieces,
                    record.pieces_to_buy,
                    "yes" if record.is_complete else "no",
                    "yes" if index < buildable else "no",
                ]
            )
