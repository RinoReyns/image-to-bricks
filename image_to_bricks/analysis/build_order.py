"""Build order of the parts that hang in the air while the model is built bottom-up, as SUB-ASSEMBLIES.

Simulated build: layers are placed from the bottom; a piece STANDS if it is on the ground layer or rests on a piece
that stands. A piece that does not stand hangs; hanging pieces that rest on each other form a HANGING GROUP. A group
is JOINED on the layer where standing pieces (its JOINING pieces) are first placed on it.

Every hanging group, whatever its size, becomes a sub-assembly together with its joining pieces (groups that share a
joining piece become one): it is built on its own pages from its lowest layer up, its joining pieces last, and then
placed as one unit on its join layer, where the joining pieces press onto the model and carry the hanging part.
Inside a sub-assembly the same rule applies again (ground = its lowest layer): parts that hang inside it are smaller
sub-assemblies (S4.1, S4.2, ...) built first and placed during its step of their join layer.

Pieces: model.PlacedBrick (rectangles and L pieces); layers are model z indices (plates)."""

from collections.abc import Sequence
from dataclasses import dataclass

from image_to_bricks.model import PlacedBrick, piece_cells

TOP_LEVEL_NAME_PREFIX = "S"
COMPACT_MAX_STEPS = 5  # sub-assemblies with at most this many steps and pieces, and no smaller sub-assembly
COMPACT_MAX_PIECES = 8  # inside, share one page per join layer; bigger ones get one page per step


@dataclass(frozen=True)
class HangingGroup:
    """Pieces that hang together while building bottom-up (build order: z, y, x), joined to the standing build on
    `join_layer` (z index) by `joining_pieces` (the standing pieces placed on the group on that layer)."""

    pieces: tuple[PlacedBrick, ...]
    join_layer: int
    joining_pieces: tuple[PlacedBrick, ...]


@dataclass(frozen=True)
class Build:
    """How a set of pieces is built bottom-up from `ground_layer`: the main model (ground 0) or a sub-assembly
    (ground = its lowest layer).
    subassemblies: its direct sub-assemblies in page order; subassembly_of: {id(piece): the direct sub-assembly the
    piece belongs to}; floating: {id(piece): clamping layer or -1} for pieces with nothing under them that are in no
    sub-assembly (hold them in place; only a hanging group that never joins, which tie repair prevents)."""

    pieces: tuple[PlacedBrick, ...]
    ground_layer: int
    subassemblies: tuple["SubAssembly", ...]
    subassembly_of: dict[int, "SubAssembly"]
    floating: dict[int, int]

    @property
    def lowest_layer(self) -> int:
        """Lowest plate layer (z) any of its pieces starts on."""
        return min(piece.z for piece in self.pieces)

    @property
    def top_layer(self) -> int:
        """Highest plate layer any of its pieces covers."""
        return max(piece.z + piece.h_plates - 1 for piece in self.pieces)

    @property
    def step_layers(self) -> list[int]:
        """Layers (z) with a step: a piece placed directly or a sub-assembly placed on it, lowest first."""
        return sorted(
            {piece.z for piece in self.pieces_placed_directly()}
            | {subassembly.join_layer for subassembly in self.subassemblies}
        )

    def pieces_placed_directly(self, layer: int | None = None) -> list[PlacedBrick]:
        """Pieces placed one by one in this build (not inside a sub-assembly), all or those starting on `layer`."""
        return [
            piece
            for piece in self.pieces
            if id(piece) not in self.subassembly_of and (layer is None or piece.z == layer)
        ]

    def is_waiting_for_attachment(self, piece: PlacedBrick, layer: int) -> bool:
        """True if the piece is in a sub-assembly that is not on the model yet while layer `layer` is built (it is
        not drawn on that page at all)."""
        subassembly = self.subassembly_of.get(id(piece))
        return subassembly is not None and layer < subassembly.join_layer

    def subassembly_placed_on(self, piece: PlacedBrick, layer: int) -> "SubAssembly | None":
        """The sub-assembly of the piece if it is placed as a unit on layer `layer` (its pieces on that layer are
        drawn faded with its name), else None."""
        subassembly = self.subassembly_of.get(id(piece))
        return subassembly if subassembly is not None and subassembly.join_layer == layer else None

    def is_in_place_before_layer(self, piece: PlacedBrick, layer: int) -> bool:
        """True if the piece is on the model when the pieces of layer `layer` are placed (sub-assemblies of that
        layer are placed first)."""
        return not self.is_waiting_for_attachment(piece, layer)

    def all_subassemblies(self) -> list["SubAssembly"]:
        """Its sub-assemblies and theirs, depth first in page order."""
        found: list[SubAssembly] = []
        for subassembly in self.subassemblies:
            found.append(subassembly)
            found += subassembly.build.all_subassemblies()
        return found


@dataclass(frozen=True)
class SubAssembly:
    """A hanging part with the pieces that clamp it (joining_pieces, on join_layer of the parent build), built on its
    own pages (`build`) and placed as one unit on join_layer. colour_index: its index among the sub-assemblies placed
    on the same layer (its outline colour on the maps). is_blocked_from_above: something built before it lies over
    one of its pieces, so it cannot be lowered straight into place (slide it in from the side)."""

    name: str
    join_layer: int
    joining_pieces: tuple[PlacedBrick, ...]
    build: Build
    colour_index: int
    is_blocked_from_above: bool

    @property
    def pieces(self) -> tuple[PlacedBrick, ...]:
        """Its hanging pieces and joining pieces, in build order."""
        return self.build.pieces

    @property
    def step_layers(self) -> list[int]:
        """Layers (z) with a step of its own build, lowest first."""
        return self.build.step_layers

    @property
    def is_compact(self) -> bool:
        """Small enough to share one page with the other small sub-assemblies of its join layer."""
        return (
            len(self.step_layers) <= COMPACT_MAX_STEPS
            and len(self.pieces) <= COMPACT_MAX_PIECES
            and not self.build.subassemblies
        )


type CellOccupants = dict[tuple[int, int, int], PlacedBrick]  # {(z, y, x): piece on that plate cell}


def _occupants_by_cell(pieces: Sequence[PlacedBrick]) -> CellOccupants:
    """{(z, y, x): piece} for every plate cell of every piece."""
    occupant: CellOccupants = {}
    for piece in pieces:
        for stud_x, stud_y in piece_cells(piece):
            for z in range(piece.z, piece.z + piece.h_plates):
                occupant[(z, stud_y, stud_x)] = piece
    return occupant


def _pieces_touching(piece: PlacedBrick, layer: int, occupant: CellOccupants) -> list[PlacedBrick]:
    """Distinct pieces in `layer` over / under the studs of `piece`, in stud order."""
    found: list[PlacedBrick] = []
    for stud_x, stud_y in piece_cells(piece):
        other = occupant.get((layer, stud_y, stud_x))
        if other is not None and other is not piece and all(other is not seen for seen in found):
            found.append(other)
    return found


def find_unsupported_pieces(pieces: Sequence[PlacedBrick], ground_layer: int = 0) -> dict[int, int]:
    """Pieces with nothing directly under them (no piece of `pieces` in the layer below touches their studs); pieces
    starting on `ground_layer` rest on the ground. Returns {id(piece): the lowest layer above the piece with any
    piece over its studs (it clamps the piece), or -1 if nothing is above}."""
    occupant = _occupants_by_cell(pieces)
    top_layer = max((piece.z + piece.h_plates for piece in pieces), default=0)
    unsupported: dict[int, int] = {}
    for piece in pieces:
        if piece.z <= ground_layer or _pieces_touching(piece, piece.z - 1, occupant):
            continue
        unsupported[id(piece)] = next(
            (
                layer
                for layer in range(piece.z + piece.h_plates, top_layer)
                if any((layer, stud_y, stud_x) in occupant for stud_x, stud_y in piece_cells(piece))
            ),
            -1,
        )
    return unsupported


class _HangingGroupTracker:
    """Open (not yet joined) hanging groups while the build is simulated layer by layer."""

    def __init__(self) -> None:
        self.group_key_of: dict[int, int] = {}  # id(piece) -> key of its open group
        self.pieces_by_group_key: dict[int, list[PlacedBrick]] = {}
        self.next_group_key = 0

    def is_hanging(self, piece: PlacedBrick) -> bool:
        """True if the piece is in an open hanging group."""
        return id(piece) in self.group_key_of

    def add_hanging_piece(self, piece: PlacedBrick, supports: Sequence[PlacedBrick]) -> None:
        """Put a hanging piece into the group(s) of the hanging pieces under it (merging them), or a new group."""
        group_keys = sorted({self.group_key_of[id(support)] for support in supports})
        if not group_keys:
            group_keys = [self.next_group_key]
            self.pieces_by_group_key[self.next_group_key] = []
            self.next_group_key += 1
        target_key = group_keys[0]
        for other_key in group_keys[1:]:
            for member in self.pieces_by_group_key.pop(other_key):
                self.group_key_of[id(member)] = target_key
                self.pieces_by_group_key[target_key].append(member)
        self.group_key_of[id(piece)] = target_key
        self.pieces_by_group_key[target_key].append(piece)

    def group_keys_under(self, supports: Sequence[PlacedBrick]) -> set[int]:
        """Keys of the open groups that the supports belong to."""
        return {self.group_key_of[id(support)] for support in supports if self.is_hanging(support)}

    def close_group(self, group_key: int) -> list[PlacedBrick]:
        """Remove a joined group from the open groups and return its pieces."""
        members = self.pieces_by_group_key.pop(group_key)
        for member in members:
            del self.group_key_of[id(member)]
        return members


def _build_order_key(piece: PlacedBrick) -> tuple[int, int, int]:
    """Sort key of the build order: layer, then y, then x."""
    return piece.z, piece.y, piece.x


def find_hanging_groups(pieces: Sequence[PlacedBrick], ground_layer: int = 0) -> list[HangingGroup]:
    """Hanging groups of the pieces built bottom-up from `ground_layer` (see the module docstring), in join order.
    On every layer the pieces resting on something that stands (as built up to the layer below) stand; the others
    join the groups under them. Groups that never join the standing build are not returned (they would be loose)."""
    occupant = _occupants_by_cell(pieces)
    pieces_by_layer: dict[int, list[PlacedBrick]] = {}
    for piece in sorted(pieces, key=_build_order_key):
        pieces_by_layer.setdefault(piece.z, []).append(piece)
    tracker = _HangingGroupTracker()
    groups: list[HangingGroup] = []
    for layer in sorted(pieces_by_layer):
        layer_pieces = pieces_by_layer[layer]
        supports_by_piece = {id(piece): _pieces_touching(piece, layer - 1, occupant) for piece in layer_pieces}
        standing_pieces = [
            piece
            for piece in layer_pieces
            if layer <= ground_layer or any(not tracker.is_hanging(support) for support in supports_by_piece[id(piece)])
        ]
        standing_ids = {id(piece) for piece in standing_pieces}
        for piece in layer_pieces:
            if id(piece) not in standing_ids:
                tracker.add_hanging_piece(piece, supports_by_piece[id(piece)])
        joined_group_keys: set[int] = set()
        for piece in standing_pieces:
            joined_group_keys |= tracker.group_keys_under(supports_by_piece[id(piece)])
        for group_key in sorted(joined_group_keys):
            members = tracker.close_group(group_key)
            groups.append(
                HangingGroup(
                    tuple(sorted(members, key=_build_order_key)),
                    layer,
                    tuple(_joining_pieces_of(members, standing_pieces, supports_by_piece)),
                )
            )
    return groups


def _joining_pieces_of(
    members: Sequence[PlacedBrick],
    standing_pieces: Sequence[PlacedBrick],
    supports_by_piece: dict[int, list[PlacedBrick]],
) -> list[PlacedBrick]:
    """Standing pieces of the join layer that rest on a member of the group; supports_by_piece = {id(piece):
    the pieces under it}."""
    member_ids = {id(member) for member in members}
    return [
        piece for piece in standing_pieces if any(id(support) in member_ids for support in supports_by_piece[id(piece)])
    ]


def _merge_groups_sharing_joining_pieces(groups: Sequence[HangingGroup]) -> list[list[HangingGroup]]:
    """Groups that share a joining piece belong to one unit (the piece clamps both)."""
    unit_index_of_joining_piece: dict[int, int] = {}
    units: list[list[HangingGroup]] = []
    for group in groups:
        unit_indexes = sorted(
            {
                unit_index_of_joining_piece[id(piece)]
                for piece in group.joining_pieces
                if id(piece) in unit_index_of_joining_piece
            }
        )
        if unit_indexes:
            target = unit_indexes[0]
            for other in unit_indexes[1:]:
                units[target] += units[other]
                units[other] = []
        else:
            target = len(units)
            units.append([])
        units[target].append(group)
        for unit_index, unit in enumerate(units):
            for member_group in unit:
                for piece in member_group.joining_pieces:
                    unit_index_of_joining_piece[id(piece)] = unit_index
    return [unit for unit in units if unit]


def _unit_pieces(unit: Sequence[HangingGroup]) -> list[PlacedBrick]:
    """Hanging pieces and joining pieces of a unit, each once, in build order."""
    pieces_by_id: dict[int, PlacedBrick] = {}
    for group in unit:
        for piece in group.pieces + group.joining_pieces:
            pieces_by_id[id(piece)] = piece
    return sorted(pieces_by_id.values(), key=_build_order_key)


def is_blocked_from_above(unit_pieces: Sequence[PlacedBrick], built_pieces: Sequence[PlacedBrick]) -> bool:
    """True if a built piece lies over a stud of a unit piece (higher up), so the unit cannot be lowered straight
    into place."""
    unit_ids = {id(piece) for piece in unit_pieces}
    lowest_unit_layer_by_stud: dict[tuple[int, int], int] = {}
    for piece in unit_pieces:
        for stud in piece_cells(piece):
            lowest_unit_layer_by_stud[stud] = min(piece.z, lowest_unit_layer_by_stud.get(stud, piece.z))
    return any(
        id(piece) not in unit_ids and piece.z > lowest_unit_layer_by_stud.get(stud, piece.z)
        for piece in built_pieces
        for stud in piece_cells(piece)
    )


def _colour_indexes(join_layers: Sequence[int]) -> list[int]:
    """Index of each sub-assembly among those with the same join layer (in order)."""
    seen_count_by_layer: dict[int, int] = {}
    indexes: list[int] = []
    for layer in join_layers:
        indexes.append(seen_count_by_layer.get(layer, 0))
        seen_count_by_layer[layer] = indexes[-1] + 1
    return indexes


def plan_build(pieces: Sequence[PlacedBrick], ground_layer: int = 0, name_prefix: str = TOP_LEVEL_NAME_PREFIX) -> Build:
    """Build plan of the pieces built bottom-up from ground_layer (see the module docstring). Sub-assemblies are
    named name_prefix + 1, 2, ... in page order (join layer, then lowest piece); theirs get "<name>.1", ..."""
    units = sorted(
        _merge_groups_sharing_joining_pieces(find_hanging_groups(pieces, ground_layer)),
        key=lambda unit: (unit[0].join_layer, _build_order_key(_unit_pieces(unit)[0])),
    )
    unit_pieces_list = [_unit_pieces(unit) for unit in units]
    join_layers = [unit[0].join_layer for unit in units]
    later_unit_piece_ids: list[set[int]] = [
        {
            id(piece)
            for other_index, other_pieces in enumerate(unit_pieces_list)
            if join_layers[other_index] >= join_layers[unit_index] and other_index != unit_index
            for piece in other_pieces
        }
        for unit_index in range(len(units))
    ]
    colour_indexes = _colour_indexes(join_layers)
    subassemblies = []
    for unit_index, (unit, unit_pieces) in enumerate(zip(units, unit_pieces_list, strict=True)):
        name = f"{name_prefix}{unit_index + 1}"
        unit_ids = {id(piece) for piece in unit_pieces}
        built_before = [
            piece
            for piece in pieces
            if piece.z < join_layers[unit_index]
            and id(piece) not in unit_ids
            and id(piece) not in later_unit_piece_ids[unit_index]
        ]
        joining = {id(piece): piece for group in unit for piece in group.joining_pieces}
        subassemblies.append(
            SubAssembly(
                name,
                join_layers[unit_index],
                tuple(joining.values()),
                plan_build(unit_pieces, min(piece.z for piece in unit_pieces), name + "."),
                colour_indexes[unit_index],
                is_blocked_from_above(unit_pieces, built_before),
            )
        )
    subassembly_of = {id(piece): subassembly for subassembly in subassemblies for piece in subassembly.pieces}
    floating = {
        piece_id: clamp_layer
        for piece_id, clamp_layer in find_unsupported_pieces(pieces, ground_layer).items()
        if piece_id not in subassembly_of
    }
    return Build(
        tuple(sorted(pieces, key=_build_order_key)), ground_layer, tuple(subassemblies), subassembly_of, floating
    )


def empty_build(pieces: Sequence[PlacedBrick]) -> Build:
    """Plan with no sub-assemblies (every piece placed directly on its layer)."""
    return Build(tuple(sorted(pieces, key=_build_order_key)), 0, (), {}, {})
