"""The cases that used to be "add later from below" / "hold it" (step 73) are now small sub-assemblies
(build_order.plan_build): a hanging piece is built together with the piece(s) that clamp it on top and placed as one
unit on the clamping layer. Only a group that nothing ever clamps stays "hold it" (red "!")."""

import sys

import support

support.stub_mesh_libraries()
import numpy as np
from reportlab.lib.styles import getSampleStyleSheet

from image_to_bricks.analysis import build_order
from image_to_bricks.instructions import fast_render as fr
from image_to_bricks.instructions import outputs_handler as oh
from image_to_bricks.model import LegoModel, PlacedBrick

BLACK = 6
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def model_of(pieces, shape=(4, 3, 6)):
    grid = np.zeros(shape, bool)
    for piece in pieces:
        grid[piece.z : piece.z + piece.h_plates, piece.y : piece.y + piece.d, piece.x : piece.x + piece.w] = True
    return LegoModel(grid=grid, bricks=list(pieces))


def heel_nub():
    """Layer 0: 3x1. Layer 1: 3x1 on it + a 1x1 at (1, 1) over air. Layer 2: a 1x2 over the 3x1 and the 1x1."""
    nub = PlacedBrick(1, 1, 1, 1, 1, 1, BLACK)
    holder = PlacedBrick(1, 0, 2, 1, 2, 1, BLACK)
    pieces = [PlacedBrick(0, 0, 0, 3, 1, 1, BLACK), PlacedBrick(0, 0, 1, 3, 1, 1, BLACK), nub, holder]
    return model_of(pieces), nub, holder


def ids_of(pieces):
    return {id(piece) for piece in pieces}


def test_nub_and_its_holder_are_one_small_subassembly():
    model, nub, holder = heel_nub()
    plan = build_order.plan_build(model.bricks)
    subassembly = plan.subassemblies[0] if plan.subassemblies else None
    check(
        "one sub-assembly = the nub + its holder",
        len(plan.subassemblies) == 1 and ids_of(subassembly.pieces) == ids_of([nub, holder]),
        plan.subassemblies,
    )
    check(
        "placed on layer 2 (the holder's layer), joined by the holder",
        subassembly.join_layer == 2 and subassembly.joining_pieces == (holder,),
    )
    check("small: shares a page", subassembly.is_compact)
    check("nothing left to hold", not plan.floating)


def test_holder_resting_only_on_the_nub_is_never_clamped_so_held():
    nub = PlacedBrick(3, 0, 1, 1, 1, 1, BLACK)
    holder = PlacedBrick(3, 0, 2, 2, 1, 1, BLACK)  # over the nub and over air
    model = model_of([PlacedBrick(0, 0, 0, 3, 1, 1, BLACK), PlacedBrick(0, 0, 1, 3, 1, 1, BLACK), nub, holder])
    plan = build_order.plan_build(model.bricks)
    check(
        "nothing clamps them: no sub-assembly, the nub stays 'hold it'",
        not plan.subassemblies and id(nub) in plan.floating,
    )


def test_two_single_steps_are_two_subassemblies_or_one_when_they_share_the_clamp():
    step1 = PlacedBrick(3, 0, 1, 1, 1, 1, BLACK)  # each step one stud further out, over air
    step2 = PlacedBrick(4, 0, 2, 1, 1, 1, BLACK)
    top = PlacedBrick(2, 0, 3, 3, 1, 1, BLACK)  # rests on the column, holds step 2 only
    column = [PlacedBrick(0, 0, z, 3, 1, 1, BLACK) for z in range(3)]
    plan = build_order.plan_build(model_of(column + [step1, step2, top]).bricks)
    check(
        "step 2 + the top piece are one sub-assembly; step 1 is clamped by nothing: held",
        len(plan.subassemblies) == 1
        and ids_of(plan.subassemblies[0].pieces) == ids_of([step2, top])
        and id(step1) in plan.floating,
        plan.floating,
    )
    nub_a, nub_b = PlacedBrick(3, 0, 1, 1, 1, 1, BLACK), PlacedBrick(3, 1, 1, 1, 1, 1, BLACK)
    shared_clamp = PlacedBrick(2, 0, 2, 2, 2, 1, BLACK)  # one 2x2 on the column and on both nubs
    column = [PlacedBrick(0, 0, z, 3, 2, 1, BLACK) for z in range(2)]
    shared = build_order.plan_build(model_of(column + [nub_a, nub_b, shared_clamp]).bricks)
    check(
        "two nubs under one clamp: one sub-assembly of 3 pieces",
        len(shared.subassemblies) == 1 and len(shared.subassemblies[0].pieces) == 3,
    )


def paragraph_texts(notes):
    return " ".join(getattr(note, "text", "") for note in notes)


def test_pdf_notes_no_more_add_later():
    model, nub, holder = heel_nub()
    plan = build_order.plan_build(model.bricks)
    context = oh.LayerPageContext(plan, {}, 0)
    styles = getSampleStyleSheet()
    own_page = paragraph_texts(oh._layer_notes(model, 1, plan.pieces_placed_directly(1), context, styles))
    holder_page = paragraph_texts(oh._layer_notes(model, 2, plan.pieces_placed_directly(2), context, styles))
    check("layer of the nub: no note (the nub is not there yet)", own_page == "", own_page)
    check(
        "holder layer: place sub-assembly S1 first",
        "First place sub-assembly" in holder_page and "S1" in holder_page,
        holder_page,
    )
    check(
        "layer pages place neither piece directly",
        nub not in plan.pieces_placed_directly(1) and holder not in plan.pieces_placed_directly(2),
    )


def test_map_has_no_add_later_mark_or_legend_entry():
    check("no add-later mark any more", not hasattr(fr, "ADD_LATER_MARK"))
    check(
        "default legend: place now / continues / layer below",
        [text for _, text in fr.legend_entries()] == ["place now", "brick from below continues", "layer below"],
    )


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
