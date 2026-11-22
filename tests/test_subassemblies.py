"""Parts that hang in the air while building bottom-up become sub-assemblies (build_order.plan_build): the hanging
pieces together with the pieces that clamp them, built on their own page(s) right before the layer they are placed
on, then placed as one unit. Before that layer they are not drawn at all; on it they are drawn faded with their name
in their own colour. Small ones share one page per layer, big ones get one page per step, parts that hang inside a
sub-assembly are sub-assemblies of it. Only the instructions change: the model is never touched."""

import contextlib
import io
import os
import sys
import tempfile

import support

support.stub_mesh_libraries()
import numpy as np
from reportlab.lib.styles import getSampleStyleSheet
from support import load_image

from image_to_bricks.analysis import build_order
from image_to_bricks.instructions import fast_render as fr
from image_to_bricks.instructions import outputs_handler as oh
from image_to_bricks.instructions import subassembly_pages
from image_to_bricks.instructions.instruction_parts import colour_rgb
from image_to_bricks.model import LegoModel, PlacedBrick
from image_to_bricks.options import OutputOptions

BLACK, ORANGE = 6, 9  # palette ids (consts.COLOR_NAMES)
failed = False


def check(name, ok, info=None):
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok or info is None else f"  {info}"))
    failed |= not ok


def model_of(pieces, shape=(9, 2, 9)):
    grid = np.zeros(shape, bool)
    for piece in pieces:
        grid[piece.z : piece.z + piece.h_plates, piece.y : piece.y + piece.d, piece.x : piece.x + piece.w] = True
    return LegoModel(grid=grid, bricks=list(pieces))


def tail_model(with_inner_nub=False):
    """Black column x 0-1 on layers 0-3. Orange tail beside it, over air: layer 1 a 2x1 at x 4-5, layer 2 a 3x1 at
    x 3-5 on it, layer 3 a 1x1 at x 3. Layer 4: a black 3x1 at x 1-3 rests on the column AND the tail's top: it
    clamps the tail. Layer 5: a black 2x1 on the column. with_inner_nub: an orange 1x1 at x 6 on layer 2 over air,
    held by an orange 2x1 at x 5-6 on layer 3 that rests on the tail's 3x1 (it hangs inside the tail)."""
    column = [PlacedBrick(0, 0, z, 2, 1, 1, BLACK) for z in range(4)]
    tail = [
        PlacedBrick(4, 0, 1, 2, 1, 1, ORANGE),
        PlacedBrick(3, 0, 2, 3, 1, 1, ORANGE),
        PlacedBrick(3, 0, 3, 1, 1, 1, ORANGE),
    ]
    if with_inner_nub:
        tail += [PlacedBrick(6, 0, 2, 1, 1, 1, ORANGE), PlacedBrick(5, 0, 3, 2, 1, 1, ORANGE)]
    clamp = PlacedBrick(1, 0, 4, 3, 1, 1, BLACK)
    top = PlacedBrick(0, 0, 5, 2, 1, 1, BLACK)
    return model_of(column + tail + [clamp, top], shape=(6, 2, 9)), tail, clamp


def long_tail_model():
    """Column x 0-1 on layers 0-6; an orange 1x1 stack at x 4 on layers 1-6 over air; layer 7: a black 4x1 at x 1-4
    clamps it (7 steps: a big sub-assembly)."""
    column = [PlacedBrick(0, 0, z, 2, 1, 1, BLACK) for z in range(7)]
    tail = [PlacedBrick(4, 0, z, 1, 1, 1, ORANGE) for z in range(1, 7)]
    clamp = PlacedBrick(1, 0, 7, 4, 1, 1, BLACK)
    return model_of(column + tail + [clamp], shape=(8, 2, 9)), tail, clamp


def ids_of(pieces):
    return {id(piece) for piece in pieces}


def test_tail_is_one_group_joined_on_the_layer_that_clamps_it():
    model, tail, clamp = tail_model()
    groups = build_order.find_hanging_groups(model.bricks)
    check(
        "one hanging group = the 3 tail pieces, joined on layer 4 by the black 3x1",
        len(groups) == 1
        and ids_of(groups[0].pieces) == ids_of(tail)
        and groups[0].join_layer == 4
        and groups[0].joining_pieces == (clamp,),
        groups,
    )


def test_subassembly_is_the_tail_plus_its_clamp():
    model, tail, clamp = tail_model()
    plan = build_order.plan_build(model.bricks)
    subassembly = plan.subassemblies[0]
    check(
        "one sub-assembly S1 = tail + clamp",
        len(plan.subassemblies) == 1
        and subassembly.name == "S1"
        and ids_of(subassembly.pieces) == ids_of(tail + [clamp]),
    )
    check(
        "steps: layers 1-4 (the clamp last), placed on layer 4",
        subassembly.step_layers == [1, 2, 3, 4] and subassembly.join_layer == 4,
    )
    check("4 pieces / 4 steps: small (shared page)", subassembly.is_compact)
    check("nothing to hold, nothing above it", not plan.floating and not subassembly.is_blocked_from_above)
    check(
        "the main build places neither the tail nor the clamp",
        not ids_of(plan.pieces_placed_directly()) & ids_of(tail + [clamp]),
    )


def test_long_part_is_big():
    model, tail, _ = long_tail_model()
    subassembly = build_order.plan_build(model.bricks).subassemblies[0]
    check("7 steps: big (one page per step)", len(subassembly.step_layers) == 7 and not subassembly.is_compact)


def test_numbered_by_join_layer_and_coloured_apart_on_the_same_layer():
    """Row y 0: column on layers 0-4, a 4-piece stack at x 6 on layers 1-4, clamped on layer 5. Row y 1: column on
    layers 0-2, a 2-piece stack on layers 1-2, clamped on layer 3. Row y 2: the same as row 1."""
    pieces = (
        [PlacedBrick(0, 0, z, 4, 1, 1, BLACK) for z in range(5)]
        + [PlacedBrick(6, 0, z, 1, 1, 1, ORANGE) for z in range(1, 5)]
        + [PlacedBrick(3, 0, 5, 4, 1, 1, BLACK)]
    )
    for row in (1, 2):
        pieces += (
            [PlacedBrick(0, row, z, 4, 1, 1, BLACK) for z in range(3)]
            + [PlacedBrick(6, row, z, 1, 1, 1, ORANGE) for z in range(1, 3)]
            + [PlacedBrick(3, row, 3, 4, 1, 1, BLACK)]
        )
    plan = build_order.plan_build(model_of(pieces, shape=(6, 3, 9)).bricks)
    summary = [
        (subassembly.name, subassembly.join_layer, subassembly.colour_index) for subassembly in plan.subassemblies
    ]
    check(
        "S1, S2 on layer 3 (colours 0, 1), S3 on layer 5 (colour 0)",
        summary == [("S1", 3, 0), ("S2", 3, 1), ("S3", 5, 0)],
        summary,
    )


def test_group_that_never_joins_stays_held():
    loose = [PlacedBrick(5, 0, 1, 1, 1, 1, ORANGE), PlacedBrick(5, 0, 2, 1, 1, 1, ORANGE)]
    plan = build_order.plan_build(model_of([PlacedBrick(0, 0, 0, 2, 1, 1, BLACK)] + loose).bricks)
    check(
        "loose group: no sub-assembly, its lowest piece is held",
        not plan.subassemblies and id(loose[0]) in plan.floating,
    )


def test_part_hanging_inside_a_subassembly_is_its_own_subassembly():
    model, tail, _ = tail_model(with_inner_nub=True)
    subassembly = build_order.plan_build(model.bricks).subassemblies[0]
    inner_nub, inner_holder = tail[3], tail[4]
    children = subassembly.build.subassemblies
    check(
        "S1 has S1.1 = the nub + its holder",
        len(children) == 1
        and children[0].name == "S1.1"
        and ids_of(children[0].pieces) == ids_of([inner_nub, inner_holder]),
        children,
    )
    check(
        "S1.1 is placed on S1's layer 3; nothing inside S1 to hold",
        children[0].join_layer == 3 and not subassembly.build.floating,
    )
    check("S1 with its own sub-assembly is big", not subassembly.is_compact)


def test_blocked_from_above():
    unit = [PlacedBrick(4, 0, 2, 1, 1, 1, ORANGE)]
    check(
        "a built piece above it: blocked",
        build_order.is_blocked_from_above(unit, [PlacedBrick(3, 0, 3, 2, 1, 1, BLACK)]),
    )
    check(
        "built pieces only below / beside: not blocked",
        not build_order.is_blocked_from_above(
            unit, [PlacedBrick(0, 0, 0, 8, 1, 1, BLACK), PlacedBrick(0, 0, 3, 2, 1, 1, BLACK)]
        ),
    )


def test_pieces_waiting_placed_and_in_place():
    model, tail, clamp = tail_model()
    plan = build_order.plan_build(model.bricks)
    check(
        "waiting (not drawn) before layer 4 only",
        plan.is_waiting_for_attachment(tail[0], 3) and not plan.is_waiting_for_attachment(tail[0], 4),
    )
    check(
        "placed as S1 on layer 4 (the clamp and nothing else of layer 4)",
        plan.subassembly_placed_on(clamp, 4) is plan.subassemblies[0] and plan.subassembly_placed_on(clamp, 5) is None,
    )
    check(
        "in place before the pieces of layer 4 (S1 goes first)",
        plan.is_in_place_before_layer(tail[2], 4) and not plan.is_in_place_before_layer(tail[2], 3),
    )


def count_pixels(image, rgb):
    return int((np.asarray(image) == np.array(rgb)).all(axis=2).sum())


def build_maps_of(model, plan, folder):
    with contextlib.redirect_stdout(io.StringIO()):
        return oh.generate_build_layers(model, folder, show_layer_below=True, plan=plan)


def test_maps_hide_the_part_before_it_is_placed_and_name_it_on_its_layer():
    model, _, _ = tail_model()
    plan = build_order.plan_build(model.bricks)
    s1_colour = fr.subassembly_outline_colour(0)
    with tempfile.TemporaryDirectory() as folder:
        with_plan = [load_image(path) for path in build_maps_of(model, plan, os.path.join(folder, "plan"))]
        without = [
            load_image(path)
            for path in build_maps_of(model, build_order.empty_build(model.bricks), os.path.join(folder, "none"))
        ]
        orange = colour_rgb(ORANGE)
        check(
            "layers 2-4 (z 1-3): no orange at all (the tail is not drawn, not hatched)",
            all(count_pixels(with_plan[z], orange) == 0 for z in (1, 2, 3))
            and all(count_pixels(without[z], orange) > 0 for z in (1, 2, 3)),
        )
        check(
            "layer 5 (z 4): the clamp has S1's outline colour; no colour before",
            count_pixels(with_plan[4], s1_colour) > 50 and count_pixels(with_plan[3], s1_colour) == 0,
        )
        check(
            "layer 6: the tail and clamp are in place below (hatching), no S1 marks",
            count_pixels(with_plan[5], s1_colour) == 0,
        )
        check("the map with S1 has the extra legend row entry", with_plan[4].height >= without[4].height)


def test_legend_entries_only_when_needed():
    check("plain map: 3 entries", len(fr.legend_entries()) == 3)
    check(
        "with hold marks and sub-assembly pieces: 5 entries",
        len(both := fr.legend_entries(has_hold_marks=True, has_subassembly_pieces=True)) == 5
        and both[-1] == fr.SUBASSEMBLY_LEGEND_ENTRY,
    )


def test_main_3d_views_leave_the_part_out_until_it_is_placed():
    model, _, _ = tail_model()
    plan = build_order.plan_build(model.bricks)
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
        with_plan = oh.generate_cumulative_layer_renders(model, os.path.join(folder, "plan"), plan)
        without = oh.generate_cumulative_layer_renders(model, os.path.join(folder, "none"))
        same = [
            np.array_equal(np.asarray(load_image(a)), np.asarray(load_image(b))) for a, b in zip(with_plan, without)
        ]
    check("layer 2 view differs (tail left out), from the placing layer on identical", not same[1] and all(same[4:]))


def headings_and_texts(story):
    """(Heading2 texts in order, all paragraph texts joined)."""
    headings, texts = [], []
    for flowable in story:
        for item in getattr(flowable, "_content", [flowable]):
            text = getattr(item, "text", None)
            if text is None:
                continue
            texts.append(text)
            if item.style.name == "Heading2":
                headings.append(text)
    return headings, " ".join(texts)


def parts_quantity_on_page(story, heading):
    """Total Qty of the parts table on the layer page with this heading."""
    index = [getattr(flowable, "text", None) for flowable in story].index(heading)
    left_cell = story[index + 1]._cellvalues[0][0]
    return 0 if len(left_cell) < 2 else sum(int(row[4]) for row in left_cell[1]._cellvalues[1:])


def build_pages(model, plan, folder):
    with contextlib.redirect_stdout(io.StringIO()):
        maps = oh.generate_build_layers(model, os.path.join(folder, "layers"), show_layer_below=True, plan=plan)
        images = subassembly_pages.generate_subassembly_images(model, plan, os.path.join(folder, "sub"))
    return oh.build_step_pages(
        model, oh.InstructionImages(maps, [], images), oh.LayerPageContext(plan, {}, 0), getSampleStyleSheet()
    )


def test_pdf_small_part_on_a_shared_page_right_before_its_layer():
    model, tail, clamp = tail_model()
    plan = build_order.plan_build(model.bricks)
    with tempfile.TemporaryDirectory() as folder:
        story = build_pages(model, plan, folder)
        headings, texts = headings_and_texts(story)
    check(
        "page order: S1 page right before Layer 5",
        headings == ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Sub-assembly S1 (for Layer 5)", "Layer 5", "Layer 6"],
        headings,
    )
    check(
        "layer pages place only the column and top pieces",
        parts_quantity_on_page(story, "Layer 2") == 1
        and parts_quantity_on_page(story, "Layer 5") == 0
        and parts_quantity_on_page(story, "Layer 6") == 1,
    )
    check("placing page: first place S1", "First place sub-assembly" in texts and "S1" in texts)
    check("no add-later or hold notes", "Skip" not in texts and "nothing under them" not in texts, texts)


def test_pdf_big_part_one_page_per_step():
    model, _, _ = long_tail_model()
    plan = build_order.plan_build(model.bricks)
    with tempfile.TemporaryDirectory() as folder:
        headings, texts = headings_and_texts(build_pages(model, plan, folder))
    step_headings = [heading for heading in headings if heading.startswith("Sub-assembly S1 - step")]
    check(
        "7 step pages, then Layer 8",
        len(step_headings) == 7 and headings.index("Layer 8") == headings.index(step_headings[-1]) + 1,
        headings,
    )
    check("intro and complete notes", "start on the table" in texts and "is complete" in texts)


def test_pdf_nested_part_is_a_pre_step_on_its_parents_step_page():
    model, _, _ = tail_model(with_inner_nub=True)
    plan = build_order.plan_build(model.bricks)
    check(
        "S1.1 is labelled 3a (step 3 of S1)",
        subassembly_pages.inline_step_labels(plan) == {"S1.1": "3a"},
        subassembly_pages.inline_step_labels(plan),
    )
    with tempfile.TemporaryDirectory() as folder:
        story = build_pages(model, plan, folder)
        headings, texts = headings_and_texts(story)
        map_title_ok = os.path.exists(os.path.join(folder, "sub", "S1.1_step_01.png"))
    check("no own page or heading for the nested part", not any("S1.1" in heading for heading in headings), headings)
    step_3 = headings.index("Sub-assembly S1 - step 3 / 4 (Layer 4)")
    check(
        "step 3 page: 3a connect, then 3b put it in place",
        "3a:</b></font> first connect these 2 pieces" in texts
        and "<b>3b:</b> put <b>3a</b> in its place" in texts
        and step_3 >= 0,
        texts,
    )
    check("its maps are written", map_title_ok)


def test_many_pre_steps_get_their_own_page():
    """With a low height limit even one pre-step gets its own page, right before its step page."""
    model, _, _ = tail_model(with_inner_nub=True)
    plan = build_order.plan_build(model.bricks)
    original = subassembly_pages.INLINE_MAX_HEIGHT_ON_STEP_PAGE_INCH
    subassembly_pages.INLINE_MAX_HEIGHT_ON_STEP_PAGE_INCH = 0.5  # force the own page with one pre-step
    try:
        with tempfile.TemporaryDirectory() as folder:
            headings, _ = headings_and_texts(build_pages(model, plan, folder))
    finally:
        subassembly_pages.INLINE_MAX_HEIGHT_ON_STEP_PAGE_INCH = original
    pre_step = "Sub-assembly S1 - step 3 / 4 (Layer 4): pre-steps 3a"
    check(
        "pre-step page right before the step page",
        pre_step in headings and headings[headings.index(pre_step) + 1] == "Sub-assembly S1 - step 3 / 4 (Layer 4)",
        headings,
    )


def test_3d_view_draws_a_brick_as_one_block():
    cells = np.ones((3, 1, 1), bool)  # one stud, 3 plates high
    colours = np.full((3, 1, 1, 3), 200, np.uint8)
    outline = (30, 30, 30)

    def outline_pixels(piece_numbers):
        return count_pixels(fr.iso_render(cells, colours, fr.IsoView(px=20, piece_ids_zyx=piece_numbers)), outline)

    per_cell = count_pixels(fr.iso_render(cells, colours, fr.IsoView(px=20)), outline)
    one_brick = outline_pixels(np.ones((3, 1, 1), np.int32))
    three_plates = outline_pixels(np.array([1, 2, 3], np.int32).reshape(3, 1, 1))
    check(
        "one brick: no lines between its plate layers; three plates: the lines stay",
        one_brick < three_plates == per_cell,
        (one_brick, three_plates, per_cell),
    )


def test_layers_with_nothing_to_place_get_no_page():
    """Layer 0 a 2x1 plate, layers 1-3 a 2x1 brick, layer 4 a 2x1 plate: layers 2 and 3 (figure "Layer 3-4") only
    have the brick reaching up through them."""
    pieces = [
        PlacedBrick(0, 0, 0, 2, 1, 1, BLACK),
        PlacedBrick(0, 0, 1, 2, 1, 3, BLACK),
        PlacedBrick(0, 0, 4, 2, 1, 1, BLACK),
    ]
    model = model_of(pieces, shape=(5, 1, 2))
    with tempfile.TemporaryDirectory() as folder:
        headings, texts = headings_and_texts(build_pages(model, build_order.plan_build(model.bricks), folder))
    check("no pages for layers 3-4", headings == ["Layer 1", "Layer 2", "Layer 5"], headings)
    check("the next page says why", "Layers 3-4: nothing new to place" in texts, texts)


def test_map_crop_leaves_out_parts_placed_later():
    model, _, _ = tail_model()
    plan = build_order.plan_build(model.bricks)
    check(
        "layer 3 map (z 2) crops to the column only",
        oh.map_region(model, 2, show_layer_below=True, margin_studs=0, plan=plan) == (0, 2, 0, 1),
        oh.map_region(model, 2, show_layer_below=True, margin_studs=0, plan=plan),
    )
    check(
        "without the plan the tail is inside the crop",
        oh.map_region(model, 2, show_layer_below=True, margin_studs=0)[1] == 6,
    )


def test_save_outputs_prints_the_build_plan_and_keeps_the_model():
    model, _, _ = long_tail_model()
    before = [(piece.x, piece.y, piece.z, piece.w, piece.d, piece.h_plates, piece.color) for piece in model.bricks]
    grid_before = model.grid.copy()
    output = io.StringIO()
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(output):
        oh.save_outputs(model, "t", folder, options=OutputOptions(make_pdf=True))
        has_pdf = os.path.exists(os.path.join(folder, "t_instructions.pdf"))
        has_step_map = os.path.exists(os.path.join(folder, "layers", "subassemblies", "S1_step_01.png"))
    after = [(piece.x, piece.y, piece.z, piece.w, piece.d, piece.h_plates, piece.color) for piece in model.bricks]
    check(
        "console: build plan line",
        "Build plan: 1 sub-assemblies for the parts that hang while building (S1 7 "
        "pieces (layers 2-8, placed on Layer 8); 0 small one(s) on shared pages)" in output.getvalue(),
        output.getvalue(),
    )
    check("PDF and sub-assembly images written", has_pdf and has_step_map)
    check("model unchanged", before == after and np.array_equal(grid_before, model.grid))


def test_no_cli_option_any_more():
    source = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image_to_bricks", "cli.py"), encoding="utf-8"
    ).read()
    check("--subassembly-min-pieces removed", "--subassembly-min-pieces" not in source)


if __name__ == "__main__":
    for test in [value for key, value in list(globals().items()) if key.startswith("test_")]:
        test()
    sys.exit(1 if failed else 0)
