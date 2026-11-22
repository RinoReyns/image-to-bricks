HOW TO RUN
  python tests/run_all.py              every test file, one summary; one file per CPU core, slowest first (1-2 minutes)
  python tests/run_all.py --fast       also skip the golden builds and the CLI end-to-end runs (under a minute)
  python tests/run_all.py --exhaustive sweep tests check every configuration, not a fixed sample; the base sweep is
                                       split into one process per job (about 2 minutes with --jobs 4)
  python tests/run_all.py --jobs 2     only 2 test processes at a time (default: number of CPU cores)
  python tests/run_all.py golden       only files whose name contains "golden"
  python tests/test_x.py [name_part]   one file (function-style files: only tests whose name contains name_part)

LINT RATCHET (tests/test_lint_ratchet.py): fails when a file / module in the "not clean yet" lists of pyproject.toml
is already clean (delete its entry); skipped when ruff / mypy are not installed (pip install -r requirements-dev.txt).

GOLDEN BUILDS = the safety net for refactoring (tests/test_golden_builds.py)
  Fixed commands on fixed models must give EXACTLY the frozen result (every piece position, inventory, stock report).
  tests/fixtures/public/  robot / frog / bear: our own figures (committed). golden_cases.json = their commands.
  tests/fixtures/local/   the local reference build (third-party model, NEVER committed, see .gitignore). Skipped when missing.
  Freeze once on the machine you refactor on, BEFORE changing code:
      python tests/test_golden_builds.py --update-golden
  then after every refactor step:
      python tests/test_golden_builds.py
  Results depend on numpy / scipy / OpenCV versions (k-means colour groups shift on big models), so each case also
  freezes its environment; a case frozen with other versions is SKIPPED with the version difference (re-freeze).
  Re-freeze only after a change that is MEANT to change the output, and say so in the commit message.
  New public figure shapes: edit tests/fixtures/public/figure_shapes.py, run make_public_fixtures.py (needs
  scikit-image), then --update-golden.

FUNCTION-STYLE TESTS (need the real trimesh; helpers in tests/support.py)
  python tests/test_mesh_geometry.py        load_mesh (scene node transform), rotate around the centroid, scale to
                                            the target studs + Z stretch, voxelise (supersampling keeps true volume)
  python tests/test_colour_mapping.py       nearest palette colour, allowed list, colour boost, k-means groups (light
                                            and shadow one colour, dark neutral -> Black unless Black not allowed),
                                            palette limit, smoothing, absorb, overrides, barycentric, denoise, unify
  python tests/test_colour_sampling.py      assign_voxel_colors from vertex colours and from a UV texture (GLB round
                                            trip), interior filled, --color-alias, --paint, --missing-rgb, debug PNGs,
                                            regression: meshes with fewer than 16 triangles no longer crash
  python tests/test_rarely_used_colour_options.py  --carve-gap (cape side found, gap cut, body recoloured),
                                            --color-groups 0 (region growing), --harmonize (band columns)
  python tests/test_cli_options.py          argparse value types, box / front-oval parsers, layered-base option
                                            checks, real CLI processes rejecting bad options with clear messages
  python tests/test_stock_ledger.py         stock CSV loading rules, row status, report + buy list files, summary,
                                            give back, recount, fork isolation
  python tests/test_catalog_tools.py        build_catalog.py (hand edits kept, backup, specials) and check_cost.py
                                            (shortfall only, colour price, compatible fallback, factor, unpriced
                                            guess) on copies in temp folders: the real catalogue is never touched
  python tests/test_colorize_mesh_helpers.py colorize_mesh.py helpers without OpenGL (silhouettes, sampling mask,
                                            look-at pose, adjacency, highlight denoise, palette, fill unseen)
  python tests/test_pipeline_end_to_end.py  whole CLI on a synthetic Y-up two-colour figure: every output file, PDF
                                            pages, exact cover, one assembly, head on top after rotation, allowed
                                            colours, width = target studs, inventory / stock report / buy list agree,
                                            same command -> same build, no stock -> no stock files, --debug writes
                                            the placement check (and nothing without it)
  python tests/test_golden_builds.py        see GOLDEN BUILDS above

SCRIPT-STYLE TESTS (older; synthetic, no mesh needed)
Synthetic regression tests (no mesh needed). Run from the project folder:
  python tests/test_stability.py   graph / centre-of-mass checks on hand-built models
  python tests/test_base.py        base: exact coverage, figure unchanged by the base, PDF/CSV
  python tests/test_base_holds_together.py  base shapes x plates x figure sizes x margins: exact cover, one assembly,
                                   no loose base piece; 160 seeded cases, all 3200 with --exhaustive (about 5 min)
  python tests/test_tie.py         tie repair + interior growth on a synthetic figure
  python tests/test_footing_margin.py  single-stud footing merge + automatic base margin
  python tests/test_stock.py       --stock: owned pieces first, size fallback, colours as hard requirements, base + tie repair ledger
  python tests/test_cleanup.py     --cleanup-small (2 random shapes, 4 with --exhaustive): lossless, never more 1x1 / pieces, wall not weaker, stock ledger
  python tests/test_voxel.py       --voxel-supersample: true volume, wider gaps, thin parts stay connected
  python tests/test_trim.py        --trim-colors full rings per part, --color-alias
  python tests/test_thicken.py     --thicken-box: 1-wide parts in a box get a plus cross-section (joints 2+ studs)
  python tests/test_floating.py    pieces with nothing under them (PDF marks) + --prop-box post
  python tests/test_accent.py      --accent special part + special stock rows are never packed, running ledger == final pieces, small plates of the wanted colour merged into the accent plate
  python tests/test_retile.py      --retile: re-tiling around single-stud joints, drop / box options, stock ledger
  python tests/test_axes.py        layer-map axes from the base plate corner, --front arrow, {front_stud}, stability plate_x / plate_y
  python tests/test_base_zones.py  --base-zone: pond / path colours, one colour per base column, visible cells keep the zone colour (with and without --stock)
  python tests/test_scale_boxes.py --boxes-for-studs: boxes scaled to another figure size
  python tests/test_sheets.py      --thicken-sheet-box: one-plate sheets (hat brim) get a 2nd plate, no floating ring
They stub trimesh/tqdm if those are not installed.
  python tests/test_base_bricks.py  --base-bricks: bricks in the lowest base band, no loose base pieces
  python tests/test_details.py     Dark Pink in the palette, --smooth-iters / --absorb-min-region keep 1-2 cell details (eyes, nose)
  python tests/test_log.py         run_log.txt: console output copied to <outdir>/run_log.txt
  python tests/test_paint.py       --paint: surface cells in a box get a colour (eyes, nose); interior / empty cells untouched
  python tests/test_views.py       flat views named by the figure's own sides (--front), asymmetric marker (raised right paw, nose) for every front, figure-grid axes, nearest cell wins, isometric view names from the camera position
  python tests/test_nose_brick.py  --paint-deep: owned pink 1x2 brick as a nose (one stud hidden in the wall) is used, also through tie repair + clean-up
  python tests/test_l_pieces.py    --use-l-pieces: owned L pieces (corner plate / brick 2x2) replace small pieces, colour rules, stock ledger, layer maps / PDF, whole pipeline
  python tests/test_check_cost_l.py check_cost.py prices the L corner pieces (owned ones subtracted, others from price_list.csv)
  python tests/test_base_packer.py  --base-hidden-layers: layered base of big plates (exact cover, zone colours on top, hidden layers any owned colour, joints spanned >= 75 %, --base-max-plate, stock ledger, CLI checks)
  python tests/test_base_ties.py    --base-sparse-layers: sparse tie layer (every seam chunk tied, corner feet, holes in the grid, one assembly, tip angle, stock ledger)
  python tests/test_tie_speed.py    tie repair speed-ups: fast _grow_options / _sample_tilings / Stock.sizes_with == plain definitions, progress lines
  python tests/test_color_cleanup.py --open-gaps / --fill-pinholes / --despeckle / --neutral-enclaves: gap spurs removed (details kept), pinholes closed in the sheet, real gaps kept, lone patches, grey enclaves, trim colours / paints locked, CLI
  python tests/test_trim_snap.py    --trim-snap: ring-less trim snaps to the ring layer of the part under it (none beside a part), rings one layer, visible band, strays recoloured (also after smoothing), no ring on a smaller part (tail) or on a separate body (--trim-snap-similarity)
  python tests/test_hat_shell.py    --join-corners: diagonal thin flap side-connected, corner gaps between big parts / separate bodies stay; --dark-enclaves: small spots inside Black become Black, painted / locked / colourful areas kept
  python tests/test_owned_layers.py --stock ownership per layer (owned pieces handed out bottom-up, Own / Buy columns, _owned_layers.csv) and --base-fill-colors (base re-tiled with spare owned plates, figure pieces never taken, hidden layers any colour, loose check)
  python tests/test_base_layout.py --base-layout: hand-built base CSV loaded / validated, seam + corner check, placement (feet on top plates, largest tip angle, centred), final base == layout after tie repair, technic special owned, ledger
  python tests/test_anchor_floating.py --anchor-floating: floating plate re-tiled with a supported same-colour neighbour, level 2 splits a neighbouring brick, other colours / weaker results untouched, cells unchanged, ledger
  python tests/test_ground_overhangs.py --ground-overhangs: tall wall columns starting 1-2 plates in the air extended down (column colour), specks / brims / tails left, dents closed
  python tests/test_merge_plates.py --merge-plates: 2-3 neighbouring plates filling a standard plate become one (1x1+1x1 -> 1x2, 1x2+2x1+hidden 2x1 -> 3x2), visible colours kept, buying never grows, weaker results / bricks / base untouched
  python tests/test_layer_maps.py  layer maps show the layer below (hatched where the new layer leaves it uncovered, light colours darkened, L pieces, crop includes it, legend row; --no-layer-below = old map; model unchanged) and the PDF counts the note lines so the map stays on its layer page
  python tests/test_add_later.py   the former "add later from below" cases are now small sub-assemblies (nub + its holder); only a never-clamped group is held
  python tests/test_subassemblies.py every hanging part + its clamping pieces = a sub-assembly (nested ones too), built on pages before the layer it is placed on (small ones share a page); not drawn before that layer, faded + named on it; model unchanged
  python tests/test_drop_box.py    --drop-box: filled cells in a figure-grid box removed after every other shape fix (ground overhangs cannot put them back), colours cleared, WARNING for an empty box
  python tests/test_prefer_bricks.py  --prefer-bricks: bricks first in 3 band shifts, bought even when owned plates of the colour are left; colours kept, fewer plates; off by default
  python tests/test_front_oval.py  --front-oval: oval colour bands on the front fur (middle colour first), wall behind painted too, belt / non-fur cells kept, --front decides the painted face
