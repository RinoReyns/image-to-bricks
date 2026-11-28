# image-to-bricks (pictures -> coloured 3D model (GLB) -> LEGO build + instructions)

Layout:
- `model_to_bricks.py`: command-line entry point (same commands as always); the code is in the `image_to_bricks/` package:
  `cli.py` (command line -> `options.BuildOptions`), `pipeline.py` (the whole build, `build_model`), `options.py`
  (every setting, typed groups), `model.py` (LegoModel, PlacedBrick), `consts.py` (palette), `stock.py`, and:
  - `mesh/`: loading, voxelize (cells, hollowing, `surface_mask`), shape_edits, drop_cells, ground_overhangs
  - `colours/`: colour_sampling (`assign_voxel_colors`), palette_matching, surface_cleanup, hand_edits, cape_carve and the
    single clean-up rules (trim_snap, color_cleanup, gap_spurs, pinholes, corner_joins, fur_ovals)
  - `packing/`: placement (`optimize_brick_placement`), layer_packer, tie_repair, cleanup, l_pieces, accents, ...
  - `base/`: stand (base cells, margin), base_pieces (base packing), base_packer, base_layout, base_ties, base_fill
  - `analysis/` (stability, build order, debug images), `instructions/` (PDF, layer maps, renders).
- `images_to_model.py` / `image_to_bricks/images_to_model/`: pictures -> GLB (TRELLIS.2 shape, colours projected from
  the pictures, licence package; `verification/` = metrics and lock-file checks of the GPU setup). GPU only:
  `scripts/setup_images_to_model.sh` (Linux / WSL2) adds it to the same `.venv`, pinned in `lock/images_to_model/`;
  `scripts/reproduce_images_to_model.sh` checks the results (`tools/images_to_model/`). The GPU libraries are imported
  with importlib inside functions only, so the package and its tests run without them.
- `images_to_bricks.py` / `image_to_bricks/images_to_bricks.py`: both steps in one command.
- `tools/colorize_mesh.py`: colour a plain mesh from photos. `my_bricks_catalog/`: the user's bricks + catalogue scripts.
- `tests/` (`python tests/run_all.py`, see `tests/README.txt`), `docs/` (plans and the clean-code rules).
History: `git log` (one commit per step). Plan and current state: the Project docs (claude.ai Project "Lego Project").
Local only, never in git: `private/` (files kept out of the public repo, e.g. the image-to-3D engine evaluation),
`notes/` (old development diary), the catalogue data (`my_bricks_catalog/*.csv`, its
README), `build_optimization/` (hand-built bases of local models).

## Always
- Before ANY code change, read `docs/CLEAN_CODE_SKILL.md` and follow it (self-explanatory names, small single-purpose functions, no magic numbers, no duplication, type hints, docstrings, no new global state, no dead code).
- New feature goes in its own module or function; keep modules under about 600 lines.
- Code stays generic: model-specific values (coordinates, colours) are CLI options, never hard-coded.
- Every new option or behaviour gets a test in `tests/` (synthetic data). Every bug fix gets a regression test.
- After any change to the pipeline (mesh, colours, packing, base, stability, stock), run the whole test suite (`python tests/run_all.py`, see `tests/README.txt`) and report the result. Golden builds must stay identical.
- A refactor must not change output: compare the piece list / inventory CSV before and after on the synthetic fixtures.
- Never overwrite user data: new `--outdir` for every run; do not edit `catalog_stock.csv` unless asked.
- Deterministic output: seed every random choice.

## Working style
- Short answers. Analyse first, propose a short plan, then apply step by step.
- After each change give the exact full command (no `<placeholders>`, cmd.exe treats `<` as a redirect) with a NEW `--outdir`; the user runs it and we analyse the output together.
- One commit per step, tests green first (`python tests/run_all.py`; `--exhaustive` before a release). The commit message says what changed and
  how to check it. NEVER add `Co-Authored-By` or any other attribution line to commit messages.
- Update the Project docs when the plan or the current state changes.
- Nothing is pushed to GitHub until the clean-code refactor is finished.

## Checks (pre-commit, CI)
- `pre-commit run --all-files` runs ruff (lint + format), mypy and basic file checks; CI (`.github/workflows/ci.yml`)
  runs the same plus `tests/run_all.py` on Linux, Windows and macOS. Settings live in `pyproject.toml`.
- Strict rules everywhere; files that do not meet them yet are in the "not clean yet" lists in `pyproject.toml` (ruff
  per-file codes, mypy `ignore_errors` modules). When you clean a file, delete its entry in the same commit
  (`tests/test_lint_ratchet.py` fails on stale entries). Never add a new entry: new code is clean.
- Dependencies: edit `requirements.in` / `requirements-dev.in` (direct packages, exact versions), then
  `pip-compile requirements.in` and `pip-compile requirements-dev.in`. Never edit the generated `.txt` files.
  Keep the venv on pip 26.0.1 (`python -m pip install pip==26.0.1`): pip-tools 7.5.3 / 7.6.0 do not work with newer
  pip yet. Annotate arrays as `NDArray[...]`, never bare `np.ndarray` (fails mypy with the pinned numpy 2.2).

## Environment
- Windows, run from the project folder inside `.venv` (Python 3.12, see `.python-version`; CI also runs 3.13, on Linux, Windows and macOS).
- Windows locks open images: never reuse an `--outdir`.
