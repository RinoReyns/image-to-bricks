---
name: "clean-code"
description: "Use for ANY code change in the image-to-bricks project (write, edit, refactor, review): enforces clean-code rules, tests and the project's working style."
---

# Clean code rules for image-to-bricks

Apply to every change in the project (Python, `model_to_bricks.py`, `pipeline.py`, `placement.py`, `stability.py`, `tie_repair.py`, `stock.py`, `l_pieces.py`, ...). Before finishing, run the checklist at the end and say in one line which rules you touched.

## 1. Structure
- **Single responsibility**: one function does one thing, one module has one reason to change. A function that packs AND scores AND logs gets split.
- **Small functions**: aim for <= 30 lines, <= 4 parameters (group related ones in a dataclass, e.g. `PackOptions`, `BaseOptions`, `TieOptions`). Nesting depth <= 3; use early returns / guard clauses.
- **Small modules**: a file over ~600 lines is a smell. Do not add a new feature to a big file (`colour_sampling.py`, `placement.py`): put it in its own module (as `l_pieces.py`, `cleanup.py`, `stock.py` already do) and call it from one place in the pipeline.
- **Pipeline stays readable**: `optimize_brick_placement` should read like a list of named steps (voxelise -> colour -> pack -> base -> tie repair -> cleanup -> L pieces). Each step is a function with explicit inputs and outputs.
- **Dependencies point one way**: `consts` / `utils` <- domain modules <- pipeline <- CLI. No imports from the CLI layer into domain code; no circular imports.

## 2. Naming: self-explanatory names (variables, functions, classes, files)
A reader must understand what a name means WITHOUT reading its body or its call sites. If a name needs a comment to explain it, rename it instead.
- **Functions are verbs that say exactly what they do and what they return**: `count_single_stud_joints`, `find_weak_joints`, `build_stud_contacts`, `recolour_from_neighbours`. Not `process`, `handle`, `do_check`, `run2`, `fix`, `helper`, `stuff`. Predicates start with `is_` / `has_` / `can_` (`is_hidden_cell`, `has_owned_piece`). Getters that compute say so (`compute_`, `find_`, `count_`, `build_`), they do not hide behind `get_`.
- **Variables are nouns that say what the value is, including its unit or kind**: `layer_count`, `base_margin_studs`, `visible_cell_mask`, `owned_piece_counts`, `tip_angle_degrees`. Not `n`, `tmp`, `res`, `data`, `val`, `arr`, `lst`, `d`, `m`, `flag`, `cnt`.
- **No abbreviations** unless they are universal in this domain (`rgb`, `csv`, `pdf`, `ldraw`, `x/y/z`). Write `neighbour_count`, not `nbr_cnt`; `candidate`, not `cand`; `placed_brick`, not `pb`.
- **Name the role, not the type**: `loose_pieces`, not `piece_list`; `weak_joints`, not `joint_set`.
- **Collections are plural, single items singular**: `for piece in pieces`, `for layer_index in range(layer_count)`. Dicts say the mapping: `colour_by_cell`, `stock_count_by_piece_key`.
- **Loop indices**: only `i`, `j` for trivial loops over an index with no meaning; anything with meaning gets a name (`layer_index`, `stud_x`). Coordinates `x, y, z, w, d` are fine only inside the grid / geometry code.
- **Booleans read as true/false statements**: `is_visible`, `has_stock`, `should_recolour`; never `flag`, `mode`, `check`, or negatives like `not_hidden` / `no_ties` inside code (the CLI flag `--no-ties` maps to `use_tie_repair`).
- **Same word, same concept** everywhere (layer / plate / brick / piece; figure grid vs plate coordinates). Say which coordinate system or unit a name is in when not obvious (`grid_x`, `plate_x`, `height_plates`, `width_studs`).
- **Constants are UPPER_SNAKE and describe meaning plus unit**: `DEFAULT_MIN_TIP_ANGLE_DEGREES`, not `MTA` or `LIMIT`.
- **Classes are nouns** (`PlacedBrick`, `StockLedger`); dataclass fields follow the same variable rules.
- **Modules and tests**: file names say the topic (`l_pieces.py`, `stability.py`); tests are named for the behaviour (`test_hidden_cells_take_any_owned_colour`).
- **Length is fine**: a long clear name beats a short vague one (`count_pieces_hanging_on_single_stud` > `cnt_hang`). If a name gets very long, the function is probably doing too much: split it.
- **Rename on sight**: when you edit code that has a cryptic name, rename it (rename-only commit, tests green) as part of the boy-scout rule. Do not leave a new cryptic name behind.
- Comments explain WHY (a decision, a trap, a geometry fact), never WHAT. Delete comments that restate code. No commented-out code and no `step 17`-style history in code (history lives in git).
- Docstring on every public function / class: purpose, arguments, return value, units / coordinate system, side effects.

## 3. No magic numbers, no duplication (DRY)
- Every tuning value is a named constant in `consts.py` (or the top of its module) with a comment on its unit and meaning, AND exposed as a CLI option when a user may want to change it. Defaults live in ONE place, shared by the CLI and the code.
- If the same logic appears twice, extract it. Three similar loops over 2x2 boxes / neighbours = one helper. Colour names, palette and piece sizes come only from the palette table in `consts.py`.
- Do not copy-paste an option parser (`Z0:Z1,Y0:Y1,X0:X1`) - one `parse_box` used by all box options.

## 4. Data and functions
- **Prefer pure functions**: inputs -> outputs, no hidden global state. Mutate only what the docstring says. Do not mutate arguments silently; if a function changes the model in place (ledger, grid), name it as such (`apply_...`, `_inplace`).
- Use dataclasses / NamedTuples for records instead of positional tuples (the 7-tuple of base pieces is a smell: `PlacedBrick` fields are named). Never index tuples by magic position.
- **Type hints** on all signatures; `Optional` only where None is meaningful. numpy arrays: document shape and dtype (`grid[z, y, x]` bool).
- No global mutable state: settings come in as `options.BuildOptions` (or the one group a function needs, e.g. `ColourOptions`), the stock ledger as a `stock` argument. Upper-case module names are constants only, never set from the CLI.
- Do not use boolean flags that switch behaviour of a function (`do_x=True`): split into two functions or pass a strategy.

## 5. Errors and logging
- Fail early with a clear message that says what was wrong and how to fix it (option name, the value, the allowed range). Validate CLI input once, at the edge (`argparse` type functions), not deep in the pipeline.
- No bare `except:`; catch the specific exception. Never swallow errors silently. A fallback must print a WARNING line.
- Use one logging helper (console + `run_log.txt`), not scattered `print`s; messages are stable and greppable (tests and the docs quote them: `VERDICT`, `Accent:`, `L pieces:`).
- Warnings that are known and accepted must be either fixed or documented; a warning nobody understands (e.g. "stock ledger differs") is a bug to investigate, not noise.

## 6. Tests (non-negotiable)
- Every new option / behaviour gets a test in `tests/` (synthetic data, no mesh needed). Every bug fix gets a regression test that fails before the fix.
- Tests are small, named for the behaviour, one concept each, no dependence on test order or on files outside `tests/`.
- Invariants to assert where relevant: cells and visible colours unchanged by repair / cleanup, stock ledger == recount of final pieces, no loose pieces, exact cover (no overlap, no gap), deterministic output for a fixed seed.
- Run the whole test suite (see `tests/README.txt`) after ANY change to the pipeline modules, `stability`, `tie_repair`, `stock`, `cleanup`, `l_pieces`. Report the result (N passed). Never leave a failing or skipped test without saying why.

## 7. Refactoring rules
- **Behaviour first**: a refactor must not change output. Prove it: same piece list / inventory CSV on the synthetic fixtures (and on the saved `*_model.pkl` of the last real run when available) before and after.
- Refactor in small steps, tests green after each; one kind of change per step (rename OR move OR split, not all together). Never mix a refactor and a feature in the same step.
- Boy-scout rule: when you edit a function, leave it cleaner (names, dead code, missing type hints), but do not rewrite unrelated code.
- Delete dead code and unused options / imports instead of keeping them "just in case" (git remembers). Exception: options the user still uses in the documented commands.
- Keep old pickles loading (class defaults) unless the user agrees to break them. Renaming a field of a pickled class needs a compatibility alias.
- Never rename a CLI option or a quoted log line (`VERDICT`, `Accent:`) without telling the user: documented commands and tests depend on them.

## 8. Performance and safety
- Measure before optimising; keep the readable version unless it is a real bottleneck (the pipeline already prints timings). Document why an optimisation is not obvious.
- Determinism: every random choice uses a seeded generator (`--base-seed`, tiling seeds); same command -> same result.
- Never overwrite user data: output goes to a NEW `--outdir`; never edit the user's catalogue / stock CSV without being asked; the stock loader must ignore unknown rows rather than crash.

## 9. Project working style (keep)
- Short answers. Analyse first, propose a short plan, then apply step by step.
- Code stays generic: model-specific values (cell coordinates, colours) live in CLI options, never in code.
- After each change give the exact full command (no `<placeholders>`; cmd.exe treats `<` as redirect) with a NEW `--outdir`.
- One commit per step, tests green first; the commit message says what changed, how to run it, what to check. No
  `Co-Authored-By` or other attribution lines in commit messages. Update the Project docs when the plan changes.
- Device files: copy each file to a NEW staged filename before every commit, re-stage and `cmp` to verify. Edit the plan from the latest container copy, never from an older re-staged one.

## Checklist before you say "done"
1. Does each new / changed function do one thing, fit on a screen, have type hints and a docstring?
2. **Names**: can every new or touched variable, function and class be understood without reading its body (verb for functions, noun with unit for variables, no abbreviations, no `tmp` / `data` / `res`)?
3. Any magic number, duplicated block, tuple-by-position, or new global state? Fix it.
4. Any new option: validated at the edge, default in one place, documented (option help text, README)?
5. Tests added for the feature / bug, whole suite run, result reported?
6. Behaviour of untouched options unchanged (identical output on fixtures)?
7. No dead code, no commented-out code, no stale history comments, no leftover debug prints?
8. Plan doc updated, full command with a new `--outdir` given?
