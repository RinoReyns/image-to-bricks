# check_cost.py: what do the missing pieces of a build cost?

`check_cost.py` answers one question: **for a build the image-to-bricks pipeline produced, how much (in PLN) do I have to spend on pieces I do not own yet?**
It subtracts the bricks and plates I already have (`catalog_stock.csv`) from the pieces the build uses and prices only the rest. Python standard library only, nothing to install.

## Quick start

Run from the project folder (`image-to-bricks`):

```
python my_bricks_catalog\check_cost.py lego_output_figure
```

The argument is a build folder (the script picks the `*_inventory.csv` in it) or the inventory file itself.
Before running, make sure the stock is current: run `python my_bricks_catalog\build_catalog.py` after adding photos or hand-added pieces (and close the CSV in Excel first).

## How it works

1. **Read the build**: every `Plate` / `Brick` line of `*_inventory.csv` (`section`, `piece_type`, `width_studs`, `depth_studs`, `height_plates`, `color_name`, `quantity`). Base and figure lines are added together. The `TOTAL PIECES` line is skipped.
2. **Read what I own**: every `Plate` / `Brick` line of `catalog_stock.csv` (`quantity` column = photographed + hand-added pieces). Lines whose `piece_type` contains "special" (wedges, round plates, ...) are ignored.
3. **Match** by *type + size + colour name*. Size is compared without orientation (2x4 = 4x2). Colour names must be identical (`Dark Orange` is not `Orange`).
4. **Shortfall** per match: `need = max(0, used - owned)`. Pieces I own more of than the build needs cost nothing, and the surplus is not used for anything else.
5. **Price** each shortfall line (see below), multiply by the quantity, and add everything up.
6. **Report** on screen and in `<build>_cost.csv`.

## Price list (`price_list.csv`)

One line per piece type and size, price in **PLN per piece**:

| Column | Meaning |
|---|---|
| `piece_type` | `Plate` or `Brick` |
| `width_studs`, `depth_studs` | size (any orientation) |
| `color_name` | empty = same price for every colour. Fill it to give one colour its own price (a colour row wins over the empty row). |
| `price_lego_shop` | price from a LEGO single-piece shop (sourced: MojeKlocki24, 2026-10-31) |
| `price_compatible` | price of the non-LEGO brand I buy. Empty at the moment: fill it in. |
| `note` | where the price comes from; `ESTIMATE` = no real price found, a guess |

Example lines:

```
Plate,2,1,,0.39,,sourced: MojeKlocki24 ...
Plate,2,1,Black,0.35,0.12,black 1x2 plates are cheaper at my other shop
```

If a size has **no row**, the script uses a rough fallback (default 0.15 PLN per stud for plates, 0.25 for bricks; change with `--fallback-per-stud`) and marks it `UNPRICED GUESS`. Add a row to make it exact.

## Options

| Option | Default | Meaning |
|---|---|---|
| `build` (positional) | required | build folder or `*_inventory.csv` |
| `--stock FILE` | `catalog_stock.csv` next to the script | pieces I own |
| `--prices FILE` | `price_list.csv` next to the script | price table |
| `--profile lego_shop \| compatible` | `lego_shop` | which price column to use. If the cell is empty in `compatible`, the LEGO-shop price is used and the script says so. |
| `--factor X` | `1.0` | multiplies every price (discount, other brand at X times the price, ...) |
| `--fallback-per-stud PLATE BRICK` | `0.15 0.25` | PLN per stud for sizes without a price row |

Examples:

```
python my_bricks_catalog\check_cost.py lego_output_figure --profile compatible
python my_bricks_catalog\check_cost.py lego_output_figure --factor 0.5
python my_bricks_catalog\check_cost.py lego_output_figure --stock other_stock.csv
```

## Output

On screen:

```
Build:  figure_inventory.csv
Uses 2517 pieces; 96 of them I already own; 2421 to buy.
Prices: profile=lego_shop, factor=1.00 (PLN, shipping NOT included)

TOTAL TO BUY: 1282.10 PLN

By colour:
  Black                       611.58 PLN
  Dark Orange                 224.74 PLN
  ...
Biggest part types:
  Plate 1x1                472.20 PLN
  ...
Note: 23 lines use ESTIMATE prices from price_list.csv (no real price found yet).
```

(The example shows the format; the numbers come from a real 2,500-piece build.)

`<build>_cost.csv`, written next to the inventory, one line per missing piece:

| Column | Meaning |
|---|---|
| `piece_type`, `width_studs`, `depth_studs`, `height_plates`, `color_name` | the piece |
| `used` | how many the build uses |
| `have` | how many I own |
| `need` | `used - have` (only lines with `need > 0` appear) |
| `unit_pln`, `cost_pln` | price per piece (after `--factor`) and `need x unit_pln` |
| `price_source` | the `note` of the price row used, or `UNPRICED GUESS` |

The last line is the total. **Shipping is never included.**

## Limits to keep in mind

- **Exact colour only.** Spare Dark Orange does not cover a missing Orange; colour substitution is the packer's job.
- **No brick/plate swapping.** A spare brick does not cover three missing plates (possible future option).
- **Irregular pieces are ignored** in the stock, so they never count as owned. (Example: the chrome ball plate. The pipeline's own stock report counted it as an owned 2x1 plate, which made the report 1 piece lower.)
- **Prices are a snapshot.** Stock and prices change; the shop may not have hundreds of one part. The `ESTIMATE` rows are guesses.
- It prices what the build uses; it does not look for the cheapest shop or combine sellers.

## Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| `WARNING: stock file not found` | wrong `--stock` path; the script then assumes I own nothing |
| `UNPRICED GUESS: ... add rows for: Plate 8x4` | no price row for that size: add it to `price_list.csv` |
| `Note: no compatible price for ...` | `--profile compatible` but `price_compatible` is empty for that size (the LEGO-shop price was used) |
| Piece I own is still counted as missing | colour name or size differs between the build and `catalog_stock.csv` (check spelling, e.g. `Light Bluish Gray`) |
| Total differs from the pipeline's `*_stock_report.csv` | the stock changed after that report, or a special piece is involved (see limits) |
| Cannot save `catalog_stock.csv` / old numbers | the file is open in Excel; close it and rerun `build_catalog.py` |
