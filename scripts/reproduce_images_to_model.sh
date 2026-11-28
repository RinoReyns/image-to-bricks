#!/usr/bin/env bash
# Run the model step on the test characters robot, frog and bear and check the results against
# lock/images_to_model/expected_results.json (shape against the true shapes, colours against the pictures).
#   bash scripts/reproduce_images_to_model.sh                    PASS/FAIL per number (log: repro_out/reproduce.log)
#   bash scripts/reproduce_images_to_model.sh --write-expected   save this run as the baseline (only when we agree to)
# Needs bash scripts/setup_images_to_model.sh first. Runs offline, about 5 minutes (+1 the first time, for the true
# shapes).
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"
VENV="$(realpath -m "${VENV:-$PROJECT/.venv}")"
[ -f "$VENV/IMAGES_TO_MODEL_OK" ] || { echo "setup is not complete: run bash scripts/setup_images_to_model.sh first" >&2; exit 1; }
PYTHON="$VENV/bin/python"
OUT="${OUT:-repro_out}"
CHARACTERS=tests/fixtures/public/characters
CASES=(robot frog bear)
rm -rf "$OUT/runs" "$OUT/compare" && mkdir -p "$OUT"
if [ ! -f "$OUT/truth/bear.glb" ]; then
  echo "building the true shapes from the SDF scenes..."
  "$PYTHON" -m tools.images_to_model.build_truth "$CHARACTERS" --out "$OUT/truth"
fi
RUN_ARGS=() COMPARE_ARGS=()
for CASE in "${CASES[@]}"; do
  RUN_ARGS+=(--case "$CASE=$CHARACTERS/$CASE")
  COMPARE_ARGS+=(--case "$CASE" --reference "$CASE=$OUT/truth/$CASE.glb")
done
echo "running the model step on ${CASES[*]} (models load ~60 s)..."
"$PYTHON" -m tools.images_to_model.run_cases --out "$OUT/runs" "${RUN_ARGS[@]}" > "$OUT/reproduce.log" 2>&1 ||
  { tail -25 "$OUT/reproduce.log"; exit 1; }
echo "scoring against the true shapes..."
"$PYTHON" -m tools.images_to_model.compare_models --runs "$OUT/runs" "${COMPARE_ARGS[@]}" --out "$OUT/compare" \
  >> "$OUT/reproduce.log" 2>&1 || { tail -25 "$OUT/reproduce.log"; exit 1; }
"$PYTHON" -m tools.images_to_model.check_reproduction --runs "$OUT/runs" --compare "$OUT/compare/report.json" "$@" |
  tee "$OUT/check.txt"
