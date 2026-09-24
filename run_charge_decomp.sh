#!/usr/bin/env bash
# Does the partial-charge gain come from atom typing or from electronic context?
#
#   charge_z      plain charge, z-scored            <- control
#   charge_typed  per-atom-type mean charge only    <- atom typing, context removed
#   charge_resid  charge minus that mean            <- context only, typing removed
#
# All three reach the model at identical magnitude, so any accuracy difference
# is information content, not scale. Compare against runs_7251/baseline and
# runs_7251/gasteiger.
#
# Interpretation: if charge_typed recovers most of gasteiger's gain over
# baseline, the improvement is sharper atom typing. If charge_resid does, it is
# genuine electronic structure.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
source .venv/bin/activate
export PYTHONPATH=src

DATA=data/ames_7251.csv
XYZ=data/xtb_7251
OUT=runs_7251
COMMON="--dataset-path $DATA --fusion-type smiles_anchor_residual \
--token-weight-mode abs --fusion-learning-rate 2e-3 --gate-bias-init 0.0 \
--batch-size 16 --max-epochs 15 --seed 42 --n-splits 5"

if [[ ! -f data/charge_type_table.json ]]; then
  echo "[$(date -u +%H:%M)] building the atom-type charge table (labels never used)"
  python scripts/build_charge_type_table.py --dataset-path "$DATA" --charge-xyz-dir "$XYZ"
fi

run () {
  local name="$1"; shift
  if [[ -f "$OUT/$name/cross_validation_summary.csv" ]]; then
    echo "[$(date -u +%H:%M)] $name already done, skipping"; return 0
  fi
  echo "[$(date -u +%H:%M)] === $name ==="
  python -m downstream.toxicity.run_cross_validation $COMMON --output-dir "$OUT/$name" "$@"
  find "$OUT/$name" -path "*/checkpoints/*.ckpt" -delete 2>/dev/null
}

# Gasteiger source (matches runs_7251/gasteiger)
run gast_z      --token-features charge_z     --token-feature-normalize-mode none
run gast_typed  --token-features charge_typed --token-feature-normalize-mode none
run gast_resid  --token-features charge_resid --token-feature-normalize-mode none

# xTB source (matches runs_7251/xtb_charge)
run xtb_z       --token-features charge_z     --charge-xyz-dir "$XYZ" --token-feature-normalize-mode none
run xtb_typed   --token-features charge_typed --charge-xyz-dir "$XYZ" --token-feature-normalize-mode none
run xtb_resid   --token-features charge_resid --charge-xyz-dir "$XYZ" --token-feature-normalize-mode none

echo "[$(date -u +%H:%M)] ALL DONE"
python scripts/compare_runs_stats.py --runs \
  $OUT/baseline $OUT/gasteiger $OUT/gast_z $OUT/gast_typed $OUT/gast_resid \
  $OUT/xtb_charge $OUT/xtb_z $OUT/xtb_typed $OUT/xtb_resid 2>/dev/null || true
