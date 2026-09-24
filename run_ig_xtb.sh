#!/usr/bin/env bash
# Integrated gradients for the xTB-charge model on the 7,251-molecule set.
# Mirrors run_ig_gast.sh exactly; the only differences are
#   --descriptor-config xtb_charge   (instead of gasteiger)
#   --charge-xyz-dir data/xtb_7251   (forwarded only to the descriptor model)
# Training used: --token-features charge --charge-xyz-dir $XYZ
#                --token-feature-normalize-mode none   (scripts/run_table_7251.sh:31)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
source .venv/bin/activate
export PYTHONPATH=src

OUT=results/xai
XYZ=data/xtb_7251
mkdir -p "$OUT"

echo "[$(date -u +%H:%M)] 1/3 flip cohort (baseline vs xtb_charge)"
python scripts/xai/count_flips.py --runs-dir runs_7251 \
    --baseline-config baseline --descriptor-config xtb_charge \
    --baseline-token-features \
    --descriptor-token-features charge \
    --charge-xyz-dir "$XYZ" \
    --baseline-token-feature-normalize-mode none \
    --descriptor-token-feature-normalize-mode none \
    --out "$OUT/flip_cohort_xtb.csv"

echo "[$(date -u +%H:%M)] 2/3 IG, baseline model on the xtb cohort"
python scripts/xai/integrated_gradients.py \
    --runs-dir runs_7251 --config baseline \
    --token-features \
    --token-feature-normalize-mode none \
    --cohort "$OUT/flip_cohort_xtb.csv" --steps 64 \
    --out "$OUT/ig_baseline_xtb.csv"

echo "[$(date -u +%H:%M)] 3/3 IG, xtb_charge model"
python scripts/xai/integrated_gradients.py \
    --runs-dir runs_7251 --config xtb_charge \
    --token-features charge \
    --charge-xyz-dir "$XYZ" \
    --token-feature-normalize-mode none \
    --cohort "$OUT/flip_cohort_xtb.csv" --steps 64 \
    --out "$OUT/ig_xtb.csv"

echo "[$(date -u +%H:%M)] done -> $OUT/ig_baseline_xtb.csv, $OUT/ig_xtb.csv"
