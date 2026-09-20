set -euo pipefail
source .venv/bin/activate
export PYTHONPATH=src
OUT=results/xai_pest
mkdir -p "$OUT"

python scripts/xai/count_flips.py --runs-dir runs_pest \
    --baseline-config baseline --descriptor-config gasteiger \
    --baseline-token-features \
    --descriptor-token-features charge \
    --baseline-token-feature-normalize-mode none \
    --descriptor-token-feature-normalize-mode none \
    --out "$OUT/flip_cohort_gast.csv"

python scripts/xai/integrated_gradients.py \
    --runs-dir runs_pest --config baseline \
    --token-features \
    --token-feature-normalize-mode none \
    --cohort "$OUT/flip_cohort_gast.csv" --steps 64 \
    --out "$OUT/ig_baseline_gast.csv"

python scripts/xai/integrated_gradients.py \
    --runs-dir runs_pest --config gasteiger \
    --token-features charge \
    --token-feature-normalize-mode none \
    --cohort "$OUT/flip_cohort_gast.csv" --steps 64 \
    --out "$OUT/ig_gasteiger_pest.csv"
