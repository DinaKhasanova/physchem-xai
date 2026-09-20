set -euo pipefail
OUT=results/xai_pest
BT=""; DT="charge neighbor_k2 neighbor_k3 neighbor_k4"

python scripts/xai/count_flips.py --runs-dir runs_pest \
    --baseline-config baseline --descriptor-config gast_neigh \
    --baseline-token-features $BT --descriptor-token-features $DT \
    --baseline-token-feature-normalize-mode none \
    --descriptor-token-feature-normalize-mode dataset \
    --out "$OUT/flip_cohort_pest.csv"

python scripts/xai/integrated_gradients.py \
    --runs-dir runs_pest --config baseline --token-features $BT \
    --token-feature-normalize-mode none \
    --cohort "$OUT/flip_cohort_pest.csv" --steps 64 \
    --out "$OUT/ig_baseline_pest.csv"

python scripts/xai/integrated_gradients.py \
    --runs-dir runs_pest --config gast_neigh --token-features $DT \
    --token-feature-normalize-mode dataset \
    --cohort "$OUT/flip_cohort_pest.csv" --steps 64 \
    --out "$OUT/ig_gast_neigh_pest.csv"
