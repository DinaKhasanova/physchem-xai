set -euo pipefail
OUT=results/xai_pest
mkdir -p "$OUT"

BASE_TOK=""
BASE_MOL=""
DESC_TOK="charge neighbor_k2 neighbor_k3 neighbor_k4"
DESC_MOL=""

echo "baseline   : token=[$BASE_TOK] molecule=[$BASE_MOL]"
echo "gast_neigh : token=[$DESC_TOK] molecule=[$DESC_MOL]"

python scripts/xai/count_flips.py \
    --runs-dir runs_pest \
    --baseline-config baseline --descriptor-config gast_neigh \
    --baseline-token-features $BASE_TOK --baseline-molecule-features $BASE_MOL \
    --descriptor-token-features $DESC_TOK --descriptor-molecule-features $DESC_MOL \
    --out "$OUT/flip_cohort_pest.csv"

wc -l "$OUT/flip_cohort_pest.csv"

python scripts/xai/integrated_gradients.py \
    --runs-dir runs_pest --config baseline \
    --token-features $BASE_TOK --molecule-features $BASE_MOL \
    --cohort "$OUT/flip_cohort_pest.csv" --steps 64 \
    --out "$OUT/ig_baseline_pest.csv"

python scripts/xai/integrated_gradients.py \
    --runs-dir runs_pest --config gast_neigh \
    --token-features $DESC_TOK --molecule-features $DESC_MOL \
    --cohort "$OUT/flip_cohort_pest.csv" --steps 64 \
    --out "$OUT/ig_gast_neigh_pest.csv"
