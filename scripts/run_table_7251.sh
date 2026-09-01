#!/usr/bin/env bash
# Table 1 on the 7,251-molecule set. Complete xTB coverage, verified pairing
# and atom ordering; no descriptor fallback can trigger.
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

run () {
  local name="$1"; shift
  if [[ -f "$OUT/$name/cross_validation_summary.csv" ]]; then
    echo "[$(date -u +%H:%M)] $name already done, skipping"; return 0
  fi
  echo "[$(date -u +%H:%M)] === $name ==="
  python -m downstream.toxicity.run_cross_validation $COMMON --output-dir "$OUT/$name" "$@"
  find "$OUT/$name" -path "*/checkpoints/*.ckpt" -delete 2>/dev/null
  echo "[$(date -u +%H:%M)] $name done. disk: $(df -h . | awk 'NR==2{print $4}') free"
}

NB="neighbor_k2 neighbor_k3 neighbor_k4"

run baseline        --token-feature-normalize-mode none
run gasteiger       --token-features charge --token-feature-normalize-mode none
run xtb_charge      --token-features charge --charge-xyz-dir $XYZ --token-feature-normalize-mode none
run neighbors       --token-features $NB --token-feature-normalize-mode none
run xtb_polar       --token-features polarizability --charge-xyz-dir $XYZ --token-feature-normalize-mode none
run rdkit_mr        --token-features polarizability --token-feature-normalize-mode none
run xtb_orbital     --molecule-features orbital --charge-xyz-dir $XYZ --token-feature-normalize-mode none
run rdkit_physchem  --molecule-features physchem --token-feature-normalize-mode none
run gast_neigh      --token-features charge $NB --token-feature-normalize-mode dataset
run gast_neigh_mr   --token-features charge $NB polarizability --token-feature-normalize-mode dataset

echo "ALL DONE $(date -u)"
