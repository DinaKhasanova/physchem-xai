#!/usr/bin/env bash
# Baseline vs Gasteiger on the 18,754-compound pesticide genotoxicity set.
# No xTB — all descriptors are RDKit-computable, so this is a subset of the
# Ames row set and directly comparable to it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source .venv/bin/activate
export PYTHONPATH=src

DATA=data/pesticides_genotoxicity_ames_format.csv
OUT=runs_pest
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

run baseline    --token-feature-normalize-mode none
run gasteiger   --token-features charge --token-feature-normalize-mode none


PHYSCHEM="physchem_molwt physchem_tpsa physchem_logp physchem_molmr physchem_hbd physchem_hba physchem_rotatablebonds physchem_ringcount physchem_heavyatomcount physchem_fractioncsp3"

export PHYSCHEM_STATS=pesticide

run physchem_tok  --token-features $PHYSCHEM --token-feature-normalize-mode dataset

echo "ALL DONE $(date -u)"

NB="neighbor_k2 neighbor_k3 neighbor_k4"

run neighbors   --token-features $NB --token-feature-normalize-mode none
run gast_neigh  --token-features charge $NB --token-feature-normalize-mode dataset

echo "ALL DONE $(date -u)"
