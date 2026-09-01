#!/usr/bin/env bash
# Run the XAI diagnostic suite in the order that answers the cheapest question first.
#
#   bash scripts/run_xai_diagnostics.sh <CHECKPOINT> <RUN_DIR> [XYZ_DIR]
#
# CHECKPOINT  a .ckpt written by run_cross_validation / train
# RUN_DIR     the run directory containing splits/  (the --output-dir used for training)
# XYZ_DIR     optional protonation_*_xyz directory with an outs/ subdirectory
#
# Edit TOKEN_FEATURES / MOLECULE_FEATURES to match how the checkpoint was trained.
set -euo pipefail

CHECKPOINT="${1:?usage: run_xai_diagnostics.sh <checkpoint> <run_dir> [xyz_dir]}"
RUN_DIR="${2:?usage: run_xai_diagnostics.sh <checkpoint> <run_dir> [xyz_dir]}"
XYZ_DIR="${3:-}"

TOKEN_FEATURES="${TOKEN_FEATURES:-charge}"
MOLECULE_FEATURES="${MOLECULE_FEATURES:-physchem}"
DATASET="${DATASET:-data/ames_7251.csv}"
MAX_MOLECULES="${MAX_MOLECULES:-200}"

export PYTHONPATH="${PYTHONPATH:-src}"
XYZ_ARG=()
if [[ -n "${XYZ_DIR}" ]]; then
  XYZ_ARG=(--charge-xyz-dir "${XYZ_DIR}")
fi

echo "=============================================================="
echo "0/5  self-test"
echo "=============================================================="
python -m downstream.toxicity.xai.selftest

echo
echo "=============================================================="
echo "1/5  does the checkpoint use the descriptors at all?  (seconds)"
echo "=============================================================="
python -m downstream.toxicity.xai.diagnostics \
  --checkpoint-path "${CHECKPOINT}" \
  --token-features ${TOKEN_FEATURES} \
  --molecule-features ${MOLECULE_FEATURES} \
  --output-dir "${RUN_DIR}/xai_diagnostics"

if [[ -n "${XYZ_DIR}" ]]; then
  echo
  echo "=============================================================="
  echo "2/5  are the semiempirical values real, and on the right atoms?"
  echo "=============================================================="
  python -m downstream.toxicity.xai.coverage \
    --dataset-path "${DATASET}" \
    --charge-xyz-dir "${XYZ_DIR}" \
    --output-dir "${RUN_DIR}/xai_diagnostics"
fi

echo
echo "=============================================================="
echo "3/5  descriptor baselines (replaces the zero baseline)"
echo "=============================================================="
python -m downstream.toxicity.xai.baselines \
  --dataset-path "${DATASET}" \
  --molecule-features ${MOLECULE_FEATURES} \
  "${XYZ_ARG[@]}" \
  --output-dir "${RUN_DIR}/xai_diagnostics"

echo
echo "=============================================================="
echo "4/5  ablations"
echo "=============================================================="
python -m downstream.toxicity.xai.ablation \
  --checkpoint-path "${CHECKPOINT}" \
  --output-dir "${RUN_DIR}" \
  --token-features ${TOKEN_FEATURES} \
  --molecule-features ${MOLECULE_FEATURES} \
  "${XYZ_ARG[@]}"

echo
echo "=============================================================="
echo "5/5  explanation quality"
echo "=============================================================="
python -m downstream.toxicity.xai.faithfulness \
  --checkpoint-path "${CHECKPOINT}" \
  --output-dir "${RUN_DIR}" \
  --token-features ${TOKEN_FEATURES} \
  --molecule-features ${MOLECULE_FEATURES} \
  "${XYZ_ARG[@]}" \
  --max-molecules "${MAX_MOLECULES}" \
  --tag augmented

echo
echo "Results in ${RUN_DIR}/xai_diagnostics"
echo
echo "Not run automatically, because they need a second checkpoint or are slow:"
echo "  python -m downstream.toxicity.xai.counterfactual_occlusion --token-mode recompute ..."
echo "  python -m downstream.toxicity.xai.wins --baseline-csv ... --augmented-csv ..."
