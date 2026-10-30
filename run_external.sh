#!/usr/bin/env bash
# External validation: Ames-trained models on non-overlapping pesticide compounds.
#
#   cd ~/Downloads/physchem-xai-main
#   bash run_external.sh
#
# Expects:
#   - external_validation.py in this folder (or already at scripts/external_validation.py)
#   - the five runs_7251__*.tar files in ~/Downloads  OR already unpacked into runs_7251/
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
ROOT="$PWD"
echo "repo root: $ROOT"
echo

# ---------------------------------------------------------------- 1. python
echo "=== 1. Python environment ==="
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -z "$PY" ] && { echo "No python3 found. Install Python 3.11 or 3.12 and re-run."; exit 1; }
echo "using $PY ($($PY -V 2>&1))"

if [ ! -d .venv ]; then
  echo "creating .venv ..."
  "$PY" -m venv .venv || { echo "venv creation failed"; exit 1; }
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip

echo "installing dependencies (a few minutes the first time) ..."
pip install -q torch rdkit lightning transformers scikit-learn pandas || {
  echo "dependency install failed"; exit 1; }

python - <<'PYEOF'
import torch, platform
print(f"  torch {torch.__version__} on {platform.machine()}")
if torch.cuda.is_available():
    print(f"  GPU: CUDA — {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("  GPU: Apple Silicon (MPS)")
else:
    print("  GPU: none detected — will run on CPU (slow, allow several hours)")
PYEOF
echo

# ---------------------------------------------------------------- 2. checkpoints
echo "=== 2. Checkpoints ==="
NEEDED="baseline gasteiger neighbors gast_neigh physchem_tok"
for cfg in $NEEDED; do
  if [ -f "runs_7251/$cfg/fold_5/toxicity_classifier.ckpt" ]; then
    echo "  $cfg already unpacked"
    continue
  fi
  TAR=""
  for d in "$HOME/Downloads" "/tmp/ckpt" "$ROOT"; do
    [ -f "$d/runs_7251__$cfg.tar" ] && TAR="$d/runs_7251__$cfg.tar" && break
  done
  if [ -z "$TAR" ]; then
    echo "  MISSING: runs_7251__$cfg.tar"
    echo "           download it from"
    echo "           https://github.com/DinaKhasanova/physchem-xai/releases/tag/checkpoints-v1"
    echo "           and leave it in ~/Downloads, then re-run this script."
    exit 1
  fi
  echo "  unpacking $TAR ..."
  tar -xf "$TAR" -C "$ROOT" || { echo "  tar failed for $cfg"; exit 1; }
done
N=$(find runs_7251 -name 'toxicity_classifier.ckpt' 2>/dev/null | wc -l | tr -d ' ')
echo "  checkpoints present: $N (expect 25 for the five configs)"
echo

# ---------------------------------------------------------------- 3. external set
echo "=== 3. External test set ==="
if [ ! -f data/pesticides_external.csv ]; then
  python - <<'PYEOF'
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(s))
a = pd.read_csv("data/ames_7251.csv");  a["c"] = a.smiles.map(canon)
p = pd.read_csv("data/pesticides_genotoxicity_ames_format.csv"); p["c"] = p.smiles.map(canon)
A = set(a.c.dropna())
ext = p[p.c.notna() & ~p.c.isin(A)].copy()
ext.to_csv("data/pesticides_external.csv", index=False)
print(f"  overlap discarded : {int(p.c.isin(A).sum())}")
print(f"  external test set : {len(ext)} "
      f"({int((ext.label==1).sum())} positive, {int((ext.label==0).sum())} negative)")
PYEOF
else
  echo "  data/pesticides_external.csv already built"
fi
echo

# ---------------------------------------------------------------- 4. run
echo "=== 4. Scoring ==="
[ -f external_validation.py ] && cp -f external_validation.py scripts/external_validation.py
[ -f scripts/external_validation.py ] || {
  echo "scripts/external_validation.py not found. Put external_validation.py in $ROOT and re-run."
  exit 1; }

unset PHYSCHEM_STATS              # Ames-trained models need the Ames statistics
export PYTHONPATH=src
export PYTORCH_ENABLE_MPS_FALLBACK=1   # harmless on CUDA/CPU

python scripts/external_validation.py \
  --runs-dir runs_7251 \
  --external-csv data/pesticides_external.csv \
  --batch-size 64 \
  --out results/external

echo
echo "Done. Results in $ROOT/results/external/"
