#!/usr/bin/env bash
# One-time VM setup. Run from the directory that contains the repository.
#
#   bash setup_vm.sh
#
# Installs the Python dependencies and checks that the data, the pretrained
# encoder and the patched code are all present.
set -euo pipefail

echo "=== Python ==="
python3 --version

if [[ ! -d .venv ]]; then
  echo "=== creating virtualenv ==="
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== installing dependencies (a few minutes) ==="
pip install --quiet --upgrade pip
pip install --quiet \
  "torch" \
  "lightning" \
  "transformers" \
  "rdkit" \
  "pandas" \
  "scikit-learn" \
  "matplotlib"

echo
echo "=== versions ==="
python -c "
import torch, lightning, transformers, rdkit, sklearn, pandas
print('torch       ', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('lightning   ', lightning.__version__)
print('transformers', transformers.__version__)
print('rdkit       ', rdkit.__version__)
print('sklearn     ', sklearn.__version__)
print('pandas      ', pandas.__version__)
"

echo
echo "=== files ==="
for path in data/ames_7251.csv src/downstream/toxicity/train.py src/downstream/toxicity/xai/ckpt_peek.py models/molencoder; do
  if [[ -e "$path" ]]; then echo "  ok      $path"; else echo "  MISSING $path"; fi
done
if [[ -d protonation_federico_corina_7246_xyz/outs ]]; then
  echo "  ok      protonation_federico_corina_7246_xyz/outs ($(ls protonation_federico_corina_7246_xyz/outs | wc -l) files)"
else
  echo "  MISSING protonation_federico_corina_7246_xyz/outs (only needed for semiempirical features)"
fi

echo
echo "=== is the head learning rate patch present? ==="
if grep -q "head_learning_rate" src/downstream/toxicity/train.py; then
  echo "  ok      train.py has --head-learning-rate"
else
  echo "  MISSING train.py is the unpatched version - copy the patched src over it"
fi

echo
echo "=== xai self-test ==="
PYTHONPATH=src python -m downstream.toxicity.xai.selftest

echo
echo "Setup done. Next:  bash scripts/run_head_lr_experiment.sh"
