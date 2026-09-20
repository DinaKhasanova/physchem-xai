set -euo pipefail
source .venv/bin/activate
export PYTHONPATH=src

python scripts/xai/count_flips.py --runs-dir runs_7251 \
    --baseline-config baseline --descriptor-config gasteiger \
    --baseline-token-features \
    --descriptor-token-features charge \
    --baseline-token-feature-normalize-mode none \
    --descriptor-token-feature-normalize-mode none \
    --folds 1 --out results/xai_pest/ames_check_fold1.csv

python -c "
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
d = pd.read_csv('results/xai_pest/ames_check_fold1.csv')
for m in ['baseline','descriptor']:
    print(m, 'auc=%.4f bal=%.4f' % (roc_auc_score(d.label, d['prob_'+m]),
                                    balanced_accuracy_score(d.label, d['pred_'+m])))
print()
print('recorded fold-1 rows:')
for run in ['baseline','gasteiger']:
    rows = open('runs_7251/'+run+'/cross_validation_fold_metrics.csv').read().strip().split(chr(10))
    print(' ', run, rows[0])
    print(' ', run, rows[1])
"
