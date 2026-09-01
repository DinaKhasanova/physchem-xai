"""Paired statistics across cross-validation folds.

Reads ``cross_validation_fold_metrics.csv`` from two or more run directories and
reports, for each run against a reference run, the mean difference in ROC-AUC, a
paired t-test across folds, and the number of folds in which the run improved.

Folds are paired by fold index, which is meaningful because every run uses the
same stratified split with seed 42.

Usage
-----
    python scripts/compare_runs_stats.py \\
        --baseline runs_7251/baseline \\
        --runs runs_7251/gasteiger runs_7251/xtb_charge \\
        --metric roc_auc

    # whole table at once, sorted by mean
    python scripts/compare_runs_stats.py \\
        --baseline runs_7251/baseline \\
        --runs runs_7251/*/ --sort
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

FOLD_METRICS_FILENAME = "cross_validation_fold_metrics.csv"


def load_fold_metric(run_dir: Path, metric: str) -> pd.Series:
    """Return the per-fold values of ``metric`` for one run, indexed by fold."""
    path = run_dir / FOLD_METRICS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"no {FOLD_METRICS_FILENAME} in {run_dir}")

    frame = pd.read_csv(path)
    if metric not in frame.columns:
        raise KeyError(
            f"{path} has no column {metric!r}; available: {list(frame.columns)}"
        )

    fold_column = next(
        (c for c in ("fold", "fold_index", "fold_number") if c in frame.columns),
        None,
    )
    if fold_column is None:
        frame = frame.reset_index().rename(columns={"index": "fold"})
        fold_column = "fold"

    series = frame.set_index(fold_column)[metric].sort_index()
    series.name = run_dir.name
    return series


def compare(reference: pd.Series, candidate: pd.Series) -> dict[str, float]:
    """Paired comparison of one run against the reference, across folds."""
    paired = pd.concat([reference, candidate], axis=1, join="inner")
    if len(paired) < 2:
        raise ValueError(
            f"only {len(paired)} folds shared between "
            f"{reference.name} and {candidate.name}"
        )

    ref_values = paired.iloc[:, 0]
    cand_values = paired.iloc[:, 1]
    differences = cand_values - ref_values

    result = stats.ttest_rel(cand_values, ref_values)

    return {
        "run": str(candidate.name),
        "n_folds": len(paired),
        "mean": float(cand_values.mean()),
        "sd": float(cand_values.std(ddof=1)),
        "delta": float(differences.mean()),
        "t": float(result.statistic),
        "p": float(result.pvalue),
        "folds_improved": int((differences > 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="reference run directory, e.g. runs_7251/baseline",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="run directories to compare against the baseline",
    )
    parser.add_argument(
        "--metric",
        default="roc_auc",
        help="column in cross_validation_fold_metrics.csv (default: roc_auc)",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="sort output by mean metric rather than by argument order",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="also write the table to this path",
    )
    args = parser.parse_args()

    reference = load_fold_metric(args.baseline, args.metric)

    rows: list[dict[str, float]] = []
    for run_dir in args.runs:
        run_dir = Path(str(run_dir).rstrip("/"))
        if run_dir.resolve() == args.baseline.resolve():
            continue
        try:
            candidate = load_fold_metric(run_dir, args.metric)
        except FileNotFoundError:
            continue  # not a run directory
        rows.append(compare(reference, candidate))

    if not rows:
        raise SystemExit("no comparable runs found")

    table = pd.DataFrame(rows)
    if args.sort:
        table = table.sort_values("mean")

    baseline_mean = reference.mean()
    baseline_sd = reference.std(ddof=1)
    print(
        f"baseline {args.baseline.name}: "
        f"{baseline_mean:.4f} +/- {baseline_sd:.4f} "
        f"({len(reference)} folds, metric={args.metric})"
    )
    print()

    header = f"{'run':<22}{'mean':>9}{'sd':>9}{'delta':>10}{'t':>8}{'p':>9}{'wins':>7}"
    print(header)
    print("-" * len(header))
    for row in table.to_dict("records"):
        print(
            f"{row['run']:<22}"
            f"{row['mean']:>9.4f}"
            f"{row['sd']:>9.4f}"
            f"{row['delta']:>+10.4f}"
            f"{row['t']:>8.2f}"
            f"{row['p']:>9.4f}"
            f"{row['folds_improved']:>5}/{row['n_folds']}"
        )

    print()
    print(
        "Paired t-test across folds, two-tailed. With 5 folds, t(4), the "
        "critical value at alpha=0.05 is 2.776."
    )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output_csv, index=False)
        print(f"\nwritten: {args.output_csv}")


if __name__ == "__main__":
    main()
