from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def format_mean_and_error(mean_value: float, error_value: float, digits: int = 5) -> str:
    return f"{mean_value:.{digits}f} ± {error_value:.{digits}f}"


def summarize_cv_suite(labels: list[str], run_dirs: list[Path], error_metric: str = "se") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, run_dir in zip(labels, run_dirs, strict=True):
        summary_path = run_dir / "cross_validation_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Expected cross-validation summary at {summary_path}")
        summary = pd.read_csv(summary_path).iloc[0]
        roc_error = float(summary[f"roc_auc_{error_metric}"])
        balacc_error = float(summary[f"balanced_accuracy_{error_metric}"])
        rows.append(
            {
                "Model / feature set": label,
                "ROC AUC mean": float(summary["roc_auc_mean"]),
                f"ROC AUC {error_metric.upper()}": roc_error,
                "Balanced accuracy mean": float(summary["balanced_accuracy_mean"]),
                f"Balanced accuracy {error_metric.upper()}": balacc_error,
                "ROC AUC": format_mean_and_error(float(summary["roc_auc_mean"]), roc_error),
                "Balanced accuracy": format_mean_and_error(float(summary["balanced_accuracy_mean"]), balacc_error),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a publication-ready summary table from CV run directories.")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--error-metric", choices=("se", "sd"), default="se")
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.labels) != len(args.run_dirs):
        raise ValueError("--labels and --run-dirs must have the same length.")
    summary_df = summarize_cv_suite(args.labels, args.run_dirs, error_metric=args.error_metric)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_path, index=False)
    print(f"Saved suite summary to {args.output_path}")


if __name__ == "__main__":
    main()
