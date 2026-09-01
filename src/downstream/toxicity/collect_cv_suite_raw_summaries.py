from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def collect_raw_summaries(labels: list[str], run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, run_dir in zip(labels, run_dirs, strict=True):
        summary_path = run_dir / "cross_validation_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Expected cross-validation summary at {summary_path}")
        row = pd.read_csv(summary_path).iloc[0].to_dict()
        row["label"] = label
        row["run_dir"] = str(run_dir)
        rows.append(row)
    columns = ["label", "run_dir", *[key for key in rows[0].keys() if key not in {"label", "run_dir"}]]
    return pd.DataFrame(rows)[columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect raw CV summary rows from multiple run directories.")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.labels) != len(args.run_dirs):
        raise ValueError("--labels and --run-dirs must have the same length.")
    summary_df = collect_raw_summaries(args.labels, args.run_dirs)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_path, index=False)
    print(f"Saved raw suite summaries to {args.output_path}")


if __name__ == "__main__":
    main()
