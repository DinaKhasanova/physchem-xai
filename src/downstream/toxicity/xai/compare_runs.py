"""Put two or more cross-validation runs side by side, with a paired fold test.

Reads ``cross_validation_fold_metrics.csv`` from each run directory, so it needs
no torch and no GPU. When a checkpoint is present it also reports how much
weight the classifier head puts on the descriptor block, which is the quantity
that should change if the head learning rate fix worked.

Usage
-----
    python -m downstream.toxicity.xai.compare_runs \
        --run "baseline (old lr)=runs_head_lr/baseline_old" \
        --run "baseline (head lr)=runs_head_lr/baseline_new" \
        --run "descriptors (old lr)=runs_head_lr/descriptors_old" \
        --run "descriptors (head lr)=runs_head_lr/descriptors_new" \
        --pair "descriptors (old lr)|descriptors (head lr)" \
        --pair "baseline (head lr)|descriptors (head lr)" \
        --token-features neighbor_k2 neighbor_k3 neighbor_k4 \
        --molecule-features physchem
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("roc_auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity")


def load_folds(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "cross_validation_fold_metrics.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "fold" not in frame.columns:
        frame["fold"] = range(1, len(frame) + 1)
    return frame


def summarise(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, frame in runs.items():
        row: dict[str, object] = {"run": label, "n_folds": len(frame)}
        for metric in METRICS:
            if metric in frame.columns:
                values = frame[metric].astype(float)
                row[metric] = values.mean()
                row[f"{metric}_sem"] = values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict[str, float]:
    """Paired comparison across matching folds. Returns mean difference and a t statistic."""

    merged = a[["fold", metric]].merge(b[["fold", metric]], on="fold", suffixes=("_a", "_b"))
    if merged.empty:
        return {}
    diff = merged[f"{metric}_b"].astype(float) - merged[f"{metric}_a"].astype(float)
    n = len(diff)
    mean = float(diff.mean())
    if n < 2 or diff.std(ddof=1) == 0:
        return {"n": n, "mean_diff": mean, "t": float("nan"), "wins": int((diff > 0).sum())}
    t = mean / (float(diff.std(ddof=1)) / math.sqrt(n))
    return {"n": n, "mean_diff": mean, "t": t, "wins": int((diff > 0).sum())}


def descriptor_share(
    run_dir: Path,
    token_features: tuple[str, ...],
    molecule_features: tuple[str, ...],
) -> dict[str, float] | None:
    """Fraction of classifier weight mass sitting on the descriptor blocks."""

    from .ckpt_peek import MOLECULE_FEATURE_DIMS, load_tensors  # noqa: PLC0415

    checkpoints = sorted(run_dir.glob("fold_*/toxicity_classifier.ckpt"))
    if not checkpoints:
        return None

    shares: list[float] = []
    ratios: list[float] = []
    init_ratios: list[float] = []
    for checkpoint in checkpoints:
        try:
            tensors = load_tensors(checkpoint)
        except Exception:
            continue
        gain = tensors.get("model.classifier.0.weight")
        linear = tensors.get("model.classifier.1.weight")
        if gain is None or linear is None:
            continue
        mass = np.abs(gain) * np.linalg.norm(linear, axis=0)
        in_features, encoder_hidden = int(gain.shape[0]), int(linear.shape[0])
        stats_dim = len(token_features) * 4
        molecule_dim = sum(MOLECULE_FEATURE_DIMS[name] for name in molecule_features)
        if encoder_hidden * 2 + stats_dim + molecule_dim != in_features:
            continue
        descriptor_start = encoder_hidden * 2
        descriptor_mass = mass[descriptor_start:]
        cls_mass = mass[:encoder_hidden]
        if descriptor_mass.size == 0:
            continue
        shares.append(float(descriptor_mass.sum() / mass.sum()))
        ratios.append(float(descriptor_mass.mean() / cls_mass.mean()))
        init_ratios.append(float(mass.mean() / math.sqrt(encoder_hidden / (3.0 * in_features))))

    if not shares:
        return None
    return {
        "n_checkpoints": len(shares),
        "descriptor_share_of_mass": float(np.mean(shares)),
        "descriptor_vs_cls_per_dim": float(np.mean(ratios)),
        "head_mass_vs_init": float(np.mean(init_ratios)),
    }


def describe(
    summary: pd.DataFrame,
    pairs: list[tuple[str, str, dict[str, dict[str, float]]]],
    weights: dict[str, dict[str, float] | None],
) -> str:
    lines = ["# Cross-validation comparison", ""]

    display = summary.copy()
    for metric in METRICS:
        sem = f"{metric}_sem"
        if metric in display.columns and sem in display.columns:
            display[metric] = [
                f"{value:.4f} ± {error:.4f}"
                for value, error in zip(display[metric], display[sem])
            ]
            display = display.drop(columns=[sem])
    lines.append(display.to_string(index=False))
    lines.append("")

    if pairs:
        lines.append("## Paired fold comparisons")
        lines.append("")
        lines.append("Same folds, same seed, so the folds pair up. `wins` counts folds where the")
        lines.append("second run beat the first. With 5 folds, 5/5 wins is about as strong as this")
        lines.append("design can get; |t| above ~2.8 is p < 0.05 at 4 degrees of freedom.")
        lines.append("")
        for first, second, results in pairs:
            lines.append(f"### {first}  ->  {second}")
            for metric, stats in results.items():
                if not stats:
                    continue
                lines.append(
                    f"  {metric:<20} mean diff {stats['mean_diff']:+.4f}   "
                    f"t = {stats['t']:+.2f}   wins {stats['wins']}/{stats['n']}"
                )
            lines.append("")

    reported = {label: value for label, value in weights.items() if value}
    if reported:
        lines.append("## Where the classifier head puts its weight")
        lines.append("")
        lines.append("`descriptor_vs_cls_per_dim` is the per-dimension weight mass of the descriptor")
        lines.append("blocks divided by that of the CLS block. At 1.00 the head treats a descriptor")
        lines.append("exactly like an average embedding dimension, i.e. it has learned nothing about")
        lines.append("them. `head_mass_vs_init` at 1.00 means the head weights are still at their")
        lines.append("random initialisation scale.")
        lines.append("")
        lines.append(
            f"  {'run':<28}{'descr share':>13}{'descr/cls per dim':>20}{'head vs init':>14}"
        )
        for label, value in reported.items():
            lines.append(
                f"  {label:<28}{value['descriptor_share_of_mass']:>13.4f}"
                f"{value['descriptor_vs_cls_per_dim']:>20.3f}{value['head_mass_vs_init']:>14.3f}"
            )
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--pair", action="append", default=[], metavar="LABEL_A|LABEL_B")
    parser.add_argument("--token-features", nargs="*", default=())
    parser.add_argument("--molecule-features", nargs="*", default=(), choices=("physchem", "orbital"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs: dict[str, pd.DataFrame] = {}
    directories: dict[str, Path] = {}
    for item in args.run:
        label, _, directory = item.partition("=")
        path = Path(directory)
        directories[label] = path
        frame = load_folds(path)
        if frame is None:
            print(f"! no cross_validation_fold_metrics.csv in {path} - skipping '{label}'")
            continue
        runs[label] = frame

    if not runs:
        raise SystemExit("No runs with results found.")

    summary = summarise(runs)

    pairs = []
    for item in args.pair:
        first, _, second = item.partition("|")
        if first not in runs or second not in runs:
            print(f"! cannot pair '{first}' with '{second}' - one of them has no results")
            continue
        results = {
            metric: paired_test(runs[first], runs[second], metric)
            for metric in ("roc_auc", "balanced_accuracy")
            if metric in runs[first].columns and metric in runs[second].columns
        }
        pairs.append((first, second, results))

    weights = {
        label: descriptor_share(path, tuple(args.token_features), tuple(args.molecule_features))
        for label, path in directories.items()
        if label in runs
    }

    text = describe(summary, pairs, weights)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
