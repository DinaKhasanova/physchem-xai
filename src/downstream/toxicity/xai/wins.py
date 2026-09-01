"""Which molecules does the descriptor-augmented model actually fix?

A 0.004 ROC AUC difference is not a result anyone can defend. A McNemar test on
the paired predictions is, because it asks the question the manuscript is
really making: on the same molecules, does adding descriptors turn more errors
into successes than the reverse?

This module pairs two prediction files on SMILES and reports

* the 2x2 discordance table and an exact McNemar p-value,
* the molecules the augmented model fixes and the ones it breaks,
* optional SMARTS-alert enrichment, so "descriptors help most for aromatic
  amines" becomes a number rather than two example figures.

Inputs are any CSVs carrying ``smiles``, ``label`` and either
``positive_probability`` or ``predicted_label`` -- the occlusion summary files
already written by this project qualify.

Usage
-----
    python -m downstream.toxicity.xai.wins \
        --baseline-csv runs_cv/baseline/occlusion/test_pad_all/test_occlusion_summary.csv \
        --augmented-csv runs_cv/physchem/occlusion/test_pad_all/test_occlusion_summary.csv \
        --alerts-smarts alerts.csv \
        --output-dir runs_cv/diagnostics
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from .common import ensure_dir


def _predicted(frame: pd.DataFrame, threshold: float) -> pd.Series:
    if "predicted_label" in frame.columns:
        return frame["predicted_label"].astype(int)
    if "positive_probability" in frame.columns:
        return (frame["positive_probability"] >= threshold).astype(int)
    raise ValueError("CSV needs a 'predicted_label' or 'positive_probability' column.")


def _load(path: Path, threshold: float, tag: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "smiles" not in frame.columns or "label" not in frame.columns:
        raise ValueError(f"{path} must contain 'smiles' and 'label' columns.")
    out = pd.DataFrame(
        {
            "smiles": frame["smiles"].astype(str),
            "label": frame["label"].astype(int),
            f"pred_{tag}": _predicted(frame, threshold),
        }
    )
    if "positive_probability" in frame.columns:
        out[f"prob_{tag}"] = frame["positive_probability"].astype(float)
    return out.drop_duplicates(subset="smiles")


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts ``b`` and ``c``."""

    n = b + c
    if n == 0:
        return 1.0
    smaller = min(b, c)
    tail = sum(math.comb(n, k) for k in range(0, smaller + 1)) * (0.5**n)
    return float(min(1.0, 2.0 * tail))


def compare(
    baseline_csv: Path,
    augmented_csv: Path,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    baseline = _load(baseline_csv, threshold, "baseline")
    augmented = _load(augmented_csv, threshold, "augmented")
    merged = baseline.merge(augmented.drop(columns=["label"]), on="smiles", how="inner")

    merged["baseline_correct"] = merged["pred_baseline"] == merged["label"]
    merged["augmented_correct"] = merged["pred_augmented"] == merged["label"]
    merged["outcome"] = "both_wrong"
    merged.loc[merged["baseline_correct"] & merged["augmented_correct"], "outcome"] = "both_correct"
    merged.loc[~merged["baseline_correct"] & merged["augmented_correct"], "outcome"] = "fixed"
    merged.loc[merged["baseline_correct"] & ~merged["augmented_correct"], "outcome"] = "broken"

    fixed = int((merged["outcome"] == "fixed").sum())
    broken = int((merged["outcome"] == "broken").sum())
    summary = {
        "n_paired": int(len(merged)),
        "both_correct": int((merged["outcome"] == "both_correct").sum()),
        "both_wrong": int((merged["outcome"] == "both_wrong").sum()),
        "fixed_by_descriptors": fixed,
        "broken_by_descriptors": broken,
        "net_gain": fixed - broken,
        "baseline_accuracy": float(merged["baseline_correct"].mean()),
        "augmented_accuracy": float(merged["augmented_correct"].mean()),
        "mcnemar_p_value": exact_mcnemar(broken, fixed),
    }
    return merged, summary


def alert_enrichment(merged: pd.DataFrame, alerts: dict[str, str]) -> pd.DataFrame:
    """Per-alert hit rate among fixed molecules versus everything else."""

    from rdkit import Chem  # noqa: PLC0415

    patterns = {}
    for name, smarts in alerts.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            patterns[name] = pattern
    if not patterns:
        return pd.DataFrame()

    molecules = {smiles: Chem.MolFromSmiles(smiles) for smiles in merged["smiles"].unique()}
    fixed_set = set(merged.loc[merged["outcome"] == "fixed", "smiles"])
    other_set = set(merged["smiles"]) - fixed_set

    rows = []
    for name, pattern in patterns.items():
        def hit_rate(group: set[str]) -> tuple[int, int]:
            hits = 0
            total = 0
            for smiles in group:
                mol = molecules.get(smiles)
                if mol is None:
                    continue
                total += 1
                if mol.HasSubstructMatch(pattern):
                    hits += 1
            return hits, total

        fixed_hits, fixed_total = hit_rate(fixed_set)
        other_hits, other_total = hit_rate(other_set)
        fixed_rate = fixed_hits / fixed_total if fixed_total else 0.0
        other_rate = other_hits / other_total if other_total else 0.0
        rows.append(
            {
                "alert": name,
                "fixed_hits": fixed_hits,
                "fixed_total": fixed_total,
                "fixed_rate": fixed_rate,
                "other_hits": other_hits,
                "other_total": other_total,
                "other_rate": other_rate,
                "enrichment": (fixed_rate / other_rate) if other_rate > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("enrichment", ascending=False).reset_index(drop=True)


def describe(summary: dict, enrichment: pd.DataFrame | None = None) -> str:
    lines = [
        "Paired comparison: SMILES-only baseline vs descriptor-augmented model",
        "",
        f"  molecules paired           : {summary['n_paired']}",
        f"  both correct               : {summary['both_correct']}",
        f"  both wrong                 : {summary['both_wrong']}",
        f"  fixed by descriptors       : {summary['fixed_by_descriptors']}",
        f"  broken by descriptors      : {summary['broken_by_descriptors']}",
        f"  net gain                   : {summary['net_gain']:+d}",
        "",
        f"  baseline accuracy          : {summary['baseline_accuracy']:.4f}",
        f"  augmented accuracy         : {summary['augmented_accuracy']:.4f}",
        f"  exact McNemar p-value      : {summary['mcnemar_p_value']:.4g}",
        "",
    ]
    if summary["mcnemar_p_value"] < 0.05:
        lines.append(
            "  The paired test is significant: the descriptor model fixes reliably more "
            "molecules than it breaks. This is a defensible claim even when the ROC AUC "
            "difference is small, and it is the sentence the results section should carry."
        )
    else:
        lines.append(
            "  The paired test is not significant: on these predictions the descriptor model "
            "fixes about as many molecules as it breaks. Reporting the aggregate metric "
            "difference as an improvement would not survive review."
        )
    if enrichment is not None and not enrichment.empty:
        lines.append("")
        lines.append("Structural-alert enrichment among the fixed molecules:")
        lines.append(enrichment.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--augmented-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--alerts-smarts",
        type=Path,
        default=None,
        help="Optional CSV with 'name' and 'smarts' columns for enrichment analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged, summary = compare(args.baseline_csv, args.augmented_csv, args.threshold)

    enrichment = None
    if args.alerts_smarts is not None and args.alerts_smarts.exists():
        alerts_frame = pd.read_csv(args.alerts_smarts)
        alerts = dict(zip(alerts_frame["name"].astype(str), alerts_frame["smarts"].astype(str)))
        enrichment = alert_enrichment(merged, alerts)

    out_dir = ensure_dir(args.output_dir)
    merged.to_csv(out_dir / "paired_predictions.csv", index=False)
    merged.loc[merged["outcome"] == "fixed"].to_csv(out_dir / "fixed_by_descriptors.csv", index=False)
    merged.loc[merged["outcome"] == "broken"].to_csv(out_dir / "broken_by_descriptors.csv", index=False)
    if enrichment is not None and not enrichment.empty:
        enrichment.to_csv(out_dir / "alert_enrichment.csv", index=False)

    text = describe(summary, enrichment)
    (out_dir / "paired_comparison.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
