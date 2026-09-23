"""Does attribution move off the toxicophore when the descriptor model corrects
a false positive?

This is the measurement the whole project has been building toward. Everything
before it was predictions and SMARTS patterns; this is the first analysis that
looks at attributions.

    cd ~/physchem-xai_upload && source .venv/bin/activate && export PYTHONPATH=src
    python delta_analysis.py

Inputs
------
runs_7251/{baseline,gasteiger}/fold_{1..5}/counterfactual_occlusion/
    test_recompute_all/atom_occlusion.csv
flip_cohort.csv          (from count_flips.py)
toxicophores/alerts_kazius29.csv

What it computes
----------------
occlusion_score = positive_probability - perturbed_probability, so a positive
value means removing that atom lowered the predicted toxicity: the atom argued
for toxicity.

Raw scores are NOT comparable between models -- a confident model has more
probability to lose, so its scores are larger everywhere. The comparison is
therefore made on the *distribution* of attribution over atoms:

    frac_alert = (attribution on alert atoms) / (attribution on all atoms)

which is what the published figures show when they colour atoms relative to
each other. Reported two ways:

    _abs  denominator sum|score|          -- uses every atom, symmetric
    _pos  denominator sum max(score, 0)   -- only atoms arguing for toxicity

and as enrichment, frac_alert / (n_alert_atoms / n_atoms), where 1.0 means the
alert receives no more attribution than its share of the molecule.

The test
--------
    delta = frac_alert(baseline) - frac_alert(gasteiger)

positive delta = the descriptor model puts less of its attention on the alert.

    PRIMARY  76 corrected false positives -- expect delta > 0
    CONTROL  molecules both models get right -- expect delta ~ 0
    CONTROL  molecules the descriptor model broke -- if delta is equally
             positive here, the shift does not track correctness and there
             is no result
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------- alerts
def load_alerts(path: Path):
    out = []
    with open(path, encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            pattern = Chem.MolFromSmarts(record["smarts"])
            if pattern is None:
                raise ValueError(f"unparseable SMARTS: {record['alert_id']}")
            out.append((record["alert_id"], record["name"], pattern))
    return out


def alert_atoms(smiles: str, alerts) -> tuple[set[int], int, list[str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set(), 0, []
    atoms, names = set(), []
    for _, name, pattern in alerts:
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            names.append(name)
            for match in matches:
                atoms.update(match)
    return atoms, mol.GetNumAtoms(), names


# ---------------------------------------------------------------- attribution
def per_molecule_attribution(frame: pd.DataFrame, alerts, min_ok: float):
    """One row per molecule: how much of its attribution sits on the alert."""
    rows = []
    for smiles, group in frame.groupby("smiles", sort=False):
        n_rows = len(group)
        ok = group[group["status"] == "ok"] if "status" in group else group
        if n_rows == 0 or len(ok) / n_rows < min_ok:
            continue

        atoms, n_atoms, names = alert_atoms(smiles, alerts)
        if not atoms or n_atoms == 0:
            continue

        idx = ok["atom_index"].to_numpy()
        score = ok["occlusion_score"].to_numpy(dtype=float)
        in_alert = np.isin(idx, list(atoms))

        abs_all = np.abs(score).sum()
        pos_all = np.clip(score, 0, None).sum()
        if abs_all <= 1e-12:
            continue

        frac_abs = np.abs(score[in_alert]).sum() / abs_all
        frac_pos = (np.clip(score[in_alert], 0, None).sum() / pos_all
                    if pos_all > 1e-12 else np.nan)

        share = len(atoms) / n_atoms   # the alert's share of the molecule
        rows.append({
            "smiles": smiles,
            "label": int(group["label"].iloc[0]),
            "base_prob": float(group["positive_probability"].iloc[0]),
            "n_atoms": n_atoms,
            "n_alert_atoms": len(atoms),
            "alert_share": share,
            "frac_abs": frac_abs,
            "frac_pos": frac_pos,
            "enrich_abs": frac_abs / share if share > 0 else np.nan,
            "enrich_pos": frac_pos / share if share > 0 else np.nan,
            "frac_ok": len(ok) / n_rows,
            "alert_names": "|".join(names),
        })
    return pd.DataFrame(rows)


def load_config(runs: Path, config: str, folds, subdir: str, alerts, min_ok: float):
    parts, missing, status_total, status_ok = [], [], 0, 0
    for fold in folds:
        path = runs / config / f"fold_{fold}" / "counterfactual_occlusion" / subdir / "atom_occlusion.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        raw = pd.read_csv(path)
        if "status" in raw:
            status_total += len(raw)
            status_ok += (raw["status"] == "ok").sum()
        per_mol = per_molecule_attribution(raw, alerts, min_ok)
        per_mol["fold"] = fold
        parts.append(per_mol)

    if missing:
        print(f"  MISSING for {config}:")
        for m in missing:
            print(f"    {m}")
    if status_total:
        pct = 100 * status_ok / status_total
        flag = "  <-- high failure rate" if pct < 90 else ""
        print(f"  {config}: {status_ok}/{status_total} atom rows ok ({pct:.1f}%){flag}")

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------------- report
def test_stratum(name: str, delta: pd.Series, expect: str = ""):
    d = delta.dropna()
    if len(d) < 3:
        print(f"  {name:34s} n={len(d):5d}   too few to test")
        return None
    t, p = stats.ttest_1samp(d, 0.0)
    w = stats.wilcoxon(d)[1] if len(d) >= 10 else float("nan")
    effect = d.mean() / d.std() if d.std() > 0 else np.nan
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {name:34s} n={len(d):5d}  mean {d.mean():+.4f}  median {d.median():+.4f}"
          f"  t={t:+6.2f}  p={p:8.2g} {star:3s} d={effect:+.2f}  {expect}")
    return {"stratum": name, "n": len(d), "mean": d.mean(), "median": d.median(),
            "t": t, "p": p, "wilcoxon_p": w, "cohens_d": effect}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_7251"))
    parser.add_argument("--baseline-config", default="baseline")
    parser.add_argument("--descriptor-config", default="gasteiger")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--subdir", default="test_recompute_all",
                        help="use test_mask_all to analyse the smoke-test output")
    parser.add_argument("--cohort", type=Path, default=Path("flip_cohort.csv"))
    parser.add_argument("--alerts", type=Path, default=Path("toxicophores/alerts_kazius29.csv"))
    parser.add_argument("--metric", default="frac_abs",
                        choices=("frac_abs", "frac_pos", "enrich_abs", "enrich_pos"))
    parser.add_argument("--min-ok", type=float, default=0.8,
                        help="skip molecules where fewer than this fraction of atoms occluded cleanly")
    parser.add_argument("--out", type=Path, default=Path("delta_results.csv"))
    args = parser.parse_args()

    alerts = load_alerts(args.alerts)
    print(f"{len(alerts)} alerts\n")

    print("loading occlusion output")
    base = load_config(args.runs_dir, args.baseline_config, args.folds, args.subdir, alerts, args.min_ok)
    desc = load_config(args.runs_dir, args.descriptor_config, args.folds, args.subdir, alerts, args.min_ok)
    if base.empty or desc.empty:
        raise SystemExit("\nno occlusion output found -- has the run finished?")
    print(f"\n  {args.baseline_config:12s} {len(base):5d} alert-bearing molecules")
    print(f"  {args.descriptor_config:12s} {len(desc):5d} alert-bearing molecules")

    merged = base.merge(desc, on=["smiles", "fold"], suffixes=("_base", "_desc"))
    print(f"  paired                {len(merged):5d}")

    if not (merged["label_base"] == merged["label_desc"]).all():
        raise SystemExit("labels disagree between configs -- splits are misaligned")

    cohort = pd.read_csv(args.cohort)[["smiles", "fold", "stratum", "label",
                                       "prob_baseline", "prob_descriptor"]]
    merged = merged.merge(cohort, on=["smiles", "fold"], how="left")
    if merged["stratum"].isna().any():
        n = int(merged["stratum"].isna().sum())
        print(f"  WARNING: {n} molecules absent from {args.cohort}; dropping")
        merged = merged.dropna(subset=["stratum"])

    m = args.metric
    merged["delta"] = merged[f"{m}_base"] - merged[f"{m}_desc"]

    print("\n" + "=" * 96)
    print(f"DELTA = {m}(baseline) - {m}(descriptor)")
    print("positive delta = the descriptor model puts LESS attribution on the alert")
    print("=" * 96 + "\n")

    fp_fixed = merged[(merged["stratum"] == "flip_to_correct") & (merged["label"] == 0)]
    fn_fixed = merged[(merged["stratum"] == "flip_to_correct") & (merged["label"] == 1)]
    broke = merged[merged["stratum"] == "flip_to_wrong"]
    broke_fp = broke[broke["label"] == 0]
    both_ok = merged[merged["stratum"] == "both_correct"]

    results = [
        test_stratum("PRIMARY: false positives fixed", fp_fixed["delta"], "<-- the claim"),
        test_stratum("false negatives fixed", fn_fixed["delta"]),
        test_stratum("CONTROL: both correct", both_ok["delta"], "<-- expect ~0"),
        test_stratum("CONTROL: descriptor broke it", broke["delta"], "<-- expect ~0"),
        test_stratum("  of those, true non-mutagens", broke_fp["delta"]),
    ]

    print("\n" + "-" * 96)
    print("THE DECIDING COMPARISON: is the shift specific to corrected molecules?")
    print("-" * 96)
    for other_name, other in (("both correct", both_ok), ("broke it", broke)):
        a, b = fp_fixed["delta"].dropna(), other["delta"].dropna()
        if len(a) >= 3 and len(b) >= 3:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  fp_fixed ({a.mean():+.4f}) vs {other_name} ({b.mean():+.4f}): "
                  f"t={t:+.2f} p={p:.3g} {star}  |  Mann-Whitney p={pu:.3g}")

    print("\n" + "-" * 96)
    print("RAW LEVELS (not differences) — sanity check")
    print("-" * 96)
    print(f"  {'stratum':34s} {'baseline':>10s} {'descriptor':>11s} {'alert share':>12s}")
    for name, frame in (("false positives fixed", fp_fixed), ("false negatives fixed", fn_fixed),
                        ("both correct", both_ok), ("descriptor broke it", broke)):
        if len(frame):
            print(f"  {name:34s} {frame[f'{m}_base'].mean():10.4f} "
                  f"{frame[f'{m}_desc'].mean():11.4f} {frame['alert_share_base'].mean():12.4f}")

    print("\n" + "-" * 96)
    print("BY ALERT, within the primary cohort")
    print("-" * 96)
    from collections import defaultdict
    by_alert = defaultdict(list)
    for _, row in fp_fixed.iterrows():
        for name in str(row["alert_names_base"]).split("|"):
            if name:
                by_alert[name].append(row["delta"])
    rows = sorted(((k, np.mean(v), len(v)) for k, v in by_alert.items() if len(v) >= 5),
                  key=lambda r: -r[1])
    for name, mean_delta, n in rows:
        vals = by_alert[name]
        p = stats.ttest_1samp(vals, 0)[1] if len(vals) >= 3 else float("nan")
        star = "*" if p < 0.05 else ""
        print(f"  {name[:50]:50s} n={n:3d}  mean delta {mean_delta:+.4f}  p={p:.3g} {star}")

    merged.to_csv(args.out, index=False)
    summary = pd.DataFrame([r for r in results if r])
    summary.to_csv(str(args.out).replace(".csv", "_summary.csv"), index=False)
    print(f"\nwrote {args.out} and {str(args.out).replace('.csv', '_summary.csv')}")

    print("\n" + "=" * 96)
    print("HOW TO READ THIS")
    print("=" * 96)
    print("  Primary positive and significant, controls near zero")
    print("      -> the descriptor model moves attribution off the alert specifically")
    print("         when it corrects a false positive. That is the paper's claim,")
    print("         measured over the cohort instead of illustrated with two figures.")
    print("  Primary positive but 'broke it' equally positive")
    print("      -> the shift is what descriptors do to alert-bearing molecules in")
    print("         general, not something that tracks correctness. No claim.")
    print("  Primary near zero")
    print("      -> the published figures are not representative of the cohort.")
    print("         Worth knowing before a reviewer finds it.")


if __name__ == "__main__":
    main()
