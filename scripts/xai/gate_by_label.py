"""Is the descriptor branch's behaviour explained by chemistry, or by the cohort?

The strata in gate_decomposition.py are defined by the model's own errors: a
molecule is in "FP corrected" *because* the descriptor branch pushed it across
the threshold. So a large contribution there is partly guaranteed by
construction, and the -1.39 vs +0.44 asymmetry cannot be read as evidence about
chemistry.

This script re-cuts the same data by TRUE LABEL and ALERT PRESENCE only --
nothing defined by model behaviour. If the descriptor branch still separates
active from deactivated toxicophores, that is a claim about chemistry rather
than about how the cohorts were drawn.

    python scripts/xai/gate_by_label.py

Reads results/xai/gate_decomposition.csv (all 7,251 molecules, no exclusions).
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


def load_alerts(path: Path):
    out = []
    with open(path, encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            pat = Chem.MolFromSmarts(r["smarts"])
            if pat is None:
                raise ValueError(r["alert_id"])
            out.append((r["name"], pat))
    return out


def summarise(name, vals, expect=""):
    v = pd.Series(vals).dropna()
    if len(v) < 3:
        print(f"  {name:44s} n={len(v):5d}   too few")
        return
    p = stats.ttest_1samp(v, 0.0)[1]
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {name:44s} n={len(v):5d}  mean {v.mean():+8.4f}  "
          f"median {v.median():+8.4f}  p {p:9.2g} {star:3s} {expect}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, default=Path("results/xai/gate_decomposition.csv"))
    parser.add_argument("--alerts", type=Path, default=Path("toxicophores/alerts_kazius29.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/xai/gate_by_label.csv"))
    args = parser.parse_args()

    data = pd.read_csv(args.gate)
    alerts = load_alerts(args.alerts)
    print(f"{len(data)} molecules, {len(alerts)} alerts\n")

    # alert membership, per molecule, independent of any model output
    names, has_alert = [], []
    for smi in data["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            names.append(""); has_alert.append(False); continue
        hits = [n for n, p in alerts if mol.GetSubstructMatches(p)]
        names.append("|".join(hits)); has_alert.append(bool(hits))
    data["alert_names"] = names
    data["has_alert"] = has_alert

    c = "descriptor_contribution"

    print("=" * 104)
    print("A. CUT BY TRUE LABEL AND ALERT ONLY — no cohort defined by model behaviour")
    print("=" * 104)
    print("  A structural alert is necessary but not sufficient. Among molecules that")
    print("  CARRY an alert, does the descriptor branch tell active from deactivated?\n")

    alert_pos = data[data["has_alert"]]
    toxic = alert_pos[alert_pos["label"] == 1]
    safe = alert_pos[alert_pos["label"] == 0]

    summarise("alert-bearing, truly TOXIC", toxic[c], "expect positive")
    summarise("alert-bearing, truly SAFE (deactivated)", safe[c], "expect negative / lower")

    if len(toxic) >= 3 and len(safe) >= 3:
        t, p = stats.ttest_ind(toxic[c], safe[c], equal_var=False)
        u, pu = stats.mannwhitneyu(toxic[c], safe[c], alternative="two-sided")
        pooled = np.sqrt((toxic[c].var() + safe[c].var()) / 2)
        d = (toxic[c].mean() - safe[c].mean()) / pooled if pooled > 0 else np.nan
        print(f"\n  SEPARATION: toxic - safe = {toxic[c].mean() - safe[c].mean():+.4f}")
        print(f"    Welch t = {t:+.2f}, p = {p:.3g}   Mann-Whitney p = {pu:.3g}   Cohen's d = {d:+.2f}")
        print("    This is the non-circular test. Cohorts are true labels, not model errors.")

    print("\n  For contrast, molecules with NO alert:")
    no_alert = data[~data["has_alert"]]
    summarise("no alert, truly TOXIC", no_alert[no_alert["label"] == 1][c])
    summarise("no alert, truly SAFE", no_alert[no_alert["label"] == 0][c])

    print("\n" + "=" * 104)
    print("B. BOTH-CORRECT STRATUM SPLIT BY LABEL")
    print("=" * 104)
    print("  gate_decomposition reported +0.06 overall but +1.55 on alert-bearing")
    print("  both-correct molecules. Splitting by label shows whether that is just")
    print("  most alert-bearing molecules being genuinely toxic.\n")
    bc = data[data["stratum"] == "both_correct"]
    summarise("both correct, all", bc[c])
    summarise("both correct, alert-bearing, toxic", bc[bc["has_alert"] & (bc["label"] == 1)][c])
    summarise("both correct, alert-bearing, safe", bc[bc["has_alert"] & (bc["label"] == 0)][c])
    summarise("both correct, no alert, toxic", bc[~bc["has_alert"] & (bc["label"] == 1)][c])
    summarise("both correct, no alert, safe", bc[~bc["has_alert"] & (bc["label"] == 0)][c])

    print("\n" + "=" * 104)
    print("C. PER ALERT — where does the branch discriminate?")
    print("=" * 104)
    print(f"  {'alert':44s} {'n tox':>6s} {'n safe':>7s} {'toxic':>9s} {'safe':>9s} {'diff':>9s} {'p':>10s}")
    print("  " + "-" * 100)
    rows = []
    for alert_name, _ in alerts:
        sub = alert_pos[alert_pos["alert_names"].str.contains(alert_name, regex=False, na=False)]
        tox, sfe = sub[sub["label"] == 1][c], sub[sub["label"] == 0][c]
        if len(tox) < 10 or len(sfe) < 10:
            continue
        t, p = stats.ttest_ind(tox, sfe, equal_var=False)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {alert_name[:44]:44s} {len(tox):6d} {len(sfe):7d} "
              f"{tox.mean():+9.4f} {sfe.mean():+9.4f} {tox.mean()-sfe.mean():+9.4f} {p:10.2g} {star}")
        rows.append({"alert": alert_name, "n_toxic": len(tox), "n_safe": len(sfe),
                     "mean_toxic": tox.mean(), "mean_safe": sfe.mean(),
                     "diff": tox.mean() - sfe.mean(), "p": p})

    print("\n" + "=" * 104)
    print("D. AS A CLASSIFIER — how much does the branch alone know?")
    print("=" * 104)
    from sklearn.metrics import roc_auc_score
    print("  Within alert-bearing molecules, alert presence carries no information")
    print("  (they all have one). Any discrimination comes from the descriptors.\n")
    if alert_pos["label"].nunique() == 2:
        auc = roc_auc_score(alert_pos["label"], alert_pos[c])
        print(f"    descriptor contribution alone, alert-bearing : ROC AUC {auc:.4f}  (n={len(alert_pos)})")
    if data["label"].nunique() == 2:
        auc_all = roc_auc_score(data["label"], data[c])
        print(f"    descriptor contribution alone, all molecules : ROC AUC {auc_all:.4f}  (n={len(data)})")
        auc_base = roc_auc_score(data["label"], data["base_logit_diff"])
        print(f"    baseline logit alone                         : ROC AUC {auc_base:.4f}")

    if rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    print("\n" + "=" * 104)
    print("HOW TO READ THIS")
    print("=" * 104)
    print("  Section A separation large and significant")
    print("      -> the descriptor branch distinguishes active from deactivated")
    print("         toxicophores. Cohorts are true labels, so this is a claim about")
    print("         chemistry, not an artefact of how the strata were drawn.")
    print("  Section A separation near zero")
    print("      -> the earlier asymmetry was cohort construction. Drop that sentence.")
    print("  Section D AUC near 0.5")
    print("      -> the branch's contribution carries little label information on its")
    print("         own, whatever it does at the margin.")


if __name__ == "__main__":
    main()
