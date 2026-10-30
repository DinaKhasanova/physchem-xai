"""Does descriptor quality change WHERE the model looks, not just how well it scores?

Gasteiger and xTB charges reach statistically indistinguishable ROC AUC on the
Ames set, so accuracy cannot separate them. This script measures the other
axis: how much of the integrated-gradients attribution each descriptor moves
onto the atoms matched by a Kazius structural alert.

For every molecule the shift is

    shift = alert-atom fraction (descriptor model)
          - alert-atom fraction (that model's OWN SMILES-only baseline)

so each descriptor is compared only with itself and the two numbers are never
contaminated by a difference between the two baselines.

Two cohorts are reported, and they answer different questions:

  own cohort     each descriptor's own alert-bearing corrections (294 Gasteiger,
                 278 xTB). These are the headline per-descriptor shifts, but the
                 two means come from non-identical molecule sets, so their RATIO
                 is not a paired comparison.
  common cohort  the 212 molecules both descriptors corrected. The paired test
                 requires identical molecules, so the signed-rank test, the
                 bootstrap interval and the xTB - Gasteiger difference are all
                 computed here.

Usage
-----
    cd physchem-xai && source .venv/bin/activate && export PYTHONPATH=src

    python scripts/xai/attribution_shift.py

Reads results/xai/ig_{gasteiger,xtb,baseline,baseline_xtb}.csv, the two flip
cohorts from count_flips.py, toxicophores/alerts_kazius29.csv and the xTB
outputs in data/xtb_7251. Writes a per-alert breakdown (Supporting Information
Table S2) to results/xai/attribution_shift.csv.
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

RNG_SEED = 42
N_BOOTSTRAP = 10000


def load_alerts(path: Path):
    """SMARTS patterns for the Kazius alert set, in file order."""
    out = []
    with open(path, encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            pat = Chem.MolFromSmarts(r["smarts"])
            if pat is None:
                raise ValueError(r["alert_id"])
            out.append((r["name"], pat))
    return out


def alert_atoms(smiles: str, alerts):
    """Atom indices matched by any alert, and the names of the alerts that hit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set(), set()
    atoms, names = set(), set()
    for name, pat in alerts:
        for match in mol.GetSubstructMatches(pat):
            atoms.update(match)
            names.add(name)
    return atoms, names


def load_net_charges(xtb_dir: Path):
    """{canonical SMILES: net charge} as GFN2-xTB actually computed the molecule.

    The ionisation state is a property of the Epik-protonated structure at pH
    7.4, not of the SMILES the model reads: the cleaned TDC strings are written
    neutral almost throughout. Reading the formal charge off the SMILES would
    therefore report nearly every compound as neutral. The authoritative value
    is the `charge = N` line of each xTB output.
    """
    stems = {}
    with open(xtb_dir / "smiles_to_out_stem.csv", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            stems[r["canonical_smiles"]] = r["out_stem"]
    charges = {}
    for smiles, stem in stems.items():
        path = xtb_dir / "outs" / f"{stem}.out"
        if not path.exists():
            continue
        match = re.search(
            r"^charge\s*=\s*(-?\d+)", path.read_text(errors="ignore"), re.M
        )
        if match:
            charges[smiles] = int(match.group(1))
    return charges


def load_attributions(path: Path):
    """{smiles: {atom_index: attr_total}}; molecules with any blank cell are dropped.

    A handful of molecules (e.g. ClCl) have no defined descriptor channel and
    carry an empty attr_descriptor, so their attr_total is empty too. Those are
    excluded rather than silently read as zero.
    """
    table = collections.defaultdict(dict)
    with open(path, encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            value = r["attr_total"]
            table[r["smiles"]][int(r["atom_index"])] = float(value) if value else None
    return {s: a for s, a in table.items() if not any(v is None for v in a.values())}


def alert_fraction(attrs, atoms):
    """Share of total absolute attribution sitting on alert-matched atoms."""
    if attrs is None:
        return None
    total = sum(abs(v) for v in attrs.values())
    if total == 0:
        return None
    return sum(abs(v) for idx, v in attrs.items() if idx in atoms) / total


def load_cohort(path: Path):
    """Alert-bearing molecules the descriptor model corrected over the baseline."""
    with open(path, encoding="utf-8") as handle:
        return {
            r["smiles"]
            for r in csv.DictReader(handle)
            if r["stratum"] == "flip_to_correct"
            and r["alert_positive"] in ("1", "True", "true")
        }


def shifts(molecules, model, baseline, alerts):
    """Per-molecule shift, plus the alert names each molecule matched."""
    values, names, kept = [], [], []
    for smiles in sorted(molecules):
        atoms, matched = alert_atoms(smiles, alerts)
        if not atoms:
            continue
        f_model = alert_fraction(model.get(smiles), atoms)
        f_base = alert_fraction(baseline.get(smiles), atoms)
        if f_model is None or f_base is None:
            continue
        values.append(f_model - f_base)
        names.append(matched)
        kept.append(smiles)
    return np.asarray(values), names, kept


def report(label, values, expect=None):
    tail = f"   (paper {expect})" if expect else ""
    print(f"  {label:34s} {values.mean():+.3f}   n = {len(values):3d}{tail}")


def paired_test(diff, rng):
    _, p = stats.wilcoxon(diff)
    boot = np.array(
        [rng.choice(diff, diff.size, replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    return p, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/xai"))
    parser.add_argument(
        "--alerts", type=Path, default=Path("toxicophores/alerts_kazius29.csv")
    )
    parser.add_argument(
        "--xtb-dir", type=Path, default=Path("data/xtb_7251"),
        help="GFN2-xTB outputs; supplies the ionisation state of each compound",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-alert-n", type=int, default=20,
                        help="alerts below this many molecules are not broken out")
    args = parser.parse_args()

    rd = args.results_dir
    out_path = args.out or rd / "attribution_shift.csv"
    alerts = load_alerts(args.alerts)
    rng = np.random.default_rng(RNG_SEED)

    ig_gast = load_attributions(rd / "ig_gasteiger.csv")
    ig_gast_base = load_attributions(rd / "ig_baseline.csv")
    ig_xtb = load_attributions(rd / "ig_xtb.csv")
    ig_xtb_base = load_attributions(rd / "ig_baseline_xtb.csv")

    cohort_gast = load_cohort(rd / "flip_cohort.csv")
    cohort_xtb = load_cohort(rd / "flip_cohort_xtb.csv")
    common = cohort_gast & cohort_xtb
    print(f"cohorts: Gasteiger {len(cohort_gast)}, xTB {len(cohort_xtb)}, "
          f"common {len(common)}")

    # --- headline per-descriptor shifts, each on its own cohort ---
    print("\nOwn cohort (not paired -- the two means are different molecule sets):")
    own_gast, _, _ = shifts(cohort_gast, ig_gast, ig_gast_base, alerts)
    own_xtb, _, _ = shifts(cohort_xtb, ig_xtb, ig_xtb_base, alerts)
    report("Gasteiger", own_gast, "0.031")
    report("xTB", own_xtb, "0.057")

    # --- paired comparison on identical molecules ---
    print("\nCommon cohort (paired -- use these for any ratio or test):")
    pair_gast, names, kept = shifts(common, ig_gast, ig_gast_base, alerts)
    pair_xtb, _, _ = shifts(common, ig_xtb, ig_xtb_base, alerts)
    report("Gasteiger", pair_gast)
    report("xTB", pair_xtb)
    diff = pair_xtb - pair_gast
    report("xTB - Gasteiger", diff, "0.024")
    print(f"  {'median difference':34s} {np.median(diff):+.3f}   (paper 0.017)")
    p, lo, hi = paired_test(diff, rng)
    print(f"  {'Wilcoxon signed-rank p':34s} {p:.4f}   (paper 0.004)")
    print(f"  {'bootstrap 95% CI':34s} {lo:+.3f} to {hi:+.3f}   (paper 0.009 to 0.043)")
    print(f"  {'ratio xTB / Gasteiger':34s} {pair_xtb.mean() / pair_gast.mean():.2f}x")

    # --- the advantage is not an artefact of ionisation ---
    # NB the SI calls this difference 0.024 and "unchanged to three decimal
    # places" from the full cohort; it is 0.0245 against 0.0238, so the two
    # agree to two decimal places, not three.
    net_charges = load_net_charges(args.xtb_dir)
    charges = np.array([net_charges.get(s) for s in kept], dtype=object)
    if any(c is None for c in charges):
        missing = sum(c is None for c in charges)
        raise SystemExit(f"{missing} cohort molecules absent from {args.xtb_dir}")
    neutral = charges.astype(int) == 0
    print(f"\nNeutral-only subset ({int(neutral.sum())} neutral, "
          f"{int((~neutral).sum())} ionised):")
    report("xTB - Gasteiger", diff[neutral], "0.024, actually 0.025")
    print(f"  {'median difference':34s} {np.median(diff[neutral]):+.3f}   "
          f"(paper 0.015)")
    p_n, lo_n, hi_n = paired_test(diff[neutral], rng)
    print(f"  {'Wilcoxon signed-rank p':34s} {p_n:.4f}   (paper 0.013)")
    print(f"  {'bootstrap 95% CI':34s} {lo_n:+.3f} to {hi_n:+.3f}   "
          f"(paper 0.006 to 0.043)")
    _, p_split = stats.mannwhitneyu(diff[neutral], diff[~neutral])
    print(f"  {'ionised vs neutral (Mann-Whitney)':34s} {p_split:.2f}   (paper 0.94)")

    # --- per-alert breakdown (Supporting Information Table S2) ---
    rows = []
    for alert in sorted({n for matched in names for n in matched}):
        mask = np.array([alert in matched for matched in names])
        if mask.sum() < args.min_alert_n:
            continue
        rows.append(
            {
                "alert": alert,
                "n": int(mask.sum()),
                "gasteiger": round(float(pair_gast[mask].mean()), 3),
                "xtb": round(float(pair_xtb[mask].mean()), 3),
                "xtb_minus_gasteiger": round(float(diff[mask].mean()), 3),
            }
        )
    table = pd.DataFrame(rows).sort_values("xtb_minus_gasteiger", ascending=False)
    print(f"\nPer-alert breakdown (>= {args.min_alert_n} molecules):")
    print(table.to_string(index=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
