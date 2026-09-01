"""
Structural Alert Explainability Analysis
=========================================
Compares the MolEncoder baseline vs. the RDKit-physchem-augmented model on:

  1. False-positive rates on structurally-alerted but actually non-toxic compounds
     (systematic scaffold over-reliance test)

  2. Attribution concentration metric: what fraction of the occlusion signal sits
     on known alert atoms?  A well-explaining model should NOT concentrate
     attribution on the alert when the molecule is non-toxic.

Usage (from the project root, with the xai_dft conda env active):
  python scripts/structural_alert_analysis.py

Outputs (written to runs_cv/explainability_analysis/):
  per_compound_predictions.csv   — one row per compound per fold, both models
  alert_subset_fp_rates.csv      — FP rates per alert type, both models
  alert_subset_summary.csv       — overall FP rate comparison
  attribution_concentration.csv  — per-compound alert-atom fraction (where occlusion data exists)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ── project paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "ames_7251.csv"
CV_DIR = ROOT / "runs_cv" / "cv_original_ames"
MODEL_CACHE = ROOT / "models" / "molencoder"
OUTPUT_DIR = ROOT / "runs_cv" / "explainability_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

import json
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# ── model imports ──────────────────────────────────────────────────────────────
from molencoder.config import MolEncoderConfig
from molencoder.model import MolEncoderWrapper
from downstream.toxicity.train import ToxicityLightningModule, ToxicityClassifier, create_dataloaders
from downstream.toxicity.features import molecule_feature_dimension
from downstream.toxicity.dataset import ToxicityDataset, ToxicityCollator


# ══════════════════════════════════════════════════════════════════════════════
# Structural alert SMARTS  (standard Ames / genotoxicity alerts)
# ══════════════════════════════════════════════════════════════════════════════
STRUCTURAL_ALERTS: dict[str, str] = {
    "primary_aromatic_amine":   "[NH2]c",
    "secondary_aromatic_amine":  "[NH1](c)c",
    "aromatic_nitro":           "[$([N+](=O)[O-]),$([n+](=O)[O-])]c",
    "nitroso":                  "N=O",
    "epoxide":                  "C1OC1",
    "alkyl_halide":             "[CX4][F,Cl,Br,I]",
    "acrylate_michael":         "[CX3]=[CX3][CX3](=O)",
    "hydrazine":                "[NH]-[NH]",
    "hydroxylamine":            "[NH][OH]",
    "aliphatic_aldehyde":       "[CH1](=O)[#6]",
    "quinone":                  "O=C1C=CC(=O)C=C1",
    "azide":                    "[N-]=[N+]=[N-]",
    "acyl_halide":              "[CX3](=O)[F,Cl,Br,I]",
    "beta_lactam":              "C1(=O)CCN1",
    "furan":                    "c1ccoc1",
    "thiophene_reactive":       "[SX2]([#6])[#6]",
}

def compile_alerts() -> dict[str, object]:
    compiled = {}
    for name, smarts in STRUCTURAL_ALERTS.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            print(f"WARNING: could not compile SMARTS for '{name}': {smarts}")
        else:
            compiled[name] = pat
    return compiled

ALERT_PATTERNS = compile_alerts()


def get_alerts(smiles: str) -> list[str]:
    """Return list of alert names matched by this SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [name for name, pat in ALERT_PATTERNS.items() if mol.HasSubstructMatch(pat)]


def get_alert_atom_indices(smiles: str) -> set[int]:
    """Return the union of atom indices covered by any structural alert."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    indices: set[int] = set()
    for pat in ALERT_PATTERNS.values():
        for match in mol.GetSubstructMatches(pat):
            indices.update(match)
    return indices


# ══════════════════════════════════════════════════════════════════════════════
# Model loading helpers
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIGS = {
    "baseline": {
        "token_feature_names": (),
        "molecule_feature_names": (),
        "run_subdir": "baseline_smiles",
    },
    "physchem": {
        "token_feature_names": (),
        "molecule_feature_names": ("physchem",),
        "run_subdir": "smiles_rdkit_global_physchem",
    },
}


def load_encoder() -> MolEncoderWrapper:
    cfg = MolEncoderConfig(
        pretrained_model_name="fabikru/MolEncoder",
        max_sequence_length=256,
        num_labels=2,
        model_cache_dir=MODEL_CACHE,
    )
    return MolEncoderWrapper(cfg)


def load_model(
    encoder: MolEncoderWrapper,
    ckpt_path: Path,
    token_feature_names: tuple[str, ...],
    molecule_feature_names: tuple[str, ...],
) -> ToxicityClassifier:
    classifier = ToxicityClassifier(
        encoder=encoder,
        num_labels=2,
        token_feature_dim=len(token_feature_names),
        molecule_feature_dim=molecule_feature_dimension(molecule_feature_names),
    )
    lightning_model = ToxicityLightningModule.load_from_checkpoint(
        str(ckpt_path),
        model=classifier,
        learning_rate=2e-5,
        weight_decay=1e-2,
    )
    lightning_model.eval()
    return lightning_model.model


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(
    model: ToxicityClassifier,
    test_frame: pd.DataFrame,
    encoder: MolEncoderWrapper,
    token_feature_names: tuple[str, ...],
    molecule_feature_names: tuple[str, ...],
) -> tuple[list[float], list[int]]:
    """Return (positive_scores, predictions) aligned to test_frame rows."""
    collator = ToxicityCollator(
        encoder,
        feature_names=list(token_feature_names),
        molecule_feature_names=list(molecule_feature_names),
        charge_xyz_dir=None,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ToxicityDataset(test_frame),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    positive_scores: list[float] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_features=batch.get("token_features"),
                molecule_features=batch.get("molecule_features"),
            )
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            positive_scores.extend(probs[:, 1].tolist())
            predictions.extend(preds.tolist())
    return positive_scores, predictions


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1: per-compound predictions across all folds
# ══════════════════════════════════════════════════════════════════════════════
def collect_predictions() -> pd.DataFrame:
    print("\n=== Collecting per-compound predictions ===")
    encoder = load_encoder()
    rows: list[dict] = []

    for fold_idx in range(1, 6):
        print(f"  Fold {fold_idx}...")
        # Load test split from baseline (same split for all models)
        test_csv = CV_DIR / "baseline_smiles" / f"fold_{fold_idx}" / "splits" / "test.csv"
        test_frame = pd.read_csv(test_csv)

        # Annotate structural alerts
        test_frame["alerts"] = test_frame["smiles"].apply(
            lambda s: ";".join(get_alerts(s)) if get_alerts(s) else ""
        )
        test_frame["has_alert"] = test_frame["alerts"].str.len() > 0

        fold_row_base: dict = {
            "fold": fold_idx,
        }

        # Collect predictions from each model config
        model_scores: dict[str, list] = {}
        model_preds: dict[str, list] = {}

        for model_name, cfg in MODEL_CONFIGS.items():
            ckpt_path = CV_DIR / cfg["run_subdir"] / f"fold_{fold_idx}" / "toxicity_classifier.ckpt"
            if not ckpt_path.exists():
                print(f"    WARNING: checkpoint not found: {ckpt_path}")
                model_scores[model_name] = [float("nan")] * len(test_frame)
                model_preds[model_name] = [-1] * len(test_frame)
                continue

            model = load_model(
                encoder=encoder,
                ckpt_path=ckpt_path,
                token_feature_names=cfg["token_feature_names"],
                molecule_feature_names=cfg["molecule_feature_names"],
            )
            scores, preds = run_inference(
                model=model,
                test_frame=test_frame,
                encoder=encoder,
                token_feature_names=cfg["token_feature_names"],
                molecule_feature_names=cfg["molecule_feature_names"],
            )
            model_scores[model_name] = scores
            model_preds[model_name] = preds

        for i, (_, row) in enumerate(test_frame.iterrows()):
            entry = {
                "fold": fold_idx,
                "smiles": row["smiles"],
                "true_label": int(row["label"]),
                "alerts": row["alerts"],
                "has_alert": bool(row["has_alert"]),
            }
            for model_name in MODEL_CONFIGS:
                entry[f"{model_name}_score"] = model_scores[model_name][i]
                entry[f"{model_name}_pred"] = model_preds[model_name][i]
            rows.append(entry)

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "per_compound_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1b: FP rate comparison on alert-containing non-toxics
# ══════════════════════════════════════════════════════════════════════════════
def compute_fp_rates(df: pd.DataFrame) -> None:
    print("\n=== False-positive rate analysis ===")

    # Overall on all non-toxics
    non_toxic = df[df["true_label"] == 0]
    alerted_nontoxic = non_toxic[non_toxic["has_alert"]]
    clean_nontoxic = non_toxic[~non_toxic["has_alert"]]

    summary_rows = []
    for subset_name, subset in [
        ("all_non_toxic", non_toxic),
        ("alerted_non_toxic", alerted_nontoxic),
        ("clean_non_toxic", clean_nontoxic),
    ]:
        n = len(subset)
        if n == 0:
            continue
        row: dict = {"subset": subset_name, "n": n}
        for model_name in MODEL_CONFIGS:
            fp = (subset[f"{model_name}_pred"] == 1).sum()
            row[f"{model_name}_fp"] = int(fp)
            row[f"{model_name}_fpr"] = round(fp / n, 4)
        summary_rows.append(row)

    # Per-alert-type breakdown
    alert_rows = []
    for alert_name in STRUCTURAL_ALERTS:
        subset = non_toxic[non_toxic["alerts"].str.contains(alert_name, na=False)]
        n = len(subset)
        if n < 3:
            continue
        row = {"alert_type": alert_name, "n_nontoxic_with_alert": n}
        for model_name in MODEL_CONFIGS:
            fp = (subset[f"{model_name}_pred"] == 1).sum()
            row[f"{model_name}_fp"] = int(fp)
            row[f"{model_name}_fpr"] = round(fp / n, 4)
        alert_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    alert_df = pd.DataFrame(alert_rows)

    summary_df.to_csv(OUTPUT_DIR / "alert_subset_summary.csv", index=False)
    alert_df.to_csv(OUTPUT_DIR / "alert_subset_fp_rates.csv", index=False)

    print("\n  Overall false-positive rates:")
    print(summary_df.to_string(index=False))
    print("\n  Per alert-type false-positive rates:")
    print(alert_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2: Attribution concentration on alert atoms (occlusion maps)
# ══════════════════════════════════════════════════════════════════════════════
def load_occlusion_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def attribution_concentration_from_occlusion(
    occlusion_df: pd.DataFrame,
    smiles: str,
) -> float | None:
    """
    Compute what fraction of total occlusion signal sits on structural-alert atoms.
    Occlusion CSV is expected to have columns: token_index, occlusion_score (or similar).
    Returns None if alert atoms cannot be determined or data is missing.
    """
    alert_indices = get_alert_atom_indices(smiles)
    if not alert_indices:
        return None

    score_col = None
    for col in ["occlusion_score", "score", "delta_prob", "importance"]:
        if col in occlusion_df.columns:
            score_col = col
            break
    if score_col is None:
        return None

    idx_col = None
    for col in ["token_index", "atom_index", "position"]:
        if col in occlusion_df.columns:
            idx_col = col
            break
    if idx_col is None:
        return None

    total_abs = occlusion_df[score_col].abs().sum()
    if total_abs < 1e-9:
        return None

    alert_mask = occlusion_df[idx_col].isin(alert_indices)
    alert_abs = occlusion_df.loc[alert_mask, score_col].abs().sum()
    return float(alert_abs / total_abs)


def collect_attribution_concentrations(df: pd.DataFrame) -> None:
    print("\n=== Attribution concentration metric ===")

    # Only look at non-toxic compounds that have structural alerts and were
    # misclassified by baseline but correctly classified by physchem
    target = df[
        (df["true_label"] == 0) &
        (df["has_alert"]) &
        (df["baseline_pred"] == 1) &
        (df["physchem_pred"] == 0)
    ].copy()

    print(f"  Compounds where baseline=FP, physchem=TN, has alert: {len(target)}")

    concentration_rows = []
    for _, row in target.iterrows():
        fold = int(row["fold"])
        smiles = row["smiles"]

        # Look for occlusion files for both models in this fold
        for model_name, cfg in MODEL_CONFIGS.items():
            occlusion_dir = CV_DIR / cfg["run_subdir"] / f"fold_{fold}"
            # Occlusion files may be in subdirectories named by row index
            occ_files = list(occlusion_dir.glob("**/test_occlusion_summary.csv"))
            occ_files += list(occlusion_dir.glob("**/occlusion_scores.csv"))
            occ_files += list(occlusion_dir.glob("**/*occlusion*.csv"))

            for occ_file in occ_files:
                occ_df = load_occlusion_csv(occ_file)
                if occ_df is None:
                    continue
                # Check if SMILES column matches
                if "smiles" in occ_df.columns:
                    occ_df = occ_df[occ_df["smiles"] == smiles]
                if occ_df.empty:
                    continue
                conc = attribution_concentration_from_occlusion(occ_df, smiles)
                if conc is not None:
                    concentration_rows.append({
                        "fold": fold,
                        "smiles": smiles,
                        "true_label": int(row["true_label"]),
                        "alerts": row["alerts"],
                        "model": model_name,
                        "alert_attribution_fraction": conc,
                        "baseline_pred": int(row["baseline_pred"]),
                        "physchem_pred": int(row["physchem_pred"]),
                    })

    if concentration_rows:
        conc_df = pd.DataFrame(concentration_rows)
        conc_df.to_csv(OUTPUT_DIR / "attribution_concentration.csv", index=False)
        print(f"  Saved attribution_concentration.csv ({len(conc_df)} rows)")

        # Summary: mean concentration on alert atoms per model
        summary = conc_df.groupby("model")["alert_attribution_fraction"].agg(["mean", "std", "count"])
        print("\n  Mean alert-atom attribution fraction:")
        print(summary.to_string())
    else:
        print("  No occlusion data found for the key compounds.")
        print("  → To generate occlusion maps, run compute_occlusion.py for both models.")
        print("  → Once available, re-run this script to populate attribution_concentration.csv")

        # Still save the target compounds so the user knows which ones to prioritize
        target[["fold", "smiles", "true_label", "alerts", "baseline_score", "physchem_score"]].to_csv(
            OUTPUT_DIR / "priority_compounds_for_occlusion.csv", index=False
        )
        print(f"  Saved priority_compounds_for_occlusion.csv — {len(target)} compounds to compute occlusion for")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"Output directory: {OUTPUT_DIR}")

    # Step 1: collect predictions
    pred_csv = OUTPUT_DIR / "per_compound_predictions.csv"
    if pred_csv.exists():
        print(f"\nLoading existing predictions from {pred_csv}")
        df = pd.read_csv(pred_csv)
    else:
        df = collect_predictions()

    # Step 2: FP rate analysis
    compute_fp_rates(df)

    # Step 3: attribution concentration
    collect_attribution_concentrations(df)

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
