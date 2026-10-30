"""External validation: Ames-trained models applied to non-overlapping pesticide compounds.

Answers the reviewer question: do descriptor-augmented models transfer better than
the SMILES-only baseline to chemistry they have never seen?

The 11,591 pesticide compounds used here are those whose RDKit canonical SMILES do
NOT appear anywhere in ames_7251.csv, so no compound was in any Ames training,
validation or test fold. All five Ames fold-models are therefore valid for every
compound, and their probabilities are averaged.

IMPORTANT: do NOT set PHYSCHEM_STATS. It defaults to "ames" (features.py:666),
which is what Ames-trained models require. run_pesticides.sh sets it to "pesticide";
copying that here would silently feed the models wrongly scaled descriptors.

Usage
-----
    cd <repo root> && source .venv/bin/activate && export PYTHONPATH=src
    python scripts/external_validation.py \
        --runs-dir runs_7251 \
        --external-csv data/pesticides_external.csv \
        --out results/external

Expects checkpoints at <runs-dir>/<config>/fold_<n>/toxicity_classifier.ckpt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "xai"))

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
)

from downstream.toxicity.attribution_base import create_loader
from downstream.toxicity.features import molecule_feature_dimension
from fusion_loader import load_fusion_model

RDLogger.DisableLog("rdApp.*")

PRETRAINED_MODEL_NAME = "fabikru/MolEncoder"
MAX_SEQUENCE_LENGTH = 128

PHYSCHEM = (
    "physchem_molwt", "physchem_tpsa", "physchem_logp", "physchem_molmr",
    "physchem_hbd", "physchem_hba", "physchem_rotatablebonds",
    "physchem_ringcount", "physchem_heavyatomcount", "physchem_fractioncsp3",
)
NB = ("neighbor_k2", "neighbor_k3", "neighbor_k4")

# (token features, token_feature_normalize_mode) — must match how each model was
# trained. Taken verbatim from scripts/run_table_7251.sh and run_pesticides.sh.
CONFIGS: dict[str, tuple[tuple[str, ...], str]] = {
    "baseline":     ((),                  "none"),
    "gasteiger":    (("charge",),         "none"),
    "neighbors":    (NB,                  "none"),
    "gast_neigh":   (("charge",) + NB,    "dataset"),
    "physchem_tok": (PHYSCHEM,            "dataset"),
}


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def score_frame(checkpoint: Path, frame: pd.DataFrame, token_features: tuple[str, ...],
                normalize_mode: str, batch_size: int, num_workers: int,
                device: torch.device) -> np.ndarray:
    """Probability of the positive class for every row of `frame`, in row order."""
    lightning_model = load_fusion_model(
        checkpoint_path=checkpoint,
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        model_dir=None,
        weight_mode="abs",
        expected_token_dim=len(token_features),
        expected_molecule_dim=molecule_feature_dimension(()),
    )
    lightning_model.to(device).eval()
    model = lightning_model.model

    loader = create_loader(
        encoder=model.encoder,
        frame=frame,
        batch_size=batch_size,
        num_workers=num_workers,
        token_feature_names=token_features,
        molecule_feature_names=(),
        charge_xyz_dir=None,
        token_feature_normalize_mode=normalize_mode,
    )

    out: list[np.ndarray] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            tf = batch.get("token_features")
            mf = batch.get("molecule_features")
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_features=tf.to(device) if tf is not None else None,
                molecule_features=mf.to(device) if mf is not None else None,
            )
            out.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
            if (i + 1) % 25 == 0:
                print(f"      batch {i + 1}/{len(loader)}", flush=True)

    del lightning_model, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


def metrics(label: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(label, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(label, prob),
        "balanced_accuracy": balanced_accuracy_score(label, pred),
        "sensitivity": recall_score(label, pred, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "precision": precision_score(label, pred, zero_division=0),
        "f1": f1_score(label, pred, zero_division=0),
        "accuracy": accuracy_score(label, pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("runs_7251"))
    ap.add_argument("--external-csv", type=Path, default=Path("data/pesticides_external.csv"))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--checkpoint-name", default="toxicity_classifier.ckpt")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N compounds (smoke test; 0 = all)")
    ap.add_argument("--out", type=Path, default=Path("results/external"))
    args = ap.parse_args()

    import os
    if "PHYSCHEM_STATS" in os.environ and os.environ["PHYSCHEM_STATS"] != "ames":
        raise SystemExit(
            f"PHYSCHEM_STATS={os.environ['PHYSCHEM_STATS']!r} is set. Ames-trained models "
            "need the 'ames' statistics. Unset it (`unset PHYSCHEM_STATS`) and re-run."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    frame = pd.read_csv(args.external_csv)[["smiles", "label"]].reset_index(drop=True)
    if args.limit:
        # keep both classes present so the metrics still compute
        pos = frame[frame.label == 1].head(max(args.limit // 2, 1))
        neg = frame[frame.label == 0].head(args.limit - len(pos))
        frame = pd.concat([pos, neg]).reset_index(drop=True)
        print(f"*** SMOKE TEST: {len(frame)} compounds only, metrics are meaningless ***\n")
    label = frame["label"].to_numpy()
    print(f"external set: {len(frame)} compounds "
          f"({label.sum()} positive, {(1 - label).sum()} negative) on {device}\n", flush=True)

    predictions = frame.copy()
    rows = []

    for config in args.configs:
        if config not in CONFIGS:
            raise SystemExit(f"unknown config {config!r}; known: {list(CONFIGS)}")
        token_features, normalize_mode = CONFIGS[config]
        print(f"=== {config}  (features={token_features or 'none'}, "
              f"normalize={normalize_mode}) ===", flush=True)

        fold_probs = []
        for fold in args.folds:
            ckpt = args.runs_dir / config / f"fold_{fold}" / args.checkpoint_name
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
            print(f"   fold {fold} ...", flush=True)
            p = score_frame(ckpt, frame, token_features, normalize_mode,
                            args.batch_size, args.num_workers, device)
            fold_probs.append(p)
            predictions[f"{config}_fold{fold}"] = p

        mean_prob = np.mean(fold_probs, axis=0)
        predictions[f"{config}_mean"] = mean_prob

        m = metrics(label, mean_prob)
        m["config"] = config
        m["roc_auc_fold_mean"] = float(np.mean([roc_auc_score(label, p) for p in fold_probs]))
        m["roc_auc_fold_sd"] = float(np.std([roc_auc_score(label, p) for p in fold_probs], ddof=1))
        rows.append(m)
        print(f"   -> ROC AUC {m['roc_auc']:.4f} (ensemble), "
              f"{m['roc_auc_fold_mean']:.4f} ± {m['roc_auc_fold_sd']:.4f} (per fold), "
              f"balanced accuracy {m['balanced_accuracy']:.4f}\n", flush=True)

    summary = pd.DataFrame(rows)[[
        "config", "roc_auc", "roc_auc_fold_mean", "roc_auc_fold_sd", "balanced_accuracy",
        "sensitivity", "specificity", "precision", "f1", "accuracy", "tn", "fp", "fn", "tp",
    ]].sort_values("roc_auc", ascending=False)

    summary.to_csv(args.out / "external_summary.csv", index=False)
    predictions.to_csv(args.out / "external_predictions.csv", index=False)

    print("=" * 78)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 78)

    if "baseline" in summary.config.values:
        base = float(summary.loc[summary.config == "baseline", "roc_auc"].iloc[0])
        print("\nDelta vs SMILES-only baseline (ensemble ROC AUC):")
        for _, r in summary.iterrows():
            if r.config == "baseline":
                continue
            print(f"  {r.config:<14} {r.roc_auc - base:+.4f}")

    print(f"\nwrote {args.out / 'external_summary.csv'} and "
          f"{args.out / 'external_predictions.csv'}")


if __name__ == "__main__":
    main()
