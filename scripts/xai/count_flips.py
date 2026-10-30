"""Step 1: count the flipped cohort.

Scores every fold's test split with two checkpoints and reports molecules where
the baseline is wrong and the descriptor model is right, restricted to
alert-bearing true non-mutagens -- the population the attribution comparison
will be computed over.

Reuses the loader and model-loading path in attribution_base.py, so the
probabilities match what the attribution scripts report as
positive_probability.

Usage
-----
    cd physchem-xai && source .venv/bin/activate && export PYTHONPATH=src

    python count_flips.py \
        --runs-dir runs_7251 \
        --baseline-config baseline \
        --descriptor-config gasteiger \
        --descriptor-token-features charge \
        --alerts toxicophores/alerts_kazius29.csv \
        --out flip_cohort.csv

If gasteiger needs --charge-xyz-dir, pass it; it is forwarded only to the
descriptor model, never to the baseline.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch

from downstream.toxicity.attribution_base import (
    AttributionConfig,
    create_loader,
    load_split_dataframe,
)
from downstream.toxicity.xai.common import resolve_device

from fusion_loader import load_fusion_model

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Must match training. Override with --pretrained-model-name if different.
PRETRAINED_MODEL_NAME = "fabikru/MolEncoder"
MAX_SEQUENCE_LENGTH = 128


def score_split(
    checkpoint: Path,
    fold_dir: Path,
    token_features: tuple[str, ...],
    molecule_features: tuple[str, ...],
    charge_xyz_dir: Path | None,
    split_name: str,
    num_workers: int,
    weight_mode: str,
    target_class: int,
    token_feature_normalize_mode: str = "per_molecule",
) -> pd.DataFrame:
    """Probability of the positive class for every molecule in the split."""
    from downstream.toxicity.features import molecule_feature_dimension

    frame = load_split_dataframe(fold_dir, split_name)

    lightning_model = load_fusion_model(
        checkpoint_path=checkpoint,
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        model_dir=None,
        weight_mode=weight_mode,
        expected_token_dim=len(token_features),
        expected_molecule_dim=molecule_feature_dimension(molecule_features),
    )
    device = resolve_device()
    lightning_model.to(device)
    model = lightning_model.model

    loader = create_loader(
        encoder=model.encoder,
        frame=frame,
        batch_size=1,
        num_workers=num_workers,
        token_feature_names=token_features,
        molecule_feature_names=molecule_features,
        charge_xyz_dir=charge_xyz_dir,
        token_feature_normalize_mode=token_feature_normalize_mode,
    )

    rows = []
    idx = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tf = batch.get("token_features")
        mf = batch.get("molecule_features")
        if tf is not None:
            tf = tf.to(device)
        if mf is not None:
            mf = mf.to(device)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_features=tf,
                molecule_features=mf,
            )
            prob = float(torch.softmax(logits, dim=-1)[0, target_class].item())

        row = frame.iloc[idx]
        rows.append(
            {
                "smiles": str(row["smiles"]),
                "label": int(batch["labels"][0].item()),
                "prob": prob,
            }
        )
        idx += 1

    return pd.DataFrame(rows)


def load_alerts(path: Path) -> list[tuple[str, str, Chem.Mol]]:
    out = []
    with open(path, encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            pattern = Chem.MolFromSmarts(record["smarts"])
            if pattern is None:
                raise ValueError(f"SMARTS failed to parse: {record['alert_id']}")
            out.append((record["alert_id"], record["name"], pattern))
    return out


def alert_hits(smiles: str, alerts) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [name for _, name, pat in alerts if mol.GetSubstructMatches(pat)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_7251"))
    parser.add_argument("--baseline-config", default="baseline")
    parser.add_argument("--descriptor-config", default="gasteiger")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--checkpoint-name", default="toxicity_classifier.ckpt")
    parser.add_argument("--baseline-token-features", nargs="*", default=[])
    parser.add_argument("--baseline-molecule-features", nargs="*", default=[])
    parser.add_argument("--descriptor-token-features", nargs="*", default=["charge"])
    parser.add_argument("--descriptor-molecule-features", nargs="*", default=[])
    parser.add_argument("--charge-xyz-dir", type=Path, default=None)
    parser.add_argument("--baseline-token-feature-normalize-mode",
                        default="per_molecule", choices=("per_molecule", "none", "dataset"),
                        help="must match how the baseline model was trained")
    parser.add_argument("--descriptor-token-feature-normalize-mode",
                        default="per_molecule", choices=("per_molecule", "none", "dataset"),
                        help="must match how the descriptor model was trained")
    parser.add_argument("--alerts", type=Path, default=Path("toxicophores/alerts_kazius29.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weight-mode", default="abs", choices=("abs", "positive"))
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--pretrained-model-name", default=PRETRAINED_MODEL_NAME)
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--out", type=Path, default=Path("flip_cohort.csv"))
    args = parser.parse_args()

    globals()["PRETRAINED_MODEL_NAME"] = args.pretrained_model_name
    globals()["MAX_SEQUENCE_LENGTH"] = args.max_sequence_length

    alerts = load_alerts(args.alerts)
    print(f"loaded {len(alerts)} alerts from {args.alerts}\n")

    frames = []
    for fold in args.folds:
        base_dir = args.runs_dir / args.baseline_config / f"fold_{fold}"
        desc_dir = args.runs_dir / args.descriptor_config / f"fold_{fold}"
        for directory in (base_dir, desc_dir):
            if not (directory / args.checkpoint_name).exists():
                raise FileNotFoundError(directory / args.checkpoint_name)

        print(f"fold {fold}: scoring {args.baseline_config} ...", flush=True)
        base_scores = score_split(
            base_dir / args.checkpoint_name,
            base_dir,
            tuple(args.baseline_token_features),
            tuple(args.baseline_molecule_features),
            None,
            args.split_name,
            args.num_workers,
            args.weight_mode,
            args.target_class,
            args.baseline_token_feature_normalize_mode,
        )

        print(f"fold {fold}: scoring {args.descriptor_config} ...", flush=True)
        desc_scores = score_split(
            desc_dir / args.checkpoint_name,
            desc_dir,
            tuple(args.descriptor_token_features),
            tuple(args.descriptor_molecule_features),
            args.charge_xyz_dir,
            args.split_name,
            args.num_workers,
            args.weight_mode,
            args.target_class,
            args.descriptor_token_feature_normalize_mode,
        )

        # The two configs share a split (same seed, same n_splits), so rows
        # correspond. Verify rather than assume.
        if len(base_scores) != len(desc_scores):
            raise RuntimeError(f"fold {fold}: split lengths differ")
        if not (base_scores["smiles"] == desc_scores["smiles"]).all():
            raise RuntimeError(f"fold {fold}: split molecule order differs between configs")
        if not (base_scores["label"] == desc_scores["label"]).all():
            raise RuntimeError(f"fold {fold}: labels differ between configs")

        merged = base_scores.rename(columns={"prob": "prob_baseline"}).copy()
        merged["prob_descriptor"] = desc_scores["prob"].values
        merged["fold"] = fold
        frames.append(merged)

    data = pd.concat(frames, ignore_index=True)

    data["pred_baseline"] = (data["prob_baseline"] >= args.threshold).astype(int)
    data["pred_descriptor"] = (data["prob_descriptor"] >= args.threshold).astype(int)
    data["alert_names"] = [ "|".join(alert_hits(s, alerts)) for s in data["smiles"] ]
    data["alert_positive"] = data["alert_names"].str.len() > 0

    base_ok = data["pred_baseline"] == data["label"]
    desc_ok = data["pred_descriptor"] == data["label"]

    data["stratum"] = "both_wrong"
    data.loc[base_ok & desc_ok, "stratum"] = "both_correct"
    data.loc[~base_ok & desc_ok, "stratum"] = "flip_to_correct"
    data.loc[base_ok & ~desc_ok, "stratum"] = "flip_to_wrong"

    data.to_csv(args.out, index=False)

    n = len(data)
    print(f"\n{'='*64}\nscored {n} molecules across folds {args.folds}")
    print(f"  baseline   accuracy {base_ok.mean():.4f}")
    print(f"  descriptor accuracy {desc_ok.mean():.4f}")

    print("\nstrata (all molecules):")
    for name, count in data["stratum"].value_counts().items():
        print(f"  {name:18s} {count:5d}")

    print("\n" + "=" * 64)
    print("THE COHORT: baseline wrong, descriptor right, alert-bearing")
    print("=" * 64)
    flip = data[(data["stratum"] == "flip_to_correct") & data["alert_positive"]]
    fp = flip[flip["label"] == 0]   # false positive corrected -- the figures
    fn = flip[flip["label"] == 1]   # false negative corrected

    print(f"  true non-mutagens (baseline said toxic, wrongly) : {len(fp):5d}   <-- PRIMARY")
    print(f"  true mutagens     (baseline said safe, wrongly)  : {len(fn):5d}")
    print(f"  control cohort: both correct, alert-bearing      : "
          f"{len(data[(data['stratum']=='both_correct') & data['alert_positive']]):5d}")

    print(f"\n  reverse direction (descriptor broke it): "
          f"{len(data[(data['stratum']=='flip_to_wrong') & data['alert_positive']]):5d}")

    if len(fp) >= 50:
        verdict = "one-sample t-test on delta, with the both-correct control"
    elif len(fp) >= 15:
        verdict = "report the effect with n stated plainly; a t-test will be fragile"
    else:
        verdict = f"too small for statistics -- write 'we examined all {len(fp)} such cases'"
    print(f"\n  => {verdict}")

    if len(fp):
        print("\n  most frequent alerts in the primary cohort:")
        from collections import Counter
        counts = Counter(a for names in fp["alert_names"] for a in names.split("|") if a)
        for name, count in counts.most_common(8):
            print(f"    {count:4d}  {name}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
