"""Exact decomposition of the descriptor pathway's contribution.

SmilesAnchorResidualAuxClassifier.forward ends with

    return base_logits + gate * aux_delta

so the prediction is a SMILES-only logit plus a gated descriptor correction.
That makes the model additively separable at the output: `gate * aux_delta` IS
the descriptor pathway's contribution to this molecule's prediction, in logits.

No attribution method, no baseline choice, no perturbation, and therefore no
molecules lost -- it runs on all 7,251.

    cd physchem-xai && source .venv/bin/activate && export PYTHONPATH=src
    python scripts/xai/gate_decomposition.py \
        --descriptor-config gasteiger --descriptor-token-features charge \
        --out results/xai/gate_decomposition.csv

Question it answers
-------------------
On molecules where the descriptor model corrects a false positive, does the
descriptor branch actively pull the prediction away from "toxic"? Is it quiet
on molecules both models already get right?

If `gate * aux_delta` is near zero on the corrected molecules, the descriptor
pathway is not driving the correction and no attribution method will show that
it is. Worth knowing before investing in integrated gradients.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from downstream.toxicity.attribution_base import create_loader, load_split_dataframe
from downstream.toxicity.features import molecule_feature_dimension
from downstream.toxicity.xai.common import resolve_device

from fusion_loader import load_fusion_model

PRETRAINED_MODEL_NAME = "fabikru/MolEncoder"
MAX_SEQUENCE_LENGTH = 128


def decompose(model, batch, device, target_class: int) -> dict:
    """Re-run forward, keeping the intermediates instead of only the logits.

    Mirrors SmilesAnchorResidualAuxClassifier.forward exactly. If that method
    changes, this must change with it.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_features = batch.get("token_features")
    molecule_features = batch.get("molecule_features")
    if token_features is not None:
        token_features = token_features.to(device)
    if molecule_features is not None:
        molecule_features = molecule_features.to(device)

    with torch.no_grad():
        token_states = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = token_states[:, 0, :]
        base_repr = model.base_proj(cls_embedding)
        base_logits = model.base_classifier(base_repr)

        aux_parts = []
        if (token_features is not None and token_features.shape[-1] > 0
                and model.token_proj is not None):
            feature_weights, feature_stats = model.summarize_token_features(
                token_features, attention_mask
            )
            pooled = (token_states * feature_weights.to(token_states.dtype)).sum(dim=1)
            pooled = model.charge_context_scale * pooled
            feature_stats = model.feature_stats_scale * feature_stats.to(cls_embedding.dtype)
            aux_parts.append(model.token_proj(torch.cat([pooled, feature_stats], dim=-1)))

        if (molecule_features is not None and molecule_features.shape[-1] > 0
                and model.molecule_proj is not None):
            aux_parts.append(model.molecule_proj(molecule_features.to(cls_embedding.dtype)))

        if not aux_parts:
            # baseline: no descriptor branch at all
            probs = torch.softmax(base_logits, dim=-1)
            return {
                "base_logit_diff": float((base_logits[0, 1] - base_logits[0, 0]).item()),
                "gate": np.nan,
                "aux_delta_diff": 0.0,
                "descriptor_contribution": 0.0,
                "prob": float(probs[0, target_class].item()),
                "base_only_prob": float(probs[0, target_class].item()),
            }

        aux_repr = torch.stack(aux_parts, dim=0).mean(dim=0)
        aux_repr = model.aux_proj(aux_repr)
        aux_delta = model.aux_classifier(aux_repr)

        base_probs = torch.softmax(base_logits, dim=-1)
        confidence = base_probs.max(dim=-1, keepdim=True).values
        margin = (base_probs[:, 1:2] - base_probs[:, 0:1]).abs()
        gate = torch.sigmoid(model.gate_mlp(torch.cat([base_repr, confidence, margin], dim=-1)))

        final_logits = base_logits + gate * aux_delta

        # sanity: the reconstruction must match the model's own forward
        check = model(input_ids=input_ids, attention_mask=attention_mask,
                      token_features=token_features, molecule_features=molecule_features)
        if not torch.allclose(final_logits, check, atol=1e-4):
            raise RuntimeError(
                "decomposition does not reproduce forward() -- the architecture "
                "has changed and this script must be updated"
            )

        contribution = (gate * aux_delta)
        return {
            "base_logit_diff": float((base_logits[0, 1] - base_logits[0, 0]).item()),
            "gate": float(gate[0, 0].item()),
            "aux_delta_diff": float((aux_delta[0, 1] - aux_delta[0, 0]).item()),
            "descriptor_contribution": float((contribution[0, 1] - contribution[0, 0]).item()),
            "prob": float(torch.softmax(final_logits, dim=-1)[0, target_class].item()),
            "base_only_prob": float(base_probs[0, target_class].item()),
        }


def run_fold(runs: Path, config: str, fold: int, token_features, molecule_features,
             charge_xyz_dir, split_name, weight_mode, target_class, num_workers):
    fold_dir = runs / config / f"fold_{fold}"
    checkpoint = fold_dir / "toxicity_classifier.ckpt"
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
        encoder=model.encoder, frame=frame, batch_size=1, num_workers=num_workers,
        token_feature_names=token_features, molecule_feature_names=molecule_features,
        charge_xyz_dir=charge_xyz_dir,
    )

    rows = []
    for i, batch in enumerate(loader):
        out = decompose(model, batch, device, target_class)
        out["smiles"] = str(frame.iloc[i]["smiles"])
        out["label"] = int(batch["labels"][0].item())
        out["fold"] = fold
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_7251"))
    parser.add_argument("--descriptor-config", default="gasteiger")
    parser.add_argument("--descriptor-token-features", nargs="*", default=["charge"])
    parser.add_argument("--descriptor-molecule-features", nargs="*", default=[])
    parser.add_argument("--charge-xyz-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--cohort", type=Path, default=Path("results/xai/flip_cohort.csv"))
    parser.add_argument("--weight-mode", default="abs")
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/xai/gate_decomposition.csv"))
    args = parser.parse_args()

    frames = []
    for fold in args.folds:
        print(f"fold {fold} ...", flush=True)
        frames.append(run_fold(
            args.runs_dir, args.descriptor_config, fold,
            tuple(args.descriptor_token_features), tuple(args.descriptor_molecule_features),
            args.charge_xyz_dir, args.split_name, args.weight_mode,
            args.target_class, args.num_workers,
        ))
    data = pd.concat(frames, ignore_index=True)
    print(f"\ndecomposed {len(data)} molecules -- no exclusions\n")

    cohort = pd.read_csv(args.cohort)[["smiles", "fold", "stratum", "alert_positive"]]
    data = data.merge(cohort, on=["smiles", "fold"], how="left")

    # how much did the descriptor branch move the prediction, in probability?
    data["prob_shift"] = data["prob"] - data["base_only_prob"]

    print("=" * 84)
    print("DESCRIPTOR PATHWAY CONTRIBUTION  (gate x aux_delta, in logits)")
    print("=" * 84)
    print(f"  {'stratum':26s} {'n':>6s} {'gate':>8s} {'contrib':>10s} {'|contrib|':>10s} {'prob shift':>11s}")
    print("  " + "-" * 74)
    for stratum in ["flip_to_correct", "flip_to_wrong", "both_correct", "both_wrong"]:
        sub = data[data["stratum"] == stratum]
        if not len(sub):
            continue
        print(f"  {stratum:26s} {len(sub):6d} {sub['gate'].mean():8.3f} "
              f"{sub['descriptor_contribution'].mean():+10.4f} "
              f"{sub['descriptor_contribution'].abs().mean():10.4f} "
              f"{sub['prob_shift'].mean():+11.4f}")

    print("\n" + "=" * 84)
    print("BY DIRECTION — does the descriptor branch push the right way?")
    print("=" * 84)
    groups = {
        "FP corrected (true safe)":   (data["stratum"] == "flip_to_correct") & (data["label"] == 0),
        "FN corrected (true toxic)":  (data["stratum"] == "flip_to_correct") & (data["label"] == 1),
        "broke it, true safe":        (data["stratum"] == "flip_to_wrong") & (data["label"] == 0),
        "broke it, true toxic":       (data["stratum"] == "flip_to_wrong") & (data["label"] == 1),
        "both correct":               (data["stratum"] == "both_correct"),
    }
    print(f"  {'group':28s} {'n':>6s} {'contribution':>13s} {'p vs 0':>10s}   expected")
    print("  " + "-" * 78)
    expectations = {
        "FP corrected (true safe)": "negative (pull away from toxic)",
        "FN corrected (true toxic)": "positive (pull toward toxic)",
        "broke it, true safe": "positive = wrong direction",
        "broke it, true toxic": "negative = wrong direction",
        "both correct": "~0",
    }
    rows = []
    for name, mask in groups.items():
        vals = data.loc[mask, "descriptor_contribution"].dropna()
        if len(vals) < 3:
            continue
        p = stats.ttest_1samp(vals, 0.0)[1]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:28s} {len(vals):6d} {vals.mean():+13.4f} {p:10.2g} {star:3s} "
              f"{expectations[name]}")
        rows.append({"group": name, "n": len(vals), "mean_contribution": vals.mean(), "p": p})

    print("\n" + "=" * 84)
    print("ALERT-BEARING ONLY (the cohort the attribution analysis uses)")
    print("=" * 84)
    alert = data[data["alert_positive"] == True]  # noqa: E712
    print(f"  {'group':28s} {'n':>6s} {'contribution':>13s} {'p vs 0':>10s}")
    print("  " + "-" * 60)
    for name, mask in groups.items():
        vals = data.loc[mask & (data["alert_positive"] == True), "descriptor_contribution"].dropna()  # noqa: E712
        if len(vals) < 3:
            continue
        p = stats.ttest_1samp(vals, 0.0)[1]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:28s} {len(vals):6d} {vals.mean():+13.4f} {p:10.2g} {star}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 84)
    print("HOW TO READ THIS")
    print("=" * 84)
    print("  FP corrected negative and FN corrected positive")
    print("      -> the descriptor branch pushes each molecule the right way. It is")
    print("         causing the corrections, not merely correlated with them.")
    print("  Both near zero")
    print("      -> the descriptor branch contributes little; the gain comes from")
    print("         elsewhere and no attribution method will show otherwise.")
    print("  'broke it' rows large in the wrong direction")
    print("      -> the branch is confident and wrong there, which is a finding in")
    print("         itself and belongs in the paper alongside the corrections.")


if __name__ == "__main__":
    main()
