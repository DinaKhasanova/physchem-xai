"""Descriptor ablations: does the model depend on the descriptors at all?

A model can improve by 0.004 ROC AUC either because the descriptors carry real
complementary signal or because the extra parameters happened to help. These
ablations separate the two, on an already-trained checkpoint, without retraining.

Ablations
---------
``identity``                 unmodified reference run
``molecule_zero``            molecule descriptors set to 0 (the current occlusion baseline)
``molecule_median``          molecule descriptors set to the split median
``molecule_shuffle``         molecule descriptors permuted across molecules
``molecule_noise:<f>``       Gaussian noise of ``f`` x per-descriptor std added
``token_zero``               token descriptors set to 0
``token_shuffle_positions``  token descriptors permuted *within* each molecule
``token_noise:<f>``          Gaussian noise of ``f`` x per-molecule std added

How to read the result
----------------------
* ``molecule_shuffle`` barely hurts  -> the head is not using the molecule
  descriptors; any measured gain came from elsewhere.
* ``token_shuffle_positions`` barely hurts -> the model only uses the *summary
  statistics* of the atomic values, not which atom carries which value. Atom-level
  attribution over those features cannot be meaningful, because the model never
  had access to the atom-level structure of the descriptor.
* a large ``token_noise:0.1`` drop -> the model is brittle to descriptor error,
  which matters because semiempirical values carry error of that order.

Usage
-----
    python -m downstream.toxicity.xai.ablation \
        --checkpoint-path runs_cv/.../best.ckpt \
        --output-dir runs_cv/fold0 \
        --token-features charge \
        --molecule-features physchem
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from ..attribution_base import (
    AttributionConfig,
    create_loader,
    load_lightning_model,
    load_split_dataframe,
)
from .common import ensure_dir, resolve_device

DEFAULT_ABLATIONS = (
    "identity",
    "molecule_zero",
    "molecule_median",
    "molecule_shuffle",
    "molecule_noise:0.1",
    "molecule_noise:0.5",
    "token_zero",
    "token_shuffle_positions",
    "token_noise:0.1",
)


def _metrics(labels: list[int], probabilities: list[float]) -> dict[str, float]:
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        balanced_accuracy_score,
        roc_auc_score,
    )

    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }
    if len(set(labels)) > 1:
        out["roc_auc"] = float(roc_auc_score(labels, probabilities))
    else:
        out["roc_auc"] = float("nan")
    return out


def _collect_molecule_features(loader, device) -> torch.Tensor:
    chunks = []
    for batch in loader:
        features = batch.get("molecule_features")
        if features is None:
            return torch.zeros((0, 0))
        chunks.append(features.to(device))
    return torch.cat(chunks, dim=0) if chunks else torch.zeros((0, 0))


def _apply_molecule_ablation(
    features: torch.Tensor,
    name: str,
    pool: torch.Tensor,
    offset: int,
    permutation: torch.Tensor | None,
) -> torch.Tensor:
    if features is None or features.numel() == 0:
        return features
    if name == "molecule_zero":
        return torch.zeros_like(features)
    if name == "molecule_median":
        median = pool.median(dim=0).values.to(features.device)
        return median.unsqueeze(0).expand_as(features).clone()
    if name == "molecule_shuffle":
        if permutation is None:
            return features
        rows = permutation[offset : offset + features.shape[0]].to(pool.device)
        return pool[rows].to(features.device)
    if name.startswith("molecule_noise:"):
        factor = float(name.split(":", 1)[1])
        std = pool.std(dim=0).to(features.device)
        return features + torch.randn_like(features) * std.unsqueeze(0) * factor
    return features


def _apply_token_ablation(features: torch.Tensor, attention_mask: torch.Tensor, name: str) -> torch.Tensor:
    if features is None or features.numel() == 0:
        return features
    if name == "token_zero":
        return torch.zeros_like(features)
    if name == "token_shuffle_positions":
        out = features.clone()
        for row in range(features.shape[0]):
            valid = int(attention_mask[row].sum().item())
            # keep the two special positions in place, permute the interior
            interior = slice(1, max(valid - 1, 1))
            width = interior.stop - interior.start
            if width > 1:
                order = torch.randperm(width, device=features.device)
                out[row, interior, :] = features[row, interior, :][order]
        return out
    if name.startswith("token_noise:"):
        factor = float(name.split(":", 1)[1])
        std = features.std(dim=1, keepdim=True)
        return features + torch.randn_like(features) * std * factor
    return features


def run_ablations(
    config: AttributionConfig,
    ablations: tuple[str, ...] = DEFAULT_ABLATIONS,
    seed: int = 0,
) -> pd.DataFrame:
    torch.manual_seed(seed)

    frame = load_split_dataframe(config.output_dir, config.split_name)
    lightning_model = load_lightning_model(config)
    device = resolve_device()
    lightning_model.to(device)

    def make_loader():
        return create_loader(
            encoder=lightning_model.model.encoder,
            frame=frame,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            token_feature_names=config.token_feature_names,
            molecule_feature_names=config.molecule_feature_names,
            charge_xyz_dir=config.charge_xyz_dir,
        )

    pool = _collect_molecule_features(make_loader(), device)
    permutation = torch.randperm(pool.shape[0]) if pool.numel() else None

    rows: list[dict[str, object]] = []
    for ablation in ablations:
        labels_all: list[int] = []
        probabilities_all: list[float] = []
        offset = 0

        for batch in make_loader():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            token_features = batch.get("token_features")
            molecule_features = batch.get("molecule_features")
            if token_features is not None:
                token_features = token_features.to(device)
            if molecule_features is not None:
                molecule_features = molecule_features.to(device)

            if ablation.startswith("molecule_"):
                molecule_features = _apply_molecule_ablation(
                    molecule_features, ablation, pool, offset, permutation
                )
            elif ablation.startswith("token_"):
                token_features = _apply_token_ablation(token_features, attention_mask, ablation)

            with torch.no_grad():
                logits = lightning_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_features=token_features,
                    molecule_features=molecule_features,
                )
                probabilities = torch.softmax(logits, dim=-1)[:, config.target_class]

            labels_all.extend(int(value) for value in labels.tolist())
            probabilities_all.extend(float(value) for value in probabilities.detach().cpu().tolist())
            offset += input_ids.shape[0]

        metrics = _metrics(labels_all, probabilities_all)
        metrics["ablation"] = ablation
        rows.append(metrics)

    result = pd.DataFrame(rows)[["ablation", "roc_auc", "balanced_accuracy", "accuracy"]]
    reference = result.loc[result["ablation"] == "identity"]
    if not reference.empty:
        base_auc = float(reference["roc_auc"].iloc[0])
        base_bacc = float(reference["balanced_accuracy"].iloc[0])
        result["delta_roc_auc"] = result["roc_auc"] - base_auc
        result["delta_balanced_accuracy"] = result["balanced_accuracy"] - base_bacc
    return result


def describe(frame: pd.DataFrame) -> str:
    lines = ["Descriptor ablation results (negative delta = ablation hurt the model)", ""]
    lines.append(frame.to_string(index=False, float_format=lambda value: f"{value: .4f}"))
    lines.append("")

    def delta(name: str) -> float | None:
        row = frame.loc[frame["ablation"] == name]
        if row.empty or "delta_roc_auc" not in frame:
            return None
        return float(row["delta_roc_auc"].iloc[0])

    shuffle = delta("molecule_shuffle")
    if shuffle is not None:
        if abs(shuffle) < 0.005:
            lines.append(
                f"molecule_shuffle changes ROC AUC by {shuffle:+.4f}: permuting the molecule "
                "descriptors across molecules costs the model almost nothing, i.e. the head is "
                "not actually reading them."
            )
        else:
            lines.append(
                f"molecule_shuffle changes ROC AUC by {shuffle:+.4f}: the head does depend on "
                "the molecule descriptors."
            )

    positions = delta("token_shuffle_positions")
    if positions is not None:
        if abs(positions) < 0.005:
            lines.append(
                f"token_shuffle_positions changes ROC AUC by {positions:+.4f}: which atom carries "
                "which value is irrelevant to the prediction. Only the summary statistics matter, "
                "so atom-level attribution over these features has no mechanism to be faithful."
            )
        else:
            lines.append(
                f"token_shuffle_positions changes ROC AUC by {positions:+.4f}: the model uses the "
                "positional arrangement of the atomic descriptors."
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", type=str, default="test", choices=("train", "validation", "test"))
    parser.add_argument("--pretrained-model-name", type=str, default="fabikru/MolEncoder")
    parser.add_argument("--model-dir", type=Path, default=Path("models/molencoder"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--token-features", nargs="*", default=())
    parser.add_argument("--molecule-features", nargs="*", default=())
    parser.add_argument("--charge-xyz-dir", type=Path, default=None)
    parser.add_argument("--ablations", nargs="*", default=DEFAULT_ABLATIONS)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AttributionConfig(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        split_name=args.split_name,
        pretrained_model_name=args.pretrained_model_name,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_sequence_length=args.max_sequence_length,
        token_feature_names=tuple(args.token_features),
        molecule_feature_names=tuple(args.molecule_features),
        charge_xyz_dir=args.charge_xyz_dir,
        target_class=args.target_class,
    )
    frame = run_ablations(config, ablations=tuple(args.ablations), seed=args.seed)

    out_dir = ensure_dir(args.output_dir / "xai_diagnostics")
    frame.to_csv(out_dir / "ablation.csv", index=False)
    text = describe(frame)
    (out_dir / "ablation.txt").write_text(text, encoding="utf-8")
    (out_dir / "ablation.json").write_text(json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(text)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
