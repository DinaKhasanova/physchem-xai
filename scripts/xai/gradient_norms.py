"""Measure gradient norms in the fusion pathway relative to the encoder pathway.

This is the measurement behind the Methods claim that, under a single global
learning rate, the randomly-initialised descriptor pathway receives far smaller
gradients than the pretrained encoder and therefore never trains.

It reproduces that setting deliberately: ONE learning rate for every parameter
(no --fusion-learning-rate), model freshly built exactly as run_table_7251.sh
builds it, real training batches from a real fold split. Gradient norms are
read after backward() and before any optimiser step, so nothing here depends on
the learning rate at step 0; the optimiser is stepped only so later batches
reflect a model that has started to move.

    cd ~/physchem-xai_upload && source .venv/bin/activate && export PYTHONPATH=src
    python scripts/xai/gradient_norms.py --config gasteiger --token-features charge

Reports, per batch and pooled, the ratio

    ||grad(fusion params)|| / ||grad(encoder params)||

as a percentage. FUSION_PARAM_MARKERS in train.py defines which parameters
count as fusion; the same split drives the --fusion-learning-rate group.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from downstream.toxicity.compute_occlusion import create_loader, load_split_dataframe
from downstream.toxicity.features import molecule_feature_dimension
from downstream.toxicity.finetune import SmilesAnchorResidualAuxClassifier
from downstream.toxicity.train import ToxicityLightningModule
from molencoder.config import MolEncoderConfig
from molencoder.model import MolEncoderWrapper

PRETRAINED = "fabikru/MolEncoder"


def group_grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item()) ** 2
    return total ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("runs_7251"))
    ap.add_argument("--config", default="gasteiger")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--token-features", nargs="*", default=["charge"])
    ap.add_argument("--molecule-features", nargs="*", default=[])
    ap.add_argument("--token-feature-normalize-mode", default="none",
                    choices=("per_molecule", "none", "dataset"),
                    help="must match how this config was trained")
    ap.add_argument("--charge-xyz-dir", type=Path, default=None)
    ap.add_argument("--token-weight-mode", default="abs")
    ap.add_argument("--gate-bias-init", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-sequence-length", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=2e-5,
                    help="single global lr, i.e. the condition being diagnosed")
    ap.add_argument("--batches", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="local MolEncoder copy; defaults to models/molencoder when present")
    args = ap.parse_args()

    model_dir = args.model_dir
    if model_dir is None and Path("models/molencoder").exists():
        model_dir = Path("models/molencoder")

    torch.manual_seed(args.seed)

    encoder = MolEncoderWrapper(MolEncoderConfig(
        pretrained_model_name=PRETRAINED,
        max_sequence_length=args.max_sequence_length,
        num_labels=2,
        model_cache_dir=model_dir,
    ))
    model = SmilesAnchorResidualAuxClassifier(
        encoder=encoder,
        num_labels=2,
        token_feature_dim=len(args.token_features),
        molecule_feature_dim=molecule_feature_dimension(tuple(args.molecule_features)),
        weight_mode=args.token_weight_mode,
        gate_bias_init=args.gate_bias_init,
    )
    lm = ToxicityLightningModule(model=model, learning_rate=args.learning_rate,
                                 weight_decay=1e-2)
    encoder_params, fusion_params, head_params = lm.split_parameter_groups()
    print(f"parameter tensors: encoder={len(encoder_params)} "
          f"fusion={len(fusion_params)} head={len(head_params)}")
    if not fusion_params:
        sys.exit("no fusion parameters found -- check --token-features")

    fold_dir = args.runs_dir / args.config / f"fold_{args.fold}"
    frame = load_split_dataframe(fold_dir, "train")
    loader = create_loader(
        encoder=model.encoder, frame=frame, batch_size=args.batch_size, num_workers=0,
        token_feature_names=tuple(args.token_features),
        molecule_feature_names=tuple(args.molecule_features),
        charge_xyz_dir=args.charge_xyz_dir,
        token_feature_normalize_mode=args.token_feature_normalize_mode,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-2)
    model.train()

    ratios = []
    print(f"\n{'batch':>6} {'||g_enc||':>12} {'||g_fus||':>12} {'fusion/encoder':>16}")
    for step, batch in enumerate(loader):
        if step >= args.batches:
            break
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_features=batch.get("token_features"),
            molecule_features=batch.get("molecule_features"),
        )
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward()

        g_enc = group_grad_norm(encoder_params)
        g_fus = group_grad_norm(fusion_params)
        ratio = 100.0 * g_fus / g_enc if g_enc > 0 else float("nan")
        ratios.append(ratio)
        if step % 10 == 0 or step < 5:
            print(f"{step:>6} {g_enc:>12.5f} {g_fus:>12.5f} {ratio:>15.3f}%")
        optimizer.step()

    if ratios:
        t = torch.tensor(ratios)
        print(f"\nover {len(ratios)} batches, single lr = {args.learning_rate:g}")
        print(f"  fusion/encoder gradient-norm ratio:")
        print(f"    min    {t.min():.3f}%")
        print(f"    median {t.median():.3f}%")
        print(f"    mean   {t.mean():.3f}%")
        print(f"    max    {t.max():.3f}%")
        print(f"\n  quote as: {t.min():.2f}-{t.max():.2f}% of the encoder pathway")


if __name__ == "__main__":
    main()
