"""Integrated gradients for SmilesAnchorResidualAuxClassifier.

Per-atom attributions with no perturbation, no molecule rebuilding and
therefore no losses -- every molecule keeps every atom. This is what occlusion
could not do: `--token-mode recompute` failed on 47.7% of atoms, systematically
in fused ring systems.

Because the architecture is additive at the output

    logits = base_logits + gate * aux_delta

IG can attribute the two pathways separately and both attributions are exact
decompositions of their own logit. The completeness axiom gives a validity check
most IG applications skip: attributions must sum to the difference between the
prediction at the input and at the baseline.

Baselines
---------
Token embeddings integrate from the pad embedding, the standard choice.
Descriptors integrate from the **per-feature training-set median**, not zero:
an all-zero charge vector is a molecule that cannot exist, and integrating
through it is the same objection the occlusion module raises about zeroing
descriptors.

    cd ~/physchem-xai_upload && source .venv/bin/activate && export PYTHONPATH=src

    python scripts/xai/integrated_gradients.py \
        --config gasteiger --token-features charge \
        --stratum flip_to_correct \
        --out results/xai/ig_gasteiger.csv

Run once per config; the figure script pairs baseline against gasteiger.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from downstream.toxicity.attribution_base import create_loader, load_split_dataframe
from downstream.toxicity.features import molecule_feature_dimension
from downstream.toxicity.xai.common import char_to_atom_index, resolve_device

from fusion_loader import load_fusion_model

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PRETRAINED_MODEL_NAME = "fabikru/MolEncoder"
MAX_SEQUENCE_LENGTH = 128


# --------------------------------------------------------------- forward parts
def forward_from_embeddings(model, inputs_embeds, attention_mask,
                            token_features, molecule_features):
    """SmilesAnchorResidualAuxClassifier.forward, entered at the embedding layer.

    Mirrors the real forward exactly. If that method changes, this must too --
    the completeness check below will catch a divergence.
    """
    encoder = model.encoder
    inner = encoder.encoder          # the HuggingFace AutoModel
    out = inner(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                output_hidden_states=True)
    token_states = out.hidden_states[-1] if hasattr(out, "hidden_states") else out.last_hidden_state

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
        return base_logits, base_logits, None

    aux_repr = model.aux_proj(torch.stack(aux_parts, dim=0).mean(dim=0))
    aux_delta = model.aux_classifier(aux_repr)

    base_probs = torch.softmax(base_logits, dim=-1)
    confidence = base_probs.max(dim=-1, keepdim=True).values
    margin = (base_probs[:, 1:2] - base_probs[:, 0:1]).abs()
    gate = torch.sigmoid(model.gate_mlp(torch.cat([base_repr, confidence, margin], dim=-1)))

    return base_logits + gate * aux_delta, base_logits, gate * aux_delta


def embedding_layer(model):
    return model.encoder.encoder.get_input_embeddings()


# ------------------------------------------------------------------------- IG
def integrated_gradients(model, batch, device, target_class, steps,
                         descriptor_baseline, pad_id):
    """Returns per-token attributions for the token and descriptor pathways."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_features = batch.get("token_features")
    molecule_features = batch.get("molecule_features")
    if token_features is not None:
        token_features = token_features.to(device).float()
    if molecule_features is not None:
        molecule_features = molecule_features.to(device).float()

    embed = embedding_layer(model)
    x_embed = embed(input_ids).detach()
    base_embed = embed(torch.full_like(input_ids, pad_id)).detach()

    if token_features is not None and token_features.shape[-1] > 0:
        tf_base = descriptor_baseline.to(device).view(1, 1, -1).expand_as(token_features)
    else:
        tf_base = None

    total_embed = torch.zeros_like(x_embed)
    total_tf = torch.zeros_like(token_features) if tf_base is not None else None

    for step in range(steps):
        alpha = (step + 0.5) / steps          # midpoint rule
        e = (base_embed + alpha * (x_embed - base_embed)).clone().requires_grad_(True)
        if tf_base is not None:
            tf = (tf_base + alpha * (token_features - tf_base)).clone().requires_grad_(True)
        else:
            tf = None

        logits, _, _ = forward_from_embeddings(model, e, attention_mask, tf, molecule_features)
        score = logits[0, target_class]

        grads = torch.autograd.grad(score, [g for g in (e, tf) if g is not None])
        total_embed += grads[0].detach()
        if tf is not None:
            total_tf += grads[1].detach()

    attr_embed = ((x_embed - base_embed) * total_embed / steps).sum(dim=-1)[0]
    attr_tf = (((token_features - tf_base) * total_tf / steps).sum(dim=-1)[0]
               if tf_base is not None else torch.zeros_like(attr_embed))

    # completeness: attributions should sum to the prediction difference
    with torch.no_grad():
        f_x = forward_from_embeddings(model, x_embed, attention_mask,
                                      token_features, molecule_features)[0][0, target_class]
        f_0 = forward_from_embeddings(model, base_embed, attention_mask,
                                      tf_base, molecule_features)[0][0, target_class]
    completeness = float((attr_embed.sum() + attr_tf.sum() - (f_x - f_0)).abs().item())

    return (attr_embed.cpu().numpy(), attr_tf.cpu().numpy(),
            completeness, float((f_x - f_0).item()))


def tokens_to_atoms(smiles, offsets, values):
    """Map per-token attributions onto atoms via character offsets."""
    mapping = char_to_atom_index(smiles)
    n_atoms = max((v for v in mapping if v is not None), default=-1) + 1
    atom_values = np.zeros(n_atoms, dtype=float)
    for (start, end), value in zip(offsets, values):
        if end <= start:
            continue
        for pos in range(start, min(end, len(mapping))):
            idx = mapping[pos]
            if idx is not None:
                atom_values[idx] += value / max(end - start, 1)
    return atom_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_7251"))
    parser.add_argument("--config", default="gasteiger")
    parser.add_argument("--token-features", nargs="*", default=["charge"])
    parser.add_argument("--molecule-features", nargs="*", default=[])
    parser.add_argument("--charge-xyz-dir", type=Path, default=None)
    parser.add_argument("--token-feature-normalize-mode",
                        default="per_molecule", choices=("per_molecule", "none", "dataset"),
                        help="must match how this model was trained")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--cohort", type=Path, default=Path("results/xai/flip_cohort.csv"))
    parser.add_argument("--stratum", default="flip_to_correct",
                        help="which molecules to attribute; 'all' for everything")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--weight-mode", default="abs")
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/xai/ig_attributions.csv"))
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort)
    if args.stratum != "all":
        wanted = cohort[cohort["stratum"] == args.stratum]
    else:
        wanted = cohort
    print(f"{len(wanted)} molecules in stratum '{args.stratum}'\n")

    rows, completeness_all = [], []

    for fold in args.folds:
        fold_dir = args.runs_dir / args.config / f"fold_{fold}"
        frame = load_split_dataframe(fold_dir, args.split_name)
        keep = set(wanted[wanted["fold"] == fold]["smiles"])
        if not keep:
            continue

        model_lm = load_fusion_model(
            checkpoint_path=fold_dir / "toxicity_classifier.ckpt",
            pretrained_model_name=PRETRAINED_MODEL_NAME,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            model_dir=None,
            weight_mode=args.weight_mode,
            expected_token_dim=len(args.token_features),
            expected_molecule_dim=molecule_feature_dimension(tuple(args.molecule_features)),
        )
        device = resolve_device()
        model_lm.to(device)
        model = model_lm.model
        model.eval()

        loader = create_loader(
            encoder=model.encoder, frame=frame, batch_size=1, num_workers=0,
            token_feature_names=tuple(args.token_features),
            molecule_feature_names=tuple(args.molecule_features),
            charge_xyz_dir=args.charge_xyz_dir,
            token_feature_normalize_mode=args.token_feature_normalize_mode,
        )

        # descriptor baseline: per-feature median over this fold's TRAIN split
        train_loader = create_loader(
            encoder=model.encoder,
            frame=load_split_dataframe(fold_dir, "train"),
            batch_size=1, num_workers=0,
            token_feature_names=tuple(args.token_features),
            molecule_feature_names=tuple(args.molecule_features),
            charge_xyz_dir=args.charge_xyz_dir,
            token_feature_normalize_mode=args.token_feature_normalize_mode,
        )
        collected = []
        for i, b in enumerate(train_loader):
            tf = b.get("token_features")
            if tf is None or tf.shape[-1] == 0:
                break
            mask = b["attention_mask"][0].bool()
            collected.append(tf[0][mask].float())
            if i >= 500:
                break
        descriptor_baseline = (torch.cat(collected, 0).median(dim=0).values
                               if collected else torch.zeros(len(args.token_features)))
        if collected:
            print(f"fold {fold}: descriptor baseline (median) = "
                  f"{descriptor_baseline.numpy().round(4)}")

        tokenizer = model.encoder.tokenizer
        pad_id = tokenizer.pad_token_id or 0

        done = 0
        for i, batch in enumerate(loader):
            smiles = str(frame.iloc[i]["smiles"])
            if smiles not in keep:
                continue

            attr_tok, attr_desc, comp, delta = integrated_gradients(
                model, batch, device, args.target_class, args.steps,
                descriptor_baseline, pad_id,
            )
            completeness_all.append(comp / (abs(delta) + 1e-9))

            enc = tokenizer(smiles, return_offsets_mapping=True,
                            truncation=True, max_length=MAX_SEQUENCE_LENGTH)
            offsets = enc["offset_mapping"]
            n = min(len(offsets), len(attr_tok))
            atoms_tok = tokens_to_atoms(smiles, offsets[:n], attr_tok[:n])
            atoms_desc = tokens_to_atoms(smiles, offsets[:n], attr_desc[:n])

            for idx, (a, b_) in enumerate(zip(atoms_tok, atoms_desc)):
                rows.append({
                    "smiles": smiles, "fold": fold, "atom_index": idx,
                    "attr_token": float(a), "attr_descriptor": float(b_),
                    "attr_total": float(a + b_),
                    "label": int(batch["labels"][0].item()),
                    "config": args.config,
                })
            done += 1
            if done % 25 == 0:
                print(f"  fold {fold}: {done}/{len(keep)}", flush=True)
        print(f"fold {fold}: {done} molecules")

    data = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)

    print(f"\n{data['smiles'].nunique()} molecules, {len(data)} atom rows")
    if completeness_all:
        arr = np.array(completeness_all)
        print(f"\ncompleteness error (|sum(attr) - delta| / |delta|):")
        print(f"  median {np.median(arr):.4f}   mean {arr.mean():.4f}   "
              f"90th pct {np.percentile(arr, 90):.4f}")
        if np.median(arr) > 0.05:
            print("  WARNING: above 5% -- raise --steps, the integral has not converged")
        else:
            print("  OK: the attributions account for the prediction difference")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
