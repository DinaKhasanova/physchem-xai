"""Quantitative explanation quality, so "better explanations" stops being a claim about two figures.

Metrics
-------
deletion AUC
    Remove atoms in decreasing order of attribution and track the predicted
    toxic probability. A faithful explanation makes it fall fast, so a *low*
    area under the deletion curve is good.
insertion AUC
    Start from a fully occluded molecule and add atoms back in decreasing order
    of attribution. A *high* area is good.
comprehensiveness
    ``p(x) - p(x without the top-k atoms)``. Higher is better.
sufficiency
    ``p(x) - p(only the top-k atoms)``. Lower is better.
stability
    Spearman correlation between the atom attributions obtained from the
    canonical SMILES and from ``--n-randomizations`` randomised SMILES of the
    same molecule. A SMILES-based model that gives a different story for the
    same molecule written a different way is not explaining chemistry, and this
    is the number that shows it.

Every metric is reported per molecule, so two checkpoints (SMILES-only and
descriptor-augmented) can be compared with a paired test rather than by eye.

Usage
-----
    python -m downstream.toxicity.xai.faithfulness \
        --checkpoint-path runs_cv/.../best.ckpt \
        --output-dir runs_cv/fold0 \
        --token-features charge --molecule-features physchem \
        --max-molecules 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from ..attribution_base import (
    AttributionConfig,
    create_loader,
    load_lightning_model,
    load_split_dataframe,
)
from .common import (
    char_to_atom_index,
    curve_auc as _curve_auc,
    ensure_dir,
    resolve_device,
    spearman as _spearman,
)


def _atom_char_positions(smiles: str) -> list[list[int]]:
    mapping = char_to_atom_index(smiles)
    n_atoms = 0
    for value in mapping:
        if value is not None:
            n_atoms = max(n_atoms, value + 1)
    spans: list[list[int]] = [[] for _ in range(n_atoms)]
    for char_index, atom_index in enumerate(mapping):
        if atom_index is not None:
            spans[atom_index].append(char_index)
    return spans


def _occlude_atoms(
    input_ids: torch.Tensor,
    token_features: torch.Tensor | None,
    spans: list[list[int]],
    atoms: list[int],
    replacement_id: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    occluded_ids = input_ids.clone()
    occluded_features = token_features.clone() if token_features is not None else None
    width = occluded_ids.shape[1]
    for atom_index in atoms:
        if atom_index >= len(spans):
            continue
        for char_position in spans[atom_index]:
            token_position = char_position + 1
            if token_position >= width:
                continue
            occluded_ids[0, token_position] = replacement_id
            if occluded_features is not None:
                occluded_features[0, token_position, :] = 0.0
    return occluded_ids, occluded_features


def evaluate(
    config: AttributionConfig,
    max_molecules: int | None = None,
    top_k_fraction: float = 0.2,
    n_randomizations: int = 3,
    use_mask_token: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    frame = load_split_dataframe(config.output_dir, config.split_name)
    if max_molecules:
        frame = frame.iloc[:max_molecules].reset_index(drop=True)

    lightning_model = load_lightning_model(config)
    device = resolve_device()
    lightning_model.to(device)
    model = lightning_model.model
    tokenizer = model.encoder.tokenizer

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mask_id = getattr(tokenizer, "mask_token_id", None)
    replacement_id = mask_id if (use_mask_token and mask_id is not None) else pad_id

    torch.manual_seed(seed)

    loader = create_loader(
        encoder=model.encoder,
        frame=frame,
        batch_size=1,
        num_workers=config.num_workers,
        token_feature_names=config.token_feature_names,
        molecule_feature_names=config.molecule_feature_names,
        charge_xyz_dir=config.charge_xyz_dir,
    )

    def predict(input_ids, attention_mask, token_features, molecule_features) -> float:
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_features=token_features,
                molecule_features=molecule_features,
            )
            return float(torch.softmax(logits, dim=-1)[0, config.target_class].item())

    def attributions_for(smiles: str) -> tuple[list[float], float]:
        """Single-atom occlusion attributions for an arbitrary SMILES string."""

        from ..features import build_molecule_feature_vector, build_token_feature_matrix  # noqa: PLC0415

        encoded = model.encoder.tokenize([smiles])
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        length = int(ids.shape[1])
        tokens = None
        if config.token_feature_names:
            tokens = build_token_feature_matrix(
                smiles=smiles,
                feature_names=config.token_feature_names,
                encoded_length=length,
                charge_xyz_dir=config.charge_xyz_dir,
            ).unsqueeze(0).to(device)
        molecules = None
        if config.molecule_feature_names:
            molecules = build_molecule_feature_vector(
                smiles=smiles,
                feature_names=config.molecule_feature_names,
                charge_xyz_dir=config.charge_xyz_dir,
            ).unsqueeze(0).to(device)

        reference = predict(ids, mask, tokens, molecules)
        spans = _atom_char_positions(smiles)
        scores: list[float] = []
        for atom_index in range(len(spans)):
            occluded_ids, occluded_tokens = _occlude_atoms(
                ids, tokens, spans, [atom_index], replacement_id
            )
            scores.append(reference - predict(occluded_ids, mask, occluded_tokens, molecules))
        return scores, reference

    rows: list[dict[str, object]] = []
    running_index = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = int(batch["labels"][0].item())
        token_features = batch.get("token_features")
        molecule_features = batch.get("molecule_features")
        if token_features is not None:
            token_features = token_features.to(device)
        if molecule_features is not None:
            molecule_features = molecule_features.to(device)

        smiles = str(frame.iloc[running_index]["smiles"])
        spans = _atom_char_positions(smiles)
        n_atoms = len(spans)
        if n_atoms < 2:
            running_index += 1
            continue

        base_prob = predict(input_ids, attention_mask, token_features, molecule_features)

        scores: list[float] = []
        for atom_index in range(n_atoms):
            occluded_ids, occluded_tokens = _occlude_atoms(
                input_ids, token_features, spans, [atom_index], replacement_id
            )
            scores.append(base_prob - predict(occluded_ids, attention_mask, occluded_tokens, molecule_features))

        ranking = sorted(range(n_atoms), key=lambda i: scores[i], reverse=True)

        deletion = [base_prob]
        for step in range(1, n_atoms + 1):
            occluded_ids, occluded_tokens = _occlude_atoms(
                input_ids, token_features, spans, ranking[:step], replacement_id
            )
            deletion.append(predict(occluded_ids, attention_mask, occluded_tokens, molecule_features))

        insertion = []
        for step in range(0, n_atoms + 1):
            hidden = ranking[step:]
            occluded_ids, occluded_tokens = _occlude_atoms(
                input_ids, token_features, spans, hidden, replacement_id
            )
            insertion.append(predict(occluded_ids, attention_mask, occluded_tokens, molecule_features))

        k = max(1, int(round(top_k_fraction * n_atoms)))
        comp_ids, comp_tokens = _occlude_atoms(input_ids, token_features, spans, ranking[:k], replacement_id)
        comprehensiveness = base_prob - predict(comp_ids, attention_mask, comp_tokens, molecule_features)
        suff_ids, suff_tokens = _occlude_atoms(input_ids, token_features, spans, ranking[k:], replacement_id)
        sufficiency = base_prob - predict(suff_ids, attention_mask, suff_tokens, molecule_features)

        stability_values: list[float] = []
        if n_randomizations > 0:
            try:
                from rdkit import Chem  # noqa: PLC0415

                from .counterfactual_occlusion import _smiles_output_order  # noqa: PLC0415

                mol = Chem.MolFromSmiles(smiles)
                canonical_order = None
                if mol is not None:
                    Chem.MolToSmiles(mol, canonical=True)
                    canonical_order = _smiles_output_order(mol)

                for _ in range(n_randomizations):
                    if mol is None or canonical_order is None:
                        break
                    random_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                    random_order = _smiles_output_order(mol)
                    random_scores, _ = attributions_for(random_smiles)
                    if len(random_scores) != len(random_order) or len(scores) != len(canonical_order):
                        continue
                    by_atom_canonical = {atom: scores[pos] for pos, atom in enumerate(canonical_order)}
                    by_atom_random = {atom: random_scores[pos] for pos, atom in enumerate(random_order)}
                    shared = sorted(set(by_atom_canonical) & set(by_atom_random))
                    if len(shared) >= 3:
                        stability_values.append(
                            _spearman(
                                [by_atom_canonical[a] for a in shared],
                                [by_atom_random[a] for a in shared],
                            )
                        )
            except Exception:
                pass

        finite_stability = [value for value in stability_values if value == value]
        rows.append(
            {
                "row_index": running_index,
                "smiles": smiles,
                "label": label,
                "n_atoms": n_atoms,
                "positive_probability": base_prob,
                "deletion_auc": _curve_auc(deletion),
                "insertion_auc": _curve_auc(insertion),
                "comprehensiveness": comprehensiveness,
                "sufficiency": sufficiency,
                "top_k": k,
                "stability_spearman": (
                    sum(finite_stability) / len(finite_stability) if finite_stability else float("nan")
                ),
                "n_stability_samples": len(finite_stability),
            }
        )
        running_index += 1

    return pd.DataFrame(rows)


def describe(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No molecules evaluated."
    columns = [
        "deletion_auc",
        "insertion_auc",
        "comprehensiveness",
        "sufficiency",
        "stability_spearman",
    ]
    lines = [f"Explanation quality over {len(frame)} molecules", ""]
    lines.append(f"  {'metric':<24}{'mean':>10}{'sd':>10}{'better':>10}")
    better = {
        "deletion_auc": "lower",
        "insertion_auc": "higher",
        "comprehensiveness": "higher",
        "sufficiency": "lower",
        "stability_spearman": "higher",
    }
    for column in columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        lines.append(
            f"  {column:<24}{values.mean():>10.4f}{values.std():>10.4f}{better[column]:>10}"
        )
    lines.append("")
    lines.append(
        "  Run this for the SMILES-only checkpoint and the descriptor-augmented one, "
        "then compare per molecule with a paired Wilcoxon test. That is the claim the "
        "manuscript currently supports with two hand-picked figures."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", type=str, default="test", choices=("train", "validation", "test"))
    parser.add_argument("--pretrained-model-name", type=str, default="fabikru/MolEncoder")
    parser.add_argument("--model-dir", type=Path, default=Path("models/molencoder"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--token-features", nargs="*", default=())
    parser.add_argument("--molecule-features", nargs="*", default=())
    parser.add_argument("--charge-xyz-dir", type=Path, default=None)
    parser.add_argument("--max-molecules", type=int, default=200)
    parser.add_argument("--top-k-fraction", type=float, default=0.2)
    parser.add_argument("--n-randomizations", type=int, default=3)
    parser.add_argument("--tag", type=str, default="model", help="Name used in the output filenames")
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
        batch_size=1,
        num_workers=args.num_workers,
        max_sequence_length=args.max_sequence_length,
        token_feature_names=tuple(args.token_features),
        molecule_feature_names=tuple(args.molecule_features),
        charge_xyz_dir=args.charge_xyz_dir,
        target_class=args.target_class,
    )
    frame = evaluate(
        config,
        max_molecules=args.max_molecules,
        top_k_fraction=args.top_k_fraction,
        n_randomizations=args.n_randomizations,
        seed=args.seed,
    )
    out_dir = ensure_dir(args.output_dir / "xai_diagnostics")
    frame.to_csv(out_dir / f"faithfulness_{args.tag}.csv", index=False)
    text = describe(frame)
    (out_dir / f"faithfulness_{args.tag}.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
