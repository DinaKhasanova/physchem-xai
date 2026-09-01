"""Occlusion that keeps the molecule consistent with the perturbation.

Two problems with the occlusion currently used in the manuscript, both visible
in ``compute_occlusion.py`` and ``compute_global_occlusion.py``:

1. **Token occlusion does not update the molecule descriptors.** When a token is
   replaced by the padding id, ``molecule_features`` is passed through unchanged
   (``compute_occlusion.py``, the ``molecule_feature_vector`` argument). The
   classifier therefore still receives the true molecular weight, logP, TPSA and
   orbital energies of the *intact* molecule. Any pathway that runs through the
   descriptors is invisible to the attribution, and the measured effect of
   removing an atom is systematically too small. This is the most likely reason
   the descriptor-augmented attributions look barely different from the
   SMILES-only ones.

2. **Global occlusion sets descriptors to zero.** Zero is not a neutral value for
   scaled descriptors (see :mod:`baselines`); it is an out-of-distribution point,
   so the score conflates importance with extrapolation.

This module provides consistent alternatives.

Token modes
-----------
``pad``        legacy behaviour, kept for exact comparison
``mask``       use the tokenizer mask token instead of padding, when one exists
``recompute``  delete the atom from the molecule, regenerate the SMILES, and
               rebuild **both** token-level and molecule-level features from the
               resulting structure

Global modes
------------
``zero``       legacy behaviour
``median``     replace the descriptor with the split median
``marginal``   average the effect over ``--marginal-samples`` values drawn from
               the empirical distribution of that descriptor

Descriptors that cannot be recomputed without new quantum-chemical calculations
(the ``orbital`` block, and atomic values read from xTB output) are *frozen* by
default: the parent molecule's value is carried over rather than silently
falling back to the RDKit approximation, which would otherwise show up as a
large spurious attribution. Pass ``--recompute-all`` to change that, and read
the coverage report first.

Usage
-----
    python -m downstream.toxicity.xai.counterfactual_occlusion \
        --checkpoint-path runs_cv/.../best.ckpt \
        --output-dir runs_cv/fold0 \
        --token-features charge --molecule-features physchem orbital \
        --token-mode recompute --global-mode marginal \
        --row-indices 3 17 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from ..compute_occlusion import (
    OcclusionConfig,
    create_loader,
    load_lightning_model,
    load_split_dataframe,
)
from ..features import (
    FEATURE_BUILDERS,
    _build_xyz_index,
    _charge_values_for_smiles,
    _polarizability_values_for_smiles,
    build_molecule_feature_vector,
    build_physchem_descriptor_vector,
    molecule_feature_labels,
)
from .baselines import DescriptorBaseline, compute_molecule_feature_baseline
from .common import (
    char_to_atom_index,
    ensure_dir,
    resolve_device,
    spread_atom_values_over_characters as _spread_atom_values_over_characters,
)

# Feature blocks that require a quantum-chemical calculation on the perturbed
# structure and therefore cannot be honestly recomputed here.
QUANTUM_MOLECULE_FEATURES = frozenset({"orbital"})


@dataclass(slots=True)
class CounterfactualConfig:
    base: OcclusionConfig
    token_mode: str = "recompute"
    global_mode: str = "marginal"
    marginal_samples: int = 16
    recompute_all: bool = False
    baseline_path: Path | None = None
    baseline_max_molecules: int | None = 2000
    seed: int = 0
    frozen_molecule_features: tuple[str, ...] = field(default=())


# --------------------------------------------------------------------------
# atom-level values, carried across a perturbation without re-deriving them
# --------------------------------------------------------------------------
def parent_atom_values(
    smiles: str,
    feature_names: tuple[str, ...],
    charge_xyz_dir: Path | None,
    charge_xyz_index: dict | None,
) -> dict[str, list[float]]:
    """Atom-level values for the intact molecule, one list per token feature."""

    values: dict[str, list[float]] = {}
    for name in feature_names:
        if name == "charge":
            values[name] = _charge_values_for_smiles(smiles, charge_xyz_dir, charge_xyz_index)
        elif name == "polarizability":
            values[name] = _polarizability_values_for_smiles(smiles, charge_xyz_dir, charge_xyz_index)
        else:
            values[name] = FEATURE_BUILDERS[name](smiles)
    return values


def _smiles_output_order(mol) -> list[int]:
    """Atom indices of ``mol`` in the order they appear in its canonical SMILES."""

    raw = mol.GetProp("_smilesAtomOutputOrder")
    inner = raw.strip().lstrip("[").rstrip("]").rstrip(",")
    if not inner:
        return []
    return [int(part) for part in inner.split(",") if part.strip() != ""]


def delete_atom(smiles: str, atom_index: int):
    """Remove one atom and return ``(new_smiles, parent_index_per_new_smiles_atom)``.

    The second element maps each atom position of the *new* SMILES back to an
    atom index of the parent molecule, so atomic values computed once for the
    parent can be carried over instead of recomputed with a different method.
    Returns ``(None, None)`` when the deletion cannot produce a valid molecule.
    """

    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or atom_index >= mol.GetNumAtoms():
        return None, None

    editable = Chem.RWMol(mol)
    editable.RemoveAtom(atom_index)
    new_mol = editable.GetMol()

    try:
        Chem.SanitizeMol(new_mol)
    except Exception:
        return None, None

    fragments = Chem.GetMolFrags(new_mol, asMols=False)
    if len(fragments) > 1:
        largest = max(fragments, key=len)
        keep = set(largest)
        editable = Chem.RWMol(new_mol)
        for index in sorted((i for i in range(new_mol.GetNumAtoms()) if i not in keep), reverse=True):
            editable.RemoveAtom(index)
        new_mol = editable.GetMol()
        try:
            Chem.SanitizeMol(new_mol)
        except Exception:
            return None, None
        survivors = sorted(keep)
    else:
        survivors = list(range(new_mol.GetNumAtoms()))

    if new_mol.GetNumAtoms() == 0:
        return None, None

    new_smiles = Chem.MolToSmiles(new_mol, canonical=True)
    try:
        order = _smiles_output_order(new_mol)
    except KeyError:
        order = list(range(new_mol.GetNumAtoms()))

    # index in the post-deletion molecule -> index in the parent molecule
    def to_parent(index_after_delete: int) -> int:
        original = survivors[index_after_delete] if survivors else index_after_delete
        return original if original < atom_index else original + 1

    parent_index_per_position = [to_parent(index) for index in order]
    return new_smiles, parent_index_per_position


def token_matrix_from_atom_values(
    smiles: str,
    feature_names: tuple[str, ...],
    atom_values: dict[str, list[float]],
    encoded_length: int,
    normalize_mode: str = "per_molecule",
) -> torch.Tensor:
    """Reimplementation of ``features.build_token_feature_matrix`` with supplied values.

    Kept byte-for-byte consistent with the original in clamping, log1p and
    normalisation so that perturbed and unperturbed inputs are on the same scale.
    """

    names = list(feature_names)
    if not names:
        return torch.zeros((encoded_length, 0), dtype=torch.float32)

    columns: list[list[float]] = []
    for name in names:
        char_values = _spread_atom_values_over_characters(smiles, atom_values.get(name, []))
        token_values = [0.0] + char_values + [0.0]
        if len(token_values) < encoded_length:
            token_values.extend([0.0] * (encoded_length - len(token_values)))
        else:
            token_values = token_values[:encoded_length]
        columns.append(token_values)

    matrix = torch.tensor(columns, dtype=torch.float32).transpose(0, 1)
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    for index, name in enumerate(names):
        if name == "charge":
            matrix[:, index] = torch.clamp(matrix[:, index], -2.0, 2.0)
        if name.startswith("neighbor"):
            matrix[:, index] = torch.log1p(matrix[:, index].clamp(min=0.0))
    if normalize_mode == "per_molecule":
        mean = matrix.mean(dim=0, keepdim=True)
        std = matrix.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (matrix - mean) / std
    return matrix


def molecule_vector_for_perturbed(
    new_smiles: str,
    molecule_feature_names: tuple[str, ...],
    parent_vector: torch.Tensor,
    frozen: frozenset[str],
    charge_xyz_dir: Path | None,
    charge_xyz_index: dict | None,
) -> torch.Tensor:
    """Recompute the recomputable descriptor blocks, carry the frozen ones over."""

    from ..features import MOLECULE_FEATURE_DIMS  # noqa: PLC0415

    pieces: list[torch.Tensor] = []
    offset = 0
    for name in molecule_feature_names:
        width = MOLECULE_FEATURE_DIMS[name]
        if name in frozen:
            pieces.append(parent_vector[offset : offset + width].clone())
        elif name == "physchem":
            pieces.append(torch.tensor(build_physchem_descriptor_vector(new_smiles), dtype=torch.float32))
        else:
            pieces.append(
                build_molecule_feature_vector(
                    smiles=new_smiles,
                    feature_names=(name,),
                    charge_xyz_dir=charge_xyz_dir,
                    charge_xyz_index=charge_xyz_index,
                )
            )
        offset += width
    if not pieces:
        return parent_vector.clone()
    return torch.cat(pieces, dim=0).to(parent_vector.dtype)


# --------------------------------------------------------------------------
# main routine
# --------------------------------------------------------------------------
def compute(config: CounterfactualConfig) -> dict[str, Path]:
    base = config.base
    frame = load_split_dataframe(base.output_dir, base.split_name)
    source_indices = frame.index.tolist()
    if base.row_indices:
        selected = list(base.row_indices)
        frame = frame.iloc[selected].reset_index(drop=True)
        source_indices = selected

    lightning_model = load_lightning_model(base)
    device = resolve_device()
    lightning_model.to(device)
    model = lightning_model.model
    tokenizer = model.encoder.tokenizer

    torch.manual_seed(config.seed)

    frozen = frozenset(config.frozen_molecule_features) or (
        frozenset() if config.recompute_all else QUANTUM_MOLECULE_FEATURES
    )
    charge_xyz_index = _build_xyz_index(base.charge_xyz_dir) if base.charge_xyz_dir else None

    baseline: DescriptorBaseline | None = None
    if base.molecule_feature_names and config.global_mode in {"median", "marginal"}:
        if config.baseline_path and Path(config.baseline_path).exists():
            baseline = DescriptorBaseline.load(config.baseline_path)
        else:
            full_frame = load_split_dataframe(base.output_dir, "train")
            baseline = compute_molecule_feature_baseline(
                smiles_list=[str(value) for value in full_frame["smiles"].tolist()],
                molecule_feature_names=base.molecule_feature_names,
                charge_xyz_dir=base.charge_xyz_dir,
                max_molecules=config.baseline_max_molecules,
            )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if config.token_mode == "mask" and mask_id is None:
        raise ValueError("Tokenizer has no mask token; use --token-mode pad or recompute.")

    loader = create_loader(
        encoder=model.encoder,
        frame=frame,
        batch_size=1,
        num_workers=base.num_workers,
        token_feature_names=base.token_feature_names,
        molecule_feature_names=base.molecule_feature_names,
        charge_xyz_dir=base.charge_xyz_dir,
    )

    atom_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    feature_labels = molecule_feature_labels(base.molecule_feature_names) if base.molecule_feature_names else ()

    def predict(input_ids, attention_mask, token_features, molecule_features) -> float:
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_features=token_features,
                molecule_features=molecule_features,
            )
            return float(torch.softmax(logits, dim=-1)[0, base.target_class].item())

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
        source_row_index = source_indices[running_index]
        base_prob = predict(input_ids, attention_mask, token_features, molecule_features)

        # ---------------- atom-level attribution ----------------
        mapping = char_to_atom_index(smiles)
        n_atoms = 0
        for value in mapping:
            if value is not None:
                n_atoms = max(n_atoms, value + 1)

        atom_values = (
            parent_atom_values(smiles, base.token_feature_names, base.charge_xyz_dir, charge_xyz_index)
            if base.token_feature_names
            else {}
        )

        for atom_index in range(n_atoms):
            char_positions = [i for i, value in enumerate(mapping) if value == atom_index]
            if not char_positions:
                continue

            if config.token_mode in {"pad", "mask"}:
                replacement = pad_id if config.token_mode == "pad" else mask_id
                occluded_ids = input_ids.clone()
                occluded_token_features = token_features.clone() if token_features is not None else None
                for char_position in char_positions:
                    token_position = char_position + 1
                    if token_position >= occluded_ids.shape[1]:
                        continue
                    occluded_ids[0, token_position] = replacement
                    if occluded_token_features is not None:
                        occluded_token_features[0, token_position, :] = 0.0
                occluded_prob = predict(
                    occluded_ids, attention_mask, occluded_token_features, molecule_features
                )
                new_smiles = ""
                status = "ok"
            else:
                new_smiles, parent_positions = delete_atom(smiles, atom_index)
                if new_smiles is None:
                    atom_rows.append(
                        {
                            "row_index": running_index,
                            "source_row_index": source_row_index,
                            "smiles": smiles,
                            "label": label,
                            "positive_probability": base_prob,
                            "atom_index": atom_index,
                            "occlusion_score": 0.0,
                            "perturbed_smiles": "",
                            "status": "deletion_failed",
                        }
                    )
                    continue

                encoded = model.encoder.tokenize([new_smiles])
                new_input_ids = encoded["input_ids"].to(device)
                new_attention_mask = encoded["attention_mask"].to(device)

                new_token_features = None
                if base.token_feature_names and atom_values:
                    carried = {
                        name: [
                            values[parent_index] if parent_index < len(values) else 0.0
                            for parent_index in parent_positions
                        ]
                        for name, values in atom_values.items()
                    }
                    new_token_features = token_matrix_from_atom_values(
                        smiles=new_smiles,
                        feature_names=base.token_feature_names,
                        atom_values=carried,
                        encoded_length=int(new_input_ids.shape[1]),
                    ).unsqueeze(0).to(device)

                new_molecule_features = None
                if molecule_features is not None:
                    new_molecule_features = molecule_vector_for_perturbed(
                        new_smiles=new_smiles,
                        molecule_feature_names=base.molecule_feature_names,
                        parent_vector=molecule_features[0].detach().cpu(),
                        frozen=frozen,
                        charge_xyz_dir=base.charge_xyz_dir,
                        charge_xyz_index=charge_xyz_index,
                    ).unsqueeze(0).to(device)

                occluded_prob = predict(
                    new_input_ids, new_attention_mask, new_token_features, new_molecule_features
                )
                status = "ok"

            atom_rows.append(
                {
                    "row_index": running_index,
                    "source_row_index": source_row_index,
                    "smiles": smiles,
                    "label": label,
                    "positive_probability": base_prob,
                    "atom_index": atom_index,
                    "occlusion_score": float(base_prob - occluded_prob),
                    "perturbed_probability": occluded_prob,
                    "perturbed_smiles": new_smiles or "",
                    "status": status,
                    "token_mode": config.token_mode,
                }
            )

        # ---------------- global descriptor attribution ----------------
        if molecule_features is not None and feature_labels:
            parent_vector = molecule_features[0].clone()
            for feature_index, feature_name in enumerate(feature_labels):
                if config.global_mode == "zero":
                    replacements = [torch.zeros(1, dtype=parent_vector.dtype, device=device)]
                elif config.global_mode == "median":
                    value = baseline.median[feature_index] if baseline else torch.zeros(1)
                    replacements = [value.to(device).reshape(1)]
                else:
                    if baseline is None or baseline.pool.numel() == 0:
                        replacements = [torch.zeros(1, dtype=parent_vector.dtype, device=device)]
                    else:
                        draws = baseline.sample(config.marginal_samples)
                        replacements = [draws[k, feature_index].to(device).reshape(1) for k in range(draws.shape[0])]

                deltas: list[float] = []
                for replacement in replacements:
                    perturbed = parent_vector.clone()
                    perturbed[feature_index] = replacement.to(perturbed.dtype)[0]
                    perturbed_prob = predict(
                        input_ids, attention_mask, token_features, perturbed.unsqueeze(0)
                    )
                    deltas.append(base_prob - perturbed_prob)

                mean_delta = sum(deltas) / len(deltas)
                global_rows.append(
                    {
                        "row_index": running_index,
                        "source_row_index": source_row_index,
                        "smiles": smiles,
                        "label": label,
                        "positive_probability": base_prob,
                        "feature_name": feature_name,
                        "feature_value": float(parent_vector[feature_index].item()),
                        "occlusion_score": float(mean_delta),
                        "occlusion_score_sd": float(
                            (sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)) ** 0.5
                        ),
                        "n_samples": len(deltas),
                        "global_mode": config.global_mode,
                    }
                )

        running_index += 1

    suffix = "all" if not base.row_indices else "rows_" + "_".join(str(i) for i in base.row_indices)
    out_dir = ensure_dir(
        base.output_dir / "counterfactual_occlusion" / f"{base.split_name}_{config.token_mode}_{suffix}"
    )

    paths: dict[str, Path] = {}
    atom_frame = pd.DataFrame(atom_rows)
    atom_path = out_dir / "atom_occlusion.csv"
    atom_frame.to_csv(atom_path, index=False)
    paths["atom"] = atom_path

    if global_rows:
        global_frame = pd.DataFrame(global_rows)
        global_path = out_dir / "global_occlusion.csv"
        global_frame.to_csv(global_path, index=False)
        paths["global"] = global_path

        aggregate = (
            global_frame.groupby("feature_name", as_index=False)
            .agg(
                mean_occlusion_score=("occlusion_score", "mean"),
                mean_absolute_occlusion_score=("occlusion_score", lambda values: values.abs().mean()),
                max_absolute_occlusion_score=("occlusion_score", lambda values: values.abs().max()),
            )
            .sort_values("mean_absolute_occlusion_score", ascending=False)
            .reset_index(drop=True)
        )
        aggregate_path = out_dir / "global_occlusion_aggregate.csv"
        aggregate.to_csv(aggregate_path, index=False)
        paths["global_aggregate"] = aggregate_path

    meta = {
        "token_mode": config.token_mode,
        "global_mode": config.global_mode,
        "marginal_samples": config.marginal_samples,
        "frozen_molecule_features": sorted(frozen),
        "token_features": list(base.token_feature_names),
        "molecule_features": list(base.molecule_feature_names),
        "n_molecules": running_index,
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["config"] = out_dir / "config.json"
    return paths


def parse_args() -> CounterfactualConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", type=str, default="test", choices=("train", "validation", "test"))
    parser.add_argument("--pretrained-model-name", type=str, default="fabikru/MolEncoder")
    parser.add_argument("--model-dir", type=Path, default=Path("models/molencoder"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--row-indices", nargs="*", type=int, default=())
    parser.add_argument("--token-features", nargs="*", default=())
    parser.add_argument("--molecule-features", nargs="*", default=())
    parser.add_argument("--charge-xyz-dir", type=Path, default=None)
    parser.add_argument("--token-mode", choices=("pad", "mask", "recompute"), default="recompute")
    parser.add_argument("--global-mode", choices=("zero", "median", "marginal"), default="marginal")
    parser.add_argument("--marginal-samples", type=int, default=16)
    parser.add_argument("--recompute-all", action="store_true")
    parser.add_argument("--baseline-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = OcclusionConfig(
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
        row_indices=tuple(args.row_indices),
        target_class=args.target_class,
    )
    return CounterfactualConfig(
        base=base,
        token_mode=args.token_mode,
        global_mode=args.global_mode,
        marginal_samples=args.marginal_samples,
        recompute_all=args.recompute_all,
        baseline_path=args.baseline_path,
        seed=args.seed,
    )


def main() -> None:
    paths = compute(parse_args())
    for name, path in paths.items():
        print(f"{name:<18} {path}")


if __name__ == "__main__":
    main()
