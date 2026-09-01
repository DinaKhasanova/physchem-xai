"""Does the trained model actually use the physicochemical descriptors?

This module answers that from the checkpoint alone -- no encoder download, no
dataset, no GPU. It is the cheapest and most informative thing to run first.

Background
----------
``ToxicityClassifier`` concatenates four blocks before the classifier head::

    [ cls_embedding | pooled_context | feature_stats | molecule_features ]
      768             768              4*n_token_feat  n_mol_feat

and feeds that through ``LayerNorm -> Linear``. Two consequences matter:

* ``pooled_context`` and ``feature_stats`` are multiplied by the learned scalars
  ``charge_context_scale`` and ``feature_stats_scale`` (both initialised to 0.1).
  If training drove them toward zero, the descriptor pathway was switched off.
* the ``molecule_features`` block is typically 10-16 dims inside a ~1550-dim
  vector. Whether the head listens to it is decided by the LayerNorm gain and
  the first Linear's column norms, which is exactly what we measure here.

Usage
-----
    python -m downstream.toxicity.xai.diagnostics \
        --checkpoint-path runs_cv/.../best.ckpt \
        --token-features charge \
        --molecule-features physchem orbital
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..features import molecule_feature_dimension, molecule_feature_labels
from .common import classifier_block_slices, ensure_dir


def _get(state: dict, *names: str):
    for name in names:
        if name in state:
            return state[name]
    return None


def load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    return {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}


def infer_head_type(state: dict[str, torch.Tensor]) -> str:
    keys = set(state)
    if any(key.startswith("model.conv_stack") for key in keys):
        return "ToxicityCnnClassifier"
    if any(key.startswith("model.feature_proj") for key in keys):
        return "GatedFusionClassifier"
    if any(key.startswith("model.aux_classifier") for key in keys):
        return "SmilesAnchorResidualAuxClassifier"
    if any(key.startswith("model.base_classifier") for key in keys):
        return "ConfidenceResidualFusionClassifier"
    if "model.classifier.0.weight" in keys:
        size = int(state["model.classifier.0.weight"].shape[0])
        encoder_hidden = _encoder_hidden_size(state)
        if encoder_hidden is not None and size == encoder_hidden:
            return "LegacyClsToxicityClassifier"
        return "ToxicityClassifier"
    return "unknown"


def _encoder_hidden_size(state: dict[str, torch.Tensor]) -> int | None:
    for key, value in state.items():
        if key.endswith("encoder.embeddings.word_embeddings.weight"):
            return int(value.shape[1])
    linear = state.get("model.classifier.1.weight")
    if linear is not None:
        return int(linear.shape[0])
    return None


def fusion_scales(state: dict[str, torch.Tensor]) -> dict[str, float]:
    """Learned scalars that gate the descriptor pathway (init value 0.1)."""

    out: dict[str, float] = {}
    for key in ("model.charge_context_scale", "model.feature_stats_scale"):
        tensor = state.get(key)
        if tensor is not None:
            out[key.replace("model.", "")] = float(tensor.reshape(-1)[0].item())
    return out


def block_weight_mass(
    state: dict[str, torch.Tensor],
    token_feature_dim: int,
    molecule_feature_dim: int,
) -> dict[str, dict[str, float]]:
    """Effective per-input-dimension weight mass of each concatenated block.

    For input dim ``i`` the head computes ``LayerNorm`` (elementwise gain
    ``g_i``) then a ``Linear`` with column ``W[:, i]``. The product
    ``|g_i| * ||W[:, i]||_2`` is how strongly that input dimension can move the
    hidden representation, so it is the right quantity to compare across blocks
    of different widths.
    """

    gain = _get(state, "model.classifier.0.weight")
    linear = _get(state, "model.classifier.1.weight")
    if gain is None or linear is None:
        return {}

    in_features = int(gain.shape[0])
    encoder_hidden = int(linear.shape[0])
    slices = classifier_block_slices(encoder_hidden, token_feature_dim, molecule_feature_dim)

    expected = sum(s.stop - s.start for s in slices.values())
    if expected != in_features:
        raise ValueError(
            f"Block layout does not match the checkpoint: computed {expected} input dims "
            f"from token_feature_dim={token_feature_dim}, molecule_feature_dim={molecule_feature_dim}, "
            f"encoder_hidden={encoder_hidden}, but the checkpoint has {in_features}. "
            "Pass the same --token-features/--molecule-features that were used for training."
        )

    column_mass = gain.abs() * linear.norm(dim=0)

    report: dict[str, dict[str, float]] = {}
    for name, block in slices.items():
        width = block.stop - block.start
        if width == 0:
            continue
        values = column_mass[block]
        report[name] = {
            "width": float(width),
            "mean_mass": float(values.mean().item()),
            "total_mass": float(values.sum().item()),
            "max_mass": float(values.max().item()),
        }

    total = sum(entry["total_mass"] for entry in report.values())
    for entry in report.values():
        entry["share_of_total"] = entry["total_mass"] / total if total > 0 else 0.0
    return report


def per_descriptor_mass(
    state: dict[str, torch.Tensor],
    token_feature_dim: int,
    molecule_feature_names: tuple[str, ...],
) -> list[dict[str, float | str]]:
    """Per-named-descriptor weight mass inside the ``molecule_features`` block."""

    if not molecule_feature_names:
        return []
    gain = _get(state, "model.classifier.0.weight")
    linear = _get(state, "model.classifier.1.weight")
    if gain is None or linear is None:
        return []

    molecule_feature_dim = molecule_feature_dimension(molecule_feature_names)
    labels = molecule_feature_labels(molecule_feature_names)
    encoder_hidden = int(linear.shape[0])
    slices = classifier_block_slices(encoder_hidden, token_feature_dim, molecule_feature_dim)
    block = slices["molecule_features"]

    column_mass = (gain.abs() * linear.norm(dim=0))[block]
    if column_mass.numel() != len(labels):
        return []

    rows = [
        {"descriptor": label, "mass": float(column_mass[index].item())}
        for index, label in enumerate(labels)
    ]
    rows.sort(key=lambda row: row["mass"], reverse=True)
    return rows


def describe(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"checkpoint : {report['checkpoint']}")
    lines.append(f"head type  : {report['head_type']}")

    scales = report.get("fusion_scales") or {}
    if scales:
        lines.append("")
        lines.append("Learned descriptor-pathway scales (initialised at 0.100):")
        for name, value in scales.items():
            if abs(value) < 0.02:
                verdict = "COLLAPSED - the descriptor pathway was effectively switched off"
            elif abs(value) < 0.1:
                verdict = "shrunk below init - pathway down-weighted during training"
            else:
                verdict = "grew above init - pathway actively used"
            lines.append(f"  {name:<22} = {value: .4f}   {verdict}")

    blocks = report.get("block_weight_mass") or {}
    if blocks:
        lines.append("")
        lines.append("Classifier input blocks (|LayerNorm gain| x first-Linear column norm):")
        lines.append(f"  {'block':<18}{'width':>7}{'mean/dim':>12}{'total':>12}{'share':>9}")
        for name, entry in blocks.items():
            lines.append(
                f"  {name:<18}{int(entry['width']):>7}{entry['mean_mass']:>12.4f}"
                f"{entry['total_mass']:>12.2f}{entry['share_of_total']:>8.1%}"
            )
        cls_mean = blocks.get("cls", {}).get("mean_mass")
        mol_mean = blocks.get("molecule_features", {}).get("mean_mass")
        if cls_mean and mol_mean:
            ratio = mol_mean / cls_mean
            lines.append("")
            lines.append(
                f"  molecule_features carries {ratio:.2f}x the per-dimension mass of the CLS block."
            )
            if ratio < 1.0:
                lines.append(
                    "  Below 1.0 means each descriptor dimension is weighted no more than an "
                    "average CLS dimension despite being far scarcer - the descriptor signal is "
                    "diluted by the shared LayerNorm over the whole concatenated vector."
                )

    descriptors = report.get("per_descriptor_mass") or []
    if descriptors:
        lines.append("")
        lines.append("Per-descriptor weight mass (highest first):")
        for row in descriptors:
            lines.append(f"  {str(row['descriptor']):<24}{float(row['mass']):.4f}")

    return "\n".join(lines)


def run(
    checkpoint_path: Path,
    token_feature_names: tuple[str, ...],
    molecule_feature_names: tuple[str, ...],
    output_dir: Path | None = None,
) -> dict:
    state = load_state_dict(checkpoint_path)
    token_feature_dim = len(token_feature_names)
    molecule_feature_dim = molecule_feature_dimension(molecule_feature_names)

    report: dict = {
        "checkpoint": str(checkpoint_path),
        "head_type": infer_head_type(state),
        "token_feature_names": list(token_feature_names),
        "molecule_feature_names": list(molecule_feature_names),
        "fusion_scales": fusion_scales(state),
    }
    try:
        report["block_weight_mass"] = block_weight_mass(state, token_feature_dim, molecule_feature_dim)
        report["per_descriptor_mass"] = per_descriptor_mass(state, token_feature_dim, molecule_feature_names)
    except ValueError as exc:
        report["error"] = str(exc)
        report["block_weight_mass"] = {}
        report["per_descriptor_mass"] = []

    if output_dir is not None:
        path = ensure_dir(output_dir) / "fusion_diagnostics.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["saved_to"] = str(path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--token-features", nargs="*", default=())
    parser.add_argument("--molecule-features", nargs="*", default=())
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(
        checkpoint_path=args.checkpoint_path,
        token_feature_names=tuple(args.token_features),
        molecule_feature_names=tuple(args.molecule_features),
        output_dir=args.output_dir,
    )
    print(describe(report))
    if "saved_to" in report:
        print(f"\nSaved JSON to {report['saved_to']}")


if __name__ == "__main__":
    main()
