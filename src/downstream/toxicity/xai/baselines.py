"""Empirical reference values for descriptor occlusion.

The existing global occlusion sets a descriptor to ``0.0``. That is not a
neutral value: ``features.build_physchem_descriptor_vector`` divides by fixed
constants, so ``MolWt = 0`` means a molecule of zero mass, and the orbital block
stores negative energies, so ``HOMO = 0`` is far outside anything the model saw
in training. The resulting score measures "how far is this point from an
impossible molecule", not "how much does this descriptor matter".

Replacing zero with the training-set median puts the perturbed point inside the
data distribution. Averaging over samples drawn from the empirical column
(``marginal`` mode) approximates the conditional expectation used by SHAP-style
attribution and removes the dependence on any single reference.

Usage
-----
    python -m downstream.toxicity.xai.baselines \
        --dataset-path data/ames_7251.csv \
        --molecule-features physchem orbital \
        --charge-xyz-dir protonation_federico_corina_7246_xyz \
        --output-dir runs_cv/diagnostics
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ..features import (
    _build_xyz_index,
    build_molecule_feature_vector,
    molecule_feature_labels,
)
from .common import ensure_dir


@dataclass(slots=True)
class DescriptorBaseline:
    """Reference values and the empirical column pool for each descriptor."""

    labels: tuple[str, ...]
    median: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    pool: torch.Tensor  # (n_molecules, n_descriptors)

    def reference(self, mode: str) -> torch.Tensor:
        if mode == "median":
            return self.median.clone()
        if mode == "mean":
            return self.mean.clone()
        if mode == "zero":
            return torch.zeros_like(self.median)
        raise ValueError(f"Unsupported baseline mode: {mode}")

    def sample(self, n_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``n_samples`` rows from the empirical pool (for marginal occlusion)."""

        if self.pool.numel() == 0:
            return self.median.unsqueeze(0).repeat(n_samples, 1)
        indices = torch.randint(
            low=0,
            high=self.pool.shape[0],
            size=(n_samples,),
            generator=generator,
        )
        return self.pool[indices].clone()

    def to_json(self) -> dict:
        return {
            "labels": list(self.labels),
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json()
        payload["pool"] = self.pool.tolist()
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DescriptorBaseline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            labels=tuple(payload["labels"]),
            median=torch.tensor(payload["median"], dtype=torch.float32),
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            std=torch.tensor(payload["std"], dtype=torch.float32),
            pool=torch.tensor(payload.get("pool", []), dtype=torch.float32),
        )


def compute_molecule_feature_baseline(
    smiles_list: list[str],
    molecule_feature_names: tuple[str, ...],
    charge_xyz_dir: Path | None = None,
    max_molecules: int | None = None,
) -> DescriptorBaseline:
    if not molecule_feature_names:
        raise ValueError("At least one molecule feature set is required.")

    xyz_index = _build_xyz_index(charge_xyz_dir) if charge_xyz_dir is not None else None
    selected = smiles_list[:max_molecules] if max_molecules else smiles_list

    vectors: list[torch.Tensor] = []
    for smiles in selected:
        vectors.append(
            build_molecule_feature_vector(
                smiles=smiles,
                feature_names=molecule_feature_names,
                charge_xyz_dir=charge_xyz_dir,
                charge_xyz_index=xyz_index,
            )
        )

    pool = torch.stack(vectors, dim=0) if vectors else torch.zeros((0, 0))
    labels = molecule_feature_labels(molecule_feature_names)

    if pool.numel() == 0:
        zeros = torch.zeros(len(labels), dtype=torch.float32)
        return DescriptorBaseline(labels=labels, median=zeros, mean=zeros, std=zeros, pool=pool)

    return DescriptorBaseline(
        labels=labels,
        median=pool.median(dim=0).values,
        mean=pool.mean(dim=0),
        std=pool.std(dim=0),
        pool=pool,
    )


def describe(baseline: DescriptorBaseline) -> str:
    lines = [
        "Descriptor baselines (values are in the scaled units the model consumes)",
        f"  molecules pooled: {baseline.pool.shape[0] if baseline.pool.numel() else 0}",
        "",
        f"  {'descriptor':<24}{'median':>12}{'mean':>12}{'std':>12}{'zero is':>14}",
    ]
    for index, label in enumerate(baseline.labels):
        median = float(baseline.median[index])
        mean = float(baseline.mean[index])
        std = float(baseline.std[index])
        if std > 1e-9:
            z_of_zero = abs(0.0 - mean) / std
            verdict = f"{z_of_zero:.1f} sd away"
        else:
            verdict = "constant"
        lines.append(f"  {label:<24}{median:>12.4f}{mean:>12.4f}{std:>12.4f}{verdict:>14}")
    lines.append("")
    lines.append(
        "  Any descriptor whose zero sits several standard deviations from the mean is a "
        "descriptor whose zero-baseline occlusion score is measuring extrapolation, not importance."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--molecule-features", nargs="+", required=True)
    parser.add_argument("--charge-xyz-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-molecules", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    import pandas as pd  # noqa: PLC0415

    args = parse_args()
    frame = pd.read_csv(args.dataset_path)
    column = "smiles" if "smiles" in frame.columns else frame.columns[0]
    smiles_list = [str(value) for value in frame[column].tolist()]

    baseline = compute_molecule_feature_baseline(
        smiles_list=smiles_list,
        molecule_feature_names=tuple(args.molecule_features),
        charge_xyz_dir=args.charge_xyz_dir,
        max_molecules=args.max_molecules,
    )
    out_dir = ensure_dir(args.output_dir)
    path = baseline.save(out_dir / "descriptor_baseline.json")
    text = describe(baseline)
    (out_dir / "descriptor_baseline.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved baseline to {path}")


if __name__ == "__main__":
    main()
