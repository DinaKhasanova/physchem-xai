"""Are the "semiempirical" features really semiempirical, and are they on the right atoms?

``features._charge_values_for_smiles`` and ``_polarizability_values_for_smiles``
fall back **silently**:

* charge          -> RDKit Gasteiger charges, if no xTB ``.out`` is found
* polarizability  -> Crippen molar-refractivity contributions, if no ``.out`` is found
* orbital         -> a vector of six zeros, if no ``.out`` is found

so a run labelled "SMILES + semiempirical charges" can quietly be a Gasteiger
run for part or all of the dataset, and an "orbital descriptors" run can be
feeding the classifier constant zeros. Nothing in the training logs would say so.

A second, subtler failure: the ``.out`` file lists heavy atoms in the order of
the 3D structure that was generated from the SMILES, while the token mapping
assumes that order equals RDKit's atom order for the canonical SMILES. If the
two orders differ, every atomic value is attached to the wrong atom, which
would look exactly like "atomic charges do not help".

This module checks all of that and writes a per-molecule report.

Usage
-----
    python -m downstream.toxicity.xai.coverage \
        --dataset-path data/ames_7251.csv \
        --charge-xyz-dir protonation_federico_corina_7246_xyz \
        --output-dir runs_cv/diagnostics
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from ..features import (
    _build_xyz_index,
    _lookup_out_path_for_smiles,
    _read_orbital_descriptors,
    _read_xtb_heavy_charges,
    _read_xtb_heavy_polarizabilities,
)
from .common import ensure_dir, parsed_atom_count


def _read_xtb_heavy_atomic_numbers(out_path: Path) -> list[int]:
    """Atomic numbers of the heavy atoms, in the order the ``.out`` file lists them."""

    with out_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    start = end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(">atom;charge;pol"):
            start = i + 1
        if stripped.startswith("<atom;charge;pol"):
            end = i
            break

    if start is None or end is None or end <= start:
        return []

    numbers: list[int] = []
    for line in lines[start:end]:
        parts = line.strip().split(";")
        if len(parts) < 2:
            continue
        try:
            atomic_number = int(parts[0])
        except ValueError:
            continue
        if atomic_number != 1:
            numbers.append(atomic_number)
    return numbers


def audit_dataset(
    smiles_list: list[str],
    charge_xyz_dir: Path,
) -> pd.DataFrame:
    from rdkit import Chem  # noqa: PLC0415

    xyz_index = _build_xyz_index(charge_xyz_dir)
    rows: list[dict[str, object]] = []

    for smiles in smiles_list:
        row: dict[str, object] = {"smiles": smiles}
        mol = Chem.MolFromSmiles(smiles)
        row["rdkit_parsed"] = mol is not None
        if mol is None:
            rows.append(row)
            continue

        heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        row["rdkit_heavy_atoms"] = len(heavy_atoms)
        row["walk_atoms"] = parsed_atom_count(smiles)
        row["walk_matches_rdkit"] = row["walk_atoms"] == mol.GetNumAtoms()

        out_path = _lookup_out_path_for_smiles(smiles, charge_xyz_dir, xyz_index)
        has_out = out_path is not None and out_path.exists()
        row["out_found"] = has_out
        row["out_path"] = str(out_path) if out_path is not None else ""

        if not has_out:
            row["charge_source"] = "gasteiger_fallback"
            row["polarizability_source"] = "crippen_mr_fallback"
            row["orbital_source"] = "zeros_fallback"
            rows.append(row)
            continue

        charges = _read_xtb_heavy_charges(out_path)
        polarizabilities = _read_xtb_heavy_polarizabilities(out_path)
        orbitals = _read_orbital_descriptors(out_path)
        xtb_numbers = _read_xtb_heavy_atomic_numbers(out_path)

        row["charge_source"] = "xtb" if charges else "gasteiger_fallback"
        row["polarizability_source"] = "xtb" if polarizabilities else "crippen_mr_fallback"
        row["orbital_source"] = "xtb" if any(abs(value) > 0 for value in orbitals) else "zeros_fallback"
        row["n_xtb_heavy_atoms"] = len(charges)
        row["heavy_count_matches"] = len(charges) == len(heavy_atoms)

        rdkit_numbers = [atom.GetAtomicNum() for atom in heavy_atoms]
        if xtb_numbers and len(xtb_numbers) == len(rdkit_numbers):
            row["element_order_matches"] = xtb_numbers == rdkit_numbers
            row["n_element_mismatches"] = sum(
                1 for a, b in zip(xtb_numbers, rdkit_numbers) if a != b
            )
        else:
            row["element_order_matches"] = None
            row["n_element_mismatches"] = None

        row["homo"] = orbitals[0]
        row["lumo"] = orbitals[1]
        row["gap"] = orbitals[2]
        rows.append(row)

    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> str:
    total = len(frame)
    lines = [f"Molecules audited: {total}", ""]

    for column, label in (
        ("charge_source", "charge"),
        ("polarizability_source", "polarizability"),
        ("orbital_source", "orbital"),
    ):
        if column not in frame:
            continue
        counts = frame[column].value_counts(dropna=False)
        lines.append(f"{label} feature source:")
        for value, count in counts.items():
            lines.append(f"  {str(value):<26}{count:>7}  ({count / total:.1%})")
        lines.append("")

    if "heavy_count_matches" in frame:
        checked = frame["heavy_count_matches"].notna().sum()
        bad = int((frame["heavy_count_matches"] == False).sum())  # noqa: E712
        lines.append(f"heavy-atom count mismatch (xTB vs RDKit): {bad} of {checked}")

    if "element_order_matches" in frame:
        checked = int(frame["element_order_matches"].notna().sum())
        bad = int((frame["element_order_matches"] == False).sum())  # noqa: E712
        lines.append(f"element-order mismatch (xTB vs RDKit)   : {bad} of {checked}")
        if checked and bad / max(checked, 1) > 0.01:
            lines.append(
                "  ==> atom ordering disagrees for a non-trivial fraction of molecules. "
                "Every per-atom xTB value for those molecules is attached to the wrong atom, "
                "which alone can explain why atomic charges did not improve the model."
            )

    if "walk_matches_rdkit" in frame:
        bad = int((frame["walk_matches_rdkit"] == False).sum())  # noqa: E712
        lines.append(f"SMILES-walk vs RDKit atom count mismatch: {bad} of {total}")

    if "gap" in frame:
        gaps = frame["gap"].dropna()
        finite = [value for value in gaps if isinstance(value, (int, float)) and math.isfinite(value)]
        zeros = sum(1 for value in finite if value == 0.0)
        if finite:
            lines.append("")
            lines.append(
                f"HOMO-LUMO gap: {zeros} of {len(finite)} molecules have gap exactly 0.0 "
                "(these are the zeros-fallback rows feeding the classifier)"
            )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, required=True, help="CSV with a 'smiles' column")
    parser.add_argument("--charge-xyz-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Audit only the first N molecules")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.dataset_path)
    column = "smiles" if "smiles" in frame.columns else frame.columns[0]
    smiles_list = [str(value) for value in frame[column].tolist()]
    if args.limit:
        smiles_list = smiles_list[: args.limit]

    report = audit_dataset(smiles_list, args.charge_xyz_dir)
    out_dir = ensure_dir(args.output_dir)
    path = out_dir / "feature_source_coverage.csv"
    report.to_csv(path, index=False)

    text = summarize(report)
    (out_dir / "feature_source_coverage.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved per-molecule report to {path}")


if __name__ == "__main__":
    main()
