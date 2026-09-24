"""Build data/charge_type_table.json: per-atom-type mean charge + per-variant z-scoring stats.

    PYTHONPATH=src python scripts/build_charge_type_table.py \
        --dataset-path data/ames_7251.csv --charge-xyz-dir data/xtb_7251

Uses charges only, never labels, so it is safe to compute over the whole dataset.
"""
from __future__ import annotations

import argparse, json, statistics
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from downstream.toxicity.charge_decomposition import atom_type_key, GLOBAL_KEY
from downstream.toxicity.features import (
    _charge_values_for_smiles, _build_xyz_index, _gasteiger_atom_charges,
)

RDLogger.DisableLog("rdApp.*")


def collect(smiles_list, xyz_dir):
    idx = _build_xyz_index(xyz_dir) if xyz_dir else None
    per_type: dict[str, list[float]] = {}
    rows: list[tuple[str, float]] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        vals = (_charge_values_for_smiles(smi, xyz_dir=xyz_dir, xyz_index=idx)
                if xyz_dir else _gasteiger_atom_charges(smi))
        if len(vals) != mol.GetNumAtoms():
            continue
        for atom, v in zip(mol.GetAtoms(), vals):
            if v != v:          # NaN
                continue
            k = atom_type_key(atom)
            per_type.setdefault(k, []).append(float(v))
            rows.append((k, float(v)))
    return per_type, rows


def block(per_type, rows):
    means = {k: statistics.fmean(v) for k, v in per_type.items()}
    allv = [v for _, v in rows]
    means[GLOBAL_KEY] = statistics.fmean(allv)
    typed = [means[k] for k, _ in rows]
    resid = [v - means[k] for k, v in rows]
    def st(x):
        return {"mean": statistics.fmean(x), "std": max(statistics.pstdev(x), 1e-6)}
    ss_tot = sum((v - means[GLOBAL_KEY]) ** 2 for v in allv)
    ss_res = sum(r * r for r in resid)
    return {
        "type_mean": means,
        "stats": {"z": st(allv), "typed": st(typed), "resid": st(resid)},
        "n_atoms": len(allv),
        "n_types": len(per_type) ,
        "r2_atom_type": 1 - ss_res / ss_tot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", type=Path, default=Path("data/ames_7251.csv"))
    ap.add_argument("--charge-xyz-dir", type=Path, default=Path("data/xtb_7251"))
    ap.add_argument("--out", type=Path, default=Path("data/charge_type_table.json"))
    a = ap.parse_args()

    smiles = pd.read_csv(a.dataset_path)["smiles"].tolist()
    table = {}
    for source, xyz in (("gasteiger", None), ("xtb", a.charge_xyz_dir)):
        pt, rows = collect(smiles, xyz)
        table[source] = block(pt, rows)
        b = table[source]
        print(f"{source:10s} atoms={b['n_atoms']:7d}  types={b['n_types']:5d}  "
              f"R^2(atom type)={b['r2_atom_type']:.4f}  "
              f"sd: z={b['stats']['z']['std']:.4f} "
              f"typed={b['stats']['typed']['std']:.4f} "
              f"resid={b['stats']['resid']['std']:.4f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(table))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
