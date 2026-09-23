#!/usr/bin/env python3
"""Build the AMES_xTB dataset tables for public release (Zenodo).

Produces two linked CSV tables from the GFN2-xTB outputs in data/xtb_7251/:

  AMES_xTB_molecules.csv  one row per compound: SMILES, Ames label and global
                          (molecule-level) xTB descriptors
  AMES_xTB_tokens.csv     one row per SMILES character token: local (atom-level)
                          xTB descriptors mapped onto the MolEncoder tokens

Parsing and token mapping are imported from the training code
(src/downstream/toxicity/features.py and xai/common.py), so the token values are
exactly the model inputs before standardisation. Values are not standardised.

Every molecule is checked before anything is written: xTB heavy-atom order and
elements must match the RDKit atoms of the SMILES, the character walk must find
the same number of atoms, and summing token values per atom must give back the
xTB atom value. The script exits non-zero if any check fails.

Usage:  PYTHONPATH=src python scripts/build_ames_xtb_dataset.py [--out AMES_xTB]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import types
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # features.py imports torch at module level; the parsers used here do not need it
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = types.ModuleType("torch")

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from downstream.toxicity.features import (  # noqa: E402
    _build_xyz_index,
    _lookup_out_path_for_smiles,
    _map_atom_values_to_smiles_characters,
    _read_orbital_descriptors,
    _read_xtb_heavy_charges,
    _read_xtb_heavy_polarizabilities,
)
from downstream.toxicity.xai.common import char_to_atom_index  # noqa: E402

PT = Chem.GetPeriodicTable()
TOTAL_ENERGY = re.compile(r"^Total Energy\s*=\s*(\S+)\s*$", re.M)
TOTAL_POL = re.compile(r"^\s*Total Polarizability\s+(\S+)\s*$", re.M)
CALC_CHARGE = re.compile(r"^charge\s*=\s*(-?\d+)\s*$", re.M)
BRACKET_ELEMENT = re.compile(r"^\[\d*([A-Z][a-z]?|[a-z][a-z]?)")


def read_atom_block(text: str) -> list[tuple[int, float, float]]:
    """All atoms (including H) from the >atom;charge;pol ... <atom;charge;pol block."""
    block = text.split(">atom;charge;pol", 1)[1].split("<atom;charge;pol", 1)[0]
    atoms = []
    for line in block.strip().splitlines():
        z, q, p = line.strip().split(";")[:3]
        atoms.append((int(z), float(q), float(p)))
    return atoms


def token_element(chars: str) -> str:
    """Element symbol written in the SMILES for one atom's characters."""
    if chars.startswith("["):
        m = BRACKET_ELEMENT.match(chars)
        return m.group(1).capitalize() if m else "?"
    return chars.capitalize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "AMES_xTB"))
    args = ap.parse_args()

    xyz = ROOT / "data" / "xtb_7251"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ames = pd.read_csv(ROOT / "data" / "ames_7251.csv")
    asg = pd.read_csv(xyz / "ames_out_assignment.csv")
    stem = pd.read_csv(xyz / "smiles_to_out_stem.csv")

    asg = asg[asg.matched_by != "NOT_CALCULATED"]
    df = ames.merge(asg[["Drug_ID", "out_file"]], on="Drug_ID", how="left", validate="1:1")
    assert df.out_file.notna().all(), "compound without xTB output"
    assert df.out_file.is_unique, "two compounds share one xTB output"
    df["out_stem"] = df.out_file.str.removesuffix(".out")
    stem_to_can = dict(zip(stem.out_stem, stem.canonical_smiles))
    df["canonical_smiles"] = df.out_stem.map(stem_to_can)
    assert df.canonical_smiles.notna().all(), "output file missing from smiles_to_out_stem.csv"

    xyz_index = _build_xyz_index(xyz)

    problems: dict[str, list[str]] = defaultdict(list)
    lookup_not_found = 0
    mol_rows, tok_rows = [], []
    n_charge_beyond_2 = 0
    open_shell = 0
    n_charge_state_differs = 0

    for r in df.itertuples(index=False):
        smiles, did = r.smiles, r.Drug_ID
        out_path = xyz / "outs" / r.out_file
        text = out_path.read_text(encoding="utf-8")

        # the lookup the training code performs must land on the same file
        looked_up = _lookup_out_path_for_smiles(smiles, xyz, xyz_index)
        if looked_up is None:
            lookup_not_found += 1  # RDKit-version canonicalisation drift only
        elif looked_up.resolve() != out_path.resolve():
            problems["lookup_points_to_other_file"].append(did)

        atoms = read_atom_block(text)
        heavy = [a for a in atoms if a[0] != 1]
        q = _read_xtb_heavy_charges(out_path)
        p = _read_xtb_heavy_polarizabilities(out_path)
        if q != [a[1] for a in heavy] or p != [a[2] for a in heavy]:
            problems["parser_disagrees_with_atom_block"].append(did)
        n_charge_beyond_2 += sum(abs(v) > 2.0 for v in q)

        if "closed-shell calculation" not in text:
            open_shell += 1

        homo, lumo, gap, _mu, _eta, omega = _read_orbital_descriptors(out_path)
        if homo == 0.0 and lumo == 0.0:
            problems["orbital_parse_failed"].append(did)
        e_tot = TOTAL_ENERGY.findall(text)
        a_tot = TOTAL_POL.findall(text)
        c_tot = CALC_CHARGE.findall(text)
        if len(e_tot) != 1 or len(a_tot) != 1 or len(c_tot) != 1:
            problems["total_energy_polarizability_or_charge_not_unique"].append(did)
            continue

        mol = Chem.MolFromSmiles(smiles)
        n_charge_state_differs += int(c_tot[0]) != Chem.GetFormalCharge(mol)
        rd_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        xtb_symbols = [PT.GetElementSymbol(a[0]) for a in heavy]
        if rd_symbols != xtb_symbols:
            problems["atom_order_or_element_mismatch"].append(did)

        char_atom = char_to_atom_index(smiles)
        q_tok = _map_atom_values_to_smiles_characters(smiles, q)
        p_tok = _map_atom_values_to_smiles_characters(smiles, p)
        if not (len(q_tok) == len(p_tok) == len(char_atom) == len(smiles)):
            problems["token_length_mismatch"].append(did)
            continue
        walk_atoms = sorted({a for a in char_atom if a is not None})
        if walk_atoms != list(range(len(heavy))):
            problems["character_walk_atom_count_mismatch"].append(did)
            continue

        # per-atom checks: element written in SMILES, and token values sum back to xTB value
        chars_of = defaultdict(str)
        q_sum, p_sum = defaultdict(float), defaultdict(float)
        for ch, a, qv, pv in zip(smiles, char_atom, q_tok, p_tok):
            if a is None:
                if qv != 0.0 or pv != 0.0:
                    problems["value_on_non_atom_token"].append(did)
                continue
            chars_of[a] += ch
            q_sum[a] += qv
            p_sum[a] += pv
        for a in walk_atoms:
            if token_element(chars_of[a]) != xtb_symbols[a]:
                problems["smiles_element_mismatch"].append(f"{did}:{a}:{chars_of[a]}")
            if not (math.isclose(q_sum[a], q[a], abs_tol=1e-12)
                    and math.isclose(p_sum[a], p[a], abs_tol=1e-12)):
                problems["token_values_do_not_sum_to_atom"].append(did)

        for i, (ch, a, qv, pv) in enumerate(zip(smiles, char_atom, q_tok, p_tok)):
            tok_rows.append((did, i, ch, a, xtb_symbols[a] if a is not None else None, qv, pv))

        mol_rows.append({
            "Drug_ID": did,
            "smiles": smiles,
            "canonical_smiles": r.canonical_smiles,
            "ames_label": int(r.label),
            "xtb_output_file": r.out_file,
            "xtb_total_charge": int(c_tot[0]),
            "n_heavy_atoms": len(heavy),
            "n_atoms_total": len(atoms),
            "n_tokens": len(smiles),
            "homo_Eh": homo,
            "lumo_Eh": lumo,
            "homo_lumo_gap_Eh": round(gap, 10),
            "electrophilicity_index_Eh": round(omega, 10),
            "total_energy_Eh": float(e_tot[0]),
            "total_polarizability_bohr3": float(a_tot[0]),
        })

    print(f"molecules processed           : {len(mol_rows)} / {len(df)}")
    print(f"token rows                    : {len(tok_rows)}")
    print(f"open-shell calculations       : {open_shell}")
    print(f"xTB charge != SMILES charge  : {n_charge_state_differs}  (protonation state of the calculation differs)")
    print(f"heavy-atom |charge| > 2       : {n_charge_beyond_2}  (model clamped to +-2)")
    print(f"canonical lookup not found    : {lookup_not_found}  (RDKit version drift; file link taken from ames_out_assignment.csv)")
    for k, v in problems.items():
        print(f"PROBLEM {k}: {len(v)}  e.g. {v[:5]}")
    if problems or len(mol_rows) != len(df):
        print("NOT WRITTEN: fix the problems above first")
        return 1

    mols = pd.DataFrame(mol_rows)
    toks = pd.DataFrame(tok_rows, columns=[
        "Drug_ID", "token_index", "token", "atom_index", "element",
        "xtb_charge_e", "xtb_polarizability_bohr3",
    ])
    toks["atom_index"] = toks["atom_index"].astype("Int64")
    mols.to_csv(out_dir / "AMES_xTB_molecules.csv", index=False)
    toks.to_csv(out_dir / "AMES_xTB_tokens.csv", index=False)
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
