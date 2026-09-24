"""Split an atomic partial charge into an atom-type component and a context residual.

Motivation
----------
~86% (Gasteiger) / ~90% (xTB) of the variance in atomic partial charge across
ames_7251.csv is predictable from purely LOCAL atom typing -- element,
aromaticity, degree, formal charge, hybridisation and attached-hydrogen count --
with no long-range or conjugative information at all.

So a model that improves when given partial charges may be exploiting either

  (i)  a sharper, continuous atom-typing signal than the character tokenizer
       provides on its own, or
  (ii) the ~10-14% context-dependent residual, which is where genuine
       electronic structure lives.

These two features separate those hypotheses:

  charge_typed  q_type(a)            -- atom typing only, all context removed
  charge_resid  q(a) - q_type(a)     -- context only, atom typing removed
  charge_z      q(a)                 -- the undecomposed control

All three are z-scored with constants stored in the lookup table so that the
three variants reach the model at identical magnitude; any difference in
downstream accuracy is then attributable to information content, not scale.

The lookup table is built by scripts/build_charge_type_table.py. It is derived
from charges only and never touches the labels, so computing it over the full
dataset introduces no label leakage into the cross-validation folds.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from rdkit import Chem

TABLE_PATH = Path(__file__).resolve().parents[3] / "data" / "charge_type_table.json"
GLOBAL_KEY = "__global__"


def atom_type_key(atom: Chem.Atom) -> str:
    """The purely local descriptor of an atom. No neighbours beyond degree/H count."""
    return "|".join((
        atom.GetSymbol(),
        "a" if atom.GetIsAromatic() else "A",
        str(atom.GetDegree()),
        str(atom.GetFormalCharge()),
        str(atom.GetHybridization()),
        str(atom.GetTotalNumHs()),
    ))


@lru_cache(maxsize=1)
def _table(path: str | None = None) -> dict:
    p = Path(path) if path else TABLE_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"charge type table not found at {p}. Build it first:\n"
            f"  python scripts/build_charge_type_table.py"
        )
    return json.loads(p.read_text())


def decompose_charges(
    smiles: str,
    atom_values: list[float],
    mode: str,
    source: str,
    table_path: str | None = None,
) -> list[float]:
    """mode: 'typed' | 'resid' | 'z'.   source: 'gasteiger' | 'xtb'."""
    if mode not in ("typed", "resid", "z"):
        raise ValueError(f"mode must be typed|resid|z, got {mode!r}")
    tab = _table(table_path)
    if source not in tab:
        raise KeyError(f"no '{source}' block in the charge type table")
    means = tab[source]["type_mean"]
    stats = tab[source]["stats"][mode]
    centre, scale = float(stats["mean"]), float(stats["std"])
    fallback = float(means.get(GLOBAL_KEY, 0.0))

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != len(atom_values):
        # cannot type the atoms; fall back to the undecomposed values
        return [(v - centre) / scale for v in atom_values]

    out: list[float] = []
    for atom, value in zip(mol.GetAtoms(), atom_values):
        typed = float(means.get(atom_type_key(atom), fallback))
        if mode == "typed":
            raw = typed
        elif mode == "resid":
            raw = value - typed
        else:
            raw = value
        out.append((raw - centre) / scale)
    return out
