from __future__ import annotations
import os

import csv
import math
from collections.abc import Iterable
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors


UNSUPPORTED_TOKEN_CHARS = set("#()+-/12345678=@[]\\")


def _map_atom_values_to_smiles_characters(smiles: str, atom_values: list[float]) -> list[float]:
    values: list[float] = []
    atom_index = 0
    token_index = 0

    while token_index < len(smiles):
        token = smiles[token_index]

        if token == "[":
            end = smiles.find("]", token_index)
            if end == -1:
                values.append(0.0)
                token_index += 1
                continue

            value = atom_values[atom_index] if atom_index < len(atom_values) else 0.0
            atom_index += 1
            block = [0.0] * (end - token_index + 1)

            atom_pos = None
            for j in range(token_index + 1, end):
                if smiles[j].isalpha():
                    atom_pos = j
                    break
            if atom_pos is not None:
                block[atom_pos - token_index] = value

            values.extend(block)
            token_index = end + 1
            continue

        if token == "C" and token_index + 1 < len(smiles) and smiles[token_index + 1] == "l":
            value = atom_values[atom_index] if atom_index < len(atom_values) else 0.0
            values.extend([value / 2.0, value / 2.0])
            atom_index += 1
            token_index += 2
            continue

        if token == "B" and token_index + 1 < len(smiles) and smiles[token_index + 1] == "r":
            value = atom_values[atom_index] if atom_index < len(atom_values) else 0.0
            values.extend([value / 2.0, value / 2.0])
            atom_index += 1
            token_index += 2
            continue

        if token in UNSUPPORTED_TOKEN_CHARS or not token.isalpha():
            values.append(0.0)
        else:
            value = atom_values[atom_index] if atom_index < len(atom_values) else 0.0
            values.append(value)
            atom_index += 1
        token_index += 1

    return values


def _gasteiger_atom_charges(smiles: str) -> list[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    AllChem.ComputeGasteigerCharges(mol)
    charges: list[float] = []
    for atom in mol.GetAtoms():
        charge = float(atom.GetProp("_GasteigerCharge"))
        charges.append(charge if math.isfinite(charge) else 0.0)
    return charges


def _mr_atom_contributions(smiles: str) -> list[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    contributions = rdMolDescriptors._CalcCrippenContribs(mol)
    values: list[float] = []
    for _, mr in contributions:
        value = float(mr)
        values.append(value if math.isfinite(value) else 0.0)
    return values


def _heavy_neighbor_counts(smiles: str, distance_cutoff: int) -> list[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    distance_matrix = Chem.GetDistanceMatrix(mol)
    heavy_mask = [atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()]
    counts: list[float] = []
    for i in range(mol.GetNumAtoms()):
        count = 0
        for j in range(mol.GetNumAtoms()):
            if i == j or not heavy_mask[j]:
                continue
            if distance_matrix[i, j] <= distance_cutoff:
                count += 1
        counts.append(float(count))
    return counts


def _charge_variant_builder(mode: str):
    """Fallback builder (Gasteiger source). build_token_feature_matrix overrides
    this with the xTB source when --charge-xyz-dir is supplied."""
    def _build(smiles: str) -> list[float]:
        from downstream.toxicity.charge_decomposition import decompose_charges
        return decompose_charges(smiles, _gasteiger_atom_charges(smiles),
                                 mode=mode, source="gasteiger")
    return _build


CHARGE_VARIANTS = {"charge_z": "z", "charge_typed": "typed", "charge_resid": "resid"}

FEATURE_BUILDERS = {
    "charge": _gasteiger_atom_charges,
    "charge_z": _charge_variant_builder("z"),
    "charge_typed": _charge_variant_builder("typed"),
    "charge_resid": _charge_variant_builder("resid"),
    "mr": _mr_atom_contributions,
    "polarizability": _mr_atom_contributions,
    "neighbor": lambda smiles: _heavy_neighbor_counts(smiles, distance_cutoff=3),
    "neighbor_k2": lambda smiles: _heavy_neighbor_counts(smiles, distance_cutoff=2),
    "neighbor_k3": lambda smiles: _heavy_neighbor_counts(smiles, distance_cutoff=3),
    "neighbor_k4": lambda smiles: _heavy_neighbor_counts(smiles, distance_cutoff=4),
    "orbital_homo": (lambda smiles, _p=0: _broadcast_molecule_value(smiles, "orbital", _p)),
    "orbital_lumo": (lambda smiles, _p=1: _broadcast_molecule_value(smiles, "orbital", _p)),
    "orbital_electrophilicity": (lambda smiles, _p=2: _broadcast_molecule_value(smiles, "orbital", _p)),
    "physchem_molwt": (lambda smiles, _p=0: _broadcast_molecule_value(smiles, "physchem", _p)),
    "physchem_tpsa": (lambda smiles, _p=1: _broadcast_molecule_value(smiles, "physchem", _p)),
    "physchem_logp": (lambda smiles, _p=2: _broadcast_molecule_value(smiles, "physchem", _p)),
    "physchem_molmr": (lambda smiles, _p=3: _broadcast_molecule_value(smiles, "physchem", _p)),
    "physchem_hbd": (lambda smiles, _p=4: _broadcast_molecule_value(smiles, "physchem", _p)),
    "physchem_hba": (lambda smiles, _p=5: _broadcast_molecule_value(smiles, "physchem", _p)),
    "physchem_rotatablebonds": (lambda smiles, _p=6: _broadcast_molecule_value(smiles, "physchem", _p)),
    "physchem_ringcount": (lambda smiles, _p=7: _broadcast_molecule_value(smiles, "physchem", _p)),
    "physchem_heavyatomcount": (lambda smiles, _p=8: _broadcast_molecule_value(smiles, "physchem", _p)),
    "physchem_fractioncsp3": (lambda smiles, _p=9: _broadcast_molecule_value(smiles, "physchem", _p)),
}

MOLECULE_FEATURE_BUILDERS = {
    "physchem": lambda smiles: build_physchem_descriptor_vector(smiles),
    "orbital": lambda smiles: [0.0] * 3,
}

MOLECULE_FEATURE_DIMS = {
    "physchem": 10,
    "orbital": 3,
}

MOLECULE_FEATURE_LABELS = {
    "physchem": (
        "MolWt",
        "TPSA",
        "LogP",
        "MolMR",
        "HBD",
        "HBA",
        "RotatableBonds",
        "RingCount",
        "HeavyAtomCount",
        "FractionCSP3",
    ),
    "orbital": (
        "HOMO",
        "LUMO",
        "electrophilicity_index",
    ),
}


def available_feature_names() -> tuple[str, ...]:
    return tuple(FEATURE_BUILDERS.keys())


def available_molecule_feature_names() -> tuple[str, ...]:
    return tuple(MOLECULE_FEATURE_BUILDERS.keys())


def molecule_feature_dimension(feature_names: Iterable[str]) -> int:
    return sum(MOLECULE_FEATURE_DIMS[name] for name in feature_names)


def molecule_feature_labels(feature_names: Iterable[str]) -> tuple[str, ...]:
    labels: list[str] = []
    for name in feature_names:
        if name not in MOLECULE_FEATURE_LABELS:
            raise ValueError(
                f"Unsupported molecule feature '{name}'. Choices: {', '.join(available_molecule_feature_names())}"
            )
        labels.extend(MOLECULE_FEATURE_LABELS[name])
    return tuple(labels)


def build_physchem_descriptor_vector(smiles: str) -> list[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0] * 10

    values = [
        Descriptors.MolWt(mol) / 1000.0,
        rdMolDescriptors.CalcTPSA(mol) / 250.0,
        Crippen.MolLogP(mol) / 10.0,
        Crippen.MolMR(mol) / 200.0,
        Lipinski.NumHDonors(mol) / 10.0,
        Lipinski.NumHAcceptors(mol) / 20.0,
        Lipinski.NumRotatableBonds(mol) / 20.0,
        Lipinski.RingCount(mol) / 10.0,
        Lipinski.HeavyAtomCount(mol) / 100.0,
        Lipinski.FractionCSP3(mol),
    ]
    return [float(value) if math.isfinite(value) else 0.0 for value in values]


def _read_orbital_descriptors(out_path: Path) -> list[float]:
    with out_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    occupations: list[float] | None = None
    energies: list[float] | None = None
    for i, line in enumerate(lines):
        if line.strip().startswith("occupation vector"):
            if i + 1 < len(lines):
                occupations = [float(value) for value in lines[i + 1].split()]
        if line.strip().startswith("Optimized orbital energies:"):
            _, values = line.split(":", maxsplit=1)
            energies = [float(value) for value in values.split()]
            break

    if not occupations or not energies:
        return [0.0] * 6

    n = min(len(occupations), len(energies))
    homo_index = None
    lumo_index = None
    for index in range(n):
        if occupations[index] > 0:
            homo_index = index
        elif lumo_index is None:
            lumo_index = index
            break

    if homo_index is None or lumo_index is None:
        return [0.0] * 6

    homo = energies[homo_index]
    lumo = energies[lumo_index]
    gap = lumo - homo
    mu = (homo + lumo) / 2.0
    eta = gap
    omega = (mu * mu) / (2.0 * eta) if eta > 1e-8 else 0.0
    values = [homo, lumo, gap, mu, eta, omega]
    return [float(value) if math.isfinite(value) else 0.0 for value in values]


def _orbital_descriptors_for_smiles(
    smiles: str,
    xyz_dir: Path | None = None,
    xyz_index: dict[str, Path] | None = None,
) -> list[float]:
    if xyz_dir is None or xyz_index is None:
        return [0.0] * 6

    out_path = _lookup_out_path_for_smiles(smiles, xyz_dir, xyz_index)
    if out_path is None:
        return [0.0] * 6

    if not out_path.exists():
        return [0.0] * 3
    values = _read_orbital_descriptors(out_path)
    if len(values) < 6:
        return [0.0] * 3
    # The six conceptual-DFT quantities have rank 3: gap = LUMO - HOMO,
    # chemical potential = (HOMO + LUMO)/2, and hardness = gap/2 are all
    # algebraic restatements of HOMO and LUMO (gap and hardness are in fact
    # numerically identical here). Keeping the three independent quantities
    # avoids double-counting them in the projection.
    return [values[0], values[1], values[5]]  # HOMO, LUMO, electrophilicity


def _canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _read_smiles_from_xyz(xyz_path: Path) -> str:
    with xyz_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    return lines[1].strip()


def _xyz_to_out_path(xyz_path: Path, xyz_dir: Path) -> Path:
    return xyz_dir / "outs" / f"{xyz_path.stem}.out"


def _lookup_out_path_for_smiles(
    smiles: str,
    xyz_dir: Path,
    xyz_index: dict[str, Path],
) -> Path | None:
    key = _canonical_smiles(smiles)
    indexed_path = xyz_index.get(key) if key is not None else None
    if indexed_path is None:
        return None
    if indexed_path.suffix == ".out":
        return indexed_path
    return _xyz_to_out_path(indexed_path, xyz_dir)


def _read_xtb_heavy_charges(out_path: Path) -> list[float]:
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

    charges: list[float] = []
    for line in lines[start:end]:
        parts = line.strip().split(";")
        if len(parts) < 2:
            continue
        atomic_number = int(parts[0])
        charge = float(parts[1])
        if atomic_number != 1:
            charges.append(charge if math.isfinite(charge) else 0.0)
    return charges


def _read_xtb_heavy_polarizabilities(out_path: Path) -> list[float]:
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

    polarizabilities: list[float] = []
    for line in lines[start:end]:
        parts = line.strip().split(";")
        if len(parts) < 3:
            continue
        atomic_number = int(parts[0])
        polarizability = float(parts[2])
        if atomic_number != 1:
            polarizabilities.append(polarizability if math.isfinite(polarizability) else 0.0)
    return polarizabilities


def _build_xyz_index(xyz_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not xyz_dir.exists():
        return index

    precomputed_index_path = xyz_dir / "smiles_to_out_stem.csv"
    if precomputed_index_path.exists():
        with precomputed_index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                smiles = row.get("canonical_smiles")
                stem = row.get("out_stem")
                if not smiles or not stem:
                    continue
                index[smiles] = xyz_dir / "outs" / f"{stem}.out"
        return index

    for xyz_path in xyz_dir.glob("*.xyz"):
        key = _canonical_smiles(_read_smiles_from_xyz(xyz_path))
        if key is not None:
            index[key] = xyz_path
    return index


def _charge_values_for_smiles(
    smiles: str,
    xyz_dir: Path | None = None,
    xyz_index: dict[str, Path] | None = None,
) -> list[float]:
    if xyz_dir is not None and xyz_index is not None:
        out_path = _lookup_out_path_for_smiles(smiles, xyz_dir, xyz_index)
        if out_path is not None and out_path.exists():
            xtb_charges = _read_xtb_heavy_charges(out_path)
            if xtb_charges:
                return xtb_charges
    return _gasteiger_atom_charges(smiles)


def _polarizability_values_for_smiles(
    smiles: str,
    xyz_dir: Path | None = None,
    xyz_index: dict[str, Path] | None = None,
) -> list[float]:
    if xyz_dir is not None and xyz_index is not None:
        out_path = _lookup_out_path_for_smiles(smiles, xyz_dir, xyz_index)
        if out_path is not None and out_path.exists():
            xtb_polarizabilities = _read_xtb_heavy_polarizabilities(out_path)
            if xtb_polarizabilities:
                return xtb_polarizabilities
    return _mr_atom_contributions(smiles)


TOKEN_FEATURE_SCALES = {
    "charge": 0.120,
    "charge_z": 1.0,
    "charge_typed": 1.0,
    "charge_resid": 1.0,
    "polarizability": 0.929,
    "neighbor_k2": 0.424,
    "neighbor_k3": 0.456,
    "neighbor_k4": 0.483,
    "orbital_homo": 1.0,
    "orbital_lumo": 1.0,
    "orbital_electrophilicity": 1.0,
    "physchem_molwt": 1.0,
    "physchem_tpsa": 1.0,
    "physchem_logp": 1.0,
    "physchem_molmr": 1.0,
    "physchem_hbd": 1.0,
    "physchem_hba": 1.0,
    "physchem_rotatablebonds": 1.0,
    "physchem_ringcount": 1.0,
    "physchem_heavyatomcount": 1.0,
    "physchem_fractioncsp3": 1.0,
}

TOKEN_FEATURE_MEANS = {
    "charge": -0.041,
    "charge_z": 0.0,
    "charge_typed": 0.0,
    "charge_resid": 0.0,
    "polarizability": 3.242,
    "neighbor_k2": 1.659,
    "neighbor_k3": 2.050,
    "neighbor_k4": 2.286,
    "orbital_homo": 0.0,
    "orbital_lumo": 0.0,
    "orbital_electrophilicity": 0.0,
    "physchem_molwt": 0.0,
    "physchem_tpsa": 0.0,
    "physchem_logp": 0.0,
    "physchem_molmr": 0.0,
    "physchem_hbd": 0.0,
    "physchem_hba": 0.0,
    "physchem_rotatablebonds": 0.0,
    "physchem_ringcount": 0.0,
    "physchem_heavyatomcount": 0.0,
    "physchem_fractioncsp3": 0.0,
}


# ---------------------------------------------------------------------------
# Broadcast (molecule-level) descriptors exposed as TOKEN features.
#
# A molecule-level quantity such as HOMO has no per-atom value, so the same
# number is written to every atom of the molecule. This routes global
# descriptors through the same late-fusion pathway the per-atom descriptors
# use (token_proj), making the two directly comparable: identical mechanism,
# only local-vs-global chemistry differs.
#
# Note the attention component of that pathway
# (feature_strength = token_features.abs().sum(-1), normalised over tokens)
# is necessarily uniform for a broadcast descriptor -- there is no atom to
# single out. The descriptor reaches the model through the feature_stats
# component instead. That is the correct behaviour for a global quantity, not
# a limitation of the encoding.
#
# Values are standardised with the same dataset constants used for the
# molecule-level path (MOLECULE_FEATURE_STATS), so the two encodings of the
# same descriptor carry identical numbers.
def _broadcast_molecule_value(smiles, block, position, xyz_dir=None, xyz_index=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    if block == "orbital":
        raw = _standardise_molecule_values(
            "orbital",
            _orbital_descriptors_for_smiles(
                smiles=smiles, xyz_dir=xyz_dir, xyz_index=xyz_index
            ),
        )
    else:
        raw = _standardise_molecule_values(
            block, list(MOLECULE_FEATURE_BUILDERS[block](smiles))
        )
    if position >= len(raw):
        return [0.0] * mol.GetNumAtoms()
    return [float(raw[position])] * mol.GetNumAtoms()


def build_token_feature_matrix(
    smiles: str,
    feature_names: Iterable[str],
    encoded_length: int,
    normalize_mode: str = "per_molecule",
    charge_xyz_dir: str | Path | None = None,
    charge_xyz_index: dict[str, Path] | None = None,
) -> torch.Tensor:
    """Build optional token-level physicochemical features.

    MolEncoder uses a character-level tokenizer, so we align atom-derived values to
    SMILES characters and reserve zeros for special and padding tokens.
    """

    names = list(feature_names)
    if not names:
        return torch.zeros((encoded_length, 0), dtype=torch.float32)

    xyz_dir = Path(charge_xyz_dir) if charge_xyz_dir is not None else None
    xyz_index = charge_xyz_index
    if xyz_dir is not None and xyz_index is None:
        xyz_index = _build_xyz_index(xyz_dir)

    per_feature_columns: list[list[float]] = []
    for feature_name in names:
        if feature_name not in FEATURE_BUILDERS:
            raise ValueError(
                f"Unsupported feature '{feature_name}'. Choices: {', '.join(available_feature_names())}"
            )

        if feature_name == "charge" or feature_name in CHARGE_VARIANTS:
            atom_values = _charge_values_for_smiles(
                smiles=smiles,
                xyz_dir=xyz_dir,
                xyz_index=xyz_index,
            )
            if feature_name in CHARGE_VARIANTS:
                from downstream.toxicity.charge_decomposition import decompose_charges
                atom_values = decompose_charges(
                    smiles, atom_values,
                    mode=CHARGE_VARIANTS[feature_name],
                    source="xtb" if xyz_dir is not None else "gasteiger",
                )
        elif feature_name == "polarizability":
            atom_values = _polarizability_values_for_smiles(
                smiles=smiles,
                xyz_dir=xyz_dir,
                xyz_index=xyz_index,
            )
        elif feature_name.startswith("orbital_"):
            atom_values = _broadcast_molecule_value(
                smiles,
                "orbital",
                ("orbital_homo", "orbital_lumo",
                 "orbital_electrophilicity").index(feature_name),
                xyz_dir=xyz_dir,
                xyz_index=xyz_index,
            )
        else:
            atom_values = FEATURE_BUILDERS[feature_name](smiles)
        token_values = _map_atom_values_to_smiles_characters(smiles, atom_values)
        token_values = [0.0] + token_values + [0.0]
        if len(token_values) < encoded_length:
            token_values.extend([0.0] * (encoded_length - len(token_values)))
        else:
            token_values = token_values[:encoded_length]
        per_feature_columns.append(token_values)

    matrix = torch.tensor(per_feature_columns, dtype=torch.float32).transpose(0, 1)
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    for feature_index, feature_name in enumerate(names):
        if feature_name == "charge":
            matrix[:, feature_index] = torch.clamp(matrix[:, feature_index], -2.0, 2.0)
        if feature_name in CHARGE_VARIANTS:
            matrix[:, feature_index] = torch.clamp(matrix[:, feature_index], -6.0, 6.0)
        if feature_name.startswith("neighbor"):
            matrix[:, feature_index] = torch.log1p(matrix[:, feature_index].clamp(min=0.0))
    if normalize_mode == "per_molecule":
        mean = matrix.mean(dim=0, keepdim=True)
        std = matrix.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (matrix - mean) / std
    if normalize_mode == "none":
        return matrix
    if normalize_mode == "dataset":
        # Fixed per-feature scales measured over the Ames training data, so every
        # descriptor reaches `feature_strength = token_features.abs().sum(-1)`
        # with comparable magnitude. Without this, molar refractivity (mean 3.24)
        # and neighborhood counts (1.7-2.3) swamp partial charges (mean |q| 0.041)
        # by ~50-80x and alone determine which atoms receive attention weight.
        # Division only: signs and within-molecule structure are preserved.
        for feature_index, feature_name in enumerate(names):
            scale = TOKEN_FEATURE_SCALES.get(feature_name, 1.0)
            centre = TOKEN_FEATURE_MEANS.get(feature_name, 0.0)
            column = matrix[:, feature_index]
            padding = column == 0.0
            standardised = (column - centre) / scale
            standardised[padding] = 0.0
            matrix[:, feature_index] = standardised
        return matrix
    raise ValueError(f"Unsupported token feature normalize mode: {normalize_mode}")


# Dataset-level mean/sd per molecule-level descriptor, measured over all 7,251
# molecules of ames_7251.csv. Applied in build_molecule_feature_vector so that
# each descriptor arrives on a comparable scale ACROSS molecules.
#
# This replaces the per-molecule nn.LayerNorm that used to sit at the front of
# molecule_proj. That LayerNorm normalised each molecule's descriptor vector
# against itself, which destroys the absolute level -- and the level is the
# signal. Measured on 400 random molecules, the correlation between the raw
# value and the LayerNorm output across molecules was 0.215 for HOMO and
# -0.380 for chemical potential: not merely lost but inverted.
# Statistics are dataset-specific: each dataset is standardised against its own
# distribution, so the same procedure is applied to every experiment. Select
# with the PHYSCHEM_STATS environment variable; "ames" is the default and is
# what every runs_7251/ model was trained with. Do not change the "ames" values,
# or those checkpoints can no longer be reproduced and any attribution computed
# from them would use a different standardisation than the model saw in training.
_MOLECULE_FEATURE_STATS_BY_DATASET = {
    # measured over all 7,251 molecules of data/ames_7251.csv
    "ames": {
        "orbital": {
            "mean": [-0.38236, -0.26563, 0.65388],
            "sd":   [0.02813, 0.06543, 0.56662],
        },
        "physchem": {
            "mean": [0.24272, 0.22176, 0.25086, 0.32736, 0.10650,
                     0.15709, 0.13977, 0.20876, 0.16752, 0.34368],
            "sd":   [0.10807, 0.16827, 0.20065, 0.14731, 0.13332,
                     0.11347, 0.16466, 0.16131, 0.07693, 0.33098],
        },
    },
    # measured over all 18,754 molecules of
    # data/pesticides_genotoxicity_ames_format.csv
    "pesticide": {
        # no xTB calculations for this dataset; orbital stats are not defined
        "physchem": {
            "mean": [0.27764, 0.24716, 0.26911, 0.36869, 0.11777,
                     0.17985, 0.19475, 0.20912, 0.19079, 0.39947],
            "sd":   [0.14425, 0.20631, 0.22524, 0.19109, 0.15753,
                     0.14158, 0.21488, 0.17269, 0.10133, 0.32488],
        },
    },
}

_PHYSCHEM_STATS_DATASET = os.environ.get("PHYSCHEM_STATS", "ames")
if _PHYSCHEM_STATS_DATASET not in _MOLECULE_FEATURE_STATS_BY_DATASET:
    raise ValueError(
        f"PHYSCHEM_STATS={_PHYSCHEM_STATS_DATASET!r} is not one of "
        f"{sorted(_MOLECULE_FEATURE_STATS_BY_DATASET)}"
    )
MOLECULE_FEATURE_STATS = _MOLECULE_FEATURE_STATS_BY_DATASET[_PHYSCHEM_STATS_DATASET]


def _standardise_molecule_values(feature_name, values):
    stats = MOLECULE_FEATURE_STATS.get(feature_name)
    if stats is None or len(values) != len(stats["mean"]):
        return values
    return [
        (v - m) / (sd if sd > 1e-8 else 1.0)
        for v, m, sd in zip(values, stats["mean"], stats["sd"])
    ]


def build_molecule_feature_vector(
    smiles: str,
    feature_names: Iterable[str],
    charge_xyz_dir: str | Path | None = None,
    charge_xyz_index: dict[str, Path] | None = None,
) -> torch.Tensor:
    names = list(feature_names)
    if not names:
        return torch.zeros((0,), dtype=torch.float32)

    xyz_dir = Path(charge_xyz_dir) if charge_xyz_dir is not None else None
    xyz_index = charge_xyz_index
    if xyz_dir is not None and xyz_index is None:
        xyz_index = _build_xyz_index(xyz_dir)

    values: list[float] = []
    for feature_name in names:
        if feature_name not in MOLECULE_FEATURE_BUILDERS:
            raise ValueError(
                f"Unsupported molecule feature '{feature_name}'. Choices: {', '.join(available_molecule_feature_names())}"
            )
        if feature_name == "orbital":
            values.extend(
                _standardise_molecule_values(
                    "orbital",
                    _orbital_descriptors_for_smiles(
                        smiles=smiles,
                        xyz_dir=xyz_dir,
                        xyz_index=xyz_index,
                    ),
                )
            )
        else:
            values.extend(
                _standardise_molecule_values(
                    feature_name,
                    list(MOLECULE_FEATURE_BUILDERS[feature_name](smiles)),
                )
            )

    vector = torch.tensor(values, dtype=torch.float32)
    return torch.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
