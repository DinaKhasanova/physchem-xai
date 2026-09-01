from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import SaltRemover
from torch.utils.data import Dataset

from molencoder.model import MolEncoderWrapper
from .features import _build_xyz_index, build_molecule_feature_vector, build_token_feature_matrix


@dataclass(slots=True)
class ToxicityExample:
    smiles: str
    label: int


def clean_smiles_dataframe(raw_smiles: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Clean and canonicalize a dataframe with `smiles` and `label` columns."""
    data = raw_smiles.copy()
    data.dropna(inplace=True)
    indices_to_drop: list[int] = []

    for index, row in data.iterrows():
        smiles = row["smiles"]
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                if verbose:
                    print(f"Failed to parse SMILES: {smiles}")
                indices_to_drop.append(index)
                continue

            Chem.SanitizeMol(mol)
            mol = SaltRemover.SaltRemover().StripMol(mol, dontRemoveEverything=True)
            if len(Chem.GetMolFrags(mol)) > 1:
                if verbose:
                    print(f"Multiple fragments for {smiles}; dropping row.")
                indices_to_drop.append(index)
                continue

            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            if not canonical_smiles or len(canonical_smiles) > 200:
                if verbose:
                    print(f"Invalid canonical SMILES for {smiles}; dropping row.")
                indices_to_drop.append(index)
                continue
            data.at[index, "smiles"] = canonical_smiles
        except Exception as exc:  # pragma: no cover - defensive path
            if verbose:
                print(f"Failed to clean {smiles}: {exc}")
            indices_to_drop.append(index)

    data.drop(indices_to_drop, inplace=True)
    data = data.sort_values(["smiles", "label"])
    duplicates = data.duplicated(subset=["smiles", "label"], keep="first")
    data = data[~duplicates]

    conflicting = data.groupby("smiles")["label"].nunique()
    conflicting_smiles = conflicting[conflicting > 1].index
    if len(conflicting_smiles) > 0:
        data = data[~data["smiles"].isin(conflicting_smiles)]

    return data.reset_index(drop=True)


def load_toxicity_dataframe(csv_path: str | Path, clean: bool = True) -> pd.DataFrame:
    """Load toxicity data from csv and normalize required columns."""
    path = Path(csv_path)
    data = pd.read_csv(path)
    normalized_columns = {column.lower(): column for column in data.columns}

    if "smiles" not in normalized_columns:
        raise ValueError(f"{path} must contain a 'smiles' column.")
    if "label" not in normalized_columns:
        raise ValueError(f"{path} must contain a 'label' column.")

    data = data.rename(
        columns={
            normalized_columns["smiles"]: "smiles",
            normalized_columns["label"]: "label",
        }
    )[["smiles", "label"]]
    data["label"] = data["label"].astype(int)
    return clean_smiles_dataframe(data) if clean else data


def split_toxicity_dataframe(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits from a toxicity dataframe."""
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, validation_ratio > 0, and train+validation < 1.")

    shuffled = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_end = int(len(shuffled) * train_ratio)
    validation_end = int(len(shuffled) * (train_ratio + validation_ratio))
    train = shuffled.iloc[:train_end].reset_index(drop=True)
    validation = shuffled.iloc[train_end:validation_end].reset_index(drop=True)
    test = shuffled.iloc[validation_end:].reset_index(drop=True)
    return train, validation, test


class ToxicityDataset(Dataset[ToxicityExample]):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> ToxicityExample:
        row = self.frame.iloc[index]
        return ToxicityExample(smiles=str(row["smiles"]), label=int(row["label"]))


class ToxicityCollator:
    def __init__(
        self,
        encoder: MolEncoderWrapper,
        feature_names: list[str] | None = None,
        molecule_feature_names: list[str] | None = None,
        token_feature_normalize_mode: str = "per_molecule",
        charge_xyz_dir: str | Path | None = None,
    ) -> None:
        self.encoder = encoder
        self.feature_names = feature_names or []
        self.molecule_feature_names = molecule_feature_names or []
        self.token_feature_normalize_mode = token_feature_normalize_mode
        self.charge_xyz_dir = Path(charge_xyz_dir) if charge_xyz_dir is not None else None
        self.charge_xyz_index = _build_xyz_index(self.charge_xyz_dir) if self.charge_xyz_dir is not None else None

    def __call__(self, batch: Sequence[ToxicityExample]) -> dict[str, torch.Tensor]:
        smiles = [example.smiles for example in batch]
        labels = torch.tensor([example.label for example in batch], dtype=torch.long)
        encoded = self.encoder.tokenize(smiles)
        if self.feature_names:
            encoded_length = int(encoded["input_ids"].shape[1])
            encoded["token_features"] = torch.stack(
                [
                    build_token_feature_matrix(
                        smiles=smiles_item,
                        feature_names=self.feature_names,
                        encoded_length=encoded_length,
                        normalize_mode=self.token_feature_normalize_mode,
                        charge_xyz_dir=self.charge_xyz_dir,
                        charge_xyz_index=self.charge_xyz_index,
                    )
                    for smiles_item in smiles
                ],
                dim=0,
            )
        if self.molecule_feature_names:
            encoded["molecule_features"] = torch.stack(
                [
                    build_molecule_feature_vector(
                        smiles=smiles_item,
                        feature_names=self.molecule_feature_names,
                        charge_xyz_dir=self.charge_xyz_dir,
                        charge_xyz_index=self.charge_xyz_index,
                    )
                    for smiles_item in smiles
                ],
                dim=0,
            )
        encoded["labels"] = labels
        return encoded
