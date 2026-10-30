"""Shared infrastructure for the attribution scripts.

Holds what every attribution method needs and none of the methods themselves:
the run configuration, the split loader, the dataloader factory, and checkpoint
loading for both the current fusion architecture and the legacy CLS classifier.

The attribution methods live outside this module:

* ``scripts/xai/integrated_gradients.py``  -- integrated gradients, used in the manuscript
* ``scripts/xai/gate_decomposition.py``    -- exact decomposition of the gated descriptor term
* ``scripts/xai/count_flips.py``           -- prediction changes relative to the baseline model
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from molencoder.config import MolEncoderConfig
from molencoder.model import MolEncoderWrapper

from .dataset import ToxicityCollator, ToxicityDataset
from .features import molecule_feature_dimension
from .finetune import ToxicityClassifier
from .train import ToxicityLightningModule


@dataclass(slots=True)
class AttributionConfig:
    checkpoint_path: Path
    output_dir: Path
    split_name: str = "test"
    pretrained_model_name: str = "fabikru/MolEncoder"
    model_dir: Path | None = Path("models/molencoder")
    dataset_path: Path | None = None
    batch_size: int = 1
    num_workers: int = 0
    max_sequence_length: int = 256
    token_feature_names: tuple[str, ...] = ()
    molecule_feature_names: tuple[str, ...] = ()
    charge_xyz_dir: Path | None = None
    image_label_filter: int | None = None
    row_indices: tuple[int, ...] = ()
    target_class: int = 1


class LegacyClsToxicityClassifier(nn.Module):
    """Compatibility wrapper for older checkpoints trained with CLS-only input."""

    def __init__(self, encoder: MolEncoderWrapper, num_labels: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_features: torch.Tensor | None = None,
        molecule_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del token_features, molecule_features
        token_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(token_states[:, 0, :])


def load_split_dataframe(output_dir: Path, split_name: str) -> pd.DataFrame:
    split_path = output_dir / "splits" / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Expected existing split file at {split_path}.")
    return pd.read_csv(split_path)


def create_loader(
    encoder: MolEncoderWrapper,
    frame: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    token_feature_names: tuple[str, ...],
    molecule_feature_names: tuple[str, ...],
    charge_xyz_dir: Path | None,
    token_feature_normalize_mode: str = "per_molecule",
):
    collator = ToxicityCollator(
        encoder,
        feature_names=list(token_feature_names),
        molecule_feature_names=list(molecule_feature_names),
        token_feature_normalize_mode=token_feature_normalize_mode,
        charge_xyz_dir=charge_xyz_dir,
    )
    return torch.utils.data.DataLoader(
        ToxicityDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )


def load_lightning_model(config: AttributionConfig) -> ToxicityLightningModule:
    encoder_config = MolEncoderConfig(
        pretrained_model_name=config.pretrained_model_name,
        max_sequence_length=config.max_sequence_length,
        num_labels=2,
        model_cache_dir=config.model_dir,
    )
    encoder = MolEncoderWrapper(encoder_config)
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    state = checkpoint["state_dict"]

    if "model.base_classifier.weight" in state:
        # SmilesAnchorResidualAuxClassifier: confidence-gated residual fusion.
        # Feature dimensions are read back from tensor shapes rather than taken
        # from the CLI, so a mismatch fails loudly instead of scoring wrongly.
        from downstream.toxicity.finetune import SmilesAnchorResidualAuxClassifier

        hidden = int(state["model.base_classifier.weight"].shape[1])
        ckpt_token_dim = 0
        if "model.token_proj.0.weight" in state:
            ckpt_token_dim = (int(state["model.token_proj.0.weight"].shape[0]) - hidden) // 4
        ckpt_molecule_dim = 0
        if "model.molecule_proj.1.weight" in state:
            ckpt_molecule_dim = int(state["model.molecule_proj.1.weight"].shape[1])

        want_token_dim = len(config.token_feature_names)
        want_molecule_dim = molecule_feature_dimension(config.molecule_feature_names)
        if ckpt_token_dim != want_token_dim:
            raise ValueError(
                f"{config.checkpoint_path}: checkpoint token_feature_dim="
                f"{ckpt_token_dim} but --token-features implies {want_token_dim}"
            )
        if ckpt_molecule_dim != want_molecule_dim:
            raise ValueError(
                f"{config.checkpoint_path}: checkpoint molecule_feature_dim="
                f"{ckpt_molecule_dim} but --molecule-features implies {want_molecule_dim}"
            )

        classifier = SmilesAnchorResidualAuxClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=ckpt_token_dim,
            molecule_feature_dim=ckpt_molecule_dim,
            weight_mode=getattr(config, "token_weight_mode", "abs"),
        )
    else:
        classifier_input_size = int(state["model.classifier.0.weight"].shape[0])
        if classifier_input_size == encoder.hidden_size:
            classifier = LegacyClsToxicityClassifier(encoder=encoder, num_labels=2)
        else:
            classifier = ToxicityClassifier(
                encoder=encoder,
                num_labels=2,
                token_feature_dim=len(config.token_feature_names),
                molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
            )
    lightning_model = ToxicityLightningModule(model=classifier, learning_rate=2e-5, weight_decay=1e-2)
    lightning_model.load_state_dict(checkpoint["state_dict"], strict=True)
    lightning_model.eval()
    return lightning_model
