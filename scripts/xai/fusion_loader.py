"""Load SmilesAnchorResidualAuxClassifier checkpoints for inference.

``compute_occlusion.load_lightning_model`` predates the fusion architecture: it
probes ``model.classifier.0.weight``, which does not exist in a
``SmilesAnchorResidualAuxClassifier`` checkpoint (that has ``base_proj`` /
``base_classifier`` / ``aux_proj`` / ``aux_classifier`` / ``gate_mlp``).

This module reconstructs the right class. Feature dimensions are read back from
the checkpoint's own tensor shapes rather than trusted from CLI flags, so a
mismatch between what you pass and what was trained is caught immediately
instead of silently producing wrong probabilities.

``hyper_parameters`` is empty in these checkpoints, hence the shape inference.
"""
from __future__ import annotations

from pathlib import Path

import torch

from downstream.toxicity.finetune import SmilesAnchorResidualAuxClassifier
from downstream.toxicity.compute_occlusion import ToxicityLightningModule
from molencoder.config import MolEncoderConfig
from molencoder.model import MolEncoderWrapper


def infer_feature_dims(state: dict) -> tuple[int, int]:
    """Recover (token_feature_dim, molecule_feature_dim) from tensor shapes.

    token_proj.0 is LayerNorm(hidden + token_dim * 4)  -> token_dim
    molecule_proj.1 is Linear(molecule_dim, hidden)    -> molecule_dim
    Absent keys mean the branch was disabled, i.e. dimension 0.
    """
    hidden = int(state["model.base_classifier.weight"].shape[1])

    token_dim = 0
    key = "model.token_proj.0.weight"
    if key in state:
        token_dim = (int(state[key].shape[0]) - hidden) // 4

    molecule_dim = 0
    key = "model.molecule_proj.1.weight"
    if key in state:
        molecule_dim = int(state[key].shape[1])

    return token_dim, molecule_dim


def load_fusion_model(
    checkpoint_path: Path,
    pretrained_model_name: str,
    max_sequence_length: int,
    model_dir: Path | None,
    weight_mode: str = "abs",
    expected_token_dim: int | None = None,
    expected_molecule_dim: int | None = None,
) -> ToxicityLightningModule:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["state_dict"]

    token_dim, molecule_dim = infer_feature_dims(state)

    if expected_token_dim is not None and token_dim != expected_token_dim:
        raise ValueError(
            f"{checkpoint_path}: checkpoint has token_feature_dim={token_dim}, "
            f"but the features you passed imply {expected_token_dim}. "
            f"The --token-features you gave do not match how this model was trained."
        )
    if expected_molecule_dim is not None and molecule_dim != expected_molecule_dim:
        raise ValueError(
            f"{checkpoint_path}: checkpoint has molecule_feature_dim={molecule_dim}, "
            f"but the features you passed imply {expected_molecule_dim}."
        )

    encoder = MolEncoderWrapper(
        MolEncoderConfig(
            pretrained_model_name=pretrained_model_name,
            max_sequence_length=max_sequence_length,
            num_labels=2,
            model_cache_dir=model_dir,
        )
    )

    classifier = SmilesAnchorResidualAuxClassifier(
        encoder=encoder,
        num_labels=2,
        token_feature_dim=token_dim,
        molecule_feature_dim=molecule_dim,
        weight_mode=weight_mode,
    )

    lightning_model = ToxicityLightningModule(
        model=classifier, learning_rate=2e-5, weight_decay=1e-2
    )
    # strict=True: a silent key mismatch here would mean scoring with a
    # partially-initialised model.
    lightning_model.load_state_dict(state, strict=True)
    lightning_model.eval()
    return lightning_model
