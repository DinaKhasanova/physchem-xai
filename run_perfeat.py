"""Per-feature attention over token descriptors — standalone experiment.

Nothing under src/ is modified. This script imports the existing pipeline,
defines one new classifier locally, and substitutes it into
run_cross_validation._build_classifier at import time. Every other detail of
training (splits, seeds, callbacks, early stopping, checkpointing) is the
untouched project code.

WHY
---
SmilesAnchorResidualAuxClassifier.summarize_token_features collapses all token
descriptors into ONE attention weight per token:

    feature_strength = token_features.abs().sum(dim=-1)   # sums ACROSS features
    pooled_context   = (token_states * feature_weights).sum(dim=1)

With charge + neighbour counts, an atom scoring 1.7 could be strongly charged
or highly connected; the model cannot tell. Measured: Gasteiger alone 0.874647,
Gasteiger + neighbourhoods + MR 0.874626 -- a difference of 0.00002.

This class gives every descriptor its OWN attention weight, pools token_states
separately with each, and concatenates. F descriptors give F context vectors.

INITIALISATION
--------------
__init__ deliberately does NOT call the parent's __init__. It replicates it
layer for layer, in the same order, changing only token_proj's width. Building
token_proj once (rather than letting the parent build it and then replacing it)
means the random number generator is consumed identically, so at
token_feature_dim == 1 -- where per-feature and single-weight attention are
mathematically the same -- this class reproduces the parent's result exactly.
That makes runs_7251/gasteiger a valid control for the combination runs.

USAGE
-----
    PYTHONPATH=src python run_perfeat.py \
        --dataset-path data/ames_7251.csv --output-dir runs_7251/pf_gasteiger \
        --token-features charge \
        --token-weight-mode abs --token-feature-normalize-mode none \
        --fusion-learning-rate 2e-3 --gate-bias-init 0.0 \
        --batch-size 16 --max-epochs 15 --seed 42 --n-splits 5

Takes every flag run_cross_validation takes; --fusion-type is ignored.
"""
from __future__ import annotations

import torch
from torch import nn

from downstream.toxicity import run_cross_validation as RCV
from downstream.toxicity.finetune import (
    SmilesAnchorResidualAuxClassifier,
    _summarize_token_features,
)
from downstream.toxicity.features import molecule_feature_dimension


class PerFeatureAttentionClassifier(SmilesAnchorResidualAuxClassifier):
    """Anchor-residual fusion with one attention weight per token descriptor."""

    def __init__(self, encoder, hidden_size=None, num_labels=2,
                 token_feature_dim=0, molecule_feature_dim=0,
                 weight_mode="positive", gate_bias_init=-2.0):
        # nn.Module.__init__, NOT the parent's -- see INITIALISATION above.
        nn.Module.__init__(self)
        self.encoder = encoder
        self.encoder_hidden_size = hidden_size or encoder.hidden_size
        self.weight_mode = weight_mode
        self.token_feature_dim = token_feature_dim
        self.molecule_feature_dim = molecule_feature_dim
        self.base_proj = nn.Sequential(
            nn.LayerNorm(self.encoder_hidden_size),
            nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.base_classifier = nn.Linear(self.encoder_hidden_size, num_labels)
        self.charge_context_scale = nn.Parameter(torch.tensor(0.1))
        self.feature_stats_scale = nn.Parameter(torch.tensor(0.1))
        # ONLY DIFFERENCE: hidden_size * F instead of hidden_size, because each
        # descriptor contributes its own pooled context. Identical at F == 1.
        token_width = self.encoder_hidden_size * token_feature_dim + token_feature_dim * 4
        self.token_proj = (
            nn.Sequential(
                nn.LayerNorm(token_width),
                nn.Linear(token_width, self.encoder_hidden_size),
                nn.GELU(),
            )
            if token_feature_dim > 0
            else None
        )
        self.molecule_proj = (
            nn.Sequential(
                nn.Identity(),
                nn.Linear(molecule_feature_dim, self.encoder_hidden_size),
                nn.GELU(),
            )
            if molecule_feature_dim > 0
            else None
        )
        self.aux_proj = nn.Sequential(
            nn.LayerNorm(self.encoder_hidden_size),
            nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.aux_classifier = nn.Linear(self.encoder_hidden_size, num_labels)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.encoder_hidden_size + 2, self.encoder_hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.encoder_hidden_size // 2, 1),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, gate_bias_init)

    def per_feature_weights(self, token_features, attention_mask):
        """(B, T, F) attention weights, normalised over tokens for each feature."""
        if self.weight_mode == "abs":
            strength = token_features.abs()
        elif self.weight_mode == "positive":
            strength = token_features.clamp(min=0.0)
        else:
            raise ValueError(f"Unsupported weight mode: {self.weight_mode}")
        strength = strength * attention_mask.unsqueeze(-1).to(strength.dtype)
        denom = strength.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return strength / denom

    def forward(self, input_ids, attention_mask, token_features=None,
                molecule_features=None):
        token_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = token_states[:, 0, :]
        base_repr = self.base_proj(cls_embedding)
        base_logits = self.base_classifier(base_repr)

        aux_parts: list[torch.Tensor] = []

        if (token_features is not None and token_features.shape[-1] > 0
                and self.token_proj is not None):
            _, _, _, feature_stats = _summarize_token_features(token_features, attention_mask)
            weights = self.per_feature_weights(token_features, attention_mask)
            # (B,T,H) x (B,T,F) -> (B,F,H) -> (B, F*H)
            contexts = torch.einsum("bth,btf->bfh", token_states,
                                    weights.to(token_states.dtype))
            contexts = (self.charge_context_scale * contexts).flatten(start_dim=1)
            feature_stats = self.feature_stats_scale * feature_stats.to(cls_embedding.dtype)
            aux_parts.append(self.token_proj(torch.cat([contexts, feature_stats], dim=-1)))

        if (molecule_features is not None and molecule_features.shape[-1] > 0
                and self.molecule_proj is not None):
            aux_parts.append(self.molecule_proj(molecule_features.to(cls_embedding.dtype)))

        if not aux_parts:
            return base_logits

        aux_repr = torch.stack(aux_parts, dim=0).mean(dim=0)
        aux_repr = self.aux_proj(aux_repr)
        aux_delta = self.aux_classifier(aux_repr)
        base_probs = torch.softmax(base_logits, dim=-1)
        confidence = base_probs.max(dim=-1, keepdim=True).values
        margin = (base_probs[:, 1:2] - base_probs[:, 0:1]).abs()
        gate = torch.sigmoid(self.gate_mlp(torch.cat([base_repr, confidence, margin], dim=-1)))
        return base_logits + gate * aux_delta


def _build_per_feature_classifier(config, encoder):
    return PerFeatureAttentionClassifier(
        encoder=encoder,
        num_labels=2,
        token_feature_dim=len(config.token_feature_names),
        molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
        weight_mode=config.token_weight_mode,
        gate_bias_init=getattr(config, "gate_bias_init", -2.0),
    )


RCV._build_classifier = _build_per_feature_classifier


if __name__ == "__main__":
    print("[run_perfeat] per-feature attention classifier active", flush=True)
    RCV.main()
