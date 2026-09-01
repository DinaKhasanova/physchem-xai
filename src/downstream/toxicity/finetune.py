from __future__ import annotations

import torch
from torch import nn

from molencoder.model import MolEncoderWrapper


def _summarize_token_features(
    token_features: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = attention_mask.unsqueeze(-1).to(token_features.dtype)
    masked_features = token_features * mask
    token_count = mask.sum(dim=1).clamp(min=1.0)
    feature_mean = masked_features.sum(dim=1) / token_count
    feature_std = torch.sqrt(
        ((masked_features - feature_mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / token_count
    )

    negative_inf = torch.full_like(masked_features, float("-inf"))
    positive_inf = torch.full_like(masked_features, float("inf"))
    feature_max = torch.where(mask.bool(), masked_features, negative_inf).amax(dim=1)
    feature_min = torch.where(mask.bool(), masked_features, positive_inf).amin(dim=1)
    feature_max = torch.where(torch.isfinite(feature_max), feature_max, torch.zeros_like(feature_max))
    feature_min = torch.where(torch.isfinite(feature_min), feature_min, torch.zeros_like(feature_min))
    feature_stats = torch.cat([feature_mean, feature_std, feature_max, feature_min], dim=-1)
    return mask, token_count, feature_mean, feature_stats


class ToxicityClassifier(nn.Module):
    """Current fusion head using CLS plus weighted token-feature pooling."""

    def __init__(
        self,
        encoder: MolEncoderWrapper,
        hidden_size: int | None = None,
        num_labels: int = 2,
        token_feature_dim: int = 0,
        molecule_feature_dim: int = 0,
        weight_mode: str = "abs",
        descriptor_projection_dim: int | None = None,
    ) -> None:
        """Fusion head over CLS, descriptor-weighted token pooling and molecule descriptors.

        ``descriptor_projection_dim`` addresses a measured dilution problem. The
        four blocks are concatenated and passed through one shared LayerNorm, so
        a 6-16 dimensional descriptor block sits inside a ~780 dimensional vector
        and is normalised jointly with 768 embedding dimensions. Reading trained
        checkpoints shows the consequence: every descriptor dimension ends up
        with the same per-dimension weight mass as an average CLS dimension
        (0.98x), i.e. the head never learns to treat descriptors as special.

        Setting this gives the descriptor block its own LayerNorm and a linear
        projection up to ``descriptor_projection_dim`` before concatenation, so
        it is normalised on its own scale and occupies a share of the input
        proportional to its importance rather than to its raw dimensionality.
        Leaving it ``None`` reproduces the original architecture exactly.
        """

        super().__init__()
        self.encoder = encoder
        self.encoder_hidden_size = hidden_size or encoder.hidden_size
        self.token_feature_dim = token_feature_dim
        self.molecule_feature_dim = molecule_feature_dim
        self.weight_mode = weight_mode
        self.descriptor_projection_dim = descriptor_projection_dim
        stats_dim = token_feature_dim * 4 if token_feature_dim > 0 else 0

        self.descriptor_proj = None
        effective_molecule_dim = molecule_feature_dim
        if descriptor_projection_dim and molecule_feature_dim > 0:
            self.descriptor_proj = nn.Sequential(
                nn.Identity(),  # was LayerNorm: per-molecule normalisation destroyed
                # the cross-molecule level, which is the signal. Descriptors are
                # now standardised per-feature at the data level instead.
                nn.Linear(molecule_feature_dim, descriptor_projection_dim),
                nn.GELU(),
            )
            effective_molecule_dim = descriptor_projection_dim

        classifier_input_size = self.encoder_hidden_size + self.encoder_hidden_size + stats_dim + effective_molecule_dim
        self.charge_context_scale = nn.Parameter(torch.tensor(0.1))
        self.feature_stats_scale = nn.Parameter(torch.tensor(0.1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_size),
            nn.Linear(classifier_input_size, self.encoder_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.encoder_hidden_size, num_labels),
        )

    def summarize_token_features(
        self,
        token_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, _, feature_stats = _summarize_token_features(token_features, attention_mask)

        if self.weight_mode == "abs":
            feature_strength = token_features.abs().sum(dim=-1)
        elif self.weight_mode == "positive":
            feature_strength = token_features.clamp(min=0.0).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported weight mode: {self.weight_mode}")
        feature_strength = feature_strength * attention_mask.to(feature_strength.dtype)
        weight_denom = feature_strength.sum(dim=1, keepdim=True).clamp(min=1e-6)
        feature_weights = (feature_strength / weight_denom).unsqueeze(-1)
        return feature_weights, feature_stats

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_features: torch.Tensor | None = None,
        molecule_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The encoder ignores token_features unless it was built with early
        # fusion enabled, in which case it injects them into the token
        # embeddings before the transformer layers. Passing them is safe either
        # way, and is what makes early fusion reachable at all.
        token_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_features=token_features,
        )
        cls_embedding = token_states[:, 0, :]
        if token_features is None or token_features.shape[-1] == 0:
            pooled_context = torch.zeros_like(cls_embedding)
            feature_stats = torch.zeros(
                cls_embedding.shape[0],
                0,
                device=cls_embedding.device,
                dtype=cls_embedding.dtype,
            )
        else:
            feature_weights, feature_stats = self.summarize_token_features(token_features, attention_mask)
            pooled_context = (token_states * feature_weights.to(token_states.dtype)).sum(dim=1)
            pooled_context = self.charge_context_scale * pooled_context
            feature_stats = self.feature_stats_scale * feature_stats.to(cls_embedding.dtype)
        if molecule_features is None:
            molecule_features = torch.zeros(
                cls_embedding.shape[0],
                0,
                device=cls_embedding.device,
                dtype=cls_embedding.dtype,
            )
        else:
            molecule_features = molecule_features.to(cls_embedding.dtype)
            if self.descriptor_proj is not None and molecule_features.shape[-1] > 0:
                molecule_features = self.descriptor_proj(molecule_features)
        classifier_input = torch.cat([cls_embedding, pooled_context, feature_stats, molecule_features], dim=-1)
        return self.classifier(classifier_input)


class ConfidenceResidualFusionClassifier(nn.Module):
    """Base token fusion plus confidence-gated residual adjustment from molecule descriptors."""

    def __init__(
        self,
        encoder: MolEncoderWrapper,
        hidden_size: int | None = None,
        num_labels: int = 2,
        token_feature_dim: int = 0,
        molecule_feature_dim: int = 0,
        weight_mode: str = "positive",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_hidden_size = hidden_size or encoder.hidden_size
        self.token_feature_dim = token_feature_dim
        self.molecule_feature_dim = molecule_feature_dim
        self.weight_mode = weight_mode

        stats_dim = token_feature_dim * 4 if token_feature_dim > 0 else 0
        base_input_size = self.encoder_hidden_size + self.encoder_hidden_size + stats_dim
        self.charge_context_scale = nn.Parameter(torch.tensor(0.1))
        self.feature_stats_scale = nn.Parameter(torch.tensor(0.1))
        self.base_proj = nn.Sequential(
            nn.LayerNorm(base_input_size),
            nn.Linear(base_input_size, self.encoder_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.base_classifier = nn.Linear(self.encoder_hidden_size, num_labels)

        self.descriptor_proj = (
            nn.Sequential(
                nn.Identity(),  # was LayerNorm: per-molecule normalisation destroyed
                # the cross-molecule level, which is the signal. Descriptors are
                # now standardised per-feature at the data level instead.
                nn.Linear(molecule_feature_dim, self.encoder_hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
            )
            if molecule_feature_dim > 0
            else None
        )
        gate_input_dim = self.encoder_hidden_size + 2
        self.gate_mlp = (
            nn.Sequential(
                nn.Linear(gate_input_dim, self.encoder_hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.encoder_hidden_size // 2, 1),
            )
            if molecule_feature_dim > 0
            else None
        )
        if self.gate_mlp is not None:
            final_linear = self.gate_mlp[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(final_linear.bias, -2.0)

        self.final_classifier = nn.Sequential(
            nn.LayerNorm(self.encoder_hidden_size),
            nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.encoder_hidden_size, num_labels),
        )

    def summarize_token_features(
        self,
        token_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, _, feature_stats = _summarize_token_features(token_features, attention_mask)
        if self.weight_mode == "abs":
            feature_strength = token_features.abs().sum(dim=-1)
        elif self.weight_mode == "positive":
            feature_strength = token_features.clamp(min=0.0).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported weight mode: {self.weight_mode}")
        feature_strength = feature_strength * attention_mask.to(feature_strength.dtype)
        weight_denom = feature_strength.sum(dim=1, keepdim=True).clamp(min=1e-6)
        feature_weights = (feature_strength / weight_denom).unsqueeze(-1)
        return feature_weights, feature_stats

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_features: torch.Tensor | None = None,
        molecule_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = token_states[:, 0, :]
        if token_features is None or token_features.shape[-1] == 0:
            pooled_context = torch.zeros_like(cls_embedding)
            feature_stats = torch.zeros(
                cls_embedding.shape[0],
                0,
                device=cls_embedding.device,
                dtype=cls_embedding.dtype,
            )
        else:
            feature_weights, feature_stats = self.summarize_token_features(token_features, attention_mask)
            pooled_context = (token_states * feature_weights.to(token_states.dtype)).sum(dim=1)
            pooled_context = self.charge_context_scale * pooled_context
            feature_stats = self.feature_stats_scale * feature_stats.to(cls_embedding.dtype)

        base_input = torch.cat([cls_embedding, pooled_context, feature_stats], dim=-1)
        base_repr = self.base_proj(base_input)
        base_logits = self.base_classifier(base_repr)

        if molecule_features is None or molecule_features.shape[-1] == 0 or self.descriptor_proj is None or self.gate_mlp is None:
            return self.final_classifier(base_repr)

        descriptor_adjustment = self.descriptor_proj(molecule_features.to(cls_embedding.dtype))
        base_probs = torch.softmax(base_logits, dim=-1)
        confidence = base_probs.max(dim=-1, keepdim=True).values
        margin = (base_probs[:, 1:2] - base_probs[:, 0:1]).abs()
        gate_input = torch.cat([base_repr, confidence, margin], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        fused_repr = base_repr + gate * descriptor_adjustment
        return self.final_classifier(fused_repr)


class SmilesAnchorResidualAuxClassifier(nn.Module):
    """SMILES-only anchor with confidence-gated auxiliary logit correction from token and molecule features."""

    def __init__(
        self,
        encoder: MolEncoderWrapper,
        hidden_size: int | None = None,
        num_labels: int = 2,
        token_feature_dim: int = 0,
        molecule_feature_dim: int = 0,
        weight_mode: str = "positive",
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
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
        self.token_proj = (
            nn.Sequential(
                nn.LayerNorm(self.encoder_hidden_size + token_feature_dim * 4),
                nn.Linear(self.encoder_hidden_size + token_feature_dim * 4, self.encoder_hidden_size),
                nn.GELU(),
            )
            if token_feature_dim > 0
            else None
        )
        self.molecule_proj = (
            nn.Sequential(
                nn.Identity(),  # was LayerNorm: per-molecule normalisation destroyed
                # the cross-molecule level, which is the signal. Descriptors are
                # now standardised per-feature at the data level instead.
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

    def summarize_token_features(
        self,
        token_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, _, feature_stats = _summarize_token_features(token_features, attention_mask)
        if self.weight_mode == "abs":
            feature_strength = token_features.abs().sum(dim=-1)
        elif self.weight_mode == "positive":
            feature_strength = token_features.clamp(min=0.0).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported weight mode: {self.weight_mode}")
        feature_strength = feature_strength * attention_mask.to(feature_strength.dtype)
        weight_denom = feature_strength.sum(dim=1, keepdim=True).clamp(min=1e-6)
        feature_weights = (feature_strength / weight_denom).unsqueeze(-1)
        return feature_weights, feature_stats

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_features: torch.Tensor | None = None,
        molecule_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = token_states[:, 0, :]

        base_repr = self.base_proj(cls_embedding)
        base_logits = self.base_classifier(base_repr)

        aux_parts: list[torch.Tensor] = []
        if token_features is not None and token_features.shape[-1] > 0 and self.token_proj is not None:
            feature_weights, feature_stats = self.summarize_token_features(token_features, attention_mask)
            pooled_context = (token_states * feature_weights.to(token_states.dtype)).sum(dim=1)
            pooled_context = self.charge_context_scale * pooled_context
            feature_stats = self.feature_stats_scale * feature_stats.to(cls_embedding.dtype)
            aux_parts.append(self.token_proj(torch.cat([pooled_context, feature_stats], dim=-1)))
        if molecule_features is not None and molecule_features.shape[-1] > 0 and self.molecule_proj is not None:
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


class GatedFusionClassifier(nn.Module):
    """Learned gating over token states plus projected token features."""

    def __init__(
        self,
        encoder: MolEncoderWrapper,
        hidden_size: int | None = None,
        num_labels: int = 2,
        token_feature_dim: int = 0,
        molecule_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_hidden_size = hidden_size or encoder.hidden_size
        self.feature_proj = nn.Linear(token_feature_dim, self.encoder_hidden_size) if token_feature_dim > 0 else None
        self.gate_mlp = (
            nn.Sequential(
                nn.Linear(self.encoder_hidden_size * 2, self.encoder_hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.encoder_hidden_size // 2, 1),
            )
            if token_feature_dim > 0
            else None
        )
        self.molecule_proj = (
            nn.Sequential(
                nn.Identity(),  # was LayerNorm: per-molecule normalisation destroyed
                # the cross-molecule level, which is the signal. Descriptors are
                # now standardised per-feature at the data level instead.
                nn.Linear(molecule_feature_dim, self.encoder_hidden_size // 2),
                nn.GELU(),
            )
            if molecule_feature_dim > 0
            else None
        )
        feature_stats_dim = token_feature_dim * 2 if token_feature_dim > 0 else 0
        molecule_branch_dim = self.encoder_hidden_size // 2 if molecule_feature_dim > 0 else 0
        classifier_input_size = self.encoder_hidden_size + self.encoder_hidden_size + feature_stats_dim + molecule_branch_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_size),
            nn.Linear(classifier_input_size, self.encoder_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.encoder_hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_features: torch.Tensor | None = None,
        molecule_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = token_states[:, 0, :]
        if token_features is None or token_features.shape[-1] == 0 or self.feature_proj is None or self.gate_mlp is None:
            pooled_context = torch.zeros_like(cls_embedding)
            feature_stats = torch.zeros(cls_embedding.shape[0], 0, device=cls_embedding.device, dtype=cls_embedding.dtype)
        else:
            projected = self.feature_proj(token_features.to(token_states.dtype))
            gate_input = torch.cat([token_states, projected], dim=-1)
            gate_logits = self.gate_mlp(gate_input).squeeze(-1)
            gate_logits = gate_logits.masked_fill(~attention_mask.bool(), float("-inf"))
            gate_weights = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
            pooled_context = ((token_states + projected) * gate_weights).sum(dim=1)

            mask = attention_mask.unsqueeze(-1).to(token_features.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            feature_mean = (token_features * mask).sum(dim=1) / denom
            positive_mass = (token_features.clamp(min=0.0) * mask).sum(dim=1) / denom
            feature_stats = torch.cat([feature_mean, positive_mass], dim=-1).to(cls_embedding.dtype)

        if molecule_features is None or self.molecule_proj is None:
            molecule_branch = torch.zeros(cls_embedding.shape[0], 0, device=cls_embedding.device, dtype=cls_embedding.dtype)
        else:
            molecule_branch = self.molecule_proj(molecule_features.to(cls_embedding.dtype))
        classifier_input = torch.cat([cls_embedding, pooled_context, feature_stats, molecule_branch], dim=-1)
        return self.classifier(classifier_input)


class ToxicityCnnClassifier(nn.Module):
    """CNN fusion over token states plus token features with a separate molecule branch."""

    def __init__(
        self,
        encoder: MolEncoderWrapper,
        hidden_size: int | None = None,
        num_labels: int = 2,
        token_feature_dim: int = 0,
        molecule_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_hidden_size = hidden_size or encoder.hidden_size
        input_channels = self.encoder_hidden_size + token_feature_dim
        conv_hidden = self.encoder_hidden_size // 2
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_channels, conv_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_hidden, conv_hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(conv_hidden, conv_hidden, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.molecule_proj = (
            nn.Sequential(
                nn.Identity(),  # was LayerNorm: per-molecule normalisation destroyed
                # the cross-molecule level, which is the signal. Descriptors are
                # now standardised per-feature at the data level instead.
                nn.Linear(molecule_feature_dim, self.encoder_hidden_size // 2),
                nn.GELU(),
            )
            if molecule_feature_dim > 0
            else None
        )
        molecule_branch_dim = self.encoder_hidden_size // 2 if molecule_feature_dim > 0 else 0
        classifier_input_size = self.encoder_hidden_size + conv_hidden * 2 + molecule_branch_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_size),
            nn.Linear(classifier_input_size, self.encoder_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.encoder_hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_features: torch.Tensor | None = None,
        molecule_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = token_states[:, 0, :]
        if token_features is None or token_features.shape[-1] == 0:
            cnn_input = token_states
        else:
            cnn_input = torch.cat([token_states, token_features.to(token_states.dtype)], dim=-1)

        mask = attention_mask.unsqueeze(-1).to(cnn_input.dtype)
        cnn_input = cnn_input * mask
        conv_features = self.conv_stack(cnn_input.transpose(1, 2))
        conv_features = conv_features * attention_mask.unsqueeze(1).to(conv_features.dtype)

        negative_inf = torch.full_like(conv_features, float("-inf"))
        max_pool = torch.where(attention_mask.unsqueeze(1).bool(), conv_features, negative_inf).amax(dim=-1)
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(conv_features.dtype)
        mean_pool = conv_features.sum(dim=-1) / denom
        if molecule_features is None or self.molecule_proj is None:
            molecule_branch = torch.zeros(
                cls_embedding.shape[0],
                0,
                device=cls_embedding.device,
                dtype=cls_embedding.dtype,
            )
        else:
            molecule_branch = self.molecule_proj(molecule_features.to(cls_embedding.dtype))
        classifier_input = torch.cat([cls_embedding, max_pool, mean_pool, molecule_branch], dim=-1)
        return self.classifier(classifier_input)
