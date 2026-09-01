from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .config import MolEncoderConfig


class MolEncoderWrapper(nn.Module):
    """Thin wrapper around the pretrained Hugging Face MolEncoder model.

    This is the default encoder for downstream tasks in this project.
    """

    def __init__(self, config: MolEncoderConfig, token_feature_dim: int = 0) -> None:
        super().__init__()
        self.config = config
        load_path = self._resolve_model_load_path()
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path
        )
        self.encoder = AutoModel.from_pretrained(
            load_path
        )
        self.hidden_size = int(self.encoder.config.hidden_size)

        # ---- early fusion -------------------------------------------------
        # Injects per-token physicochemical values into the token embeddings
        # *before* the transformer layers, so self-attention operates on
        # charge-aware representations and the value influences every layer.
        #
        # The alternative -- the existing late fusion in ToxicityClassifier --
        # was measured to be inert: zeroing every charge changed test ROC AUC by
        # 0.0010 and shuffling charges between atoms by 0.0002, because
        # `pooled_context` normalises its weights to sum to one (destroying
        # absolute magnitude, which is the reactivity signal) and both paths are
        # scaled by 0.1.
        #
        # Zero-initialised output projection: at step 0 the sum is exactly the
        # pretrained embedding, so the model starts from the SMILES-only
        # behaviour and learns to use the descriptor from there. Without this,
        # a random projection perturbs a pretrained encoder and training starts
        # from a worse point than the baseline.
        self.token_feature_dim = token_feature_dim
        self.feature_embedding = None
        if token_feature_dim > 0:
            self.feature_embedding = nn.Sequential(
                nn.Linear(token_feature_dim, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            nn.init.zeros_(self.feature_embedding[-1].weight)
            nn.init.zeros_(self.feature_embedding[-1].bias)
            # Learned gate on the whole injection, so the model can also scale
            # the contribution up or down globally. Starts at 1.0 because the
            # zero-init projection already makes the initial contribution zero.
            self.feature_gate = nn.Parameter(torch.tensor(1.0))

    def _resolve_model_load_path(self) -> str:
        """Resolve where to load MolEncoder from.

        If `model_cache_dir` is set, we use a project-local directory:
        - if the model is already downloaded there, load it from disk
        - otherwise download it there so future runs are local and portable
        """
        if self.config.model_cache_dir is None:
            return self.config.pretrained_model_name

        model_dir = Path(self.config.model_cache_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        has_local_model = (model_dir / "config.json").exists()
        if has_local_model:
            return str(model_dir)

        AutoTokenizer.from_pretrained(
            self.config.pretrained_model_name,
            cache_dir=str(model_dir),
        ).save_pretrained(str(model_dir))
        AutoModel.from_pretrained(
            self.config.pretrained_model_name,
            cache_dir=str(model_dir),
        ).save_pretrained(str(model_dir))
        return str(model_dir)

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_early_fusion = (
            self.feature_embedding is not None
            and token_features is not None
            and token_features.shape[-1] == self.token_feature_dim
            and token_features.shape[-1] > 0
        )

        if not use_early_fusion:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state

        embeddings = self._input_embeddings(input_ids)
        injection = self.feature_embedding(token_features.to(embeddings.dtype))
        if attention_mask is not None:
            injection = injection * attention_mask.unsqueeze(-1).to(injection.dtype)
        outputs = self.encoder(
            inputs_embeds=embeddings + self.feature_gate * injection,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    def cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the hidden state of the first token, used as a CLS-style embedding."""
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_features=token_features,
        )[:, 0, :]

    def tokenize(self, smiles: list[str] | str) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt",
        )
