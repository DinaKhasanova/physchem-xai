from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import pandas as pd
import torch

torch.set_num_threads(2)  # this VM has 2 vCPUs; torch defaults to 1
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import AdamW
from torch.utils.data import DataLoader

from molencoder.config import MolEncoderConfig
from molencoder.model import MolEncoderWrapper

from .dataset import (
    ToxicityCollator,
    ToxicityDataset,
    load_toxicity_dataframe,
    split_toxicity_dataframe,
)
from .evaluate import binary_classification_metrics, classification_metrics_from_predictions, save_metrics
from .features import available_feature_names, available_molecule_feature_names, molecule_feature_dimension
from .finetune import (
    ConfidenceResidualFusionClassifier,
    GatedFusionClassifier,
    SmilesAnchorResidualAuxClassifier,
    ToxicityClassifier,
    ToxicityCnnClassifier,
)


@dataclass(slots=True)
class TrainConfig:
    dataset_path: Path
    output_dir: Path
    pretrained_model_name: str = "fabikru/MolEncoder"
    max_sequence_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    head_learning_rate: float | None = None
    fusion_learning_rate: float | None = None
    descriptor_projection_dim: int | None = None
    weight_decay: float = 1e-2
    max_epochs: int = 5
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0
    token_feature_names: tuple[str, ...] = ()
    molecule_feature_names: tuple[str, ...] = ()
    fusion_type: str = "current"
    token_feature_normalize_mode: str = "per_molecule"
    token_weight_mode: str = "abs"
    charge_xyz_dir: Path | None = None
    reuse_existing_splits: bool = True
    model_dir: Path | None = Path("models/molencoder")
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-3


def save_split_dataframes(
    output_dir: Path,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    validation_path = splits_dir / "validation.csv"
    test_path = splits_dir / "test.csv"

    train_frame.to_csv(train_path, index=False)
    validation_frame.to_csv(validation_path, index=False)
    test_frame.to_csv(test_path, index=False)

    return train_path, validation_path, test_path


def load_or_create_split_dataframes(
    output_dir: Path,
    frame: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
    reuse_existing_splits: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    splits_dir = output_dir / "splits"
    train_path = splits_dir / "train.csv"
    validation_path = splits_dir / "validation.csv"
    test_path = splits_dir / "test.csv"

    if reuse_existing_splits and train_path.exists() and validation_path.exists() and test_path.exists():
        train_frame = pd.read_csv(train_path)
        validation_frame = pd.read_csv(validation_path)
        test_frame = pd.read_csv(test_path)
        return train_frame, validation_frame, test_frame, train_path, validation_path, test_path

    train_frame, validation_frame, test_frame = split_toxicity_dataframe(
        frame,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    train_path, validation_path, test_path = save_split_dataframes(
        output_dir=output_dir,
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
    )
    return train_frame, validation_frame, test_frame, train_path, validation_path, test_path


class ToxicityLightningModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        weight_decay: float,
        head_learning_rate: float | None = None,
        fusion_learning_rate: float | None = None,
    ) -> None:
        """Fine-tuning module for the toxicity classifier.

        ``head_learning_rate`` lets the freshly initialised classification head
        train at a different rate from the pretrained encoder. Leaving it at
        ``None`` reproduces the original single-rate behaviour exactly.
        """

        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.head_learning_rate = head_learning_rate
        self.fusion_learning_rate = fusion_learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_features: torch.Tensor | None = None,
        molecule_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_features=token_features,
            molecule_features=molecule_features,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_features"),
            batch.get("molecule_features"),
        )
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_features"),
            batch.get("molecule_features"),
        )
        loss = self.loss_fn(logits, batch["labels"])
        metrics = binary_classification_metrics(logits, batch["labels"])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", metrics["accuracy"], prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_features"),
            batch.get("molecule_features"),
        )
        loss = self.loss_fn(logits, batch["labels"])
        metrics = binary_classification_metrics(logits, batch["labels"])
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_accuracy", metrics["accuracy"], prog_bar=True, on_step=False, on_epoch=True)

    FUSION_PARAM_MARKERS = (
        "feature_embedding", "feature_gate",
        "token_proj", "molecule_proj", "aux_proj", "aux_classifier",
        "gate_mlp", "charge_context_scale", "feature_stats_scale",
    )

    def split_parameter_groups(self) -> tuple[list, list, list]:
        """Split into (pretrained encoder, descriptor-injection, everything else).

        The injection group is the early-fusion projection that maps token-level
        descriptors into the encoder's embedding space. It is separated because
        it starts from a **zero-initialised** output layer and therefore needs a
        far larger learning rate than a pretrained encoder to become useful.

        Measured on the first early-fusion run: after ~1270 optimiser steps at
        lr 2e-5, ``feature_embedding.2.weight`` had grown from exactly 0 to std
        0.00075, while the layer feeding it sat at its random-init std of 0.58.
        The injection was therefore ~0.1% of the token-embedding magnitude --
        live, but far too small to carry information. Adam's per-step size is
        bounded by the learning rate, so the ceiling over that many steps was
        ~0.025 even moving in a straight line.

        "Everything else" is the head: ``classifier.*``, the fusion scalars
        ``charge_context_scale`` / ``feature_stats_scale``, and the projections
        of the residual/gated/CNN variants.
        """

        encoder_params, fusion_params, head_params = [], [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(marker in name for marker in self.FUSION_PARAM_MARKERS):
                fusion_params.append(parameter)
            elif name.startswith("encoder."):
                encoder_params.append(parameter)
            else:
                head_params.append(parameter)
        return encoder_params, fusion_params, head_params

    def configure_optimizers(self) -> AdamW:
        if self.head_learning_rate is None and self.fusion_learning_rate is None:
            return AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        encoder_params, fusion_params, head_params = self.split_parameter_groups()
        head_lr = self.head_learning_rate if self.head_learning_rate is not None else self.learning_rate
        fusion_lr = self.fusion_learning_rate if self.fusion_learning_rate is not None else head_lr

        groups = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": self.learning_rate})
        if fusion_params:
            groups.append({"params": fusion_params, "lr": fusion_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        if not groups:  # pragma: no cover - defensive
            return AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        print(
            f"[optimizer] encoder={len(encoder_params)} lr={self.learning_rate:g} | "
            f"fusion={len(fusion_params)} lr={fusion_lr:g} | "
            f"head={len(head_params)} lr={head_lr:g}"
        )
        return AdamW(groups, weight_decay=self.weight_decay)


def create_dataloaders(
    encoder: MolEncoderWrapper,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    token_feature_names: tuple[str, ...],
    molecule_feature_names: tuple[str, ...],
    token_feature_normalize_mode: str,
    charge_xyz_dir: Path | None,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
]:
    collator = ToxicityCollator(
        encoder,
        feature_names=list(token_feature_names),
        molecule_feature_names=list(molecule_feature_names),
        token_feature_normalize_mode=token_feature_normalize_mode,
        charge_xyz_dir=charge_xyz_dir,
    )
    train_loader = DataLoader(
        ToxicityDataset(train_frame),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        ToxicityDataset(validation_frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        ToxicityDataset(test_frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    return train_loader, validation_loader, test_loader


def train_toxicity_model(config: TrainConfig) -> Path:
    L.seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    encoder_config = MolEncoderConfig(
        pretrained_model_name=config.pretrained_model_name,
        max_sequence_length=config.max_sequence_length,
        num_labels=2,
        model_cache_dir=config.model_dir,
    )
    encoder = MolEncoderWrapper(encoder_config)
    frame = load_toxicity_dataframe(config.dataset_path)
    train_frame, validation_frame, test_frame, train_path, validation_path, test_path = load_or_create_split_dataframes(
        output_dir=config.output_dir,
        frame=frame,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
        reuse_existing_splits=config.reuse_existing_splits,
    )
    train_loader, validation_loader, test_loader = create_dataloaders(
        encoder=encoder,
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        token_feature_names=config.token_feature_names,
        molecule_feature_names=config.molecule_feature_names,
        token_feature_normalize_mode=config.token_feature_normalize_mode,
        charge_xyz_dir=config.charge_xyz_dir,
    )

    if config.fusion_type == "cnn":
        classifier = ToxicityCnnClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=len(config.token_feature_names),
            molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
        )
    elif config.fusion_type == "smiles_anchor_residual":
        classifier = SmilesAnchorResidualAuxClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=len(config.token_feature_names),
            molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
            weight_mode=config.token_weight_mode,
        )
    elif config.fusion_type == "confidence_residual":
        classifier = ConfidenceResidualFusionClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=len(config.token_feature_names),
            molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
            weight_mode=config.token_weight_mode,
        )
    elif config.fusion_type == "gated":
        classifier = GatedFusionClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=len(config.token_feature_names),
            molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
        )
    else:
        classifier = ToxicityClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=len(config.token_feature_names),
            molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
            weight_mode=config.token_weight_mode,
            descriptor_projection_dim=config.descriptor_projection_dim,
        )

    lightning_model = ToxicityLightningModule(
        model=classifier,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        head_learning_rate=config.head_learning_rate,
        fusion_learning_rate=config.fusion_learning_rate,
    )

    trainer = L.Trainer(
        default_root_dir=str(config.output_dir),
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
            ),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch:02d}-{val_loss:.4f}",
            ),
        ],
    )
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
    if best_checkpoint_path:
        best_model = ToxicityLightningModule.load_from_checkpoint(
            best_checkpoint_path,
            model=classifier,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            head_learning_rate=config.head_learning_rate,
        fusion_learning_rate=config.fusion_learning_rate,
        )
    else:
        best_model = lightning_model
        best_checkpoint_path = str(config.output_dir / "toxicity_classifier.ckpt")

    trainer.test(best_model, dataloaders=test_loader)

    labels: list[int] = []
    predictions: list[int] = []
    positive_scores: list[float] = []
    best_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            logits = best_model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_features=batch.get("token_features"),
                molecule_features=batch.get("molecule_features"),
            )
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            labels.extend(batch["labels"].tolist())
            predictions.extend(preds.tolist())
            positive_scores.extend(probs[:, 1].tolist())

    metrics = classification_metrics_from_predictions(
        labels=labels,
        positive_scores=positive_scores,
        predictions=predictions,
    )
    best_model_score = trainer.checkpoint_callback.best_model_score
    best_epoch_match = re.search(r"epoch=(\d+)", trainer.checkpoint_callback.best_model_path)
    metrics["best_val_loss"] = float(best_model_score) if best_model_score is not None else float("nan")
    metrics["best_epoch"] = int(best_epoch_match.group(1)) if best_epoch_match else int(trainer.current_epoch)

    checkpoint_path = config.output_dir / "toxicity_classifier.ckpt"
    if best_checkpoint_path and Path(best_checkpoint_path).exists():
        shutil.copyfile(best_checkpoint_path, checkpoint_path)
    else:
        trainer.save_checkpoint(str(checkpoint_path))
    metrics_csv_path, metrics_json_path = save_metrics(config.output_dir, metrics)
    print(f"Saved train split to {train_path}")
    print(f"Saved validation split to {validation_path}")
    print(f"Saved test split to {test_path}")
    print(f"Saved metrics csv to {metrics_csv_path}")
    print(f"Saved metrics json to {metrics_json_path}")
    return checkpoint_path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune MolEncoder on a toxicity dataset.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="CSV file with `smiles` and `label` columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/toxicity"))
    parser.add_argument("--pretrained-model-name", type=str, default="fabikru/MolEncoder")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--descriptor-projection-dim", type=int, default=None)
    parser.add_argument("--fusion-learning-rate", type=float, default=None)
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=None,
        help=(
            "Separate learning rate for the randomly initialised classification head. "
            "Omit to reproduce the original single-rate behaviour."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--fusion-type",
        type=str,
        choices=("current", "gated", "cnn", "confidence_residual", "smiles_anchor_residual"),
        default="current",
        help="Fusion strategy for token and molecule features.",
    )
    parser.add_argument(
        "--token-feature-normalize-mode",
        type=str,
        choices=("per_molecule", "none"),
        default="per_molecule",
        help="Per-molecule token feature normalization mode.",
    )
    parser.add_argument(
        "--token-weight-mode",
        type=str,
        choices=("abs", "positive"),
        default="abs",
        help="Token weighting rule used by the current fusion strategy.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/molencoder"),
        help="Project-local directory for the pretrained MolEncoder files.",
    )
    parser.add_argument(
        "--reuse-existing-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse split CSV files from the output directory if they already exist.",
    )
    parser.add_argument(
        "--token-features",
        nargs="*",
        default=(),
        choices=available_feature_names(),
        help="Optional token-level features: charge, mr, neighbor.",
    )
    parser.add_argument(
        "--molecule-features",
        nargs="*",
        default=(),
        choices=available_molecule_feature_names(),
        help="Optional molecule-level feature sets appended after the encoder, e.g. physchem.",
    )
    parser.add_argument(
        "--charge-xyz-dir",
        type=Path,
        default=None,
        help="Optional directory with xyz files and an outs/ subdirectory for ORCA/XTB charge lookup; falls back to Gasteiger when unavailable.",
    )
    args = parser.parse_args()
    return TrainConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        pretrained_model_name=args.pretrained_model_name,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        fusion_learning_rate=args.fusion_learning_rate,
        descriptor_projection_dim=args.descriptor_projection_dim,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        token_feature_names=tuple(args.token_features),
        molecule_feature_names=tuple(args.molecule_features),
        fusion_type=args.fusion_type,
        token_feature_normalize_mode=args.token_feature_normalize_mode,
        token_weight_mode=args.token_weight_mode,
        charge_xyz_dir=args.charge_xyz_dir,
        reuse_existing_splits=args.reuse_existing_splits,
        model_dir=args.model_dir,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


def main() -> None:
    checkpoint_path = train_toxicity_model(parse_args())
    print(f"Saved fine-tuned checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
