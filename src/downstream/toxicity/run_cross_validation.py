from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from molencoder.config import MolEncoderConfig
from molencoder.model import MolEncoderWrapper

from .dataset import load_toxicity_dataframe
from .evaluate import (
    classification_metrics_from_predictions,
    save_metrics,
    summarize_cross_validation_metrics,
)
from .features import molecule_feature_dimension
from .finetune import (
    ConfidenceResidualFusionClassifier,
    GatedFusionClassifier,
    SmilesAnchorResidualAuxClassifier,
    ToxicityClassifier,
    ToxicityCnnClassifier,
)
from .train import ToxicityLightningModule, TrainConfig, create_dataloaders, save_split_dataframes


@dataclass(slots=True)
class CrossValidationConfig:
    dataset_path: Path
    output_dir: Path
    pretrained_model_name: str = "fabikru/MolEncoder"
    model_dir: Path | None = Path("models/molencoder")
    max_sequence_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    head_learning_rate: float | None = None
    early_fusion: bool = False
    fusion_learning_rate: float | None = None
    descriptor_projection_dim: int | None = None
    weight_decay: float = 1e-2
    max_epochs: int = 15
    seed: int = 42
    num_workers: int = 0
    token_feature_names: tuple[str, ...] = ()
    molecule_feature_names: tuple[str, ...] = ()
    fusion_type: str = "current"
    gate_bias_init: float = -2.0
    token_feature_normalize_mode: str = "per_molecule"
    token_weight_mode: str = "abs"
    charge_xyz_dir: Path | None = None
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-3
    n_splits: int = 5
    validation_fraction_within_train: float = 0.125
    fold_indices: tuple[int, ...] = ()
    skip_completed_folds: bool = False


def _make_train_config(config: CrossValidationConfig, fold_output_dir: Path) -> TrainConfig:
    return TrainConfig(
        dataset_path=config.dataset_path,
        output_dir=fold_output_dir,
        pretrained_model_name=config.pretrained_model_name,
        max_sequence_length=config.max_sequence_length,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        head_learning_rate=config.head_learning_rate,
        fusion_learning_rate=config.fusion_learning_rate,
        descriptor_projection_dim=config.descriptor_projection_dim,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        seed=config.seed,
        num_workers=config.num_workers,
        token_feature_names=config.token_feature_names,
        molecule_feature_names=config.molecule_feature_names,
        fusion_type=config.fusion_type,
        token_feature_normalize_mode=config.token_feature_normalize_mode,
        token_weight_mode=config.token_weight_mode,
        charge_xyz_dir=config.charge_xyz_dir,
        reuse_existing_splits=True,
        model_dir=config.model_dir,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
    )


def _build_classifier(config: CrossValidationConfig, encoder: MolEncoderWrapper) -> torch.nn.Module:
    if config.fusion_type == "cnn":
        return ToxicityCnnClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=len(config.token_feature_names),
            molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
        )
    if config.fusion_type == "smiles_anchor_residual":
        return SmilesAnchorResidualAuxClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=len(config.token_feature_names),
            molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
            weight_mode=config.token_weight_mode,
            gate_bias_init=config.gate_bias_init,
        )
    if config.fusion_type == "confidence_residual":
        return ConfidenceResidualFusionClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=len(config.token_feature_names),
            molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
            weight_mode=config.token_weight_mode,
        )
    if config.fusion_type == "gated":
        return GatedFusionClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=len(config.token_feature_names),
            molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
        )
    return ToxicityClassifier(
        encoder=encoder,
        num_labels=2,
        token_feature_dim=len(config.token_feature_names),
        molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
        weight_mode=config.token_weight_mode,
        descriptor_projection_dim=config.descriptor_projection_dim,
    )


def _train_single_fold(
    config: CrossValidationConfig,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    fold_output_dir: Path,
) -> dict[str, float | int]:
    L.seed_everything(config.seed)
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    save_split_dataframes(fold_output_dir, train_frame, validation_frame, test_frame)

    encoder_config = MolEncoderConfig(
        pretrained_model_name=config.pretrained_model_name,
        max_sequence_length=config.max_sequence_length,
        num_labels=2,
        model_cache_dir=config.model_dir,
    )
    encoder = MolEncoderWrapper(
        encoder_config,
        token_feature_dim=len(config.token_feature_names) if config.early_fusion else 0,
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

    classifier = _build_classifier(config, encoder)
    lightning_model = ToxicityLightningModule(
        model=classifier,
        learning_rate=config.learning_rate,
        head_learning_rate=config.head_learning_rate,
        fusion_learning_rate=config.fusion_learning_rate,
        weight_decay=config.weight_decay,
    )

    trainer = L.Trainer(
        default_root_dir=str(fold_output_dir),
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
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
        head_learning_rate=config.head_learning_rate,
        fusion_learning_rate=config.fusion_learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        best_model = lightning_model

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
    metrics["stopped_epoch"] = int(trainer.current_epoch)

    checkpoint_path = fold_output_dir / "toxicity_classifier.ckpt"
    if best_checkpoint_path and Path(best_checkpoint_path).exists():
        shutil.copyfile(best_checkpoint_path, checkpoint_path)
    else:
        trainer.save_checkpoint(str(checkpoint_path))
    save_metrics(fold_output_dir, metrics)
    return metrics


def run_cross_validation(config: CrossValidationConfig) -> tuple[Path, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_toxicity_dataframe(config.dataset_path)
    labels = frame["label"].to_numpy()
    selected_folds = set(config.fold_indices)

    outer_cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    fold_metrics: list[dict[str, float | int]] = []

    for fold_index, (train_val_indices, test_indices) in enumerate(outer_cv.split(frame, labels), start=1):
        fold_dir = config.output_dir / f"fold_{fold_index}"
        if selected_folds and fold_index not in selected_folds:
            continue
        existing_metrics_path = fold_dir / "test_metrics.csv"
        if config.skip_completed_folds and existing_metrics_path.exists():
            existing_metrics = pd.read_csv(existing_metrics_path).iloc[0].to_dict()
            existing_metrics["fold"] = fold_index
            fold_metrics.append(existing_metrics)
            continue
        train_val_frame = frame.iloc[train_val_indices].reset_index(drop=True)
        test_frame = frame.iloc[test_indices].reset_index(drop=True)

        inner_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.validation_fraction_within_train,
            random_state=config.seed + fold_index,
        )
        inner_labels = train_val_frame["label"].to_numpy()
        train_indices, validation_indices = next(inner_splitter.split(train_val_frame, inner_labels))
        train_frame = train_val_frame.iloc[train_indices].reset_index(drop=True)
        validation_frame = train_val_frame.iloc[validation_indices].reset_index(drop=True)

        fold_metric = _train_single_fold(
            config=config,
            train_frame=train_frame,
            validation_frame=validation_frame,
            test_frame=test_frame,
            fold_output_dir=fold_dir,
        )
        fold_metric["fold"] = fold_index
        fold_metrics.append(fold_metric)

    folds_path = config.output_dir / "cross_validation_fold_metrics.csv"
    pd.DataFrame(fold_metrics).to_csv(folds_path, index=False)
    summary = summarize_cross_validation_metrics(fold_metrics)
    summary.update(
        {
            "n_samples": int(len(frame)),
            "n_positive": int(frame["label"].sum()),
            "n_negative": int((frame["label"] == 0).sum()),
        }
    )
    summary_path_csv, _ = save_metrics(config.output_dir, summary, filename_stem="cross_validation_summary")
    return folds_path, summary_path_csv


def parse_args() -> CrossValidationConfig:
    parser = argparse.ArgumentParser(description="Run 5-fold cross-validation for MolEncoder toxicity models.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model-name", type=str, default="fabikru/MolEncoder")
    parser.add_argument("--model-dir", type=Path, default=Path("models/molencoder"))
    parser.add_argument("--gate-bias-init", type=float, default=-2.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--descriptor-projection-dim",
        type=int,
        default=None,
        help=(
            "Give the molecule-descriptor block its own LayerNorm and project it to this "
            "width before concatenation, so it is not diluted by the shared LayerNorm over "
            "the full ~780-dim vector. Omit for the original architecture."
        ),
    )
    parser.add_argument(
        "--fusion-learning-rate",
        type=float,
        default=None,
        help=(
            "Learning rate for the early-fusion descriptor projection only. It is "
            "zero-initialised, so at the encoder's 2e-5 it cannot grow enough to matter: "
            "measured std 0.00075 after 4 epochs against a 0.58 input layer. Try 1e-3."
        ),
    )
    parser.add_argument(
        "--early-fusion",
        action="store_true",
        help=(
            "Inject token-level features into the encoder input embeddings before the "
            "transformer layers, instead of appending them to the pooled vector after. "
            "The late-fusion path was measured inert: zeroing all charges moved test "
            "ROC AUC by 0.0010."
        ),
    )
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=None,
        help="Separate learning rate for the classification head (omit for original behaviour).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--fusion-type",
        type=str,
        choices=("current", "gated", "cnn", "confidence_residual", "smiles_anchor_residual"),
        default="current",
    )
    parser.add_argument(
        "--token-feature-normalize-mode",
        type=str,
        choices=("per_molecule", "none", "dataset"),
        default="per_molecule",
    )
    parser.add_argument(
        "--token-weight-mode",
        type=str,
        choices=("abs", "positive"),
        default="abs",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--validation-fraction-within-train", type=float, default=0.125)
    parser.add_argument(
        "--fold-indices",
        nargs="*",
        type=int,
        default=(),
        help="Optional 1-based fold numbers to run, for example: --fold-indices 4 5",
    )
    parser.add_argument(
        "--skip-completed-folds",
        action="store_true",
        help="Skip folds that already have test_metrics.csv and reuse their recorded metrics.",
    )
    parser.add_argument(
        "--token-features",
        nargs="*",
        default=(),
        help="Optional token-level features used by the model.",
    )
    parser.add_argument(
        "--molecule-features",
        nargs="*",
        default=(),
        help="Optional molecule-level features used by the model.",
    )
    parser.add_argument("--charge-xyz-dir", type=Path, default=None)
    args = parser.parse_args()
    return CrossValidationConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        pretrained_model_name=args.pretrained_model_name,
        model_dir=args.model_dir,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        early_fusion=args.early_fusion,
        fusion_learning_rate=args.fusion_learning_rate,
        descriptor_projection_dim=args.descriptor_projection_dim,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        token_feature_names=tuple(args.token_features),
        molecule_feature_names=tuple(args.molecule_features),
        fusion_type=args.fusion_type,
        gate_bias_init=args.gate_bias_init,
        token_feature_normalize_mode=args.token_feature_normalize_mode,
        token_weight_mode=args.token_weight_mode,
        charge_xyz_dir=args.charge_xyz_dir,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        n_splits=args.n_splits,
        validation_fraction_within_train=args.validation_fraction_within_train,
        fold_indices=tuple(args.fold_indices),
        skip_completed_folds=args.skip_completed_folds,
    )


def main() -> None:
    folds_path, summary_path = run_cross_validation(parse_args())
    print(f"Saved fold metrics to {folds_path}")
    print(f"Saved cross-validation summary to {summary_path}")


if __name__ == "__main__":
    main()
