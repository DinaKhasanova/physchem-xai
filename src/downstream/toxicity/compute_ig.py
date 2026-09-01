from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from molencoder.config import MolEncoderConfig
from molencoder.model import MolEncoderWrapper

from .compute_occlusion import (
    LegacyClsToxicityClassifier,
    atom_scores_from_char_scores,
    load_split_dataframe,
    save_smiles_token_heatmap_plot,
)
from .dataset import ToxicityCollator, ToxicityDataset
from .features import available_feature_names, available_molecule_feature_names, molecule_feature_dimension
from .finetune import ToxicityClassifier
from .train import ToxicityLightningModule


@dataclass(slots=True)
class IgConfig:
    checkpoint_path: Path
    output_dir: Path
    split_name: str = "test"
    pretrained_model_name: str = "fabikru/MolEncoder"
    model_dir: Path | None = Path("models/molencoder")
    max_sequence_length: int = 256
    token_feature_names: tuple[str, ...] = ()
    molecule_feature_names: tuple[str, ...] = ()
    charge_xyz_dir: Path | None = None
    row_indices: tuple[int, ...] = ()
    target_class: int = 1
    ig_steps: int = 32


def json_list(values: list[float] | list[str]) -> str:
    return json.dumps(values, ensure_ascii=True)


def load_lightning_model(config: IgConfig) -> ToxicityLightningModule:
    encoder_config = MolEncoderConfig(
        pretrained_model_name=config.pretrained_model_name,
        max_sequence_length=config.max_sequence_length,
        num_labels=2,
        model_cache_dir=config.model_dir,
    )
    encoder = MolEncoderWrapper(encoder_config)
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    classifier_input_size = int(checkpoint["state_dict"]["model.classifier.0.weight"].shape[0])
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


def classifier_from_token_states(
    classifier: LegacyClsToxicityClassifier | ToxicityClassifier,
    token_states: torch.Tensor,
    attention_mask: torch.Tensor,
    token_features: torch.Tensor | None,
    molecule_features: torch.Tensor | None,
) -> torch.Tensor:
    cls_embedding = token_states[:, 0, :]
    if isinstance(classifier, LegacyClsToxicityClassifier):
        return classifier.classifier(cls_embedding)

    if token_features is None or token_features.shape[-1] == 0:
        pooled_context = torch.zeros_like(cls_embedding)
        feature_stats = torch.zeros(cls_embedding.shape[0], 0, device=cls_embedding.device, dtype=cls_embedding.dtype)
    else:
        feature_weights, feature_stats = classifier.summarize_token_features(token_features, attention_mask)
        pooled_context = (token_states * feature_weights.to(token_states.dtype)).sum(dim=1)
        pooled_context = classifier.charge_context_scale * pooled_context
        feature_stats = classifier.feature_stats_scale * feature_stats.to(cls_embedding.dtype)

    if molecule_features is None:
        molecule_features = torch.zeros(cls_embedding.shape[0], 0, device=cls_embedding.device, dtype=cls_embedding.dtype)
    else:
        molecule_features = molecule_features.to(cls_embedding.dtype)
    classifier_input = torch.cat([cls_embedding, pooled_context, feature_stats, molecule_features], dim=-1)
    return classifier.classifier(classifier_input)


def integrated_gradients_for_batch(
    lightning_model: ToxicityLightningModule,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_features: torch.Tensor | None,
    molecule_features: torch.Tensor | None,
    target_class: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    classifier = lightning_model.model
    hf_model = classifier.encoder.encoder
    embedding_layer = hf_model.get_input_embeddings()

    with torch.no_grad():
        input_embeds = embedding_layer(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)
        base_logits = classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_features=token_features,
            molecule_features=molecule_features,
        )
        base_probabilities = torch.softmax(base_logits, dim=-1)
        base_predictions = torch.argmax(base_probabilities, dim=-1)

    total_gradients = torch.zeros_like(input_embeds)
    for alpha in torch.linspace(0.0, 1.0, steps + 1, device=input_embeds.device)[1:]:
        scaled_embeds = (baseline_embeds + alpha * (input_embeds - baseline_embeds)).detach()
        scaled_embeds.requires_grad_(True)
        token_states = hf_model(inputs_embeds=scaled_embeds, attention_mask=attention_mask).last_hidden_state
        logits = classifier_from_token_states(
            classifier=classifier,
            token_states=token_states,
            attention_mask=attention_mask,
            token_features=token_features,
            molecule_features=molecule_features,
        )
        target = logits[:, target_class].sum()
        gradients = torch.autograd.grad(target, scaled_embeds)[0]
        total_gradients += gradients.detach()

    avg_gradients = total_gradients / float(steps)
    attributions = (input_embeds - baseline_embeds) * avg_gradients
    token_scores = attributions.sum(dim=-1)
    return token_scores.detach(), base_probabilities.detach(), base_predictions.detach()


def compute_ig_for_split(config: IgConfig) -> tuple[Path, Path]:
    frame = load_split_dataframe(config.output_dir, config.split_name)
    source_indices = frame.index.tolist()
    if config.row_indices:
        selected_rows = list(config.row_indices)
        frame = frame.iloc[selected_rows].reset_index(drop=True)
        source_indices = selected_rows

    lightning_model = load_lightning_model(config)
    device = torch.device("cpu")
    lightning_model.to(device)

    collator = ToxicityCollator(
        lightning_model.model.encoder,
        feature_names=list(config.token_feature_names),
        molecule_feature_names=list(config.molecule_feature_names),
        charge_xyz_dir=config.charge_xyz_dir,
    )
    loader = torch.utils.data.DataLoader(ToxicityDataset(frame), batch_size=1, shuffle=False, collate_fn=collator)

    subset_suffix = "all" if not config.row_indices else "rows_" + "_".join(str(idx) for idx in config.row_indices)
    ig_dir = config.output_dir / "ig" / f"{config.split_name}_zero_{subset_suffix}"
    images_dir = ig_dir / "images"
    ig_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    token_rows: list[dict[str, object]] = []
    for running_index, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_features = batch.get("token_features")
        if token_features is not None:
            token_features = token_features.to(device)
        molecule_features = batch.get("molecule_features")
        if molecule_features is not None:
            molecule_features = molecule_features.to(device)

        token_scores, probabilities, predictions = integrated_gradients_for_batch(
            lightning_model=lightning_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_features=token_features,
            molecule_features=molecule_features,
            target_class=config.target_class,
            steps=config.ig_steps,
        )

        row = frame.iloc[running_index]
        source_row_index = source_indices[running_index]
        smiles = str(row["smiles"])
        valid_length = int(attention_mask[0].sum().item())
        tokens = lightning_model.model.encoder.tokenizer.convert_ids_to_tokens(input_ids[0, :valid_length].tolist())
        score_values = [float(score) for score in token_scores[0, :valid_length].cpu().tolist()]
        if score_values:
            score_values[0] = 0.0
            score_values[-1] = 0.0
        char_scores = score_values[1:-1]
        atom_scores = atom_scores_from_char_scores(smiles, char_scores)

        image_path = images_dir / f"{source_row_index:04d}.png"
        save_smiles_token_heatmap_plot(
            smiles=smiles,
            atom_scores=atom_scores,
            token_chars=list(smiles),
            token_scores=char_scores,
            output_path=image_path,
            title=f"Integrated Gradients, {config.ig_steps} steps",
        )

        summary_rows.append(
            {
                "row_index": running_index,
                "source_row_index": source_row_index,
                "smiles": smiles,
                "label": int(labels[0].item()),
                "predicted_label": int(predictions[0].item()),
                "positive_probability": float(probabilities[0, config.target_class].item()),
                "tokens": json_list(tokens),
                "token_scores": json_list(score_values),
                "char_scores": json_list(char_scores),
                "image_path": str(image_path.resolve()),
            }
        )
        for token_index, (token, token_score) in enumerate(zip(tokens, score_values, strict=True)):
            token_rows.append(
                {
                    "row_index": running_index,
                    "source_row_index": source_row_index,
                    "token_index": token_index,
                    "token": token,
                    "token_score": float(token_score),
                }
            )

    summary_path = ig_dir / f"{config.split_name}_ig_summary.csv"
    token_path = ig_dir / f"{config.split_name}_ig_token_scores.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(token_rows).to_csv(token_path, index=False)
    return summary_path, token_path


def parse_args() -> IgConfig:
    parser = argparse.ArgumentParser(description="Compute embedding-level Integrated Gradients for toxicity.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", type=str, default="test", choices=("train", "validation", "test"))
    parser.add_argument("--pretrained-model-name", type=str, default="fabikru/MolEncoder")
    parser.add_argument("--model-dir", type=Path, default=Path("models/molencoder"))
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--row-indices", nargs="*", type=int, default=())
    parser.add_argument("--token-features", nargs="*", default=(), choices=available_feature_names())
    parser.add_argument("--molecule-features", nargs="*", default=(), choices=available_molecule_feature_names())
    parser.add_argument("--charge-xyz-dir", type=Path, default=None)
    args = parser.parse_args()
    return IgConfig(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        split_name=args.split_name,
        pretrained_model_name=args.pretrained_model_name,
        model_dir=args.model_dir,
        max_sequence_length=args.max_sequence_length,
        token_feature_names=tuple(args.token_features),
        molecule_feature_names=tuple(args.molecule_features),
        charge_xyz_dir=args.charge_xyz_dir,
        row_indices=tuple(args.row_indices),
        target_class=args.target_class,
        ig_steps=args.ig_steps,
    )


def main() -> None:
    summary_path, token_path = compute_ig_for_split(parse_args())
    print(f"Saved IG summary to {summary_path}")
    print(f"Saved IG token scores to {token_path}")


if __name__ == "__main__":
    main()
