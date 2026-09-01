from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .compute_occlusion import (
    create_loader,
    load_lightning_model,
    load_split_dataframe,
)
from .features import (
    available_feature_names,
    available_molecule_feature_names,
    molecule_feature_labels,
)


@dataclass(slots=True)
class GlobalOcclusionConfig:
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
    row_indices: tuple[int, ...] = ()
    target_class: int = 1


def compute_global_occlusion_for_split(config: GlobalOcclusionConfig) -> tuple[Path, Path]:
    if not config.molecule_feature_names:
        raise ValueError("Global occlusion requires at least one molecule-level feature set.")

    frame = load_split_dataframe(config.output_dir, config.split_name)
    source_indices = frame.index.tolist()
    if config.row_indices:
        selected_rows = list(config.row_indices)
        frame = frame.iloc[selected_rows].reset_index(drop=True)
        source_indices = selected_rows

    lightning_model = load_lightning_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    lightning_model.to(device)

    loader = create_loader(
        encoder=lightning_model.model.encoder,
        frame=frame,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        token_feature_names=config.token_feature_names,
        molecule_feature_names=config.molecule_feature_names,
        charge_xyz_dir=config.charge_xyz_dir,
    )

    feature_labels = molecule_feature_labels(config.molecule_feature_names)
    suffix = "all" if not config.row_indices else "rows_" + "_".join(str(idx) for idx in config.row_indices)
    out_dir = config.output_dir / "global_occlusion" / f"{config.split_name}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    running_index = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_features = batch.get("token_features")
        if token_features is not None:
            token_features = token_features.to(device)
        molecule_features = batch.get("molecule_features")
        if molecule_features is None:
            raise ValueError("The selected checkpoint/input batch does not contain molecule-level features.")
        molecule_features = molecule_features.to(device)

        with torch.no_grad():
            logits = lightning_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_features=token_features,
                molecule_features=molecule_features,
            )
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        for batch_index in range(input_ids.shape[0]):
            row = frame.iloc[running_index]
            source_row_index = source_indices[running_index]
            smiles = str(row["smiles"])
            base_prob = float(probabilities[batch_index, config.target_class].item())

            feature_vector = molecule_features[batch_index].clone()
            feature_scores: list[float] = []
            for feature_index, feature_name in enumerate(feature_labels):
                occluded_vector = feature_vector.clone()
                occluded_vector[feature_index] = 0.0
                with torch.no_grad():
                    occluded_logits = lightning_model.model(
                        input_ids=input_ids[batch_index].unsqueeze(0),
                        attention_mask=attention_mask[batch_index].unsqueeze(0),
                        token_features=(
                            token_features[batch_index].unsqueeze(0) if token_features is not None else None
                        ),
                        molecule_features=occluded_vector.unsqueeze(0),
                    )
                    occluded_prob = torch.softmax(occluded_logits, dim=-1)[0, config.target_class].item()
                score = float(base_prob - occluded_prob)
                feature_scores.append(score)
                detail_rows.append(
                    {
                        "row_index": running_index,
                        "source_row_index": source_row_index,
                        "smiles": smiles,
                        "label": int(labels[batch_index].item()),
                        "predicted_label": int(predictions[batch_index].item()),
                        "positive_probability": base_prob,
                        "feature_index": feature_index,
                        "feature_name": feature_name,
                        "feature_value": float(feature_vector[feature_index].item()),
                        "occlusion_score": score,
                    }
                )

            summary_rows.append(
                {
                    "row_index": running_index,
                    "source_row_index": source_row_index,
                    "smiles": smiles,
                    "label": int(labels[batch_index].item()),
                    "predicted_label": int(predictions[batch_index].item()),
                    "positive_probability": base_prob,
                    **{f"{name}_score": score for name, score in zip(feature_labels, feature_scores, strict=True)},
                }
            )
            running_index += 1

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = (
        detail_df.groupby("feature_name", as_index=False)
        .agg(
            mean_occlusion_score=("occlusion_score", "mean"),
            mean_absolute_occlusion_score=("occlusion_score", lambda values: values.abs().mean()),
            max_absolute_occlusion_score=("occlusion_score", lambda values: values.abs().max()),
        )
        .sort_values("mean_absolute_occlusion_score", ascending=False)
        .reset_index(drop=True)
    )

    summary_path = out_dir / f"{config.split_name}_global_occlusion_summary.csv"
    detail_path = out_dir / f"{config.split_name}_global_occlusion_details.csv"
    aggregate_path = out_dir / f"{config.split_name}_global_occlusion_aggregate.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    return aggregate_path, detail_path


def parse_args() -> GlobalOcclusionConfig:
    parser = argparse.ArgumentParser(
        description="Compute descriptor-level occlusion importance for molecule-level toxicity features."
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", type=str, default="test", choices=("train", "validation", "test"))
    parser.add_argument("--pretrained-model-name", type=str, default="fabikru/MolEncoder")
    parser.add_argument("--model-dir", type=Path, default=Path("models/molencoder"))
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--row-indices", nargs="*", type=int, default=())
    parser.add_argument(
        "--token-features",
        nargs="*",
        default=(),
        choices=available_feature_names(),
        help="Optional token-level features used by the saved checkpoint.",
    )
    parser.add_argument(
        "--molecule-features",
        nargs="*",
        default=(),
        choices=available_molecule_feature_names(),
        help="Molecule-level feature sets used by the saved checkpoint.",
    )
    parser.add_argument(
        "--charge-xyz-dir",
        type=Path,
        default=None,
        help="Optional directory with xyz files and an outs/ subdirectory for ORCA/XTB feature lookup.",
    )
    args = parser.parse_args()
    return GlobalOcclusionConfig(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        split_name=args.split_name,
        pretrained_model_name=args.pretrained_model_name,
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_sequence_length=args.max_sequence_length,
        token_feature_names=tuple(args.token_features),
        molecule_feature_names=tuple(args.molecule_features),
        charge_xyz_dir=args.charge_xyz_dir,
        row_indices=tuple(args.row_indices),
        target_class=args.target_class,
    )


def main() -> None:
    aggregate_path, detail_path = compute_global_occlusion_for_split(parse_args())
    print(f"Saved global occlusion aggregate to {aggregate_path}")
    print(f"Saved global occlusion details to {detail_path}")


if __name__ == "__main__":
    main()
