from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from molencoder.config import MolEncoderConfig
from molencoder.model import MolEncoderWrapper

from .dataset import ToxicityCollator, ToxicityDataset, load_toxicity_dataframe
from .features import (
    UNSUPPORTED_TOKEN_CHARS,
    available_feature_names,
    available_molecule_feature_names,
    molecule_feature_dimension,
)
from .finetune import ToxicityClassifier
from .train import ToxicityLightningModule


@dataclass(slots=True)
class OcclusionConfig:
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


def json_list(values: list[float] | list[str]) -> str:
    return json.dumps(values, ensure_ascii=True)


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
):
    collator = ToxicityCollator(
        encoder,
        feature_names=list(token_feature_names),
        molecule_feature_names=list(molecule_feature_names),
        charge_xyz_dir=charge_xyz_dir,
    )
    return torch.utils.data.DataLoader(
        ToxicityDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )


def load_lightning_model(config: OcclusionConfig) -> ToxicityLightningModule:
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


def normalize_relative_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if abs(max_score - min_score) < 1e-12:
        return [0.0 for _ in scores]
    return [((score - min_score) / (max_score - min_score)) * 2.0 - 1.0 for score in scores]


def atom_scores_from_char_scores(smiles: str, char_scores: list[float]) -> list[float]:
    atom_scores: list[float] = []
    token_index = 0

    while token_index < len(smiles):
        token = smiles[token_index]

        if token_index >= len(char_scores):
            atom_scores.append(0.0)
            token_index += 1
            continue

        if token == "[":
            end = smiles.find("]", token_index)
            if end == -1:
                token_index += 1
                continue
            block_scores = char_scores[token_index : end + 1]
            atom_scores.append(float(sum(block_scores)))
            token_index = end + 1
            continue

        if token == "C" and token_index + 1 < len(smiles) and smiles[token_index + 1] == "l":
            block_scores = char_scores[token_index : token_index + 2]
            atom_scores.append(float(sum(block_scores)))
            token_index += 2
            continue

        if token == "B" and token_index + 1 < len(smiles) and smiles[token_index + 1] == "r":
            block_scores = char_scores[token_index : token_index + 2]
            atom_scores.append(float(sum(block_scores)))
            token_index += 2
            continue

        if token in UNSUPPORTED_TOKEN_CHARS or not token.isalpha():
            token_index += 1
            continue

        atom_scores.append(float(char_scores[token_index]))
        token_index += 1

    return atom_scores


def _draw_molecule_png_bytes(smiles: str, atom_scores: list[float] | None = None) -> bytes | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    drawer = rdMolDraw2D.MolDraw2DCairo(900, 500)
    options = drawer.drawOptions()
    options.padding = 0.08
    highlight_atoms: list[int] = []
    highlight_colors: dict[int, tuple[float, float, float]] = {}
    if atom_scores is not None:
        n_atoms = mol.GetNumAtoms()
        if len(atom_scores) < n_atoms:
            atom_scores = atom_scores + [0.0] * (n_atoms - len(atom_scores))
        else:
            atom_scores = atom_scores[:n_atoms]
        normalized_scores = normalize_relative_scores(atom_scores)
        highlight_atoms = [idx for idx, score in enumerate(normalized_scores) if abs(score) > 1e-8]
        for idx, score in enumerate(normalized_scores):
            if abs(score) <= 1e-8:
                continue
            intensity = min(abs(score), 1.0)
            if score >= 0:
                highlight_colors[idx] = (0.35 + 0.4 * (1.0 - intensity), 1.0, 0.35 + 0.4 * (1.0 - intensity))
            else:
                highlight_colors[idx] = (1.0, 0.35 + 0.4 * (1.0 - intensity), 0.35 + 0.4 * (1.0 - intensity))

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def save_smiles_token_heatmap_plot(
    smiles: str,
    atom_scores: list[float],
    token_chars: list[str],
    token_scores: list[float],
    output_path: str | Path,
    title: str = "Occlusion Token Importance",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    png_bytes = _draw_molecule_png_bytes(smiles, atom_scores=atom_scores)
    if png_bytes is None:
        return

    molecule_image = plt.imread(BytesIO(png_bytes), format="png")
    fig_width = max(8, len(token_chars) * 0.45)
    fig = plt.figure(figsize=(fig_width, 7.5), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.6], hspace=0.08)

    ax_mol = fig.add_subplot(gs[0, 0])
    ax_mol.imshow(molecule_image)
    ax_mol.axis("off")

    ax_heat = fig.add_subplot(gs[1, 0])
    if token_scores:
        values = np.array(token_scores, dtype=float)
        min_value = float(values.min())
        max_value = float(values.max())
        if abs(max_value - min_value) < 1e-12:
            normalized = np.zeros_like(values)
        else:
            normalized = ((values - min_value) / (max_value - min_value)) * 2.0 - 1.0
        heat = normalized[None, :]
        ax_heat.imshow(heat, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
        ax_heat.set_xticks(range(len(token_chars)))
        ax_heat.set_xticklabels(token_chars, fontsize=10)
        ax_heat.set_yticks([])
        for idx, (char, raw_value, norm_value) in enumerate(zip(token_chars, values, normalized, strict=True)):
            text_color = "black" if abs(norm_value) < 0.45 else "white"
            ax_heat.text(idx, 0, char, ha="center", va="center", fontsize=12, color=text_color)
            ax_heat.text(
                idx,
                0.42,
                f"{raw_value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                rotation=90,
            )
        ax_heat.set_title(title, fontsize=12, pad=8)
    else:
        ax_heat.axis("off")

    fig.suptitle(smiles, fontsize=13, y=0.99)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compute_occlusion_for_split(config: OcclusionConfig) -> tuple[Path, Path]:
    frame = load_split_dataframe(config.output_dir, config.split_name)
    if config.dataset_path is not None:
        _ = load_toxicity_dataframe(config.dataset_path)
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

    subset_suffix = "all" if not config.row_indices else "rows_" + "_".join(str(idx) for idx in config.row_indices)
    occ_dir = config.output_dir / "occlusion" / f"{config.split_name}_pad_{subset_suffix}"
    images_dir = occ_dir / "images"
    occ_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    pad_token_id = lightning_model.model.encoder.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    summary_rows: list[dict[str, object]] = []
    token_rows: list[dict[str, object]] = []

    running_index = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_features = batch.get("token_features")
        if token_features is not None:
            token_features = token_features.to(device)
        molecule_features = batch.get("molecule_features")
        if molecule_features is not None:
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

            token_ids = input_ids[batch_index].clone()
            mask_ids = attention_mask[batch_index].clone()
            feature_matrix = token_features[batch_index].clone() if token_features is not None else None
            molecule_feature_vector = (
                molecule_features[batch_index].clone() if molecule_features is not None else None
            )
            base_prob = float(probabilities[batch_index, config.target_class].item())
            valid_length = int(mask_ids.sum().item())
            tokens = lightning_model.model.encoder.tokenizer.convert_ids_to_tokens(token_ids[:valid_length].tolist())

            token_scores = [0.0] * valid_length
            for token_position in range(1, max(valid_length - 1, 1)):
                occluded_ids = token_ids.clone()
                occluded_ids[token_position] = pad_token_id
                occluded_features = feature_matrix.clone() if feature_matrix is not None else None
                if occluded_features is not None:
                    occluded_features[token_position, :] = 0.0

                with torch.no_grad():
                    occluded_logits = lightning_model.model(
                        input_ids=occluded_ids.unsqueeze(0),
                        attention_mask=mask_ids.unsqueeze(0),
                        token_features=occluded_features.unsqueeze(0) if occluded_features is not None else None,
                        molecule_features=(
                            molecule_feature_vector.unsqueeze(0)
                            if molecule_feature_vector is not None
                            else None
                        ),
                    )
                    occluded_prob = torch.softmax(occluded_logits, dim=-1)[0, config.target_class].item()
                token_scores[token_position] = float(base_prob - occluded_prob)

            if valid_length > 0:
                token_scores[0] = 0.0
                token_scores[valid_length - 1] = 0.0

            score_values = [float(score) for score in token_scores[:valid_length]]
            char_scores = score_values[1:-1]

            image_path_str = ""
            should_save_image = config.image_label_filter is None or int(labels[batch_index].item()) == config.image_label_filter
            if should_save_image:
                image_path = images_dir / f"{source_row_index:04d}.png"
                atom_scores = atom_scores_from_char_scores(smiles, char_scores)
                save_smiles_token_heatmap_plot(
                    smiles=smiles,
                    atom_scores=atom_scores,
                    token_chars=list(smiles),
                    token_scores=char_scores,
                    output_path=image_path,
                )
                image_path_str = str(image_path.resolve())

            summary_rows.append(
                {
                    "row_index": running_index,
                    "source_row_index": source_row_index,
                    "smiles": smiles,
                    "label": int(labels[batch_index].item()),
                    "predicted_label": int(predictions[batch_index].item()),
                    "positive_probability": base_prob,
                    "tokens": json_list(tokens),
                    "token_scores": json_list(score_values),
                    "char_scores": json_list(char_scores),
                    "image_path": image_path_str,
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
            running_index += 1

    summary_path = occ_dir / f"{config.split_name}_occlusion_summary.csv"
    token_path = occ_dir / f"{config.split_name}_occlusion_token_scores.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(token_rows).to_csv(token_path, index=False)
    return summary_path, token_path


def parse_args() -> OcclusionConfig:
    parser = argparse.ArgumentParser(description="Compute token occlusion importance for a saved toxicity checkpoint.")
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
    parser.add_argument("--image-label-filter", type=int, default=None, choices=(0, 1))
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
        help="Optional molecule-level feature sets used by the saved checkpoint.",
    )
    parser.add_argument(
        "--charge-xyz-dir",
        type=Path,
        default=None,
        help="Optional directory with xyz files and an outs/ subdirectory for ORCA/XTB feature lookup.",
    )
    args = parser.parse_args()
    return OcclusionConfig(
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
        image_label_filter=args.image_label_filter,
        row_indices=tuple(args.row_indices),
        target_class=args.target_class,
    )


def main() -> None:
    summary_path, token_path = compute_occlusion_for_split(parse_args())
    print(f"Saved occlusion summary to {summary_path}")
    print(f"Saved occlusion token scores to {token_path}")


if __name__ == "__main__":
    main()
