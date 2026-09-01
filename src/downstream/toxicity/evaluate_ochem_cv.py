from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .dataset import clean_smiles_dataframe, load_toxicity_dataframe
from .evaluate import classification_metrics_from_predictions, save_metrics, summarize_cross_validation_metrics


@dataclass(slots=True)
class OchemCvConfig:
    dataset_path: Path
    ochem_path: Path
    output_dir: Path
    smiles_column: str = "SMILES"
    molecule_id_column: str = "MOLECULEID"
    seed: int = 42
    n_splits: int = 5


def _load_ochem_predictions(config: OchemCvConfig) -> dict[str, int]:
    ochem_df = pd.read_excel(config.ochem_path)
    if config.smiles_column not in ochem_df.columns:
        raise ValueError(f"{config.ochem_path} does not contain the column '{config.smiles_column}'.")
    assay_columns = [column for column in ochem_df.columns if column not in {config.smiles_column, config.molecule_id_column}]
    if not assay_columns:
        raise ValueError("No assay columns found in the OCHEM file.")

    def to_binary_prediction(row: pd.Series) -> int:
        for value in row[assay_columns]:
            if pd.isna(value):
                continue
            if str(value).strip() == "-":
                continue
            try:
                if float(value) == 1.0:
                    return 1
            except (TypeError, ValueError):
                continue
        return 0

    smiles_df = pd.DataFrame(
        {
            "smiles": ochem_df[config.smiles_column].astype(str),
            "label": ochem_df.apply(to_binary_prediction, axis=1).astype(int),
        }
    )
    cleaned = clean_smiles_dataframe(smiles_df)
    return dict(zip(cleaned["smiles"].tolist(), cleaned["label"].astype(int).tolist(), strict=True))


def evaluate_ochem_cv(config: OchemCvConfig) -> tuple[Path, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_toxicity_dataframe(config.dataset_path)
    ochem_predictions = _load_ochem_predictions(config)
    labels = frame["label"].to_numpy()

    outer_cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    fold_metrics: list[dict[str, float | int]] = []

    for fold_index, (_, test_indices) in enumerate(outer_cv.split(frame, labels), start=1):
        test_frame = frame.iloc[test_indices].reset_index(drop=True)
        predictions = [int(ochem_predictions.get(smiles, 0)) for smiles in test_frame["smiles"].tolist()]
        positive_scores = [float(prediction) for prediction in predictions]
        metrics = classification_metrics_from_predictions(
            labels=test_frame["label"].astype(int).tolist(),
            positive_scores=positive_scores,
            predictions=predictions,
        )
        metrics["fold"] = fold_index
        metrics["n_test"] = int(len(test_frame))
        fold_metrics.append(metrics)

    folds_path = config.output_dir / "ochem_cross_validation_fold_metrics.csv"
    pd.DataFrame(fold_metrics).to_csv(folds_path, index=False)
    summary = summarize_cross_validation_metrics(fold_metrics)
    summary.update(
        {
            "n_samples": int(len(frame)),
            "n_positive": int(frame["label"].sum()),
            "n_negative": int((frame["label"] == 0).sum()),
        }
    )
    summary_path_csv, _ = save_metrics(config.output_dir, summary, filename_stem="ochem_cross_validation_summary")
    return folds_path, summary_path_csv


def parse_args() -> OchemCvConfig:
    parser = argparse.ArgumentParser(description="Evaluate the fixed OCHEM toxicophore rule with stratified cross-validation folds.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--ochem-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smiles-column", type=str, default="SMILES")
    parser.add_argument("--molecule-id-column", type=str, default="MOLECULEID")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    return OchemCvConfig(
        dataset_path=args.dataset_path,
        ochem_path=args.ochem_path,
        output_dir=args.output_dir,
        smiles_column=args.smiles_column,
        molecule_id_column=args.molecule_id_column,
        seed=args.seed,
        n_splits=args.n_splits,
    )


def main() -> None:
    folds_path, summary_path = evaluate_ochem_cv(parse_args())
    print(f"Saved OCHEM fold metrics to {folds_path}")
    print(f"Saved OCHEM summary to {summary_path}")


if __name__ == "__main__":
    main()
