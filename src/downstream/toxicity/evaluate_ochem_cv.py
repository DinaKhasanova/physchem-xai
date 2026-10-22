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


def _load_ochem_predictions(config: OchemCvConfig) -> dict[str, dict[str, int]]:
    ochem_df = pd.read_excel(config.ochem_path)
    if config.smiles_column not in ochem_df.columns:
        raise ValueError(f"{config.ochem_path} does not contain the column '{config.smiles_column}'.")
    assay_columns = [column for column in ochem_df.columns if column not in {config.smiles_column, config.molecule_id_column}]
    if not assay_columns:
        raise ValueError("No assay columns found in the OCHEM file.")

    def alert_count(row: pd.Series) -> int:
        count = 0
        for value in row[assay_columns]:
            if pd.isna(value):
                continue
            if str(value).strip() == "-":
                continue
            try:
                if float(value) == 1.0:
                    count += 1
            except (TypeError, ValueError):
                continue
        return count

    counts = ochem_df.apply(alert_count, axis=1).astype(int)
    smiles_df = pd.DataFrame(
        {
            "smiles": ochem_df[config.smiles_column].astype(str),
            # Binary positive label (>=1 ToxAlert fired): used for balanced
            # accuracy, sensitivity, specificity, precision and F1.
            "label": (counts > 0).astype(int),
            # Number of ToxAlerts fired: used as the continuous ranking score
            # for ROC AUC. A binary 0/1 score makes ROC AUC degenerate (it
            # collapses to balanced accuracy), so the alert count is used.
            "count": counts,
        }
    )
    cleaned = clean_smiles_dataframe(smiles_df)
    return {
        str(smiles): {"label": int(label), "count": int(count)}
        for smiles, label, count in zip(
            cleaned["smiles"].tolist(),
            cleaned["label"].astype(int).tolist(),
            cleaned["count"].astype(int).tolist(),
            strict=True,
        )
    }


def evaluate_ochem_cv(config: OchemCvConfig) -> tuple[Path, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_toxicity_dataframe(config.dataset_path)
    ochem_predictions = _load_ochem_predictions(config)
    labels = frame["label"].to_numpy()

    outer_cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    fold_metrics: list[dict[str, float | int]] = []

    for fold_index, (_, test_indices) in enumerate(outer_cv.split(frame, labels), start=1):
        test_frame = frame.iloc[test_indices].reset_index(drop=True)
        test_smiles = test_frame["smiles"].tolist()
        # Binary prediction (>=1 alert) for threshold metrics; alert count as
        # the continuous score for ROC AUC ranking.
        predictions = [int(ochem_predictions.get(smiles, {"label": 0})["label"]) for smiles in test_smiles]
        positive_scores = [float(ochem_predictions.get(smiles, {"count": 0})["count"]) for smiles in test_smiles]
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
