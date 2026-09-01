from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score


def binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == labels).float().mean()
    return {"accuracy": accuracy}


def classification_metrics_from_predictions(
    labels: list[int],
    positive_scores: list[float],
    predictions: list[int],
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )
    roc_auc = roc_auc_score(labels, positive_scores)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_metrics(
    output_dir: Path,
    metrics: dict[str, float | int],
    filename_stem: str = "test_metrics",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{filename_stem}.csv"
    json_path = output_dir / f"{filename_stem}.json"
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    pd.Series(metrics).to_json(json_path, indent=2)
    return csv_path, json_path


def summarize_cross_validation_metrics(
    fold_metrics: list[dict[str, float | int]],
    metric_names: tuple[str, ...] = ("roc_auc", "balanced_accuracy"),
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"n_folds": len(fold_metrics)}
    if not fold_metrics:
        return summary

    for metric_name in metric_names:
        values = [float(fold_metrics_item[metric_name]) for fold_metrics_item in fold_metrics]
        metric_mean = mean(values)
        metric_sd = stdev(values) if len(values) > 1 else 0.0
        metric_se = metric_sd / (len(values) ** 0.5) if values else 0.0
        ci95_half_width = 1.96 * metric_se
        summary[f"{metric_name}_mean"] = metric_mean
        summary[f"{metric_name}_sd"] = metric_sd
        summary[f"{metric_name}_se"] = metric_se
        summary[f"{metric_name}_ci95_low"] = metric_mean - ci95_half_width
        summary[f"{metric_name}_ci95_high"] = metric_mean + ci95_half_width
    return summary


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)
