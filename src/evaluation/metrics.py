"""Расширенные метрики для BCI P300 классификации."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def binary_epoch_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Полный набор метрик для бинарной P300 классификации.

    Parameters
    ----------
    y_true : np.ndarray
        Истинные метки (0/1)
    y_pred : np.ndarray
        Предсказанные метки (0/1)
    y_prob : np.ndarray | None
        Вероятности P(target) для ROC/PR-AUC

    Returns
    -------
    dict
        Словарь метрик

    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1],
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_nontarget": float(precision[0]),
        "precision_target": float(precision[1]),
        "recall_nontarget": float(recall[0]),
        "recall_target": float(recall[1]),
        "f1_nontarget": float(f1[0]),
        "f1_target": float(f1[1]),
        "specificity": float(
            cm[0, 0] / (cm[0, 0] + cm[0, 1])
            if (cm[0, 0] + cm[0, 1]) > 0
            else 0.0,
        ),
        "confusion_matrix": cm,
        "n_target": int(support[1]) if support is not None else 0,
        "n_nontarget": int(support[0]) if support is not None else 0,
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.0

        try:
            metrics["pr_auc"] = float(
                average_precision_score(y_true, y_prob),
            )
        except ValueError:
            metrics["pr_auc"] = 0.0

    return metrics


def letter_prediction_metrics(
    true_letters: list[str],
    predicted_letters: list[str],
    all_scores: list[dict[str, float]] | None = None,
) -> dict:
    """Метрики для letter-level предсказания.

    Parameters
    ----------
    true_letters : list[str]
        Истинные целевые буквы
    predicted_letters : list[str]
        Предсказанные буквы
    all_scores : list[dict[str, float]] | None
        Scores для всех букв на каждом trial (для top-K)

    Returns
    -------
    dict
        top1, top3, mrr и т.д.

    """
    n = len(true_letters)
    correct_top1 = sum(
        1 for t, p in zip(true_letters, predicted_letters)
        if t == p
    )

    metrics = {
        "top1_accuracy": correct_top1 / n if n > 0 else 0.0,
        "n_trials": n,
        "correct_top1": correct_top1,
    }

    if all_scores is not None:
        correct_top3 = 0
        reciprocal_ranks = []

        for true_letter, scores in zip(true_letters, all_scores):
            sorted_letters = sorted(
                scores, key=scores.get, reverse=True,
            )
            top3 = sorted_letters[:3]
            correct_top3 += int(true_letter in top3)

            if true_letter in sorted_letters:
                rank = sorted_letters.index(true_letter) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        metrics["top3_accuracy"] = correct_top3 / n if n > 0 else 0.0
        metrics["mrr"] = float(np.mean(reciprocal_ranks))

    return metrics


def print_metrics_summary(metrics: dict, title: str = "Metrics") -> None:
    """Печатает метрики в читаемом формате."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")

    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"  Confusion Matrix:")
            print(f"    {value}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"{'=' * 50}\n")
