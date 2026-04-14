"""Визуализация ЭЭГ данных и результатов P300 классификации."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_erp_comparison(
    target_epochs: np.ndarray,
    nontarget_epochs: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
    channels_to_plot: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grand average ERP: target vs nontarget.

    Parameters
    ----------
    target_epochs : np.ndarray
        shape (n_target, n_channels, n_times)
    nontarget_epochs : np.ndarray
        shape (n_nontarget, n_channels, n_times)
    times : np.ndarray
        Временная ось в секундах
    channel_names : list[str]
        Имена каналов
    channels_to_plot : list[str] | None
        Каналы для отображения (None -> Pz, Cz, Fz)
    save_path : str | Path | None
        Путь для сохранения

    """
    if channels_to_plot is None:
        channels_to_plot = ["Pz", "Cz", "Fz"]

    ch_indices = []
    for ch in channels_to_plot:
        if ch in channel_names:
            ch_indices.append(channel_names.index(ch))

    if not ch_indices:
        ch_indices = list(range(min(3, len(channel_names))))
        channels_to_plot = [channel_names[i] for i in ch_indices]

    n_plots = len(ch_indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]

    target_mean = np.mean(target_epochs, axis=0)
    nontarget_mean = np.mean(nontarget_epochs, axis=0)

    target_sem = np.std(target_epochs, axis=0) / np.sqrt(len(target_epochs))
    nontarget_sem = (
        np.std(nontarget_epochs, axis=0) / np.sqrt(len(nontarget_epochs))
    )

    times_ms = times * 1000

    for ax, ch_idx, ch_name in zip(axes, ch_indices, channels_to_plot):
        ax.plot(
            times_ms, target_mean[ch_idx] * 1e6,
            "r-", linewidth=2, label=f"Target (n={len(target_epochs)})",
        )
        ax.fill_between(
            times_ms,
            (target_mean[ch_idx] - target_sem[ch_idx]) * 1e6,
            (target_mean[ch_idx] + target_sem[ch_idx]) * 1e6,
            color="red", alpha=0.2,
        )

        ax.plot(
            times_ms, nontarget_mean[ch_idx] * 1e6,
            "b-", linewidth=2,
            label=f"Nontarget (n={len(nontarget_epochs)})",
        )
        ax.fill_between(
            times_ms,
            (nontarget_mean[ch_idx] - nontarget_sem[ch_idx]) * 1e6,
            (nontarget_mean[ch_idx] + nontarget_sem[ch_idx]) * 1e6,
            color="blue", alpha=0.2,
        )

        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
        ax.axvline(x=300, color="green", linestyle=":", alpha=0.5,
                   label="300 ms")
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

        ax.set_title(f"Channel {ch_name}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (uV)")
        ax.legend(loc="upper right")
        ax.set_xlim(times_ms[0], times_ms[-1])

    fig.suptitle("Grand Average ERP: Target vs Nontarget", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_difference_wave(
    target_epochs: np.ndarray,
    nontarget_epochs: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
    channels_to_plot: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Разностная волна (target - nontarget) для выделения P300.

    Parameters
    ----------
    target_epochs, nontarget_epochs : np.ndarray
        Данные эпох
    times : np.ndarray
        Временная ось
    channel_names : list[str]
        Имена каналов
    channels_to_plot : list[str] | None
        Каналы для отображения
    save_path : str | Path | None
        Путь для сохранения

    """
    if channels_to_plot is None:
        channels_to_plot = ["Pz", "Cz", "Fz", "Oz"]

    ch_indices = [
        channel_names.index(ch)
        for ch in channels_to_plot
        if ch in channel_names
    ]
    if not ch_indices:
        ch_indices = list(range(min(4, len(channel_names))))
        channels_to_plot = [channel_names[i] for i in ch_indices]

    diff = np.mean(target_epochs, axis=0) - np.mean(nontarget_epochs, axis=0)
    times_ms = times * 1000

    fig, ax = plt.subplots(figsize=(12, 5))

    for ch_idx, ch_name in zip(ch_indices, channels_to_plot):
        ax.plot(times_ms, diff[ch_idx] * 1e6, linewidth=1.5, label=ch_name)

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=300, color="green", linestyle=":", alpha=0.5, label="300 ms")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    ax.set_title("Difference Wave (Target - Nontarget)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix_binary(
    cm: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Матрица ошибок для бинарной классификации."""
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    labels = ["Nontarget", "Target"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        ylabel="True", xlabel="Predicted",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_trial_scores(
    letter_scores: dict[str, float],
    true_target: str,
    trial_index: int,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart scores по буквам для одного trial.

    Parameters
    ----------
    letter_scores : dict[str, float]
        {буква: P(target)}
    true_target : str
        Истинная целевая буква
    trial_index : int
        Номер trial

    """
    sorted_items = sorted(
        letter_scores.items(), key=lambda x: x[1], reverse=True,
    )
    letters = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(max(12, len(letters) * 0.4), 5))

    colors = [
        "red" if l == true_target else "steelblue"
        for l in letters
    ]
    ax.bar(range(len(letters)), scores, color=colors)
    ax.set_xticks(range(len(letters)))
    ax.set_xticklabels(letters, fontsize=8)
    ax.set_ylabel("P(target)")
    ax.set_title(
        f"Trial {trial_index}: target='{true_target}', "
        f"predicted='{letters[0]}'"
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_training_history(
    history: dict,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Кривые обучения: loss, F1, AUC."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0, 0].plot(history["epoch"], history["val_loss"], label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("Epoch")

    # F1
    axes[0, 1].plot(history["epoch"], history["val_f1_target"], label="F1 Target")
    axes[0, 1].set_title("F1 Score (Target)")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Epoch")

    # Accuracy
    axes[1, 0].plot(history["epoch"], history["train_accuracy"], label="Train")
    axes[1, 0].plot(history["epoch"], history["val_accuracy"], label="Val (BAcc)")
    axes[1, 0].set_title("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("Epoch")

    # ROC AUC
    if "ROC_AUC" in history:
        axes[1, 1].plot(history["epoch"], history["ROC_AUC"], label="ROC AUC")
    axes[1, 1].set_title("ROC AUC")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("Epoch")

    fig.suptitle("Training History", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
