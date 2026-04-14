"""Главный скрипт оркестрации: загрузка -> предобработка -> обучение -> оценка.

Запуск:
    python -m src.models.run_training --data-dir data/raw --settings BCI_P300/settings.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    ALPHABET,
    ARTIFACT_REJECT_UV,
    DEFAULT_BALANCE_RATIO,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TARGET_WEIGHT,
    EPOCH_TMAX,
    EPOCH_TMIN,
    N_SYMBOLS,
    RESAMPLE_HZ,
)
from src.data.edf_loader import load_session
from src.data.epoch_extractor import (
    create_mne_epochs,
    extract_epoch_events,
    get_epoch_metadata,
)
from src.data.preprocessing import preprocess_raw
from src.data.synchronization import build_sync_mapping, validate_sync_quality
from src.evaluation.metrics import (
    binary_epoch_metrics,
    letter_prediction_metrics,
    print_metrics_summary,
)
from src.features.p300_dataset import (
    P300BinaryDataset,
    P300TrialDataset,
    prepare_binary_data,
    split_by_trials,
)
from src.models.p300net import P300EEGNet
from src.models.train_model import train_model, test_model, predict_probabilities
from src.prediction.letter_predictor import LetterPredictor, compute_itr
from src.visualization.erp_plots import (
    plot_confusion_matrix_binary,
    plot_difference_wave,
    plot_erp_comparison,
    plot_training_history,
    plot_trial_scores,
)


def run_pipeline(
    edf_path: str,
    log_path: str,
    settings_path: str | None = None,
    channel_set: str = "core",
    resample_hz: int = RESAMPLE_HZ,
    balance_ratio: float | None = DEFAULT_BALANCE_RATIO,
    target_weight: float = DEFAULT_TARGET_WEIGHT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_epochs: int = 200,
    patience: int = 15,
    output_dir: str = "reports/figures",
    model_save_path: str = "models/p300_model.pth",
) -> dict:
    """Полный пайплайн: от сырых данных до предсказания букв.

    Parameters
    ----------
    edf_path : str
        Путь к .edf файлу
    log_path : str
        Путь к .txt файлу лога
    settings_path : str | None
        Путь к settings.json
    channel_set : str
        Набор каналов: 'core', 'extended', 'all'
    resample_hz : int
        Частота ресемплинга
    balance_ratio : float
        Соотношение nontarget:target для балансировки
    target_weight : float
        Вес target класса в loss
    batch_size : int
        Размер батча
    num_epochs : int
        Максимальное число эпох обучения
    patience : int
        Early stopping patience
    output_dir : str
        Папка для сохранения графиков
    model_save_path : str
        Путь для сохранения модели

    Returns
    -------
    dict
        Результаты: метрики, history, predictions

    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ===== ФАЗА 1: Загрузка данных =====
    print("\n" + "=" * 60)
    print("  ФАЗА 1: Загрузка данных")
    print("=" * 60)

    raw, log_events, settings = load_session(
        edf_path, log_path, settings_path,
    )

    # ===== ФАЗА 2: Синхронизация =====
    print("\n" + "=" * 60)
    print("  ФАЗА 2: Синхронизация EDF <-> Лог")
    print("=" * 60)

    trial_sync_pairs = build_sync_mapping(raw, log_events)
    sync_stats = validate_sync_quality(trial_sync_pairs)
    print(f"Sync пар: {sync_stats['n_pairs']}, trials: {sync_stats['n_trials']}")
    if "drift_mean_ms" in sync_stats:
        print(f"Drift: mean={sync_stats['drift_mean_ms']:.2f} ms, "
              f"max={sync_stats['drift_max_ms']:.2f} ms")

    # ===== ФАЗА 3: Предобработка =====
    print("\n" + "=" * 60)
    print("  ФАЗА 3: Предобработка ЭЭГ")
    print("=" * 60)

    from src.data.preprocessing import get_channel_set
    channels = get_channel_set(channel_set)
    raw = preprocess_raw(raw, channel_set=channels, resample_hz=resample_hz)

    # ===== ФАЗА 4: Извлечение эпох =====
    print("\n" + "=" * 60)
    print("  ФАЗА 4: Извлечение эпох")
    print("=" * 60)

    epochs_info = extract_epoch_events(log_events, trial_sync_pairs)
    metadata = get_epoch_metadata(epochs_info)
    print(f"Всего эпох: {metadata['n_total']} "
          f"(target={metadata['n_target']}, "
          f"nontarget={metadata['n_nontarget']}, "
          f"ratio={metadata['ratio']:.1f}:1)")
    print(f"Trials: {metadata['n_trials']}")

    mne_epochs, epochs_info = create_mne_epochs(
        raw, epochs_info,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        reject_uv=ARTIFACT_REJECT_UV,
    )

    epochs_data = mne_epochs.get_data()  # (n, ch, t)
    channel_names = mne_epochs.ch_names
    times = mne_epochs.times

    print(f"Эпохи после отбраковки: {epochs_data.shape}")

    # ===== Визуализация ERP =====
    print("\nСтроим ERP графики...")

    target_mask = np.array([e.is_target for e in epochs_info])
    target_data = epochs_data[target_mask]
    nontarget_data = epochs_data[~target_mask]

    plot_erp_comparison(
        target_data, nontarget_data, times, channel_names,
        save_path=output_path / "erp_comparison.png",
    )
    plot_difference_wave(
        target_data, nontarget_data, times, channel_names,
        save_path=output_path / "difference_wave.png",
    )
    print("ERP графики сохранены.")

    # ===== ФАЗА 5: Подготовка данных =====
    print("\n" + "=" * 60)
    print("  ФАЗА 5: Подготовка данных для обучения")
    print("=" * 60)

    splits = split_by_trials(epochs_data, epochs_info)

    # Балансировка train (если balance_ratio задан)
    train_data_raw, _, train_info_raw = splits["train"]
    train_data, train_labels, train_info = prepare_binary_data(
        train_data_raw, train_info_raw, balance_ratio=balance_ratio,
    )

    val_data, _, val_info = splits["val"]
    val_labels = np.array([1 if e.is_target else 0 for e in val_info])

    test_data, _, test_info = splits["test"]
    test_labels = np.array([1 if e.is_target else 0 for e in test_info])

    # Авто-расчёт веса target класса из соотношения классов в train
    n_target_train = int(train_labels.sum())
    n_nontarget_train = len(train_labels) - n_target_train
    if n_target_train > 0:
        target_weight = n_nontarget_train / n_target_train
        print(f"Auto target_weight: {target_weight:.1f} "
              f"({n_nontarget_train} nontarget / {n_target_train} target)")

    # PyTorch Datasets — нормализация по статистикам train
    train_dataset = P300BinaryDataset(
        train_data, train_labels, augment=True, normalize=True,
    )
    train_norm_stats = train_dataset.norm_stats

    val_dataset = P300BinaryDataset(
        val_data, val_labels, augment=False, normalize=True,
        norm_stats=train_norm_stats,
    )
    test_dataset = P300BinaryDataset(
        test_data, test_labels, augment=False, normalize=True,
        norm_stats=train_norm_stats,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # ===== ФАЗА 6: Обучение =====
    print("\n" + "=" * 60)
    print("  ФАЗА 6: Обучение модели")
    print("=" * 60)

    n_channels = epochs_data.shape[1]
    n_times = epochs_data.shape[2]

    model = P300EEGNet(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=2,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Модель: P300EEGNet ({n_channels} каналов, {n_times} точек)")
    print(f"Параметров: {n_params:,}")

    history = train_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs,
        target_class_weight=target_weight,
    )

    plot_training_history(
        history, save_path=output_path / "training_history.png",
    )

    # Сохранение модели
    model_path = Path(model_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_channels": n_channels,
        "n_times": n_times,
        "channel_names": channel_names,
        "norm_stats": train_norm_stats,
        "history": history,
    }, str(model_path))
    print(f"Модель сохранена: {model_path}")

    # ===== ФАЗА 7: Тестирование =====
    print("\n" + "=" * 60)
    print("  ФАЗА 7: Тестирование")
    print("=" * 60)

    predictions, labels, accuracy = test_model(model, test_loader)
    probs, _ = predict_probabilities(model, test_loader)

    binary_metrics = binary_epoch_metrics(
        np.array(labels),
        np.array(predictions),
        np.array(probs),
    )
    print_metrics_summary(binary_metrics, "Binary Classification (Test)")

    if "confusion_matrix" in binary_metrics:
        plot_confusion_matrix_binary(
            binary_metrics["confusion_matrix"],
            save_path=output_path / "confusion_matrix.png",
        )

    # ===== ФАЗА 8: Предсказание букв =====
    print("\n" + "=" * 60)
    print("  ФАЗА 8: Предсказание букв")
    print("=" * 60)

    predictor = LetterPredictor(model, norm_stats=train_norm_stats)
    test_trial_dataset = P300TrialDataset(test_data, test_info)
    trial_results = predictor.evaluate_trials(test_trial_dataset)

    true_letters = [
        p["true_letter"] for p in trial_results["predictions"]
    ]
    pred_letters = [
        p["predicted_letter"] for p in trial_results["predictions"]
    ]
    all_scores = [
        p["scores"] for p in trial_results["predictions"]
    ]

    letter_metrics = letter_prediction_metrics(
        true_letters, pred_letters, all_scores,
    )
    print_metrics_summary(letter_metrics, "Letter Prediction (Test)")

    # ITR
    avg_trial_duration = 30.0  # примерная оценка (сек)
    itr = compute_itr(N_SYMBOLS, letter_metrics["top1_accuracy"],
                      avg_trial_duration)
    print(f"ITR: {itr:.2f} bits/min")

    # Визуализация по trials
    for pred in trial_results["predictions"][:5]:
        plot_trial_scores(
            pred["scores"],
            pred["true_letter"],
            pred["trial_index"],
            save_path=output_path / f"trial_{pred['trial_index']}_scores.png",
        )

    return {
        "binary_metrics": binary_metrics,
        "letter_metrics": letter_metrics,
        "itr": itr,
        "history": history,
        "trial_results": trial_results,
    }


def main() -> None:
    """Entry point для CLI."""
    parser = argparse.ArgumentParser(
        description="P300 BCI Training Pipeline",
    )
    parser.add_argument(
        "--edf", type=str, default=r"data\raw\NeoRec_2026-03-30_11-01-10.edf",
        help="Путь к .edf файлу",
    )
    parser.add_argument(
        "--log", type=str, default=r"data\raw\logs_2026-03-30_11-00-12.txt",
        help="Путь к .txt файлу лога",
    )
    parser.add_argument(
        "--settings", type=str, default="BCI_P300/settings.json",
        help="Путь к settings.json",
    )
    parser.add_argument(
        "--channels", type=str, default="extended",
        choices=["core", "extended", "all", "clear"],
        help="Набор каналов",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Максимальное число эпох обучения",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Размер батча",
    )
    parser.add_argument(
        "--balance-ratio", type=float, default=DEFAULT_BALANCE_RATIO,
        help="Соотношение nontarget:target",
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports/figures",
        help="Папка для графиков",
    )
    parser.add_argument(
        "--model-path", type=str, default="models/p300_model.pth",
        help="Путь для сохранения модели",
    )

    args = parser.parse_args()

    run_pipeline(
        edf_path=args.edf,
        log_path=args.log,
        settings_path=args.settings,
        channel_set=args.channels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        balance_ratio=args.balance_ratio,
        output_dir=args.output_dir,
        model_save_path=args.model_path,
    )


if __name__ == "__main__":
    main()
