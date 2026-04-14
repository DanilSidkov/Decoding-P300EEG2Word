"""PyTorch Dataset для бинарной P300 классификации и trial-level предсказания."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.epoch_extractor import EpochInfo


class P300BinaryDataset(Dataset):
    """Dataset для бинарной классификации target/nontarget эпох.

    Parameters
    ----------
    epochs_data : np.ndarray
        Данные эпох, shape (n_epochs, n_channels, n_times)
    labels : np.ndarray
        Метки: 0 (nontarget) или 1 (target), shape (n_epochs,)
    augment : bool
        Применять аугментацию при обучении
    normalize : bool
        Применять нормализацию по каналам

    """

    def __init__(
        self,
        epochs_data: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        normalize: bool = True,
        norm_stats: list[tuple[float, float]] | None = None,
    ) -> None:
        self.data = epochs_data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.norm_stats = norm_stats

        if normalize:
            if self.norm_stats is None:
                self.norm_stats = self._compute_norm_stats()
            self._apply_normalize()

    def _compute_norm_stats(self) -> list[tuple[float, float]]:
        """Вычисляет статистики нормализации (median, scale) по каналам."""
        stats = []
        for ch in range(self.data.shape[1]):
            channel_data = self.data[:, ch, :]
            median = float(np.median(channel_data))
            mad = float(np.median(np.abs(channel_data - median)))
            scale = mad * 1.4826 if mad > 0 else float(np.std(channel_data))
            stats.append((median, scale))
        return stats

    def _apply_normalize(self) -> None:
        """Применяет нормализацию используя сохранённые статистики."""
        for ch, (median, scale) in enumerate(self.norm_stats):
            if scale > 0:
                self.data[:, ch, :] = (self.data[:, ch, :] - median) / scale

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx].copy()

        if self.augment:
            x = self._augment(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long,
        )

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Аугментация для ЭЭГ (без временного сдвига — он разрушает P300)."""
        # Гауссов шум
        if np.random.random() < 0.5:
            noise_scale = 0.05 * np.std(x)
            x = x + np.random.randn(*x.shape).astype(np.float32) * noise_scale

        # Масштабирование амплитуды
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

        return x


def prepare_binary_data(
    epochs_data: np.ndarray,
    epochs_info: list[EpochInfo],
    balance_ratio: float | None = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[EpochInfo]]:
    """Подготавливает данные для бинарной классификации.

    Создаёт метки и (опционально) балансирует классы
    через undersampling non-target.

    Parameters
    ----------
    epochs_data : np.ndarray
        Данные эпох, shape (n_epochs, n_channels, n_times)
    epochs_info : list[EpochInfo]
        Метаданные эпох
    balance_ratio : float | None
        Целевое соотношение nontarget:target. None для отключения.
    seed : int
        Seed для воспроизводимости

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[EpochInfo]]
        (данные, метки, обновлённый список эпох)

    """
    labels = np.array([1 if e.is_target else 0 for e in epochs_info])

    if balance_ratio is None:
        return epochs_data, labels, epochs_info

    rng = np.random.RandomState(seed)

    target_idx = np.where(labels == 1)[0]
    nontarget_idx = np.where(labels == 0)[0]

    n_target = len(target_idx)
    n_keep_nontarget = int(n_target * balance_ratio)
    n_keep_nontarget = min(n_keep_nontarget, len(nontarget_idx))

    selected_nontarget = rng.choice(
        nontarget_idx, size=n_keep_nontarget, replace=False,
    )

    keep_idx = np.sort(np.concatenate([target_idx, selected_nontarget]))

    balanced_data = epochs_data[keep_idx]
    balanced_labels = labels[keep_idx]
    balanced_info = [epochs_info[i] for i in keep_idx]

    print(
        f"Балансировка: {len(epochs_data)} -> {len(balanced_data)} эпох "
        f"(target={n_target}, nontarget={n_keep_nontarget}, "
        f"ratio={n_keep_nontarget / n_target:.1f}:1)"
    )

    return balanced_data, balanced_labels, balanced_info


def split_by_trials(
    epochs_data: np.ndarray,
    epochs_info: list[EpochInfo],
    train_ratio: float = 0.71,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[EpochInfo]]]:
    """Разбивает данные по trials (НЕ рандомно по эпохам).

    Trials перемешиваются для устранения temporal bias.

    Parameters
    ----------
    epochs_data : np.ndarray
        Данные эпох
    epochs_info : list[EpochInfo]
        Метаданные эпох
    train_ratio : float
        Доля trials для обучения
    val_ratio : float
        Доля trials для валидации
    seed : int
        Seed для воспроизводимости

    Returns
    -------
    dict[str, tuple]
        {'train': (data, labels, info), 'val': ..., 'test': ...}

    """
    labels = np.array([1 if e.is_target else 0 for e in epochs_info])
    trial_indices = np.array([e.trial_index for e in epochs_info])
    unique_trials = sorted(set(trial_indices))
    n_trials = len(unique_trials)

    rng = np.random.RandomState(seed)
    shuffled = np.array(unique_trials)
    rng.shuffle(shuffled)

    n_train = int(n_trials * train_ratio)
    n_val = int(n_trials * val_ratio)

    train_trials = set(shuffled[:n_train])
    val_trials = set(shuffled[n_train:n_train + n_val])
    test_trials = set(shuffled[n_train + n_val:])

    result = {}
    for split_name, split_trials in [
        ("train", train_trials),
        ("val", val_trials),
        ("test", test_trials),
    ]:
        mask = np.array([e.trial_index in split_trials for e in epochs_info])
        split_data = epochs_data[mask]
        split_labels = labels[mask]
        split_info = [e for e, m in zip(epochs_info, mask) if m]

        n_tar = int(split_labels.sum())
        n_ntar = len(split_labels) - n_tar
        print(
            f"{split_name}: {len(split_data)} эпох "
            f"({len(split_trials)} trials, "
            f"target={n_tar}, nontarget={n_ntar})"
        )
        result[split_name] = (split_data, split_labels, split_info)

    return result


class P300TrialDataset:
    """Группирует эпохи по trials для letter-level предсказания.

    Parameters
    ----------
    epochs_data : np.ndarray
        Данные эпох, shape (n_epochs, n_channels, n_times)
    epochs_info : list[EpochInfo]
        Метаданные эпох

    """

    def __init__(
        self,
        epochs_data: np.ndarray,
        epochs_info: list[EpochInfo],
    ) -> None:
        self.data = epochs_data
        self.info = epochs_info
        self._group_by_trial()

    def _group_by_trial(self) -> None:
        """Группирует эпохи по trial_index."""
        self.trials: dict[int, dict] = {}

        for i, ep_info in enumerate(self.info):
            tid = ep_info.trial_index
            if tid not in self.trials:
                self.trials[tid] = {
                    "target_letter": ep_info.target_letter,
                    "indices": [],
                    "letters": [],
                    "is_target": [],
                }
            self.trials[tid]["indices"].append(i)
            self.trials[tid]["letters"].append(ep_info.letter)
            self.trials[tid]["is_target"].append(ep_info.is_target)

    def get_trial(self, trial_index: int) -> dict:
        """Возвращает все эпохи для данного trial.

        Returns
        -------
        dict
            target_letter, data (n, ch, t), letters, is_target

        """
        trial = self.trials[trial_index]
        indices = trial["indices"]
        return {
            "target_letter": trial["target_letter"],
            "data": self.data[indices],
            "letters": trial["letters"],
            "is_target": trial["is_target"],
        }

    def get_letter_epochs(
        self, trial_index: int, letter: str,
    ) -> np.ndarray | None:
        """Возвращает эпохи конкретной буквы в trial.

        Returns
        -------
        np.ndarray | None
            shape (n_epochs_letter, n_channels, n_times) или None

        """
        trial = self.trials[trial_index]
        indices = [
            trial["indices"][i]
            for i, l in enumerate(trial["letters"])
            if l == letter
        ]
        if not indices:
            return None
        return self.data[indices]

    @property
    def trial_indices(self) -> list[int]:
        return sorted(self.trials.keys())
