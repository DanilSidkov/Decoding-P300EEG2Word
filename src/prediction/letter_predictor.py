"""Предсказание букв на основе P300 детекции.

Алгоритм: для каждого trial группируем эпохи по буквам,
вычисляем средний P(target) для каждой буквы,
буква с максимальным P(target) = предсказание.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.epoch_extractor import EpochInfo
from src.features.p300_dataset import P300TrialDataset


class LetterPredictor:
    """Предсказание букв по P(target) модели.

    Parameters
    ----------
    model : nn.Module
        Обученная модель бинарной классификации
    device : str
        Устройство для inference

    """

    def __init__(
        self,
        model: nn.Module,
        device: str | None = None,
        norm_stats: list[tuple[float, float]] | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.norm_stats = norm_stats

    def predict_trial(
        self,
        trial_data: np.ndarray,
        trial_letters: list[str],
        normalize: bool = True,
    ) -> dict[str, float]:
        """Предсказывает score для каждой буквы в trial.

        Parameters
        ----------
        trial_data : np.ndarray
            shape (n_epochs, n_channels, n_times)
        trial_letters : list[str]
            Какая буква соответствует каждой эпохе
        normalize : bool
            Нормализовать данные

        Returns
        -------
        dict[str, float]
            {буква: средний P(target)}

        """
        data = trial_data.astype(np.float32)
        if normalize:
            data = self._normalize(data)

        # Прогоняем через модель
        probs = self._get_probabilities(data)

        # Группируем по буквам и считаем средний P(target)
        letter_scores = defaultdict(list)
        for prob, letter in zip(probs, trial_letters):
            letter_scores[letter].append(prob)

        return {
            letter: float(np.mean(scores))
            for letter, scores in letter_scores.items()
        }

    def predict_letter(
        self,
        trial_data: np.ndarray,
        trial_letters: list[str],
        normalize: bool = True,
    ) -> tuple[str, dict[str, float]]:
        """Предсказывает целевую букву для trial.

        Returns
        -------
        tuple[str, dict[str, float]]
            (предсказанная буква, все scores)

        """
        scores = self.predict_trial(trial_data, trial_letters, normalize)
        predicted = max(scores, key=scores.get)
        return predicted, scores

    def evaluate_trials(
        self,
        trial_dataset: P300TrialDataset,
        normalize: bool = True,
    ) -> dict:
        """Оценивает предсказание букв на всех trials.

        Parameters
        ----------
        trial_dataset : P300TrialDataset
            Группированные по trials эпохи
        normalize : bool
            Нормализовать данные

        Returns
        -------
        dict
            top1_accuracy, top3_accuracy, predictions, itr

        """
        predictions = []
        correct_top1 = 0
        correct_top3 = 0

        for tid in trial_dataset.trial_indices:
            trial = trial_dataset.get_trial(tid)
            predicted, scores = self.predict_letter(
                trial["data"], trial["letters"], normalize,
            )

            true_letter = trial["target_letter"]

            # Top-3: сортируем по score, берём 3 лучших
            sorted_letters = sorted(
                scores, key=scores.get, reverse=True,
            )
            top3 = sorted_letters[:3]

            is_top1 = predicted == true_letter
            is_top3 = true_letter in top3
            correct_top1 += int(is_top1)
            correct_top3 += int(is_top3)

            predictions.append({
                "trial_index": tid,
                "true_letter": true_letter,
                "predicted_letter": predicted,
                "is_correct": is_top1,
                "is_top3": is_top3,
                "scores": scores,
                "top3": top3,
            })

        n = len(predictions)
        top1_acc = correct_top1 / n if n > 0 else 0.0
        top3_acc = correct_top3 / n if n > 0 else 0.0

        return {
            "top1_accuracy": top1_acc,
            "top3_accuracy": top3_acc,
            "n_trials": n,
            "predictions": predictions,
        }

    def _get_probabilities(self, data: np.ndarray) -> np.ndarray:
        """Возвращает P(target) для каждой эпохи."""
        tensor = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Robust нормализация по каналам (используя train статистики)."""
        data = data.copy()
        if self.norm_stats is not None:
            for ch, (median, scale) in enumerate(self.norm_stats):
                if scale > 0:
                    data[:, ch, :] = (data[:, ch, :] - median) / scale
        else:
            for ch in range(data.shape[1]):
                channel = data[:, ch, :]
                median = np.median(channel)
                mad = np.median(np.abs(channel - median))
                scale = mad * 1.4826 if mad > 0 else np.std(channel)
                if scale > 0:
                    data[:, ch, :] = (channel - median) / scale
        return data


def compute_itr(
    n_symbols: int,
    accuracy: float,
    trial_duration_sec: float,
) -> float:
    """Вычисляет Information Transfer Rate (bits/min).

    Стандартная метрика BCI по Wolpaw et al. (2000).

    Parameters
    ----------
    n_symbols : int
        Число возможных символов
    accuracy : float
        Точность предсказания (0-1)
    trial_duration_sec : float
        Средняя длительность одного trial (секунды)

    Returns
    -------
    float
        ITR в bits/min

    """
    if accuracy <= 0 or accuracy >= 1:
        accuracy = np.clip(accuracy, 0.001, 0.999)

    n = n_symbols
    p = accuracy

    bits_per_trial = (
        np.log2(n)
        + p * np.log2(p)
        + (1 - p) * np.log2((1 - p) / (n - 1))
    )

    trials_per_min = 60.0 / trial_duration_sec
    return float(bits_per_trial * trials_per_min)
