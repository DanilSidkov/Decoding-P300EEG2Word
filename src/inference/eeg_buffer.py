"""Кольцевой буфер ЭЭГ с LSL-метками времени.

Хранит последние `capacity_sec` секунд сигнала вместе с timestamp'ами LSL,
позволяет вырезать окно [t0, t1] по временным меткам LSL.
"""

from __future__ import annotations

import threading

import numpy as np


class EEGRingBuffer:
    """Потокобезопасный кольцевой буфер samples + timestamps.

    Parameters
    ----------
    n_channels : int
        Число каналов ЭЭГ.
    sfreq : float
        Частота дискретизации исходного потока.
    capacity_sec : float
        Сколько секунд сигнала хранить.

    """

    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        capacity_sec: float = 10.0,
    ) -> None:
        self.n_channels = n_channels
        self.sfreq = float(sfreq)
        self.capacity = int(capacity_sec * sfreq)
        self._samples = np.zeros((self.capacity, n_channels), dtype=np.float32)
        self._timestamps = np.zeros(self.capacity, dtype=np.float64)
        self._write_idx = 0
        self._n_written = 0
        self._lock = threading.Lock()

    def push_chunk(
        self,
        samples: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        """Добавляет чанк сэмплов в буфер.

        Parameters
        ----------
        samples : np.ndarray, shape (n_samples, n_channels)
        timestamps : np.ndarray, shape (n_samples,) — LSL timestamps (sec)

        """
        samples = np.asarray(samples, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != self.n_channels:
            raise ValueError(
                f"Ожидалось (N, {self.n_channels}), получено {samples.shape}"
            )
        n = samples.shape[0]
        if n == 0:
            return

        with self._lock:
            end = self._write_idx + n
            if end <= self.capacity:
                self._samples[self._write_idx:end] = samples
                self._timestamps[self._write_idx:end] = timestamps
            else:
                first = self.capacity - self._write_idx
                self._samples[self._write_idx:] = samples[:first]
                self._timestamps[self._write_idx:] = timestamps[:first]
                rest = n - first
                self._samples[:rest] = samples[first:]
                self._timestamps[:rest] = timestamps[first:]
            self._write_idx = end % self.capacity
            self._n_written += n

    @property
    def latest_timestamp(self) -> float:
        """LSL timestamp последнего записанного сэмпла (или 0 если пусто)."""
        with self._lock:
            if self._n_written == 0:
                return 0.0
            last_idx = (self._write_idx - 1) % self.capacity
            return float(self._timestamps[last_idx])

    def get_window(
        self,
        t_start: float,
        t_end: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Возвращает (samples, timestamps) в окне [t_start, t_end].

        Returns None если окно ещё не полностью записано или уже
        вышло за пределы буфера.

        samples shape: (n_samples, n_channels).
        """
        with self._lock:
            if self._n_written == 0:
                return None
            n_valid = min(self._n_written, self.capacity)
            if n_valid < self.capacity:
                # буфер ещё не переполнился — валидная часть [0, _write_idx)
                samples = self._samples[:self._write_idx].copy()
                timestamps = self._timestamps[:self._write_idx].copy()
            else:
                samples = np.concatenate([
                    self._samples[self._write_idx:],
                    self._samples[:self._write_idx],
                ], axis=0)
                timestamps = np.concatenate([
                    self._timestamps[self._write_idx:],
                    self._timestamps[:self._write_idx],
                ], axis=0)

        if t_start < timestamps[0] or t_end > timestamps[-1]:
            return None

        # бинарный поиск
        i0 = int(np.searchsorted(timestamps, t_start, side="left"))
        i1 = int(np.searchsorted(timestamps, t_end, side="right"))
        if i1 <= i0:
            return None
        return samples[i0:i1].copy(), timestamps[i0:i1].copy()
