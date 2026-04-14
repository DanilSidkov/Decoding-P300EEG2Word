"""Онлайн-препроцессинг одной эпохи.

Пайплайн совпадает с offline `preprocess_raw` + `create_mne_epochs`:
выбор каналов -> bandpass -> avg re-reference -> resample -> baseline -> normalize.

Для корректной работы фильтров берём окно с запасом `filter_pad` с обоих
концов, затем обрезаем до [tmin, tmax].
"""

from __future__ import annotations

import mne
import numpy as np

from src.config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    BASELINE,
    EPOCH_TMAX,
    EPOCH_TMIN,
    NOTCH_FREQ,
    RESAMPLE_HZ,
)


class OnlineEpochPreprocessor:
    """Готовит одну эпоху для подачи в модель.

    Parameters
    ----------
    raw_channel_names : list[str]
        Имена каналов в потоке LSL (как идут в EEG chunk).
    target_channel_names : list[str]
        Имена каналов, на которых обучена модель
        (из checkpoint['channel_names']).
    sfreq : float
        Частота дискретизации входного потока.
    norm_stats : list[tuple[float, float]] | None
        Статистики нормализации (median, scale) на канал из обучения.
    tmin, tmax : float
        Границы эпохи относительно маркера (сек).
    resample_hz : float
        Целевая частота после ресемплинга (должна совпадать с обучением).
    bandpass : tuple[float, float]
        (low, high) в Гц.
    notch : float | None
        Частота notch (None — без notch).
    filter_pad_sec : float
        Сколько секунд запаса с каждого края для устойчивой фильтрации.
    baseline : tuple | None
        baseline correction, (None, 0) → от начала до 0 сек.

    """

    def __init__(
        self,
        raw_channel_names: list[str],
        target_channel_names: list[str],
        sfreq: float,
        norm_stats: list[tuple[float, float]] | None = None,
        tmin: float = EPOCH_TMIN,
        tmax: float = EPOCH_TMAX,
        resample_hz: float = RESAMPLE_HZ,
        bandpass: tuple[float, float] = (BANDPASS_LOW, BANDPASS_HIGH),
        notch: float | None = NOTCH_FREQ,
        filter_pad_sec: float = 1.0,
        baseline: tuple | None = BASELINE,
        eeg_unit_scale: float = 1e-6,
    ) -> None:
        self.raw_channel_names = list(raw_channel_names)
        self.target_channel_names = list(target_channel_names)
        self.sfreq = float(sfreq)
        self.norm_stats = norm_stats
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.resample_hz = float(resample_hz)
        self.bandpass = bandpass
        self.notch = notch
        self.filter_pad_sec = float(filter_pad_sec)
        self.baseline = baseline
        self.eeg_unit_scale = eeg_unit_scale

        # проверка: все нужные каналы есть в потоке
        raw_lower = {n.lower(): i for i, n in enumerate(self.raw_channel_names)}
        self._pick_idx: list[int] = []
        self._pick_names: list[str] = []
        missing = []
        for name in self.target_channel_names:
            i = raw_lower.get(name.lower())
            if i is None:
                missing.append(name)
            else:
                self._pick_idx.append(i)
                self._pick_names.append(name)
        if missing:
            print(
                f"[OnlineEpochPreprocessor] Внимание: каналы не найдены в "
                f"LSL-потоке: {missing}"
            )
        if not self._pick_idx:
            raise RuntimeError(
                "Ни один целевой канал не найден в EEG-потоке. "
                f"Ожидались {self.target_channel_names}, пришли "
                f"{self.raw_channel_names}"
            )

        # MNE Info для выбранных каналов
        self._info = mne.create_info(
            ch_names=self._pick_names,
            sfreq=self.sfreq,
            ch_types="eeg",
        )

    @property
    def n_output_times(self) -> int:
        """Число точек в итоговой эпохе после ресемплинга."""
        return int(round((self.tmax - self.tmin) * self.resample_hz)) + 1

    @property
    def n_channels(self) -> int:
        return len(self._pick_idx)

    def window_limits(self, t_marker: float) -> tuple[float, float]:
        """Возвращает (t_start, t_end) окна-с-запасом для выборки из буфера."""
        t_start = t_marker + self.tmin - self.filter_pad_sec
        t_end = t_marker + self.tmax + self.filter_pad_sec
        return t_start, t_end

    def process(
        self,
        samples: np.ndarray,
        timestamps: np.ndarray,
        t_marker: float,
    ) -> np.ndarray | None:
        """Обрабатывает одну эпоху.

        Parameters
        ----------
        samples : np.ndarray, shape (n_samples, n_all_channels)
            Окно-с-запасом из EEG буфера.
        timestamps : np.ndarray, shape (n_samples,)
            LSL timestamps этих сэмплов.
        t_marker : float
            LSL timestamp события (начало движения буквы).

        Returns
        -------
        np.ndarray | None
            Эпоха shape (n_channels, n_output_times), нормализованная,
            или None при ошибке.

        """
        if samples.shape[0] < 2:
            return None

        # 1. выбор каналов + перевод в V (mne ждёт вольты)
        picked = samples[:, self._pick_idx].T.astype(np.float64)
        picked = picked * self.eeg_unit_scale  # µV -> V

        # 2. строим RawArray
        try:
            raw = mne.io.RawArray(
                picked, self._info.copy(), verbose=False, first_samp=0,
            )
        except Exception as e:
            print(f"[OnlinePrep] RawArray error: {e}")
            return None

        # 3. фильтры
        if self.notch is not None:
            try:
                raw.notch_filter(self.notch, verbose=False)
            except Exception:
                pass
        try:
            raw.filter(
                l_freq=self.bandpass[0],
                h_freq=self.bandpass[1],
                method="fir",
                verbose=False,
            )
        except Exception as e:
            print(f"[OnlinePrep] filter error: {e}")
            return None

        # 4. avg re-reference
        try:
            raw.set_eeg_reference("average", verbose=False)
        except Exception:
            pass

        # 5. ресемплинг
        if abs(raw.info["sfreq"] - self.resample_hz) > 1e-3:
            try:
                raw.resample(self.resample_hz, verbose=False)
            except Exception as e:
                print(f"[OnlinePrep] resample error: {e}")
                return None

        data = raw.get_data()  # (ch, n) в вольтах
        # обратно в µV, т.к. модель училась на µV после mne.get_data()?
        # Нет — в обучении epochs.get_data() тоже в вольтах. Оставляем V.

        # 6. вычисляем индексы под tmin/tmax относительно маркера
        # после ресемплинга новая sfreq, timestamps старые.
        # берём линейный пересчёт: t_start_buf = timestamps[0]
        t_start_buf = float(timestamps[0])
        new_sfreq = raw.info["sfreq"]
        # offset относительно t_start_buf в сек
        offset_marker = t_marker - t_start_buf
        i_mark = int(round(offset_marker * new_sfreq))
        i0 = i_mark + int(round(self.tmin * new_sfreq))
        i1 = i0 + self.n_output_times  # inclusive length

        if i0 < 0 or i1 > data.shape[1]:
            print(
                f"[OnlinePrep] окно выходит за границы: "
                f"i0={i0}, i1={i1}, data.shape[1]={data.shape[1]}"
            )
            return None

        epoch = data[:, i0:i1].copy()  # (ch, n_out)

        # 7. baseline correction
        if self.baseline is not None:
            b_start_sec, b_end_sec = self.baseline
            if b_start_sec is None:
                bi0 = 0
            else:
                bi0 = max(
                    0,
                    int(round((b_start_sec - self.tmin) * new_sfreq)),
                )
            if b_end_sec is None:
                bi1 = epoch.shape[1]
            else:
                bi1 = min(
                    epoch.shape[1],
                    int(round((b_end_sec - self.tmin) * new_sfreq)),
                )
            if bi1 > bi0:
                baseline_mean = epoch[:, bi0:bi1].mean(axis=1, keepdims=True)
                epoch = epoch - baseline_mean

        # 8. нормализация
        if self.norm_stats is not None:
            for ch, (median, scale) in enumerate(self.norm_stats):
                if ch >= epoch.shape[0]:
                    break
                if scale > 0:
                    epoch[ch] = (epoch[ch] - median) / scale

        return epoch.astype(np.float32)
