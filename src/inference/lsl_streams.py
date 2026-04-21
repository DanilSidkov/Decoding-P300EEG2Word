"""LSL-обёртки: EEG-инлет (фоновый поток) и маркеры (non-blocking polling)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from pylsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    local_clock,
    resolve_byprop,
    resolve_streams,
)

from src.inference.eeg_buffer import EEGRingBuffer


@dataclass
class EEGStreamMeta:
    """Метаданные EEG-потока."""

    name: str
    n_channels: int
    sfreq: float
    channel_names: list[str]


def _extract_channel_names(info) -> list[str]:
    """Достаёт имена каналов из LSL StreamInfo.desc()."""
    names: list[str] = []
    try:
        desc = info.desc()
        chs = desc.child("channels").child("channel")
        for _ in range(info.channel_count()):
            label = chs.child_value("label")
            if not label:
                label = chs.child_value("name")
            names.append(label or "")
            chs = chs.next_sibling()
    except Exception:
        pass
    if len(names) != info.channel_count() or any(not n for n in names):
        names = [f"ch{i}" for i in range(info.channel_count())]
    return names


def resolve_eeg_stream(
    timeout: float = 10.0,
    preferred_names: tuple[str, ...] = ("NeoRec", "NVX136", "EEG"),
    min_channels: int = 2,
) -> tuple[StreamInlet, EEGStreamMeta]:
    """Находит EEG-поток через LSL.

    Сначала пытается найти по имени, затем любой с type='EEG'.
    """
    def _is_eeg_like(s) -> bool:
        return s.channel_count() >= min_channels and s.nominal_srate() > 0.0

    # сначала точное совпадение по имени
    for name in preferred_names:
        streams = resolve_byprop("name", name, timeout=1.0)
        streams = [s for s in streams if _is_eeg_like(s)]
        if streams:
            info = streams[0]
            inlet = StreamInlet(info, max_buflen=60)
            meta = EEGStreamMeta(
                name=info.name(),
                n_channels=info.channel_count(),
                sfreq=info.nominal_srate(),
                channel_names=_extract_channel_names(info),
            )
            return inlet, meta

    # по типу EEG
    streams = resolve_byprop("type", "EEG", timeout=timeout)
    streams = [s for s in streams if _is_eeg_like(s)]

    # фолбэк: все доступные потоки, ищем по подстроке в name/type
    all_streams = resolve_streams(wait_time=2.0)
    if not streams:
        candidates = []
        for s in all_streams:
            if not _is_eeg_like(s):
                continue
            n = (s.name() or "").lower()
            t = (s.type() or "").lower()
            if (
                t == "eeg"
                or "eeg" in t
                or "neorec" in n
                or "nvx" in n
                or any(p.lower() in n for p in preferred_names)
            ):
                candidates.append(s)
        streams = candidates

    if not streams:
        visible = ", ".join(
            f"{s.name()!r}(type={s.type()!r}, ch={s.channel_count()})"
            for s in all_streams
        ) or "<нет потоков>"
        raise RuntimeError(
            "LSL EEG-поток не найден. Запущен ли NeoRec с LSL broadcast?\n"
            f"Видимые LSL-потоки: {visible}"
        )
    info = streams[0]
    inlet = StreamInlet(info, max_buflen=60)
    meta = EEGStreamMeta(
        name=info.name(),
        n_channels=info.channel_count(),
        sfreq=info.nominal_srate(),
        channel_names=_extract_channel_names(info),
    )
    return inlet, meta


def resolve_marker_stream(
    name: str = "annotations",
    timeout: float = 10.0,
) -> StreamInlet:
    """Находит marker stream по имени."""
    streams = resolve_byprop("name", name, timeout=timeout)
    if not streams:
        raise RuntimeError(
            f"LSL marker stream '{name}' не найден. Запущен ли стимулятор?"
        )
    return StreamInlet(streams[0], max_buflen=360)


def make_feedback_outlet(
    name: str = "p300_feedback",
    source_id: str = "p300_fb",
) -> StreamOutlet:
    """Outlet для обратных маркеров (predicted letter -> стимулятор)."""
    info = StreamInfo(
        name=name,
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id=source_id,
    )
    return StreamOutlet(info)


class EEGReader(threading.Thread):
    """Фоновый поток: читает сэмплы из EEG-инлета в ring buffer.

    Если EEG-поток идёт с другого компьютера, его timestamps могут быть
    в чужих «часах». inlet.time_correction() возвращает смещение, которое
    нужно прибавить, чтобы перевести удалённое время в локальное LSL-время.
    Коррекция пересчитывается раз в `tc_interval_sec` секунд.
    """

    def __init__(
        self,
        inlet: StreamInlet,
        buffer: EEGRingBuffer,
        chunk_timeout: float = 0.2,
        tc_interval_sec: float = 5.0,
    ) -> None:
        super().__init__(daemon=True)
        self.inlet = inlet
        self.buffer = buffer
        self.chunk_timeout = chunk_timeout
        self.tc_interval_sec = tc_interval_sec
        self._stop_event = threading.Event()
        self.n_samples_total = 0
        self._tc_offset: float = 0.0       # time_correction кэш
        self._tc_last_update: float = -1.0

    def stop(self) -> None:
        self._stop_event.set()

    def _refresh_tc(self) -> None:
        """Обновляет кэш time_correction (не чаще раз в tc_interval_sec)."""
        now = time.monotonic()
        if now - self._tc_last_update < self.tc_interval_sec:
            return
        try:
            self._tc_offset = self.inlet.time_correction(timeout=0.5)
            self._tc_last_update = now
        except Exception as e:
            print(f"[EEGReader] time_correction error: {e}")

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._refresh_tc()
            try:
                chunk, timestamps = self.inlet.pull_chunk(
                    timeout=self.chunk_timeout,
                    max_samples=1024,
                )
            except Exception as e:
                print(f"[EEGReader] pull_chunk error: {e}")
                time.sleep(0.1)
                continue
            if chunk:
                samples = np.asarray(chunk, dtype=np.float32)
                ts = np.asarray(timestamps, dtype=np.float64) + self._tc_offset
                self.buffer.push_chunk(samples, ts)
                self.n_samples_total += samples.shape[0]


__all__ = [
    "EEGReader",
    "EEGStreamMeta",
    "local_clock",
    "make_feedback_outlet",
    "resolve_eeg_stream",
    "resolve_marker_stream",
]
