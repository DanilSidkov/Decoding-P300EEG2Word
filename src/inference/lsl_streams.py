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
) -> tuple[StreamInlet, EEGStreamMeta]:
    """Находит EEG-поток через LSL.

    Сначала пытается найти по имени, затем любой с type='EEG'.
    """
    # сначала по имени
    for name in preferred_names:
        streams = resolve_byprop("name", name, timeout=1.0)
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

    # затем по типу
    streams = resolve_byprop("type", "EEG", timeout=timeout)
    if not streams:
        # в крайнем случае — любые потоки
        streams = resolve_streams(wait_time=timeout)
        streams = [s for s in streams if s.type().lower() == "eeg"]
    if not streams:
        raise RuntimeError(
            "LSL EEG-поток не найден. Запущен ли NeoRec с LSL broadcast?"
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
    """Фоновый поток: читает сэмплы из EEG-инлета в ring buffer."""

    def __init__(
        self,
        inlet: StreamInlet,
        buffer: EEGRingBuffer,
        chunk_timeout: float = 0.2,
    ) -> None:
        super().__init__(daemon=True)
        self.inlet = inlet
        self.buffer = buffer
        self.chunk_timeout = chunk_timeout
        self._stop_event = threading.Event()
        self.n_samples_total = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
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
                ts = np.asarray(timestamps, dtype=np.float64)
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
