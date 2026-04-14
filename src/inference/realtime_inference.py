"""Главный оркестратор real-time P300 инференса.

Протокол маркеров (inbound, от стимулятора):
    "T:<letter>"      — начало trial, target letter = <letter>
    "F:<letter>"      — начало движения буквы <letter> (epoch onset)
    "E"               — конец trial, требуется предсказание

Протокол маркеров (outbound, в стимулятор):
    "P:<letter>"      — предсказанная таргет-буква
    "READY"           — инференс поднят, можно запускать стимуляцию
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from src.inference.eeg_buffer import EEGRingBuffer
from src.inference.letter_aggregator import TrialAggregator
from src.inference.lsl_streams import (
    EEGReader,
    make_feedback_outlet,
    resolve_eeg_stream,
    resolve_marker_stream,
)
from src.inference.online_preprocessor import OnlineEpochPreprocessor
from src.models.p300net import P300EEGNet


@dataclass
class InferenceConfig:
    model_path: str
    marker_stream_name: str = "annotations"
    feedback_stream_name: str = "p300_feedback"
    buffer_capacity_sec: float = 10.0
    filter_pad_sec: float = 1.0
    average_last: int | None = None
    device: str | None = None
    eeg_unit_scale: float = 1e-6
    log_path: str | None = None


@dataclass
class PendingEpoch:
    letter: str
    t_marker: float     # LSL timestamp маркера
    ready_at: float     # local_clock() когда можно пытаться вырезать


class RealtimeInference:
    """Чтение EEG + маркеров, инференс, отсылка предсказаний."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.device = (
            config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- модель + метаданные из чекпоинта ---
        print(f"[RT] Загружаем чекпоинт: {config.model_path}")
        ckpt = torch.load(
            config.model_path, map_location=self.device, weights_only=False,
        )
        self.target_channel_names: list[str] = ckpt["channel_names"]
        self.n_times_model: int = ckpt["n_times"]
        self.norm_stats = ckpt.get("norm_stats")
        n_channels = ckpt["n_channels"]

        self.model = P300EEGNet(
            n_channels=n_channels,
            n_times=self.n_times_model,
            n_classes=2,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(
            f"[RT] Модель загружена: {n_channels} каналов, "
            f"{self.n_times_model} точек, device={self.device}"
        )
        print(f"[RT] Каналы модели: {self.target_channel_names}")

        # --- LSL: EEG inlet + reader ---
        print("[RT] Ищем EEG LSL stream...")
        self.eeg_inlet, self.eeg_meta = resolve_eeg_stream()
        print(
            f"[RT] EEG stream: name='{self.eeg_meta.name}', "
            f"sfreq={self.eeg_meta.sfreq}, n_ch={self.eeg_meta.n_channels}"
        )
        print(f"[RT] LSL каналы: {self.eeg_meta.channel_names}")

        self.buffer = EEGRingBuffer(
            n_channels=self.eeg_meta.n_channels,
            sfreq=self.eeg_meta.sfreq,
            capacity_sec=config.buffer_capacity_sec,
        )
        self.eeg_reader = EEGReader(self.eeg_inlet, self.buffer)

        # --- препроцессор ---
        self.preproc = OnlineEpochPreprocessor(
            raw_channel_names=self.eeg_meta.channel_names,
            target_channel_names=self.target_channel_names,
            sfreq=self.eeg_meta.sfreq,
            norm_stats=self.norm_stats,
            filter_pad_sec=config.filter_pad_sec,
            eeg_unit_scale=config.eeg_unit_scale,
        )

        # --- LSL: markers in + feedback out ---
        print(f"[RT] Ищем marker stream '{config.marker_stream_name}'...")
        self.marker_inlet = resolve_marker_stream(
            config.marker_stream_name,
        )
        print("[RT] Marker stream найден.")
        self.feedback_out = make_feedback_outlet(
            name=config.feedback_stream_name,
        )

        # --- очереди ---
        self._pending: deque[PendingEpoch] = deque()
        self._aggregator = TrialAggregator(
            average_last=config.average_last,
        )
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        # лог трайлов
        self._trial_log: list[dict] = []
        self._trial_idx = -1

    # ------------------------------------------------------------------
    # жизненный цикл
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.eeg_reader.start()
        # прогрев буфера
        print("[RT] Прогрев EEG буфера (2с)...")
        time.sleep(2.0)
        print(
            f"[RT] Получено {self.eeg_reader.n_samples_total} сэмплов. Готов."
        )
        self.feedback_out.push_sample(["READY"])
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.eeg_reader.stop()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        self._save_log()

    def _save_log(self) -> None:
        if not self.config.log_path or not self._trial_log:
            return
        import json
        from pathlib import Path
        p = Path(self.config.log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._trial_log, f, ensure_ascii=False, indent=2)
        print(f"[RT] Лог сохранён: {p}")

    # ------------------------------------------------------------------
    # главный цикл
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        from pylsl import local_clock

        min_wait_sec = self.preproc.tmax + self.preproc.filter_pad_sec + 0.05

        while not self._stop_event.is_set():
            # 1. читаем маркеры неблокирующе
            try:
                sample, ts = self.marker_inlet.pull_sample(timeout=0.05)
            except Exception as e:
                print(f"[RT] marker pull error: {e}")
                sample, ts = None, None

            if sample is not None and sample[0]:
                self._handle_marker(sample[0], float(ts))

            # 2. обрабатываем pending эпохи, у которых подоспел хвост EEG
            now = local_clock()
            while self._pending and self._pending[0].ready_at <= now:
                pe = self._pending.popleft()
                self._process_epoch(pe)

            # 3. тайм-ауты pending (если EEG не догнал)
            cutoff = now - 5.0  # 5с без сигнала — мусор
            while self._pending and self._pending[0].ready_at < cutoff:
                pe = self._pending.popleft()
                print(
                    f"[RT] Сбросили устаревшую эпоху '{pe.letter}' "
                    f"(t={pe.t_marker:.3f})"
                )

            time.sleep(0.005)

    # ------------------------------------------------------------------
    # обработка маркеров
    # ------------------------------------------------------------------

    def _handle_marker(self, payload: str, t_marker: float) -> None:
        payload = payload.strip()
        if not payload:
            return

        if payload.startswith("T:"):
            letter = payload[2:]
            self._trial_idx += 1
            self._aggregator.reset()
            self._aggregator.target_letter = letter
            self._pending.clear()
            print(
                f"\n[RT] === Trial {self._trial_idx}: target='{letter}' ==="
            )
            return

        if payload.startswith("F:"):
            letter = payload[2:]
            if not letter:
                return
            # когда можно будет вырезать: t_marker + tmax + pad
            ready_at = (
                t_marker
                + self.preproc.tmax
                + self.preproc.filter_pad_sec
                + 0.02
            )
            self._pending.append(PendingEpoch(
                letter=letter, t_marker=t_marker, ready_at=ready_at,
            ))
            return

        if payload == "E":
            self._finalize_trial()
            return

        # sync-маркеры от старого experiment.py: просто число
        # игнорируем их
        if payload.lstrip("-").isdigit():
            return

        print(f"[RT] Неизвестный маркер: '{payload}'")

    # ------------------------------------------------------------------
    # инференс одной эпохи
    # ------------------------------------------------------------------

    def _process_epoch(self, pe: PendingEpoch) -> None:
        t_start, t_end = self.preproc.window_limits(pe.t_marker)
        window = self.buffer.get_window(t_start, t_end)
        if window is None:
            latest = self.buffer.latest_timestamp
            print(
                f"[RT] Нет окна EEG для '{pe.letter}' "
                f"(need [{t_start:.3f}, {t_end:.3f}], latest={latest:.3f})"
            )
            return
        samples, timestamps = window
        epoch = self.preproc.process(samples, timestamps, pe.t_marker)
        if epoch is None:
            return

        # аккуратно приводим число точек к n_times_model
        if epoch.shape[1] != self.n_times_model:
            if epoch.shape[1] > self.n_times_model:
                epoch = epoch[:, : self.n_times_model]
            else:
                pad = self.n_times_model - epoch.shape[1]
                epoch = np.pad(
                    epoch, ((0, 0), (0, pad)), mode="edge",
                )

        with torch.no_grad():
            t = torch.from_numpy(epoch).unsqueeze(0).to(self.device)
            logits = self.model(t)
            p_target = float(
                torch.softmax(logits, dim=1)[0, 1].cpu().item()
            )

        self._aggregator.add(pe.letter, p_target)
        cnt = len(self._aggregator.scores[pe.letter])
        print(
            f"[RT]  · '{pe.letter}' #{cnt}: P(target)={p_target:.3f}"
        )

    # ------------------------------------------------------------------
    # завершение trial
    # ------------------------------------------------------------------

    def _finalize_trial(self) -> None:
        # ждём пока все pending обработаются
        deadline = time.time() + 3.0
        while self._pending and time.time() < deadline:
            time.sleep(0.05)

        predicted, scores = self._aggregator.predict()
        target = self._aggregator.target_letter

        if predicted is None:
            print("[RT] Trial завершился без эпох — нет предсказания.")
            return

        # топ-3 для отладки
        top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3_s = ", ".join(f"{l}={p:.3f}" for l, p in top3)
        mark = "✓" if predicted == target else "✗"
        print(
            f"[RT] Trial end. Target='{target}', pred='{predicted}' {mark} "
            f"| top3: {top3_s}"
        )

        # отправляем предсказание в стимулятор
        self.feedback_out.push_sample([f"P:{predicted}"])

        # лог
        self._trial_log.append({
            "trial_index": self._trial_idx,
            "target": target,
            "predicted": predicted,
            "correct": predicted == target,
            "scores": scores,
            "counts": self._aggregator.counts(),
        })
