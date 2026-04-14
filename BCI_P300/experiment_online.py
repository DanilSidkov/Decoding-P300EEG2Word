"""Online-версия P300 спеллера с feedback от инференс-процесса.

Отличия от experiment.py:
* В LSL отправляются именованные маркеры (протокол синхронизирован с
  src/inference/realtime_inference.py):
    "T:<letter>"  — начало trial
    "F:<letter>"  — начало активного движения буквы (onset эпохи)
    "E"           — конец trial (когда каждая буква стартовала
                    n_repetitions раз)
* Дополнительный LSL-инлет 'p300_feedback' слушает ответ модели:
    "P:<letter>"  — предсказанная таргет-буква → подсветить
    "READY"       — инференс готов
* После trial показывается фидбэк: предсказанная буква подсвечивается
  зелёным (совпала с таргетом) или красным (не совпала) в течение
  feedback_duration_ms.
"""

from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime

import numpy as np
import pygame
from pylsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    local_clock,
    resolve_byprop,
)


class OnlineExperiment:

    def __init__(self, data_config: dict) -> None:
        self.data_config = data_config
        self.sentence = data_config["sentence"]
        self.n_repetitions = int(data_config.get("n_repetitions", 10))
        self.feedback_duration_ms = int(
            data_config.get("feedback_duration_ms", 2000),
        )
        self.wait_feedback_timeout_ms = int(
            data_config.get("wait_feedback_timeout_ms", 5000),
        )
        self.wait_inference_ready_sec = float(
            data_config.get("wait_inference_ready_sec", 15.0),
        )

    # ------------------------------------------------------------------

    def fit(self) -> None:
        self.preinit()
        self.init()
        self.lsl_init()
        self._wait_for_inference_ready()

        for target_letter in self.sentence:
            self.postinit()
            self.perform(target_letter)
            if not self.running_global:
                break

        self.end_exp()

    # ------------------------------------------------------------------

    def preinit(self) -> None:
        x = self.data_config["window_x"]
        y = self.data_config["window_y"]
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
        os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"

    def init(self) -> None:
        pygame.init()
        flags = pygame.FULLSCREEN
        self.screen = pygame.display.set_mode(size=(0, 0), flags=flags)

        self.width = self.screen.get_width()
        self.height = self.screen.get_height()

        self.n_cols = self.data_config["num_cols"]
        self.n_rows = self.data_config["num_rows"]
        self.alphabet = self.data_config["alphabet"].replace("\n", "")

        self.w = self.width // self.n_cols
        self.h = self.height // self.n_rows
        self.sec_in_msec = 1e-3

        self.t0_mean = self.data_config["t0_mean"]
        self.t1_a = self.data_config["t1_a"]
        self.t1_b = self.data_config["t1_b"]
        self.t2_a = self.data_config["t2_a"]
        self.t2_b = self.data_config["t2_b"]

        self.delay = self.data_config["delay"]

        self.f = open(
            f'logs_online_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt',
            "w", encoding="utf-8",
        )

        self.running_global = True

    def lsl_init(self) -> None:
        # outlet маркеров стимулятора
        info = StreamInfo(
            name="annotations",
            type="Events",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id="p300_stim_markers",
        )
        self.outlet = StreamOutlet(info)

        # inlet фидбэка от инференса
        print("[STIM] Ищем feedback stream 'p300_feedback'...")
        streams = resolve_byprop("name", "p300_feedback", timeout=2.0)
        if streams:
            self.feedback_inlet: StreamInlet | None = StreamInlet(streams[0])
            print("[STIM] Feedback stream найден.")
        else:
            self.feedback_inlet = None
            print(
                "[STIM] Feedback stream не найден сразу — буду периодически "
                "проверять."
            )

    def _wait_for_inference_ready(self) -> None:
        """Ждём READY от инференса (или пока не таймаут)."""
        deadline = local_clock() + self.wait_inference_ready_sec
        print(
            f"[STIM] Ждём сигнал READY от инференса "
            f"(до {self.wait_inference_ready_sec}s)..."
        )
        while local_clock() < deadline:
            if self.feedback_inlet is None:
                streams = resolve_byprop("name", "p300_feedback", timeout=0.5)
                if streams:
                    self.feedback_inlet = StreamInlet(streams[0])
            if self.feedback_inlet is not None:
                try:
                    sample, _ = self.feedback_inlet.pull_sample(timeout=0.2)
                except Exception:
                    sample = None
                if sample and sample[0] == "READY":
                    print("[STIM] Инференс готов.")
                    return
        print("[STIM] Таймаут ожидания READY, запускаем без подтверждения.")

    # ------------------------------------------------------------------
    # speed func (как в оригинале)
    # ------------------------------------------------------------------

    def set_t1(self) -> float:
        return float(np.random.uniform(self.t1_a, self.t1_b))

    def set_t0(self) -> float:
        return self.t0_mean

    def set_t2(self) -> float:
        return float(np.random.uniform(self.t2_a, self.t2_b))

    def speed_func(self, t, t0=1, t1=0.5, t2=0.5, forward=True):
        if t0 <= 0:
            raise ValueError("freq must be positive")
        if t1 < 0:
            raise ValueError("t1 must be non negative")
        if t2 < 0:
            raise ValueError("t2 must be non negative")

        if t1 / self.sec_in_msec <= t <= (t1 + t0) / self.sec_in_msec:
            return math.sin(
                (t - t1 / self.sec_in_msec)
                * math.pi * self.sec_in_msec / t0
            ) * (2 * forward - 1)
        if (0 <= t < t1 / self.sec_in_msec
                or (t1 + t0) / self.sec_in_msec < t
                <= (t1 + t0 + t2) / self.sec_in_msec):
            return 0
        raise ValueError(
            f"t must be between 0 and "
            f"{(t1 + t0 + t2) / self.sec_in_msec}"
        )

    # ------------------------------------------------------------------

    def postinit(self) -> None:
        self.clock = pygame.time.Clock()
        self.FPS = self.data_config["FPS"]

        self.font_name = self.data_config["font_name"]
        self.font_size = self.data_config["font_size"]
        self.font = pygame.font.SysFont(
            name=self.font_name, size=self.font_size, bold=True,
        )

        self.cells = {}
        self.amplitude_x_scale = self.data_config["amplitude_x_scale"]
        self.amplitude_y_scale = self.data_config["amplitude_y_scale"]

        self.letter_foreground = (0, 0, 0)
        self.background = (255, 255, 255)
        # цвета подсветки
        self.color_correct = (0, 200, 0)
        self.color_wrong = (220, 0, 0)
        self.color_target_cue = (0, 0, 200)

        for id_, char in enumerate(self.alphabet):
            letter = self.font.render(char, True, self.letter_foreground)
            self.cells[char] = {
                "id": id_,
                "letter": letter,
                "amplitude_x": (self.w / 2) * self.amplitude_x_scale,
                "amplitude_y": (self.h / 2) * self.amplitude_y_scale,
                "t0": self.set_t0(),
                "t1": self.set_t1(),
                "t2": self.set_t2(),
                "dir": random.randint(0, 1),
                "start_t": 0,
                "start_marker": False,
                "end_marker": False,
                "n_starts": 0,
            }

        self.running = True
        self.start_experiment = False
        self.prev_t = -self.delay

    # ------------------------------------------------------------------

    def _render_letter(self, char, params, dx=0.0, dy=0.0, surface=None):
        if surface is None:
            surface = params["letter"]
        j = params["id"] // self.n_cols
        i = params["id"] % self.n_cols
        self.screen.blit(
            surface,
            (
                i * self.w + self.w / 2 - surface.get_width() / 2 + dx,
                j * self.h + self.h / 2 - surface.get_height() / 2 - dy,
            ),
        )

    def _render_rect(self, char, color):
        params = self.cells[char]
        j = params["id"] // self.n_cols
        i = params["id"] % self.n_cols
        rect = pygame.Rect(i * self.w, j * self.h, self.w, self.h)
        pygame.draw.rect(self.screen, color, rect, width=6)

    def _show_target_cue(self, target_letter: str, duration_ms: int = 1500):
        """Показывает таргет-букву (статично, с синей рамкой) перед trial."""
        t0 = pygame.time.get_ticks()
        while pygame.time.get_ticks() - t0 < duration_ms:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.end_exp()
                    return
                if (event.type == pygame.KEYDOWN
                        and event.key == pygame.K_ESCAPE):
                    self.end_exp()
                    return
            self.screen.fill(self.background)
            for char, params in self.cells.items():
                if char == target_letter:
                    self._render_rect(char, self.color_target_cue)
                    self._render_letter(char, params)
            pygame.display.flip()
            self.clock.tick(self.FPS)

    # ------------------------------------------------------------------

    def perform(self, target_letter: str) -> None:
        self.f.write(f"target letter {target_letter}\n")

        # 1) показать cue
        self._show_target_cue(target_letter)

        # 2) маркер начала trial
        self.outlet.push_sample([f"T:{target_letter}"], local_clock())

        # 3) основной цикл: ждём пока каждая буква не стартует
        #    n_repetitions раз, затем отправим "E"
        t0_ticks = pygame.time.get_ticks()
        self.prev_t = -self.delay
        self.start_experiment = True

        while self.running:
            dt = self.clock.tick(self.FPS)  # noqa: F841
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.end_exp()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.end_exp()
                        return
                    if event.key == pygame.K_SPACE:
                        # SPACE во время trial = аварийный выход trial
                        self.running = False

            self.screen.fill(self.background)

            t = pygame.time.get_ticks() - t0_ticks
            if t - self.prev_t >= self.delay:
                self.prev_t = t
                # heartbeat sync (совместимость с логом)
                self.outlet.push_sample([str(t)], local_clock())
                self.f.write(f"sync {t}\n")

            for char, params in self.cells.items():
                try:
                    speed = self.speed_func(
                        t - params["start_t"],
                        params["t0"], params["t1"], params["t2"],
                        params["dir"],
                    )

                    if (t >= params["start_t"]
                            + params["t1"] / self.sec_in_msec
                            and not params["start_marker"]):
                        # момент onset эпохи → LSL-маркер + лог
                        self.outlet.push_sample(
                            [f"F:{char}"], local_clock(),
                        )
                        self.f.write(f"letter {char} start {t}\n")
                        params["start_marker"] = True
                        params["n_starts"] += 1

                    if (t >= params["start_t"]
                            + (params["t1"] + params["t0"])
                            / self.sec_in_msec
                            and not params["end_marker"]):
                        params["end_marker"] = True

                except ValueError:
                    params["start_t"] = t
                    params["t1"] = self.set_t1()
                    params["t2"] = self.set_t2()
                    params["dir"] = random.randint(0, 1)
                    params["start_marker"] = False
                    params["end_marker"] = False
                    speed = self.speed_func(
                        t - params["start_t"],
                        params["t0"], params["t1"], params["t2"],
                        params["dir"],
                    )

                dx = params["amplitude_x"] * speed \
                    if self.data_config["x_move"] else 0
                dy = params["amplitude_y"] * speed \
                    if self.data_config["y_move"] else 0

                self._render_letter(char, params, dx=dx, dy=dy)

            pygame.display.flip()

            # условие конца trial: каждая буква стартовала ≥ n_repetitions раз
            if min(p["n_starts"] for p in self.cells.values()) \
                    >= self.n_repetitions:
                self.running = False

        # 4) end-of-trial marker
        self.outlet.push_sample(["E"], local_clock())
        self.f.write("trial end\n")
        self.start_experiment = False

        # 5) ждём P:* от инференса
        predicted = self._wait_for_prediction()

        # 6) фидбэк
        if predicted is not None:
            color = (
                self.color_correct
                if predicted == target_letter
                else self.color_wrong
            )
            self._show_feedback(predicted, target_letter, color)
        else:
            print("[STIM] Предсказание не получено — пропускаем фидбэк.")

    # ------------------------------------------------------------------

    def _wait_for_prediction(self) -> str | None:
        if self.feedback_inlet is None:
            return None
        deadline_ms = pygame.time.get_ticks() + self.wait_feedback_timeout_ms
        while pygame.time.get_ticks() < deadline_ms:
            # обрабатываем события, чтобы окно не "заморозилось"
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if (event.type == pygame.KEYDOWN
                        and event.key == pygame.K_ESCAPE):
                    return None
            try:
                sample, _ = self.feedback_inlet.pull_sample(timeout=0.05)
            except Exception:
                sample = None
            if sample and isinstance(sample[0], str):
                payload = sample[0]
                if payload.startswith("P:"):
                    return payload[2:]
            # держим экран белым (пустое межтрайловое состояние)
            self.screen.fill(self.background)
            pygame.display.flip()
            self.clock.tick(30)
        return None

    def _show_feedback(
        self, predicted: str, target: str, color: tuple[int, int, int],
    ) -> None:
        self.f.write(f"predicted {predicted} target {target}\n")
        t0 = pygame.time.get_ticks()
        while pygame.time.get_ticks() - t0 < self.feedback_duration_ms:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.end_exp()
                    return
                if (event.type == pygame.KEYDOWN
                        and event.key == pygame.K_ESCAPE):
                    self.end_exp()
                    return
            self.screen.fill(self.background)
            for char, params in self.cells.items():
                if char == predicted:
                    self._render_rect(char, color)
                self._render_letter(char, params)
            pygame.display.flip()
            self.clock.tick(self.FPS)

    # ------------------------------------------------------------------

    def end_exp(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass
        pygame.quit()
        self.running_global = False


def main() -> None:
    with open(
        os.path.join(os.path.dirname(__file__), "settings.json"),
        encoding="utf-8",
    ) as file:
        data_config = json.load(file)
    exp = OnlineExperiment(data_config)
    exp.fit()


if __name__ == "__main__":
    main()
