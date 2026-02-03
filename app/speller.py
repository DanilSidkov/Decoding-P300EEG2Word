# [file name]: speller.py
import logging
import sys
import tkinter as tk
from tkinter import messagebox

from pylsl import StreamInfo, StreamOutlet

from app.get_logger import setup_logger
from app.theme import ThemeManager
from monitor_config import setup_window_on_target_monitor

class SSVEPSpellerExperiment:
    def __init__(self, root):
        self.root = root
        self.logger = logging.getLogger("BCI")
        setup_logger(self.logger, "Experiment")

        # Инициализация менеджера тем
        self.theme_manager = ThemeManager()
        self.theme = self.theme_manager.get_theme()

        # Настраиваем окно на целевом мониторе
        if not setup_window_on_target_monitor(self.root):
            # Если не удалось, используем обычный fullscreen
            self.root.attributes("-fullscreen", True)
        self.root.configure(bg=self.theme["bg_primary"])
        
        self.root.bind("<Escape>", self._exit_program)
        self.root.protocol("WM_DELETE_WINDOW", self._exit_program)
        # Горячие клавиши для переключения темы
        self.root.bind("<Control-t>", lambda e: self._toggle_theme())
        self.root.bind("<Control-T>", lambda e: self._toggle_theme())

        self.codelen = 9
        self.cycle_duration = 9
        self.num_cycles = 10
        self.base_interval = self.cycle_duration / 9
        self.transition_duration = self.base_interval * 0.2

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self.controller = None
        info = StreamInfo(
            name="annotations",
            type="Events",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id="my_marker_stream",
        )
        self.outlet = StreamOutlet(info)
        
        # Текущее окно
        self.current_window = None

    def send_event_marker(self, event_type, **kwargs):
        """Отправка маркера события через LSL с поддержкой движения"""
        formatted_kwargs = {}
        for key, value in kwargs.items():
            if key == "states" and isinstance(value, str) and len(value) == 36:
                formatted_kwargs[key] = value
            else:
                formatted_kwargs[key] = value
        
        # Добавляем информацию о движении, если она есть
        if 'movement_direction' in kwargs:
            movement_info = {
                'direction': kwargs.get('movement_direction', 'N'),
                'direction_idx': kwargs.get('movement_direction_idx', 0),
                'amplitude': kwargs.get('movement_amplitude', 'small'),
                'amplitude_idx': kwargs.get('movement_amplitude_idx', 0),
                'description': kwargs.get('movement_description', 'N_small')
            }
            formatted_kwargs['movement'] = movement_info
        
        marker_string = f"{event_type} - {formatted_kwargs}"
        self.outlet.push_sample([marker_string])
        self.logger.info(f"Событие: {event_type} {formatted_kwargs}")

    def start(self):
        """Запускает последовательность окон"""
        self.logger.info("Начало работы")
        self._show_welcome()

    def _show_welcome(self):
        """Показывает приветственное окно"""
        from app.welcome import WelcomeWindow
        self.send_event_marker("WINDOW_OPEN_hello", window="welcome")
        self.logger.info("Показ приветственного окна")
        if self.current_window:
            self.current_window.window.destroy()
        self.current_window = WelcomeWindow(self._show_instructions, self.root, self.theme_manager)

    def _show_instructions(self):
        """Показывает окно инструкций"""
        from app.instructions import InstructionWindow
        self.send_event_marker("WINDOW_OPEN_instruct", window="instructions")
        self.logger.info("Показ окна инструкций")
        if self.current_window:
            self.current_window.window.destroy()
        self.current_window = InstructionWindow(self._show_preparation, self.root, self.theme_manager)

    def _show_preparation(self):
        """Показывает окно подготовки"""
        self.send_event_marker("WINDOW_OPEN_preparation", window="preparation")
        self.logger.info("Показ окна подготовки")
        if self.current_window:
            self.current_window.window.destroy()
        from app.preparation_window import PreparationWindow
        self.current_window = PreparationWindow(self._start_experiment_from_prep, self.root, self.theme_manager, self)
        self.current_window.window.deiconify()
        self.current_window.window.lift()

    def _start_experiment_from_prep(self, text, codelen, cycle_duration, num_cycles, stimulus_type="Мигание", motion_type="Дрожание"):
        """Начинает эксперимент с параметрами из окна подготовки"""
        try:
            self.codelen = int(codelen)
            self.cycle_duration = float(cycle_duration)
            self.num_cycles = int(num_cycles)
            self.base_interval = self.cycle_duration / self.codelen
            self.transition_duration = self.base_interval * 0.2

            if self.codelen <= 0 or self.cycle_duration <= 0 or self.num_cycles <= 0:
                raise ValueError("Все параметры должны быть больше 0")

            self.send_event_marker(
                "EXPERIMENT_START",
                text=text,
                codelen=self.codelen,
                duration=self.cycle_duration,
                cycles=self.num_cycles,
                stimulus_type=stimulus_type,
                motion_type=motion_type
            )

            self._setup_experiment(text.upper(), stimulus_type, motion_type)

        except ValueError as e:
            self.logger.critical(f"Некорректные данные: {e}")
            messagebox.showerror("Ошибка", f"Некорректные данные: {e}")

    def _setup_experiment(self, text, stimulus_type, motion_type):
        """Настраивает и запускает главное окно эксперимента"""
        self.logger.info("Настройка эксперимента")
        if self.current_window:
            self.current_window.window.destroy()
        from app.experiment_window import ExperimentWindow
        self.current_window = ExperimentWindow(self.root, self.theme_manager, self, text, stimulus_type, motion_type)

    def _toggle_theme(self):
        """Переключение темы для всего приложения"""
        self.theme = self.theme_manager.toggle_theme()
        if self.current_window and hasattr(self.current_window, '_update_theme'):
            self.current_window._update_theme()
        self.logger.info(f"Тема изменена на: {self.theme['name']}")

    def _exit_program(self, event=None):
        """Закрытие программы"""
        self.logger.info("Программа завершена пользователем")
        if messagebox.askyesno("Выход", "Вы уверены, что хотите выйти?"):
            if self.current_window:
                self.current_window.window.destroy()
            self.root.destroy()
            sys.exit(0)