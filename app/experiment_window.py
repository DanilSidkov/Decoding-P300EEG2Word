import logging
import sys
import threading
import time
import tkinter as tk
import math
from tkinter import messagebox, ttk

from app.code_generator import CodeGen
from app.theme import ThemeManager


class ExperimentWindow:
    """Главное окно эксперимента"""
    
    def __init__(self, root_window, theme_manager, experiment_instance, target_text, stimulus_type="Мигание", motion_type="Дрожание"):
        self.root = root_window
        self.theme_manager = theme_manager
        self.experiment_instance = experiment_instance
        self.theme = theme_manager.get_theme()
        self.target_text = target_text
        self.stimulus_type = stimulus_type  # "Мигание", "Движение", "Комбинированный"
        self.motion_type = motion_type  # "Дрожание", "Колебание размера"

        self.window = root_window
        self.window.title("BCI Speller - Эксперимент")
        
        # Используем тему
        self.window.attributes("-fullscreen", True)
        
        # Холст для эффектов
        self.canvas = tk.Canvas(self.window, bg=self.theme["bg_primary"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Привязка клавиш
        self.window.bind("<Escape>", self._exit_program)
        self.window.protocol("WM_DELETE_WINDOW", self._exit_program)

        # Инициализация переменных эксперимента
        self.is_running = False
        self.current_interval = 0
        self.current_cycle = 0
        self.output_text = ""
        self.flash_thread = None
        
        self.target_symbol = ""
        self.target_symbols = list(target_text)
        self.current_target_index = 0
        
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        # Словарь для хранения оригинальных положений и свойств символов
        self.symbol_properties = {}  # key: label, value: dict with original properties
        
        # Инициализация символов и паттернов
        self.setup_symbols()
        
        # Создаем все виджеты
        self._create_ui()

    def setup_symbols(self):
        """Настройка символов"""
        CG = CodeGen(self.experiment_instance.codelen)
        symbols_alphabetical = CG.alphabet
        
        keyboard_order = [
            "Й", "Ц", "У", "К", "Е", "Н", "Г", "Ш", "Щ", "З", "Х", "Ъ",
            "Ф", "Ы", "В", "А", "П", "Р", "О", "Л", "Д", "Ж", "Э", "Ё",
            "Я", "Ч", "С", "М", "И", "Т", "Ь", "Б", "Ю", ",", ".", "_"
        ]

        if set(keyboard_order) != set(symbols_alphabetical):
            raise ValueError(
                "Набор символов в keyboard_order не совпадает с symbols_alphabetical"
            )
        
        self.symbol_indices = [
            symbols_alphabetical.index(sym) for sym in keyboard_order
        ]
        self.symbols = keyboard_order
        self.patterns = [CG.patterns[i] for i in self.symbol_indices]

    def _create_ui(self):
        """Создает весь интерфейс окна"""
        # Очищаем холст
        self.canvas.delete("all")
        
        # Устанавливаем фон окна
        self.window.configure(bg=self.theme["bg_primary"])
        self.canvas.configure(bg=self.theme["bg_primary"])
        
        # Создаем кнопки управления (тема и выход)
        self._create_control_buttons()
        
        # Создаем основной контент
        self._create_main_content()

    def _create_control_buttons(self):
        """Создает кнопки управления (тема и выход)"""
        # Кнопка переключения темы
        self.theme_button = tk.Button(
            self.canvas,
            text="☀️" if self.theme_manager.is_dark_mode else "🌙",
            font=("Segoe UI", 12, "bold"),
            command=self._toggle_theme,
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["accent_primary"],
            activeforeground=self.theme["text_primary"],
            relief="flat",
            width=3,
            height=1,
            bd=2,
            highlightthickness=2,
            highlightbackground=self.theme["border_primary"],
            highlightcolor=self.theme["border_primary"],
            cursor="hand2"
        )
        self.theme_button.place(x=self.screen_width - 60, y=20)
        self._create_glow_effect(self.theme_button, self.theme["accent_primary"])

        # Кнопка выхода
        self.exit_button = tk.Button(
            self.canvas,
            text="✕",
            font=("Segoe UI", 14, "bold"),
            command=self._exit_program,
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["accent_warning"],
            activeforeground=self.theme["text_primary"],
            relief="flat",
            width=3,
            height=1,
            bd=0,
            cursor="hand2"
        )
        self.exit_button.place(x=20, y=20)
        self._create_glow_effect(self.exit_button, self.theme["accent_warning"])

    def _create_main_content(self):
        """Создает основной контент главного окна"""
        # Основной контейнер
        main_container = tk.PanedWindow(
            self.canvas,
            orient=tk.VERTICAL,
            bg=self.theme["bg_primary"],
            sashwidth=5,
            sashrelief="flat",
            sashpad=3,
            opaqueresize=False
        )
        main_container.place(relx=0.5, rely=0.5, anchor="center", 
                           width=self.screen_width*0.9, height=self.screen_height*0.9)

        # Верхняя панель (80%)
        top_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        main_container.add(top_frame, height=int(self.screen_height * 0.9 * 0.8))

        # Панель информации
        info_frame = tk.Frame(top_frame, bg=self.theme["bg_primary"])
        info_frame.pack(fill=tk.X, padx=20, pady=(20, 10))

        # Индикатор статуса
        status_group = tk.Frame(info_frame, bg=self.theme["bg_primary"])
        status_group.pack(side=tk.LEFT)

        self.status_light = tk.Canvas(status_group, width=20, height=20, 
                                     bg=self.theme["bg_primary"], highlightthickness=0)
        self.status_light.pack(side=tk.LEFT, padx=(0, 10))
        self.status_indicator = self.status_light.create_oval(2, 2, 18, 18, 
                                                            fill=self.theme["text_tertiary"], outline="")

        # Обновляем текст статуса в зависимости от типа стимула
        stimulus_text = {
            "Мигание": "Мигание",
            "Движение": "Движение",
            "Комбинированный": "Мигание+Движение"
        }
        
        self.status_label = tk.Label(
            status_group,
            text=f"Тип стимула: {stimulus_text.get(self.stimulus_type, 'Мигание')}",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
        )
        self.status_label.pack(side=tk.LEFT)

        # Индикатор стимуляции
        stimulus_group = tk.Frame(info_frame, bg=self.theme["bg_primary"])
        stimulus_group.pack(side=tk.RIGHT)

        self.stimulus_indicator = tk.Label(
            stimulus_group,
            text="○",
            font=("Segoe UI", 16),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_tertiary"],
        )
        self.stimulus_indicator.pack(side=tk.LEFT, padx=(0, 5))

        stimulus_type_text = {
            "Мигание": "МИГАНИЕ",
            "Движение": "ДВИЖЕНИЕ",
            "Комбинированный": "МИГАНИЕ+ДВИЖ"
        }
        
        tk.Label(
            stimulus_group,
            text=stimulus_type_text.get(self.stimulus_type, "СТИМУЛЯЦИЯ"),
            font=("Segoe UI", 11),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"],
        ).pack(side=tk.LEFT)

        # Панель прогресса
        progress_frame = tk.Frame(top_frame, bg=self.theme["bg_primary"])
        progress_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.current_symbol_label = tk.Label(
            progress_frame,
            text="Ожидание целевого символа...",
            font=("Segoe UI", 14),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_primary"],
        )
        self.current_symbol_label.pack(anchor="w", pady=(0, 5))

        # Прогресс-бар эксперимента
        progress_bar_frame = tk.Frame(progress_frame, bg=self.theme["bg_primary"])
        progress_bar_frame.pack(fill=tk.X, pady=(0, 10))

        self.experiment_progress = tk.Label(
            progress_bar_frame,
            text=f"Прогресс: 0/{len(self.target_symbols)}",
            font=("Segoe UI", 11),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"],
        )
        self.experiment_progress.pack(side=tk.LEFT)
        
        # Сетка символов
        grid_container = tk.Frame(top_frame, bg=self.theme["bg_primary"])
        grid_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self._create_symbol_grid(grid_container)

        # Нижняя панель (20%)
        bottom_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        main_container.add(bottom_frame, height=int(self.screen_height * 0.9 * 0.2))

        bottom_paned = tk.PanedWindow(
            bottom_frame,
            orient=tk.HORIZONTAL,
            bg=self.theme["bg_primary"],
            sashwidth=3,
            sashrelief="flat",
            opaqueresize=False
        )
        bottom_paned.pack(fill=tk.BOTH, expand=True)

        # Левая панель (параметры)
        left_bottom_frame = tk.Frame(bottom_paned, bg=self.theme["bg_secondary"])
        bottom_paned.add(left_bottom_frame, width=int(self.screen_width * 0.9 * 0.4))

        # Заголовок параметров
        tk.Label(
            left_bottom_frame,
            text="ПАРАМЕТРЫ ЭКСПЕРИМЕНТА",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["accent_primary"],
        ).pack(anchor="w", padx=20, pady=(15, 10))

        # Настройки
        settings_text = f"Тип: {self.stimulus_type} | "
        if self.stimulus_type != "Мигание":
            settings_text += f"Движение: {self.motion_type} | "
        settings_text += f"Цикл: {self.experiment_instance.cycle_duration} сек | Циклов: {self.experiment_instance.num_cycles}"
        
        self.settings_label = tk.Label(
            left_bottom_frame,
            text=settings_text,
            font=("Segoe UI", 10),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_secondary"],
            wraplength=350,
            justify="left"
        )
        self.settings_label.pack(anchor="w", padx=20, pady=(0, 5))

        # Прогресс стимуляции
        self.progress_label = tk.Label(
            left_bottom_frame,
            text=f"Цикл: 0/{self.experiment_instance.num_cycles} | Интервал: 0/{self.experiment_instance.codelen}",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["accent_success"],
        )
        self.progress_label.pack(anchor="w", padx=20, pady=(5, 5))

        # Правая панель (результаты)
        right_bottom_frame = tk.Frame(bottom_paned, bg=self.theme["bg_secondary"])
        bottom_paned.add(right_bottom_frame, width=int(self.screen_width * 0.9 * 0.6))

        # Заголовок результатов
        tk.Label(
            right_bottom_frame,
            text="РЕЗУЛЬТАТ ВВОДА",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["accent_primary"],
        ).pack(anchor="w", padx=20, pady=(15, 10))

        # Текстовое поле для результатов
        text_container = tk.Frame(right_bottom_frame, bg=self.theme["bg_secondary"])
        text_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))

        # Кастомный скроллбар
        style = ttk.Style()
        style.configure("Results.Vertical.TScrollbar", 
                       background=self.theme["scrollbar_bg"],
                       troughcolor=self.theme["scrollbar_trough"],
                       bordercolor=self.theme["bg_secondary"],
                       arrowcolor=self.theme["accent_primary"],
                       relief="flat")

        self.text_display = tk.Text(
            text_container,
            font=("Consolas", 14),
            height=3,
            wrap=tk.WORD,
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
            insertbackground=self.theme["accent_primary"],
            selectbackground=self.theme["accent_primary"],
            selectforeground=self.theme["text_primary"],
            relief="flat",
            bd=2,
            highlightthickness=1,
            highlightbackground=self.theme["border_secondary"],
            highlightcolor=self.theme["border_secondary"],
            padx=15,
            pady=10
        )

        scrollbar = ttk.Scrollbar(text_container, orient="vertical", 
                                 command=self.text_display.yview,
                                 style="Results.Vertical.TScrollbar")
        self.text_display.configure(yscrollcommand=scrollbar.set)

        self.text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Запускаем эксперимент
        self.window.after(100, self._show_next_target)

    def _create_symbol_grid(self, parent):
        """Создает сетку СИМВОЛОВ с использованием place для возможности движения"""
        self.grid_frame = tk.Frame(parent, bg=self.theme["bg_primary"])
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Ждем обновления геометрии для получения реальных размеров
        self.grid_frame.update_idletasks()
        
        rows = 3
        cols = 12
        
        # Вычисляем размеры ячеек
        cell_width = self.grid_frame.winfo_width() // cols
        cell_height = self.grid_frame.winfo_height() // rows
        
        self.labels = []
        
        for i, symbol in enumerate(self.symbols):
            row = i // cols
            col = i % cols
            
            # Вычисляем координаты центра ячейки
            x = col * cell_width + cell_width // 2
            y = row * cell_height + cell_height // 2
            
            # СОЗДАЕМ ЛЕЙБЛЫ С СИМВОЛАМИ с использованием place
            label = tk.Label(
                self.grid_frame,
                text=symbol,
                font=("Segoe UI", 22, "bold"),
                bg=self.theme["bg_primary"],
                fg=self.theme["text_tertiary"],
                width=3,
                height=1,
            )
            label.place(x=x, y=y, anchor="center")
            self.labels.append(label)
            
            # Сохраняем оригинальные свойства символа
            self.symbol_properties[label] = {
                'row': row,
                'col': col,
                'symbol': symbol,
                'font_size': 22,
                'base_x': x,
                'base_y': y,
                'cell_width': cell_width,
                'cell_height': cell_height
            }

    def _create_glow_effect(self, widget, color=None):
        """Создает эффект свечения для виджета"""
        if color is None:
            color = self.theme["accent_primary"]
            
        def on_enter(e):
            if hasattr(widget, 'config'):
                widget.config(highlightbackground=color, highlightcolor=color)
            
        def on_leave(e):
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            darker = f'#{max(0, r-30):02x}{max(0, g-30):02x}{max(0, b-30):02x}'
            if hasattr(widget, 'config'):
                widget.config(highlightbackground=darker, highlightcolor=darker)
            
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _toggle_theme(self):
        """Переключение темы"""
        self.theme = self.theme_manager.toggle_theme()
        self._update_theme()
        
    def _update_theme(self):
        """Обновление цветов согласно теме"""
        # Полностью пересоздаем интерфейс с новой темой
        self._create_ui()

    def _exit_program(self, event=None):
        """Закрытие программы"""
        if messagebox.askyesno("Выход", "Вы уверены, что хотите выйти?"):
            self.window.destroy()
            sys.exit(0)

    def _show_next_target(self):
        """Показывает окно с целевым символом для ввода"""
        if self.current_target_index < len(self.target_symbols):
            self.target_symbol = self.target_symbols[self.current_target_index]

            self.experiment_instance.send_event_marker(
                "Смотрите на целевой символ и нажмите ПРОБЕЛ",
                symbol=self.target_symbol,
                index=self.current_target_index,
                total=len(self.target_symbols),
                stimulus_type=self.stimulus_type,
                motion_type=self.motion_type
            )

            # Обновляем индикатор статуса
            self.status_light.itemconfig(self.status_indicator, fill="#f39c12")
            self.status_label.config(
                text=f"Целевой символ: '{self.target_symbol}' ({self.current_target_index + 1}/{len(self.target_symbols)})",
                fg="#f39c12"
            )
            
            self.current_symbol_label.config(
                text=f"СИМВОЛ: '{self.target_symbol}' • ПОЗИЦИЯ: {self.current_target_index + 1}/{len(self.target_symbols)} • ТИП: {self.stimulus_type}"
            )
            
            self.experiment_progress.config(
                text=f"Прогресс: {self.current_target_index}/{len(self.target_symbols)}"
            )

            from app.target_window import TargetWindow
            self.target_win = TargetWindow(self.window, self.target_symbol, self._on_target_confirmed, self.theme_manager)

        else:
            self._finish_experiment()

    def _on_target_confirmed(self):
        """Вызывается после подтверждения целевого символа"""
        self.experiment_instance.send_event_marker(
            "TARGET_CONFIRMED",
            symbol=self.target_symbol,
            index=self.current_target_index,
            stimulus_type=self.stimulus_type,
            motion_type=self.motion_type
        )
        self.window.deiconify()

        # Обновляем индикатор статуса
        self.status_light.itemconfig(self.status_indicator, fill=self.theme["accent_warning"])
        
        stimulus_text = {
            "Мигание": "Мигание... Смотрите на целевой символ",
            "Движение": "Движение... Смотрите на целевой символ",
            "Комбинированный": "Мигание+Движение... Смотрите на целевой символ"
        }
        
        self.status_label.config(
            text=stimulus_text.get(self.stimulus_type, "Стимуляция... Смотрите на целевой символ"),
            fg=self.theme["accent_warning"]
        )

        self._start_stimulation()

    def _start_stimulation(self):
        """Запускает стимуляцию для текущего символа"""
        if not self.is_running:
            self.experiment_instance.send_event_marker(
                "STIMULATION_START",
                symbol=self.target_symbol,
                duration=self.experiment_instance.cycle_duration,
                cycles=self.experiment_instance.num_cycles,
                codelen=self.experiment_instance.codelen,
                stimulus_type=self.stimulus_type,
                motion_type=self.motion_type
            )
            self.is_running = True
            self.current_interval = 0
            self.current_cycle = 0

            self._update_progress_display()
            self._highlight_target_symbol()

            self.stimulation_thread = threading.Thread(
                target=self._stimulation_sequence, daemon=True
            )
            self.stimulation_thread.start()

    def _highlight_target_symbol(self):
        """Подсвечивает целевой символ в сетке"""
        target_index = self.symbols.index(self.target_symbol) if self.target_symbol in self.symbols else -1
        if target_index != -1:
            for i, label in enumerate(self.labels):
                if i == target_index:
                    label.config(fg=self.theme["symbol_target_0"])
                else:
                    label.config(fg=self.theme["symbol_custom_0"])

    def _update_progress_display(self):
        """Обновляет отображение прогресса стимуляции"""
        self.progress_label.config(
            text=f"Цикл: {self.current_cycle}/{self.experiment_instance.num_cycles} | Интервал: {self.current_interval}/{self.experiment_instance.codelen}"
        )
        
        # Обновляем индикатор стимуляции
        if self.is_running:
            self.stimulus_indicator.config(text="●", fg=self.theme["accent_warning"])
        else:
            self.stimulus_indicator.config(text="○", fg=self.theme["text_tertiary"])

    def _apply_motion(self, label, active, interval):
        """Применяет движение к символу в зависимости от типа движения"""
        if not active:
            # Сбрасываем движение для неактивных символов
            props = self.symbol_properties[label]
            
            # ВАЖНО: Используем place с абсолютными координатами вместо grid
            # Вычисляем абсолютные координаты для центрирования
            cell_width = self.grid_frame.winfo_width() // 12
            cell_height = self.grid_frame.winfo_height() // 3
            
            x = props['col'] * cell_width + cell_width // 2
            y = props['row'] * cell_height + cell_height // 2
            
            label.place(x=x, y=y, anchor="center")
            label.config(font=("Segoe UI", props['font_size'], "bold"))
            return
        
        # Вычисляем фазу движения на основе интервала
        phase = (interval % 10) / 10.0 * 2 * math.pi
        
        # Получаем исходные свойства
        props = self.symbol_properties[label]
        cell_width = self.grid_frame.winfo_width() // 12
        cell_height = self.grid_frame.winfo_height() // 3
        
        # Базовые координаты (центр ячейки)
        base_x = props['col'] * cell_width + cell_width // 2
        base_y = props['row'] * cell_height + cell_height // 2
        
        if self.motion_type == "Дрожание":
            # Дрожание: небольшие случайные смещения
            import random
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            
            label.place(x=base_x + x_offset, y=base_y + y_offset, anchor="center")
            
        elif self.motion_type == "Колебание размера":
            # Колебание размера: синусоидальное изменение размера шрифта
            scale = 0.5 + 0.5 * math.sin(phase * 2)  # От 0.8 до 1.2
            font_size = int(props['font_size'] * scale)
            label.config(font=("Segoe UI", font_size, "bold"))
            label.place(x=base_x, y=base_y, anchor="center")

    def _stimulation_sequence(self):
        """Выполняет последовательность стимуляции с маркерами"""
        for cycle in range(self.experiment_instance.num_cycles):
            if not self.is_running:
                break
            self.current_cycle = cycle + 1
            self.window.after(0, self._update_progress_display)

            target_index = (
                self.symbols.index(self.target_symbol)
                if self.target_symbol in self.symbols
                else -1
            )

            # ВАЖНЫЙ МАРКЕР: НАЧАЛО ЦИКЛА
            self.experiment_instance.send_event_marker(
                "CYCLE_START",
                cycle=cycle + 1,
                total_cycles=self.experiment_instance.num_cycles,
                target_symbol=self.target_symbol,
                target_index=target_index,
                stimulus_type=self.stimulus_type,
                motion_type=self.motion_type
            )

            for interval in range(self.experiment_instance.codelen):
                if not self.is_running:
                    break
                self.current_interval = interval + 1
                self.window.after(0, self._update_progress_display)

                states = []
                for i in range(len(self.symbols)):
                    if i < len(self.patterns):
                        states.append(self.patterns[i][interval])
                    else:
                        states.append(0)

                states_str = "".join(str(s) for s in states)

                # ВАЖНЫЙ МАРКЕР: НАЧАЛО ИНТЕРВАЛА СТИМУЛЯЦИИ
                self.experiment_instance.send_event_marker(
                    "STIMULUS_INTERVAL_START",
                    interval=interval + 1,
                    cycle=cycle + 1,
                    target_symbol=self.target_symbol,
                    target_index=target_index,
                    states=states_str,
                    target_state=self.patterns[target_index][interval]
                    if target_index != -1
                    else 0,
                    stimulus_type=self.stimulus_type,
                    motion_type=self.motion_type
                )

                # Применяем стимуляцию в зависимости от типа
                for i, label in enumerate(self.labels):
                    if i < len(self.patterns):
                        active = self.patterns[i][interval] == 1
                        
                        if self.stimulus_type == "Мигание":
                            # Только мигание
                            if active:
                                if i == target_index:
                                    label.config(fg=self.theme["symbol_target_1"])
                                else:
                                    label.config(fg=self.theme["symbol_custom_1"])
                            else:
                                if i == target_index:
                                    label.config(fg=self.theme["symbol_target_0"])
                                else:
                                    label.config(fg=self.theme["symbol_custom_0"])
                                    
                        elif self.stimulus_type == "Движение":
                            # Только движение
                            self._apply_motion(label, active, interval)
                            # Сохраняем цветовую схему
                            if i == target_index:
                                label.config(fg=self.theme["symbol_target_1"] if active else self.theme["symbol_target_0"])
                            else:
                                label.config(fg=self.theme["symbol_custom_1"] if active else self.theme["symbol_custom_0"])
                                
                        elif self.stimulus_type == "Комбинированный":
                            # Комбинированный: и мигание, и движение
                            if active:
                                if i == target_index:
                                    label.config(fg=self.theme["symbol_target_1"])
                                else:
                                    label.config(fg=self.theme["symbol_custom_1"])
                            else:
                                if i == target_index:
                                    label.config(fg=self.theme["symbol_target_0"])
                                else:
                                    label.config(fg=self.theme["symbol_custom_0"])
                            self._apply_motion(label, active, interval)

                # Очень короткая задержка для визуального эффекта (20% от интервала)
                time.sleep(self.experiment_instance.base_interval * 0.1)

                # Фаза 2: Основное состояние
                if self.stimulus_type == "Мигание" or self.stimulus_type == "Комбинированный":
                    for i, label in enumerate(self.labels):
                        if i < len(self.patterns):
                            active = self.patterns[i][interval] == 1
                            if active:
                                # Символы с 1 становятся серыми после вспышки
                                if i == target_index:
                                    label.config(fg=self.theme["symbol_target_0"])
                                else:
                                    label.config(fg=self.theme["symbol_custom_0"])

                # Оставшаяся часть интервала
                time.sleep(self.experiment_instance.base_interval * 0.9)

                # Сбрасываем движение в конце интервала
                if self.stimulus_type != "Мигание":
                    for i, label in enumerate(self.labels):
                        if i < len(self.patterns):
                            active = self.patterns[i][interval] == 1
                            if active:
                                self._apply_motion(label, False, interval)

            # ВАЖНЫЙ МАРКЕР: КОНЕЦ ЦИКЛА
            self.experiment_instance.send_event_marker(
                "CYCLE_END",
                cycle=cycle + 1,
                target_symbol=self.target_symbol,
                total_intervals=self.experiment_instance.codelen,
                stimulus_type=self.stimulus_type,
                motion_type=self.motion_type
            )

        self.current_interval = 0
        if self.is_running:
            self.window.after(0, self._finish_symbol)

    def _finish_symbol(self):
        """Завершает ввод текущего символа"""
        self.is_running = False
        self.current_cycle = 0
        self.current_interval = 0

        self._update_progress_display()

        # Сбрасываем все свойства символов
        for label in self.labels:
            # Сбрасываем движение и возвращаем в исходное положение
            props = self.symbol_properties[label]
            label.place(x=props['base_x'], y=props['base_y'], anchor="center")
            label.config(
                font=("Segoe UI", props['font_size'], "bold"),
                fg=self.theme["text_tertiary"],
                bg=self.theme["bg_primary"]
            )

        self.output_text += self.target_symbol

        # ВАЖНЫЙ МАРКЕР: СИМВОЛ ЗАВЕРШЕН
        self.experiment_instance.send_event_marker(
            "SYMBOL_FINISHED",
            symbol=self.target_symbol,
            output_text=self.output_text,
            index=self.current_target_index,
            stimulus_type=self.stimulus_type,
            motion_type=self.motion_type
        )
        
        # Обновляем индикатор статуса
        self.status_light.itemconfig(self.status_indicator, fill=self.theme["accent_success"])
        self.status_label.config(
            text=f"Символ '{self.target_symbol}' добавлен",
            fg=self.theme["accent_success"]
        )

        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.output_text)
        self.text_display.see(tk.END)

        self.current_target_index += 1

        self.window.after(1000, self._show_next_target)

    def _finish_experiment(self):
        """Завершает эксперимент"""
        # ВАЖНЫЙ МАРКЕР: ЭКСПЕРИМЕНТ ЗАВЕРШЕН
        self.experiment_instance.send_event_marker(
            "EXPERIMENT_END",
            final_text=self.output_text,
            total_symbols=len(self.target_symbols),
            stimulus_type=self.stimulus_type,
            motion_type=self.motion_type
        )
        
        # Обновляем индикатор статуса
        self.status_light.itemconfig(self.status_indicator, fill=self.theme["accent_success"])
        self.status_label.config(text="Эксперимент завершен!", fg=self.theme["accent_success"])
        self.progress_label.config(text="Завершено", fg=self.theme["accent_success"])

        # Стилизованное сообщение о завершении
        result_window = tk.Toplevel(self.window)
        result_window.title("Эксперимент завершен")
        result_window.configure(bg=self.theme["bg_primary"])
        result_window.geometry("600x450")
        result_window.resizable(False, False)
        
        # Центрируем окно
        x = (self.screen_width - 600) // 2
        y = (self.screen_height - 450) // 2
        result_window.geometry(f"600x450+{x}+{y}")
        
        # Содержимое
        canvas = tk.Canvas(result_window, bg=self.theme["bg_primary"], highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Иконка успеха
        tk.Label(
            canvas,
            text="✓",
            font=("Segoe UI", 72),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_success"],
        ).place(relx=0.5, rely=0.25, anchor="center")
        
        # Заголовок
        tk.Label(
            canvas,
            text="ЭКСПЕРИМЕНТ ЗАВЕРШЁН",
            font=("Segoe UI", 24, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
        ).place(relx=0.5, rely=0.4, anchor="center")
        
        # Результат
        tk.Label(
            canvas,
            text=f"Введённый текст: {self.output_text}",
            font=("Consolas", 14),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_primary"],
        ).place(relx=0.5, rely=0.5, anchor="center")
        
        # Параметры
        params_text = f"Тип стимула: {self.stimulus_type}\n"
        if self.stimulus_type != "Мигание":
            params_text += f"Тип движения: {self.motion_type}\n"
        params_text += f"Параметры: {self.experiment_instance.num_cycles} циклов × {self.experiment_instance.cycle_duration} сек"
        
        tk.Label(
            canvas,
            text=params_text,
            font=("Segoe UI", 12),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"],
            justify="center"
        ).place(relx=0.5, rely=0.65, anchor="center")
        
        # Кнопка закрытия
        close_button = tk.Button(
            canvas,
            text="ЗАКРЫТЬ",
            font=("Segoe UI", 12, "bold"),
            command=lambda: [result_window.destroy(), self.window.quit()],
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["accent_warning"],
            activeforeground=self.theme["text_primary"],
            relief="flat",
            width=15,
            height=1,
            bd=2,
            highlightthickness=2,
            highlightbackground=self.theme["accent_warning"],
            highlightcolor=self.theme["accent_warning"],
            cursor="hand2"
        )
        close_button.place(relx=0.5, rely=0.85, anchor="center")
        
        result_window.focus_force()