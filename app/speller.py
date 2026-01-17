import logging
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from pylsl import StreamInfo, StreamOutlet

from app.code_generator import CodeGen
from app.get_logger import setup_logger


class SSVEPSpellerExperiment:
    def __init__(self, root):
        self.root = root
        self.logger = logging.getLogger("BCI")
        setup_logger(self.logger, "Experiment")

        # Технологичный дизайн
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#0a0e17")
        
        self.root.bind("<Escape>", self._exit_program)
        self.root.protocol("WM_DELETE_WINDOW", self._exit_program)

        self.codelen = 9
        self.cycle_duration = 0.5
        self.num_cycles = 10
        self.base_interval = self.cycle_duration / 9
        self.transition_duration = self.base_interval * 0.2

        self.is_running = False
        self.current_interval = 0
        self.output_text = ""
        self.flash_thread = None

        self.target_symbol = ""
        self.target_symbols = []
        self.current_target_index = 0

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
        
        # Стиль для виджетов
        self._setup_styles()

    def _setup_styles(self):
        """Настраивает стили для виджетов"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Стиль для кнопок
        style.configure("Tech.TButton",
                       background="#1a2238",
                       foreground="#ffffff",
                       borderwidth=2,
                       focusthickness=3,
                       focuscolor="#3498db",
                       font=("Segoe UI", 10, "bold"))
        
        style.map("Tech.TButton",
                 background=[("active", "#2c3e50")],
                 foreground=[("active", "#ffffff")])

    def _exit_program(self, event=None):
        """Закрытие программы"""
        self.logger.info("Программа завершена пользователем")
        if messagebox.askyesno("Выход", "Вы уверены, что хотите выйти?"):
            self.root.destroy()
            sys.exit(0)

    def send_event_marker(self, event_type, **kwargs):
        """Отправка маркера события через LSL"""
        formatted_kwargs = {}
        for key, value in kwargs.items():
            if key == "states" and isinstance(value, str) and len(value) == 36:
                formatted_kwargs[key] = value
            else:
                formatted_kwargs[key] = value
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
        WelcomeWindow(self._show_instructions, self.root)

    def _show_instructions(self):
        """Показывает окно инструкций"""
        from app.instructions import InstructionWindow
        self.send_event_marker("WINDOW_OPEN_instruct", window="instructions")
        self.logger.info("Показ окна инструкций")
        InstructionWindow(self._show_preparation, self.root)

    def _show_preparation(self):
        """Окно подготовки"""
        self.send_event_marker("WINDOW_OPEN_preparation", window="preparation")
        self.root.withdraw()

        self.prep_window = tk.Toplevel(self.root)
        self.prep_window.title("BCI Speller - Настройки")
        
        # Технологичный дизайн
        self.prep_window.attributes("-fullscreen", True)
        self.prep_window.configure(bg="#0a0e17")
        
        # Холст для фона
        self.prep_canvas = tk.Canvas(self.prep_window, bg="#0a0e17", highlightthickness=0)
        self.prep_canvas.pack(fill=tk.BOTH, expand=True)
        
        self._create_prep_background()

        self.prep_window.bind("<Escape>", lambda e: self._exit_program(e))
        self.prep_window.protocol("WM_DELETE_WINDOW", lambda: self._exit_program())

        # Кнопка выхода
        exit_button = tk.Button(
            self.prep_canvas,
            text="✕",
            font=("Segoe UI", 14, "bold"),
            command=self._exit_program,
            bg="#0a0e17",
            fg="#ffffff",
            activebackground="#f53939",
            activeforeground="#ffffff",
            relief="flat",
            width=3,
            height=1,
            bd=0,
            cursor="hand2"
        )
        exit_button.place(x=20, y=20)
        self._create_glow_effect(exit_button)

        self.logger.info("Ввод данных")

        # Основной контейнер
        main_container = tk.Frame(self.prep_canvas, bg="#0a0e17")
        main_container.place(relx=0.5, rely=0.5, anchor="center", width=800, height=600)

        # Заголовок
        tk.Label(
            main_container,
            text="НАСТРОЙКИ ЭКСПЕРИМЕНТА",
            font=("Segoe UI", 28, "bold"),
            bg="#0a0e17",
            fg="#ffffff",
        ).pack(pady=(0, 40))

        # Форма ввода
        form_frame = tk.Frame(main_container, bg="#0a0e17")
        form_frame.pack(pady=(0, 40))

        # Поле ввода текста
        input_group = tk.Frame(form_frame, bg="#0a0e17")
        input_group.pack(pady=(0, 20), fill=tk.X)
        
        tk.Label(
            input_group,
            text="ТЕКСТ ДЛЯ ВВОДА",
            font=("Segoe UI", 12, "bold"),
            bg="#0a0e17",
            fg="#3498db",
        ).pack(anchor="w", pady=(0, 5))
        
        self.text_entry = tk.Entry(
            input_group,
            font=("Consolas", 14),
            bg="#1a2238",
            fg="#ffffff",
            insertbackground="#3498db",
            relief="flat",
            width=30
        )
        self.text_entry.insert(0, "ПРИВЕТ")
        self.text_entry.pack(fill=tk.X, pady=(0, 10), ipady=8)
        
        # Горизонтальный раздел для параметров
        params_frame = tk.Frame(form_frame, bg="#0a0e17")
        params_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Длительность цикла
        duration_group = tk.Frame(params_frame, bg="#0a0e17")
        duration_group.pack(side=tk.LEFT, padx=(0, 40))
        
        tk.Label(
            duration_group,
            text="ДЛИТЕЛЬНОСТЬ ЦИКЛА",
            font=("Segoe UI", 11, "bold"),
            bg="#0a0e17",
            fg="#95a5a6",
        ).pack(anchor="w", pady=(0, 5))
        
        self.duration_entry = tk.Entry(
            duration_group,
            font=("Consolas", 12),
            bg="#1a2238",
            fg="#ffffff",
            insertbackground="#3498db",
            relief="flat",
            width=12,
            justify="center"
        )
        self.duration_entry.insert(0, "0.5")
        self.duration_entry.pack(ipady=6)
        
        # Количество циклов
        cycles_group = tk.Frame(params_frame, bg="#0a0e17")
        cycles_group.pack(side=tk.LEFT, padx=(0, 40))
        
        tk.Label(
            cycles_group,
            text="ЦИКЛОВ МИГАНИЯ",
            font=("Segoe UI", 11, "bold"),
            bg="#0a0e17",
            fg="#95a5a6",
        ).pack(anchor="w", pady=(0, 5))
        
        self.cycles_entry = tk.Entry(
            cycles_group,
            font=("Consolas", 12),
            bg="#1a2238",
            fg="#ffffff",
            insertbackground="#3498db",
            relief="flat",
            width=12,
            justify="center"
        )
        self.cycles_entry.insert(0, "10")
        self.cycles_entry.pack(ipady=6)
        
        # Длина кода
        codelen_group = tk.Frame(params_frame, bg="#0a0e17")
        codelen_group.pack(side=tk.LEFT)
        
        tk.Label(
            codelen_group,
            text="ДЛИНА КОДА",
            font=("Segoe UI", 11, "bold"),
            bg="#0a0e17",
            fg="#95a5a6",
        ).pack(anchor="w", pady=(0, 5))
        
        self.codelen_entry = tk.Entry(
            codelen_group,
            font=("Consolas", 12),
            bg="#1a2238",
            fg="#ffffff",
            insertbackground="#3498db",
            relief="flat",
            width=12,
            justify="center"
        )
        self.codelen_entry.insert(0, "9")
        self.codelen_entry.pack(ipady=6)

        # Подсказка
        hint_label = tk.Label(
            form_frame,
            text="1 цикл = N (длина кода) интервалов мигания",
            font=("Segoe UI", 10),
            bg="#0a0e17",
            fg="#7f8c8d",
        )
        hint_label.pack(pady=(10, 0))

        # Кнопка запуска
        button_frame = tk.Frame(main_container, bg="#0a0e17")
        button_frame.pack()
        
        start_button = tk.Button(
            button_frame,
            text="🚀 НАЧАТЬ ЭКСПЕРИМЕНТ",
            font=("Segoe UI", 14, "bold"),
            command=self._start_experiment,
            bg="#0a0e17",
            fg="#ffffff",
            activebackground="#2ecc71",
            activeforeground="#ffffff",
            relief="flat",
            width=25,
            height=2,
            bd=2,
            highlightthickness=2,
            highlightbackground="#2ecc71",
            highlightcolor="#2ecc71",
            cursor="hand2"
        )
        start_button.pack()
        self._create_glow_effect(start_button, "#2ecc71")

    def _create_prep_background(self):
        """Создает фоновые эффекты для окна подготовки"""
        width = self.prep_window.winfo_screenwidth()
        height = self.prep_window.winfo_screenheight()
        
        # Сетка
        for x in range(0, width, 60):
            self.prep_canvas.create_line(x, 0, x, height, fill="#1a2238", width=1, dash=(3, 6))
        
        for y in range(0, height, 60):
            self.prep_canvas.create_line(0, y, width, y, fill="#1a2238", width=1, dash=(3, 6))

    def _create_glow_effect(self, widget, color="#3498db"):
        """Создает эффект свечения для виджета"""
        def on_enter(e):
            widget.config(highlightbackground=color, highlightcolor=color)
            
        def on_leave(e):
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            darker = f'#{max(0, r-30):02x}{max(0, g-30):02x}{max(0, b-30):02x}'
            widget.config(highlightbackground=darker, highlightcolor=darker)
            
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _start_experiment(self):
        """Начинает эксперимент"""
        try:
            text = self.text_entry.get().upper()
            if not text:
                self.logger.critical("Текст не введен")
                messagebox.showerror("Ошибка", "Введите текст")
                return

            self.logger.info("Запуск эксперимента!")
            self.codelen = int(self.codelen_entry.get())
            self.setup_symbols()

            self.target_symbols = list(text)
            self.current_target_index = 0

            duration = float(self.duration_entry.get())
            if duration <= 0:
                self.logger.error("Ошибка длительности")
                raise ValueError("Длительность должна быть > 0")

            self.cycle_duration = duration
            self.base_interval = self.cycle_duration / self.codelen
            self.transition_duration = self.base_interval * 0.2

            cycles = int(self.cycles_entry.get())
            if cycles <= 0:
                self.logger.error("Ошибка количества циклов")
                raise ValueError("Количество циклов должно быть > 0")

            self.num_cycles = cycles

            self.send_event_marker(
                "EXPERIMENT_START",
                text=self.text_entry.get(),
                codelen=self.codelen,
                duration=self.cycle_duration,
                cycles=self.num_cycles,
            )

            self.prep_window.destroy()
            self._setup_main_ui()
            self._show_next_target()

        except ValueError as e:
            self.logger.critical("Некорректные данные")
            messagebox.showerror("Ошибка", f"Некорректные данные: {e}")

    def _setup_main_ui(self):
        """Настраивает главное окно (скрытое)"""
        self.root.deiconify()
        self.root.title("BCI Speller - Эксперимент")
        
        # Холст для фона
        self.main_canvas = tk.Canvas(self.root, bg="#0a0e17", highlightthickness=0)
        self.main_canvas.pack(fill=tk.BOTH, expand=True)
        
        self._create_main_background()

        # Кнопка выхода
        exit_button = tk.Button(
            self.main_canvas,
            text="✕",
            font=("Segoe UI", 14, "bold"),
            command=self._exit_program,
            bg="#0a0e17",
            fg="#ffffff",
            activebackground="#f53939",
            activeforeground="#ffffff",
            relief="flat",
            width=3,
            height=1,
            bd=0,
            cursor="hand2"
        )
        exit_button.place(x=20, y=20)
        self._create_glow_effect(exit_button)

        # Основной контейнер
        main_container = tk.PanedWindow(
            self.main_canvas,
            orient=tk.VERTICAL,
            bg="#0a0e17",
            sashwidth=5,
            sashrelief="flat",
            sashpad=3,
            opaqueresize=False
        )
        main_container.place(relx=0.5, rely=0.5, anchor="center", 
                           width=self.screen_width*0.9, height=self.screen_height*0.9)

        # Верхняя панель (80%)
        top_frame = tk.Frame(main_container, bg="#0a0e17")
        main_container.add(top_frame, height=int(self.screen_height * 0.9 * 0.8))

        # Панель информации
        info_frame = tk.Frame(top_frame, bg="#0a0e17")
        info_frame.pack(fill=tk.X, padx=20, pady=(20, 10))

        # Индикатор статуса
        status_group = tk.Frame(info_frame, bg="#0a0e17")
        status_group.pack(side=tk.LEFT)

        self.status_light = tk.Canvas(status_group, width=20, height=20, 
                                     bg="#0a0e17", highlightthickness=0)
        self.status_light.pack(side=tk.LEFT, padx=(0, 10))
        self.status_indicator = self.status_light.create_oval(2, 2, 18, 18, 
                                                            fill="#7f8c8d", outline="")

        self.status_label = tk.Label(
            status_group,
            text="Готов к началу эксперимента",
            font=("Segoe UI", 12, "bold"),
            bg="#0a0e17",
            fg="#ffffff",
        )
        self.status_label.pack(side=tk.LEFT)

        # Индикатор мигания
        flash_group = tk.Frame(info_frame, bg="#0a0e17")
        flash_group.pack(side=tk.RIGHT)

        self.flash_indicator = tk.Label(
            flash_group,
            text="○",
            font=("Segoe UI", 16),
            bg="#0a0e17",
            fg="#7f8c8d",
        )
        self.flash_indicator.pack(side=tk.LEFT, padx=(0, 5))

        tk.Label(
            flash_group,
            text="МИГАНИЕ",
            font=("Segoe UI", 11),
            bg="#0a0e17",
            fg="#95a5a6",
        ).pack(side=tk.LEFT)

        # Панель прогресса
        progress_frame = tk.Frame(top_frame, bg="#0a0e17")
        progress_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.current_symbol_label = tk.Label(
            progress_frame,
            text="Ожидание целевого символа...",
            font=("Segoe UI", 14),
            bg="#0a0e17",
            fg="#3498db",
        )
        self.current_symbol_label.pack(anchor="w", pady=(0, 5))

        # Прогресс-бар эксперимента
        progress_bar_frame = tk.Frame(progress_frame, bg="#0a0e17")
        progress_bar_frame.pack(fill=tk.X, pady=(0, 10))

        self.experiment_progress = tk.Label(
            progress_bar_frame,
            text="Прогресс: 0/0",
            font=("Segoe UI", 11),
            bg="#0a0e17",
            fg="#95a5a6",
        )
        self.experiment_progress.pack(side=tk.LEFT)

        # Сетка символов - ТОЛЬКО СИМВОЛЫ
        grid_container = tk.Frame(top_frame, bg="#0a0e17")
        grid_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self._create_symbol_grid(grid_container)

        # Нижняя панель (20%)
        bottom_frame = tk.Frame(main_container, bg="#0a0e17")
        main_container.add(bottom_frame, height=int(self.screen_height * 0.9 * 0.2))

        bottom_paned = tk.PanedWindow(
            bottom_frame,
            orient=tk.HORIZONTAL,
            bg="#0a0e17",
            sashwidth=3,
            sashrelief="flat",
            opaqueresize=False
        )
        bottom_paned.pack(fill=tk.BOTH, expand=True)

        # Левая панель (параметры)
        left_bottom_frame = tk.Frame(bottom_paned, bg="#1a2238")
        bottom_paned.add(left_bottom_frame, width=int(self.screen_width * 0.9 * 0.4))

        # Заголовок параметров
        tk.Label(
            left_bottom_frame,
            text="ПАРАМЕТРЫ ЭКСПЕРИМЕНТА",
            font=("Segoe UI", 12, "bold"),
            bg="#1a2238",
            fg="#3498db",
        ).pack(anchor="w", padx=20, pady=(15, 10))

        # Настройки
        self.settings_label = tk.Label(
            left_bottom_frame,
            text=f"Длительность цикла: {self.cycle_duration} сек | Циклов: {self.num_cycles}",
            font=("Segoe UI", 10),
            bg="#1a2238",
            fg="#95a5a6",
        )
        self.settings_label.pack(anchor="w", padx=20, pady=(0, 5))

        # Прогресс мигания
        self.progress_label = tk.Label(
            left_bottom_frame,
            text=f"Цикл: 0/{self.num_cycles} | Интервал: 0/{self.codelen}",
            font=("Segoe UI", 11, "bold"),
            bg="#1a2238",
            fg="#2ecc71",
        )
        self.progress_label.pack(anchor="w", padx=20, pady=(5, 5))

        # Правая панель (результаты)
        right_bottom_frame = tk.Frame(bottom_paned, bg="#1a2238")
        bottom_paned.add(right_bottom_frame, width=int(self.screen_width * 0.9 * 0.6))

        # Заголовок результатов
        tk.Label(
            right_bottom_frame,
            text="РЕЗУЛЬТАТ ВВОДА",
            font=("Segoe UI", 12, "bold"),
            bg="#1a2238",
            fg="#3498db",
        ).pack(anchor="w", padx=20, pady=(15, 10))

        # Текстовое поле для результатов
        text_container = tk.Frame(right_bottom_frame, bg="#1a2238")
        text_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))

        # Кастомный скроллбар
        style = ttk.Style()
        style.configure("Results.Vertical.TScrollbar", 
                       background="#2c3e50",
                       troughcolor="#1a2238",
                       bordercolor="#1a2238",
                       arrowcolor="#3498db",
                       relief="flat")

        self.text_display = tk.Text(
            text_container,
            font=("Consolas", 14),
            height=3,
            wrap=tk.WORD,
            bg="#0a0e17",
            fg="#ffffff",
            insertbackground="#3498db",
            selectbackground="#3498db",
            selectforeground="#ffffff",
            relief="flat",
            bd=2,
            highlightthickness=1,
            highlightbackground="#2c3e50",
            highlightcolor="#2c3e50",
            padx=15,
            pady=10
        )

        scrollbar = ttk.Scrollbar(text_container, orient="vertical", 
                                 command=self.text_display.yview,
                                 style="Results.Vertical.TScrollbar")
        self.text_display.configure(yscrollcommand=scrollbar.set)

        self.text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_main_background(self):
        """Создает фоновые эффекты для главного окна"""
        width = self.screen_width
        height = self.screen_height
        
        # Динамические линии
        for i in range(10):
            x1 = width // 10 * i
            y1 = 0
            x2 = width // 10 * (10 - i)
            y2 = height
            self.main_canvas.create_line(x1, y1, x2, y2, fill="#1a2238", width=1)

    def _create_symbol_grid(self, parent):
        """Создает сетку СИМВОЛОВ (без плиток/карточек)"""
        self.grid_frame = tk.Frame(parent, bg="#0a0e17")
        self.grid_frame.pack(fill=tk.BOTH, expand=True)

        rows = 3
        cols = 12

        for i in range(rows):
            self.grid_frame.grid_rowconfigure(i, weight=1, uniform="row")
        for j in range(cols):
            self.grid_frame.grid_columnconfigure(j, weight=1, uniform="col")

        self.labels = []
        
        for i, symbol in enumerate(self.symbols):
            row = i // cols
            col = i % cols

            # СОЗДАЕМ ТОЛЬКО ЛЕЙБЛЫ С СИМВОЛАМИ (без фреймов)
            label = tk.Label(
                self.grid_frame,
                text=symbol,
                font=("Segoe UI", 22, "bold"),  # Увеличим размер шрифта
                bg="#0a0e17",  # Прозрачный фон
                fg="#7f8c8d",  # Серый цвет по умолчанию
                width=3,
                height=1,
            )
            label.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.labels.append(label)
            
            # Эффект при наведении (опционально)
            def on_enter(e, l=label, s=symbol):
                if self.target_symbol == s and self.is_running:
                    l.config(fg="#ffffff")  # Белый при мигании цели
                elif not self.is_running:
                    l.config(fg="#95a5a6")  # Светло-серый при наведении
                    
            def on_leave(e, l=label, s=symbol):
                if self.target_symbol == s and self.is_running:
                    l.config(fg="#f53939")  # Красный для целевого символа
                elif not self.is_running:
                    l.config(fg="#7f8c8d")  # Темно-серый по умолчанию
                else:
                    l.config(fg="#7f8c8d")
            
            label.bind("<Enter>", on_enter)
            label.bind("<Leave>", on_leave)

    def setup_symbols(self):
        """Настройка символов"""
        CG = CodeGen(self.codelen)
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

    def _show_next_target(self):
        """Показывает окно с целевым символом для ввода"""
        if self.current_target_index < len(self.target_symbols):
            self.target_symbol = self.target_symbols[self.current_target_index]

            self.send_event_marker(
                "Смотрите на целевой символ и нажмите ПРОБЕЛ",
                symbol=self.target_symbol,
                index=self.current_target_index,
                total=len(self.target_symbols),
            )

            # Обновляем индикатор статуса
            self.status_light.itemconfig(self.status_indicator, fill="#f39c12")
            self.status_label.config(
                text=f"Целевой символ: '{self.target_symbol}' ({self.current_target_index + 1}/{len(self.target_symbols)})",
                fg="#f39c12"
            )
            
            self.current_symbol_label.config(
                text=f"СИМВОЛ: '{self.target_symbol}' • ПОЗИЦИЯ: {self.current_target_index + 1}/{len(self.target_symbols)}"
            )
            
            self.experiment_progress.config(
                text=f"Прогресс: {self.current_target_index}/{len(self.target_symbols)}"
            )

            from app.target_window import TargetWindow
            target_win = TargetWindow(self.root, self.target_symbol, self._on_target_confirmed)

        else:
            self._finish_experiment()

    def _on_target_confirmed(self):
        """Вызывается после подтверждения целевого символа"""
        self.send_event_marker(
            "TARGET_CONFIRMED",
            symbol=self.target_symbol,
            index=self.current_target_index,
        )
        self.root.deiconify()

        # Обновляем индикатор статуса
        self.status_light.itemconfig(self.status_indicator, fill="#f53939")
        self.status_label.config(
            text="Мигание... Смотрите на целевой символ",
            fg="#f53939"
        )

        self._start_flashing()

    def _start_flashing(self):
        """Запускает мигание для текущего символа"""
        if not self.is_running:
            self.send_event_marker(
                "FLASHING_START",
                symbol=self.target_symbol,
                duration=self.cycle_duration,
                cycles=self.num_cycles,
                codelen=self.codelen,
            )
            self.is_running = True
            self.current_interval = 0
            self.current_cycle = 0

            self._update_progress_display()
            self._highlight_target_symbol()

            self.flash_thread = threading.Thread(
                target=self._flash_sequence, daemon=True
            )
            self.flash_thread.start()

    def _highlight_target_symbol(self):
        """Подсвечивает целевой символ в сетке (ТОЛЬКО ЦВЕТ ТЕКСТА)"""
        target_index = self.symbols.index(self.target_symbol) if self.target_symbol in self.symbols else -1
        if target_index != -1:
            for i, label in enumerate(self.labels):
                if i == target_index:
                    label.config(fg="#f53939")  # Красный для целевого символа
                else:
                    label.config(fg="#7f8c8d")  # Серый для остальных

    def _update_progress_display(self):
        """Обновляет отображение прогресса мигания"""
        self.progress_label.config(
            text=f"Цикл: {self.current_cycle}/{self.num_cycles} | Интервал: {self.current_interval}/{self.codelen}"
        )
        
        # Обновляем индикатор мигания
        if self.is_running:
            self.flash_indicator.config(text="●", fg="#f53939")
        else:
            self.flash_indicator.config(text="○", fg="#7f8c8d")

    def _flash_sequence(self):
        """Выполняет последовательность мигания с маркерами - ТОЛЬКО ЦВЕТ ТЕКСТА"""
        for cycle in range(self.num_cycles):
            if not self.is_running:
                break
            self.current_cycle = cycle + 1
            self.root.after(0, self._update_progress_display)

            target_index = (
                self.symbols.index(self.target_symbol)
                if self.target_symbol in self.symbols
                else -1
            )

            self.send_event_marker(
                "CYCLE_START",
                cycle=cycle + 1,
                total_cycles=self.num_cycles,
                target_symbol=self.target_symbol,
                target_index=target_index,
            )

            for interval in range(self.codelen):
                if not self.is_running:
                    break
                self.current_interval = interval + 1
                self.root.after(0, self._update_progress_display)

                states = []
                for i in range(len(self.symbols)):
                    if i < len(self.patterns):
                        states.append(self.patterns[i][interval])
                    else:
                        states.append(0)

                states_str = "".join(str(s) for s in states)

                self.send_event_marker(
                    "FLASH_INTERVAL_START",
                    interval=interval + 1,
                    cycle=cycle + 1,
                    target_symbol=self.target_symbol,
                    target_index=target_index,
                    states=states_str,
                    target_state=self.patterns[target_index][interval]
                    if target_index != -1
                    else 0,
                )

                # Фаза 1: Основное состояние - МЕНЯЕМ ТОЛЬКО ЦВЕТ ТЕКСТА
                for i, label in enumerate(self.labels):
                    if i < len(self.patterns):
                        if self.patterns[i][interval] == 1:
                            # Активное мигание - яркий цвет
                            if i == target_index:
                                label.config(fg="#f53939")  # Белый для целевого
                            else:
                                label.config(fg="#ffffff")  # Белый для остальных
                        else:
                            # Неактивное мигание
                            if i == target_index:
                                label.config(fg="#37130f")  # Красный для целевого
                            else:
                                label.config(fg="#7f8c8d")  # Серый для остальных

                time.sleep(self.base_interval * 0.9)

                # Фаза 2: Кратковременное отключение - МЕНЯЕМ ТОЛЬКО ЦВЕТ ТЕКСТА
                if self.is_running:
                    for i, label in enumerate(self.labels):
                        if i == target_index:
                            label.config(fg="#37130f")  # Красный для целевого
                        else:
                            label.config(fg="#7f8c8d")  # Серый для остальных
                    time.sleep(self.base_interval * 0.1)

            self.send_event_marker(
                "CYCLE_END",
                cycle=cycle + 1,
                target_symbol=self.target_symbol,
                total_intervals=self.codelen,
            )

        self.current_interval = 0
        if self.is_running:
            self.root.after(0, self._finish_symbol)

    def _finish_symbol(self):
        """Завершает ввод текущего символа"""
        self.is_running = False
        self.current_cycle = 0
        self.current_interval = 0

        self._update_progress_display()

        # Сбрасываем цвета всех символов (ТОЛЬКО ЦВЕТ ТЕКСТА)
        for label in self.labels:
            label.config(fg="#7f8c8d")  # Все символы серые

        self.output_text += self.target_symbol

        self.send_event_marker(
            "SYMBOL_FINISHED",
            symbol=self.target_symbol,
            output_text=self.output_text,
            index=self.current_target_index,
        )
        
        # Обновляем индикатор статуса
        self.status_light.itemconfig(self.status_indicator, fill="#2ecc71")
        self.status_label.config(
            text=f"Символ '{self.target_symbol}' добавлен",
            fg="#2ecc71"
        )

        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.output_text)
        self.text_display.see(tk.END)

        self.current_target_index += 1

        self.root.after(1000, self._show_next_target)

    def _finish_experiment(self):
        """Завершает эксперимент"""
        self.send_event_marker(
            "EXPERIMENT_END",
            final_text=self.output_text,
            total_symbols=len(self.target_symbols),
        )
        
        # Обновляем индикатор статуса
        self.status_light.itemconfig(self.status_indicator, fill="#2ecc71")
        self.status_label.config(text="Эксперимент завершен!", fg="#2ecc71")
        self.progress_label.config(text="Завершено", fg="#2ecc71")

        # Стилизованное сообщение о завершении
        from tkinter import Toplevel
        result_window = Toplevel(self.root)
        result_window.title("Эксперимент завершен")
        result_window.configure(bg="#0a0e17")
        result_window.geometry("600x400")
        result_window.resizable(False, False)
        
        # Центрируем окно
        x = (self.screen_width - 600) // 2
        y = (self.screen_height - 400) // 2
        result_window.geometry(f"600x400+{x}+{y}")
        
        # Содержимое
        canvas = tk.Canvas(result_window, bg="#0a0e17", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Иконка успеха
        tk.Label(
            canvas,
            text="✓",
            font=("Segoe UI", 72),
            bg="#0a0e17",
            fg="#2ecc71",
        ).place(relx=0.5, rely=0.3, anchor="center")
        
        # Заголовок
        tk.Label(
            canvas,
            text="ЭКСПЕРИМЕНТ ЗАВЕРШЁН",
            font=("Segoe UI", 24, "bold"),
            bg="#0a0e17",
            fg="#ffffff",
        ).place(relx=0.5, rely=0.5, anchor="center")
        
        # Результат
        tk.Label(
            canvas,
            text=f"Введённый текст: {self.output_text}",
            font=("Consolas", 14),
            bg="#0a0e17",
            fg="#3498db",
        ).place(relx=0.5, rely=0.6, anchor="center")
        
        # Параметры
        tk.Label(
            canvas,
            text=f"Параметры: {self.num_cycles} циклов × {self.cycle_duration} сек",
            font=("Segoe UI", 12),
            bg="#0a0e17",
            fg="#95a5a6",
        ).place(relx=0.5, rely=0.7, anchor="center")
        
        # Кнопка закрытия
        close_button = tk.Button(
            canvas,
            text="ЗАКРЫТЬ",
            font=("Segoe UI", 12, "bold"),
            command=lambda: [result_window.destroy(), self.root.quit()],
            bg="#0a0e17",
            fg="#ffffff",
            activebackground="#f53939",
            activeforeground="#ffffff",
            relief="flat",
            width=15,
            height=1,
            bd=2,
            highlightthickness=2,
            highlightbackground="#f53939",
            highlightcolor="#f53939",
            cursor="hand2"
        )
        close_button.place(relx=0.5, rely=0.85, anchor="center")
        
        result_window.focus_force()