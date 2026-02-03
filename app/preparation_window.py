# [file name]: preparation_window.py
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from monitor_config import setup_window_on_target_monitor

class PreparationWindow:
    """Окно подготовки к эксперименту"""
    
    def __init__(self, parent_callback, root_window, theme_manager, experiment_instance):
        self.callback = parent_callback
        self.root = root_window
        self.theme_manager = theme_manager
        self.experiment_instance = experiment_instance
        self.theme = theme_manager.get_theme()

        self.window = tk.Toplevel()
        self.window.title("BCI Speller - Настройки")
        
        # Настраиваем окно на целевом мониторе
        if not setup_window_on_target_monitor(self.window):
            # Если не удалось, используем обычный fullscreen
            self.window.attributes("-fullscreen", True)
        
        # Холст для эффектов
        self.canvas = tk.Canvas(self.window, bg=self.theme["bg_primary"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Привязка клавиш
        self.window.bind("<Escape>", self._exit_program)
        self.window.protocol("WM_DELETE_WINDOW", self._exit_program)

        # Создаем все виджеты
        self._create_ui()

    def _create_ui(self):
        """Создает весь интерфейс окна"""
        # Очищаем холст
        self.canvas.delete("all")
        
        # Устанавливаем фон окна
        self.window.configure(bg=self.theme["bg_primary"])
        self.canvas.configure(bg=self.theme["bg_primary"])
        
        self.window.update_idletasks()
        # Создаем фоновые эффекты
        self._create_background_effects()
        
        # Создаем кнопки управления (тема и выход)
        self._create_control_buttons()
        
        # Создаем основной контент
        self._create_main_content()

    def _create_background_effects(self):
        """Создает фоновые эффекты для окна подготовки"""
        width = self.window.winfo_screenwidth()
        height = self.window.winfo_screenheight()
        
        # Сетка
        for x in range(0, width, 60):
            self.canvas.create_line(x, 0, x, height, 
                                   fill=self.theme["grid_lines"], width=1, dash=(3, 6))
        
        for y in range(0, height, 60):
            self.canvas.create_line(0, y, width, y, 
                                   fill=self.theme["grid_lines"], width=1, dash=(3, 6))

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
        self.theme_button.place(x=self.window.winfo_screenwidth() - 60, y=20)
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
        """Создает основной контент окна"""
        # Основной контейнер
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        main_container = tk.Frame(self.canvas, bg=self.theme["bg_primary"])
        main_container.place(relx=0.5, rely=0.5, anchor="center", width=screen_width//2, height=int(screen_height*0.8))

        # Заголовок
        tk.Label(
            main_container,
            text="НАСТРОЙКИ ЭКСПЕРИМЕНТА",
            font=("Segoe UI", 28, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
        ).pack(pady=(0, 30))

        # Форма ввода
        form_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        form_frame.pack(pady=(0, 40), fill=tk.BOTH, expand=True)

        # Поле ввода текста
        input_group = tk.Frame(form_frame, bg=self.theme["bg_primary"])
        input_group.pack(pady=(0, 20), fill=tk.X)
        
        tk.Label(
            input_group,
            text="ТЕКСТ ДЛЯ ВВОДА",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_primary"],
        ).pack(anchor="w", pady=(0, 5))
        
        self.text_entry = tk.Entry(
            input_group,
            font=("Consolas", 14),
            bg=self.theme["entry_bg"],
            fg=self.theme["entry_fg"],
            insertbackground=self.theme["accent_primary"],
            relief="flat",
            width=30
        )
        self.text_entry.insert(0, "ПРИВЕТ")
        self.text_entry.pack(fill=tk.X, pady=(0, 10), ipady=8)

        # Тип стимула
        stimulus_group = tk.Frame(form_frame, bg=self.theme["bg_primary"])
        stimulus_group.pack(pady=(0, 20), fill=tk.X)
        
        tk.Label(
            stimulus_group,
            text="ТИП СТИМУЛА",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_primary"],
        ).pack(anchor="w", pady=(0, 5))
        
        # Переменная для типа стимула
        self.stimulus_type = tk.StringVar(value="Мигание")
        
        # Фрейм для радиокнопок
        radio_frame = tk.Frame(stimulus_group, bg=self.theme["bg_primary"])
        radio_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Радиокнопки для выбора типа стимула
        rb1 = tk.Radiobutton(
            radio_frame,
            text="Только мигание (стандартный SSVEP)",
            variable=self.stimulus_type,
            value="Мигание",
            font=("Segoe UI", 11),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_tertiary"],
            activebackground=self.theme["bg_primary"],
            activeforeground=self.theme["accent_primary"],
            cursor="hand2"
        )
        rb1.pack(anchor="w", pady=(5, 0))
        
        rb2 = tk.Radiobutton(
            radio_frame,
            text="Только движение буквы",
            variable=self.stimulus_type,
            value="Движение",
            font=("Segoe UI", 11),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_tertiary"],
            activebackground=self.theme["bg_primary"],
            activeforeground=self.theme["accent_primary"],
            cursor="hand2"
        )
        rb2.pack(anchor="w", pady=(5, 0))
        
        rb3 = tk.Radiobutton(
            radio_frame,
            text="Мигание + движение (комбинированный)",
            variable=self.stimulus_type,
            value="Комбинированный",
            font=("Segoe UI", 11),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_tertiary"],
            activebackground=self.theme["bg_primary"],
            activeforeground=self.theme["accent_primary"],
            cursor="hand2"
        )
        rb3.pack(anchor="w", pady=(5, 0))

        # Тип движения
        motion_group = tk.Frame(form_frame, bg=self.theme["bg_primary"])
        motion_group.pack(pady=(0, 20), fill=tk.X)
        
        tk.Label(
            motion_group,
            text="ТИП ДВИЖЕНИЯ",
            font=("Segoe UI", 12, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_primary"],
        ).pack(anchor="w", pady=(0, 5))
        
        # Переменная для типа движения
        self.motion_type = tk.StringVar(value="Дрожание")
        
        # Выпадающий список для типа движения
        motion_options = ["Дрожание", "Колебание размера", "Направленное движение"]
        self.motion_combo = ttk.Combobox(
            motion_group,
            textvariable=self.motion_type,
            values=motion_options,
            font=("Segoe UI", 11),
            state="readonly",
            width=30
        )
        self.motion_combo.pack(anchor="w", pady=(0, 5), ipady=6)
        
        # Описание выбранного типа движения
        self.motion_desc = tk.Label(
            motion_group,
            text="Дрожание: буква слегка вибрирует на месте (1-2 пикселя)",
            font=("Segoe UI", 10),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"],
            wraplength=500,
            justify="left"
        )
        self.motion_desc.pack(anchor="w", pady=(5, 0))
        
        # Привязываем изменение выбора типа движения
        self.motion_combo.bind("<<ComboboxSelected>>", self._update_motion_desc)
        
        # Горизонтальный раздел для параметров
        params_frame = tk.Frame(form_frame, bg=self.theme["bg_primary"])
        params_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Длительность цикла
        duration_group = tk.Frame(params_frame, bg=self.theme["bg_primary"])
        duration_group.pack(side=tk.LEFT, padx=(0, 40))
        
        tk.Label(
            duration_group,
            text="ДЛИТЕЛЬНОСТЬ ЦИКЛА",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"],
        ).pack(anchor="w", pady=(0, 5))
        
        self.duration_entry = tk.Entry(
            duration_group,
            font=("Consolas", 12),
            bg=self.theme["entry_bg"],
            fg=self.theme["entry_fg"],
            insertbackground=self.theme["accent_primary"],
            relief="flat",
            width=12,
            justify="center"
        )
        self.duration_entry.insert(0, "9")
        self.duration_entry.pack(ipady=6)
        
        # Количество циклов
        cycles_group = tk.Frame(params_frame, bg=self.theme["bg_primary"])
        cycles_group.pack(side=tk.LEFT, padx=(0, 40))
        
        tk.Label(
            cycles_group,
            text="ЦИКЛОВ МИГАНИЯ",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"],
        ).pack(anchor="w", pady=(0, 5))
        
        self.cycles_entry = tk.Entry(
            cycles_group,
            font=("Consolas", 12),
            bg=self.theme["entry_bg"],
            fg=self.theme["entry_fg"],
            insertbackground=self.theme["accent_primary"],
            relief="flat",
            width=12,
            justify="center"
        )
        self.cycles_entry.insert(0, "10")
        self.cycles_entry.pack(ipady=6)
        
        # Длина кода
        codelen_group = tk.Frame(params_frame, bg=self.theme["bg_primary"])
        codelen_group.pack(side=tk.LEFT)
        
        tk.Label(
            codelen_group,
            text="ДЛИНА КОДА",
            font=("Segoe UI", 11, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"],
        ).pack(anchor="w", pady=(0, 5))
        
        self.codelen_entry = tk.Entry(
            codelen_group,
            font=("Consolas", 12),
            bg=self.theme["entry_bg"],
            fg=self.theme["entry_fg"],
            insertbackground=self.theme["accent_primary"],
            relief="flat",
            width=12,
            justify="center"
        )
        self.codelen_entry.insert(0, "9")
        self.codelen_entry.pack(ipady=6)

        # Подсказка
        hint_label = tk.Label(
            form_frame,
            text="1 цикл = N (длина кода) интервалов стимуляции",
            font=("Segoe UI", 10),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_tertiary"],
        )
        hint_label.pack(pady=(10, 0))

        # Кнопка запуска
        button_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        button_frame.pack(pady=(10, 0))
        
        start_button = tk.Button(
            button_frame,
            text="🚀 НАЧАТЬ ЭКСПЕРИМЕНТ",
            font=("Segoe UI", 14, "bold"),
            command=self._start_experiment,
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["accent_success"],
            activeforeground=self.theme["text_primary"],
            relief="flat",
            width=25,
            height=2,
            bd=2,
            highlightthickness=2,
            highlightbackground=self.theme["accent_success"],
            highlightcolor=self.theme["accent_success"],
            cursor="hand2"
        )
        start_button.pack()
        self._create_glow_effect(start_button, self.theme["accent_success"])

    # Обновим метод _update_motion_desc:
    def _update_motion_desc(self, event=None):
        """Обновляет описание выбранного типа движения"""
        motion_type = self.motion_type.get()
        descriptions = {
            "Дрожание": "Дрожание: буква вибрирует случайным образом (1-2 пикселя)",
            "Колебание размера": "Колебание размера: буква плавно увеличивается и уменьшается",
            "Направленное движение": "Направленное движение: буква двигается в 8 направлениях (N, NE, E, SE, S, SW, W, NW) с двумя амплитудами (малая: 2px, большая: 5px)"
        }
        self.motion_desc.config(text=descriptions.get(motion_type, ""))

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
        self.window.destroy()
        self.root.destroy()
        sys.exit(0)

    def _start_experiment(self):
        """Начинает эксперимент с введенными параметрами"""
        text = self.text_entry.get().upper()
        if not text:
            self.experiment_instance.logger.critical("Текст не введен")
            messagebox.showerror("Ошибка", "Введите текст")
            return
            
        self.callback(
            text, 
            self.codelen_entry.get(), 
            self.duration_entry.get(), 
            self.cycles_entry.get(),
            self.stimulus_type.get(),
            self.motion_type.get()
        )