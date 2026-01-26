import sys
import tkinter as tk
from tkinter import font


class WelcomeWindow:
    def __init__(self, parent_callback, root_window, theme_manager):
        self.callback = parent_callback
        self.root = root_window
        self.theme_manager = theme_manager
        self.theme = theme_manager.get_theme()

        self.window = tk.Toplevel()
        self.window.title("BCI Speller - Добро пожаловать")
        
        # Используем тему
        self.window.attributes("-fullscreen", True)
        
        # Холст для эффектов
        self.canvas = tk.Canvas(self.window, bg=self.theme["bg_primary"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Привязка клавиш
        self.window.bind("<Escape>", self._exit_program)
        self.window.protocol("WM_DELETE_WINDOW", self._exit_program)
        self.window.bind("<space>", lambda e: self._on_continue())

        # Создаем все виджеты
        self._create_ui()

    def _create_ui(self):
        """Создает весь интерфейс окна"""
        # Очищаем холст
        self.canvas.delete("all")
        
        # Устанавливаем фон окна
        self.window.configure(bg=self.theme["bg_primary"])
        self.canvas.configure(bg=self.theme["bg_primary"])
        
        # Создаем фоновую сетку
        self._create_grid_background()
        
        # Создаем кнопки управления (тема и выход)
        self._create_control_buttons()
        
        # Создаем основной контент
        self._create_main_content()

    def _create_grid_background(self):
        """Создает анимированную сетку на фоне"""
        width = self.window.winfo_screenwidth()
        height = self.window.winfo_screenheight()
        
        # Вертикальные линии
        for x in range(0, width, 50):
            self.canvas.create_line(x, 0, x, height, fill=self.theme["grid_lines"], width=1, dash=(2, 4))
        
        # Горизонтальные линии
        for y in range(0, height, 50):
            self.canvas.create_line(0, y, width, y, fill=self.theme["grid_lines"], width=1, dash=(2, 4))
        
        # Точки на пересечениях
        for x in range(25, width, 50):
            for y in range(25, height, 50):
                self.canvas.create_oval(x-1, y-1, x+1, y+1, fill=self.theme["bg_tertiary"], outline="")

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
        main_container = tk.Frame(self.canvas, bg=self.theme["bg_primary"])
        main_container.place(relx=0.5, rely=0.5, anchor="center", width=800, height=600)
        
        # Логотип/иконка
        logo_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        logo_frame.pack(pady=(0, 30))
        
        tk.Label(
            logo_frame,
            text="🧠",
            font=("Segoe UI", 72),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_primary"],
        ).pack()
        
        # Заголовок с градиентным эффектом
        title_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        title_frame.pack(pady=(0, 20))
        
        tk.Label(
            title_frame,
            text="BCI SPELLER",
            font=("Segoe UI", 42, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
        ).pack()
        
        # Подзаголовок с анимацией точки
        subtitle_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        subtitle_frame.pack(pady=(0, 40))
        
        tk.Label(
            subtitle_frame,
            text="ИНТЕРФЕЙС МОЗГ-КОМПЬЮТЕР",
            font=("Segoe UI", 18),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_primary"],
        ).pack()
        
        # Описание
        desc_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        desc_frame.pack(pady=(0, 60))
        
        tk.Label(
            desc_frame,
            text="Добро пожаловать в эксперимент по нейроуправлению\n"
                 "Вам предстоит вводить текст силой мысли",
            font=("Segoe UI", 14),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"],
            justify="center",
        ).pack()
        
        # Индикатор прогресса
        progress_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        progress_frame.pack(pady=(0, 30))
        
        self.progress_canvas = tk.Canvas(progress_frame, width=200, height=4, 
                                        bg=self.theme["bg_primary"], highlightthickness=0)
        self.progress_canvas.pack()
        self.progress_canvas.create_rectangle(0, 0, 200, 4, 
                                            fill=self.theme["bg_tertiary"], outline="")
        self.progress_canvas.create_rectangle(0, 0, 50, 4, 
                                            fill=self.theme["accent_primary"], outline="")
        
        # Стилизованная кнопка продолжения
        button_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        button_frame.pack()
        
        self.continue_button = tk.Button(
            button_frame,
            text="НАЧАТЬ ЭКСПЕРИМЕНТ",
            font=("Segoe UI", 14, "bold"),
            command=self._on_continue,
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["accent_primary"],
            activeforeground=self.theme["text_primary"],
            relief="flat",
            width=25,
            height=2,
            bd=2,
            highlightthickness=2,
            highlightbackground=self.theme["accent_primary"],
            highlightcolor=self.theme["accent_primary"],
            cursor="hand2"
        )
        self.continue_button.pack()
        self._create_glow_effect(self.continue_button, self.theme["accent_primary"])
        
        # Подсказка
        hint_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        hint_frame.pack(pady=(20, 0))
        
        tk.Label(
            hint_frame,
            text="Нажмите ПРОБЕЛ для продолжения",
            font=("Segoe UI", 11),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_tertiary"],
        ).pack()

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

    def _on_continue(self):
        self.window.destroy()
        self.callback()