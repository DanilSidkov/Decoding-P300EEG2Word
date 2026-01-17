import sys
import tkinter as tk
from tkinter import font


class WelcomeWindow:
    def __init__(self, parent_callback, root_window):
        self.callback = parent_callback
        self.root = root_window

        self.window = tk.Toplevel()
        self.window.title("BCI Speller - Добро пожаловать")
        
        # Технологичный дизайн
        self.window.attributes("-fullscreen", True)
        self.window.configure(bg="#0a0e17")
        
        # Добавим эффект сетки на фон
        self.canvas = tk.Canvas(self.window, bg="#0a0e17", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self._create_grid_background()
        
        self.window.bind("<Escape>", self._exit_program)
        self.window.protocol("WM_DELETE_WINDOW", self._exit_program)

        # Стилизованная кнопка выхода
        exit_button = tk.Button(
            self.canvas,
            text="✕",
            font=("Segoe UI", 14, "bold"),
            command=self._exit_program,
            bg="#0a0e17",
            fg="#ffffff",
            activebackground="#e74c3c",
            activeforeground="#ffffff",
            relief="flat",
            width=3,
            height=1,
            bd=0,
            cursor="hand2"
        )
        exit_button.place(x=20, y=20)
        self._create_glow_effect(exit_button)

        self._create_content()

    def _create_grid_background(self):
        """Создает анимированную сетку на фоне"""
        width = self.window.winfo_screenwidth()
        height = self.window.winfo_screenheight()
        
        # Вертикальные линии
        for x in range(0, width, 50):
            self.canvas.create_line(x, 0, x, height, fill="#1a2238", width=1, dash=(2, 4))
        
        # Горизонтальные линии
        for y in range(0, height, 50):
            self.canvas.create_line(0, y, width, y, fill="#1a2238", width=1, dash=(2, 4))
        
        # Точки на пересечениях
        for x in range(25, width, 50):
            for y in range(25, height, 50):
                self.canvas.create_oval(x-1, y-1, x+1, y+1, fill="#2c3e50", outline="")

    def _create_content(self):
        # Основной контейнер
        main_container = tk.Frame(self.canvas, bg="#0a0e17")
        main_container.place(relx=0.5, rely=0.5, anchor="center", width=800, height=600)
        
        # Логотип/иконка
        logo_frame = tk.Frame(main_container, bg="#0a0e17")
        logo_frame.pack(pady=(0, 30))
        
        tk.Label(
            logo_frame,
            text="🧠",
            font=("Segoe UI", 72),
            bg="#0a0e17",
            fg="#3498db",
        ).pack()
        
        # Заголовок с градиентным эффектом
        title_frame = tk.Frame(main_container, bg="#0a0e17")
        title_frame.pack(pady=(0, 20))
        
        tk.Label(
            title_frame,
            text="BCI SPELLER",
            font=("Segoe UI", 42, "bold"),
            bg="#0a0e17",
            fg="#ffffff",
        ).pack()
        
        # Подзаголовок с анимацией точки
        subtitle_frame = tk.Frame(main_container, bg="#0a0e17")
        subtitle_frame.pack(pady=(0, 40))
        
        tk.Label(
            subtitle_frame,
            text="ИНТЕРФЕЙС МОЗГ-КОМПЬЮТЕР",
            font=("Segoe UI", 18),
            bg="#0a0e17",
            fg="#3498db",
        ).pack()
        
        # Описание
        desc_frame = tk.Frame(main_container, bg="#0a0e17")
        desc_frame.pack(pady=(0, 60))
        
        tk.Label(
            desc_frame,
            text="Добро пожаловать в эксперимент по нейроуправлению\n"
                 "Вам предстоит вводить текст силой мысли",
            font=("Segoe UI", 14),
            bg="#0a0e17",
            fg="#95a5a6",
            justify="center",
        ).pack()
        
        # Индикатор прогресса
        progress_frame = tk.Frame(main_container, bg="#0a0e17")
        progress_frame.pack(pady=(0, 30))
        
        self.progress_canvas = tk.Canvas(progress_frame, width=200, height=4, bg="#0a0e17", highlightthickness=0)
        self.progress_canvas.pack()
        self.progress_canvas.create_rectangle(0, 0, 200, 4, fill="#1a2238", outline="")
        self.progress_canvas.create_rectangle(0, 0, 50, 4, fill="#3498db", outline="")
        
        # Стилизованная кнопка продолжения
        button_frame = tk.Frame(main_container, bg="#0a0e17")
        button_frame.pack()
        
        continue_button = tk.Button(
            button_frame,
            text="НАЧАТЬ ЭКСПЕРИМЕНТ",
            font=("Segoe UI", 14, "bold"),
            command=self._on_continue,
            bg="#0a0e17",
            fg="#ffffff",
            activebackground="#3498db",
            activeforeground="#ffffff",
            relief="flat",
            width=25,
            height=2,
            bd=2,
            highlightthickness=2,
            highlightbackground="#3498db",
            highlightcolor="#3498db",
            cursor="hand2"
        )
        continue_button.pack()
        self._create_glow_effect(continue_button)
        
        # Подсказка
        hint_frame = tk.Frame(main_container, bg="#0a0e17")
        hint_frame.pack(pady=(20, 0))
        
        tk.Label(
            hint_frame,
            text="Нажмите ПРОБЕЛ для продолжения",
            font=("Segoe UI", 7),
            bg="#0a0e17",
            fg="white",
        ).pack()

        self.window.bind("<space>", lambda e: self._on_continue())

    def _create_glow_effect(self, widget):
        """Создает эффект свечения для виджета"""
        def on_enter(e):
            widget.config(highlightbackground="#2980b9", highlightcolor="#2980b9")
            
        def on_leave(e):
            widget.config(highlightbackground="#3498db", highlightcolor="#3498db")
            
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _exit_program(self, event=None):
        """Закрытие программы"""
        self.window.destroy()
        self.root.destroy()
        sys.exit(0)

    def _on_continue(self):
        self.window.destroy()
        self.callback()