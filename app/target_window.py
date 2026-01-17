import sys
import tkinter as tk


class TargetWindow:
    """Окно для отображения целевого символа"""

    def __init__(self, parent, symbol, on_start_callback):
        self.on_start_callback = on_start_callback
        self.symbol = symbol
        self.parent = parent

        if self.symbol == " ":
            self.symbol = "␣"

        self.window = tk.Toplevel(parent)
        self.window.title(f"BCI Speller - Целевой символ: {self.symbol}")
        
        # Технологичный дизайн
        self.window.attributes("-fullscreen", True)
        self.window.configure(bg="#0a0e17")
        
        # Холст для эффектов
        self.canvas = tk.Canvas(self.window, bg="#0a0e17", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self._create_background_effects()
        
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

        self.window.bind("<space>", self._on_space_pressed)
        self.window.focus_force()

        self._create_content()
        
        # Запускаем анимацию фокуса
        self._start_focus_animation()

    def _create_background_effects(self):
        """Создает технологичный фон с концентрическими кругами"""
        width = self.window.winfo_screenwidth()
        height = self.window.winfo_screenheight()
        
        # Центральная точка
        center_x, center_y = width // 2, height // 2
        
        # Концентрические круги
        for radius in range(100, min(width, height)//2, 50):
            self.canvas.create_oval(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                outline="#1a2238", width=1, dash=(5, 10)
            )

    def _create_content(self):
        # Основной контейнер
        main_container = tk.Frame(self.canvas, bg="#0a0e17")
        main_container.place(relx=0.5, rely=0.5, anchor="center", width=1000, height=800)
        
        # Индикатор фокуса
        self.focus_indicator = tk.Canvas(main_container, width=400, height=400, 
                                        bg="#0a0e17", highlightthickness=0)
        self.focus_indicator.pack(pady=(0, 30))
        
        # Круг фокуса
        self.focus_circle = self.focus_indicator.create_oval(50, 50, 350, 350, 
                                                           outline="#3498db", width=3)
        
        # Целевой символ в центре
        self.symbol_label = tk.Label(
            self.focus_indicator,
            text=self.symbol,
            font=("Segoe UI", 120, "bold"),
            bg="#0a0e17",
            fg="#ffffff",
        )
        self.symbol_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Анимированные круги вокруг символа
        self.animated_circles = []
        for i in range(3):
            circle = self.focus_indicator.create_oval(
                100 + i*20, 100 + i*20,
                300 - i*20, 300 - i*20,
                outline="#3498db", width=1, state="hidden"
            )
            self.animated_circles.append(circle)
        
        # Заголовок
        title_frame = tk.Frame(main_container, bg="#0a0e17")
        title_frame.pack(pady=(0, 40))
        
        tk.Label(
            title_frame,
            text="ЦЕЛЕВОЙ СИМВОЛ",
            font=("Segoe UI", 28, "bold"),
            bg="#0a0e17",
            fg="#3498db",
        ).pack()
        
        # Инструкция
        instruction_frame = tk.Frame(main_container, bg="#0a0e17")
        instruction_frame.pack(pady=(0, 20))
        
        tk.Label(
            instruction_frame,
            text="Сфокусируйтесь на символе",
            font=("Segoe UI", 18),
            bg="#0a0e17",
            fg="#ffffff",
        ).pack()
        
        # Подсказка действия
        action_frame = tk.Frame(main_container, bg="#0a0e17")
        action_frame.pack(pady=(0, 10))
        
        tk.Label(
            action_frame,
            text="Нажмите ПРОБЕЛ для подтверждения",
            font=("Segoe UI", 14),
            bg="#0a0e17",
            fg="#2ecc71",
        ).pack()
        
        # Индикатор готовности
        self.ready_indicator = tk.Label(
            action_frame,
            text="○",
            font=("Segoe UI", 24),
            bg="#0a0e17",
            fg="#7f8c8d",
        )
        self.ready_indicator.pack(pady=(10, 0))
        
        # Прогресс-бар
        progress_frame = tk.Frame(main_container, bg="#0a0e17")
        progress_frame.pack(pady=(30, 0))
        
        self.progress_canvas = tk.Canvas(progress_frame, width=300, height=6, 
                                        bg="#0a0e17", highlightthickness=0)
        self.progress_canvas.pack()
        self.progress_bar = self.progress_canvas.create_rectangle(0, 0, 0, 6, 
                                                                fill="#3498db", outline="")
        self.progress_bg = self.progress_canvas.create_rectangle(0, 0, 300, 6, 
                                                               fill="#1a2238", outline="")

    def _start_focus_animation(self):
        """Запускает анимацию фокусировки"""
        def animate_circles(step=0):
            for i, circle in enumerate(self.animated_circles):
                if step % (len(self.animated_circles) * 2) == i * 2:
                    self.focus_indicator.itemconfig(circle, state="normal")
                elif step % (len(self.animated_circles) * 2) == i * 2 + 1:
                    self.focus_indicator.itemconfig(circle, state="hidden")
            
            ## Пульсация символа
            #if step % 10 < 5:
            #    self.symbol_label.config(fg="#3498db")
            #else:
            #    self.symbol_label.config(fg="#ffffff")
            
            # Пульсация готовности
            if step % 4 == 0:
                self.ready_indicator.config(text="●", fg="#2ecc71")
            elif step % 4 == 2:
                self.ready_indicator.config(text="○", fg="#7f8c8d")
            
            # Анимация прогресса
            progress = (step % 100) / 100
            self.progress_canvas.coords(self.progress_bar, 0, 0, 300 * progress, 6)
            
            self.window.after(100, lambda: animate_circles(step + 1))
        
        animate_circles()

    def _create_glow_effect(self, widget, color="#3498db"):
        """Создает эффект свечения для виджета"""
        def on_enter(e):
            widget.config(fg=color)
            
        def on_leave(e):
            widget.config(fg="#ffffff")
            
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _exit_program(self, event=None):
        """Закрытие программы"""
        self.window.destroy()
        self.parent.destroy()
        sys.exit(0)

    def _on_space_pressed(self, event=None):
        """Обработка нажатия пробела"""
        # Анимация подтверждения
        self.ready_indicator.config(text="✓", fg="#2ecc71", font=("Segoe UI", 28))
        self.symbol_label.config(fg="#2ecc71")
        
        # Заполняем прогресс-бар
        for i in range(10):
            progress = (i + 1) / 10
            self.window.after(i * 50, 
                            lambda p=progress: self.progress_canvas.coords(
                                self.progress_bar, 0, 0, 300 * p, 6))
        
        self.window.after(10, lambda: [self.window.destroy(), self.on_start_callback()])

    def wait_for_close(self):
        """Ожидает закрытия окна"""
        self.window.wait_window()