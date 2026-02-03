import sys
import tkinter as tk
from monitor_config import setup_window_on_target_monitor

class TargetWindow:
    """Окно для отображения целевого символа"""

    def __init__(self, parent, symbol, on_start_callback, theme_manager):
        self.on_start_callback = on_start_callback
        self.symbol = symbol
        self.parent = parent
        self.theme_manager = theme_manager
        self.theme = theme_manager.get_theme()

        if self.symbol == " ":
            self.symbol = "␣"

        self.window = tk.Toplevel(parent)
        self.window.title(f"BCI Speller - Целевой символ: {self.symbol}")
        
        if not setup_window_on_target_monitor(self.window):
            self.window.attributes("-fullscreen", True)
        self.window.configure(bg=self.theme["bg_primary"])
        
        self.canvas = tk.Canvas(self.window, bg=self.theme["bg_primary"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self._create_background_effects()
        
        self.window.bind("<Escape>", self._exit_program)
        self.window.protocol("WM_DELETE_WINDOW", self._exit_program)
        
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

        exit_button = tk.Button(
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
        exit_button.place(x=20, y=20)
        self._create_glow_effect(exit_button, self.theme["accent_warning"])

        self.window.bind("<space>", self._on_space_pressed)
        self.window.focus_force()

        self._create_content()
        
        self._start_focus_animation()

    def _create_background_effects(self):
        """Создает технологичный фон с концентрическими кругами"""
        width = self.window.winfo_screenwidth()
        height = self.window.winfo_screenheight()
        
        self.canvas.delete("all")
        
        center_x, center_y = width // 2, height // 2
        
        for radius in range(100, min(width, height)//2, 50):
            self.canvas.create_oval(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                outline=self.theme["canvas_outline"], width=1, dash=(5, 10)
            )

    def _create_content(self):
        main_container = tk.Frame(self.canvas, bg=self.theme["bg_primary"])
        main_container.place(relx=0.5, rely=0.5, anchor="center", width=1000, height=800)
        
        self.focus_indicator = tk.Canvas(main_container, width=400, height=400, 
                                        bg=self.theme["bg_primary"], highlightthickness=0)
        self.focus_indicator.pack(pady=(0, 30))
        
        self.focus_circle = self.focus_indicator.create_oval(50, 50, 350, 350, 
                                                           outline=self.theme["accent_primary"], width=3)
        
        self.symbol_label = tk.Label(
            self.focus_indicator,
            text=self.symbol,
            font=("Segoe UI", 120, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
        )
        self.symbol_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.animated_circles = []
        for i in range(3):
            circle = self.focus_indicator.create_oval(
                100 + i*20, 100 + i*20,
                300 - i*20, 300 - i*20,
                outline=self.theme["accent_primary"], width=1, state="hidden"
            )
            self.animated_circles.append(circle)
        
        title_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        title_frame.pack(pady=(0, 40))
        
        tk.Label(
            title_frame,
            text="ЦЕЛЕВОЙ СИМВОЛ",
            font=("Segoe UI", 28, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_primary"],
        ).pack()
        
        instruction_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        instruction_frame.pack(pady=(0, 20))
        
        tk.Label(
            instruction_frame,
            text="Сфокусируйтесь на символе",
            font=("Segoe UI", 18),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
        ).pack()
        
        action_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        action_frame.pack(pady=(0, 10))
        
        tk.Label(
            action_frame,
            text="Нажмите ПРОБЕЛ для подтверждения",
            font=("Segoe UI", 14),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_success"],
        ).pack()
        
        self.ready_indicator = tk.Label(
            action_frame,
            text="○",
            font=("Segoe UI", 24),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_tertiary"],
        )
        self.ready_indicator.pack(pady=(10, 0))
        
        progress_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        progress_frame.pack(pady=(30, 0))
        
        self.progress_canvas = tk.Canvas(progress_frame, width=300, height=6, 
                                        bg=self.theme["bg_primary"], highlightthickness=0)
        self.progress_canvas.pack()
        self.progress_bar = self.progress_canvas.create_rectangle(0, 0, 0, 6, 
                                                                fill=self.theme["accent_primary"], outline="")
        self.progress_bg = self.progress_canvas.create_rectangle(0, 0, 300, 6, 
                                                               fill=self.theme["bg_tertiary"], outline="")

    def _start_focus_animation(self):
        """Запускает анимацию фокусировки"""
        def animate_circles(step=0):
            for i, circle in enumerate(self.animated_circles):
                if step % (len(self.animated_circles) * 2) == i * 2:
                    self.focus_indicator.itemconfig(circle, state="normal")
                elif step % (len(self.animated_circles) * 2) == i * 2 + 1:
                    self.focus_indicator.itemconfig(circle, state="hidden")
            
            if step % 4 == 0:
                self.ready_indicator.config(text="●", fg=self.theme["accent_success"])
            elif step % 4 == 2:
                self.ready_indicator.config(text="○", fg=self.theme["text_tertiary"])
            
            progress = (step % 100) / 100
            self.progress_canvas.coords(self.progress_bar, 0, 0, 300 * progress, 6)
            
            self.window.after(100, lambda: animate_circles(step + 1))
        
        animate_circles()

    def _create_glow_effect(self, widget, color=None):
        """Создает эффект свечения для виджета"""
        if color is None:
            color = self.theme["accent_primary"]
            
        def on_enter(e):
            widget.config(fg=color)
            
        def on_leave(e):
            widget.config(fg=self.theme["button_fg"])
            
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _toggle_theme(self):
        """Переключение темы"""
        self.theme = self.theme_manager.toggle_theme()
        self._update_theme()
        
    def _update_theme(self):
        """Обновление цветов согласно теме"""
        self.window.configure(bg=self.theme["bg_primary"])
        self.canvas.configure(bg=self.theme["bg_primary"])
        
        self._create_background_effects()
        
        self.theme_button.config(
            text="☀️" if self.theme_manager.is_dark_mode else "🌙",
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            highlightbackground=self.theme["border_primary"]
        )
        
        self._create_content()
        
        self._start_focus_animation()

    def _exit_program(self, event=None):
        """Закрытие программы"""
        self.window.destroy()
        self.parent.destroy()
        sys.exit(0)

    def _on_space_pressed(self, event=None):
        """Обработка нажатия пробела"""
        self.ready_indicator.config(text="✓", fg=self.theme["accent_success"], font=("Segoe UI", 28))
        self.symbol_label.config(fg=self.theme["accent_success"])
        
        for i in range(10):
            progress = (i + 1) / 10
            self.window.after(i * 50, 
                            lambda p=progress: self.progress_canvas.coords(
                                self.progress_bar, 0, 0, 300 * p, 6))
        
        self.window.after(10, lambda: [self.window.destroy(), self.on_start_callback()])

    def wait_for_close(self):
        """Ожидает закрытия окна"""
        self.window.wait_window()