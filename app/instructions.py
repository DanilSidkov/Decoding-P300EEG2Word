import sys
import tkinter as tk
from tkinter import ttk


class InstructionWindow:
    def __init__(self, parent_callback, root_window, theme_manager):
        self.callback = parent_callback
        self.root = root_window
        self.theme_manager = theme_manager
        self.theme = theme_manager.get_theme()

        self.window = tk.Toplevel()
        self.window.title("BCI Speller - Инструкция")
        
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
        
        # Создаем фоновые эффекты
        self._create_background_effects()
        
        # Создаем кнопки управления (тема и выход)
        self._create_control_buttons()
        
        # Создаем основной контент
        self._create_main_content()

    def _create_background_effects(self):
        """Создает технологичный фон с элементами"""
        width = self.window.winfo_screenwidth()
        height = self.window.winfo_screenheight()
        
        # Круговые элементы
        for i in range(5):
            x = width // 4 + i * 100
            y = height // 3 + i * 50
            self.canvas.create_oval(x-50, y-50, x+50, y+50, 
                                  outline=self.theme["canvas_outline"], width=1, dash=(5, 5))
        
        # Линии соединения
        for i in range(10):
            x1 = width // 10 * i
            y1 = height // 5
            x2 = width // 10 * (i+1)
            y2 = height // 5 * 4
            self.canvas.create_line(x1, y1, x2, y2, fill=self.theme["canvas_outline"], width=1, dash=(3, 3))

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
        main_container.place(relx=0.5, rely=0.5, anchor="center", width=900, height=700)
        
        # Заголовок
        title_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        title_frame.pack(pady=(0, 30))
        
        tk.Label(
            title_frame,
            text="ИНСТРУКЦИЯ",
            font=("Segoe UI", 36, "bold"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
        ).pack()
        
        # Подзаголовок
        subtitle_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        subtitle_frame.pack(pady=(0, 20))
        
        tk.Label(
            subtitle_frame,
            text="Протокол проведения эксперимента SSVEP BCI",
            font=("Segoe UI", 16),
            bg=self.theme["bg_primary"],
            fg=self.theme["accent_primary"],
        ).pack()
        
        # Контейнер для текста инструкции
        text_container = tk.Frame(main_container, bg=self.theme["bg_primary"])
        text_container.pack(fill=tk.BOTH, expand=True, pady=(0, 30))
        
        # Создаем кастомный скроллбар
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Vertical.TScrollbar", 
                       background=self.theme["scrollbar_bg"],
                       troughcolor=self.theme["scrollbar_trough"],
                       bordercolor=self.theme["bg_primary"],
                       arrowcolor=self.theme["accent_primary"],
                       relief="flat")
        
        scroll_frame = tk.Frame(text_container, bg=self.theme["bg_primary"])
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # Текстовое поле в стиле терминала
        text_widget = tk.Text(
            scroll_frame,
            wrap=tk.WORD,
            font=("Consolas", 13),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
            insertbackground=self.theme["accent_primary"],
            selectbackground=self.theme["accent_primary"],
            selectforeground=self.theme["text_primary"],
            relief="flat",
            bd=0,
            padx=20,
            pady=20,
            height=15,
            width=60
        )
        
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", 
                                 command=text_widget.yview, 
                                 style="Custom.Vertical.TScrollbar")
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Содержимое инструкции
        instructions = """╔═══════════════════════════════════════════════════╗
║           ПРОТОКОЛ ЭКСПЕРИМЕНТА SSVEP BCI           ║
╚═══════════════════════════════════════════════════╝

⌨️  ПРОЦЕДУРА ВВОДА СИМВОЛА:

1. █ ЦЕЛЕВОЙ СИМВОЛ
   • На экране появится целевой символ
   • Сфокусируйте взгляд на этом символе
   
2. 🚀 ПОДТВЕРЖДЕНИЕ
   • Нажмите ПРОБЕЛ для подтверждения готовности
   • Это запустит процесс мигания
   
3. ⚡ ЦИКЛ МИГАНИЯ
   • Появится матрица с мигающими символами
   • Продолжайте смотреть на целевой символ
   • Длительность: 0.5 сек × 10 циклов
   
4. ✅ РЕЗУЛЬТАТ
   • Символ автоматически добавится в текст
   • Переход к следующему символу

⚠️  ВАЖНЫЕ РЕКОМЕНДАЦИИ:

• 🎯 Сохраняйте фокус на целевом символе
• 🧘 Держите голову неподвижно
• 👁️  Минимизируйте моргание во время мигания
• 💺 Займите удобное положение
• 🔄 Процесс повторяется для каждого символа

📊 ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ:
• Длина кода: 9 бит
• Частота мигания: регулируемая
• Количество циклов: 10
• Длительность цикла: 0.5 секунд

⚡ ГОТОВНОСТЬ К ЭКСПЕРИМЕНТУ:"""
        
        text_widget.insert(tk.END, instructions)
        
        # Добавляем цветовое форматирование
        text_widget.tag_add("title", "1.0", "3.0")
        text_widget.tag_config("title", foreground=self.theme["accent_primary"], font=("Consolas", 13, "bold"))
        
        text_widget.tag_add("section", "5.0", "5.100")
        text_widget.tag_config("section", foreground=self.theme["accent_success"], font=("Consolas", 13, "bold"))
        
        text_widget.tag_add("warning", "19.0", "19.100")
        text_widget.tag_config("warning", foreground=self.theme["accent_warning"], font=("Consolas", 13, "bold"))
        
        text_widget.tag_add("params", "27.0", "27.100")
        text_widget.tag_config("params", foreground="#9b59b6", font=("Consolas", 13, "bold"))
        
        text_widget.config(state=tk.DISABLED)
        
        # Кнопка продолжения
        button_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        button_frame.pack(pady=(10, 0))
        
        continue_button = tk.Button(
            button_frame,
            text="ПОНЯТНО →",
            font=("Segoe UI", 14, "bold"),
            command=self._on_continue,
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["accent_success"],
            activeforeground=self.theme["text_primary"],
            relief="flat",
            width=20,
            height=2,
            bd=2,
            highlightthickness=2,
            highlightbackground=self.theme["accent_success"],
            highlightcolor=self.theme["accent_success"],
            cursor="hand2"
        )
        continue_button.pack()
        self._create_glow_effect(continue_button, self.theme["accent_success"])
        
        # Подсказка
        hint_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        hint_frame.pack(pady=(10, 0))
        
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