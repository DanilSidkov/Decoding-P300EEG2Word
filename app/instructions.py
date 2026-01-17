import sys
import tkinter as tk
from tkinter import ttk


class InstructionWindow:
    def __init__(self, parent_callback, root_window):
        self.callback = parent_callback
        self.root = root_window

        self.window = tk.Toplevel()
        self.window.title("BCI Speller - Инструкция")
        
        # Технологичный дизайн
        self.window.attributes("-fullscreen", True)
        self.window.configure(bg="#0a0e17")
        
        # Холст для фоновых эффектов
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

        self._create_content()

    def _create_background_effects(self):
        """Создает технологичный фон с элементами"""
        width = self.window.winfo_screenwidth()
        height = self.window.winfo_screenheight()
        
        # Круговые элементы
        for i in range(5):
            x = width // 4 + i * 100
            y = height // 3 + i * 50
            self.canvas.create_oval(x-50, y-50, x+50, y+50, 
                                  outline="#1a2238", width=1, dash=(5, 5))
        
        # Линии соединения
        for i in range(10):
            x1 = width // 10 * i
            y1 = height // 5
            x2 = width // 10 * (i+1)
            y2 = height // 5 * 4
            self.canvas.create_line(x1, y1, x2, y2, fill="#1a2238", width=1, dash=(3, 3))

    def _create_content(self):
        # Основной контейнер
        main_container = tk.Frame(self.canvas, bg="#0a0e17")
        main_container.place(relx=0.5, rely=0.5, anchor="center", width=900, height=700)
        
        # Заголовок
        title_frame = tk.Frame(main_container, bg="#0a0e17")
        title_frame.pack(pady=(0, 30))
        
        tk.Label(
            title_frame,
            text="ИНСТРУКЦИЯ",
            font=("Segoe UI", 36, "bold"),
            bg="#0a0e17",
            fg="#ffffff",
        ).pack()
        
        # Подзаголовок
        subtitle_frame = tk.Frame(main_container, bg="#0a0e17")
        subtitle_frame.pack(pady=(0, 20))
        
        tk.Label(
            subtitle_frame,
            text="Протокол проведения эксперимента SSVEP BCI",
            font=("Segoe UI", 16),
            bg="#0a0e17",
            fg="#3498db",
        ).pack()
        
        # Контейнер для текста инструкции
        text_container = tk.Frame(main_container, bg="#0a0e17")
        text_container.pack(fill=tk.BOTH, expand=True, pady=(0, 30))
        
        # Создаем кастомный скроллбар
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Vertical.TScrollbar", 
                       background="#1a2238",
                       troughcolor="#0a0e17",
                       bordercolor="#0a0e17",
                       arrowcolor="#3498db",
                       relief="flat")
        
        scroll_frame = tk.Frame(text_container, bg="#0a0e17")
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # Текстовое поле в стиле терминала
        text_widget = tk.Text(
            scroll_frame,
            wrap=tk.WORD,
            font=("Consolas", 13),
            bg="#1a2238",
            fg="#e0e0e0",
            insertbackground="#3498db",
            selectbackground="#3498db",
            selectforeground="#ffffff",
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
        text_widget.tag_config("title", foreground="#3498db", font=("Consolas", 13, "bold"))
        
        text_widget.tag_add("section", "5.0", "5.100")
        text_widget.tag_config("section", foreground="#2ecc71", font=("Consolas", 13, "bold"))
        
        text_widget.tag_add("warning", "19.0", "19.100")
        text_widget.tag_config("warning", foreground="#e74c3c", font=("Consolas", 13, "bold"))
        
        text_widget.tag_add("params", "27.0", "27.100")
        text_widget.tag_config("params", foreground="#9b59b6", font=("Consolas", 13, "bold"))
        
        text_widget.config(state=tk.DISABLED)
        
        # Кнопка продолжения
        button_frame = tk.Frame(main_container, bg="#0a0e17")
        button_frame.pack(pady=(10, 0))
        
        continue_button = tk.Button(
            button_frame,
            text="ПОНЯТНО →",
            font=("Segoe UI", 14, "bold"),
            command=self._on_continue,
            bg="#0a0e17",
            fg="#ffffff",
            activebackground="#2ecc71",
            activeforeground="#ffffff",
            relief="flat",
            width=20,
            height=2,
            bd=2,
            highlightthickness=2,
            highlightbackground="#2ecc71",
            highlightcolor="#2ecc71",
            cursor="hand2"
        )
        continue_button.pack()
        self._create_glow_effect(continue_button, "#2ecc71")
        
        # Подсказка
        hint_frame = tk.Frame(main_container, bg="#0a0e17")
        hint_frame.pack(pady=(10, 0))
        
        tk.Label(
            hint_frame,
            text="Нажмите ПРОБЕЛ для продолжения",
            font=("Segoe UI", 11),
            bg="#0a0e17",
            fg="#7f8c8d",
        ).pack()

        self.window.bind("<space>", lambda e: self._on_continue())

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

    def _exit_program(self, event=None):
        """Закрытие программы"""
        self.window.destroy()
        self.root.destroy()
        sys.exit(0)

    def _on_continue(self):
        self.window.destroy()
        self.callback()