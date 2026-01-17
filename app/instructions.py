import sys
import tkinter as tk


class InstructionWindow:
    def __init__(self, parent_callback, root_window):
        self.callback = parent_callback
        self.root_window = root_window

        self.window = tk.Toplevel()

        self.window.attributes("-fullscreen", True)
        self.window.configure(bg="white")

        self.window.bind("<Escape>", self._exit_program)
        self.window.protocol("WM_DELETE_WINDOW", self._exit_program)

        exit_button = tk.Button(
            self.window,
            text="✕ ВЫЙТИ",
            font=("Arial", 12, "bold"),
            command=self._exit_program,
            bg="#e74c3c",
            fg="white",
            relief="flat",
            padx=20,
            pady=10,
        )
        exit_button.place(x=20, y=20)

        self._create_content()

    def _create_content(self):
        center_frame = tk.Frame(self.window, bg="white")
        center_frame.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(
            center_frame,
            text="ИНСТРУКЦИЯ",
            font=("Arial", 22, "bold"),
            bg="white",
            fg="#2c3e50",
        ).pack(pady=(0, 15))

        frame = tk.Frame(self.window, bg="white")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(
            frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            bg="white",
            fg="#2c3e50",
            yscrollcommand=scrollbar.set,
            height=15,
            width=50,
            padx=10,
            pady=10,
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        instructions = """
        ИНСТРУКЦИЯ ПО ПРОВЕДЕНИЮ ЭКСПЕРИМЕНТА:

        Для ввода КАЖДОГО символа:

        1. Вам будет показан целевой символ в отдельном окне.
        2. Сфокусируйте взгляд на этом символе.
        3. Нажмите ПРОБЕЛ, чтобы подтвердить готовность.
        4. Появится клавиатура с мигающими символами.
        5. Продолжайте смотреть на целевой символ во время мигания.
        6. После завершения цикла символ будет добавлен в текст.

        Процесс повторяется для каждого символа текста.

        ВАЖНО:
        • Держите голову неподвижно
        • Сфокусируйтесь только на целевом символе
        • Старайтесь не моргать во время мигания
        • Для перехода к следующему символу используйте ПРОБЕЛ

        УДАЧИ!
        """

        text_widget.insert(tk.END, instructions)
        text_widget.config(state=tk.DISABLED)

        tk.Button(
            self.window,
            text="ПОНЯЛ, НАЧИНАЕМ",
            font=("Arial", 12, "bold"),
            command=self._on_continue,
            bg="#3498db",
            fg="white",
            width=20,
            height=2,
        ).pack(pady=(10, 20))

        self.window.bind("<space>", lambda e: self._on_continue())

    def _exit_program(self, event=None):
        """Закрытие программы"""
        self.window.destroy()
        self.root.destroy()
        sys.exit(0)

    def _on_continue(self):
        self.window.destroy()
        self.callback()
