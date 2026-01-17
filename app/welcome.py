import sys
import tkinter as tk


class WelcomeWindow:
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
            text="ДОБРО ПОЖАЛОВАТЬ!",
            font=("Arial", 22, "bold"),
            bg="white",
            fg="#2c3e50",
        ).pack(pady=(0, 20))

        tk.Label(
            center_frame,
            text="Спасибо за участие в эксперименте",
            font=("Arial", 14),
            bg="white",
            fg="#34495e",
        ).pack(pady=(0, 20))

        tk.Label(
            center_frame,
            text="В этом эксперименте вы будете\nвводить текст с помощью\nинтерфейса мозг-компьютер.",
            font=("Arial", 12),
            bg="white",
            fg="#7f8c8d",
            justify="center",
        ).pack(pady=(0, 40))

        tk.Button(
            center_frame,
            text="ПРОДОЛЖИТЬ (ПРОБЕЛ)",
            font=("Arial", 12, "bold"),
            command=self._on_continue,
            bg="#27ae60",
            fg="white",
            width=20,
            height=2,
        ).pack()

        self.window.bind("<space>", lambda e: self._on_continue())

    def _exit_program(self, event=None):
        """Закрытие программы"""
        self.window.destroy()
        self.root.destroy()
        sys.exit(0)

    def _on_continue(self):
        self.window.destroy()
        self.callback()
