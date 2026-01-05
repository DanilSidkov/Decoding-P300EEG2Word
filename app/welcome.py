import tkinter as tk

class WelcomeWindow:
    def __init__(self, parent_callback):
        self.callback = parent_callback
        
        self.window = tk.Toplevel()
        self.window.title("Добро пожаловать!")
        self.window.geometry("500x350")
        self.window.configure(bg='white')
        self.window.resizable(False, False)
        
        self.window.attributes('-topmost', True)
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)
        
        self._create_content()
        self._center_window()
    
    def _create_content(self):
        tk.Label(
            self.window,
            text="ДОБРО ПОЖАЛОВАТЬ!",
            font=('Arial', 22, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(pady=(40, 20))
        
        tk.Label(
            self.window,
            text="Спасибо за участие в эксперименте",
            font=('Arial', 14),
            bg='white',
            fg='#34495e'
        ).pack(pady=(0, 20))
        
        tk.Label(
            self.window,
            text="В этом эксперименте вы будете\nвводить текст с помощью\nинтерфейса мозг-компьютер.",
            font=('Arial', 12),
            bg='white',
            fg='#7f8c8d',
            justify='center'
        ).pack(pady=(0, 40))
        
        tk.Button(
            self.window,
            text="ПРОДОЛЖИТЬ (ПРОБЕЛ)",
            font=('Arial', 12, 'bold'),
            command=self._on_continue,
            bg='#27ae60',
            fg='white',
            width=20,
            height=2
        ).pack()
        
        self.window.bind('<space>', lambda e: self._on_continue())
    
    def _center_window(self):
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.window.geometry(f'{width}x{height}+{x}+{y}')
    
    def _on_continue(self):
        self.window.destroy()
        self.callback()