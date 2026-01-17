import tkinter as tk
import sys

class TargetWindow:
    """Окно для отображения целевого символа"""
    def __init__(self, parent, symbol, on_start_callback):
        """
        Args:
            parent: родительское окно
            symbol: целевой символ для показа
            on_start_callback: функция, вызываемая при нажатии пробела
        """
        self.on_start_callback = on_start_callback
        self.symbol = symbol
        self.parent = parent
        
        if self.symbol == ' ':
            self.symbol = '_'
        
        self.window = tk.Toplevel(parent)

        self.window.attributes('-fullscreen', True)
        self.window.configure(bg='white')
        
        self.window.bind('<Escape>', self._exit_program)
        self.window.protocol("WM_DELETE_WINDOW", self._exit_program)

        exit_button = tk.Button(
            self.window,
            text="✕ ВЫЙТИ",
            font=('Arial', 12, 'bold'),
            command=self._exit_program,
            bg='#e74c3c',
            fg='white',
            relief='flat',
            padx=20,
            pady=10
        )
        exit_button.place(x=20, y=20)
        
        self.window.bind('<space>', self._on_space_pressed)
        
        self.window.focus_force()
        
        self._create_content()
    
    def _create_content(self):
        """Создает содержимое окна"""
        # Контейнер для центрирования
        center_frame = tk.Frame(self.window, bg='white')
        center_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        # Заголовок
        tk.Label(
            center_frame,
            text="ЦЕЛЕВОЙ СИМВОЛ",
            font=('Arial', 28, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(pady=(0, 40))
        
        # Очень большой символ
        tk.Label(
            center_frame,
            text=self.symbol,
            font=('Arial', 120, 'bold'),
            bg='white',
            fg='#e74c3c'
        ).pack(pady=(0, 60))
        
        # Инструкция
        tk.Label(
            center_frame,
            text="Сфокусируйтесь на символе и нажмите ПРОБЕЛ",
            font=('Arial', 16),
            bg='white',
            fg='#7f8c8d'
        ).pack()
        
        # Дополнительная инструкция
        tk.Label(
            center_frame,
            text="(для запуска мигания)",
            font=('Arial', 12),
            bg='white',
            fg='#95a5a6'
        ).pack(pady=(10, 0))
    
    def _exit_program(self, event=None):
        """Закрытие программы"""
        import sys
        self.window.destroy()
        self.parent.destroy()
        sys.exit(0)
    
    def _on_space_pressed(self, event=None):
        """Обработка нажатия пробела"""
        self.window.destroy()
        self.on_start_callback()
    
    def wait_for_close(self):
        """Ожидает закрытия окна"""
        self.window.wait_window()