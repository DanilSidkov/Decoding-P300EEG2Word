import tkinter as tk

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
        
        # Получаем размеры экрана
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        
        # Создаем большое окно (40% экрана)
        window_width = int(screen_width * 0.4)
        window_height = int(screen_height * 0.4)
        
        self.window = tk.Toplevel(parent)
        self.window.title(f"Целевой символ")
        self.window.geometry(f"{window_width}x{window_height}")
        self.window.configure(bg='white')
        self.window.resizable(False, False)
        
        # Делаем окно поверх всех
        self.window.attributes('-topmost', True)
        
        # Привязываем пробел
        self.window.bind('<space>', self._on_space_pressed)
        
        # Отключаем кнопку закрытия
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Создаем содержимое
        self._create_content()
        
        # Центрируем окно
        self._center_window(window_width, window_height)
        
        # Фокусируемся на окне
        self.window.focus_force()
    
    def _create_content(self):
        """Создает содержимое окна"""
        # Заголовок
        tk.Label(
            self.window,
            text="ЦЕЛЕВОЙ СИМВОЛ",
            font=('Arial', 28, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(pady=(60, 40))
        
        # Очень большой символ
        tk.Label(
            self.window,
            text=self.symbol,
            font=('Arial', 120, 'bold'),
            bg='white',
            fg='#e74c3c'
        ).pack(pady=(0, 60))
        
        # Инструкция
        tk.Label(
            self.window,
            text="Сфокусируйтесь на символе и нажмите ПРОБЕЛ",
            font=('Arial', 16),
            bg='white',
            fg='#7f8c8d'
        ).pack()
        
        # Дополнительная инструкция
        tk.Label(
            self.window,
            text="(для запуска мигания)",
            font=('Arial', 12),
            bg='white',
            fg='#95a5a6'
        ).pack(pady=(10, 0))
    
    def _center_window(self, width, height):
        """Центрирует окно на экране"""
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.window.geometry(f'{width}x{height}+{x}+{y}')
    
    def _on_space_pressed(self, event=None):
        """Обработка нажатия пробела"""
        self.window.destroy()
        self.on_start_callback()
    
    def wait_for_close(self):
        """Ожидает закрытия окна"""
        self.window.wait_window()