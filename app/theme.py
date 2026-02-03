# theme.py
class ThemeManager:
    """Менеджер тем для приложения"""
    
    # Темная тема (по умолчанию)
    DARK_THEME = {
        "name": "dark",
        "bg_primary": "#0a0e17",      # Основной фон
        "bg_secondary": "#1a2238",    # Вторичный фон
        "bg_tertiary": "#2c3e50",     # Третичный фон
        "text_primary": "#ffffff",    # Основной текст
        "text_secondary": "#95a5a6",  # Вторичный текст
        "text_tertiary": "#7f8c8d",   # Третичный текст
        "accent_primary": "#3498db",  # Основной акцент
        "accent_secondary": "#2ecc71",# Вторичный акцент
        "accent_warning": "#e74c3c",  # Цвет предупреждения
        "accent_success": "#2ecc71",  # Цвет успеха
        "border_primary": "#3498db",  # Основная граница
        "border_secondary": "#2c3e50",# Вторичная граница
        "canvas_outline": "#1a2238",  # Контур холста
        "grid_lines": "#1a2238",      # Линии сетки
        "button_bg": "#0a0e17",       # Фон кнопки
        "button_fg": "#ffffff",       # Текст кнопки
        "scrollbar_bg": "#2c3e50",    # Фон скроллбара
        "scrollbar_trough": "#1a2238",# Фон полосы прокрутки
        "entry_bg": "#1a2238",        # Фон поля ввода
        "entry_fg": "#ffffff",        # Текст поля ввода
        "symbol_target_0":"#1d2231", # Таргетного символа в неактивном состоянии
        "symbol_custom_0":"#1d2231",  # Цвет обычного символа в неактивном состоянии
        "symbol_target_1":"#ffffff",  # Таргетного символа в активном состоянии
        "symbol_custom_1":"#ffffff"   # Цвет обычного символа в активном состоянии
    }
    
    # Светлая тема
    LIGHT_THEME = {
        "name": "light",
        "bg_primary": "#f5f7fa",      # Основной фон
        "bg_secondary": "#e1e8f0",    # Вторичный фон
        "bg_tertiary": "#d1d9e6",     # Третичный фон
        "text_primary": "#000000",    # Основной текст
        "text_secondary": "#adb8c4",  # Вторичный текст
        "text_tertiary": "#cbbcbc",   # Третичный текст
        "accent_primary": "#2980b9",  # Основной акцент
        "accent_secondary": "#27ae60",# Вторичный акцент
        "accent_warning": "#c0392b",  # Цвет предупреждения
        "accent_success": "#27ae60",  # Цвет успеха
        "border_primary": "#2980b9",  # Основная граница
        "border_secondary": "#bdc3c7",# Вторичная граница
        "canvas_outline": "#d1d9e6",  # Контур холста
        "grid_lines": "#d1d9e6",      # Линии сетки
        "button_bg": "#f5f7fa",       # Фон кнопки
        "button_fg": "#2c3e50",       # Текст кнопки
        "scrollbar_bg": "#bdc3c7",    # Фон скроллбара
        "scrollbar_trough": "#e1e8f0",# Фон полосы прокрутки
        "entry_bg": "#ffffff",        # Фон поля ввода
        "entry_fg": "#2c3e50",        # Текст поля ввода
        "symbol_target_0": "#000000", # Таргетного символа в неактивном состоянии
        "symbol_custom_0":"#000000",  # Цвет обычного символа в неактивном состоянии
        "symbol_target_1":"#000000",  # Таргетного символа в активном состоянии
        "symbol_custom_1":"#000000"   # Цвет обычного символа в активном состоянии
    }
    
    def __init__(self):
        self.current_theme = self.LIGHT_THEME
        self.is_dark_mode = False
        
    def toggle_theme(self):
        """Переключение между светлой и темной темой"""
        if self.is_dark_mode:
            self.current_theme = self.LIGHT_THEME
            self.is_dark_mode = False
        else:
            self.current_theme = self.DARK_THEME
            self.is_dark_mode = True
        return self.current_theme
    
    def get_theme(self):
        """Получение текущей темы"""
        return self.current_theme
    
    def set_theme(self, theme_name):
        """Установка определенной темы"""
        if theme_name == "light":
            self.current_theme = self.LIGHT_THEME
            self.is_dark_mode = False
        else:
            self.current_theme = self.DARK_THEME
            self.is_dark_mode = True
        return self.current_theme