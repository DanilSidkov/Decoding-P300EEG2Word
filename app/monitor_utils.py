import tkinter as tk
import sys


def get_monitor_info():
    """Получает информацию о мониторах"""
    root = tk.Tk()
    root.withdraw()
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Проверяем, есть ли второй монитор
    # (предполагаем, что второй монитор справа от первого)
    if screen_width > 2560:  # Ширина больше чем у одного 4K монитора
        # Вероятно, есть несколько мониторов
        return {
            'total_width': screen_width,
            'total_height': screen_height,
            'has_second_monitor': True,
            'second_monitor_x': 1920  # Начало второго монитора
        }
    else:
        return {
            'total_width': screen_width,
            'total_height': screen_height,
            'has_second_monitor': False
        }


def move_to_monitor(window, monitor_num=2):
    """
    Перемещает окно на указанный монитор
    
    Args:
        window: Окно Tkinter
        monitor_num: Номер монитора (1 или 2)
    """
    if monitor_num == 1:
        # Первый монитор
        window.geometry("+0+0")
    elif monitor_num == 2:
        # Определяем положение второго монитора
        monitor_info = get_monitor_info()
        if monitor_info['has_second_monitor']:
            # Перемещаем на второй монитор
            window.geometry(f"+{monitor_info['second_monitor_x']}+0")
        else:
            # Если второго монитора нет, центрируем на первом
            center_window(window)
    else:
        raise ValueError("monitor_num должен быть 1 или 2")


def center_window(window):
    """Центрирует окно на экране"""
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    
    window.geometry(f"{width}x{height}+{x}+{y}")