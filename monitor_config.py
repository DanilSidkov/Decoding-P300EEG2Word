# monitor_config.py
from screeninfo import get_monitors

def get_monitor_config():
    """Возвращает конфигурацию мониторов"""
    monitors = get_monitors()
    
    # Ищем второй монитор (левый)
    target_monitor = None
    primary_monitor = None
    
    for monitor in monitors:
        if monitor.is_primary:
            primary_monitor = monitor
        else:
            target_monitor = monitor  # Левый монитор
    
    # Если не нашли второй монитор, используем основной
    if not target_monitor and primary_monitor:
        target_monitor = primary_monitor
    
    return target_monitor

# Получаем конфигурацию целевого монитора
TARGET_MONITOR = get_monitor_config()

def get_fullscreen_geometry():
    """Возвращает geometry для полноэкранного окна на целевом мониторе"""
    if TARGET_MONITOR:
        return f"{TARGET_MONITOR.width}x{TARGET_MONITOR.height}+{TARGET_MONITOR.x}+{TARGET_MONITOR.y}"
    return None

def setup_window_on_target_monitor(window):
    """Настраивает окно на целевом мониторе без использования fullscreen"""
    if TARGET_MONITOR:
        geometry = get_fullscreen_geometry()
        window.geometry(geometry)
        
        # Убираем рамку окна и делаем его поверх всех окон
        window.overrideredirect(True)  # Убирает рамку и кнопки управления
        window.attributes('-topmost', True)  # Поверх всех окон
        
        # Устанавливаем фокус
        window.focus_force()
        
        print(f"Окно настроено на мониторе: {geometry}")
        return True
    return False

# Тестирование
if __name__ == "__main__":
    print(f"Целевой монитор: {TARGET_MONITOR}")
    print(f"Fullscreen geometry: {get_fullscreen_geometry()}")