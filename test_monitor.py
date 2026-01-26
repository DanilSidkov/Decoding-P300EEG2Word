# test_monitor.py
import tkinter as tk

root = tk.Tk()
root.withdraw()

# Получаем размеры
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

print(f"Общая ширина: {screen_width}")
print(f"Общая высота: {screen_height}")

# Предположим, что мониторы одинаковые
# Если ширина ~3840, то каждый монитор ~1920
if screen_width > 2500:  # Явно больше одного монитора
    monitor_width = screen_width // 2
    print(f"Предполагаемая ширина монитора: {monitor_width}")
    print(f"Координата левого монитора: -{monitor_width}")
else:
    print("Похоже, только один монитор")

root.destroy()