import tkinter as tk

def check_monitors():
    root = tk.Tk()
    root.withdraw()
    
    print("=== Информация о мониторах ===")
    print(f"Общая ширина: {root.winfo_screenwidth()}")
    print(f"Общая высота: {root.winfo_screenheight()}")
    
    # Пробуем разные позиции
    test_positions = [
        (0, 0, "Основной монитор (0,0)"),
        (1920, 0, "Монитор справа (1920,0)"),
        (-1920, 0, "Монитор слева (-1920,0)"),
        (3840, 0, "Дальше справа (3840,0)"),
        (-3840, 0, "Дальше слева (-3840,0)")
    ]
    
    test_window = tk.Toplevel(root)
    test_window.geometry("300x100")
    test_window.title("Тестовое окно")
    
    label = tk.Label(test_window, text="Перемещаю окно...")
    label.pack(pady=20)
    
    for x, y, desc in test_positions:
        test_window.geometry(f"+{x}+{y}")
        test_window.update()
        
        actual_x = test_window.winfo_x()
        actual_y = test_window.winfo_y()
        
        print(f"{desc}: запрошено ({x}, {y}), получено ({actual_x}, {actual_y})")
        
        label.config(text=f"Позиция: {desc}\n({x}, {y}) → ({actual_x}, {actual_y})")
        test_window.update()
        root.after(1000)  # Пауза 1 секунда
    
    test_window.destroy()
    root.destroy()

if __name__ == "__main__":
    check_monitors()