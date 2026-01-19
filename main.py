import tkinter as tk

from app.speller import SSVEPSpellerExperiment
from app.monitor_utils import move_to_monitor


def main():
    root = tk.Tk()
    
    # Скрываем окно для настройки
    root.withdraw()
    
    # Перемещаем окно на второй монитор
    # (если второго монитора нет, окно останется на первом)
    move_to_monitor(root, monitor_num=2)
    
    app = SSVEPSpellerExperiment(root)
    app.start()

    root.mainloop()


if __name__ == "__main__":
    main()