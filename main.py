import tkinter as tk
from app.speller import SSVEPSpellerExperiment

def main():
    # Создаем главное окно минимально быстро
    root = tk.Tk()
    root.withdraw()  # Скрываем сразу
    
    # Создаем и запускаем приложение
    app = SSVEPSpellerExperiment(root)
    app.start()
    
    root.mainloop()

if __name__ == "__main__":
    main()