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


#from app.controller import ExperimentController
#
#def main():
#    controller = ExperimentController()
#    
#    try:
#        controller.start_experiment()
#    except KeyboardInterrupt:
#        print("Эксперимент прерван пользователем")
#    except Exception as e:
#        print(f"Ошибка: {e}")
#    finally:
#        if controller.neorec_process:
#            controller.neorec_process.terminate()#
#
#if __name__ == "__main__":
#    main()