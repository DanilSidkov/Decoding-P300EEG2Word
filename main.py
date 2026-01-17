import tkinter as tk

from app.speller import SSVEPSpellerExperiment


def main():
    root = tk.Tk()

    root.withdraw()

    app = SSVEPSpellerExperiment(root)
    app.start()

    root.mainloop()


if __name__ == "__main__":
    main()
