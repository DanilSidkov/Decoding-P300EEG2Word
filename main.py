import tkinter as tk

from app.speller import SSVEPSpellerExperiment


def main():
    root = tk.Tk()

    root.withdraw()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    if screen_width > 1920:
        root.geometry(f"+{1920}+0")

    app = SSVEPSpellerExperiment(root)
    app.start()

    root.mainloop()


if __name__ == "__main__":
    main()
