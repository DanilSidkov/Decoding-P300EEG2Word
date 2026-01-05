import tkinter as tk

class InstructionWindow:
    def __init__(self, parent_callback):
        self.callback = parent_callback
        
        self.window = tk.Toplevel()
        self.window.title("Инструкция")
        self.window.geometry("600x500")
        self.window.configure(bg='white')
        self.window.resizable(False, False)
        
        self.window.attributes('-topmost', True)
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)
        
        self._create_content()
        self._center_window()
    
    def _create_content(self):
        tk.Label(
            self.window,
            text="ИНСТРУКЦИЯ",
            font=('Arial', 22, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(pady=(20, 15))
        
        frame = tk.Frame(self.window, bg='white')
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(
            frame,
            wrap=tk.WORD,
            font=('Arial', 11),
            bg='white',
            fg='#2c3e50',
            yscrollcommand=scrollbar.set,
            height=15,
            width=50,
            padx=10,
            pady=10
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        instructions = """
        ИНСТРУКЦИЯ ПО ПРОВЕДЕНИЮ ЭКСПЕРИМЕНТА:

        Для ввода КАЖДОГО символа:

        1. Вам будет показан целевой символ в отдельном окне.
        2. Сфокусируйте взгляд на этом символе.
        3. Нажмите ПРОБЕЛ, чтобы подтвердить готовность.
        4. Появится клавиатура с мигающими символами.
        5. Продолжайте смотреть на целевой символ во время мигания.
        6. После завершения цикла символ будет добавлен в текст.

        Процесс повторяется для каждого символа текста.

        ВАЖНО:
        • Держите голову неподвижно
        • Сфокусируйтесь только на целевом символе
        • Старайтесь не моргать во время мигания
        • Для перехода к следующему символу используйте ПРОБЕЛ

        УДАЧИ!
        """
        
        text_widget.insert(tk.END, instructions)
        text_widget.config(state=tk.DISABLED)
        
        tk.Button(
            self.window,
            text="ПОНЯЛ, НАЧИНАЕМ",
            font=('Arial', 12, 'bold'),
            command=self._on_continue,
            bg='#3498db',
            fg='white',
            width=20,
            height=2
        ).pack(pady=(10, 20))
        
        self.window.bind('<space>', lambda e: self._on_continue())
    
    def _center_window(self):
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.window.geometry(f'{width}x{height}+{x}+{y}')
    
    def _on_continue(self):
        self.window.destroy()
        self.callback()