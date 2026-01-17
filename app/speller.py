import tkinter as tk
from tkinter import messagebox
import threading
import time
from app.get_logger import setup_logger
import logging
from app.code_generator import CodeGen
from pylsl import StreamInfo, StreamOutlet
import sys

class SSVEPSpellerExperiment:
    def __init__(self, root):
        self.root = root
        self.logger = logging.getLogger("BCI")
        setup_logger(self.logger, "Experiment")

        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='white')
        
        self.root.bind('<Escape>', self._exit_program)
        
        self.root.protocol("WM_DELETE_WINDOW", self._exit_program)
        
        self.codelen = 9
        self.cycle_duration = 0.5
        self.num_cycles = 10
        self.base_interval = self.cycle_duration / 9
        self.transition_duration = self.base_interval * 0.2

        self.is_running = False
        self.current_interval = 0
        self.output_text = ""
        self.flash_thread = None
        
        self.target_symbol = ""
        self.target_symbols = []
        self.current_target_index = 0
        
        self.waiting_for_space = False

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self.controller = None
        info = StreamInfo(name='annotations',
                  type='Events', # 'Markers'
                  channel_count=1,
                  nominal_srate=0,
                  channel_format='string',
                  source_id='my_marker_stream')
        self.outlet = StreamOutlet(info)

    def _exit_program(self, event=None):
        """Закрытие программы"""
        self.logger.info("Программа завершена пользователем")
        if messagebox.askyesno("Выход", "Вы уверены, что хотите выйти?"):
            self.root.destroy()
            sys.exit(0)

    def send_event_marker(self, event_type, **kwargs):
        """Отправка маркера события через LSL"""
        formatted_kwargs = {}
        for key, value in kwargs.items():
            if key == 'states' and isinstance(value, str) and len(value) == 36:
                formatted_kwargs[key] = value
            else:
                formatted_kwargs[key] = value
        marker_string = f"{event_type} - {formatted_kwargs}"

        self.outlet.push_sample([marker_string])
        #self.lsl_service.send_marker(event_type, kwargs)
        self.logger.info(f"Событие: {event_type} {formatted_kwargs}")
    
    def start(self):
        """Запускает последовательность окон"""
        self.logger.info("Начало работы")
        self._show_welcome()
    
    def _show_welcome(self):
        """Показывает приветственное окно"""
        from app.welcome import WelcomeWindow
        self.send_event_marker("WINDOW_OPEN_hello", window="welcome")
        self.logger.info("Показ приветственного окна")
        WelcomeWindow(self._show_instructions, self.root)
    
    def _show_instructions(self):
        """Показывает окно инструкций"""
        from app.instructions import InstructionWindow
        self.send_event_marker("WINDOW_OPEN_instruct", window="instructions")
        self.logger.info("Показ окна инструкций")
        InstructionWindow(self._show_preparation, self.root)
    
    def _show_preparation(self):
        """Окно подготовки"""
        self.send_event_marker("WINDOW_OPEN_preparation", window="preparation")
        self.root.withdraw()

        self.prep_window = tk.Toplevel(self.root)
        self.prep_window.attributes('-fullscreen', True)
        self.prep_window.configure(bg='white')
        
        self.prep_window.bind('<Escape>', lambda e: self._exit_program(e))
        self.prep_window.protocol("WM_DELETE_WINDOW", lambda: self._exit_program())
        
        exit_button = tk.Button(
            self.prep_window,
            text="✕",
            font=('Arial', 14, 'bold'),
            command=self._exit_program,
            bg='#e74c3c',
            fg='white',
            relief='flat',
            width=3,
            height=1
        )
        exit_button.place(x=20, y=20)
        
        self.logger.info("Ввод данных")
        
        center_frame = tk.Frame(self.prep_window, bg='white')
        center_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(
            center_frame,
            text="НАСТРОЙКИ ЭКСПЕРИМЕНТА",
            font=('Arial', 18, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(pady=(0, 30))
        
        tk.Label(
            self.prep_window,
            text="Текст для ввода:",
            font=('Arial', 11),
            bg='white',
            fg='#34495e'
        ).pack(in_=center_frame)
        
        self.text_entry = tk.Entry(
            self.prep_window,
            font=('Arial', 12),
            width=30
        )
        self.text_entry.insert(0, "ПРИВЕТ")
        self.text_entry.pack(pady=(5, 15))
        
        tk.Label(
            self.prep_window,
            text="Длительность одного цикла (сек):",
            font=('Arial', 11),
            bg='white',
            fg='#34495e'
        ).pack(in_=center_frame)
        
        self.duration_entry = tk.Entry(
            self.prep_window,
            font=('Arial', 12),
            width=10,
            justify='center'
        )
        self.duration_entry.insert(0, "0.5")
        self.duration_entry.pack(pady=(5, 25))
    
        tk.Label(
            self.prep_window,
            text="Количество циклов мигания:",
            font=('Arial', 11),
            bg='white',
            fg='#34495e'
        ).pack(in_=center_frame)

        self.cycles_entry = tk.Entry(
            self.prep_window,
            font=('Arial', 12),
            width=10,
            justify='center'
        )
        self.cycles_entry.insert(0, "10")  # По умолчанию 10 циклов
        self.cycles_entry.pack(pady=(5, 25))

        tk.Label(
            self.prep_window,
            text="(1 цикл = N (длина кода) интервалов мигания)",
            font=('Arial', 9),
            bg='white',
            fg='#7f8c8d'
        ).pack(pady=(0, 10))

        tk.Label(
            self.prep_window,
            text="Длина двоичного кода:",
            font=('Arial', 11),
            bg='white',
            fg='#34495e'
        ).pack(in_=center_frame)

        self.codelen_entry = tk.Entry(
            self.prep_window,
            font=('Arial', 12),
            width=10,
            justify='center'
        )
        self.codelen_entry.insert(0, "9")  # По умолчанию 9
        self.codelen_entry.pack(pady=(5, 25))
    
        tk.Button(
            self.prep_window,
            text="НАЧАТЬ ЭКСПЕРИМЕНТ",
            font=('Arial', 12, 'bold'),
            command=self._start_experiment,
            bg='#27ae60',
            fg='white',
            width=20,
            height=2
        ).pack(in_=center_frame)
    
    def _start_experiment(self):
        """Начинает эксперимент"""
        try:
            text = self.text_entry.get().upper()
            if not text:
                self.logger.critical("Текст не введен")
                messagebox.showerror("Ошибка", "Введите текст")
                return
            
            self.logger.info("Запуск эксперимента!")
            self.codelen = int(self.codelen_entry.get())
            self.setup_symbols()

            self.target_symbols = list(text)
            self.current_target_index = 0
            
            duration = float(self.duration_entry.get())
            if duration <= 0:
                self.logger.error("Ошибка длительности")
                raise ValueError("Длительность должна быть > 0")
            
            self.cycle_duration = duration
            self.base_interval = self.cycle_duration / self.codelen
            self.transition_duration = self.base_interval * 0.2

            cycles = int(self.cycles_entry.get())
            if cycles <= 0:
                self.logger.error("Ошибка количества циклов")
                raise ValueError("Количество циклов должно быть > 0")
            
            self.num_cycles = cycles

            self.send_event_marker("EXPERIMENT_START", 
                        text=self.text_entry.get(),
                        codelen=self.codelen,
                        duration=self.cycle_duration,
                        cycles=self.num_cycles)
            
            self.prep_window.destroy()
            
            self._setup_main_ui()
            
            self._show_next_target()
            
        except ValueError as e:
            self.logger.critical("Некорректные данные")
            messagebox.showerror("Ошибка", f"Некорректные данные: {e}")
    
    def _setup_main_ui(self):
        """Настраивает главное окно (скрытое)"""
        self.root.deiconify()
        self.root.attributes('-fullscreen', True)

        # Добавляем кнопку выхода
        exit_button = tk.Button(
            self.root,
            text="✕ ВЫЙТИ",
            font=('Arial', 12, 'bold'),
            command=self._exit_program,
            bg='#e74c3c',
            fg='white',
            relief='flat',
            padx=20,
            pady=10
        )
        exit_button.place(x=20, y=20)

        window_width = int(self.screen_width * 0.9)
        window_height = int(self.screen_height * 0.9)

        self.root.title("SSVEP BCI Эксперимент")
        self.root.configure(bg='white')

        main_container = tk.PanedWindow(self.root, orient=tk.VERTICAL, bg='white', sashwidth=5)
        main_container.pack(fill=tk.BOTH, expand=True)

        top_frame = tk.Frame(main_container, bg='white')
        main_container.add(top_frame, height=int(window_height * 0.8))

        top_paned = tk.PanedWindow(top_frame, orient=tk.VERTICAL, bg='white', sashwidth=3)
        top_paned.pack(fill=tk.BOTH, expand=True)

        info_frame = tk.Frame(top_paned, bg='white')
        top_paned.add(info_frame, height=int(window_height * 0.8 * 0.1))

        # Информация о текущем символе
        self.current_symbol_label = tk.Label(
            info_frame,
            text="Ожидание целевого символа...",
            font=('Arial', 12),
            bg='white',
            fg='#333333'
        )
        self.current_symbol_label.pack(side=tk.LEFT, padx=20)

        self.experiment_progress = tk.Label(
            info_frame,
            text="",
            font=('Arial', 11),
            bg='white',
            fg='#7f8c8d'
        )
        self.experiment_progress.pack(side=tk.LEFT, padx=20)

        self.flash_indicator = tk.Label(
            info_frame,
            text="○",
            font=('Arial', 14),
            bg='white',
            fg='#95a5a6'
        )
        self.flash_indicator.pack(side=tk.RIGHT, padx=20)

        # Фрейм сетки символов (90% верхней части)
        grid_container = tk.Frame(top_paned, bg='white')
        top_paned.add(grid_container, height=int(window_height * 0.8 * 0.9))
        

        self._create_symbol_grid(grid_container)

        # Нижняя часть: управление и результаты (20% окна)
        bottom_frame = tk.Frame(main_container, bg='white')
        main_container.add(bottom_frame, height=int(window_height * 0.2))

        # Разделяем нижнюю часть
        bottom_paned = tk.PanedWindow(bottom_frame, orient=tk.HORIZONTAL, bg='white', sashwidth=3)
        bottom_paned.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель: настройки и прогресс (40% нижней части)
        left_bottom_frame = tk.Frame(bottom_paned, bg='white')
        bottom_paned.add(left_bottom_frame, width=int(window_width * 0.4))

        # Параметры эксперимента
        tk.Label(
            left_bottom_frame,
            text="ПАРАМЕТРЫ ЭКСПЕРИМЕНТА",
            font=('Arial', 10, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(anchor='w', padx=20, pady=(15, 5))

        # Отображение настроек
        self.settings_label = tk.Label(
            left_bottom_frame,
            text=f"Длительность цикла: {self.cycle_duration} сек | Циклов: {self.num_cycles}",
            font=('Arial', 9),
            bg='white',
            fg='#7f8c8d'
        )
        self.settings_label.pack(anchor='w', padx=20, pady=(0, 5))

        # Прогресс мигания
        self.progress_label = tk.Label(
            left_bottom_frame,
            text=f"Цикл: 0/{self.num_cycles} | Интервал: 0/{self.codelen}",
            font=('Arial', 10),
            bg='white',
            fg='#3498db'
        )
        self.progress_label.pack(anchor='w', padx=20, pady=(10, 5))

        # Статус
        self.status_label = tk.Label(
            left_bottom_frame,
            text="Готов к началу эксперимента",
            font=('Arial', 10),
            bg='white',
            fg='#666666'
        )
        self.status_label.pack(anchor='w', padx=20, pady=(5, 0))

        # Правая панель: результаты (60% нижней части)
        right_bottom_frame = tk.Frame(bottom_paned, bg='white')
        bottom_paned.add(right_bottom_frame, width=int(window_width * 0.6))

        # Заголовок результатов
        tk.Label(
            right_bottom_frame,
            text="РЕЗУЛЬТАТ ВВОДА",
            font=('Arial', 10, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(anchor='w', padx=20, pady=(15, 5))

        # Текстовое поле для результата с прокруткой
        text_container = tk.Frame(right_bottom_frame, bg='white')
        text_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))

        # Полоса прокрутки
        scrollbar = tk.Scrollbar(text_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Текстовое поле
        self.text_display = tk.Text(
            text_container,
            font=('Arial', 12),
            height=3,  # Только 3 строки
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            bg='#f8f9fa',
            fg='#2c3e50',
            relief='solid',
            bd=1
        )
        self.text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text_display.yview)
        
        # Центрируем главное окно
        self.root.update_idletasks()
        x = (self.screen_width - window_width) // 2
        y = (self.screen_height - window_height) // 2
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')
    
    def _create_symbol_grid(self, parent):
        """Создает сетку символов"""
        self.grid_frame = tk.Frame(parent, bg='white')
        self.grid_frame.pack(pady=(0, 20))

        self.grid_frame = tk.Frame(parent, bg='white')
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Рассчитываем оптимальный размер шрифта на основе размера окна
        # Используем 6x6 сетку
        rows = 6
        cols = 6
        
        # Создаем сетку с весами для растяжения
        for i in range(rows):
            self.grid_frame.grid_rowconfigure(i, weight=1)
        for j in range(cols):
            self.grid_frame.grid_columnconfigure(j, weight=1)
        
        self.labels = []
        for i, symbol in enumerate(self.symbols):
            row = i // 6
            col = i % 6
            
            label = tk.Label(
                self.grid_frame,
                text=symbol,
                font=('Arial', 24, 'bold'),
                bg='white',
                fg='#999999',
                width=4,
                height=2,
                relief='flat',
                bd=2
            )
            label.grid(row=row, column=col, padx=3, pady=3, sticky='nsew')
            self.labels.append(label)
    
    def setup_symbols(self):
        """Настройка символов"""
        CG = CodeGen(self.codelen)
        self.symbols = CG.alphabet
        self.patterns = CG.patterns
    
    def _show_next_target(self):
        """Показывает окно с целевым символом для ввода"""
        if self.current_target_index < len(self.target_symbols):
            self.target_symbol = self.target_symbols[self.current_target_index]

            self.send_event_marker("Смотрите на целевой символ и нажмите ПРОБЕЛ", 
                        symbol=self.target_symbol,
                        index=self.current_target_index,
                        total=len(self.target_symbols))
            
            self.current_symbol_label.config(
                text=f"Текущий символ: '{self.target_symbol}' ({self.current_target_index + 1}/{len(self.target_symbols)})"
            )
            
            from app.target_window import TargetWindow
            target_win = TargetWindow(
                self.root,
                self.target_symbol,
                self._on_target_confirmed
            )
            
            self.status_label.config(text="Смотрите на целевой символ и нажмите ПРОБЕЛ", fg='#f39c12')
            
        else:
            self._finish_experiment()
    
    def _on_target_confirmed(self):
        """Вызывается после подтверждения целевого символа (нажатия пробела)"""
        self.send_event_marker("TARGET_CONFIRMED", 
                              symbol=self.target_symbol,
                              index=self.current_target_index)
        self.root.deiconify()
        
        self.status_label.config(text="Мигание... Смотрите на целевой символ", fg='#e74c3c')
        
        self._start_flashing()
    
    def _start_flashing(self):
        """Запускает мигание для текущего символа"""
        if not self.is_running:
            self.send_event_marker("FLASHING_START",
                                symbol=self.target_symbol,
                                duration=self.cycle_duration,
                                cycles=self.num_cycles,
                                codelen=self.codelen)
            self.is_running = True
            self.current_interval = 0
            
            self.flash_thread = threading.Thread(target=self._flash_sequence, daemon=True)
            self.flash_thread.start()
    
    def _flash_sequence(self):
        """Выполняет последовательность мигания с маркерами"""
        for cycle in range(self.num_cycles):
            if not self.is_running:
                break
            
            target_index = self.symbols.index(self.target_symbol) if self.target_symbol in self.symbols else -1

            self.send_event_marker(
                "CYCLE_START", 
                cycle=cycle+1, 
                total_cycles=self.num_cycles,
                target_symbol = self.target_symbol,
                target_index = target_index
            )
            
            for interval in range(self.codelen):
                if not self.is_running:
                    break
                
                states = []
                for i in range(len(self.symbols)):
                    if i < len(self.patterns):
                        states.append(self.patterns[i][interval])
                    else:
                        states.append(0)

                states_str = ''.join(str(s) for s in states)

                self.send_event_marker(
                    "FLASH_INTERVAL_START",
                    interval=interval+1,
                    cycle=cycle+1,
                    target_symbol = self.target_symbol,
                    target_index = target_index,
                    states = states_str,
                    target_state = self.patterns[target_index][interval] if target_index != -1 else 0
                )
                
                # Фаза 1: Основное состояние
                for i, label in enumerate(self.labels):
                    if i < len(self.patterns):
                        if self.patterns[i][interval] == 1:
                            label.config(fg='#000000')  # Черный
                        else:
                            label.config(fg='#999999')  # Серый
                
                time.sleep(self.base_interval * 0.9)
                
                # Фаза 2: Кратковременное отключение
                if self.is_running:
                    for label in self.labels:
                        label.config(fg='#999999')
                    time.sleep(self.base_interval * 0.1)
            
            # Маркер конца цикла
            self.send_event_marker(
                "CYCLE_END", 
                cycle=cycle+1,
                target_symbol = self.target_symbol,
                total_intervals=self.codelen
            )
        if self.is_running:
            self.root.after(0,self._finish_symbol)
    
    def _finish_symbol(self):
        """Завершает ввод текущего символа"""

        self.is_running = False
        
        for label in self.labels:
            label.config(fg='#999999')
        
        self.output_text += self.target_symbol

        self.send_event_marker("SYMBOL_FINISHED",
                        symbol=self.target_symbol,
                        output_text=self.output_text,
                        index=self.current_target_index)
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.output_text)
        self.text_display.see(tk.END)
        
        self.current_target_index += 1
        
        self.status_label.config(text="Символ добавлен. Переход к следующему...", fg='#27ae60')
        
        self.root.withdraw()
        
        self.root.after(1000, self._show_next_target)
    
    def _finish_experiment(self):
        """Завершает эксперимент"""
        self.send_event_marker("EXPERIMENT_END",
                        final_text=self.output_text,
                        total_symbols=len(self.target_symbols))
        self.status_label.config(text="Эксперимент завершен!", fg='#27ae60')
        self.progress_label.config(text="Завершено")

        
        # Показываем сообщение
        messagebox.showinfo(
            "Эксперимент завершен",
            f"Поздравляем! Вы успешно завершили эксперимент.\n\n"
            f"Введенный текст: {self.output_text}\n"
            f"Параметры: {self.num_cycles} циклов по {self.cycle_duration} сек каждый"
        )
        
        # Закрываем приложение
        self.root.quit()