import logging
import logging.handlers
from pathlib import Path
import datetime
from typing import Optional

import colorlog


class MicrosecondFormatter(colorlog.ColoredFormatter):
    """Кастомный форматтер с поддержкой микросекунд"""
    
    def formatTime(self, record, datefmt=None):
        """
        Переопределяем метод formatTime для поддержки микросекунд
        """
        ct = self.converter(record.created)
        if datefmt:
            if "%f" in datefmt:
                # Используем datetime для микросекунд
                dt = datetime.datetime.fromtimestamp(record.created)
                # Заменяем %f на фактическое значение микросекунд
                formatted = dt.strftime(datefmt.replace("%f", str(dt.microsecond).zfill(6)))
                return formatted
            else:
                # Стандартное форматирование
                return time.strftime(datefmt, ct)
        else:
            # Стандартное форматирование без datefmt
            t = time.strftime(self.default_time_format, ct)
            return t


def setup_logger(logger: logging.Logger, file_name: str = "logging") -> None:
    """Настраивает логгер с цветным выводом в консоль и записью в файл.

    Parameters
    ----------
    logger : logging.Logger
        Объект логгера для настройки
    file_name : str, optional
        Имя файла для записи логов (без расширения), по умолчанию "logging"

    Notes
    -----
    Создает папку logs в директории .app текущего проекта.

    """
    formatter = MicrosecondFormatter(
        "%(white)s%(asctime)s - %(funcName)s:%(lineno)d - "
        "%(log_color)s%(levelname) -8s%(reset)s%(cyan)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S.%f",  # Теперь поддерживает микросекунды
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        },
    )
    logger.setLevel(logging.DEBUG)

    # Console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logger.level)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # File handler - создаем путь к .app/logs
    current_file_path = Path(__file__).resolve()
    
    # Находим директорию .app (где находится get_logger.py)
    app_dir = current_file_path.parent
    
    # Создаем путь к папке logs внутри .app
    logs_dir = app_dir / "logs"
    logs_dir.mkdir(exist_ok=True)  # Создаем папку, если её нет
    
    log_file = logs_dir / f"{file_name}.log"
    
    f_handler = logging.FileHandler(log_file, encoding='utf-8')
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)