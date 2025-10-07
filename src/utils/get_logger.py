import logging
import logging.handlers
from pathlib import Path

import colorlog


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
    Создает папку logs в родительской директории третьего уровня от текущего файла.

    """
    formatter = colorlog.ColoredFormatter(
        "%(white)s%(asctime)s - %(funcName)s:%(lineno)d - "
        "%(log_color)s%(levelname) -8s%(reset)s%(cyan)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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

    # File handler
    log_file = (
        Path(__file__).resolve().parents[3] / "logs" / f"{file_name}.log"
    )
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
