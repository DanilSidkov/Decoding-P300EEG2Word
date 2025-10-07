import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class DataGenerator:
    """Генератор синтетических EEG данных для P300-подобных сигналов.

    Parameters
    ----------
    symbols_dict : Dict[int, str]
        Словарь соответствия индексов символов их буквенным представлениям
    n_trials : int
        Количество попыток на каждый символ
    n_letters : int
        Количество различных символов
    n_ch : int
        Количество каналов EEG

    Attributes
    ----------
    symbols_dict : Dict[int, str]
        Словарь символов
    n_trials : int
        Количество попыток
    n_letters : int
        Количество букв
    n_ch : int
        Количество каналов
    p300_wave : np.ndarray
        Волна P300 компонента
    n100_wave : np.ndarray
        Волна N100 компонента
    dataset_simul : np.ndarray
        Сгенерированный датасет
    targets : List[List[str, int]]
        Список целей
    new_dataset_simul : np.ndarray
        Длинные сгенерированные сигналы

    """

    def __init__(
        self,
        symbols_dict: dict[int, str],
        n_trials: int,
        n_letters: int,
        n_ch: int,
    ) -> None:
        self.symbols_dict = symbols_dict
        self.n_trials = n_trials
        self.n_letters = n_letters
        self.n_ch = n_ch

        self.p300_wave = (
            np.sin(np.arange(-np.pi / 2, np.pi * 2 * 0.75, 0.05)) + 1
        )
        self.n100_wave = (
            -0.5 * np.sin(3 * np.arange(-3 / 2 * np.pi, -5 / 6 * np.pi, 0.05))
            - 0.5
        )

    def generate(self) -> tuple[np.ndarray, list[list[Any]]]:
        """Генерирует синтетические EEG данные.

        Returns
        -------
        Tuple[np.ndarray, List[List[Any]]]
            Кортеж содержащий:
            - dataset_simul: массив EEG данных формы (n_samples, n_channels, seq_length)
            - targets: список целей вида [[буква, номер_попытки], ...]

        """
        self.dataset_simul = []
        self.targets = []
        for i in range(self.n_letters):
            for j in range(self.n_trials):
                n_before = random.randint(90, 150)
                n_after = 232 - n_before

                before_ = np.repeat(0, n_before)
                after_ = np.repeat(0, n_after)

                sample_wave = np.concatenate(
                    [before_, self.n100_wave, self.p300_wave, after_]
                )
                noise = np.random.normal(0, 0.3, len(sample_wave))

                data_temp = [sample_wave + noise]
                for ch in range(1, self.n_ch):
                    wave_amp = random.uniform(0.1, 0.4)
                    noise = np.random.normal(0, 0.3, len(sample_wave))
                    data_temp.append(sample_wave * wave_amp + noise)
                self.dataset_simul.append(np.array(data_temp))
                self.targets.append([self.symbols_dict[i], j])
        self.dataset_simul = np.array(self.dataset_simul)

        return self.dataset_simul, self.targets

    def imshow(self) -> None:
        """Визуализирует сгенерированные данные в виде изображения.
        """
        plt.imshow(
            abs(np.concatenate(self.dataset_simul, axis=1)[:, 10000:12000]),
            aspect="auto",
            interpolation=None,
        )

    def generate_long_signal(self) -> np.ndarray:
        """Генерирует длинные непрерывные EEG сигналы.

        Returns
        -------
        np.ndarray
            Длинный сгенерированный сигнал формы (n_channels, total_length)

        """
        targets2type = list(self.symbols_dict.values()) * 3
        random.shuffle(targets2type)

        add_timepoints = [30, 60]

        n_gazes_min = 3
        n_gazes_max = 6

        self.new_dataset_simul = []

        for letter2type in targets2type:
            n_gazes = random.randint(n_gazes_min, n_gazes_max)

            idx_data = [
                self.targets.index(tar)
                for tar in self.targets
                if tar[0] == letter2type
            ]
            data2add = self.dataset_simul[
                random.sample(idx_data, n_gazes), :, :
            ]
            new_data2add = []
            for i in range(len(data2add)):
                n_timepoints_between_samples = random.randint(
                    add_timepoints[0], add_timepoints[1]
                )
                noise = np.random.normal(
                    0, 0.3, (self.n_ch, n_timepoints_between_samples)
                )
                new_data2add.append(np.append(data2add[i], noise, axis=-1))
            self.new_dataset_simul.append(
                np.concatenate(new_data2add, axis=-1)
            )
        self.new_dataset_simul = np.concatenate(
            self.new_dataset_simul, axis=-1
        )

        return self.new_dataset_simul
