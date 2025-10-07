
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """Dataset для EEG сигналов с поддержкой аугментации.

    Parameters
    ----------
    data : np.ndarray
        Массив EEG сигналов формы (n_samples, n_channels, seq_length)
    targets : List[Tuple[str, int]]
        Список целей, где каждый элемент - кортеж (метка, номер_попытки)
    augment : bool, optional
        Флаг включения аугментации данных, по умолчанию True

    Attributes
    ----------
    data : np.ndarray
        Исходные данные EEG сигналов
    labels : List[str]
        Список меток классов
    label_encoder : LabelEncoder
        Кодировщик меток для преобразования строк в числовые индексы
    encoded_labels : np.ndarray
        Закодированные числовые метки
    augment : bool
        Флаг использования аугментации

    """

    def __init__(
        self,
        data: np.ndarray,
        targets: list[tuple[str, int]],
        augment: bool = True,
    ) -> None:
        self.data = data
        self.labels = [item[0] for item in targets]

        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        self.augment = augment

    def __len__(self) -> int:
        """Возвращает количество элементов в датасете.

        Returns
        -------
        int
            Количество элементов в датасете

        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Получает элемент по индексу.

        Parameters
        ----------
        idx : int
            Индекс элемента

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Кортеж содержащий:
            - signal: тензор EEG сигнала формы (n_channels, seq_length)
            - label: тензор метки класса формы (1,)

        """
        signal = self.data[idx]

        signal = self.normalize_channel_wise(signal)

        if self.augment and torch.rand(1) > 0.5:
            signal = self.augment_signal(signal)

        signal = torch.FloatTensor(signal)

        label = torch.LongTensor([self.encoded_labels[idx]])

        return signal, label

    def normalize_channel_wise(self, signal: np.ndarray) -> np.ndarray:
        """Нормализует EEG сигнал по каждому каналу отдельно.

        Parameters
        ----------
        signal : np.ndarray
            Входной сигнал формы (n_channels, seq_length)

        Returns
        -------
        np.ndarray
            Нормализованный сигнал формы (n_channels, seq_length)

        """
        normalized = np.zeros_like(signal)
        for channel in range(signal.shape[0]):
            channel_data = signal[channel]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:
                normalized[channel] = (channel_data - mean) / std
            else:
                normalized[channel] = channel_data - mean
        return normalized

    def augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """Применяет аугментацию к EEG сигналу.

        Parameters
        ----------
        signal : np.ndarray
            Исходный сигнал формы (n_channels, seq_length)

        Returns
        -------
        np.ndarray
            Аугментированный сигнал формы (n_channels, seq_length)

        """
        augmented = signal.copy()
        # Добавим гауссовский шум
        if torch.rand(1) > 0.5:
            noise_factor = 0.01
            noise = np.random.normal(0, noise_factor, signal.shape)
            augmented += noise
        # Случайное масштабирование амплитуды
        if torch.rand(1) > 0.5:
            scale_factor = np.random.uniform(0.8, 1.2)
            augmented *= scale_factor
        # Случайный сдвиш по времени
        if torch.rand(1) > 0.5:
            shift = np.random.randint(-10, 10)
            augmented = np.roll(augmented, shift, axis=1)

        return augmented
