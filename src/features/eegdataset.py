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
        self.current_idx = idx
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

            median = np.median(channel_data)
            mad = np.median(np.abs(channel_data - median))

            if mad > 0:
                normalized[channel] = (channel_data - median) / (mad * 1.4826)
            else:
                std = np.std(channel_data)
                if std > 0:
                    normalized[channel] = (
                        channel_data - np.mean(channel_data)
                    ) / std
                else:
                    normalized[channel] = channel_data - np.mean(channel_data)

        return normalized

    def augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """Улучшенная аугментация для ЭЭГ сигналов.

        Parameters
        ----------
        signal : np.ndarray
            Исходный сигнал формы (n_channels, seq_length)

        Returns
        -------
        np.ndarray
            Аугментированный сигнал формы (n_channels, seq_length)

        Notes
        -----
        Все аугментации имитируют реальные артефакты ЭЭГ:

        Движения глаз (low-frequency noise)
        Плохой контакт электродов (channel dropout)
        Изменения импеданса (amplitude scaling)
        Мышечные артефакты (adaptive noise)

        """
        augmented = signal.copy()

        if torch.rand(1) > 0.5:
            signal_std = np.std(signal)
            noise_factor = np.random.uniform(0.02, 0.08) * signal_std
            noise = np.random.normal(0, noise_factor, signal.shape)
            augmented += noise

        if torch.rand(1) > 0.5:
            for channel in range(augmented.shape[0]):
                channel_scale = np.random.uniform(0.7, 1.5)
                augmented[channel] *= channel_scale

        if torch.rand(1) > 0.3:
            lf_noise = np.random.normal(0, 0.05, signal.shape[1])
            for channel in range(augmented.shape[0]):
                if torch.rand(1) > 0.7:
                    augmented[channel] += lf_noise

        if torch.rand(1) > 0.8:
            n_channels_to_drop = np.random.randint(
                1, max(2, signal.shape[0] // 4)
            )
            channels_to_drop = np.random.choice(
                signal.shape[0], n_channels_to_drop, replace=False
            )
            for channel in channels_to_drop:
                noise_level = np.std(augmented[channel]) * 2
                augmented[channel] = np.random.normal(
                    0, noise_level, signal.shape[1]
                )

        if torch.rand(1) > 0.5:
            time_warp_factor = np.random.uniform(0.9, 1.1)
            original_length = signal.shape[1]
            new_length = int(original_length * time_warp_factor)

            from scipy.interpolate import interp1d

            x_original = np.linspace(0, 1, original_length)
            x_new = np.linspace(0, 1, new_length)

            for channel in range(augmented.shape[0]):
                interpolator = interp1d(
                    x_original,
                    augmented[channel],
                    kind="linear",
                    fill_value="extrapolate",
                )
                warped = interpolator(x_new)

                if new_length > original_length:
                    augmented[channel] = warped[:original_length]
                else:
                    padded = np.zeros(original_length)
                    padded[:new_length] = warped
                    augmented[channel] = padded

        if torch.rand(1) > 0.7:
            phase_shift = np.random.uniform(-0.2, 0.2)
            fft_signal = np.fft.fft(augmented, axis=1)
            frequencies = np.fft.fftfreq(signal.shape[1])
            phase_shifter = np.exp(1j * 2 * np.pi * phase_shift * frequencies)
            fft_signal *= phase_shifter
            augmented = np.real(np.fft.ifft(fft_signal, axis=1))

        return augmented
