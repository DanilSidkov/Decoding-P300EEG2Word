import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset

def compute_dataset_stats(dataset):
    """Вычисляет mean и std по тренировочным данным для нормализации"""
    data_loader = DataLoader(dataset, batch_size=len(dataset))
    all_data, all_labels = next(iter(data_loader))
    
    # Вычисляем статистики и преобразуем в Python float
    mean_val = all_data.mean().item()
    std_val = all_data.std().item()
    
    return mean_val, std_val

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
        mean=None, 
        std=None
    ) -> None:
        self.data = data
        self.labels = [item[0] for item in targets]

        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        self.augment = augment
        self.mean = mean
        self.std = std

    @property
    def targets(self):
        """Возвращает список targets в формате для совместимости"""
        return list(zip(self.labels, range(len(self.labels))))
        
    def __len__(self) -> int:
        """Возвращает количество элементов в датасете.

        Returns
        -------
        int
            Количество элементов в датасете

        """
        return len(self.data)

    def normalize_signal(self, signal):
        """Стандартная нормализация"""
        if self.mean is not None and self.std is not None:
            # Преобразуем torch tensor в numpy если нужно
            if torch.is_tensor(signal):
                signal_np = signal.numpy()
            else:
                signal_np = signal
                
            # Преобразуем статистики в numpy если они torch tensor
            if torch.is_tensor(self.mean):
                mean_val = self.mean.item()
                std_val = self.std.item()
            else:
                mean_val = self.mean
                std_val = self.std
                
            normalized = (signal_np - mean_val) / (std_val + 1e-8)
            return normalized
        return signal
    
    def __getitem__(self, idx):
        signal = self.data[idx]
        signal = self.normalize_signal(signal)
        
        if self.augment and torch.rand(1) > 0.5:
            signal = self.augment_signal(signal)
            
        signal = torch.FloatTensor(signal)
        label = torch.LongTensor([self.encoded_labels[idx]])
        return signal, label

    """
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        Получает элемент по индексу.

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

        
        self.current_idx = idx
        signal = self.data[idx]
        signal = self.normalize_channel_wise(signal)

        if self.augment and torch.rand(1) > 0.5:
            signal = self.augment_signal(signal)

        signal = torch.FloatTensor(signal)
        label = torch.LongTensor([self.encoded_labels[idx]])

        return signal, label
    """
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


def load_and_prepare_data(file_paths, test_subject=None):
    """Загружает и подготавливает данные всех субъектов

    Parameters
    ----------
    file_paths : list
        Список путей к файлам .mat
    test_subject_idx : int, optional
        Индекс субъекта для тестирования (если None, используется общее разделение)

    Returns
    -------
    tuple
        (train_dataset, val_dataset, test_dataset, label_encoder)

    """
    all_subjects_data = []
    all_subjects_targets = []
    test_subject_idx = test_subject - 1

    for i, file_path in enumerate(file_paths):
        print(f"Загрузка данных субъекта {i+1}: {file_path}")
        data = loadmat(file_path)

        tar_eeg = data["tar_eeg"]
        nontar_eeg = data["nontar_eeg"]

        tar_labels = [("target", j) for j in range(len(tar_eeg))]
        nontar_labels = [("nontarget", j) for j in range(len(nontar_eeg))]

        subject_data = np.concatenate([tar_eeg, nontar_eeg], axis=0)
        subject_targets = tar_labels + nontar_labels

        subject_data = subject_data.transpose(0, 2, 1)

        all_subjects_data.append(subject_data)
        all_subjects_targets.append(subject_targets)

    if test_subject_idx is not None:
        test_data = all_subjects_data[test_subject_idx]
        test_targets = all_subjects_targets[test_subject_idx]

        train_val_data = np.concatenate(
            [
                d
                for j, d in enumerate(all_subjects_data)
                if j != test_subject_idx
            ],
            axis=0,
        )
        train_val_targets = [
            item
            for j, sublist in enumerate(all_subjects_targets)
            if j != test_subject_idx
            for item in sublist
        ]

        train_data, val_data, train_targets, val_targets = train_test_split(
            train_val_data,
            train_val_targets,
            test_size=0.2,
            random_state=42,
            stratify=[item[0] for item in train_val_targets],
        )
        
        train_dataset = EEGDataset(train_data, train_targets, augment=False)
        val_dataset = EEGDataset(val_data, val_targets, augment=False)
        test_dataset = EEGDataset(test_data, test_targets, augment=False)

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            train_dataset.label_encoder,
        )
    else:
        all_data = np.concatenate(all_subjects_data, axis=0)
        all_targets = [
            item for sublist in all_subjects_targets for item in sublist
        ]

        train_data, test_data, train_targets, test_targets = train_test_split(
            all_data,
            all_targets,
            test_size=0.2,
            random_state=42,
            stratify=[item[0] for item in all_targets],
        )

        train_data, val_data, train_targets, val_targets = train_test_split(
            train_data,
            train_targets,
            test_size=0.25,
            random_state=42,
            stratify=[item[0] for item in train_targets],
        )

        train_dataset = EEGDataset(train_data, train_targets, augment=False)
        val_dataset = EEGDataset(val_data, val_targets, augment=False)
        test_dataset = EEGDataset(test_data, test_targets, augment=False)

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            train_dataset.label_encoder,
        )