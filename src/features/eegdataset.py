import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset

def compute_dataset_stats(dataset):
    n = 0
    mean = 0.0
    M2 = 0.0
    
    batch_size = min(len(dataset), 1024)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    
    for batch, _ in data_loader:
        # Объединяем все измерения кроме батча
        batch = batch.flatten(1) if batch.dim() > 2 else batch
        batch_np = batch.numpy() if torch.is_tensor(batch) else batch
        batch_flat = batch_np.reshape(-1)
        
        batch_size = len(batch_flat)
        delta = batch_flat - mean
        mean += delta.sum() / (n + batch_size)
        M2 += (delta * (batch_flat - mean)).sum()
        n += batch_size
    
    variance = M2 / n if n > 1 else 0.0
    std = np.sqrt(max(variance, 1e-8))
    
    return mean, std

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
        self.mean = float(mean) if mean is not None else None
        self.std = float(std) if std is not None else None

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
        if self.mean is None:
            return signal
            
        if torch.is_tensor(signal):
            signal_np = signal.numpy()
        else:
            signal_np = signal
            
        mean_val = self.mean.item() if torch.is_tensor(self.mean) else self.mean
        std_val = self.std.item() if torch.is_tensor(self.std) else self.std
            
        epsilon = 1e-6 if std_val > 1e-5 else 1e-3
        
        normalized = (signal_np - mean_val) / (std_val + epsilon)
        normalized = np.clip(normalized, -10, 10)
        
        if torch.is_tensor(signal):
            return torch.from_numpy(normalized)
        return normalized
    
    def __getitem__(self, idx):
        signal = self.data[idx].copy()
        signal = self.normalize_signal(signal)
        
        if self.augment:
            rng = np.random.RandomState(seed=idx + int(torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0))
            if rng.rand() > 0.5:
                signal = self.augment_signal(signal, rng=rng)
            
        return torch.from_numpy(signal).float(), torch.tensor(self.encoded_labels[idx], dtype=torch.long)

    def _time_warp(self, signal, max_warp=0.05, rng=None):
        """Более корректный time warping через ресемплинг"""
        from scipy.signal import resample
        
        if rng is None:
            rng = np.random
        warp_factor = rng.uniform(1-max_warp, 1+max_warp)
        new_length = int(signal.shape[1] * warp_factor)
        
        warped = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            resampled = resample(signal[ch], new_length)
            if new_length >= signal.shape[1]:
                warped[ch] = resampled[:signal.shape[1]]
            else:
                warped[ch, :new_length] = resampled
        return warped

    def _phase_shift(self, signal, max_shift=0.1, rng=None):
        """Частотно-зависимый фазовый сдвиг"""
        if rng is None:
            rng = np.random
            
        fft_signal = np.fft.rfft(signal, axis=1)
        frequencies = np.fft.rfftfreq(signal.shape[1])
        
        phase_shift = rng.uniform(-max_shift, max_shift)
        freq_dependent_shift = phase_shift * np.exp(-frequencies * 10)
        
        phase_shifter = np.exp(1j * 2 * np.pi * freq_dependent_shift)
        fft_signal *= phase_shifter
        
        return np.fft.irfft(fft_signal, n=signal.shape[1], axis=1)
        
    def _amplitude_scale(self, signal, rng=None):
        """Коррелированное масштабирование соседних каналов"""
        n_channels = signal.shape[0]
        
        base_scale = rng.uniform(0.8, 1.2)
        channel_scales = base_scale + rng.normal(0, 0.1, n_channels)
        channel_scales = np.clip(channel_scales, 0.7, 1.5)
        
        from scipy.ndimage import gaussian_filter1d
        channel_scales = gaussian_filter1d(channel_scales, sigma=1.0)
        
        return signal * channel_scales[:, np.newaxis]

    def _add_noise(self, signal, rng=None):
        signal_std = np.std(signal)
        noise_factor = rng.uniform(0.02, 0.08) * signal_std
        noise = rng.normal(0, noise_factor, signal.shape)
        return signal + noise

    def _low_freq_noise(self, signal, rng=None):
        lf_noise = rng.normal(0, 0.05, signal.shape[1])
        for channel in range(signal.shape[0]):
            if torch.rand(1) > 0.7:
                signal[channel] += lf_noise
        return signal

    def _channel_dropout(self, signal, rng=None):
        n_channels_to_drop = rng.randint(
            1, max(2, signal.shape[0] // 4)
        )
        channels_to_drop = rng.choice(
            signal.shape[0], n_channels_to_drop, replace=False
        )
        for channel in channels_to_drop:
            noise_level = np.std(signal[channel]) * 2
            signal[channel] = rng.normal(
                0, noise_level, signal.shape[1]
            )
        return signal
    
    def augment_signal(self, signal: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
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

        aug_pipeline = [
            (rng.rand() > 0.5, lambda x: self._add_noise(x, rng=rng)),
            (rng.rand() > 0.5, lambda x: self._amplitude_scale(x, rng=rng)),
            (rng.rand() > 0.3, lambda x: self._low_freq_noise(x, rng=rng)),
            (rng.rand() > 0.8, lambda x: self._channel_dropout(x, rng=rng)),
            (rng.rand() > 0.5, lambda x: self._time_warp(x, rng=rng)),
            (rng.rand() > 0.7, lambda x: self._phase_shift(x, rng=rng)),
        ]

        for should_apply, aug_func in aug_pipeline:
            if should_apply:
                augmented = aug_func(augmented)
        
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