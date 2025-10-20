import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """Improved EEG dataset with neuro-informed augmentation without data leakage.

    Parameters
    ----------
    data : np.ndarray
        EEG signals array of shape (n_samples, n_channels, seq_length)
    targets : List[Tuple[str, int]]
        List of targets where each element is (label, trial_number)
    augment : bool, optional
        Flag to enable data augmentation, default True
    augmentation_config : dict, optional
        Configuration parameters for augmentation

    Attributes
    ----------
    data : np.ndarray
        Original EEG signals data
    labels : List[str]
        List of class labels
    label_encoder : LabelEncoder
        Label encoder for converting strings to numeric indices
    encoded_labels : np.ndarray
        Encoded numeric labels
    augment : bool
        Flag for using augmentation
    config : dict
        Augmentation configuration

    """

    def __init__(
        self,
        data: np.ndarray,
        targets: List[Tuple[str, int]],
        augment: bool = True,
        augmentation_config: Optional[dict] = None,
    ) -> None:
        self.data = data
        self.labels = [item[0] for item in targets]

        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        self.augment = augment

        self.config = {
            "max_amplitude_change": 1.8,
        }
        if augmentation_config:
            self.config.update(augmentation_config)

        self.channel_groups = {
            "frontal": [0, 1],  # FZ, CZ
            "parietal_left": [2],  # P3
            "parietal_center": [3],  # PZ
            "parietal_right": [4],  # P4
            "occipito_parietal_left": [5],  # PO7
            "occipito_parietal_right": [6],  # PO8
            "occipital": [7],  # OZ
        }

        self.functional_pathways = {
            "visual_attention": [
                5,
                6,
                7,
                0,
                1,
            ],  # PO7,PO8,OZ -> FZ,CZ (visual attention)
            "sensory_integration": [
                2,
                3,
                4,
                1,
            ],  # P3,PZ,P4 -> CZ (sensory integration)
            "cross_hemisphere": [
                2,
                4,
                5,
                6,
            ],  # P3-P4, PO7-PO8 (cross-hemisphere)
        }

        self.frequency_bands = {
            "delta": (0.5, 4, 0.1, 0.3),
            "theta": (4, 8, 0.1, 0.4),
            "alpha": (8, 13, 0.2, 0.5),
            "beta": (13, 30, 0.1, 0.3),
            "gamma": (30, 45, 0.05, 0.2),
        }

    def neuro_informed_augmentation(self, signal: np.ndarray) -> np.ndarray:
        augmented = signal.copy()

        if random.random() > 0.5:
            augmentation_type = random.choice(
                ["artifact", "connectivity", "rhythm", "combined"]
            )

            if augmentation_type == "artifact":
                augmented = self.add_functional_artifacts(augmented)
            elif augmentation_type == "connectivity":
                augmented = self.add_functional_connectivity(augmented)
            elif augmentation_type == "rhythm":
                augmented = self.rhythm_specific_augmentation(augmented)
            else:
                if random.random() > 0.5:
                    augmented = self.add_adaptive_gaussian_noise(augmented)
                if random.random() > 0.5:
                    augmented = self.neuro_informed_amplitude_scaling(
                        augmented
                    )

        return augmented

    def add_functional_artifacts(self, signal: np.ndarray) -> np.ndarray:
        """Adding artifacts specific to functional systems"""
        augmented = signal.copy()

        if random.random() > 0.8:
            blink_duration = random.randint(3, 8)
            blink_start = random.randint(50, 150)
            blink_amplitude = random.uniform(0.1, 0.3)

            frontal_channels = self.channel_groups["frontal"]
            for channel in frontal_channels:
                blink_shape = self._generate_blink_shape(
                    blink_duration, blink_amplitude
                )
                end_idx = min(blink_start + blink_duration, signal.shape[1])
                actual_duration = end_idx - blink_start
                augmented[channel, blink_start:end_idx] += blink_shape[
                    :actual_duration
                ] * (1.0 if channel == 0 else 0.6)

        if random.random() > 0.9:
            saccade_amplitude = random.uniform(0.1, 0.3)
            occipito_parietal = (
                self.channel_groups["occipito_parietal_left"]
                + self.channel_groups["occipito_parietal_right"]
            )
            for channel in occipito_parietal:
                augmented[channel] += np.random.normal(
                    0, saccade_amplitude, signal.shape[1]
                )

        if random.random() > 0.85:
            muscle_amplitude = random.uniform(0.2, 0.5)
            muscle_channels = random.choice(
                self.channel_groups["parietal_left"]
                + self.channel_groups["parietal_right"]
            )
            high_freq_noise = np.random.normal(
                0, muscle_amplitude, signal.shape[1]
            )

            b, a = butter(
                3, [20 / 125, 60 / 125], btype="bandpass"
            )
            high_freq_noise = filtfilt(b, a, high_freq_noise)
            augmented[muscle_channels] += high_freq_noise

        return augmented

    def _generate_blink_shape(
        self, duration: int, amplitude: float
    ) -> np.ndarray:
        """Generate blink artifact shape"""
        t = np.linspace(0, 1, duration)
        blink = np.exp(-((t - 0.3) ** 2) / 0.1) - 0.5 * np.exp(
            -((t - 0.7) ** 2) / 0.2
        )
        return blink * amplitude

    def add_functional_connectivity(self, signal: np.ndarray) -> np.ndarray:
        """Modeling functional connectivity between regions"""
        augmented = signal.copy()

        for pathway, channels in self.functional_pathways.items():
            if random.random() > 0.7:
                network_coherence = random.uniform(0.8, 1.2)
                temporal_pattern = np.random.normal(0, 0.1, signal.shape[1])

                for channel in channels:
                    augmented[channel] = (
                        augmented[channel] * network_coherence
                        + temporal_pattern * 0.1
                    )

        if random.random() > 0.6:
            left_parietal = self.channel_groups["parietal_left"]
            right_parietal = self.channel_groups["parietal_right"]

            hemisphere_balance = random.uniform(0.95, 1.05)
            for channel in left_parietal:
                augmented[channel] *= hemisphere_balance
            for channel in right_parietal:
                augmented[channel] *= 2.0 - hemisphere_balance

        return augmented

    def rhythm_specific_augmentation(self, signal: np.ndarray) -> np.ndarray:
        """Augmentation considering EEG frequency bands"""
        augmented = signal.copy()

        for band, (
            f_low,
            f_high,
            min_amp,
            max_amp,
        ) in self.frequency_bands.items():
            if random.random() > 0.8:
                if band == "alpha":
                    channels = (
                        self.channel_groups["occipital"]
                        + self.channel_groups["occipito_parietal_left"]
                        + self.channel_groups["occipito_parietal_right"]
                    )
                elif band == "theta":
                    channels = self.channel_groups["frontal"]
                else:
                    channels = list(range(signal.shape[0]))

                for channel in channels:
                    band_amplitude = random.uniform(
                        1.0 + min_amp, 1.0 + max_amp
                    )
                    augmented[channel] *= band_amplitude

        return augmented

    def neuro_informed_amplitude_scaling(
        self, signal: np.ndarray
    ) -> np.ndarray:
        """Amplitude scaling considering typical amplitudes in different regions"""
        augmented = signal.copy()

        for region, channels in self.channel_groups.items():
            if random.random() > 0.3:
                if region in ["frontal", "parietal_center"]:
                    scale_range = (0.8, 1.4)
                elif region in ["occipital"]:
                    scale_range = (0.9, 1.2)
                else:
                    scale_range = (0.85, 1.3)

                region_scale = random.uniform(scale_range[0], scale_range[1])
                for channel in channels:
                    augmented[channel] *= region_scale

        return augmented

    def normalize_channel_wise(self, signal: np.ndarray) -> np.ndarray:
        """Improved normalization considering neuroanatomy"""
        normalized = np.zeros_like(signal)

        for channel in range(signal.shape[0]):
            channel_data = signal[channel]

            if channel in self.channel_groups["frontal"]:
                normalized[channel] = self.robust_normalize(channel_data)
            elif channel in self.channel_groups["occipital"]:
                normalized[channel] = self.rhythm_preserving_normalize(
                    channel_data
                )
            else:
                normalized[channel] = self.standard_normalize(channel_data)

        return normalized

    def rhythm_preserving_normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalization preserving oscillatory activity"""
        data = data - np.median(data)
        mad = np.median(np.abs(data))
        if mad > 0:
            return data / (mad * 1.4826)
        else:
            return data / (np.std(data) + 1e-8)

    def robust_normalize(self, data: np.ndarray) -> np.ndarray:
        """Robust normalization for frontal channels"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad > 0:
            return (data - median) / (mad * 1.4826)
        else:
            return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def standard_normalize(self, data: np.ndarray) -> np.ndarray:
        """Standard normalization"""
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def add_adaptive_gaussian_noise(self, signal: np.ndarray) -> np.ndarray:
        """Adaptive Gaussian noise"""
        signal_std = np.std(signal)
        noise_factor = random.uniform(0.02, 0.08) * signal_std
        noise = np.random.normal(0, noise_factor, signal.shape)
        return signal + noise

    def validate_augmentation_quality(
        self, original: np.ndarray, augmented: np.ndarray
    ) -> bool:
        """Validate that augmentation doesn't break physiological plausibility"""
        original_std = np.std(original)
        augmented_std = np.std(augmented)

        if (
            augmented_std > self.config["max_amplitude_change"] * original_std
            or augmented_std < 0.3 * original_std
        ):
            return False

        original_corr = np.corrcoef(original.reshape(original.shape[0], -1))
        augmented_corr = np.corrcoef(augmented.reshape(augmented.shape[0], -1))
        corr_diff = np.mean(np.abs(original_corr - augmented_corr))

        if corr_diff > 0.5:
            return False

        return True

    def __len__(self) -> int:
        """Return the number of items in dataset"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset with optional augmentation"""
        signal = self.data[idx]
        signal = self.normalize_channel_wise(signal)

        if self.augment and random.random() > 0.5:
            augmented_signal = self.neuro_informed_augmentation(signal)
            if self.validate_augmentation_quality(signal, augmented_signal):
                signal = augmented_signal

        signal = torch.FloatTensor(signal)
        label = torch.LongTensor([self.encoded_labels[idx]])

        return signal, label

    def get_augmentation_stats(self) -> dict:
        """Get statistics about augmentation parameters"""
        return {
            "total_samples": len(self.data),
            "augmentation_enabled": self.augment,
            "augmentation_config": self.config,
            "channel_groups": self.channel_groups,
            "functional_pathways": list(self.functional_pathways.keys()),
            "frequency_bands": list(self.frequency_bands.keys()),
        }


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
