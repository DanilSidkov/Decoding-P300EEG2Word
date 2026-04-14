"""P300 EEGNet — модель для бинарной классификации P300 эпох.

Адаптированная EEGNet архитектура с корректной поддержкой
произвольного числа каналов (12, 18, 34 и т.д.).
Исправлен баг grouped conv из оригинального eegnet1d.py.
"""

from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation attention по каналам."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ELU(),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class TemporalAttention(nn.Module):
    """Attention по временной оси."""

    def __init__(self, seq_length: int, reduction: int = 8) -> None:
        super().__init__()
        reduced = max(seq_length // reduction, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(seq_length, reduced),
            nn.ELU(),
            nn.Linear(reduced, seq_length),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.size()
        # Средний по каналам -> attention по времени
        w = self.pool(x.transpose(1, 2)).view(b, t)
        w = self.fc(w).view(b, 1, t)
        return x * w


class P300EEGNet(nn.Module):
    """EEGNet-подобная модель для P300 классификации.

    Двойные темпоральные ветки (малый + большой kernel) ->
    depthwise spatial conv -> separable conv -> classifier.

    Поддерживает произвольное число каналов (12, 18, 34 и т.д.).

    Parameters
    ----------
    n_channels : int
        Число EEG каналов
    n_times : int
        Число временных точек (после ресемплинга)
    n_classes : int
        Число классов (2 для target/nontarget)
    F1 : int
        Число темпоральных фильтров на ветку
    D : int
        Depth multiplier для spatial conv
    dropout : float
        Dropout rate

    """

    def __init__(
        self,
        n_channels: int = 12,
        n_times: int = 250,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        F2 = F1 * D  # каналов после spatial conv

        # --- Двойные темпоральные ветки ---
        # Depthwise temporal: каждый канал обрабатывается независимо
        # out_channels = in_channels (depthwise), затем pointwise до F1
        self.temporal_small = nn.Sequential(
            nn.Conv1d(
                n_channels, n_channels,
                kernel_size=16, padding="same",
                groups=n_channels, bias=False,
            ),
            nn.Conv1d(n_channels, F1, kernel_size=1, bias=False),
            nn.BatchNorm1d(F1),
            nn.ELU(),
        )

        self.temporal_large = nn.Sequential(
            nn.Conv1d(
                n_channels, n_channels,
                kernel_size=64, padding="same",
                groups=n_channels, bias=False,
            ),
            nn.Conv1d(n_channels, F1, kernel_size=1, bias=False),
            nn.BatchNorm1d(F1),
            nn.ELU(),
        )

        # Merge двух веток: 2*F1 -> 2*F1
        merged_channels = 2 * F1
        self.temporal_merge = nn.Sequential(
            nn.Conv1d(merged_channels, merged_channels, kernel_size=1),
            nn.BatchNorm1d(merged_channels),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(dropout),
        )

        seq_after_merge = n_times // 2
        self.temporal_att = TemporalAttention(seq_after_merge)

        # --- Spatial convolution ---
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(merged_channels, F2, kernel_size=1),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(dropout),
        )

        self.channel_att = ChannelAttention(F2, reduction=4)

        # --- Separable convolution ---
        F3 = F2 * 2
        self.separable_conv = nn.Sequential(
            nn.Conv1d(F2, F2, kernel_size=16, padding="same", groups=F2, bias=False),
            nn.Conv1d(F2, F3, kernel_size=1, bias=False),
            nn.BatchNorm1d(F3),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(dropout),
        )

        # --- Classifier ---
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(F3, F3 // 2),
            nn.BatchNorm1d(F3 // 2),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(F3 // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            shape (batch, n_channels, n_times)

        Returns
        -------
        torch.Tensor
            shape (batch, n_classes) — logits

        """
        # Двойные темпоральные ветки
        x_small = self.temporal_small(x)
        x_large = self.temporal_large(x)
        x = torch.cat([x_small, x_large], dim=1)

        x = self.temporal_merge(x)
        x = self.temporal_att(x)

        x = self.spatial_conv(x)
        x = self.channel_att(x)

        x = self.separable_conv(x)

        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)
