import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Модуль внимания к каналам (Squeeze-and-Excitation)"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, time = x.size()
        avg_out = self.global_avgpool(x).view(batch, channels)
        weights = self.fc(avg_out).view(batch, channels, 1)
        return x * weights.expand_as(x)


class TemporalAttention(nn.Module):
    """Простой модуль внимания ко временной оси"""

    def __init__(self, seq_length, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(seq_length, seq_length // reduction),
            nn.ELU(),
            nn.Linear(seq_length // reduction, seq_length),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, time = x.size()
        avg_out = self.global_avgpool(x.transpose(1, 2)).view(
            batch, time
        )
        weights = self.fc(avg_out).view(batch, 1, time)
        return x * weights.expand_as(x)


class OptimizedEEGNet1D_v2(nn.Module):
    """Улучшенная версия с оптимальными параметрами для 250 точек"""
    
    def __init__(self, input_channels=8, seq_length=250, num_classes=2):
        super().__init__()
        
        # Временные фильтры - мультимасштабные
        self.temporal_branch_small = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=16, padding=8, groups=input_channels),
            nn.BatchNorm1d(8),
            nn.ELU(),
        )
        self.temporal_branch_medium = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=32, padding=16, groups=input_channels),
            nn.BatchNorm1d(8),
            nn.ELU(),
        )
        self.temporal_branch_large = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=64, padding=32, groups=input_channels),
            nn.BatchNorm1d(8),
            nn.ELU(),
        )
        
        self.temporal_merge = nn.Sequential(
            nn.Conv1d(24, 24, kernel_size=1),
            nn.BatchNorm1d(24),
            nn.ELU(),
            nn.AvgPool1d(2),  # 250 -> 125
            nn.Dropout(0.3),
        )
        
        # Остальные слои остаются прежними...
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(4),  # 125 -> 31
            nn.Dropout(0.3),
        )
        
        self.separable_conv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, padding=8, groups=32),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(2),  # 31 -> 15
            nn.Dropout(0.3),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x):
        x_small = self.temporal_branch_small(x)
        x_medium = self.temporal_branch_medium(x)
        x_large = self.temporal_branch_large(x)
        x = torch.cat([x_small, x_medium, x_large], dim=1)
        x = self.temporal_merge(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


class OptimizedEEGNet1D(nn.Module):
    """Оптимизированная версия EEGNet с улучшенными параметрами"""
    
    def __init__(self, input_channels=8, seq_length=250, num_classes=2):
        super().__init__()
        
        # Временные фильтры оптимального размера
        self.temporal_branch = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=32, padding=16, groups=input_channels),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.3),
        )
        
        # Пространственные фильтры
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.3),
        )
        
        # Сепарабельные свертки
        self.separable_conv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, padding=8, groups=32),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x):
        x = self.temporal_branch(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)