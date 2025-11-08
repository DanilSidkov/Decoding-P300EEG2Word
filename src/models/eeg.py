import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvNet(nn.Module):
    def __init__(self, input_channels=8, num_classes=2, seq_length=62):
        super(DeepConvNet, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(input_channels, 1))
        self.bn1 = nn.BatchNorm2d(25)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True)
        self.drop1 = nn.Dropout(0.5)

        # Block 2
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(50)
        self.elu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True)
        self.drop2 = nn.Dropout(0.5)

        # Block 3
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 5), padding=(0, 2))
        self.bn3 = nn.BatchNorm2d(100)
        self.elu3 = nn.ELU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True)
        self.drop3 = nn.Dropout(0.5)

        # Block 4
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 5), padding=(0, 2))
        self.bn4 = nn.BatchNorm2d(200)
        self.elu4 = nn.ELU()
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True)
        self.drop4 = nn.Dropout(0.5)

        # Classification
        self.flatten = nn.Flatten()

        # Calculate size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_channels, seq_length)
            dummy_output = self.forward_features(dummy_input)
            fc_size = dummy_output.view(-1).shape[0]

        self.fc1 = nn.Linear(fc_size, 100)
        self.elu5 = nn.ELU()
        self.drop5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, num_classes)

    def forward_features(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Block 4
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.elu4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        return x

    def forward(self, x):
        # Input shape: (batch_size, 1, channels, timepoints)
        x = x.unsqueeze(1)

        # Проход через feature extractor
        x = self.forward_features(x)

        # Классификация
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.elu5(x)
        x = self.drop5(x)
        x = self.fc2(x)
        return x

class EEGNet(nn.Module):
    def __init__(self, input_channels=8, num_classes=2, seq_length=62, F1=8, D=2, F2=16, kernel_length1=32, kernel_length2=8, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        
        # Первый блок
        self.conv1 = nn.Conv2d(
            in_channels=1,  # Добавляем измерение канала
            out_channels=F1,
            kernel_size=(1, kernel_length1),
            padding=(0, kernel_length1 // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(input_channels, 1),
            groups=F1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Separable convolution
        self.separable_conv1 = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, kernel_length2),
            padding=(0, kernel_length2 // 2),
            groups=F1 * D,
            bias=False
        )
        self.separable_conv2 = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Полносвязный слой
        self.classifier = nn.Linear(
            in_features=self._calculate_fc_size(input_channels, seq_length, F2),
            out_features=num_classes
        )

    def _calculate_fc_size(self, input_channels, seq_length, F2):
        # Пробный проход для определения размера входа полносвязного слоя
        with torch.no_grad():
            x = torch.zeros(1, 1, input_channels, seq_length)
            x = self._forward_features(x)
            return x.view(1, -1).size(1)

    def _forward_features(self, x):
        # Первый блок
        x = self.conv1(x)
        x = self.bn1(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        # Separable convolution
        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        return x

    def forward(self, x):
        # Добавляем измерение канала (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        
        # Проход через feature extractor
        x = self._forward_features(x)
        
        # Классификация
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x