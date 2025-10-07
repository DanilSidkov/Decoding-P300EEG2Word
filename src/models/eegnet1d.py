from torch import Tensor, nn


class EEGNet1D(nn.Module):
    """Одномерная сверточная нейронная сеть для классификации EEG сигналов.

    Parameters
    ----------
    input_channels : int, optional
        Количество входных каналов EEG, по умолчанию 8
    seq_length : int, optional
        Длина временной последовательности, по умолчанию 400
    num_classes : int, optional
        Количество классов для классификации, по умолчанию 10

    Attributes
    ----------
    conv1 : torch.nn.Sequential
        Первый сверточный блок
    conv2 : torch.nn.Sequential
        Второй сверточный блок
    conv3 : torch.nn.Sequential
        Третий сверточный блок
    classifier : torch.nn.Sequential
        Классификатор на полносвязных слоях

    """

    def __init__(
        self,
        input_channels: int = 8,
        seq_length: int = 400,
        num_classes: int = 10,
    ) -> None:
        super(EEGNet1D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )

        self.after_conv_size = 50

        self.classifier = nn.Sequential(
            nn.Linear(64 * self.after_conv_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Прямой проход сети.

        Parameters
        ----------
        x : torch.Tensor
            Входной тензор формы (batch_size, input_channels, seq_length)

        Returns
        -------
        torch.Tensor
            Выходной тензор формы (batch_size, num_classes)

        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
