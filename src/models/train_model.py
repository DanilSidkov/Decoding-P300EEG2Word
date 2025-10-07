import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import DataLoader


class EarlyStopping:
    """Реализация ранней остановки для прекращения обучения при отсутствии улучшений.

    Parameters
    ----------
    patience : int, optional
        Количество эпох без улучшения перед остановкой, по умолчанию 10
    delta : float, optional
        Минимальное изменение для считать улучшением, по умолчанию 0

    """

    def __init__(self, patience: int = 10, delta: float = 0) -> None:
        self.patience = patience
        self.delta = delta
        self.best_score: float | None = None
        self.epochs_no_improve: int = 0
        self.early_stop: bool = False

    def __call__(self, val_loss: float) -> None:
        """Обновляет состояние ранней остановки на основе валидационной ошибки.

        Parameters
        ----------
        val_loss : float
            Валидационная ошибка текущей эпохи

        """
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.epochs_no_improve = 0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 200,
) -> tuple[list[float], list[float], list[float]]:
    """Обучает модель на предоставленных данных.

    Parameters
    ----------
    model : torch.nn.Module
        Модель для обучения
    train_loader : torch.utils.data.DataLoader
        DataLoader тренировочных данных
    val_loader : torch.utils.data.DataLoader
        DataLoader валидационных данных
    num_epochs : int, optional
        Количество эпох обучения, по умолчанию 200

    Returns
    -------
    Tuple[List[float], List[float], List[float]]
        Кортеж содержащий:
        - train_losses: потери на тренировке по эпохам
        - val_losses: потери на валидации по эпохам
        - val_accuracies: точность на валидации по эпохам

    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping = EarlyStopping(patience=15)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []

    for epoch in range(num_epochs):
        # Тренировка
        model.train()
        train_loss = 0.0

        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device)
                labels = labels.squeeze().to(device)

                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Статистика эпохи
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Планировщик обучения и ранняя остановка
        scheduler.step(val_loss)
        early_stopping(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.2f}%")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            print("---")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_losses, val_losses, val_accuracies


def test_model(
    model: nn.Module, test_loader: DataLoader
) -> tuple[list[int], list[int], float]:
    """Тестирует модель на тестовых данных.

    Parameters
    ----------
    model : torch.nn.Module
        Обученная модель для тестирования
    test_loader : torch.utils.data.DataLoader
        DataLoader тестовых данных

    Returns
    -------
    Tuple[List[int], List[int], float]
        Кортеж содержащий:
        - all_predictions: список предсказанных классов
        - all_labels: список истинных классов
        - accuracy: точность классификации в процентах

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.squeeze().to(device)

            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    return all_predictions, all_labels, accuracy


def predict_letter(
    model: nn.Module, signal: np.ndarray, label_encoder: LabelEncoder
) -> tuple[str, float]:
    """Предсказывает класс для одного EEG сигнала.

    Parameters
    ----------
    model : torch.nn.Module
        Обученная модель для предсказания
    signal : np.ndarray
        Входной EEG сигнал формы (channels, seq_length)
    label_encoder : sklearn.preprocessing.LabelEncoder
        Кодировщик меток для преобразования индексов в буквы

    Returns
    -------
    Tuple[str, float]
        Кортеж содержащий:
        - predicted_letter: предсказанная буква
        - confidence: уверность предсказания

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        # Нормализация
        signal_normalized = np.zeros_like(signal)
        for channel in range(signal.shape[0]):
            channel_data = signal[channel]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:
                signal_normalized[channel] = (channel_data - mean) / std
            else:
                signal_normalized[channel] = channel_data - mean

        signal_tensor = (
            torch.FloatTensor(signal_normalized).unsqueeze(0).to(device)
        )
        output = model(signal_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_letter = label_encoder.inverse_transform(
            [predicted_idx.item()]
        )[0]

        return predicted_letter, confidence.item()
