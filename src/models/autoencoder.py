import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class EEGAutoencoder(nn.Module):
    """Автоэнкодер для EEG сигналов с понижением размерности"""
    
    def __init__(self, input_channels=8, seq_length=62, embedding_dim=32):
        super(EEGAutoencoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 6, kernel_size=3, padding=1),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 62 -> 31
            
            nn.Conv1d(6, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 31 -> 15
            
            nn.Conv1d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # 15 -> 8
            
            nn.Flatten(),
            nn.Linear(4 * 8, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, embedding_dim)
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 4 * 8),
            nn.ReLU(),
            
            nn.Unflatten(1, (4, 8)),
            
            nn.Upsample(scale_factor=2),
            nn.Conv1d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            
            nn.Upsample(size=31),
            nn.Conv1d(4, 6, kernel_size=3, padding=1),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            
            nn.Upsample(size=62),
            nn.Conv1d(6, input_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return reconstructed, embedding
    
    def encode(self, x):
        """Только кодирование - для извлечения признаков"""
        return self.encoder(x)

def extract_features(autoencoder, dataloader, device):
    """Извлекает эмбеддинги из данных"""
    autoencoder.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            embeddings = autoencoder.encode(signals)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return X, y

class AutoencoderClassifier(nn.Module):
    """Комбинированная модель: автоэнкодер + классификатор"""
    
    def __init__(self, autoencoder, num_classes=2, classifier_dropout=0.4):
        super(AutoencoderClassifier, self).__init__()
        self.autoencoder = autoencoder
        self.classifier = nn.Sequential(
            nn.Linear(autoencoder.encoder[-1].out_features, 8),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(4, num_classes)
        )
    
    def forward(self, x):
        # Используем только энкодер для классификации
        embedding = self.autoencoder.encode(x)
        classification = self.classifier(embedding)
        return classification
    
    def reconstruct(self, x):
        """Только реконструкция"""
        return self.autoencoder(x)

def train_autoencoder(model, train_loader, val_loader, num_epochs=100):
    """Обучение только автоэнкодера"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        for signals, _ in train_loader:
            signals = signals.to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(signals)
            loss = criterion(reconstructed, signals)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for signals, _ in val_loader:
                signals = signals.to(device)
                reconstructed, _ = model(signals)
                loss = criterion(reconstructed, signals)
                val_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def visualize_embeddings(model, dataloader, title="Эмбеддинги EEG сигналов"):
    """Визуализация эмбеддингов с помощью t-SNE"""
    device = next(model.parameters()).device
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            embeddings = model.autoencoder.encode(signals)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)
    
    # Применяем t-SNE для визуализации
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=all_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Класс')
    plt.title(title)
    plt.xlabel('t-SNE компонента 1')
    plt.ylabel('t-SNE компонента 2')
    plt.show()
    
    return embeddings_2d, all_labels