from sklearn.neighbors import NearestNeighbors
import numpy as np
from features.eegdataset import EEGDataset
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import torch
from models.train_model import train_model

class BorderlineSMOTE1D:
    """Borderline-SMOTE адаптированный для EEG временных рядов"""
    
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        # X shape: (n_samples, n_channels, seq_length)
        # y: labels (0 - nontarget, 1 - target)
        
        X_flat = X.reshape(X.shape[0], -1)  # flatten для SMOTE
        minority_mask = (y == 1)
        minority_samples = X_flat[minority_mask]
        majority_samples = X_flat[~minority_mask]
        
        if len(minority_samples) == 0:
            return X, y
            
        # Находим пограничные minority samples
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors)
        nbrs.fit(X_flat)
        
        danger_set = []
        for i, sample_idx in enumerate(np.where(minority_mask)[0]):
            # Находим соседей
            distances, indices = nbrs.kneighbors([X_flat[sample_idx]])
            neighbor_labels = y[indices[0]]
            
            # Если больше половины соседей - majority, добавляем в danger set
            if np.sum(neighbor_labels == 0) >= len(neighbor_labels) // 2:
                danger_set.append(i)
        
        # Генерируем синтетические samples
        n_to_generate = len(X_flat) - 2 * len(minority_samples)
        if n_to_generate <= 0 or len(danger_set) == 0:
            return X, y
            
        synthetic_samples = []
        for _ in range(n_to_generate):
            # Выбираем случайный danger sample
            danger_idx = np.random.choice(danger_set)
            base_sample = minority_samples[danger_idx]
            
            # Находим k nearest neighbors среди minority class
            minority_nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(minority_samples)))
            minority_nbrs.fit(minority_samples)
            distances, indices = minority_nbrs.kneighbors([base_sample])
            
            # Выбираем случайного соседа
            neighbor_idx = np.random.choice(indices[0])
            neighbor_sample = minority_samples[neighbor_idx]
            
            # Создаем синтетический sample
            diff = neighbor_sample - base_sample
            gap = np.random.uniform(0, 1)
            synthetic = base_sample + gap * diff
            
            synthetic_samples.append(synthetic)
        
        # Преобразуем обратно в исходную форму
        synthetic_samples = np.array(synthetic_samples).reshape(-1, X.shape[1], X.shape[2])
        synthetic_labels = np.ones(len(synthetic_samples))
        
        # Объединяем с исходными данными
        X_balanced = np.concatenate([X, synthetic_samples])
        y_balanced = np.concatenate([y, synthetic_labels])
        
        return X_balanced, y_balanced

# Функция для применения SMOTE к нашим данным
def apply_smote_balancing(train_dataset):
    """Применяет Borderline-SMOTE к тренировочным данным"""
    # Собираем все данные и метки
    all_data = []
    all_labels = []
    
    for i in range(len(train_dataset)):
        signal, label = train_dataset[i]
        all_data.append(signal.numpy())
        all_labels.append(label.item())
    
    X = np.array(all_data)
    y = np.array(all_labels)
    
    print("\n=== ДИАГНОСТИКА SMOTE ===")
    print(f"ДО SMOTE:")
    print(f"  Всего samples: {len(y)}")
    print(f"  Target (1): {np.sum(y)} samples ({np.sum(y)/len(y)*100:.1f}%)")
    print(f"  Non-target (0): {len(y)-np.sum(y)} samples ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")
    
    # Применяем SMOTE
    smote = BorderlineSMOTE1D(k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Создаем новый сбалансированный датасет
    balanced_targets = [('target' if label == 1 else 'nontarget', i) 
                       for i, label in enumerate(y_balanced)]
    
    print(f"ПОСЛЕ SMOTE:")
    print(f"  Всего samples: {len(y_balanced)}")
    print(f"  Target (1): {np.sum(y_balanced)} samples ({np.sum(y_balanced)/len(y_balanced)*100:.1f}%)")
    print(f"  Non-target (0): {len(y_balanced)-np.sum(y_balanced)} samples ({(len(y_balanced)-np.sum(y_balanced))/len(y_balanced)*100:.1f}%)")
    print(f"  Добавлено synthetic samples: {len(y_balanced) - len(y)}")
    print("========================\n")
    
    return EEGDataset(X_balanced, balanced_targets, augment=False, 
                     mean=train_dataset.mean, std=train_dataset.std)

def select_optimal_time_window(data, targets, fs=250):
    """Визуализирует ERP и помогает выбрать оптимальное временное окно"""
    import matplotlib.pyplot as plt
    
    # Преобразуем targets в массив для индексации
    target_indices = [i for i, t in enumerate(targets) if t[0] == 'target']
    nontarget_indices = [i for i, t in enumerate(targets) if t[0] == 'nontarget']
    
    if len(target_indices) == 0 or len(nontarget_indices) == 0:
        print("Предупреждение: недостаточно данных для анализа временного окна")
        # Возвращаем окно по умолчанию
        start_idx = int(250)
        end_idx = int(500)
        return start_idx, end_idx
    
    target_data = data[target_indices]
    nontarget_data = data[nontarget_indices]
    
    # Усредняем по trials
    target_mean = np.nanmean(target_data, axis=0)
    nontarget_mean = np.nanmean(nontarget_data, axis=0)
    
    # Визуализируем разницу
    time = np.arange(data.shape[2]) / fs * 1000  # в мс
    
    plt.figure(figsize=(12, 6))
    for ch in range(min(8, data.shape[1])):  # первые 8 каналов
        diff = target_mean[ch] - nontarget_mean[ch]
        # Заменяем NaN на 0 для визуализации
        diff = np.nan_to_num(diff, nan=0.0)
        plt.plot(time, diff, label=f'Channel {ch}', alpha=0.7)
    
    plt.axvline(250, color='r', linestyle='--', label='250ms')
    plt.axvline(500, color='r', linestyle='--', label='500ms')
    plt.xlabel('Time (ms)')
    plt.ylabel('Target - Nontarget Amplitude')
    plt.title('ERP Difference (Target vs Nontarget)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Возвращаем оптимальное окно (250-500ms для P300)
    start_idx = int((250 + 200) * fs / 1000)  # 250ms от стимула
    end_idx = int((500 + 200) * fs / 1000)   # 500ms от стимула
    
    return start_idx, end_idx

# Применяем оптимальное окно
def apply_time_window(dataset, start_idx, end_idx):
    """Обрезает данные по выбранному временному окну"""
    windowed_data = dataset.data[:, :, start_idx:end_idx]
    
    # Создаем новые targets с правильными индексами
    new_targets = list(zip(dataset.labels, range(len(dataset.labels))))
    
    return EEGDataset(windowed_data, new_targets, 
                     augment=False, 
                     mean=dataset.mean, 
                     std=dataset.std)

def shuffle_labels_test(model, train_loader, val_loader):
    """Тест с перемешанными метками для проверки обучения"""
    # Сохраняем оригинальные метки
    original_train_labels = []
    for _, labels in train_loader:
        original_train_labels.extend(labels.squeeze().tolist())
    
    # Перемешиваем метки
    shuffled_train_loader = []
    for signals, labels in train_loader:
        shuffled_labels = labels[torch.randperm(len(labels))]
        shuffled_train_loader.append((signals, shuffled_labels))
    
    print("Тест с перемешанными метками...")
    shuffled_history = train_model(
        model=model,
        train_loader=shuffled_train_loader,
        val_loader=val_loader,
        num_epochs=50,
        target_class_weight=1.0  # равные веса для теста
    )
    
    final_f1 = shuffled_history["val_f1_target"][-1]
    print(f"F1-score с перемешанными метками: {final_f1:.4f}")
    
    if final_f1 > 0.6:
        print("ВНИМАНИЕ: Модель учится на шуме! Возможно, есть data leakage.")
    else:
        print("Тест пройден: модель не учится на случайных метках.")