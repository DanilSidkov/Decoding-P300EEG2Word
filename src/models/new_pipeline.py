from features.eegdataset import EEGDataset, load_and_prepare_data, compute_dataset_stats
from models.smote import select_optimal_time_window, apply_time_window, apply_smote_balancing, shuffle_labels_test
from models.eegnet1d import OptimizedEEGNet1D_v2
from models.train_model import train_model, test_model
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

def improved_pipeline(datapath, test_subject=8):
    """Улучшенный пайплайн обучения с учетом всех рекомендаций"""
    
    try:
        train_dataset, val_dataset, test_dataset, LE = load_and_prepare_data(datapath, test_subject=test_subject)

        train_mean, train_std = compute_dataset_stats(train_dataset)
        print(f"Статистики нормализации: mean={train_mean:.4f}, std={train_std:.4f}")
        
        # Устанавливаем статистики для всех датасетов
        train_dataset.mean = train_mean
        train_dataset.std = train_std
        val_dataset.mean = train_mean
        val_dataset.std = train_std
        test_dataset.mean = train_mean
        test_dataset.std = train_std
        
        # Визуализация ERP и выбор временного окна
        print("Анализ временного окна...")
        start_idx, end_idx = select_optimal_time_window(train_dataset.data, train_dataset.targets)
        print(f"Оптимальное окно: {start_idx}-{end_idx}")
        
        # Применяем временное окно
        print(train_dataset.data.shape)
        print(val_dataset.data.shape)
        print(test_dataset.data.shape)
        train_dataset = apply_time_window(train_dataset, start_idx, end_idx)
        val_dataset = apply_time_window(val_dataset, start_idx, end_idx)
        test_dataset = apply_time_window(test_dataset, start_idx, end_idx)
        
        # Балансировка классов с Borderline-SMOTE
        #print("Применение Borderline-SMOTE...")
        #train_dataset = apply_smote_balancing(train_dataset)
        
        # Создаем DataLoader'ы
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Получаем все метки из тренировочного датасета
        train_labels = train_loader.dataset.encoded_labels
        
        # Автоматически определяем все присутствующие классы
        unique_classes = np.unique(train_labels)
        print(f"Найдены классы в тренировочных данных: {unique_classes}")
        
        # Создаем модель с оптимизированными параметрами
        seq_length = end_idx - start_idx
        print(f"Длина последовательности после обрезки: {seq_length}")
        
        model = OptimizedEEGNet1D_v2(
            input_channels=8,
            seq_length=seq_length,
            num_classes=2
        )
        
        # Обучение с оптимизированными параметрами
        print("Начало обучения...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,
            target_class_weight=1.0
        )
        
        # Тестирование
        print("Тестирование модели...")
        predictions, true_labels, test_accuracy = test_model(model, test_loader)

        #shuffle_labels_test(model, train_loader, val_loader)
        
        print(f"Финальная точность на тесте: {test_accuracy:.2f}%")
        
        return model, history, test_accuracy
        
    except Exception as e:
        print(f"Ошибка в пайплайне: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0.0