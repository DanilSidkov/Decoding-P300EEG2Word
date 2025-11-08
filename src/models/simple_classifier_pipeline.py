from features.eegdataset import EEGDataset, load_and_prepare_data, compute_dataset_stats
from models.smote import select_optimal_time_window, apply_time_window
from models.autoencoder import EEGAutoencoder, train_autoencoder, extract_features
from models.simple_classifiers import SimpleEEGClassifier, compare_classifiers
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

def simple_classifier_pipeline(datapath, test_subject=8, classifier_type='svm'):
    """Пайплайн с автоэнкодером и простым классификатором"""
    
    try:
        # Загрузка и подготовка данных
        train_dataset, val_dataset, test_dataset, LE = load_and_prepare_data(datapath, test_subject=test_subject)
        
        # Выбор временного окна
        print("Анализ временного окна...")
        start_idx, end_idx = select_optimal_time_window(train_dataset.data, train_dataset.targets)
        print(f"Оптимальное окно: {start_idx}-{end_idx}")
        
        # Применяем временное окно
        train_dataset = apply_time_window(train_dataset, start_idx, end_idx)
        val_dataset = apply_time_window(val_dataset, start_idx, end_idx)
        test_dataset = apply_time_window(test_dataset, start_idx, end_idx)

        # Нормализация
        train_mean, train_std = compute_dataset_stats(train_dataset)
        train_dataset.mean = train_mean
        train_dataset.std = train_std
        val_dataset.mean = train_mean
        val_dataset.std = train_std
        test_dataset.mean = train_mean
        test_dataset.std = train_std
        
        # Создаем DataLoader'ы
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # shuffle=False для консистентности
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Создаем и обучаем автоэнкодер
        seq_length = end_idx - start_idx
        print(f"Длина последовательности: {seq_length}")
        
        autoencoder = EEGAutoencoder(
            input_channels=8,
            seq_length=seq_length,
            embedding_dim=12
        )
        
        print("Предварительное обучение автоэнкодера...")
        train_losses, val_losses = train_autoencoder(
            autoencoder, train_loader, val_loader, num_epochs=50
        )
        
        # Извлекаем признаки
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder.to(device)
        
        print("Извлечение признаков...")
        X_train, y_train = extract_features(autoencoder, train_loader, device)
        X_val, y_val = extract_features(autoencoder, val_loader, device)
        X_test, y_test = extract_features(autoencoder, test_loader, device)
        
        print(f"Размерности данных:")
        print(f"Train: {X_train.shape}, Labels: {y_train.shape}")
        print(f"Val: {X_val.shape}, Labels: {y_val.shape}")
        print(f"Test: {X_test.shape}, Labels: {y_test.shape}")
        
        # Балансировка классов (опционально)
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"После балансировки: {X_train_balanced.shape}, {y_train_balanced.shape}")
        
        # Выбор и обучение классификатора
        if classifier_type == 'compare':
            # Сравниваем все классификаторы
            best_classifier, results = compare_classifiers(
                X_train_balanced, y_train_balanced, X_val, y_val
            )
            classifier = best_classifier
        else:
            # Используем указанный классификатор
            simple_clf = SimpleEEGClassifier(classifier_type)
            classifier = simple_clf.fit(X_train_balanced, y_train_balanced)
            
            # Оценка на валидации
            print(f"\n=== ОЦЕНКА НА ВАЛИДАЦИИ ({classifier_type.upper()}) ===")
            val_balanced_accuracy, val_preds = simple_clf.evaluate(X_val, y_val)
        
        # Финальная оценка на тесте
        print(f"\n=== ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТЕ ===")
        if hasattr(classifier, 'predict'):
            y_test_pred = classifier.predict(X_test)
            test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
        else:
            test_accuracy = classifier.score(X_test, y_test)
            test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
        
        print(f"Test Balanced Accuracy: {test_balanced_accuracy:.4f}")
        print(f"Test Balanced Accuracy: {test_balanced_accuracy*100:.2f}%")
        
        # Визуализация эмбеддингов
        visualize_embeddings(X_test, y_test, title="Тестовые эмбеддинги")
        
        return {
            'autoencoder': autoencoder,
            'classifier': classifier,
            'test_balanced_accuracy': test_balanced_accuracy,
            'X_test': X_test,
            'y_test': y_test
        }
        
    except Exception as e:
        print(f"Ошибка в пайплайне: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_embeddings(embeddings, labels, title="Эмбеддинги"):
    """Визуализация эмбеддингов с помощью t-SNE"""
    # Уменьшаем размерность для визуализации
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Класс')
    plt.title(f'{title} (t-SNE)')
    plt.xlabel('Компонента 1')
    plt.ylabel('Компонента 2')
    plt.show()
    
    return embeddings_2d

def run_comparison(datapath, test_subject=8):
    """Сравнивает разные подходы"""
    print("=== СРАВНЕНИЕ КЛАССИФИКАТОРОВ ===")
    
    classifiers = ['svm', 'random_forest', 'logistic_regression', 'knn']
    results = {}
    
    for clf_type in classifiers:
        print(f"\n{'='*50}")
        print(f"ТЕСТИРУЕМ: {clf_type.upper()}")
        print(f"{'='*50}")
        
        result = simple_classifier_pipeline(datapath, test_subject, clf_type)
        if result is not None:
            results[clf_type] = result['test_balanced_accuracy']
    
    # Вывод результатов сравнения
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
    print(f"{'='*60}")
    for clf_type, balanced_accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{clf_type:>20}: {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)")
    
    return results