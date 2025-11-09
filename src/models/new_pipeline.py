from features.eegdataset import EEGDataset, load_and_prepare_data, compute_dataset_stats
from models.smote import select_optimal_time_window, apply_time_window, apply_smote_balancing, shuffle_labels_test
from models.eegnet1d import OptimizedEEGNet1D_v2
from models.eeg import EEGNet, DeepConvNet
from models.train_model import train_model, test_model, print_model_parameters
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from models.autoencoder import EEGAutoencoder, AutoencoderClassifier, train_autoencoder, visualize_embeddings
from models.csp import CSP, find_optimal_csp_components, plot_csp_patterns
from models.csp_smote import apply_smote_to_eeg_csp
from models.vision_transformer import EEGVisionTransformer, EnhancedEEGViT

def improved_pipeline(datapath, test_subject=8, method=None, use_autoencoder=False, use_smote='auto', model_type='vit'):
    """Улучшенный пайплайн обучения с учетом всех рекомендаций"""
    
    try:
        train_dataset, val_dataset, test_dataset, LE = load_and_prepare_data(datapath, test_subject=test_subject)
        
        print("Анализ временного окна...")
        start_idx, end_idx = select_optimal_time_window(train_dataset.data, train_dataset.targets)
        print(f"Оптимальное окно: {start_idx}-{end_idx}")
        
        print("Формы данных до обрезки:")
        print(train_dataset.data.shape)
        print(val_dataset.data.shape)
        print(test_dataset.data.shape)
        train_dataset = apply_time_window(train_dataset, start_idx, end_idx)
        val_dataset = apply_time_window(val_dataset, start_idx, end_idx)
        test_dataset = apply_time_window(test_dataset, start_idx, end_idx)

        print("Формы данных после обрезки:")
        print(f"Train: {train_dataset.data.shape}")
        print(f"Val: {val_dataset.data.shape}")
        print(f"Test: {test_dataset.data.shape}")

        if method == 'csp':
            print("Использование CSP + LDA для классификации...")
            
            X_train = train_dataset.data  # (n_trials, n_channels, n_times)
            y_train = train_dataset.encoded_labels
            X_val = val_dataset.data
            y_val = val_dataset.encoded_labels
            X_test = test_dataset.data
            y_test = test_dataset.encoded_labels

            print("\nАнализ дисбаланса классов:")
            print(f"Train - Nontarget: {np.sum(y_train == 0)}, Target: {np.sum(y_train == 1)}")
            print(f"Val - Nontarget: {np.sum(y_val == 0)}, Target: {np.sum(y_val == 1)}")
            print(f"Test - Nontarget: {np.sum(y_test == 0)}, Target: {np.sum(y_test == 1)}")

            class_counts = np.bincount(y_train)
            imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
            print(f"Коэффициент дисбаланса: {imbalance_ratio:.2f}:1")
            
            if use_smote == 'auto':
                use_smote_final = imbalance_ratio > 3.0
                print(f"Автоматический выбор: {'ИСПОЛЬЗОВАТЬ SMOTE' if use_smote_final else 'НЕ использовать SMOTE'}")
            else:
                use_smote_final = use_smote
            
            optimal_components = find_optimal_csp_components(X_train, y_train, X_val, y_val)
            
            if use_smote_final:
                print("Применение SMOTE для балансировки классов...")
                X_train_balanced, y_train_balanced = apply_smote_to_eeg_csp(
                    X_train, y_train, method='safe'
                )
            else:
                X_train_balanced, y_train_balanced = X_train, y_train
            
            print(f"После балансировки: {X_train_balanced.shape}, {y_train_balanced.shape}")
            
            csp = CSP(n_components=optimal_components)
            csp.fit(X_train_balanced, y_train_balanced)
            
            plot_csp_patterns(csp, title="CSP Spatial Patterns")
            
            print("\nОценка производительности CSP:")
            
            train_bal_acc, train_acc, train_cm = csp.detailed_evaluation(X_train, y_train, "TRAIN")
            val_bal_acc, val_acc, val_cm = csp.detailed_evaluation(X_val, y_val, "VALIDATION")
            test_bal_acc, test_acc, test_cm = csp.detailed_evaluation(X_test, y_test, "TEST")
            
            print(f"\nИтоговые результаты CSP:")
            print(f"Train Balanced Accuracy: {train_bal_acc:.3f}")
            print(f"Validation Balanced Accuracy: {val_bal_acc:.3f}")
            print(f"Test Balanced Accuracy: {test_bal_acc:.3f}")
            
            return csp, {
                "train_bal_acc": train_bal_acc, 
                "val_bal_acc": val_bal_acc,
                "train_acc": train_acc,
                "val_acc": val_acc
            }, test_bal_acc
            
        else:
            train_mean, train_std = compute_dataset_stats(train_dataset)
            print(f"Статистики нормализации: mean={train_mean:.4f}, std={train_std:.4f}")
            
            train_dataset.mean = train_mean
            train_dataset.std = train_std
            val_dataset.mean = train_mean
            val_dataset.std = train_std
            test_dataset.mean = train_mean
            test_dataset.std = train_std
    
            if use_autoencoder:
                print("Использование автоэнкодера для извлечения признаков...")
                
                autoencoder_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                autoencoder_val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                seq_length = end_idx - start_idx
                autoencoder = EEGAutoencoder(
                    input_channels=8,
                    seq_length=seq_length,
                    embedding_dim=12
                )
                
                print("Предварительное обучение автоэнкодера...")
                train_autoencoder(autoencoder, autoencoder_train_loader, autoencoder_val_loader, num_epochs=50)
                
                model = AutoencoderClassifier(autoencoder, num_classes=2)
                print_model_parameters(autoencoder, 'AutoEncoder')
                print_model_parameters(model, 'AutoEncoderClassifier')
    
            else:
                if model_type == 'vit':
                    print("Использование Vision Transformer для классификации EEG...")
                    seq_length = end_idx - start_idx
                    model = EEGVisionTransformer(
                        input_channels=8,
                        seq_length=seq_length,
                        patch_size=16,  # Можно настроить
                        embed_dim=128,
                        depth=4,
                        num_heads=4,
                        num_classes=2
                    )
                    print_model_parameters(model, 'vit')
                elif model_type == 'enhanced_vit':
                    print("Использование улучшенного ViT с канальным вниманием...")
                    seq_length = end_idx - start_idx
                    model = EnhancedEEGViT(
                        input_channels=8,
                        seq_length=seq_length,
                        patch_size=16,
                        embed_dim=128,
                        depth=4,
                        num_heads=4,
                        num_classes=2
                    )
                    print_model_parameters(model, 'enhanced_vit')
                elif model_type == 'deepconvnet':
                    print("Использование DeepConvNet...")
                    seq_length = end_idx - start_idx
                    model = DeepConvNet(
                        input_channels=8,
                        seq_length=seq_length,
                        num_classes=2
                    )
                    print_model_parameters(model, 'DeepConvNet')
                elif model_type == 'eeg':
                    print("Использование EEGNet...")
                    seq_length = end_idx - start_idx
                    model = OptimizedEEGNet1D_v2(
                        input_channels=8,
                        seq_length=seq_length,
                        num_classes=2
                    )
                    print_model_parameters(model, 'EEGNet')
                
            # Балансировка классов с Borderline-SMOTE
            #print("Применение Borderline-SMOTE...")
            #train_dataset = apply_smote_balancing(train_dataset)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            train_labels = train_loader.dataset.encoded_labels
            
            unique_classes = np.unique(train_labels)
            print(f"Найдены классы в тренировочных данных: {unique_classes}")
            print(f"Длина последовательности после обрезки: {seq_length}")
    
            print("Начало обучения...")
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=50,
                target_class_weight=1.0
            )
            
            print("Тестирование модели...")
            predictions, true_labels, test_accuracy = test_model(model, test_loader)
            
            print(f"Финальная точность на тесте: {test_accuracy:.2f}%")
    
            if use_autoencoder:
                embeddings_2d, all_labels = visualize_embeddings(model, test_loader, title="Эмбеддинги EEG сигналов")
            
            return model, history, test_accuracy
        
    except Exception as e:
        print(f"Ошибка в пайплайне: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0.0