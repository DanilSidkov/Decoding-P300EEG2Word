import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


def print_model_parameters(model, model_name="Model"):
    """
    Красиво выводит информацию о параметрах модели
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"=== Параметры модели {model_name} ===")
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    print(f"Необучаемых параметров: {total_params - trainable_params:,}")
    print(f"Размер модели: {total_params * 4 / (1024**2):.2f} MB (float32)")
    print("=" * 40)
    
    return total_params, trainable_params

class NeuroInformedEarlyStopping:
    """Ранняя остановка с учетом метрик BCI"""

    def __init__(self, patience=5, min_epochs=5):
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_ac = 10000
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, current_ac, epoch, train_loss, val_loss):
        if train_loss < self.best_ac:
            self.best_ac = train_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            
        if epoch < self.min_epochs:
            return
            
        if self.epochs_no_improve >= self.patience:
            self.early_stop = True
            print(f"Early stopping triggered.")

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.softmax(inputs, dim=1)[range(len(targets)), targets]
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_neuro_metrics(all_predictions, all_labels):
    """Специализированные метрики для BCI классификации"""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
        balanced_accuracy_score
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=[0, 1], zero_division=0
    ) #, zero_division=0

    try:
        auc = roc_auc_score(all_labels, all_predictions)
    except:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_predictions)

    return {
        "accuracy": balanced_accuracy_score(all_labels, all_predictions),
        "precision_nontarget": precision[0],
        "precision_target": precision[1],
        "recall_nontarget": recall[0],
        "recall_target": recall[1],
        "f1_nontarget": f1[0],
        "f1_target": f1[1],
        "auc_roc": auc,
        "confusion_matrix": cm,
        "specificity": cm[0, 0] / (cm[0, 0] + cm[0, 1])
        if (cm[0, 0] + cm[0, 1]) > 0
        else 0,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 200,
    target_class_weight: float = 5.0,
    update_plot_every: int = 1,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    
    # Получаем все метки из тренировочного датасета
    train_labels = train_loader.dataset.encoded_labels
    
    # Автоматически определяем все присутствующие классы
    unique_classes = np.unique(train_labels)
    print(f"Найдены классы в тренировочных данных: {unique_classes}")
    
    # Вычисляем веса только для присутствующих классов
    if len(unique_classes) > 0:
        class_weights_array = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=train_labels
        )
        class_weights = torch.tensor(class_weights_array, dtype=torch.float32).to(device)
        print(f"Вычисленные веса классов: {class_weights}")
    else:
        # Fallback если нет данных
        class_weights = torch.tensor([1.0, 1.0]).to(device)
        print("Предупреждение: используем веса по умолчанию") 

    
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_focal = FocalLoss(weight=class_weights, gamma=2.0)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001, 
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    best_ac = 0.0
    best_model_state = None
    
    accumulation_steps = 4
    steps_per_epoch = len(train_loader) // accumulation_steps
    if len(train_loader) % accumulation_steps != 0:
        steps_per_epoch += 1

    total_steps = steps_per_epoch * num_epochs

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=total_steps,
        pct_start=0.2,
        div_factor=5.0,
        final_div_factor=50.0
    )
    
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_target": [],
        "val_f1_nontarget": [],
        "val_precision_target": [],
        "val_precision_nontarget": [],
        "val_recall_target": [],
        "val_recall_nontarget": [],
        "learning_rate": [],
        "ROC_AUC": [],
        "specificity": [],
        "train_accuracy": [],
        "grad_norm": [],
    }

    early_stopping = NeuroInformedEarlyStopping(patience=10, min_epochs=15)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for i, (signals, labels) in enumerate(train_loader):
            signals, labels = signals.to(device), labels.squeeze().to(device)

            outputs = model(signals)
            
            loss_ce = criterion_ce(outputs, labels)
            loss_focal = criterion_focal(outputs, labels)
            loss = 0.7 * loss_ce + 0.3 * loss_focal
            
            loss = loss / accumulation_steps
            loss.backward()

            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if (i + 1) % accumulation_steps == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=1.0,
                    error_if_nonfinite=True
                )
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
            train_loss += loss.item() * accumulation_steps

        if len(train_loader) % accumulation_steps != 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.squeeze().to(device)
                outputs = model(signals)
                
                val_loss_ce = criterion_ce(outputs, labels)
                val_loss_focal = criterion_focal(outputs, labels)
                val_loss_combined = 0.7 * val_loss_ce + 0.3 * val_loss_focal
                val_loss += val_loss_combined.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = calculate_neuro_metrics(all_preds, all_labels)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        current_ac = metrics["accuracy"]
        if current_ac > best_ac:
            best_ac = current_ac
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'balanced_accuracy': best_ac,
                'metrics': metrics
            }

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_f1_target"].append(metrics["f1_target"])
        history["val_f1_nontarget"].append(metrics["f1_nontarget"])
        history["val_precision_target"].append(metrics["precision_target"])
        history["val_precision_nontarget"].append(metrics["precision_nontarget"])
        history["val_recall_target"].append(metrics["recall_target"])
        history["val_recall_nontarget"].append(metrics["recall_nontarget"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["ROC_AUC"].append(metrics["auc_roc"])
        history["specificity"].append(metrics["specificity"])
        history["grad_norm"].append(total_norm.item() if 'total_norm' in locals() else 0.0)

        if (epoch + 1) % update_plot_every == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Val Acc: {metrics['accuracy']:.4f}")
            print(f"Target F1: {metrics['f1_target']:.4f} | Precision: {metrics['precision_target']:.4f} | Recall: {metrics['recall_target']:.4f}")
            print(f"NonTarget F1: {metrics['f1_nontarget']:.4f} | Precision: {metrics['precision_nontarget']:.4f} | Recall: {metrics['recall_nontarget']:.4f}")
            print(f"ROC AUC: {metrics['auc_roc']:.4f} | Specificity: {metrics['specificity']:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e} | Grad Norm: {history['grad_norm'][-1]:.4f}")
            print(f"Best Balanced Accuracy: {best_ac:.4f}")
            print("-" * 60)

        early_stopping(current_ac, epoch, train_loss, val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f"\nLoaded best model from epoch {best_model_state['epoch'] + 1} with Balanced Accuracy: {best_ac:.4f}")

    history["best_ac"] = best_ac
    history["best_epoch"] = best_model_state['epoch'] if best_model_state else epoch
    
    return history

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
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
        balanced_accuracy_score
    )
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
    bac = 100 * balanced_accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test BAccuracy: {bac:.2f}%")

    return all_predictions, all_labels, bac
