import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

class NeuroInformedEarlyStopping:
    """Ранняя остановка с учетом метрик BCI"""

    def __init__(self, patience=5, min_epochs=5):
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_f1 = 0
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, current_f1, epoch, train_loss, val_loss):
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            
        if epoch < self.min_epochs:
            return
            
        if self.epochs_no_improve >= self.patience:
            self.early_stop = True
            print(f"Early stopping triggered. Best F1: {self.best_f1:.4f}")

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
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
        all_labels, all_predictions, average=None, labels=[0, 1]
    )

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
    
    class_weights = torch.tensor([1.0, target_class_weight]).to(device)
    
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_focal = FocalLoss(weight=class_weights, gamma=2.0)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    best_f1 = 0.0
    best_model_state = None
    
    accumulation_steps = 4
    steps_per_epoch = (len(train_loader) + accumulation_steps - 1) // accumulation_steps
    if len(train_loader) % accumulation_steps != 0:
        steps_per_epoch += 1

    total_steps = steps_per_epoch * num_epochs

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_target": [],
        "val_precision_target": [],
        "val_recall_target": [],
        "learning_rate": [],
        "ROC_AUC": [],
        "specificity": [],
        "train_accuracy": [],
        "grad_norm": [],
    }

    early_stopping = NeuroInformedEarlyStopping(patience=5, min_epochs=10)

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
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        current_f1 = metrics["f1_target"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict().copy(),
                'optimizer_state_dict': optimizer.state_dict().copy(),
                'f1': best_f1,
                'metrics': metrics.copy()
            }

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_f1_target"].append(current_f1)
        history["val_precision_target"].append(metrics["precision_target"])
        history["val_recall_target"].append(metrics["recall_target"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["ROC_AUC"].append(metrics["auc_roc"])
        history["specificity"].append(metrics["specificity"])
        history["train_accuracy"].append(train_accuracy)
        history["grad_norm"].append(total_norm.item() if 'total_norm' in locals() else 0.0)

        if (epoch + 1) % update_plot_every == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Train Acc: {train_accuracy:.4f} | Val Acc: {metrics['accuracy']:.4f}")
            print(f"Target F1: {current_f1:.4f} | Precision: {metrics['precision_target']:.4f} | Recall: {metrics['recall_target']:.4f}")
            print(f"ROC AUC: {metrics['auc_roc']:.4f} | Specificity: {metrics['specificity']:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e} | Grad Norm: {history['grad_norm'][-1]:.4f}")
            print(f"Best F1: {best_f1:.4f}")
            print("-" * 60)

        early_stopping(current_f1, epoch, train_loss, val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f"\nLoaded best model from epoch {best_model_state['epoch'] + 1} with F1: {best_f1:.4f}")

    history["best_f1"] = best_f1
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
    bac = balanced_accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test BAccuracy: {bac}%")

    return all_predictions, all_labels, accuracy
