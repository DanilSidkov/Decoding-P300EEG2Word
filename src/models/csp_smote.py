import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score
import warnings

def apply_smote_to_eeg_csp(X, y, method='regular', random_state=42):
    """
    Применение SMOTE к EEG данным с учетом их специфики
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
        EEG данные
    y : np.ndarray, shape (n_trials,)
        Метки классов
    method : str
        'regular', 'safe', или 'none'
    """
    
    original_shape = X.shape
    n_trials, n_channels, n_times = original_shape
    
    if method == 'none':
        return X, y
    
    X_2d = X.reshape(n_trials, -1)
    
    if method == 'safe':
        smote = SMOTE(
            random_state=random_state,
            sampling_strategy='auto',
            k_neighbors=min(3, np.sum(y == 1) - 1)
        )
    else:
        smote = SMOTE(random_state=random_state)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_balanced_2d, y_balanced = smote.fit_resample(X_2d, y)
    
    X_balanced = X_balanced_2d.reshape(-1, n_channels, n_times)
    
    print(f"После балансировки: {X_balanced.shape}, метки: {np.unique(y_balanced, return_counts=True)}")
    
    return X_balanced, y_balanced

def evaluate_smote_impact(csp, X_train, y_train, X_val, y_val, X_test, y_test):
    """Сравнение производительности с и без SMOTE"""
    
    csp_no_smote = CSP(n_components=csp.n_components)
    csp_no_smote.fit(X_train, y_train)
    
    bal_acc_no_smote = csp_no_smote.balanced_score(X_test, y_test)
    
    X_train_smote, y_train_smote = apply_smote_to_eeg_csp(X_train, y_train, method='safe')
    csp_smote = CSP(n_components=csp.n_components)
    csp_smote.fit(X_train_smote, y_train_smote)
    
    bal_acc_smote = csp_smote.balanced_score(X_test, y_test)
    
    print(f"\nСравнение SMOTE:")
    print(f"Без SMOTE - Balanced Accuracy: {bal_acc_no_smote:.3f}")
    print(f"С SMOTE  - Balanced Accuracy: {bal_acc_smote:.3f}")
    print(f"Разница: {bal_acc_smote - bal_acc_no_smote:+.3f}")
    
    return bal_acc_no_smote, bal_acc_smote