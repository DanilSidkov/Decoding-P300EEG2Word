import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

class CSP:
    """Common Spatial Patterns для извлечения признаков из EEG сигналов"""
    
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None
        self.patterns_ = None
        self.lda = LinearDiscriminantAnalysis()
        
    def fit(self, X, y):
        """
        Обучение CSP фильтров
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)
            EEG данные
        y : np.ndarray, shape (n_trials,)
            Метки классов (0 - nontarget, 1 - target)
        """
        n_trials, n_channels, n_times = X.shape
        
        # Разделяем данные по классам
        class_0 = X[y == 0]  # nontarget
        class_1 = X[y == 1]  # target
        
        # Вычисляем ковариационные матрицы для каждого класса
        cov_0 = np.zeros((n_channels, n_channels))
        cov_1 = np.zeros((n_channels, n_channels))
        
        for trial in class_0:
            cov_0 += np.cov(trial)
        cov_0 /= len(class_0)
        
        for trial in class_1:
            cov_1 += np.cov(trial)
        cov_1 /= len(class_1)
        
        # Решаем обобщенную проблему собственных значений
        eigenvalues, eigenvectors = eigh(cov_1, cov_0 + cov_1)
        
        # Сортируем собственные векторы по убыванию собственных значений
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Выбираем наиболее информативные компоненты
        self.filters_ = eigenvectors[:, :self.n_components]
        
        # Обучаем LDA на CSP признаках
        features = self.transform(X)
        self.lda.fit(features, y)
        
        return self
    
    def transform(self, X):
        """Преобразование данных с помощью CSP фильтров"""
        if self.filters_ is None:
            raise ValueError("CSP filters not fitted. Call fit() first.")
        
        n_trials = X.shape[0]
        features = np.zeros((n_trials, self.n_components))
        
        for i in range(n_trials):
            # Применяем CSP фильтры
            filtered_data = self.filters_.T @ X[i]
            # Извлекаем признаки как логарифм дисперсии
            features[i] = np.log(np.var(filtered_data, axis=1))
            
        return features
    
    def predict(self, X):
        """Предсказание классов"""
        features = self.transform(X)
        return self.lda.predict(features)
    
    def predict_proba(self, X):
        """Вероятности классов"""
        features = self.transform(X)
        return self.lda.predict_proba(features)
    
    def score(self, X, y):
        """Обычная точность классификации"""
        features = self.transform(X)
        return self.lda.score(features, y)
    
    def balanced_score(self, X, y):
        """Сбалансированная точность классификации"""
        predictions = self.predict(X)
        return balanced_accuracy_score(y, predictions)
    
    def detailed_evaluation(self, X, y, set_name=""):
        """Детальная оценка с различными метриками"""
        predictions = self.predict(X)
        
        bal_acc = balanced_accuracy_score(y, predictions)
        acc = self.score(X, y)
        cm = confusion_matrix(y, predictions)
        
        print(f"\n{set_name} - Детальная оценка:")
        print(f"Balanced Accuracy: {bal_acc:.3f}")
        print(f"Standard Accuracy: {acc:.3f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y, predictions, target_names=['nontarget', 'target'], zero_division = 0))
        
        return bal_acc, acc, cm

def find_optimal_csp_components(X_train, y_train, X_val, y_val, max_components=8):
    """Поиск оптимального количества CSP компонент"""
    best_score = 0
    best_n_components = 2
    
    for n_components in range(2, min(max_components, X_train.shape[1]) + 1, 2):
        csp = CSP(n_components=n_components)
        csp.fit(X_train, y_train)
        score = csp.score(X_val, y_val)
        
        if score > best_score:
            best_score = score
            best_n_components = n_components
            
    return best_n_components

def plot_csp_patterns(csp, channel_names=None, title="CSP Patterns"):
    """Визуализация CSP паттернов"""
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(csp.filters_.shape[0])]
    
    fig, axes = plt.subplots(1, csp.n_components, figsize=(4*csp.n_components, 4))
    
    for i in range(csp.n_components):
        pattern = csp.filters_[:, i]
        axes[i].barh(channel_names, pattern)
        axes[i].set_title(f'Component {i+1}')
        axes[i].set_xlabel('Weight')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()