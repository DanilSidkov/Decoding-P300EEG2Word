import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
import joblib

class SimpleEEGClassifier:
    """Обертка для простых классификаторов на эмбеддингах"""
    
    def __init__(self, classifier_type='svm', **kwargs):
        self.classifier_type = classifier_type
        self.classifier = self._create_classifier(classifier_type, kwargs)
        self.is_trained = False
        
    def _create_classifier(self, classifier_type, params):
        """Создает классификатор по типу"""
        if classifier_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42,
                **params
            )
        elif classifier_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                **params
            )
        elif classifier_type == 'logistic_regression':
            return LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000,
                **params
            )
        elif classifier_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=5,
                **params
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def fit(self, X, y):
        """Обучение классификатора"""
        print(f"Обучение {self.classifier_type} на {len(X)} samples...")
        self.classifier.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Предсказание"""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")
        return self.classifier.predict(X)
    
    def predict_proba(self, X):
        """Вероятности классов"""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")
        return self.classifier.predict_proba(X)
    
    def evaluate(self, X, y):
        """Оценка качества"""
        y_pred = self.predict(X)
        balanced_accuracy = balanced_accuracy_score(y, y_pred)
        
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['nontarget', 'target'], zero_division=0))
        
        return balanced_accuracy, y_pred

def compare_classifiers(X_train, y_train, X_val, y_val):
    """Сравнивает разные классификаторы"""
    classifiers = {
        'svm': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
        'knn': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n=== Обучение {name} ===")
        
        # Кросс-валидация
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='balanced_accuracy')
        print(f"Cross-val scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Обучение на всех тренировочных данных
        clf.fit(X_train, y_train)
        
        # Оценка на валидации
        y_val_pred = clf.predict(X_val)
        val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
        print(f"Validation balanced accuracy: {val_balanced_accuracy:.4f}")
        
        results[name] = {
            'model': clf,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'val_balanced_accuracy': val_balanced_accuracy
        }
    
    # Выбор лучшего классификатора
    best_name = max(results.keys(), key=lambda x: results[x]['val_balanced_accuracy'])
    best_model = results[best_name]['model']
    
    print(f"\n=== ЛУЧШИЙ КЛАССИФИКАТОР: {best_name} ===")
    print(f"Validation balanced accuracy: {results[best_name]['val_balanced_accuracy']:.4f}")
    
    return best_model, results