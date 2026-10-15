"""
Model evaluation, cross-validation, and performance metrics.
"""
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_cv(
    model: Any, X: pd.DataFrame, y: np.ndarray, n_splits: int = 10, random_state: int = 42
) -> Dict[str, float]:
    """Evaluates model using Stratified K-Fold cross validation."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = cross_validate(
        model, X, y, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    return {
        "acc_mean": results['test_accuracy'].mean(),
        "acc_std": results['test_accuracy'].std(),
        "f1_mean": results['test_f1'].mean(),
        "auc_mean": results['test_roc_auc'].mean()
    }


def evaluate_test_set(
    model: Any, X_test: pd.DataFrame, y_test: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Computes standard classification metrics and predictions on held-out test data."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba)
    }
    return metrics, y_pred, y_proba