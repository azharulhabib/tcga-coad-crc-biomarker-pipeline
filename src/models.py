"""
Machine learning model definitions and factory helpers.
"""
from typing import Any, Dict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


def get_baseline_rf(random_state: int = 42) -> RandomForestClassifier:
    """Returns default Random Forest classifier for all-gene baseline."""
    return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)


def get_optimized_rf(random_state: int = 42) -> RandomForestClassifier:
    """Returns tuned Random Forest classifier for final biomarker subset."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1
    )


def get_benchmark_models(random_state: int = 42) -> Dict[str, Any]:
    """Returns full model dictionary for comparative benchmarking."""
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": get_optimized_rf(random_state=random_state),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=random_state),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=random_state)
    }