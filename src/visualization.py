"""
Visualization suite producing publication-ready diagnostics and comparison figures.
"""
import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def plot_comprehensive_dashboard(
    y_test: np.ndarray,
    y_pred_opt: np.ndarray,
    y_proba_opt: np.ndarray,
    importances: pd.Series,
    results_df: pd.DataFrame,
    save_path: Optional[str] = "results/figures/TCGA_COAD_Results.png"
):
    """Generates integrated 4-panel analysis dashboard matching paper Figure 2."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_opt)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Normal', 'Tumor'], yticklabels=['Normal', 'Tumor'])
    axes[0, 0].set_title('Confusion Matrix (Optimized Model)', fontsize=14, pad=10)
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba_opt)
    auc_val = roc_auc_score(y_test, y_proba_opt)
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_val:.4f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=10)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Feature Importance (Top 15 Biomarkers)
    top_features = importances.head(15)
    axes[1, 0].barh(range(len(top_features)), top_features.values, color='#2c3e50')
    axes[1, 0].set_yticks(range(len(top_features)))
    axes[1, 0].set_yticklabels(top_features.index)
    axes[1, 0].set_xlabel('Mean Decrease in Impurity (Importance)')
    axes[1, 0].set_title('Top 15 Most Important Biomarkers', fontsize=14, pad=10)
    axes[1, 0].invert_yaxis()

    # 4. Model Comparison Barplot
    results_df['AUC'].plot(kind='barh', ax=axes[1, 1], color='#3498db')
    axes[1, 1].set_xlabel('AUC Score')
    axes[1, 1].set_title('Model Comparison by AUC', fontsize=14, pad=10)
    axes[1, 1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
    plt.close()