"""
Feature selection pipeline: Differential Expression Analysis (DEA), SMOTE, and Boruta.
"""
from typing import Dict, List, Tuple
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import binom, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import statsmodels.stats.multitest as smm
from tqdm import tqdm


def run_differential_expression(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    log2fc_thresh: float = 1.0,
    padj_thresh: float = 0.05
) -> Tuple[List[str], pd.DataFrame]:
    """
    Performs Welch's t-test and Benjamini-Hochberg FDR correction on training set only.
    """
    tumor_mask = (y_train == 1)
    normal_mask = (y_train == 0)
    gene_names = X_train_df.columns.values

    log2fcs, p_values = [], []
    for gene in tqdm(gene_names, desc="DEA Analysis"):
        tumor_expr = X_train_df.loc[tumor_mask, gene].values
        normal_expr = X_train_df.loc[normal_mask, gene].values
        log2fc = float(tumor_expr.mean() - normal_expr.mean())
        _, p_val = ttest_ind(tumor_expr, normal_expr, equal_var=False, nan_policy='omit')
        log2fcs.append(log2fc)
        p_values.append(p_val if not np.isnan(p_val) else 1.0)

    _, p_adj, _, _ = smm.multipletests(p_values, alpha=padj_thresh, method='fdr_bh')
    dea_df = pd.DataFrame({
        'gene': gene_names,
        'log2FC': log2fcs,
        'p_value': p_values,
        'padj': p_adj
    })

    sig_mask = (np.abs(dea_df['log2FC']) > log2fc_thresh) & (dea_df['padj'] < padj_thresh)
    selected_genes = dea_df.loc[sig_mask, 'gene'].tolist()
    return selected_genes, dea_df


def balance_classes_smote(
    X_train_de: pd.DataFrame, y_train: np.ndarray, random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Applies SMOTE oversampling on training set to resolve tumor:normal imbalance."""
    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_train_de, y_train)
    return pd.DataFrame(X_balanced, columns=X_train_de.columns), y_balanced


def custom_boruta(
    X_df: pd.DataFrame,
    y: np.ndarray,
    trials: int = 50,
    random_state: int = 42
) -> Tuple[List[str], Dict[str, int]]:
    """
    Custom Boruta all-relevant feature selection using shadow features and binomial threshold.
    """
    df = X_df.copy()
    orig_cols = list(df.columns)
    feature_hits = {col: 0 for col in orig_cols}

    # Generate shadow features all at once to prevent fragmentation
    shadow_dict = {
        f"shadow_{col}": shuffle(df[col].values, random_state=random_state)
        for col in orig_cols
    }
    df = pd.concat([df, pd.DataFrame(shadow_dict, index=df.index)], axis=1)

    for trial in tqdm(range(trials), desc="Boruta Feature Selection"):
        for col in orig_cols:
            df[f"shadow_{col}"] = shuffle(df[col].values, random_state=random_state + trial)

        rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=random_state, n_jobs=-1)
        rf.fit(df, y)
        importances = dict(zip(df.columns, rf.feature_importances_))

        max_shadow = max([v for k, v in importances.items() if k.startswith('shadow_')], default=0.0)
        for feat, imp in importances.items():
            if not feat.startswith('shadow_') and imp > max_shadow:
                feature_hits[feat] += 1

    # 95% confidence binomial distribution threshold
    pmf = [binom.pmf(x, trials, 0.5) for x in range(trials + 1)]
    cum_prob, threshold = 0.0, trials
    for i, p in enumerate(reversed(pmf)):
        cum_prob += p
        if cum_prob >= 0.05:
            threshold = trials - i
            break

    selected = [f for f, hits in feature_hits.items() if hits >= threshold]
    return selected, feature_hits