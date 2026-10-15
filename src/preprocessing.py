"""
Data preprocessing, log2 transformation, and stratified train/test split.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_tcga_barcodes(sample_ids: pd.Index) -> np.ndarray:
    """
    Extracts binary tumor vs normal labels from TCGA sample barcodes:
      - 01 to 09: Primary / recurrent tumor (label: 1)
      - 10 to 19: Normal solid tissue (label: 0)
    """
    labels = []
    for sid in sample_ids:
        parts = sid.split('-')
        sample_type = parts[3][:2] if len(parts) >= 4 else ''
        if sample_type.startswith('0') and sample_type != '00':
            labels.append(1)
        elif sample_type.startswith('1'):
            labels.append(0)
        else:
            labels.append(np.nan)
    return np.array(labels)


def apply_log2_transformation(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """Applies log2(x + 1) transformation if raw counts exceed threshold."""
    if df.max().max() > threshold:
        return np.log2(df + 1.0)
    return df


def filter_low_expression_genes(df: pd.DataFrame, threshold: float = 3.32) -> pd.DataFrame:
    """Filters out low-expressed genes where mean log2 expression < threshold."""
    mean_expr = df.mean(axis=1)
    return df.loc[mean_expr >= threshold].copy()


def prepare_train_test_split(
    df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Extracts labels, transposes samples x genes, and executes stratified train-test split.
    """
    y = parse_tcga_barcodes(df.columns)
    valid_mask = ~np.isnan(y)

    X_df = df.T.loc[valid_mask]
    y_clean = y[valid_mask].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_clean, test_size=test_size, stratify=y_clean, random_state=random_state
    )
    return X_train, X_test, y_train, y_test