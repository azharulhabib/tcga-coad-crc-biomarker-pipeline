# Machine Learning-Based Colorectal Cancer Biomarker Identification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end bioinformatics and machine learning pipeline for tumor versus normal solid tissue classification using TCGA-COAD RNA-seq transcriptomic data. The methodology integrates Differential Expression Analysis (DEA), SMOTE-based class balancing, and Boruta all-relevant feature selection to discover minimal, robust biomarker signatures.

---

## Overview

High-throughput RNA sequencing datasets present unique computational challenges, including extreme high-dimensionality ($p \gg n$) and severe class imbalance. This project implements a rigorous computational framework to classify Colorectal Adenocarcinoma (COAD) samples while mitigating data leakage through strict partition isolation.

### Key Highlights
- **99.8% Dimensionality Reduction**: Distilled 60,660 initial Ensembl transcriptomic features down to 142 robust biomarker genes.
- **Data Leakage Prevention**: All feature selection stages (differential expression analysis, custom Boruta algorithm) and SMOTE synthetic oversampling are executed exclusively within the training set ($n = 359$), preserving the held-out test set ($n = 155$) for final evaluation.
- **Classification Performance**: Five out of seven evaluated machine learning classifiers achieved 1.000 AUC and 100% test accuracy under 10-fold cross-validation.
- **Biological Concordance**: Successfully recovered established literature biomarkers (*SCARA5*, *VEGFA*, *PKIB*, *CWH43*, *TEX11*, *METTL7A*) and prioritized key transcriptional regulators including *CEMIP*, *ETV4*, and *TRIB3*.

---

## Methodology

The computational pipeline follows a sequential five-stage workflow:

1. **Data Ingestion and Annotation**: Raw UCSC Xena TCGA-COAD STAR FPKM gene expression data are loaded. Ensembl IDs are mapped to standard HGNC gene symbols using the MyGeneInfo service.
2. **Preprocessing**:
   - Barcode parsing: Primary tumor samples (`-01`) are assigned to class 1; normal solid tissue (`-11`) to class 0.
   - Normalization: $\log_2(x + 1)$ transformation applied to stabilize variance.
   - Low-expression filtering: Genes with mean $\log_2$ expression $< 3.32$ (approx. 10 FPKM counts) are excluded.
3. **Stratified Partitioning**: The cohort is split into 70% training ($n = 359$) and 30% held-out testing ($n = 155$) partitions stratified by class.
4. **Feature Selection**:
   - Differential Expression Analysis (DEA): Welch's two-sample $t$-test with Benjamini-Hochberg False Discovery Rate (FDR) adjustment ($|\log_2\text{FC}| > 1.0$, $p_{\text{adj}} < 0.05$).
   - SMOTE: Resolves the 11.4:1 tumor-to-normal imbalance by synthesizing minority class instances in the training set.
   - Boruta Algorithm: Custom implementation utilizing permuted shadow features and a 95% binomial confidence threshold to eliminate noisy genes.
5. **Model Benchmarking**: Comparative assessment of seven classification algorithms: Random Forest, Support Vector Machine (RBF kernel), $k$-Nearest Neighbors, Logistic Regression, Multi-Layer Perceptron (MLP), Decision Tree, and XGBoost.

---

## Benchmark Results

Evaluation on the independent held-out test set ($n = 155$):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Optimized)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **SVM (RBF Kernel)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **k-NN (k=5)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **MLP Classifier** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| XGBoost | 0.9935 | 1.0000 | 0.9929 | 0.9964 | 0.9996 |
| Decision Tree | 0.9677 | 0.9857 | 0.9789 | 0.9823 | 0.9395 |

---

## Repository Structure

```text
tcga-coad-crc-biomarker-pipeline/
├── config/
│   └── config.yaml               # Pipeline configurations and hyperparameters
├── data/
│   ├── raw/                      # Raw TCGA-COAD transcriptomic data (untracked)
│   └── processed/                # Intermediary datasets (untracked)
├── docs/
│   └── tcga-coad-crc-biomarker-pipeline.pdf # Full research manuscript
├── notebooks/
│   └── pipeline_walkthrough.ipynb # Modular walkthrough demonstration
├── results/
│   ├── figures/                  # Diagnostic plots and visualizations
│   └── tables/                   # Model comparison metrics (CSV)
├── scripts/
│   └── run_pipeline.py           # Command-line entry point for pipeline execution
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading and gene annotation
│   ├── preprocessing.py          # Transformation, filtering, and data splitting
│   ├── feature_selection.py      # Welch t-test, BH-FDR, SMOTE, and Boruta
│   ├── models.py                 # Estimators and hyperparameter definitions
│   ├── evaluation.py             # Metrics and cross-validation procedures
│   └── visualization.py          # Publication-ready plotting routines
├── .gitignore
├── requirements.txt
└── README.md