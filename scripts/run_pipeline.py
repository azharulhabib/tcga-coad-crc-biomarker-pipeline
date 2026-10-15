"""
End-to-End Pipeline CLI Execution Script.
"""
import argparse
import os
import yaml
import pandas as pd
from src.data_loader import load_tcga_expression, convert_ensembl_to_symbols
from src.preprocessing import (
    apply_log2_transformation, filter_low_expression_genes, prepare_train_test_split
)
from src.feature_selection import (
    run_differential_expression, balance_classes_smote, custom_boruta
)
from src.models import get_baseline_rf, get_optimized_rf, get_benchmark_models
from src.evaluation import evaluate_cv, evaluate_test_set
from src.visualization import plot_comprehensive_dashboard


def main():
    parser = argparse.ArgumentParser(description="TCGA-COAD Biomarker Discovery Pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_path = cfg['data']['raw_path']
    print(f"=== Stage 1: Loading Data from {data_path} ===")
    df = load_tcga_expression(data_path)
    df = convert_ensembl_to_symbols(df)

    print("=== Stage 2: Normalization and Low-Expression Filtering ===")
    df = apply_log2_transformation(df, threshold=cfg['preprocessing']['log2_threshold'])
    df_filtered = filter_low_expression_genes(df, threshold=cfg['preprocessing']['expression_threshold'])

    print("=== Stage 3: Stratified Train-Test Split ===")
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df_filtered,
        test_size=cfg['preprocessing']['test_size'],
        random_state=cfg['preprocessing']['random_state']
    )

    print("=== Stage 4: Differential Expression Analysis (Training Set Only) ===")
    de_genes, _ = run_differential_expression(
        X_train, y_train,
        log2fc_thresh=cfg['dea']['log2fc_threshold'],
        padj_thresh=cfg['dea']['padj_threshold']
    )
    X_train_de = X_train[de_genes]
    X_test_de = X_test[de_genes]

    print("=== Stage 5: SMOTE Balancing & Boruta Feature Selection ===")
    X_train_balanced, y_train_balanced = balance_classes_smote(
        X_train_de, y_train, random_state=cfg['boruta']['random_state']
    )
    selected_genes, _ = custom_boruta(
        X_train_balanced, y_train_balanced,
        trials=cfg['boruta']['trials'],
        random_state=cfg['boruta']['random_state']
    )

    X_train_final = X_train_balanced[selected_genes]
    X_test_final = X_test_de[selected_genes]

    print(f"Selected {len(selected_genes)} Biomarker Genes.")

    print("=== Stage 6: Training and Benchmark Comparison ===")
    benchmark_models = get_benchmark_models(random_state=cfg['preprocessing']['random_state'])
    results = {}
    for name, model in benchmark_models.items():
        model.fit(X_train_final, y_train_balanced)
        metrics, _, _ = evaluate_test_set(model, X_test_final, y_test)
        results[name] = metrics

    results_df = pd.DataFrame(results).T.round(4).sort_values("AUC", ascending=False)
    print("\n" + "=" * 60)
    print(results_df.to_string())
    print("=" * 60)

    # Save tables and figures
    os.makedirs(cfg['output']['tables_dir'], exist_ok=True)
    results_path = os.path.join(cfg['output']['tables_dir'], "model_comparison_results.csv")
    results_df.to_csv(results_path)

    # Train optimized RF for feature importance and plotting
    rf_opt = get_optimized_rf(random_state=cfg['preprocessing']['random_state'])
    rf_opt.fit(X_train_final, y_train_balanced)
    _, y_pred, y_proba = evaluate_test_set(rf_opt, X_test_final, y_test)
    importances = pd.Series(rf_opt.feature_importances_, index=selected_genes).sort_values(ascending=False)

    plot_path = os.path.join(cfg['output']['figures_dir'], "TCGA_COAD_Results.png")
    plot_comprehensive_dashboard(y_test, y_pred, y_proba, importances, results_df, save_path=plot_path)
    print("✓ Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()