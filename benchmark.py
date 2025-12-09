"""
Benchmark script to compare all tree ensemble models for Marvel Rivals win prediction.
Runs all models, compares performance, and generates summary reports.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from typing import Dict, Any

from dataprep import load_data, get_feature_sets, get_train_test_split

# Import model modules
import rf
import xgb
import cb
import ab


def run_single_model(model_name: str, train_fn, evaluate_fn,
                     X_train, X_test, y_train, y_test,
                     **kwargs) -> Dict[str, Any]:
    """Run a single model and return results.

    Args:
        model_name: Name of the model.
        train_fn: Training function.
        evaluate_fn: Evaluation function.
        X_train, X_test: Training and test features.
        y_train, y_test: Training and test labels.
        **kwargs: Additional arguments for training.

    Returns:
        Dictionary with model results.
    """
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")

    start_time = time.time()
    model = train_fn(X_train, y_train, **kwargs)
    train_time = time.time() - start_time

    metrics = evaluate_fn(model, X_test, y_test)
    metrics['train_time'] = train_time

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Train Time: {train_time:.2f}s")

    return {
        'model': model,
        'metrics': metrics
    }


def run_all_models(X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """Run all tree ensemble models.

    Args:
        X_train, X_test: Training and test features.
        y_train, y_test: Training and test labels.

    Returns:
        Dictionary with results for all models.
    """
    results = {}

    # Random Forest
    results['RandomForest'] = run_single_model(
        'Random Forest',
        rf.train, rf.evaluate,
        X_train, X_test, y_train, y_test,
        n_estimators=100
    )

    # XGBoost
    results['XGBoost'] = run_single_model(
        'XGBoost',
        xgb.train, xgb.evaluate,
        X_train, X_test, y_train, y_test,
        n_estimators=100
    )

    # AdaBoost
    results['AdaBoost'] = run_single_model(
        'AdaBoost',
        ab.train, ab.evaluate,
        X_train, X_test, y_train, y_test,
        n_estimators=100
    )

    # CatBoost
    results['CatBoost'] = run_single_model(
        'CatBoost',
        cb.train, cb.evaluate,
        X_train, X_test, y_train, y_test,
        iterations=100
    )

    return results


def run_feature_set_benchmark(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
    """Run all models on all feature sets.

    Args:
        X: Full feature DataFrame.
        y: Target Series.

    Returns:
        Nested dictionary: {feature_set: {model: metrics}}.
    """
    feature_sets = get_feature_sets(pd.concat([X, y.rename('is_winner_team_one')], axis=1))
    all_results = {}

    for fs_name, features in feature_sets.items():
        print(f"\n{'#'*60}")
        print(f"FEATURE SET: {fs_name}")
        print(f"Number of features: {len(features.columns)}")
        print(f"{'#'*60}")

        X_train, X_test, y_train, y_test = get_train_test_split(features, y)
        results = run_all_models(X_train, X_test, y_train, y_test)

        all_results[fs_name] = {
            model_name: result['metrics']
            for model_name, result in results.items()
        }

    return all_results


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a comparison table from benchmark results.

    Args:
        results: Nested dictionary from run_feature_set_benchmark.

    Returns:
        DataFrame with comparison table.
    """
    rows = []
    for fs_name, models in results.items():
        for model_name, metrics in models.items():
            rows.append({
                'Feature Set': fs_name,
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'AUC-ROC': metrics['auc_roc'],
                'F1': metrics['f1'],
                'Train Time (s)': metrics['train_time']
            })

    df = pd.DataFrame(rows)
    return df.sort_values(['Feature Set', 'Accuracy'], ascending=[True, False])


def plot_model_comparison(results: Dict[str, Dict],
                          output_path: str = 'data/model_results/benchmark_comparison.png'):
    """Create comparison visualization.

    Args:
        results: Nested dictionary from run_feature_set_benchmark.
        output_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = create_comparison_table(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Accuracy by model and feature set
    ax1 = axes[0, 0]
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        ax1.bar(
            [f"{fs}\n({model[:3]})" for fs in model_data['Feature Set']],
            model_data['Accuracy'],
            label=model,
            alpha=0.8
        )
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Model and Feature Set')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # AUC-ROC comparison
    ax2 = axes[0, 1]
    pivot_auc = df.pivot(index='Feature Set', columns='Model', values='AUC-ROC')
    pivot_auc.plot(kind='bar', ax=ax2)
    ax2.set_ylabel('AUC-ROC')
    ax2.set_title('AUC-ROC by Model and Feature Set')
    ax2.legend(loc='lower right')
    ax2.tick_params(axis='x', rotation=45)

    # Training time comparison
    ax3 = axes[1, 0]
    pivot_time = df.pivot(index='Feature Set', columns='Model', values='Train Time (s)')
    pivot_time.plot(kind='bar', ax=ax3)
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Training Time by Model and Feature Set')
    ax3.legend(loc='upper right')
    ax3.tick_params(axis='x', rotation=45)

    # Best model per feature set
    ax4 = axes[1, 1]
    best_per_fs = df.loc[df.groupby('Feature Set')['Accuracy'].idxmax()]
    colors = plt.cm.Set2(range(len(best_per_fs)))
    bars = ax4.barh(best_per_fs['Feature Set'], best_per_fs['Accuracy'], color=colors)
    ax4.set_xlabel('Accuracy')
    ax4.set_title('Best Model per Feature Set')

    # Add model names to bars
    for bar, model in zip(bars, best_per_fs['Model']):
        ax4.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 model, va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {output_path}")


def plot_roc_curves_comparison(X_test: pd.DataFrame, y_test: pd.Series,
                               models: Dict[str, Any],
                               output_path: str = 'data/model_results/benchmark_roc_curves.png'):
    """Plot ROC curves for all models.

    Args:
        X_test: Test features.
        y_test: Test labels.
        models: Dictionary of trained models.
        output_path: Path to save the plot.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green', 'purple']
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison - All Models')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curves comparison saved to {output_path}")


def generate_summary_report(results: Dict[str, Dict],
                            output_path: str = 'data/model_results/benchmark_summary.txt'):
    """Generate a text summary report.

    Args:
        results: Benchmark results.
        output_path: Path to save the report.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = create_comparison_table(results)

    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MARVEL RIVALS WIN PREDICTION - TREE ENSEMBLE BENCHMARK SUMMARY\n")
        f.write("="*70 + "\n\n")

        # Overall best
        best_overall = df.loc[df['Accuracy'].idxmax()]
        f.write("BEST OVERALL MODEL:\n")
        f.write(f"  Model: {best_overall['Model']}\n")
        f.write(f"  Feature Set: {best_overall['Feature Set']}\n")
        f.write(f"  Accuracy: {best_overall['Accuracy']:.4f}\n")
        f.write(f"  AUC-ROC: {best_overall['AUC-ROC']:.4f}\n")
        f.write(f"  F1 Score: {best_overall['F1']:.4f}\n\n")

        # Best per feature set
        f.write("-"*70 + "\n")
        f.write("BEST MODEL PER FEATURE SET:\n")
        f.write("-"*70 + "\n")
        for fs in df['Feature Set'].unique():
            fs_data = df[df['Feature Set'] == fs]
            best = fs_data.loc[fs_data['Accuracy'].idxmax()]
            f.write(f"\n{fs}:\n")
            f.write(f"  Best: {best['Model']} (Acc: {best['Accuracy']:.4f})\n")

        # Full comparison table
        f.write("\n" + "="*70 + "\n")
        f.write("FULL COMPARISON TABLE:\n")
        f.write("="*70 + "\n\n")
        f.write(df.to_string(index=False))

        # Model rankings
        f.write("\n\n" + "="*70 + "\n")
        f.write("MODEL RANKINGS (by average accuracy across feature sets):\n")
        f.write("="*70 + "\n\n")
        avg_by_model = df.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
        for rank, (model, acc) in enumerate(avg_by_model.items(), 1):
            f.write(f"  {rank}. {model}: {acc:.4f}\n")

        # Feature set rankings
        f.write("\n" + "="*70 + "\n")
        f.write("FEATURE SET RANKINGS (by best accuracy):\n")
        f.write("="*70 + "\n\n")
        max_by_fs = df.groupby('Feature Set')['Accuracy'].max().sort_values(ascending=False)
        for rank, (fs, acc) in enumerate(max_by_fs.items(), 1):
            f.write(f"  {rank}. {fs}: {acc:.4f}\n")

    print(f"\nSummary report saved to {output_path}")


def main():
    """Run full benchmark comparison."""
    print("="*70)
    print("MARVEL RIVALS WIN PREDICTION - TREE ENSEMBLE BENCHMARK")
    print("="*70)

    # Load data
    print("\nLoading data...")
    X, y = load_data()
    print(f"Loaded {len(X)} samples with {len(X.columns)} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Create output directory
    os.makedirs('data/model_results', exist_ok=True)

    # Run benchmark on all feature sets
    print("\n" + "="*70)
    print("RUNNING BENCHMARK ON ALL FEATURE SETS")
    print("="*70)
    results = run_feature_set_benchmark(X, y)

    # Create comparison table
    comparison_df = create_comparison_table(results)
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print(comparison_df.to_string(index=False))

    # Save comparison table
    comparison_df.to_csv('data/model_results/benchmark_comparison.csv', index=False)
    print("\nComparison table saved to data/model_results/benchmark_comparison.csv")

    # Plot comparison
    plot_model_comparison(results)

    # Run on full dataset for ROC comparison
    print("\n" + "="*70)
    print("GENERATING ROC CURVES COMPARISON (full feature set)")
    print("="*70)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    full_results = run_all_models(X_train, X_test, y_train, y_test)
    trained_models = {name: result['model'] for name, result in full_results.items()}
    plot_roc_curves_comparison(X_test, y_test, trained_models)

    # Generate summary report
    generate_summary_report(results)

    # Save full results
    serializable_results = {}
    for fs_name, models in results.items():
        serializable_results[fs_name] = {}
        for model_name, metrics in models.items():
            serializable_results[fs_name][model_name] = {
                k: v for k, v in metrics.items()
                if k != 'classification_report'  # This can be verbose
            }

    with open('data/model_results/benchmark_full_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print("\nFull results saved to data/model_results/benchmark_full_results.json")

    # Print final summary
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    best = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    print(f"\nBest overall: {best['Model']} on {best['Feature Set']}")
    print(f"  Accuracy: {best['Accuracy']:.4f}")
    print(f"  AUC-ROC: {best['AUC-ROC']:.4f}")
    print(f"  F1 Score: {best['F1']:.4f}")

    return results


if __name__ == '__main__':
    main()
