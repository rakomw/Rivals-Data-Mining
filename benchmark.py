"""
Benchmark script to compare all tree ensemble models for Marvel Rivals win prediction.
Runs each model's main() function which handles training and visualization.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from dataprep import load_data, get_feature_info
import rf
import xgb
import cb


def plot_model_comparison(all_results: dict, feature_info: dict, output_path: str = 'data/model_results/model_comparison.png'):
    """Create bar chart comparing model metrics."""
    models = list(all_results.keys())
    metrics = ['accuracy', 'auc_roc', 'f1']

    x = np.arange(len(metrics))
    width = 0.25

    _, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        values = [all_results[model]['metrics'][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=model)

    ax.set_ylabel('Score')
    ax.set_title(f"Model Comparison\n({feature_info['description']})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy', 'AUC-ROC', 'F1'])
    ax.legend()
    ax.set_ylim(0.8, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_roc(models_data: dict, feature_info: dict, output_path: str = 'data/model_results/combined_roc.png'):
    """Create combined ROC curve plot for all models."""
    plt.figure(figsize=(8, 6))

    colors = {'Random Forest': 'blue', 'XGBoost': 'green', 'CatBoost': 'red'}

    for name, (model, X_test, y_test) in models_data.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color=colors.get(name, 'black'), label=f'{name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curves Comparison\n({feature_info['description']})")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Run tree methods benchmark."""
    os.makedirs('data/model_results', exist_ok=True)

    print("Loading data...")
    X, y = load_data()
    feature_info = get_feature_info(X)
    print(f"  {feature_info['description']}")

    tree_methods = [
        (rf, 'Random Forest'),
        (xgb, 'XGBoost'),
        (cb, 'CatBoost')
    ]

    all_results = {}
    models_data = {}

    for module, name in tree_methods:
        print(f"Running {name}...")
        results, model, X_test, y_test = module.main(X.copy(), y.copy())
        all_results[name] = results
        models_data[name] = (model, X_test, y_test)

    print("Generating comparison plots...")
    plot_model_comparison(all_results, feature_info)
    plot_combined_roc(models_data, feature_info)

    with open('data/model_results/benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("Done.")


if __name__ == '__main__':
    main()
