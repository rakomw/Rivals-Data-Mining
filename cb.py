"""
CatBoost classifier for Marvel Rivals win prediction.
Includes training, evaluation, SHAP analysis, and hyperparameter tuning.
CatBoost excels at handling categorical features (like hero IDs) natively.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             classification_report, confusion_matrix, roc_curve)
import shap

from dataprep import load_data, get_feature_sets, get_train_test_split


def train(X_train: pd.DataFrame, y_train: pd.Series,
          iterations: int = 100,
          depth: int = 6,
          learning_rate: float = 0.1,
          l2_leaf_reg: float = 3.0,
          random_state: int = 42) -> CatBoostClassifier:
    """Train CatBoost classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        iterations: Number of boosting iterations.
        depth: Depth of the trees.
        learning_rate: Learning rate.
        l2_leaf_reg: L2 regularization coefficient.
        random_state: Random seed for reproducibility.

    Returns:
        Trained CatBoostClassifier model.
    """
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_state=random_state,
        verbose=0,
        thread_count=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: CatBoostClassifier,
             X_test: pd.DataFrame,
             y_test: pd.Series) -> dict:
    """Evaluate model performance.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary with evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }


def get_shap_values(model: CatBoostClassifier,
                    X_test: pd.DataFrame,
                    max_samples: int = 1000) -> tuple:
    """Generate SHAP values for model interpretability.

    Args:
        model: Trained model.
        X_test: Test features.
        max_samples: Maximum samples for SHAP calculation.

    Returns:
        Tuple of (shap_values, explainer, X_sample).
    """
    if len(X_test) > max_samples:
        X_sample = X_test.sample(n=max_samples, random_state=42)
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return shap_values, explainer, X_sample


def plot_shap_summary(shap_values, X_sample: pd.DataFrame,
                      output_path: str = 'data/model_results/catboost_shap_summary.png'):
    """Create and save SHAP summary plot.

    Args:
        shap_values: SHAP values from explainer.
        X_sample: Sample features used for SHAP.
        output_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {output_path}")


def plot_feature_importance(model: CatBoostClassifier,
                            feature_names: list,
                            output_path: str = 'data/model_results/catboost_feature_importance.png',
                            top_n: int = 20):
    """Plot feature importance from CatBoost.

    Args:
        model: Trained model.
        feature_names: List of feature names.
        output_path: Path to save the plot.
        top_n: Number of top features to display.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.title('CatBoost Feature Importance (Top 20)')
    plt.barh(range(top_n), importances[indices][::-1], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {output_path}")


def plot_confusion_matrix(cm: list,
                          output_path: str = 'data/model_results/catboost_confusion_matrix.png'):
    """Plot confusion matrix.

    Args:
        cm: Confusion matrix as list.
        output_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm = np.array(cm)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('CatBoost Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Team 2 Wins', 'Team 1 Wins'])
    plt.yticks([0, 1], ['Team 2 Wins', 'Team 1 Wins'])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(model: CatBoostClassifier,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   output_path: str = 'data/model_results/catboost_roc_curve.png'):
    """Plot ROC curve.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        output_path: Path to save the plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CatBoost ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {output_path}")


def tune_hyperparameters(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         cv: int = 5) -> dict:
    """Tune hyperparameters using GridSearchCV.

    Args:
        X_train: Training features.
        y_train: Training labels.
        cv: Number of cross-validation folds.

    Returns:
        Dictionary with best parameters and scores.
    """
    param_grid = {
        'iterations': [50, 100, 200],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.3],
        'l2_leaf_reg': [1, 3, 5]
    }

    model = CatBoostClassifier(random_state=42, verbose=0, thread_count=-1)
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': {
            'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': grid_search.cv_results_['std_test_score'].tolist()
        }
    }


def cross_validate(X: pd.DataFrame, y: pd.Series,
                   n_splits: int = 5,
                   **model_params) -> dict:
    """Perform cross-validation.

    Args:
        X: Features.
        y: Labels.
        n_splits: Number of CV folds.
        **model_params: Parameters for the model.

    Returns:
        Dictionary with CV scores.
    """
    model = CatBoostClassifier(random_state=42, verbose=0, thread_count=-1, **model_params)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    return {
        'cv_scores': scores.tolist(),
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }


def run_feature_set_comparison(X: pd.DataFrame, y: pd.Series) -> dict:
    """Run model on different feature sets and compare.

    Args:
        X: Full feature DataFrame.
        y: Target Series.

    Returns:
        Dictionary with results for each feature set.
    """
    results = {}
    feature_sets = get_feature_sets(pd.concat([X, y.rename('is_winner_team_one')], axis=1))

    for name, features in feature_sets.items():
        print(f"\n{'='*50}")
        print(f"Training on feature set: {name}")
        print(f"Number of features: {len(features.columns)}")
        print(f"{'='*50}")

        X_train, X_test, y_train, y_test = get_train_test_split(features, y)

        start_time = time.time()
        model = train(X_train, y_train)
        train_time = time.time() - start_time

        metrics = evaluate(model, X_test, y_test)
        metrics['train_time'] = train_time
        metrics['n_features'] = len(features.columns)

        results[name] = metrics

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Train Time: {train_time:.2f}s")

    return results


def main():
    """Run full CatBoost pipeline."""
    print("="*60)
    print("CATBOOST - Marvel Rivals Win Prediction")
    print("="*60)

    # Load data
    print("\nLoading data...")
    X, y = load_data()
    print(f"Loaded {len(X)} samples with {len(X.columns)} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    print(f"\nTrain set: {len(X_train)}, Test set: {len(X_test)}")

    # Train model
    print("\nTraining CatBoost...")
    start_time = time.time()
    model = train(X_train, y_train, iterations=100)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")

    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate(model, X_test, y_test)
    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")

    # Create output directory
    os.makedirs('data/model_results', exist_ok=True)

    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])

    # Plot ROC curve
    plot_roc_curve(model, X_test, y_test)

    # Plot feature importance
    plot_feature_importance(model, X.columns.tolist())

    # SHAP analysis
    print("\nGenerating SHAP values...")
    try:
        shap_values, explainer, X_sample = get_shap_values(model, X_test)
        plot_shap_summary(shap_values, X_sample)
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

    # Cross-validation
    print("\nRunning cross-validation...")
    cv_results = cross_validate(X, y)
    print(f"CV Accuracy: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")

    # Feature set comparison
    print("\n" + "="*60)
    print("FEATURE SET COMPARISON")
    print("="*60)
    feature_results = run_feature_set_comparison(X, y)

    # Save results
    results = {
        'model': 'CatBoost',
        'metrics': metrics,
        'cv_results': cv_results,
        'feature_set_comparison': feature_results,
        'train_time': train_time
    }

    with open('data/model_results/catboost_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to data/model_results/catboost_results.json")

    return results


if __name__ == '__main__':
    main()
