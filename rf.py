"""
Random Forest classifier for Marvel Rivals win prediction.
Includes training, evaluation, and hyperparameter tuning.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             classification_report, confusion_matrix, roc_curve)

from dataprep import load_data, get_feature_sets, get_train_test_split, get_feature_info


def train(X_train: pd.DataFrame, y_train: pd.Series,
          n_estimators: int = 100,
          max_depth: int = None,
          min_samples_split: int = 2,
          random_state: int = 42) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: RandomForestClassifier,
             X_test: pd.DataFrame,
             y_test: pd.Series) -> dict:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }


def plot_feature_importance(model: RandomForestClassifier,
                            feature_names: list,
                            output_path: str = 'data/model_results/rf_feature_importance.png',
                            top_n: int = 20):
    """Plot feature importance from Random Forest."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.title('Random Forest Feature Importance (Top 20)')
    plt.barh(range(top_n), importances[indices][::-1], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm: list,
                          output_path: str = 'data/model_results/rf_confusion_matrix.png'):
    """Plot confusion matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm = np.array(cm)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Random Forest Confusion Matrix')
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


def plot_roc_curve(model: RandomForestClassifier,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   output_path: str = 'data/model_results/rf_roc_curve.png'):
    """Plot ROC curve."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def tune_hyperparameters(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         cv: int = 5) -> dict:
    """Tune hyperparameters using GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy',
        n_jobs=-1, verbose=0
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
    """Perform cross-validation."""
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **model_params)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    return {
        'cv_scores': scores.tolist(),
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }


def run_feature_set_comparison(X: pd.DataFrame, y: pd.Series) -> dict:
    """Run model on different feature sets and compare."""
    results = {}
    feature_sets = get_feature_sets(pd.concat([X, y.rename('is_winner_team_one')], axis=1))

    for name, features in feature_sets.items():
        X_train, X_test, y_train, y_test = get_train_test_split(features, y)

        start_time = time.time()
        model = train(X_train, y_train)
        train_time = time.time() - start_time

        metrics = evaluate(model, X_test, y_test)
        metrics['train_time'] = train_time
        metrics['n_features'] = len(features.columns)

        results[name] = metrics

    return results


def main(X=None, y=None):
    """Run full Random Forest pipeline."""
    if X is None:
        X, y = load_data()

    feature_info = get_feature_info(X)

    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    start_time = time.time()
    model = train(X_train, y_train, n_estimators=100)
    train_time = time.time() - start_time

    metrics = evaluate(model, X_test, y_test)
    print(f"  Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc_roc']:.4f}")

    os.makedirs('data/model_results', exist_ok=True)

    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(model, X.columns.tolist())

    cv_results = cross_validate(X, y)
    print(f"  CV: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")

    results = {
        'model': 'RandomForest',
        'feature_info': feature_info,
        'metrics': metrics,
        'cv_results': cv_results,
        'train_time': train_time
    }

    with open('data/model_results/randomforest_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results, model, X_test, y_test


if __name__ == '__main__':
    main()
