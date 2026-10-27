"""Run a fixed training-only CV protocol and an untouched holdout evaluation."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.inspect_data import CATEGORICAL, NUMERIC, ROOT, load_data, summarize
from scripts.reporting import write_report

SEED = 42
THRESHOLD = 0.5  # Fixed before evaluation; never choose a threshold on the holdout.
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mean_cost']


def split_data(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> tuple:
    """Reserve 20% once; stratification preserves the good/bad class proportions."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def build_pipeline(model: BaseEstimator, scale_numeric: bool) -> Pipeline:
    """WHAT: Combine preprocessing and classifier. HOW: Transform columns by type.

    WHY: CV fits the entire Pipeline on each training fold, preventing leakage.
    """
    numeric_steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    # LR benefits from comparable scales; trees split by order and do not need scaling.
    if scale_numeric:
        numeric_steps.append(('scale', StandardScaler()))
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        # Unseen validation categories become all-zero indicators instead of errors.
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), NUMERIC),
        ('categorical', categorical, CATEGORICAL),
    ])
    return Pipeline([('preprocess', preprocessing), ('model', model)])


def make_models(seed: int, n_trees: int) -> dict[str, Pipeline]:
    """Use predeclared settings, not a search driven by holdout performance."""
    return {
        'Logistic Regression': build_pipeline(
            LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=seed), True),
        'Random Forest': build_pipeline(
            RandomForestClassifier(n_estimators=n_trees, max_depth=None,
                                   random_state=seed, n_jobs=1), False),
    }


def positive_probability(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Select P(bad=1) explicitly instead of assuming a probability column order."""
    column = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(X)[:, column]


def evaluate(y_true: pd.Series | list[int], probabilities: np.ndarray) -> dict:
    """Measure bad-credit detection and UCI costs at a predeclared 0.5 threshold."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 1 or len(probabilities) != len(y_true):
        raise ValueError('Expected one positive-class probability per row.')
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities must be finite and in [0, 1].')
    predicted = (probabilities >= THRESHOLD).astype(int)
    # Rows = actual, columns = predicted, labels=[good, bad]: [[TN, FP], [FN, TP]].
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    total_cost = int(fp + 5 * fn)  # Missing a bad risk costs 5; rejecting a good risk costs 1.
    return {
        'accuracy': float(accuracy_score(y_true, predicted)),
        'precision': float(precision_score(y_true, predicted, pos_label=1, zero_division=0)),
        'recall': float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        'f1': float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
        # ROC-AUC uses probabilities, not the thresholded labels.
        'roc_auc': float(roc_auc_score(y_true, probabilities)),
        'confusion_matrix': matrix.tolist(),
        'total_cost': total_cost, 'mean_cost': total_cost / len(y_true),
    }


def cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, folds: list) -> dict:
    """Fit fresh copies on identical training folds and summarize validation variability."""
    scores = []
    for fit_rows, validation_rows in folds:
        # iloc selects positions within the training set, never holdout row IDs.
        fitted = clone(pipeline).fit(X.iloc[fit_rows], y.iloc[fit_rows])
        scores.append(evaluate(y.iloc[validation_rows],
                               positive_probability(fitted, X.iloc[validation_rows])))
    return {
        'folds': scores,
        'summary': {metric: {
            'mean': float(np.mean([score[metric] for score in scores])),
            'std': float(np.std([score[metric] for score in scores], ddof=1)),
        } for metric in METRICS},
    }


def run_benchmark(output_dir: Path, seed: int = SEED, n_splits: int = 5,
                  n_trees: int = 200) -> dict:
    """Execute a reproducible comparison, saving provenance, row IDs and actual metrics."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, seed)
    # One materialized fold list ensures both models see exactly the same folds.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(X_train, y_train))
    results = {
        'dataset': summarize(X, y),
        'environment': {'python': platform.python_version(),
                        **{name: version(name) for name in
                           ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'pytest',
                            'scipy', 'pyparsing', 'joblib']}},
        'config': {'seed': seed, 'n_splits': n_splits, 'n_trees': n_trees,
                   'holdout_fraction': 0.2, 'threshold': THRESHOLD,
                   'lr_C': 1.0, 'lr_solver': 'lbfgs', 'lr_max_iter': 2000,
                   'rf_max_depth': None, 'cost_fp': 1, 'cost_fn': 5},
        'split': {'train_indices': X_train.index.tolist(),
                  'holdout_indices': X_test.index.tolist(),
                  'train_counts': y_train.value_counts().sort_index().to_dict(),
                  'holdout_counts': y_test.value_counts().sort_index().to_dict()},
        'cv_indices': [{'fit': X_train.index[a].tolist(),
                        'validation': X_train.index[b].tolist()} for a, b in folds],
        'models': {},
    }
    predictions = pd.DataFrame({'row_id': X_test.index, 'actual_bad': y_test.to_numpy()})
    for name, pipeline in make_models(seed, n_trees).items():
        cv_results = cross_validate(pipeline, X_train, y_train, folds)
        fitted = clone(pipeline).fit(X_train, y_train)
        train_probability = positive_probability(fitted, X_train)
        holdout_probability = positive_probability(fitted, X_test)
        results['models'][name] = {
            'cv': cv_results, 'train': evaluate(y_train, train_probability),
            'holdout': evaluate(y_test, holdout_probability),
            'transformed_features': len(fitted['preprocess'].get_feature_names_out()),
        }
        predictions[name + '_probability_bad'] = holdout_probability
        predictions[name + '_predicted_bad'] = (holdout_probability >= THRESHOLD).astype(int)
    # Predeclared constant baselines contextualize accuracy and the asymmetric costs.
    results['baselines'] = {
        'always_good': evaluate(y_test, np.zeros(len(y_test))),
        'always_bad': evaluate(y_test, np.ones(len(y_test))),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'benchmark_results.json').write_text(
        json.dumps(results, indent=2) + '\n', encoding='utf-8')
    predictions.to_csv(output_dir / 'holdout_predictions.csv', index=False)
    write_report(results, output_dir)
    return results


def main() -> None:
    """Expose full and small smoke configurations without changing the protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--trees', type=int, default=200)
    args = parser.parse_args()
    results = run_benchmark(args.output_dir, n_splits=args.folds, n_trees=args.trees)
    for name, result in results['models'].items():
        print(name, json.dumps(result['holdout']))
    print(f'Report: {args.output_dir / "validation_report.md"}')


if __name__ == '__main__':
    main()
