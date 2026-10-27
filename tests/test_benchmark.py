"""Test validation boundaries and known-answer metrics, then run a small real benchmark."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark import (METRICS, evaluate, make_models, positive_probability,
                                run_benchmark, split_data)
from scripts.inspect_data import NUMERIC, load_data


def test_split_separation_and_reproducibility():
    X, y = load_data()
    train, holdout, train_y, holdout_y = split_data(X, y)
    repeat = split_data(X, y)
    assert set(train.index).isdisjoint(holdout.index)
    assert set(train.index) | set(holdout.index) == set(X.index)
    assert (len(train), len(holdout)) == (800, 200)
    assert train_y.value_counts().to_dict() == {0: 560, 1: 240}
    assert holdout_y.value_counts().to_dict() == {0: 140, 1: 60}
    pd.testing.assert_frame_equal(train, repeat[0])
    pd.testing.assert_series_equal(holdout_y, repeat[3])


def test_known_metrics_and_cost_orientation():
    # A deliberately asymmetric matrix catches accidentally swapped FP and FN.
    result = evaluate([0, 0, 0, 1, 1, 1], np.array([.1, .2, .8, .3, .4, .9]))
    assert result['confusion_matrix'] == [[2, 1], [2, 1]]
    assert result['accuracy'] == pytest.approx(1 / 2)
    assert result['precision'] == pytest.approx(1 / 2)
    assert result['recall'] == pytest.approx(1 / 3)
    assert result['f1'] == pytest.approx(2 / 5)
    assert result['roc_auc'] == pytest.approx(7 / 9)
    assert result['total_cost'] == 11
    assert result['mean_cost'] == pytest.approx(11 / 6)
    assert evaluate([0, 1], np.array([.1, .5]))['confusion_matrix'] == [[1, 0], [0, 1]]
    assert evaluate([0, 1], np.array([.1, .2]))['precision'] == 0


@pytest.mark.parametrize('bad', [np.array([[.1, .2]]), np.array([np.nan, .2]), np.array([-.1, .2])])
def test_bad_probability_input_rejected(bad):
    with pytest.raises(ValueError):
        evaluate([0, 1], bad)


@pytest.mark.parametrize('name', ['Logistic Regression', 'Random Forest'])
def test_preprocessing_missing_unseen_and_train_only(name):
    X, y = load_data()
    train, holdout, train_y, _ = split_data(X, y)
    train = train.copy()
    holdout = holdout.copy()
    train.loc[train.index[0], 'credit_amount'] = np.nan
    train.loc[train.index[0], 'purpose'] = np.nan
    holdout['purpose'] = 'UNSEEN_HOLDOUT_ONLY'
    holdout['credit_amount'] = 1e12
    holdout.loc[holdout.index[0], 'duration_months'] = np.nan
    model = make_models(42, 8)[name].fit(train, train_y)
    pre = model['preprocess']
    imputer = pre.named_transformers_['numeric']['impute']
    np.testing.assert_allclose(imputer.statistics_, train[NUMERIC].median())
    encoder = pre.named_transformers_['categorical']['encode']
    assert all('UNSEEN_HOLDOUT_ONLY' not in values for values in encoder.categories_)
    transformed_train = pre.transform(train)
    transformed_test = pre.transform(holdout)
    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert np.isfinite(transformed_test).all()
    if name == 'Logistic Regression':
        expected = imputer.transform(train[NUMERIC]).mean(axis=0)
        np.testing.assert_allclose(pre.named_transformers_['numeric']['scale'].mean_, expected)
    probabilities = positive_probability(model, holdout)
    assert probabilities.shape == (200,)
    np.testing.assert_allclose(probabilities, model.predict_proba(holdout)[:, list(model.classes_).index(1)])


def test_end_to_end_small_and_reproducible(tmp_path):
    first = run_benchmark(tmp_path / 'first', n_splits=2, n_trees=8)
    second = run_benchmark(tmp_path / 'second', n_splits=2, n_trees=8)
    assert first == second
    train = set(first['split']['train_indices'])
    holdout = set(first['split']['holdout_indices'])
    validation_seen = []
    for fold in first['cv_indices']:
        fit, validation = set(fold['fit']), set(fold['validation'])
        assert fit.isdisjoint(validation)
        assert fit | validation == train
        assert (fit | validation).isdisjoint(holdout)
        validation_seen.extend(fold['validation'])
    assert sorted(validation_seen) == sorted(train)
    for model in first['models'].values():
        for score in [model['train'], model['holdout'], *model['cv']['folds']]:
            for metric in METRICS:
                assert 0 <= score[metric] <= (5 if metric == 'mean_cost' else 1)
        assert np.array(model['holdout']['confusion_matrix']).sum() == 200
        for metric in METRICS:
            values = [fold[metric] for fold in model['cv']['folds']]
            assert model['cv']['summary'][metric]['std'] == pytest.approx(np.std(values, ddof=1))
    saved = json.loads((tmp_path / 'first/benchmark_results.json').read_text())
    assert saved['models'] == first['models']
    assert pd.read_csv(tmp_path / 'first/holdout_predictions.csv').shape == (200, 6)
    assert (tmp_path / 'first/confusion_matrices.png').stat().st_size > 0
    report = (tmp_path / 'first/validation_report.md').read_text()
    assert 'FP + 5×FN' in report and 'Logistic Regression' in report
