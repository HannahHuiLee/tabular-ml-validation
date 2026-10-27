"""Check the official input contract so label or schema errors cannot pass silently."""

import numpy as np
import pandas as pd
import pytest

from scripts.inspect_data import (CATEGORICAL, DATA_PATH, FEATURES, NUMERIC,
                                  load_data, parse_data, summarize, verified_bytes)


def test_shapes_schema_and_label_convention():
    X, y = load_data()
    raw_labels = np.loadtxt(DATA_PATH, dtype=str)[:, -1].astype(int)
    assert X.shape == (1000, 20)
    assert y.shape == (1000,)
    assert X.columns.tolist() == FEATURES
    assert len(NUMERIC) == 7 and len(CATEGORICAL) == 13
    assert X.index.equals(y.index) and X.index.is_unique
    np.testing.assert_array_equal(y, raw_labels == 2)
    assert y.value_counts().to_dict() == {0: 700, 1: 300}
    assert X[NUMERIC].dtypes.eq('float64').all()
    assert X[CATEGORICAL].dtypes.eq('object').all()


def test_loading_is_reproducible():
    X1, y1 = load_data()
    X2, y2 = load_data()
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_summary_and_missing_markers():
    X, y = load_data()
    result = summarize(X, y)
    assert result['missing_feature_values'] == result['missing_labels'] == 0
    assert result['all_numeric_features_finite']
    # Deliberately inject markers into an in-memory copy, never the official file.
    rows = [line.split() for line in DATA_PATH.read_text().splitlines()]
    rows[0][0] = rows[0][1] = '?'
    changed = ('\n'.join(' '.join(row) for row in rows)).encode()
    changed_X, _ = parse_data(changed)
    assert changed_X.iloc[0, :2].isna().all()
    assert 'A65' in X['savings_status'].values


def test_invalid_target_rejected():
    rows = DATA_PATH.read_text().splitlines()
    values = rows[0].split()
    values[-1] = '3'
    rows[0] = ' '.join(values)
    with pytest.raises(ValueError, match='Targets'):
        parse_data('\n'.join(rows).encode())


def test_bad_checksum_rejected(tmp_path):
    path = tmp_path / 'german.data'
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='SHA-256'):
        verified_bytes(path)


def test_download_failure_is_explicit(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('offline')
    monkeypatch.setattr('scripts.inspect_data.urlopen', fail)
    with pytest.raises(RuntimeError, match='UCI download failed'):
        verified_bytes(tmp_path / 'absent.data')
