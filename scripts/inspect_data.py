"""Load and inspect a pinned UCI dataset so benchmark inputs are auditable."""

import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data/raw/german.data'
SOURCE_URL = 'https://archive.ics.uci.edu/static/public/144/statlog%2Bgerman%2Bcredit%2Bdata.zip'
DATA_SHA256 = 'b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871'
FEATURES = [
    'checking_status', 'duration_months', 'credit_history', 'purpose',
    'credit_amount', 'savings_status', 'employment', 'installment_rate',
    'personal_status_sex', 'other_debtors', 'residence_since', 'property',
    'age', 'other_installment_plans', 'housing', 'existing_credits',
    'job', 'dependents', 'telephone', 'foreign_worker',
]
NUMERIC = [
    'duration_months', 'credit_amount', 'installment_rate', 'residence_since',
    'age', 'existing_credits', 'dependents',
]
# A comprehension builds a list from the schema, not from observed test values.
CATEGORICAL = [name for name in FEATURES if name not in NUMERIC]


def verified_bytes(path: Path = DATA_PATH) -> bytes:
    """Read cached official bytes or fetch them; reject changed data, never fabricate it."""
    if path.exists():
        raw = path.read_bytes()
    else:
        try:
            # Context managers close the network response and archive automatically.
            with urlopen(SOURCE_URL, timeout=30) as response:
                archive_bytes = response.read()
            with ZipFile(io.BytesIO(archive_bytes)) as archive:
                raw = archive.read('german.data')
        except Exception as exc:
            raise RuntimeError('UCI download failed; restore data/raw/german.data or retry.') from exc
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError('Dataset SHA-256 mismatch; refusing unverified data.')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return raw


def parse_data(raw: bytes) -> tuple[pd.DataFrame, pd.Series]:
    """Parse the original mixed schema and map UCI 1=good, 2=bad to 0/1."""
    frame = pd.read_csv(io.BytesIO(raw), sep=r'\s+', header=None, na_values=['?'])
    if frame.shape != (1000, 21):
        raise ValueError(f'Expected 1000 rows and 21 columns; got {frame.shape}.')
    frame.columns = FEATURES + ['original_target']
    original = frame.pop('original_target')
    if original.isna().any() or set(original.unique()) != {1, 2}:
        raise ValueError('Targets must be original UCI labels 1 and 2 with no missing values.')
    # A65 (unknown/no savings) is a valid category, not a missing-value marker.
    for name in NUMERIC:
        frame[name] = pd.to_numeric(frame[name], errors='raise').astype(float)
    if np.isinf(frame[NUMERIC].to_numpy()).any():
        raise ValueError('Infinite numeric values are invalid.')
    for name in CATEGORICAL:
        frame[name] = frame[name].astype(object)
    target = original.map({1: 0, 2: 1}).astype(int).rename('bad_credit')
    return frame, target


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw features and aligned labels; learned preprocessing belongs in CV."""
    return parse_data(verified_bytes(path))


def summarize(X: pd.DataFrame, y: pd.Series) -> dict:
    """Describe input quality without learning transformations or selecting models."""
    counts = y.value_counts().sort_index()
    return {
        'n_samples': len(X), 'n_features': X.shape[1], 'target_shape': list(y.shape),
        'positive_class': '1 = bad credit; 0 = good credit (UCI 2 -> 1, 1 -> 0)',
        'class_counts': {str(label): int(count) for label, count in counts.items()},
        'class_percentages': {str(k): round(100 * v / len(y), 2) for k, v in counts.items()},
        'missing_feature_values': int(X.isna().sum().sum()),
        'missing_labels': int(y.isna().sum()),
        'all_numeric_features_finite': bool(np.isfinite(X[NUMERIC].to_numpy()).all()),
        'duplicate_feature_rows': int(X.duplicated().sum()),
        'numeric_features': NUMERIC, 'categorical_features': CATEGORICAL,
        'feature_names': X.columns.tolist(), 'sha256': DATA_SHA256,
    }


def main() -> None:
    """Write an inspection artifact independently of model training."""
    X, y = load_data()  # Tuple unpacking gives each returned object a name.
    result = json.dumps(summarize(X, y), indent=2) + '\n'
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    (reports / 'data_inspection.json').write_text(result, encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
