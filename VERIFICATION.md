# Verification

## Scope

The benchmark verifies input integrity, preprocessing boundaries, metric calculations and repeatable execution. It does not establish deployment suitability or cross-platform bitwise reproducibility.

## Checks

| Area | Evidence |
|---|---|
| Input integrity | Official data verified against a pinned SHA-256; unexpected bytes and failed downloads are rejected |
| Data contract | 1,000 rows, 20 features; original labels mapped row-by-row to good=0 and bad=1; counts 700/300 |
| Evaluation boundaries | Disjoint training/holdout row IDs; CV folds restricted to training rows |
| Preprocessing | Training-only imputation/scaling; missing values and unseen categories handled in tests |
| Metrics | Known-answer confusion matrices, positive-class probabilities, metric ranges and asymmetric costs |
| Repeatability | Two complete 2-fold, 8-tree runs compared in temporary output directories |
| Artifacts | JSON metrics, prediction CSV, Markdown report and confusion-matrix PNG generated successfully |

The existing suite contains **14 passing tests**. `pytest.ini` treats warnings as errors. Dependency consistency is checked with `pip check`.

## Commands

From the repository root, after activating the environment described in the [README](README.md):

```bash
python -m pip check
python scripts/inspect_data.py
python -m pytest -q
python -m scripts.benchmark
```

The default benchmark uses 5 folds, 200 Random Forest trees and seed 42. The full [validation report](reports/validation_report.md) and [machine-readable results](reports/benchmark_results.json) record the resulting metrics, settings and data boundaries.

## Environment

The recorded environment uses Python 3.12.14. Dependencies are pinned in [requirements-lock.txt](requirements-lock.txt); SciPy 1.14.1 and pyparsing 3.2.3 are compatible with the pinned scikit-learn and matplotlib releases. No warning exemptions are configured.

Matplotlib defaults to the ignored project-local `.mpl-cache` directory and respects an explicit `MPLCONFIGDIR` setting.

## Interpretation and remaining checks

Random Forest's training-to-holdout ROC-AUC gap and both models' costs relative to constant baselines are documented in the report. Successful execution does not resolve those statistical limitations.

A fresh operating-system installation and Windows execution have not been verified. Fairness, calibration, temporal/external validation, realistic banking costs and production readiness remain outside the completed validation scope. This is not a security audit.
