# Migration verification and review checklist

## Scope and preserved history

Updated the existing `outputs/tabular-validation` project in place. The starting directory was not a Git repository (`git rev-parse --show-toplevel` returned that fact). No repository was created and nothing was pushed. The former README, loader, tests, requirements and three inspection/report artifacts are preserved in `reports/history/stage1_breast_cancer/`.

## Modified existing files

- `README.md`: current purpose, provenance, protocol, reproducible commands, limitations and learning guide.
- `.gitignore`: environments, caches, secrets, editor/OS files and optional smoke outputs.
- `requirements.txt`, `requirements-lock.txt`: retain the five requested libraries; constrain compatible scipy/pyparsing versions.
- `scripts/inspect_data.py`: keep the loading/summary entry points, migrate to pinned UCI bytes, schema and explicit target mapping.
- `tests/test_data.py`: replace dataset-specific assertions with German Credit contracts and failure handling.
- `reports/data_inspection.json`, `reports/dataset_description.txt`, `reports/validation_report.md`: current dataset and actual results; previous files were archived first.

## Added files

- `scripts/benchmark.py`: shared split/CV, preprocessing, model comparison, metrics, UCI costs and prediction export.
- `scripts/reporting.py`: generated report, cost interpretation and confusion-matrix visualization.
- `tests/test_benchmark.py`: boundaries, known-answer metrics, missing/unseen values and reproducible end-to-end smoke execution.
- `pytest.ini`: collect current tests only; treat warnings as errors.
- `data/README.md`, `data/raw/german.data`, `data/raw/german.doc`: official source, attribution, license and unmodified cached data.
- `reports/benchmark_results.json`, `reports/holdout_predictions.csv`, `reports/confusion_matrices.png`: generated results.
- `reports/history/stage1_breast_cancer/`: archived first-stage material.
- `VERIFICATION.md`: this review record.

## Actual commands executed

Commands below ran in the existing project directory; the existing Python 3.12.14 `.venv` was reused. No new environment was claimed to have been created during this migration. README includes creation commands for a fresh checkout.

```bash
# Official data download; extracted german.data and german.doc using Python ZipFile.
curl --fail --location --max-time 60 'https://archive.ics.uci.edu/static/public/144/statlog%2Bgerman%2Bcredit%2Bdata.zip' -o ../../work/german-credit.zip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/inspect_data.py
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.benchmark
```

Also read the original files, checked Git status, computed raw-file SHA-256 hashes, inspected the generated Markdown and visually reviewed the confusion-matrix PNG. The smoke configuration (2 folds, 8 trees) ran twice inside pytest using temporary output directories, not as a separate CLI command.

## Outcomes

- Final pytest execution: **14 passed in 1.85s**, exit code 0; warnings treated as errors.
- `pip check`: **No broken requirements found**.
- Data: **1000 × 20**, original label mapping verified row-by-row; good=700, bad=300; missing feature/label values=0; completely duplicated feature rows=0.
- Formal benchmark: **5 folds, 200 trees, seed 42**, exit code 0. Both models use the same 800/200 split and training-only CV folds.
- LR holdout: accuracy 0.7800, precision 0.6667, recall 0.5333, F1 0.5926, ROC-AUC 0.8040, total cost 156.
- RF holdout: accuracy 0.7550, precision 0.6410, recall 0.4167, F1 0.5051, ROC-AUC 0.7848, total cost 189.
- The report contains training metrics, CV mean/sample standard deviation, actual confusion matrices and constant baselines. No holdout-based search or threshold selection was run.

## Issues encountered and resolution

The original lock had scipy 1.18.1 and pyparsing 3.3.3. Strict tests exposed an sklearn solver-option compatibility warning and matplotlib parsing deprecations. A temporary deprecation filter was removed after pinning scipy 1.14.1 and pyparsing 3.2.3; the final test configuration contains no warning exemptions. Model settings and evaluation rules were unchanged. The benchmark was rerun after the environment correction, and final artifacts supersede the first warning-producing run.

Matplotlib's default home cache was unwritable in this execution environment. The report module now defaults its cache to the ignored project-local `.mpl-cache`, while respecting any explicit MPLCONFIGDIR. pip could not use the user's global cache, so installation proceeded without caching; this did not block installation.

## Remaining limits

No known failing tests or blocked implementation steps. Reproducibility was verified on this machine; a fresh operating-system install and cross-platform bitwise identity were not tested. This is not a full security audit. Fairness, calibration, temporal/external validation, real banking costs and production readiness remain out of scope. Both fixed-threshold models have worse UCI holdout cost than the always-bad baseline; the report explicitly records that limitation.

## Honest resume bullet

Built a reproducible Python credit-risk model validation benchmark on UCI German Credit data, comparing Logistic Regression and Random Forest with leakage-safe preprocessing, stratified cross-validation, holdout evaluation and asymmetric error costs; added 14 passing tests and generated an auditable validation report.
