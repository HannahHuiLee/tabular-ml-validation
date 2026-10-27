# Dataset Provenance

- Dataset: **Statlog (German Credit Data)**, UCI dataset ID **144**.
- Official page: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- Official download: https://archive.ics.uci.edu/static/public/144/statlog%2Bgerman%2Bcredit%2Bdata.zip
- Citation: Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.
- UCI lists **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/ . Retain attribution, link the license, and indicate changes when sharing adaptations. This project renames columns and recodes labels in memory; the cached source files are unmodified.
- Original mixed-format `german.data`: 1,000 rows, 20 features (7 numerical, 13 categorical), plus a target column. We do not use the alternative expanded `german.data-numeric` file.
- Original labels: 1=good, 2=bad. Project labels: 0=good, 1=bad. Counts verified by execution: 700 good, 300 bad.
- Retrieved from the official ZIP; cached bytes are included for offline reproducibility. The loader verifies SHA-256 on every run. If the cache is absent, it downloads this exact official ZIP. Network errors and checksum mismatches stop execution; there is no alternate or synthetic fallback.

SHA-256:

```text
b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871  german.data
69b92cb551111bc37098698288849c96f0fd87ce81a2e54451ff39aa1e53fe49  german.doc
```

The accompanying [raw/german.doc](raw/german.doc) is the original feature dictionary and cost specification.
With actual rows and predicted columns ordered [good, bad], the cost matrix is `[[0, 1], [5, 0]]`.
This means FP + 5×FN under our positive-class convention. It is an educational benchmark's relative cost, not an estimate of actual banking losses.

## Historical limitations

UCI records donation on November 16, 1994; this is not a verified collection date. Legacy German credit categories, including Deutsche Mark bands, may not describe contemporary lending populations. The small dataset does not provide a modern representative sample or sufficient temporal/institutional metadata for out-of-time validation. Personal status/sex, age and foreign-worker attributes require careful fairness and proxy analysis before any decision use. No such deployment claim is made here.
