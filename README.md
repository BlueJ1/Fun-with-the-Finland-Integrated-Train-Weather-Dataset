# FI-TW delay modeling

## Current model with thirty-minute-old inputs

The rebuilt model achieves **8.977 minutes mean chronological CV RMSE** using input snapshots more than 30 minutes old at the scheduled prediction time. It uses only the original 80,916 development events; the holdout is unused for this rebuild. Fold RMSE values are 9.413, 7.445, 8.309, 10.464, and 9.252. All 49 tests pass, and saved fold models reproduce every CV prediction exactly.

The age cutoff applies to observed history, station context, pending status, timetable acceptance, and training-label availability. Nine events with too-recent timetable metadata remain in the cohort with masked predictors. The selected prediction recipe combines XGBoost with CV-tuned rules for missing reports and prolonged unresolved delays. This is a selected CV score; no new holdout score is reported.

```sh
.venv/bin/python -m fitw.stale prepare --archive data_archive
.venv/bin/python -m fitw.stale train --search-profile calibrated
.venv/bin/python -m fitw.stale verify
```

Read [the thirty-minute model report](results/stale30-development/REPORT.md) for the age audit, search scope, inference helper, limitations, and full reproduction commands. Existing experiments below remain available.

## Earlier operational model

`fitw.operational` adds a model for signed Oulu event delay using scheduled event metadata and earlier observations from the same dated train run. It keeps the original chronological 80/20 cohort membership and all 20,230 test events. A separate development search compares direct XGBoost prediction with XGBoost corrections to the last observed delay, using five expanding folds. The selected model is frozen before the separate holdout evaluation.

The new model achieves **10.473 RMSE minutes**, with MAE 3.557 and R² 0.803. The original full-weather chronological model scored 14.269 RMSE. This was the second holdout evaluation during feature development; the first operational version scored 12.285. Each version selected its configuration using development validation only. All 33 tests pass, and the saved model reload reproduces every prediction exactly.

Every history observation must have `actualTime` strictly before the Oulu event's scheduled prediction cutoff and `scheduledTime` strictly before the Oulu event's scheduled time. Training labels must also precede the earliest prediction cutoff in each validation or test block. Missing history stays in the evaluation. The model uses scheduled temporal encodings and equal regression weights, without focal actual-time encodings, target offsets, focal weather, or final cancellation flags.

This is an online operational prediction setting. Earlier completed events from the test period can supply history to later predictions. It requires a train-observation feed and assumes that observations are available at their recorded actual time. The default cutoff is the Oulu event's scheduled time; `--horizon-minutes` moves it earlier. It is not a forecast before the train begins its journey.

Some Oulu events occur early, so a scheduled-time cutoff can fall after their actual occurrence. The focal event remains excluded from history. The benchmark measures delay estimation at the scheduled cutoff and does not establish that every prediction precedes the physical Oulu event.

Run in the main environment with the same archive and prepared cohort as the study:

```sh
.venv/bin/python -m fitw.operational prepare
.venv/bin/python -m fitw.operational train
.venv/bin/python -m fitw.operational evaluate
.venv/bin/python -m fitw.operational verify
```

`prepare` also accepts `--archive` and `--cohort`. The other commands accept `--data` and `--output`; `verify` accepts `--cohort` and optional `--archive` to check all monthly source hashes. Training defaults to three concurrent candidates with two CPU threads each; `--workers 1` reduces concurrency. Use a distinct input filename and output directory for different prediction horizons. Searches checkpoint after each candidate and reject configuration mismatches. No additional dependencies are required.

Measured performance, saved model, predictions, source hashes, split membership, and independent verification are in [results/operational-chronological](results/operational-chronological). See [the operational model report](results/operational-chronological/REPORT.md) for the measured comparison and limitations.

## Three-scenario replication

This project runs the three XGBoost experiments in *Predicting Train Delays in Finland Using Machine Learning and Weather Data*, arXiv:2609.11277. The target is signed `differenceInMinutes` at Oulu station, not a final-arrival forecast before departure.

The three scenarios use operational features plus:

1. Instant weather and derived weather categories.
2. Instant weather only.
3. Derived weather categories only.

The supplied local archive is the source of observations. No synthetic data contribute to reported results. Native archive files are never edited. The reconstructed cohort has exactly 101,146 events, matching the paper's count; exact original row membership and experiment version remain unverified.

## Results

Read [results/REPORT.md](results/REPORT.md) for measured metrics, comparisons, protocol details, and limitations. [results/replication.html](results/replication.html) is a self-contained visual report with embedded figures. [results/summary.csv](results/summary.csv) contains all measured results.

Each completed experiment saves its candidate scores, exact split membership, selected model, scaler, predictions, and feature importances. [results/verification.json](results/verification.json) records independent model reload and metric checks.

## Setup

Use two isolated Python environments. The main `.venv` uses the packages frozen in `requirements-lock.txt`. The `.venv-author` environment uses the recovered author modeling package versions, with an explicit SciPy compatibility substitution. Both use Python 3.12. The environments used here have Python 3.12.14 on macOS; the author export records Python 3.12.3 on Windows, so Python patch version, operating system, and binary builds are not exact matches.

Create the main environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

Create the environment for profiles ending in `-author`:

```sh
python3.12 -m venv .venv-author
.venv-author/bin/python -m pip install -r requirements-author.txt
.venv-author/bin/python -m unittest discover -s tests -v
```

`requirements-author.txt` pins XGBoost 3.0.1, NumPy 2.0.1, pandas 2.2.3, and scikit-learn 1.6.1, matching the historical export. Joblib 1.4.2 also matches. The export records SciPy 1.15.1, but its macOS wheel fails dynamic-library validation on this macOS 27 host. The runnable author environment uses SciPy 1.17.1, whose PROPACK implementation avoids the old binary issue. This is an explicit dependency deviation; see the [SciPy issue](https://github.com/scipy/scipy/issues/25635). Parquet I/O and plotting packages are pinned for this project; their exact original versions were not recovered. The preserved [author environment export](references/author/c28b188948fe42ccc0da2c0f84a81405cc1e2a72/environment.decoded.yml) documents what was recorded, without proving which environment produced the published figure. Every run stores the actual modeling library versions in its `config.json`.

Place the FI-TW monthly Parquet files and station metadata in `data_archive/`. The required observation files are named `matched_data_flat_YYYY_MM.parquet` for all 84 months from January 2018 to December 2024. Additional 2025 files are permitted but excluded. The data are available from the [author's Kaggle dataset](https://www.kaggle.com/datasets/viniborin/finland-integrated-train-weather-dataset-fi-tw).

The corrected default study has two author-version profiles and six experiment runs, with 100 candidates and five folds per run. Three modern-environment profiles remain optional; explicitly selecting all five registered profiles gives 15 runs. It can take substantial CPU time. Searches checkpoint after each candidate and resume with the same configuration. Configuration differences require a new output directory; completed results are not silently overwritten with different parameters.

## Run the complete study

```sh
.venv/bin/python run_replication.py
```

This prepares the filtered cohort, runs three scenarios under `chronological-author` and `historical-author`, creates the reports and figures, and verifies saved model predictions. Start it with `.venv/bin/python`; the launcher automatically uses `.venv-author/bin/python` for profiles ending in `-author`. Both environments must exist for the default study: `.venv` handles preparation and reporting, and `.venv-author` fits the six default models. Profiles run in order; at most three models run concurrently, with two CPU threads each.

```sh
# Re-extract the archive and regenerate the input manifests
.venv/bin/python run_replication.py --prepare

# Run the three main-environment profiles only
.venv/bin/python run_replication.py --profiles chronological count-consistent historical

# Run the paper's chronological protocol in both environments
.venv/bin/python run_replication.py --profiles chronological chronological-author

# Run the historical-code diagnostic in both environments
.venv/bin/python run_replication.py --profiles historical historical-author

# Run only the author-version profiles, with the documented SciPy substitution
.venv/bin/python run_replication.py --profiles chronological-author historical-author

# Reduce concurrent CPU use
.venv/bin/python run_replication.py --workers 1
```

To run individual stages:

```sh
.venv/bin/python -m fitw.prepare
.venv/bin/python -m fitw.experiment --scenario full
.venv/bin/python -m fitw.experiment --scenario instant
.venv/bin/python -m fitw.experiment --scenario categories
.venv/bin/python -m fitw.report
.venv/bin/python -m fitw.verify
.venv/bin/python -m unittest discover -s tests -v
```

## Registered profiles

The default profiles are `chronological-author` and `historical-author`. The other three require explicit selection with `--profiles`.

| Profile | Environment | Feature counts, scenarios 1/2/3 | Split and CV | Search selection |
|---|---|---|---|---|
| `chronological` | `.venv` | 26 / 16 / 18 | Earliest 80% development, latest 20% test; five expanding-window folds | Mean validation RMSE |
| `count-consistent` | `.venv` | 28 / 18 / 19 | Same chronological split and folds | Mean validation RMSE |
| `historical` | `.venv` | 28 / 18 / 19 | Seeded random 80/20; shuffled five-fold CV | Mean validation MAE |
| `chronological-author` | `.venv-author` | 26 / 16 / 18 | Same split and folds as `chronological` | Mean validation RMSE |
| `historical-author` | `.venv-author` | 28 / 18 / 19 | Same split and folds as `historical` | Mean validation MAE |

The paper's explicit feature IDs give eight operational features, but its narrative and total counts imply nine with day of month. Its table includes cloud, while its text says cloud was omitted. The primary profile uses the explicit eight IDs and omits cloud. The count-consistent profile adds day of month and cloud to recover the printed total counts. The feature sets and all candidate parameters are recorded in each `config.json`.

`chronological-author` runs the primary chronological configuration with the recovered modeling package versions and the documented SciPy 1.17.1 substitution. Comparing it with an optional corrected `chronological` run would check software versions. `historical-author` follows public-source scaling: RobustScaler transforms only designated raw weather predictors, while operational and weather-category inputs remain unchanged. The optional modern-environment `historical` profile scales every predictor. Differences between these two historical profiles therefore combine software-version and scaling effects; they do not isolate either cause. There is no additional count-consistent profile in the author environment.

The historical profiles investigate the public author code. They use legacy category inequalities, per-month median imputation, a development-fitted scaler, and approximately reconstructed train-grouped row order. The launcher passes `--legacy-weather-scaling` for `historical-author`, which uses the serializable `WeatherScaler` helper. The original unsorted file enumeration and interactive feature selections were not recorded. This is a diagnostic, not a verified exact reproduction of the authors' execution. They also retain the primary matched cohort instead of allowing preprocessing to change sample membership between scenarios. They do not adopt the historical code's repeated test-RMSE model selection or its different per-row MAPE.

## Superseded diagnostics

Initial experiments accidentally omitted the sampled `scale_pos_weight` parameter from the regressor. A regression counterexample with target values equal to one showed that this changed fitted predictions. Those outputs are preserved under `results/parameter_omission_diagnostics/`, including twelve completed runs and three partial runs, but are excluded from the corrected report and summary. They do not count as replication results. Corrected models pass the complete sampled parameter set unchanged and record the policy in each `config.json`.

## Data and modeling decisions

- Select `stationShortCode == 'OL'` and `trainCategory == 'Long-distance'` from 2018–2024 departure-month files.
- Remove rows missing actual or scheduled event times, or a finite target. Keep negative delays and recorded cancellations with usable observations.
- Remove two duplicates under a retained-feature projection. They are arrival/departure pairs with identical retained values. The duplicate records and all source hashes are stored in `data/manifest.json`.
- Use `train_id = trainNumber` and the pinned historical author's actual-time UTC temporal encodings, including fractional hours. These predictors incorporate realized event time and cannot demonstrate prediction before departure.
- Compute the paper's ten mutually exclusive weather indicators before imputation. Missing threshold comparisons fail. Preserve the printed priority order and its unreachable Black Ice category. Category derivation can use gust, dew point, and precipitation amount even when those variables are excluded as raw model inputs.
- Use XGBoost squared-error regression with CPU histogram trees. Chronological preprocessing fits RobustScaler independently within every training fold; XGBoost handles remaining weather NaNs.
- Sample the same 100 candidates per scenario with seed 42. Written-protocol integer ranges include both endpoints; historical ranges follow SciPy's exclusive upper endpoints. `scale_pos_weight` is sampled and passed unchanged to XGBRegressor, as in the historical source. Despite its usual classification interpretation, it changes regression fits in these XGBoost implementations when target values equal one; omitting it changes the experiment.
- Interpret plotted iterations as randomized-search candidate budgets, as indicated by author code. The number of trees is a separately sampled parameter. Search-prefix curves are descriptive holdout evaluations after model selection is frozen; test scores never select a primary model or a search budget.
- Report WMAPE with `sum(abs(y))` as its denominator. A signed-denominator alternative is saved separately. The paper provides no equation; historical code's per-row MAPE differs.

## Evidence and verification

[references/protocol_audit.md](references/protocol_audit.md) cites the supplied PDF pages and tables. [references/source_search.md](references/source_search.md) documents verified primary sources and historical code differences. [references/data_audit.md](references/data_audit.md) documents archive schema and cohort reconstruction.

The historical source commit is a plausible implementation, not a verified exact experiment commit. Full upstream source snapshots and extracted paper text are local reference caches excluded from version control. [references/README.md](references/README.md) provides permanent source links and snapshot hashes. The checked-in evidence includes the environment export, source audits, and original cohort manifests. Superseded parameter-omission diagnostics also remain local and are excluded from the PR.

All 17 protocol tests pass in both `.venv` and `.venv-author`. They check weather priority and boundaries, signed-delay metrics, common cohorts, chronological boundaries, deterministic search prefixes, fold-local scaling, independence of model selection from test labels, source-consistent scaling of only raw weather predictors, saved-scaler round-trip transforms, and literal scale_pos_weight preservation with a regression counterexample. The artifact verifier separately reloads saved models, reproduces every test prediction, recomputes metrics, checks input hashes, and verifies candidate/fold completeness.

Daily paired bootstrap intervals in the report describe the observed holdout differences. They do not include model-retraining uncertainty or all temporal dependence. The paper contains no railway-only baseline; reproducing its three scenarios cannot establish the incremental value of weather.
