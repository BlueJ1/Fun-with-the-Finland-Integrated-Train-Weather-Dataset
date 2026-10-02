# Author source and dataset provenance

Verified on 2026-10-02. These are read-only source findings. No author was contacted, no credentials were accessed, and no dataset was downloaded for this source search.

## Primary sources

- [Paper](https://arxiv.org/abs/2609.11277): arXiv 2609.11277, submitted 2026-09-10, by Vinicius Pozzobon Borin, Jean Michel de Souza Sant'Ana, and Nurul Huda Mahmood. The extracted PDF text is kept in a local cache excluded from version control.
- [Dataset paper](https://arxiv.org/abs/2601.16592): earlier FI-TW description, 2018–2024, approximately 38.5 million observations, 28 engineered features. The abstract was verified through web search; direct arXiv fetching was unavailable.
- [FI-TW dataset](https://www.kaggle.com/datasets/viniborin/finland-integrated-train-weather-dataset-fi-tw) and [public metadata API](https://www.kaggle.com/api/v1/datasets/view/viniborin/finland-integrated-train-weather-dataset-fi-tw). API response preserved in `references/author/kaggle_dataset_metadata.json`.
- [Author training repository](https://github.com/borinvini/Railway-FMI-Data_training), explicitly linked by Kaggle as its preprocessing pipeline.
- [Author fetcher](https://github.com/borinvini/Railway-FMI-Data_fetcher) and [viewer](https://github.com/borinvini/Railway-FMI-Data_viewer), also linked by Kaggle.

The author uses **viniborin on Kaggle and borinvini on GitHub**. Guessing GitHub usernames viniborin or borindev failed; the verified repository links come from author-owned Kaggle metadata.

## Dataset version matters

Kaggle current version 3 was created 2026-07-21. It describes 2018–2025, approximately 48 million arrival/departure timetable events, up to 125 raw columns, 96 monthly Parquet files, and GPL 3 licensing. Version 2 introduced 78 rolling weather columns: 12h, 24h, and 72h aggregates. Version 3 added station metadata CSVs. Version 1 was released 2025-12-12.

The dataset description identifies Oulu as `stationShortCode=OL`, `stationUICCode=370`, name Oulu asema. `trainNumber` is the train identifier for the departure date; `departureDate` is its first-departure date. Each row is one station ARRIVAL or DEPARTURE event. `scheduledTime` and `actualTime` are UTC. Raw weather names include `Pressure (msl)` rather than the paper's descriptive name.

Use only 2018–2024 for the requested paper replication. Version 3's new rolling weather columns are not the paper's engineered 1h-window columns. Do not substitute its 12h/24h/72h aggregates silently.

## Preserved author code

Historical snapshot: [commit c28b188948fe42ccc0da2c0f84a81405cc1e2a72](https://github.com/borinvini/Railway-FMI-Data_training/tree/c28b188948fe42ccc0da2c0f84a81405cc1e2a72), 2026-01-28. Four relevant source files were inspected as unmodified local snapshots. Full upstream code is excluded from version control; permanent links and inspected-content hashes are in [README.md](README.md). The environment export is retained. This is a plausible pre-conference implementation, **not a verified exact experiment commit**.

Current HEAD at time checked: [714416d1160ae99a2b4b922e735cec62e14d2f5d](https://github.com/borinvini/Railway-FMI-Data_training/tree/714416d1160ae99a2b4b922e735cec62e14d2f5d), 2026-07-20. Commit inventories are preserved in `references/author/training_commits_page1.json` and page2. The history reveals substantial preprocessing and evaluation changes between these snapshots.

## Historical preprocessing conventions

References below use line numbers in the pinned historical files linked from [README.md](README.md).

- `preprocessing_pipeline.py:998–1020`: `train_id` is copied directly from `trainNumber`. It is not a newly factorized train/date identifier.
- `preprocessing_pipeline.py:2515–2537`: drop rows null in either `scheduledTime` or `actualTime`.
- `preprocessing_pipeline.py:2573–2600`: parse **actualTime**, without timezone conversion; extract its UTC month/day, and its hour plus minute as `HH:MM`. Weekday numbering is **Sunday=1, Monday=2, ..., Saturday=7**, using `np.where(pandas_dayofweek == 6, 1, pandas_dayofweek + 2)`.
- `preprocessing_pipeline.py:3418–3420`: hour cycle uses decimal hour `(hour + minute/60)/24`.
- `preprocessing_pipeline.py:3580–3581`: month sine/cosine uses `month/12`, without subtracting 1.
- `preprocessing_pipeline.py:3763–3764`: weekday sine/cosine uses `day_of_week/7`, without subtracting 1.
- `preprocessing_pipeline.py:3054`: drops missing required target columns among the configured five targets. Current HEAD instead requires only the three numeric delay targets.
- `const_preprocessing.py:54`: weather missing-column threshold **30%**. Historical `handle_missing_values` drops weather columns above this threshold within the processed monthly file, then drops rows where **all** remaining important weather columns are missing.
- `preprocessing_pipeline.py:3180–3183`: despite headings saying month-specific median, the implementation fills weather NaNs with **the single file median**, after station/row filtering. Its files are monthly; it is not a train-fold-fitted imputer.
- Weather categories are assigned earlier in the pipeline, before weather imputation. Unmet/missing comparisons leave a row Normal/Clear.

June 4 commits changed month/day cycle offsets to zero-based, numeric-target handling, and Black Ice category reachability. These later conventions must not be attributed to the historical snapshot.

## Weather category rules and ambiguity

Historical source follows this precedence, with later rules applying only to remaining Normal/Clear rows:

| Category | Historical condition |
|---|---|
| Blizzard | Temp < 0; precipitation intensity > 1 OR amount > 3; wind > 10 OR gust > 15; visibility < 1000 |
| Heavy Snow | Temp < 0; intensity > 2 OR amount > 5; snow depth > 0 |
| Extreme Cold | Temp < -20 |
| Heavy Rain | Temp > 2; intensity > 4 OR amount > 10 |
| Freezing Rain | -2 <= temp <= 2; amount > 0 OR intensity > 0 |
| Black Ice | -2 <= temp <= 2; humidity > 80; abs(dew point - temp) < 2; amount > 0 |
| Dense Fog | visibility < 1000; amount <= 0.1; humidity > 95 |
| High Winds | wind > 15 OR gust > 20 |
| Extreme Heat | temp > 30 |
| Normal/Clear | default |

**Black Ice is unreachable in this historical implementation:** Freezing Rain matches every Black Ice row first. The paper Table II puts Freezing Rain before Black Ice and gives Black Ice precipitation >0, consistent with that problem, though its displayed inequalities differ slightly from code. The 2026-06-04 commit e8296fa20322939bb72f3f8ff7bca5a049cfeb80 explicitly fixes Black Ice detectability: Black Ice moves before Freezing Rain and requires amount <=0.5. Current rules therefore differ materially from the historical implementation and the printed paper table.

One-hot categories use fixed order: Normal/Clear, Blizzard, Heavy Snow, Extreme Cold, Heavy Rain, Freezing Rain, Black Ice, Dense Fog, High Winds, Extreme Heat. All 10 columns are retained, including an all-zero Black Ice column historically.

## Evaluation code conflicts with paper protocol

The supplied paper states chronological 80/20 splitting by event timestamp, expanding-window time-series CV, fold-fitted preprocessing, 100 candidate configurations, and mean validation RMSE selection. The inspected code does **not** implement that protocol.

Historical author source:

- `const_training.py:71`: `TEST_SIZE = 0.2`.
- `training_pipeline.py:1303` and `1418–1424`: `train_test_split(df, test_size=test_size, random_state=42, stratify=None)` for regression; no timestamp sorting or chronological partition.
- `training_pipeline.py:1991–1997`: `KFold(n_splits=5, shuffle=True, random_state=42)`, `XGBRegressor(random_state=42, n_jobs=1, eval_metric='mae')`, `scoring_metric='neg_mean_absolute_error'`.
- `training_pipeline.py:1759–1775`: `RobustScaler` fits once on full training weather values, then transforms train and test. It is not fitted separately inside each CV fold. Category-only files are copied without scaling.
- `const_training.py:101–108`: `n_estimators=randint(100,400)` and `max_depth=randint(4,8)` (SciPy upper bounds exclusive), learning_rate .01/.05/.1, subsample .7/.8/.9, colsample_bytree .7/.8/1, scale_pos_weight 3.9/4.9/5.9.
- `const_training.py:132–133`: default search maximum 30 and CV folds 5. This config need not be the exact run used for the paper, which states 100.
- `training_pipeline.py:1999–2024`: `iteration_values = range(10, RANDOM_SEARCH_ITERATIONS+1, 10)`. Each trajectory point constructs a new `RandomizedSearchCV(n_iter=n_iter, random_state=42)` over the same candidate distribution. **Iterations here mean numbers of hyperparameter candidates, not boosting trees.** `n_estimators` is independently sampled as a hyperparameter.
- `training_pipeline.py:2061–2081`: historical outer search-budget winner is chosen using **test RMSE**. Thus this implementation uses the test set for repeated selection. Commit 224d40f1ef12bb1aff90a0b113343326d19344d0 on 2026-06-23 explicitly changes selection to CV score and introduces a target log transform.
- `training_pipeline.py:2133–2167`: plotted trajectory combines test scores and CV best scores, titled `XGBoost Performance vs RandomizedSearch Iterations`. This strongly supports interpreting a similar figure as search-budget performance; it does not establish that the paper's exact figure came from this commit.

Current HEAD adds an optional latest-calendar-year holdout, but still uses random `train_test_split` for the remaining development data and shuffled KFold. This is also different from the paper's stated 80/20 chronological protocol.

The primary replication should follow the supplied paper's explicit protocol. If historical-code evaluation is run to investigate the reported metrics, label it separately and retain these differences; matching its scores does not validate the paper's claimed chronological evaluation.

## Additional target, ordering, and test-distribution checks

The active historical XGBoost path reads raw `y_train = train_df[target_feature]` and raw `y_test`, with no positivity filter, target log transform, or target scaling. `TRAIN_DELAY_MINUTES=5` creates `trainDelayed` via a strict >5 comparison; it does not restrict the regression sample. The disabled target-analysis routines do contain >=5 subsets, but those do not feed the enabled training stage. Historical training state has no outlier filtering stage; that stage was introduced on June 23. `WEIGHT_DELAY_COLUMN='NONE'` disables sample weighting. The weighted objective is defined but is not wired into this active randomized-search method.

Historical config defaults to **differenceInMinutes_eachStation_offset**, not the paper's differenceInMinutes. Because selection was interactive, these checked-in constants cannot be treated as a saved exact paper run. No results JSON, models, split membership files, notebooks, or experiment assets appear in the historical Git tree.

Row order is not fully reproducible from the repository:

- `preprocessing_pipeline.py:1017–1041` groups all stops by trainNumber in a Python dictionary and emits the groups in first-encounter insertion order. Repeated train numbers from different dates are grouped together within each monthly input.
- `training_pipeline.py:861–882` uses unsorted `glob.glob` to find monthly CSVs, then concatenates them in that returned order. Filenames are sorted later only to name the output range. No record of the filesystem enumeration or first-encounter train order is saved.
- Interactive selected column numbers are sorted before selection (`training_pipeline.py:1200`), so original preprocessed column order is retained. Restricting to the paper's chosen feature IDs gives: trainStopping, month_sin, month_cos, hour_sin, hour_cos, day_week_sin, day_week_cos, train_id; then Air temperature, Wind speed, Wind direction, Relative humidity, Precipitation intensity, Snow depth, Pressure (msl), Horizontal visibility, Cloud amount; then the fixed ten category indicators. Any missing-threshold exclusions remove their columns without changing the others' order. This matches the replication's current feature order.
- Historical and current `filter_columns` remove actualTime and scheduledTime before `remove_duplicates`. The replication's projection retains actualTime, but a read-only check replacing it with source-derived month/day/weekday/hour-minute also removes exactly two rows, yielding the same 101,146-row cohort. The exact timestamp is therefore unnecessary to reproduce that count on this archive.

Read-only target-distribution checks using the reconstructed 101,146-row cohort and random 80/20 splitting with seed42 are saved in `references/author_order_audit.json`:

| Input order before random splitting | Test population std | Test mean | Test max |
|---|---:|---:|---:|
| Raw monthly archive order | 19.0079 | 5.2597 | 598 |
| Monthly trainNumber first-encounter grouping | 18.8325 | 5.1204 | 735 |
| Chronological order, then random split | 17.3943 | 5.1249 | 548 |
| Chronological 80/20 holdout | 23.6114 | 8.4253 | 734 |

The monthly grouped audit approximates historical order using first encounter among Oulu rows; the original upstream file included all stations, so its exact train first-encounter order can differ. Exact historical random membership also depends on unsaved monthly glob order. Reported R²=.78 and RMSE=8.5 imply a test population standard deviation about 18.12. These checks are consistent with the reported scores arising from a random test sample, but they do not prove a particular split or reproduce the reported model.

### Independently derived full-network train grouping

A subsequent source-faithful ordering check reads **only trainNumber** from each full-network monthly Parquet, computes `pd.unique(trainNumber)` in native row order, and saves the complete 84-month mapping in `references/global_train_order.json`. This mapping was derived from author source semantics, without selecting an order based on test performance. Monthly files are still concatenated chronologically as an explicit reproducibility assumption; historical glob order was not recorded.

Applying that full-network first-appearance ordering within each month to the same 101,146-row Oulu cohort, preserving the raw order of each trainNumber's events, and using seed42 random 80/20 gives:

- Test population standard deviation **18.101352**, mean 5.102472, mean absolute target **7.286752**, and maximum 739 minutes.
- At R²=.78, the implied RMSE is **8.490287**, which rounds to the paper's **8.5**.
- MAE=3.7 gives WMAPE=50.777%; because 3.7 is rounded to one decimal, the displayed paper WMAPE=50.5% is compatible with a true MAE about 3.680.

The audit is preserved in `references/global_train_order_audit.json`. Agreement in implied target variance strengthens the evidence that the reported results used the historical randomized partition rather than the paper's stated chronological holdout, whose target standard deviation is 23.611450. This is an inference from source ordering and metric arithmetic, not confirmation of an exact original run or model score.

## Historical software versions

The [historical environment.yml](https://github.com/borinvini/Railway-FMI-Data_training/blob/c28b188948fe42ccc0da2c0f84a81405cc1e2a72/environment.yml) pins **xgboost=3.0.1**, **libxgboost=3.0.1**, and **py-xgboost=3.0.1**. It also pins Python 3.12.3, NumPy 2.0.1, pandas 2.2.3, scikit-learn 1.6.1, and SciPy 1.15.1. The original Windows conda export is UTF-16; both its unmodified bytes and a readable `environment.decoded.yml` are saved with the historical source snapshot.

The active historical XGBRegressor creation sets seed42, n_jobs=1, and eval_metric=mae. It does not set `tree_method`, and the search space does not tune it. [Official XGBoost 3.0 parameter documentation](https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html) states that the default `tree_method=auto` uses the `hist` method. Thus explicitly using `hist` in the replication matches this historical default. The replication's XGBoost 3.4.1 differs from the recorded 3.0.1; this remains a version deviation, even though tree-method selection agrees. The repository environment export is evidence of the authors' recorded software environment, not proof that the exact reported runs used it.

## scale_pos_weight has an effect in this regression implementation

Although paper prose explains `scale_pos_weight` as a classification imbalance parameter, the authors' source passes the whole search distribution directly to `RandomizedSearchCV` without removing that parameter in its regression branch (`training_pipeline.py:1991–1997`, `2021–2028`). Its sampled values are 3.9, 4.9, and 5.9. A literal replication must retain these sampled values.

[XGBoost v3.0.1 primary source](https://github.com/dmlc/xgboost/blob/v3.0.1/src/objective/regression_obj.cu) shows why the parameter matters:

- Lines 130–131 load `param_.scale_pos_weight` into the additional gradient input.
- Lines 163–167 multiply the sample weight by this parameter **whenever the numeric label equals exactly 1.0**, and use the resulting weight in both gradient and Hessian. The operation is in the shared `RegLossObj<Loss>` template; it is not gated by a classification task check.
- Lines 221–223 register squared-error regression using that same `RegLossObj<LinearSquareLoss>` template.
- Lines 190–197 also change automatic intercept initialization to the Newton method when `scale_pos_weight` differs from 1.

Therefore this implementation weights delay targets of **exactly one minute**, rather than all positive delays, when using squared-error regression. The reconstructed cohort contains **26,908 one-minute targets**, 26.603% of its 101,146 rows, so the effect is substantial enough to preserve. The source file is saved unmodified in `references/author/xgboost-v3.0.1/regression_obj.cu`. Dropping `scale_pos_weight` on the assumption that it has no regression effect would change the trained models and cease to match the authors' literal implementation.
