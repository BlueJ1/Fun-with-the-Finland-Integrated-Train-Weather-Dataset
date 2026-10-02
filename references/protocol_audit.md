# Independent protocol audit

This audit uses the supplied six-page paper, its locally extracted text, and visual inspection of PDF pages 3, 4, and 6. It also independently inspects the author repository snapshot at commit `c28b188948fe42ccc0da2c0f84a81405cc1e2a72`. Full paper text and upstream code snapshots are local caches excluded from version control; [README.md](README.md) lists their permanent source links and hashes. Page numbers below refer to PDF pages. Reported scores are the authors' claims, not replication outputs. Implementation recommendations resolve missing details; they are not statements that the authors used those choices. The recovered historical code and the written paper use different evaluation protocols.

## Study population and target

Page 3, section III.B, restricts the experiment to Oulu asema and all long-distance train events arriving or departing during 2018 through 2024. The stated sample size is 101,146 observations. Both arrival and departure observations belong to the population; there is no instruction to select only commercial stops or delayed trains.

The regression target is `differenceInMinutes`, feature 1 in Table I on page 2. The operational API's supplied delay is the target. Pages 2 and 3 explicitly permit negative, zero, and positive delays. Neither journey-offset variant, binary `trainDelayed`, nor `cancelled` is the target. Do not clip negative values or calculate a replacement target from timestamps without labeling that as a different experiment.

The paper does not specify treatment of cancelled trains, cancelled stops, missing targets, duplicate rows, extreme outliers, or train types within the long-distance category. A defensible implementation filters station and `trainCategory == 'Long-distance'`, retains all finite target values, and records missing-target exclusions. Do not exclude target outliers or select delayed trains to improve agreement with the reported scores.

In a read-only check of the provided `matched_data_flat_2024_12.parquet`, exact station name `Oulu asema` selects 1,380 rows, all with station code `OL` and train category `Long-distance`. Sample target values include -1, 0, and 1 minutes. This is an archive observation, not evidence of the paper's exact dataset version or filtering.

## Temporal protocol and preprocessing

Page 3 specifies the following evaluation procedure:

1. Sort by event timestamp.
2. Use the earliest 80% of observations as development data and the most recent 20% as a final test set.
3. Select hyperparameters using five-fold expanding-window time-series cross-validation inside the development set.
4. Fit robust scaling and all learned transformations only on each training fold.
5. Draw 100 candidate hyperparameter configurations and minimize mean validation RMSE.
6. Retrain the selected configuration on the complete development set and evaluate once on the held-out test set.

With precisely 101,146 eligible observations, an integer floor at 80% yields 80,916 development rows and 20,230 test rows. The paper does not specify rounding or boundary ties. Freeze and export membership before training. If identical timestamps cross a boundary, disclose that or keep each timestamp group together and disclose the resulting counts.

The timestamp field is unspecified in the paper. The native archive has `scheduledTime`, `actualTime`, and sometimes estimate timestamps. The recovered author code supplies direct evidence for `actualTime` temporal predictors, so use it for fidelity in the main replication and record this choice. A scheduled-time sensitivity is scientifically useful. Using `actualTime` to derive time features embeds the realized delay in the predictors, because actual time differs from scheduled time by the target. This is a scientific limitation even when the split is chronological.

The paper does not specify timezone or whether month and weekday start at zero or one. Equation 1 on page 3 gives sin(2*pi*t/P) and cos(2*pi*t/P), with P = 24, 12, and 7 for hour, month, and weekday. The historical author preprocessing parses the UTC actual-time strings directly. It uses month values 1 through 12 divided by 12, Sunday = 1 and Monday = 2 through Saturday = 7 divided by 7, and fractional hour = hour + minute/60 divided by 24. Seconds are excluded. Preserve these recovered conventions for fidelity. A later code version can differ, so the pinned historical snapshot is necessary.

Missing-predictor handling is not specified. Page 3 says XGBoost handles missing values. Preserve missing numeric values for XGBoost and use RobustScaler inside each fold. Do not fit imputers, category mappings, or scaling on the full dataset. A stable fixed mapping for service train numbers is permissible without learning from test targets.

## The three scenarios and contradictory feature counts

Table I on page 2 defines these explicit operational feature IDs, repeated in section III.B and Table IV on page 4:

| ID | Predictor |
| --- | --- |
| 6 | `trainStopping` |
| 9 | `month_sin` |
| 10 | `month_cos` |
| 11 | `hour_sin` |
| 12 | `hour_cos` |
| 15 | `day_week_sin` |
| 16 | `day_week_cos` |
| 18 | `train_id` |

That list contains eight predictors. Page 3 says eight base operational features. Page 4 instead says nine operational predictors and explicitly mentions day of month, ID 17. Table IV totals of 28, 18, and 19 require nine operational predictors. Its feature IDs omit ID 17 in all three scenarios. Neither interpretation satisfies every statement.

Table IV instantaneous weather IDs are 19, 20, 22, 23, and 25 through 29:

| ID | Predictor |
| --- | --- |
| 19 | Air temperature |
| 20 | Wind speed |
| 22 | Wind direction |
| 23 | Relative humidity |
| 25 | Precipitation intensity |
| 26 | Snow depth |
| 27 | Pressure at mean sea level, native `Pressure (msl)` |
| 28 | Horizontal visibility |
| 29 | Cloud amount |

These are nine predictors. Gust speed and dew-point temperature are omitted from the model features according to page 4, section III.D. They can still be source measurements for weather category construction. Page 4, section III.C, separately says cloud amount was dropped because unavailable at Oulu; the wording occurs in the correlation-analysis description. Figure 2 omits cloud. Table IV nevertheless includes cloud, so cloud omission from correlation analysis cannot conclusively establish omission from the trained models.

The provided December 2024 Oulu sample contains observed cloud values including 7 and 8. Thus a blanket claim that the supplied archive has no Oulu cloud measurements is incorrect. Missingness over the complete study still needs measurement.

The intended scenario comparison is unambiguous:

| Scenario | Composition | Explicit-ID interpretation | Count-matching interpretation |
| --- | --- | ---: | ---: |
| 1, full weather | Operational + instant + ten categories | 27 | 28 |
| 2, instant only | Operational + instant | 17 | 18 |
| 3, categories only | Operational + ten categories | 18 | 19 |

The count-matching interpretation adds `day_of_month` to all scenarios. Dropping cloud subtracts one from scenarios 1 and 2 under either interpretation.

Recommended reporting is an explicit primary specification plus a small sensitivity experiment. The planned written-paper track uses the explicit eight operational predictors and omits cloud according to the availability statement, giving 26, 16, and 18 predictors. The planned historical-code track adds day of month, giving nine operational predictors. Cloud retention and explicit Table IV totals remain sensitivity choices. These configurations are declared interpretations; the historical selection function requests feature columns interactively and does not preserve the selections in the inspected snapshot. Record the chosen names and count in each result row.

The native monthly archive contains neither `train_id` nor the paper's engineered time/category columns. It contains numeric `trainNumber` and `departureDate`. The paper does not state the mapping. The recovered historical preprocessing explicitly assigns `train_id` from `trainNumber`, so numeric train number is author-supported rather than merely a proxy. Do not replace it with a unique dated journey ID or a sequential label-encoded ID.

## Weather category construction

Page 3, Table II, gives the following order. Use first-match precedence to obtain ten mutually exclusive one-hot categories. Comparisons are strict except the stated fog precipitation upper bound.

| Priority | Category | Predicate |
| ---: | --- | --- |
| 1 | Blizzard | T < 0 C; precipitation intensity > 1 mm/h or amount > 3 mm; wind > 10 m/s or gust > 15 m/s; visibility < 1,000 m |
| 2 | Heavy Snow | T < 0 C; intensity > 2 mm/h or amount > 5 mm; snow depth > 0 cm |
| 3 | Extreme Cold | T < -20 C |
| 4 | Heavy Rain | T > 2 C; intensity > 4 mm/h or amount > 10 mm |
| 5 | Freezing Rain | -2 C < T < 2 C; intensity > 0 mm/h or amount > 0 mm |
| 6 | Black Ice | -2 C < T < 2 C; humidity > 80%; dew-point minus T < 2 C; precipitation > 0 |
| 7 | Dense Fog | precipitation <= 0.1 mm; visibility < 1,000 m; humidity > 95% |
| 8 | High Winds | wind > 15 m/s or gust > 20 m/s |
| 9 | Extreme Heat | T > 30 C |
| 10 | Normal/Clear | No earlier predicate met |

Conjoin clauses separated by semicolons; use OR within each precipitation or wind/gust clause. This resolves the grammar of Table II consistently with compound weather phenomena.

There are important unresolved definitions:

- The amount-based precipitation alternatives have no stated accumulation duration. The native archive has `Precipitation amount` as well as 12h, 24h, and 72h cumulative columns. Use the native amount as a declared primary interpretation rather than silently substituting a cumulative window.
- Black Ice's precipitation predicate does not state intensity versus amount. Its dew-point predicate is printed as dew-point minus temperature, not absolute difference or temperature minus dew-point. Preserve the literal sign if reproducing the table; document its physical weakness.
- Black Ice becomes unreachable when its precipitation > 0 implies the preceding Freezing Rain predicate. This is a consequence of the literal severity order and thresholds. Do not reverse categories to obtain nonzero counts while calling it the original method.
- Missing values make comparisons false unless a policy says otherwise. Defaulting an observation with missing inputs to Normal/Clear confuses unobserved conditions with safe weather. If using literal false-comparison behavior for fidelity, record how many Normal/Clear assignments lack sufficient measurements. The paper does not specify a missing-category policy.
- Categories use source gust and dew-point even though the model inputs omit them. Removing these source measurements before category generation changes the method.

Calculate and export category counts and overlaps before first-match assignment. Verify exactly one active flag per modeled row under the mutually exclusive interpretation. Empty Black Ice flags are expected under the literal logic; they are not an implementation failure.

## Model selection

Page 3, Table III, specifies:

| Hyperparameter | Distribution |
| --- | --- |
| `n_estimators` | Uniform integer 100 through 400 |
| `max_depth` | Uniform integer 4 through 8 |
| `learning_rate` | 0.01, 0.05, 0.1 |
| `subsample` | 0.7, 0.8, 0.9 |
| `colsample_bytree` | 0.7, 0.8, 1.0 |
| `scale_pos_weight` | 3.9, 4.9, 5.9 |

Inclusive upper bounds require care because SciPy randint excludes its upper bound. If using that distribution, use 401 and 9 as the exclusive upper bounds. The paper gives no random seed, objective, XGBoost version, tree method, other regularization values, final best hyperparameters, early-stopping policy, or number of CPU threads.

The target and reported metrics are regression. The accompanying `scale_pos_weight` explanation refers to binary class imbalance, so the paper's interpretation is inconsistent with its target. Nevertheless, the historical code passes this parameter to XGBRegressor, and experiments with the supported XGBoost versions demonstrate an effect on continuous regression when some target values equal one. The literal parameter must therefore remain in the estimator; treating it as irrelevant and dropping it changes the experiment. Preserve the sampled value without describing the result as binary class balancing, and do not convert the target to classification.

Each scenario should use identical population, split, folds, scoring definition, and search budget. Fixed candidate draws and seed improve comparison reproducibility. Hyperparameter selection must not use the final test results. Retain best parameters, all candidate CV scores, and a test-prediction file.

## Iteration curve and reported results

Page 5 discusses performance at 10 through 100 iterations, and page 6 Figure 3 labels this axis only as Iterations. It does not specify whether this means trees, optimization epochs, random-search trials, or repeated runs. Figure 3's range conflicts with the 100 to 400 estimator search if iterations mean final tree counts. There are four broad plateaus, at 10/20, 30/40, 50/60/70, and 80/90/100, in every metric. That is what the plotted figure displays; it does not establish the actual training procedure.

The recovered historical author code resolves the axis. It loops over `n_iter` in RandomizedSearchCV in increments of ten, with a fixed seed of 42. These are hyperparameter candidate counts, not boosting trees. The same candidate prefixes can retain the same best model, which explains the long exact plateaus. Reproduce the figure by fitting the common 100-candidate pool once, choosing the best cross-validation candidate within each prefix of 10, 20, ..., 100, and refitting only when the winning candidate changes. Do not interpret the figure as a 10 to 100 tree learning curve. The written-paper track still uses its full 100-candidate development selection and reports its final held-out score separately.

Reported approximate convergence scores, from page 5 and the visually inspected page 6 figure:

| Scenario | R2 | RMSE, minutes | MAE, minutes | WMAPE |
| --- | ---: | ---: | ---: | ---: |
| 1, full weather | About 0.70 | About 9.9 | About 4.1 | About 56.5% |
| 2, instant only | About 0.70 | About 9.9 | About 4.1 | About 56.5%, slightly lower on the plotted line |
| 3, categories only | About 0.78 | About 8.5 | About 3.7 | About 50.5% |

Figure 3 approximate Scenario 3 plateaus are R2 0.70, 0.73, 0.76, 0.78; RMSE 9.9, 9.4, 8.9, 8.5; MAE 4.27, 4.02, 3.76, 3.69; WMAPE 58.5%, 55.1%, 51.7%, 50.6%. These are visual readings, not machine-readable author outputs. They should not be used as exact ground truth to tune models.

R2, RMSE, and MAE are standard regression quantities. Page 5 describes WMAPE as total absolute error divided by total actual values but gives no equation or signed-delay handling. Standard signed-target-safe WMAPE uses 100 * sum(abs(y - prediction)) / sum(abs(y)). A literal signed-sum denominator, 100 * sum(abs(y - prediction)) / sum(y), differs when early arrivals exist. Report the standard definition, optionally also the signed-sum variant with an explicit label. State undefined behavior if the denominator is zero.

Page 1 says 11% R2 improvement and 10% error reduction. R2 from 0.70 to 0.78 is about 11.4% relative improvement, or 0.08 absolute improvement. Page 5's conclusion says approximately 14% RMSE improvement and 10% MAE improvement. The RMSE values 9.9 to 8.5 imply about 14.1%; MAE 4.1 to 3.7 implies about 9.8%. The abstract's generic 10% error reduction is therefore not a precise summary of RMSE.

## Recovered historical author implementation

The pinned source is [author repository commit c28b188](https://github.com/borinvini/Railway-FMI-Data_training/tree/c28b188948fe42ccc0da2c0f84a81405cc1e2a72). The local snapshot is a primary implementation source. It is evidence of what the public code does, not proof that it produced the paper's results.

| Detail | Historical source evidence | Relationship to written paper |
| --- | --- | --- |
| Service identifier | `preprocessing_pipeline.py`, lines 999 to 1020, assigns train number to train ID | Resolves an omitted definition |
| Time extraction | Same file, lines 2515 to 2600, drops missing scheduled/actual times and parses actualTime | Paper does not name field |
| Cycles | Same file, hour conversion around 3380, month line 3580, weekday line 3763 | Matches equation with recovered input conventions |
| Weather thresholds | Same file, lines 2190 to 2276 | Uses native amount; includes -2 and +2 endpoints for freezing rain/black ice; uses absolute dew-point gap |
| Missing weather | Same file, lines 3070 to 3193 | Drops columns above 30% missingness and all-weather-missing rows; fills by file median before splitting |
| Predictor selection | `training_pipeline.py`, lines 1170 to 1220 | Interactive columns, no fixed scenario selection saved |
| Holdout | Same file, lines 1418 to 1423 | Random train_test_split, 20% test, seed 42; contradicts chronological split |
| Scaling | Same file, lines 1671 to 1774 | Scales weather on whole training subset before CV, rather than fold-local fitting |
| CV and score | Same file, lines 1991 to 1997 | Shuffled KFold, five folds, seed 42, negative MAE; contradicts expanding CV and RMSE selection |
| Curve axis | Same file, lines 2000 to 2030 | n_iter is RandomizedSearchCV budget, resolves unspecified Iterations |
| Choice of final prefix | Same file, lines 2062 to 2080 | Chooses the best prefix using test RMSE, so test is used for selection |
| Percentage metric | Same file, lines 2067 to 2069 and 2127 to 2130 | Per-row epsilon-clipped MAPE, not paper WMAPE |
| Search boundaries | `const_training.py`, lines 101 to 108 | randint upper endpoints exclude 400 and 8, unlike inclusive printed bounds |

The historical constant `RANDOM_SEARCH_ITERATIONS` is 30 at line 132 of `const_training.py`; the method's comment describes 10 to 100, and the paper figure reaches 100. Reaching the paper range therefore requires an explicit budget override. The constant target default is an offset target, but the target-selection preprocessing stage is disabled and training selects columns interactively. That constant alone does not establish the experiment's target. Preserve `differenceInMinutes` as requested by the paper.

The historical weather function supports literal amount-column interpretation and confirms first-match priority. Its inclusive freezing-band endpoints and absolute dew-point difference differ from printed Table II. Black Ice remains unreachable because earlier Freezing Rain still supersets its positive-amount condition. Report whether the historical-code thresholds or printed thresholds are implemented.

Two evaluation tracks are justified. The written-paper track implements chronological holdout, expanding CV, fold-local preprocessing, RMSE selection, and 100 candidates. The historical-code track implements the public random holdout, shuffled five-fold CV, MAE selection, recovered temporal conventions, and search-budget curves. The random track can approximate the displayed results without validating the paper's claim of chronological evaluation. Its pre-split weather imputation and repeated test selection also limit predictive interpretation.

## Recovered dependency versions and additional profiles

The pinned commit's `environment.yml` is a Windows conda export that records Python 3.12.3, XGBoost 3.0.1, NumPy 2.0.1, pandas 2.2.3, scikit-learn 1.6.1, SciPy 1.15.1, and joblib 1.4.2. Its original UTF-16 bytes and a readable decoded version are preserved under `references/author/c28b188948fe42ccc0da2c0f84a81405cc1e2a72/`. This primary source resolves the earlier paper-only absence of library versions, but it does not prove that this environment generated the published metrics.

The original replication environment uses XGBoost 3.4.1 and newer core dependencies. The project therefore adds `.venv-author` and `requirements-author.txt` to test the recovered modeling dependency versions. XGBoost, NumPy, pandas, scikit-learn, and joblib match the historical export. SciPy 1.15.1's macOS wheel fails dynamic-library validation on this macOS 27.0.1 host because of its PROPACK binary. The runnable environment explicitly substitutes SciPy 1.17.1, whose C PROPACK implementation avoids that failure. This substitution is a dependency deviation, documented by the [official SciPy issue](https://github.com/scipy/scipy/issues/25635). The preserved original environment export remains unchanged. `chronological-author` repeats the primary chronological configuration. `historical-author` repeats the historical random-split diagnostic and additionally enables `--legacy-weather-scaling`, which fits RobustScaler on raw weather predictors only and leaves operational and weather-category values unchanged, matching the historical source. The earlier `historical` profile scaled all predictors. These two author-version profiles form the corrected default study of six scenario runs, each with 100 candidates and five folds. `chronological`, `count-consistent`, and `historical` remain optional modern-environment profiles. Explicitly running all five registered profiles gives fifteen runs.

The chronological pair tests software-version sensitivity. The historical pair changes both software versions and scaling scope, so their differences cannot be attributed solely to library versions. The raw-weather-only scaler is a serializable helper in `fitw/scaling.py`; category-only models have no weather columns to scale, and their operational/category input values remain unchanged. These profiles do not recover absent original feature selections, exact split membership, preprocessing execution, or fitted hyperparameters. Both local environments use Python 3.12.14 on macOS instead of Python 3.12.3 on Windows. Wheel builds and operating-system dependencies also differ. Parquet I/O and plotting package versions are pinned for this replication, but their exact original versions were not established. Each experiment records its actual Python and core library versions in its configuration.

## Correction of the sampled parameter policy

The initial local implementation sampled `scale_pos_weight` but removed it before creating XGBRegressor. That assumption was incorrect. A small synthetic regression counterexample, which does not contribute any replication observations or metrics, demonstrates changed predictions when the parameter changes and labels include exactly one. The corrected implementation passes every sampled hyperparameter unchanged, consistent with the historical author source.

All initial parameter-omitted outputs are preserved as superseded diagnostics under `results/parameter_omission_diagnostics/`, including twelve completed runs and three partial runs. They are excluded from the corrected report and summary. The default six author-version experiments must be completed under the literal parameter policy before drawing numerical replication conclusions. They retain the source-derived protocol choices and the explicit portability caveats described above. A dedicated protocol test verifies both parameter preservation and its effect on a regression counterexample.

## Limits of an exact numerical replication

The paper supplies no final fitted parameters, seed, model object, membership manifest, preprocessing code, feature encoding map, weather-amount window, missingness policy, train-ID definition, or exact plotted arrays. Historical public author code resolves some choices, including seed, train identifier, temporal encoding, and the curve axis, but does not prove that this commit generated Figure 3. Its interactive feature selections and exact training input are still absent. The stated sample count and feature counts are insufficient to recover the remaining choices. Numerical agreement or disagreement alone cannot resolve them.

A completed result should therefore distinguish source claims, the implemented primary protocol, explicitly declared assumptions, sensitivity results, and measured replication scores. Do not equate implementing the same model family and three feature bundles with reproducing the numerical findings.

The categories are deterministic functions of meteorological observations. Any improvement is an empirical modeling result under the chosen fitting procedure, not proof of new information in categories. Operational predictors appear in all three scenarios, and the paper has no operational-only baseline. The reported comparison cannot establish the incremental benefit of weather over scheduling and service information, causal weather effects, advance forecast accuracy, or generalization beyond Oulu.
