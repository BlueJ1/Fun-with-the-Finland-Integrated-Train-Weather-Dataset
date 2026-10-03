# Operational model result

The new model achieves **10.473 RMSE minutes** on the existing chronological test set of 20,230 Oulu events, meeting the requested maximum of 12. MAE is 3.557 minutes and R² is 0.803. Every test row ID and signed target matches the original chronological experiment. Negative delays, extreme delays, and events without available history remain in the score.

| Model | Test RMSE, minutes | Test MAE, minutes | Test R² |
|---|---:|---:|---:|
| Original chronological full-weather XGBoost | 14.269 | 6.368 | 0.635 |
| Earlier same-run delay carried forward, missing history set to zero | 14.893 | 4.668 | 0.602 |
| First new model, basic operational history | 12.285 | 4.282 | 0.729 |
| New model, operational history and timetable context | **10.473** | **3.557** | **0.803** |

The improvement over the original model is 26.6% in observed RMSE. This comparison uses additional operational inputs and a different model configuration. It does not isolate a weather effect. Metrics were recomputed from saved predictions; see [comparison.json](comparison.json).

## Model and selection

The selected model predicts a correction to the last observed same-run delay. If no history is available, the starting prediction is zero. Its 35 inputs include scheduled temporal encodings, train and event metadata, previous delays and their ages, timetable gaps, expected but unresolved earlier route events, planned Oulu dwell, an already observed Oulu arrival, and recent delays and unresolved events from other trains at Oulu.

XGBoost uses 300 depth-3 trees, learning rate 0.05, minimum child weight 20, L2 regularization 10, subsampling 0.9, and all feature columns. Squared-error regression uses equal weights, including `scale_pos_weight=1`. No scaler or learned imputer is needed; missing predictor values remain missing.

Sixteen fixed configurations compare direct prediction and correction to persistence. The selected configuration minimizes mean RMSE over five expanding development folds. Its fold RMSE values are 8.930, 7.728, 8.893, 10.025, and 8.680 minutes, averaging **8.851**. The selection and full search are saved in [selection.json](selection.json) and [search.jsonl](search.jsonl).

The original 80,916 development rows and 20,230 test rows retain their membership. The test period runs from September 9, 2023 through January 1, 2025 under the original actual-time ordering. The final fit uses 80,915 development labels. One label is excluded because it was observed after the earliest scheduled test prediction cutoff. The fourth validation fold similarly excludes one late training label. All fits freeze before their earliest held-out prediction cutoff.

## Input availability

The default prediction cutoff is the Oulu event's scheduled time. A history observation must belong to the same departure date and train number, have an actual observation time strictly before that cutoff, and have a scheduled time strictly before the Oulu event's scheduled time. Equality is excluded. The focal event and its target-derived offsets are never predictors.

Pending events come from earlier timetable positions not yet observed at the cutoff. A later completed route event clears an earlier missing position, even when that completed event has no delay label. Planned route and dwell inputs use timetable information independently of future observation times. Other-train station context excludes the focal dated run and uses the preceding three hours. Its observation scope is the source departure-month file and the train numbers represented in that month's Oulu cohort, so context can be incomplete near month boundaries.

Focal actual-time encodings, focal weather, final cancellation flags, and delay offsets are excluded. Earlier completed events from the validation or test period can provide inputs to subsequent predictions. The fitted model never trains on test labels. This is an online operational estimate requiring a train-observation feed. It is not a forecast before the train's journey begins. Archived actual times are treated as immediate observation availability; historical feed publication and revision times are not verified. Some events occur early, so the scheduled cutoff can follow the physical Oulu event even though that event remains excluded from inputs.

## Evaluation limits

**The holdout was evaluated twice during this work.** The first feature family scored 12.285 RMSE. That result prompted further feature engineering using timetable and station context. Candidate selection within each version still used development folds only, and neither version fits on test labels. The final score is a verified result on a fixed chronological benchmark, but the benchmark was consulted during development. A new future period is needed for an independent generalization estimate.

History is available for 15,587 test events, or 77.05%. The 4,643 events without history are retained and have RMSE 9.656; the available-history subset has RMSE 10.704. Arrival RMSE is 9.655 and departure RMSE is 11.236. Severe individual errors remain; [largest_errors.csv](largest_errors.csv) preserves the 20 largest errors. [monthly_metrics.csv](monthly_metrics.csv) and [history_metrics.csv](history_metrics.csv) provide subgroup diagnostics. These scores describe this observed test period and do not guarantee RMSE below 12 on later data.

## Reproduction and verification

From the repository root, using the main environment and the original archive:

```sh
.venv/bin/python -m fitw.operational prepare
.venv/bin/python -m fitw.operational train
.venv/bin/python -m fitw.operational evaluate
.venv/bin/python -m fitw.operational verify --archive data_archive
.venv/bin/python -m unittest discover -s tests -v
```

The source archive can be supplied with `prepare --archive PATH`; the prepared cohort defaults to `data/oulu_features.parquet`. Preparation retains source hashes and cutoff provenance in its manifest and table. Training defaults to three concurrent candidates with two CPU threads each. `--workers 1` reduces concurrency. `train --feature-profile base` restricts inputs to the first version's 21 predictors; use a separate output directory for that experiment.

For new unlabeled events, combine `scheduled_features(target_events)` and `history_features(target_events, observed_feed)`, then pass the resulting table to `fitw.operational.predict`. Source observations need scheduled time, actual time, signed delay, station UIC, event type, departure date, and train number. Oulu uses station UIC 370. Focal delay labels and actual times are not needed for prediction.

All **33 tests pass**, including 16 operational tests. The saved model reload reproduces every prediction exactly, all reported metrics recompute, source cutoffs and dated-run provenance pass, chronological membership matches the original cohort, and the complete development search selects the saved configuration. Original monthly archive hashes are also verified. See [verification.json](verification.json). [config.json](config.json) records the actual Python and modeling-library versions, feature list, source hashes, and search settings. The first attempt and its verification remain in [the version-one directory](../operational-chronological-v1/).
