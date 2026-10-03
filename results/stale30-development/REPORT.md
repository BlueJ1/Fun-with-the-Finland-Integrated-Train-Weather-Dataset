# Thirty-minute-old input model

The rebuilt model achieves **8.976618 minutes mean chronological CV RMSE**, meeting the requested maximum of 9. All observed inputs come from snapshots more than 30 minutes before the scheduled prediction time. The holdout was not used for this rebuild.

| Fold | Validation events | RMSE, minutes | MAE, minutes |
|---|---:|---:|---:|
| 1 | 13,486 | 9.413 | 4.576 |
| 2 | 13,486 | 7.445 | 3.964 |
| 3 | 13,486 | 8.309 | 3.591 |
| 4 | 13,486 | 10.464 | 3.762 |
| 5 | 13,486 | 9.252 | 3.832 |
| Arithmetic mean | 67,430 total | **8.977** | 3.945 |

The goal applies to the mean across folds; individual folds can exceed 9. Carrying forward the last eligible delay averages 12.952 RMSE. The initial rebuilt model averaged 10.533 RMSE and is preserved in [the first-version directory](../stale30-development-v1/). The final result is about 30.7% lower than persistence and 14.8% lower than the initial rebuilt model.

## Thirty-minute snapshot

Prediction time is the scheduled Oulu event time. The input snapshot is exactly 30 minutes earlier. Event observations, pending-event status, other-train station context, and accepted timetable metadata all use that older snapshot. Strict comparisons exclude data exactly on the snapshot boundary too. Pending status is never updated with observations from the intervening 30 minutes.

The youngest actual observation used anywhere in a prepared row is **30 minutes and 1 second old** at prediction time. The youngest source timetable acceptance is 34.083 minutes old; the youngest focal timetable acceptance is 32.450 minutes old. Nine events had recently accepted timetable metadata. Their predictors are masked, and their labels remain in training and validation. See [age_audit.json](age_audit.json).

Raw weather has no retained weather-observation timestamp, so it is excluded. Focal actual-time encodings, focal target values, delay offsets, and final cancellation flags are excluded too. Negative delays, extreme delays, and events without history remain in the evaluation.

The final model uses 45 predictors from schedule metadata and older operational history. Planned future timetable events can be used only when their timetable acceptance precedes the old snapshot. Occurrence timestamps stand in for observation availability, and the archive does not preserve historic timetable revisions. The age audit verifies the recorded event and acceptance timestamps; it cannot certify original feed publication or revision times.

## Development-only evaluation

The cohort reader pushes `row_id < 80916` into Parquet I/O. It loads only the original 80,916 development events. Archive timetable columns and actual observations are read separately. Actual observations are filtered before the latest permissible development snapshot and never after the final development label time. Whole monthly files are hashed for provenance, but held-out labels and metrics do not enter this rebuild.

Five expanding-window folds preserve the existing development chronology. Each fit excludes labels whose actual time is not strictly before the earliest validation snapshot, so training labels also respect the 30-minute age constraint. Validation models remain fixed within their blocks. Earlier validation events can become online inputs to later predictions once they are old enough; they do not enter model fitting. [split.json](split.json) records all label cutoffs and row counts.

This is a **CV-selected score**, not an independent performance estimate. Development CV selected the model and its response rules. The final search evaluates 12 tree configurations and 720 combinations of response rules. Earlier exploration included 16 baseline configurations, other transformations, and one pending-delay residual candidate. [search_provenance.json](search_provenance.json) records those budgets. The holdout was not consulted during this rebuild. It had been evaluated for the older model in earlier work, so it is not a historically pristine holdout.

## Final model

The selected direct XGBoost model has 500 depth-4 trees, learning rate 0.03, minimum child weight 5, L2 regularization 5, row subsampling 0.9, and all input columns. It uses squared-error regression and equal weights, including `scale_pos_weight=1`.

The complete prediction recipe also includes two rules chosen by chronological CV:

- For departures with no eligible same-run history, cap the prediction at 20 minutes. This limits false alarms caused by habitual missing upstream reports.
- When eligible history exists and the oldest unresolved event at the old snapshot is more than 300 minutes overdue, floor the prediction at 1.5 times that overdue duration. This permits extrapolation for prolonged disruptions.

These rules use only the old snapshot. They change predictions, never labels or cohort membership. They can miss severe origin-departure delays or overestimate some prolonged missing-report cases. Their generalization beyond the selected CV periods is unmeasured. Load the model through `fitw.stale.predict`, which applies the complete saved recipe and checks snapshot age and model deployment time.

The final model fits all 80,916 development labels. Its simulated earliest deployment time is September 9, 2023 at 13:16 UTC, 30 minutes after the last training label. No holdout labels are included in that fit. An optional `augment` command adds age-gated summaries of prior dated service runs for future experiments; those additional features are not used by the selected model.

## Reproduction

Use the existing main Python environment and source archive:

```sh
.venv/bin/python -m fitw.stale prepare --archive data_archive
.venv/bin/python -m fitw.stale train --search-profile calibrated
.venv/bin/python -m fitw.stale verify
.venv/bin/python -m unittest discover -s tests -v
```

Preparation defaults to `data/oulu_features.parquet` as the cohort source and saves `data/oulu_stale30_development.parquet`. Override `--cohort` or `--archive` for existing files elsewhere. Training and verification accept `--data` and `--output`. Use a new output directory when changing configuration. Training checkpoints raw candidates, saves compressed out-of-fold calibration predictions, and selects response rules solely from those development predictions. Three candidates run concurrently with two CPU threads each by default; `--workers 1` reduces concurrency.

For an unlabeled prediction, call `snapshot_features(target_events, accepted_timetable, observed_feed)`, then `predict(features, model_directory)`. The timetable needs departure date, train number, station UIC, event type, scheduled time, stop metadata, train type, timetable type, and acceptance time. Observations need event keys, actual time, and signed delay. Oulu's station UIC is 370. No focal actual time or delay label is required.

## Verification

All **49 tests pass**, including 16 stale-model tests. They cover strict age boundaries, stale pending status and cleanup, context and timetable gating, future-data perturbation independence, masked unavailable metadata, pushed-down development reading, label-age purging, optional prior-service profiles, and the complete prediction rules.

The independent verifier reloads all five fold models and reproduces every saved prediction exactly. It recomputes all 720 final rule combinations from the compressed raw CV predictions, confirms the saved selection, and recomputes the arithmetic mean of the five RMSE values. It also validates age provenance and training-label cutoffs. [verification.json](verification.json) records the successful result. [config.json](config.json), [selection.json](selection.json), [calibration_search.json](calibration_search.json), [cv_metrics.csv](cv_metrics.csv), and [cv_predictions.parquet](cv_predictions.parquet) preserve the evidence.
