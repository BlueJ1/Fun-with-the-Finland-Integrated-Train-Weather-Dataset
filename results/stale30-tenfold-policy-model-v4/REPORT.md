# Native thirty-minute model, version four

**Mean chronological ten-fold CV RMSE: 7.999978167823421 minutes. The saved development score meets the eight-minute threshold.** The margin is only 2.1832176578584495e-05 minutes (0.0013099305947150697 seconds); this extremely narrow crossing does not establish independent generalization performance. All 80,916 development events and 73,560 OOF predictions are retained. No holdout was used.

There are 25 distinct sources and 26 saved roles: 250 fresh fold fits plus 25 fresh full-development fits, and 260 saved fold-role models plus 26 full-role models. `fmilight:1` serves both the departure expert and the auxiliary policy; its ten fold models and full model were copied for the second role, with identical hashes verified. The bundle reproduces the selected frozen source predictions and replays all 3 frozen bounded trials. Native reloads, source/model/frozen hashes, chronological label purging, timestamp ties, query identity, input age/count provenance and unlabeled full-model inference passed.

## Event-type experts

The older accepted timetable's `is_departure == 1` selects the departure expert. Zero or masked missing metadata selects `arrival_default`; invalid finite values are rejected. Each expert has nonnegative weights summing to one. The averaged union weights in `selection.components` identify the fitted source inventory; inference uses the two expert tables below.

### arrival_default

| Source | Weight | Backend | Snapshot profile |
|---|---:|---|---|
| service:3 | 0.05485871133026347 | xgboost | oulu_stale30_service_development.parquet |
| enriched:15 | 0.04045903563317365 | xgboost | oulu_stale30_enriched_development.parquet |
| enriched:7 | 0.03943706721607416 | xgboost | oulu_stale30_enriched_development.parquet |
| lightgbm:6 | 0.054145181351461576 | lightgbm | oulu_stale30_network_development.parquet |
| catjoint:4 | 0.16607476808390437 | catboost | oulu_stale30_network_service_development.parquet |
| robust:6 | 0.16872449485642393 | xgboost | oulu_stale30_network_service_development.parquet |
| turnaround:6 | 0.11427688708264609 | xgboost | oulu_stale30_turnaround_development.parquet |
| turnaround:2 | 0.058902285535655394 | xgboost | oulu_stale30_turnaround_development.parquet |
| catpending:6 | 0.04628079602959251 | catboost | oulu_stale30_network_service_development.parquet |
| ridge:3 | 0.09058589712954788 | ridge | oulu_stale30_turnaround_development.parquet |
| fmicat:1 | 0.009247319430165539 | catboost | oulu_stale30_fmi_development.parquet |
| originpending:2 | 0.009781826069405912 | xgboost | oulu_stale30_origin_development.parquet |
| fmilight:4 | 0.010217910914496918 | lightgbm | oulu_stale30_fmi_development.parquet |
| fmilight:6 | 0.007600201128472185 | lightgbm | oulu_stale30_fmi_development.parquet |
| readypending:2 | 0.04376050189178089 | xgboost | oulu_stale30_readiness_development.parquet |
| neuralorigin:1 | 0.05227893016551331 | neural | oulu_stale30_origin_development.parquet |
| neuralorigin:2 | 0.03336818615142217 | neural | oulu_stale30_origin_development.parquet |

### departure

| Source | Weight | Backend | Snapshot profile |
|---|---:|---|---|
| service:3 | 0.09278267858975395 | xgboost | oulu_stale30_service_development.parquet |
| enriched:7 | 0.08110664408559884 | xgboost | oulu_stale30_enriched_development.parquet |
| turnaround:8 | 0.049887525476702346 | xgboost | oulu_stale30_turnaround_development.parquet |
| fmicat:4 | 0.1110918080111753 | catboost | oulu_stale30_fmi_development.parquet |
| fmicat:3 | 0.1774278568997029 | catboost | oulu_stale30_fmi_development.parquet |
| originrobust:6 | 0.07940248006725245 | xgboost | oulu_stale30_origin_development.parquet |
| fmilight:3 | 0.04972955241755937 | lightgbm | oulu_stale30_fmi_development.parquet |
| fmilight:6 | 0.056194476613603196 | lightgbm | oulu_stale30_fmi_development.parquet |
| fmilight:1 | 0.01734132933327479 | lightgbm | oulu_stale30_fmi_development.parquet |
| fmicatrefined:4 | 0.10144489590555208 | catboost | oulu_stale30_fmi_development.parquet |
| originbagged:1 | 0.028103689095276947 | xgboost | oulu_stale30_origin_development.parquet |
| neuralorigin:1 | 0.0676035917824961 | neural | oulu_stale30_origin_development.parquet |
| neuralorigin:2 | 0.08788347172205162 | neural | oulu_stale30_origin_development.parquet |

## Ordered global policy

The six selected global parameters are shared by both experts. Apply these steps in order:

1. Add bias -0.08698015882760335 minutes to the selected expert prediction.
2. For finite older same-run previous delay at most zero, multiply by 1.1173125644211908.
3. For finite older Oulu arrival-context maximum delay at most 60 minutes, blend with auxiliary `fmilight:1` using auxiliary weight 0.2656019251844185.
4. For older previous delay at most zero and the current transformed prediction exceeding 240.0 minutes, multiply by 1.5187647904724155.
5. Cap no-history departure predictions at 25.83357851588563 minutes. For observed history with oldest pending interval exceeding 360.0 minutes, apply a floor of 2.128548475454262 times that interval.
6. Restore the value before cap/floor for no-history departures with positive older origin actual-presence counts and finite older 720-minute origin maximum last delay exceeding 120 minutes.

Missing context satisfies no corresponding gate. The ordered grammar rejects focal-outcome and identity gates. Numeric policy parameters must be finite, presence counts nonnegative integers, and role predictions finite and aligned. Exact weights, parameters and ordered gates are frozen in `config.json` and `exploration/selection.json`.

## Ten chronological folds

| Fold | RMSE, minutes |
|---:|---:|
| 1 | 8.363944354087305 |
| 2 | 9.674143992734608 |
| 3 | 7.558654344316568 |
| 4 | 7.268535166897552 |
| 5 | 7.091410541666226 |
| 6 | 6.178373321681835 |
| 7 | 8.73414282702277 |
| 8 | 8.299821811569187 |
| 9 | 7.247131925582696 |
| 10 | 9.583623392675458 |

The eight-minute target applies to the arithmetic mean of all ten folds.

## Input age and availability limits

Prediction time equals the scheduled Oulu query time. Actual observations and accepted timetables must strictly precede prediction time minus 30 minutes; equality is excluded. Focal outcomes are excluded even if unusually early. Training labels precede each validation block's earliest snapshot. Later queries can use earlier-query observations only after they pass the same input-age gate. Live estimates, final cancellation flags, untimestamped causes and original unversioned weather columns are excluded.

Recorded `actualTime` and timetable acceptance timestamps are availability proxies; retained timetables are not revision-versioned. FMI measurements must strictly precede prediction time minus 90 minutes, combining the 30-minute cutoff with an assumed 60-minute publication lag. Historical FMI publication and revision clocks are uncertified; quality-control corrections may have occurred after a historical prediction. Manifest-bound clock checks do not certify historical publication availability. Origin proximity does not identify rolling stock.

The full bundle cannot be used before `2023-09-09 13:16:00+00:00`. Its guards reject premature deployment, mismatched scheduled/query times, dated-run identities or profiles, newer timestamps, absent provenance for positive counts, negative/fractional counts, and weakened FMI lag declarations. Source manifests are retained as hashed sidecars and bound to their exact native roles and frozen source catalog.

## Inference and reproduction

Supply separate target-free prepared feature frames for each profile. Shared column names can have different values across profiles, so a merged frame is insufficient. Required filename keys are:

- `oulu_stale30_enriched_development.parquet`
- `oulu_stale30_fmi_development.parquet`
- `oulu_stale30_network_development.parquet`
- `oulu_stale30_network_service_development.parquet`
- `oulu_stale30_origin_development.parquet`
- `oulu_stale30_readiness_development.parquet`
- `oulu_stale30_service_development.parquet`
- `oulu_stale30_turnaround_development.parquet`
- `policy_context`, a separate context frame matching `oulu_stale30_origin_development.parquet`.

All frames must describe the same queries in the same order. Do not supply focal target labels.

```python
from fitw.stale_policy import predict

prediction_minutes = predict(
    profiles,  # filename -> unlabeled aged snapshot, plus policy_context
    model_directory="results/stale30-tenfold-policy-model-v4",
)
```

The bundle consumes prepared snapshots and does not provide a live feed service. Recheck all saved models:

```sh
PYTHONPATH=tmp/catboost-runtime:tmp/lightgbm-runtime:. \
  /Users/uni/Programming/FI-TW/.venv/bin/python -m fitw.stale_policy verify \
  --output results/stale30-tenfold-policy-model-v4 --data-directory data --threads 2
```

Fresh reproduction requires a new empty output directory:

```sh
PYTHONPATH=tmp/catboost-runtime:tmp/lightgbm-runtime:. \
  /Users/uni/Programming/FI-TW/.venv/bin/python -m fitw.stale_policy train \
  --recipe models/stale30_policy.json --data-directory data \
  --output results/stale30-tenfold-policy-model-v4-reproduction --threads 2
```

## Prepare the required snapshots

Use the original frozen cohort at `data/oulu_features.parquet`, the same FI-TW archive, and the recorded dependency versions. These builders restrict target rows to the original development population. Keep the cached FMI measurements under `data/fmi_pellonpaa_development/`; historical service revisions can change downloaded measurements. The recipe rejects different snapshot or manifest hashes rather than silently changing the benchmark.

```sh
.venv/bin/python -m fitw.stale prepare
.venv/bin/python -m fitw.stale augment
.venv/bin/python -m fitw.stale prepare --feature-profile enriched \
  --output data/oulu_stale30_enriched_development.parquet
.venv/bin/python -m fitw.stale prepare --feature-profile network \
  --output data/oulu_stale30_network_development.parquet
.venv/bin/python -m fitw.stale augment \
  --data data/oulu_stale30_network_development.parquet \
  --output data/oulu_stale30_network_service_development.parquet
.venv/bin/python -m fitw.stale_wide
.venv/bin/python -m fitw.stale augment \
  --data data/oulu_stale30_wide_development.parquet \
  --output data/oulu_stale30_wide_service_development.parquet
.venv/bin/python -m fitw.stale_turnaround
.venv/bin/python -m fitw.stale_fmi
.venv/bin/python -m fitw.stale_origin
.venv/bin/python -m fitw.stale_origin_boundary
.venv/bin/python -m fitw.stale_ready
```

Verify the reproduced bundle with `python -m fitw.stale_policy verify --data-directory data --output results/stale30-tenfold-policy-model-v4-reproduction`.

Runtime versions: python 3.12.14, numpy 2.5.3, pandas 3.0.6, sklearn 1.9.1, lightgbm 4.7.0, catboost 1.2.10, xgboost 3.4.1. Optional CatBoost/LightGBM pins are recorded in `requirements-models.txt`. Neural pipelines are serialized with joblib and retain training-only preprocessing/target transforms.

## Artifact provenance

The compact tracked recipe in `models/stale30_policy.json` contains the final model's 25 source configurations, selected weights and policy, three declared bounded trials, and exact input hashes. Training uses it to regenerate native weights, OOF vectors, expanded metadata, source-manifest sidecars and a local frozen recipe. It checks all three trial scores against the recorded values. No discarded search directory or original source prediction file is required.

The fitted bundle and generated artifacts remain local and gitignored. `summary.json` retains the final score and verification counts. The hashes below identify the locally verified bundle, whose fitted model and prediction bytes were preserved during packaging.

`config.json` binds all retained recipe and input-manifest hashes. `result.json` records all 286 saved role-model hashes, original source OOF hashes as historical provenance, and final predictions. Verification recomputes the score and checks model, data, recipe, query and source-age contracts.

| Artifact | SHA-256 |
|---|---|
| `config.json` | `df4958f69029137ea8cdb4545752874647052f9bcad938e78ea785232282eef8` |
| `cv_predictions.parquet` | `da84c7c5e8a6ebc6ff57c509d951f12651ab5917915f9cf53b8130b3bf67e604` |
| `result.json` | `7d63dbbb085a5e4b02b05f77c57dfac75c012ede7bb7fa4c0000d9f3da32f6a3` |
| `verification.json` | `2c90923949f533ffe9d264d8ed596731d807cd50ad63bdce3425a0d51b8aafce` |

## What this score establishes

This is a native reproduction of an extensively CV-tuned development recipe. Source families, mixtures, event-type experts and six global parameters repeatedly reused development CV labels. The score is optimistic and supplies no independent generalization estimate. A margin of 2.1832176578584495e-05 minutes can disappear under minor sampling or availability changes. The historical publication assumptions also limit any deployment claim. No holdout evaluation or external publication was performed.
