"""Development-only chronological CV with a 30-minute-old observation snapshot."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import TimeSeriesSplit
import xgboost
from xgboost import XGBRegressor

from .experiment import metrics
from .operational import FEATURES, candidates, estimator, history_features, purge_training, scheduled_features
from .prepare import sha256
from .spec import TARGET

DEVELOPMENT_ROWS = 80916
MINIMUM_AGE_MINUTES = 30.
KEY = ["departureDate", "trainNumber", "stationUICCode", "type", "scheduledTime"]
STATIC_COLUMNS = KEY + ["trainStopping", "commercialStop", "trainType", "timetableType", "timetableAcceptanceDate"]
EXTRA_FEATURES = ["train_type_ic", "train_type_s", "train_type_p", "train_type_mv",
                  "timetable_adhoc", "history_progress_fraction", "positive_delay_log",
                  "delay_vs_scheduled_gap", "projected_dwell_delay", "pending_minutes_log"]
MODEL_FEATURES = FEATURES + EXTRA_FEATURES
SERVICE_FEATURES = ["service_mean_delay_28d", "service_median_delay_28d", "service_max_delay_28d",
                    "service_p90_delay_28d", "service_last_delay", "service_last_age_minutes",
                    "service_observed_runs_28d", "service_history_coverage_28d",
                    "service_mean_pending_28d", "service_mean_previous_age_28d"]


def add_service_history(frame):
    """Online prior-run summaries. Every contributing label is old at the cutoff.

    Reporting coverage is computed only for earlier predictions whose prediction
    time and outcome time both precede the current observation snapshot.
    """
    out = frame.copy()
    stamp = lambda name: pd.to_datetime(frame[name], utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    actual, prediction, cutoff = stamp("event_time"), stamp("prediction_time"), stamp("observation_cutoff")
    dates, labels = frame.departureDate.to_numpy(), frame[TARGET].to_numpy(dtype=float)
    records, minute_ns = [], 60_000_000_000
    for _, indices in frame.groupby(["query_train_number", "type"], sort=False).groups.items():
        indices = np.asarray(indices)
        for index in indices:
            row = {"index": index}
            if frame.loc[index, "metadata_known"]:
                eligible = indices[(actual[indices] < cutoff[index]) & (prediction[indices] < cutoff[index])
                                   & (actual[indices] >= cutoff[index] - 28 * 1440 * minute_ns)
                                   & (dates[indices] != dates[index])]
                if len(eligible):
                    values = labels[eligible]
                    last = eligible[np.argmax(actual[eligible])]
                    row.update(service_mean_delay_28d=float(values.mean()), service_median_delay_28d=float(np.median(values)),
                               service_max_delay_28d=float(values.max()), service_p90_delay_28d=float(np.quantile(values, .9)),
                               service_last_delay=labels[last], service_last_age_minutes=(cutoff[index] - actual[last]) / minute_ns,
                               service_observed_runs_28d=len(eligible),
                               service_history_coverage_28d=float((frame.loc[eligible, "history_count"] > 0).mean()),
                               service_mean_pending_28d=float(frame.loc[eligible, "pending_count"].mean()),
                               service_mean_previous_age_28d=float(frame.loc[eligible, "previous_age_minutes"].mean()),
                               service_latest_available_time=pd.Timestamp(max(actual[eligible].max(), prediction[eligible].max()), unit="ns", tz="UTC"))
            records.append(row)
    profiles = pd.DataFrame(records).set_index("index").reindex(frame.index)
    for name in SERVICE_FEATURES:
        out[name] = profiles[name] if name in profiles else np.nan
    out["service_latest_available_time"] = pd.to_datetime(profiles["service_latest_available_time"], utc=True) if "service_latest_available_time" in profiles else pd.NaT
    return out


def augment(args):
    frame = add_service_history(load_data(args.data))
    output = Path(args.output)
    frame.to_parquet(output, index=False)
    manifest = json.loads(Path(args.data).with_suffix(".manifest.json").read_text())
    manifest.update(features=MODEL_FEATURES + SERVICE_FEATURES, features_sha256=sha256(output),
                    service_history_policy="same train number and event type; different dated runs; last28days; actual and prediction times strictly before snapshot")
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved age-gated service histories for {len(frame)} development rows", flush=True)


def development_cohort(path):
    """Push the frozen development-row filter into Parquet I/O, before decoding."""
    return pq.read_table(path, filters=[("row_id", "<", DEVELOPMENT_ROWS)]).to_pandas().sort_values("row_id").reset_index(drop=True)


def snapshot_features(targets, schedules, observed):
    """Mask all data newer than the per-event snapshot; never impute future events.

    `observed` may contain later development observations, but history_features
    uses only events occurring before the individual scheduled-time-minus-30m
    cutoff. Timetable rows must also have been accepted before that cutoff.
    """
    query = targets.copy()
    if "stationUICCode" not in query:
        query["stationUICCode"] = 370
    prediction = pd.to_datetime(query.scheduledTime, utc=True)
    deadline = prediction - pd.Timedelta(minutes=MINIMUM_AGE_MINUTES)
    # A full timetable is separate from actual observations. Unobserved events
    # remain NaT regardless of their eventual realization or final delay value.
    observed = observed[KEY + ["actualTime", TARGET]].drop_duplicates(KEY, keep="last")
    feed = schedules.merge(observed, on=KEY, how="left", validate="one_to_one")
    focal = query[KEY].merge(schedules[KEY + ["timetableAcceptanceDate", "trainType", "timetableType"]],
                             on=KEY, how="left", validate="one_to_one")
    focal.index = query.index
    accepted = pd.to_datetime(focal.timetableAcceptanceDate, utc=True, errors="coerce")
    known = accepted.notna() & (accepted < deadline)
    static = scheduled_features(query)
    static.loc[~known, :] = np.nan
    for suffix, name in [("ic", "IC"), ("s", "S"), ("p", "P"), ("mv", "MV")]:
        static["train_type_" + suffix] = (focal.trainType == name).astype(float).where(known)
    static["timetable_adhoc"] = (focal.timetableType == "ADHOC").astype(float).where(known)
    history = history_features(query, feed, horizon_minutes=MINIMUM_AGE_MINUTES,
                               availability_column="timetableAcceptanceDate", audit=True)
    history["history_progress_fraction"] = history.history_count / (history.history_count + history.pending_count).replace(0, np.nan)
    history["positive_delay_log"] = np.log1p(history.previous_delay.clip(lower=0))
    history["delay_vs_scheduled_gap"] = history.previous_delay / history.previous_scheduled_gap_minutes.clip(lower=1)
    history["projected_dwell_delay"] = (history.available_oulu_arrival_delay - history.scheduled_oulu_dwell_minutes).clip(lower=0)
    history["pending_minutes_log"] = np.log1p(history.oldest_pending_minutes.clip(lower=0))
    history["focal_timetable_available_time"] = accepted.where(known)
    history["prediction_time"] = prediction
    history["observation_cutoff"] = deadline
    history["snapshot_age_minutes"] = MINIMUM_AGE_MINUTES
    history["metadata_known"] = known
    # Prediction query identifiers stay in metadata; learned trainNumber is
    # masked separately when no sufficiently old accepted timetable exists.
    static = static.rename(columns={"trainNumber": "feature_train_number"})
    return pd.concat([static, history], axis=1)


def prepare(args):
    cohort = development_cohort(args.cohort)
    if len(cohort) != DEVELOPMENT_ROWS:
        raise ValueError("Frozen development cohort is incomplete")
    root, chunks, sources = Path(args.archive), [], []
    last_label_time = pd.to_datetime(cohort.event_time, utc=True).max()
    for name, targets in cohort.groupby("source_file", sort=True):
        path = root / name
        numbers = targets.trainNumber.unique().tolist()
        maximum_snapshot = min(pd.to_datetime(targets.scheduledTime, utc=True).max() - pd.Timedelta(minutes=30), last_label_time)
        timestamp = maximum_snapshot.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
        common = [("trainCategory", "=", "Long-distance"), ("trainNumber", "in", numbers),
                  ("departureDate", "<=", str(targets.departureDate.max()))]
        schedules = pq.read_table(path, columns=STATIC_COLUMNS, filters=common).to_pandas().drop_duplicates(KEY)
        observed = pq.read_table(path, columns=KEY + ["actualTime", TARGET],
                                 filters=common + [("actualTime", "<", timestamp)]).to_pandas()
        part = snapshot_features(targets, schedules, observed)
        metadata = targets[["row_id", "event_time", "scheduledTime", "departureDate", "trainNumber", "type", TARGET]].copy()
        chunks.append(pd.concat([metadata, part], axis=1))
        sources.append({"file": name, "sha256": sha256(path), "observed_rows": len(observed),
                        "observation_read_before": timestamp})
        print(f"stale snapshot {name}: {len(targets)} development events", flush=True)
    frame = pd.concat(chunks).sort_values("row_id").reset_index(drop=True)
    # Only the predictor uses masked train identity; original query identity
    # remains available for source provenance validation.
    frame["query_train_number"] = frame.trainNumber
    frame["trainNumber"] = frame.pop("feature_train_number")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    manifest = {"rows": len(frame), "development_row_id_upper_exclusive": DEVELOPMENT_ROWS,
                "minimum_input_age_minutes": MINIMUM_AGE_MINUTES, "features": MODEL_FEATURES,
                "holdout_read": False, "sources": sources, "features_sha256": sha256(output),
                "prediction_time": "scheduledTime", "snapshot_time": "scheduledTime minus 30 minutes",
                "timetable_policy": "acceptance strictly before snapshot; unknown or recent metadata masked",
                "weather_used": False, "post_snapshot_pending_status_used": False,
                "metadata_masked_rows": int((~frame.metadata_known).sum()),
                "provenance_limit": "actualTime proxies observation availability; archived acceptance does not version timetable revisions"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2))


def load_data(path):
    frame = pd.read_parquet(path).sort_values("row_id").reset_index(drop=True)
    if frame.row_id.duplicated().any() or (frame.row_id >= DEVELOPMENT_ROWS).any() or (frame.row_id < 0).any():
        raise ValueError("Development-only data must exclude every holdout row")
    if not pd.to_datetime(frame.event_time, utc=True).is_monotonic_increasing:
        raise ValueError("Development event order must be chronological")
    prediction = pd.to_datetime(frame.prediction_time, utc=True)
    cutoff = pd.to_datetime(frame.observation_cutoff, utc=True)
    if not (prediction == pd.to_datetime(frame.scheduledTime, utc=True)).all():
        raise ValueError("Prediction time must equal the scheduled query time")
    if not ((prediction - cutoff) >= pd.Timedelta(minutes=30)).all():
        raise ValueError("Snapshot is less than 30 minutes old")
    if not (pd.to_datetime(frame.prediction_cutoff, utc=True) == cutoff).all():
        raise ValueError("History cutoff disagrees with old-data snapshot")
    for name in ["source_actual_time", "input_latest_actual_time", "input_latest_timetable_time", "focal_timetable_available_time"]:
        values = pd.to_datetime(frame[name], utc=True)
        present = values.notna()
        if not (values[present] < cutoff[present]).all():
            raise ValueError(f"Input age violation in {name}")
    if "service_latest_available_time" in frame:
        values = pd.to_datetime(frame.service_latest_available_time, utc=True)
        present = values.notna()
        if not (values[present] < cutoff[present]).all():
            raise ValueError("Service profile violates old-data snapshot")
    history = frame.history_count > 0
    if frame.loc[history, "source_actual_time"].isna().any():
        raise ValueError("Present history requires observed timestamp provenance")
    static_names = FEATURES[:11] + EXTRA_FEATURES[:5]
    if frame.loc[~frame.metadata_known, static_names].notna().any().any():
        raise ValueError("Unavailable timetable predictors must remain masked")
    if not ((frame.source_departure_date[history] == frame.departureDate[history]).all()
            and (frame.source_train_number[history] == frame.query_train_number[history]).all()):
        raise ValueError("Same-run source provenance mismatch")
    return frame


def folds_for(frame):
    indices = np.arange(len(frame))
    folds = []
    for tr, va in TimeSeriesSplit(n_splits=5).split(indices):
        retained = purge_training(frame, tr, va)
        folds.append((retained, va))
    return folds


def calibrated_candidates():
    settings = [(2, 1, 1, 300, .05), (3, 1, 1, 300, .05), (4, 1, 1, 300, .05),
                (3, 5, 5, 500, .03), (3, 10, 5, 300, .05), (4, 5, 5, 500, .03)]
    return [{"mode": mode, "parameters": {"n_estimators": trees, "max_depth": depth,
             "learning_rate": rate, "min_child_weight": child, "reg_lambda": regularization,
             "subsample": .9, "colsample_bytree": 1., "scale_pos_weight": 1.}}
            for mode in ["residual", "direct"] for depth, child, regularization, trees, rate in settings]


def apply_postprocess(frame, prediction, rule=None):
    """CV-tuned response to missing reports and prolonged old pending events."""
    result = np.asarray(prediction, dtype=float).copy()
    if rule:
        history = frame.history_count.to_numpy() > 0
        departures_without_history = ~history & (frame.is_departure.to_numpy() == 1)
        result[departures_without_history] = np.minimum(result[departures_without_history], rule["nohistory_departure_cap"])
        pending = frame.oldest_pending_minutes.to_numpy()
        extreme = history & (pending > rule["pending_floor_threshold"])
        result[extreme] = np.maximum(result[extreme], rule["pending_floor_multiplier"] * pending[extreme])
    return result


def predict(features, model_directory="results/stale30-development"):
    """Predict unlabeled snapshot_features rows with the complete saved recipe."""
    output = Path(model_directory)
    config = json.loads((output / "config.json").read_text())
    selected = json.loads((output / "selection.json").read_text())
    result = json.loads((output / "result.json").read_text())
    frame = features.copy()
    if "feature_train_number" in frame:
        frame["trainNumber"] = frame.feature_train_number
    prediction_time = pd.to_datetime(frame.prediction_time, utc=True)
    cutoff = pd.to_datetime(frame.observation_cutoff, utc=True)
    if not ((prediction_time - cutoff) >= pd.Timedelta(minutes=30)).all():
        raise ValueError("Prediction inputs must come from a 30-minute-old snapshot")
    if (prediction_time < pd.Timestamp(result["deployment_not_before"])).any():
        raise ValueError("Final model was not available at this historical prediction time")
    for name in ["source_actual_time", "input_latest_actual_time", "input_latest_timetable_time", "focal_timetable_available_time", "service_latest_available_time"]:
        if name in frame:
            values = pd.to_datetime(frame[name], utc=True)
            valid = values.notna()
            if not (values[valid] < cutoff[valid]).all():
                raise ValueError(f"Recent input cannot be used: {name}")
    model = XGBRegressor()
    model.load_model(output / "model.ubj")
    prediction = model.predict(frame[config["features"]].astype(np.float32))
    if selected["configuration"]["mode"] == "residual":
        prediction += frame.previous_delay.fillna(0).to_numpy(dtype=np.float32)
    return apply_postprocess(frame, prediction, selected.get("postprocess"))


def calibration_scores(frame, row_ids, fold_ids, prediction_matrix, configurations):
    """Evaluate a fixed rule grid on genuine development out-of-fold predictions."""
    lookup = frame.set_index("row_id").loc[row_ids]
    records = []
    for index, choice in enumerate(configurations):
        for cap in [10., 20., 30.]:
            for threshold in [120., 180., 240., 300.]:
                for multiplier in [1., 1.25, 1.5, 1.75, 2.]:
                    rule = {"nohistory_departure_cap": cap, "pending_floor_threshold": threshold,
                            "pending_floor_multiplier": multiplier}
                    prediction = apply_postprocess(lookup, prediction_matrix[index], rule)
                    losses = [metrics(lookup.loc[fold_ids == i, TARGET], prediction[fold_ids == i])["rmse"] for i in range(1, 6)]
                    records.append({"candidate": index+1, "configuration": choice, "postprocess": rule,
                                    "fold_rmse": losses, "mean_cv_rmse": float(np.mean(losses))})
    return sorted(records, key=lambda record: record["mean_cv_rmse"])


def train(args):
    frame = load_data(args.data)
    manifest = json.loads(Path(args.data).with_suffix(".manifest.json").read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    folds = folds_for(frame)
    calibrated = getattr(args, "search_profile", "base") == "calibrated"
    configurations = calibrated_candidates() if calibrated else [{"mode": mode, "parameters": p} for mode in ["direct", "residual"] for p in candidates()]
    names = MODEL_FEATURES + SERVICE_FEATURES if getattr(args, "feature_profile", "base") == "service" else MODEL_FEATURES
    config = {"features": names, "data_sha256": sha256(args.data), "input_manifest": manifest,
              "configurations": configurations, "minimum_input_age_minutes": 30., "holdout_used": False,
              "selection": "minimum arithmetic mean of five chronological fold RMSEs",
              "jobs": args.jobs, "workers": args.workers,
              "search_profile": "calibrated" if calibrated else "base",
              "versions": {"python": platform.python_version(), "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    destination = output / "config.json"
    if destination.exists() and json.loads(destination.read_text()) != config:
        raise ValueError("Configuration differs; choose a new output directory")
    destination.write_text(json.dumps(config, indent=2))
    split = []
    for i, (tr, va) in enumerate(folds, 1):
        split.append({"fold": i, "train_rows": len(tr), "validation_rows": len(va),
                      "train_latest_label_time": str(frame.iloc[tr].event_time.max()),
                      "validation_earliest_snapshot": str(frame.iloc[va].observation_cutoff.min()),
                      "first_validation_row_id": int(frame.iloc[va].row_id.min()),
                      "last_validation_row_id": int(frame.iloc[va].row_id.max())})
    (output / "split.json").write_text(json.dumps(split, indent=2))
    X, y = frame[names].astype(np.float32), frame[TARGET].to_numpy(dtype=np.float32)
    base = frame.previous_delay.fillna(0).to_numpy(dtype=np.float32)
    checkpoint = output / "search.jsonl"
    cache = output / ".cv_cache"
    if calibrated:
        cache.mkdir(exist_ok=True)
    records = [json.loads(line) for line in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
    def score(index):
        choice, losses, parts = configurations[index], [], []
        residual = choice["mode"] == "residual"
        for tr, va in folds:
            model = estimator(choice["parameters"], args.jobs).fit(X.iloc[tr], y[tr] - base[tr] if residual else y[tr])
            prediction = model.predict(X.iloc[va]) + (base[va] if residual else 0)
            losses.append(metrics(y[va], prediction)["rmse"])
            if calibrated:
                parts.append(pd.DataFrame({"row_id": frame.iloc[va].row_id.to_numpy(), "fold": len(parts)+1, "prediction": prediction}))
        if calibrated:
            pd.concat(parts).to_parquet(cache / f"candidate_{index+1}_cv.parquet", index=False)
        return {"candidate": index+1, "configuration": choice, "fold_rmse": losses, "mean_cv_rmse": float(np.mean(losses))}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(score, range(len(records), len(configurations))):
            with checkpoint.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            records.append(record)
            print(json.dumps(record), flush=True)
    if calibrated:
        candidates_cv = [pd.read_parquet(cache / f"candidate_{i+1}_cv.parquet") for i in range(len(configurations))]
        row_ids, fold_ids = candidates_cv[0].row_id.to_numpy(), candidates_cv[0].fold.to_numpy()
        for part in candidates_cv:
            np.testing.assert_array_equal(part.row_id, row_ids)
            np.testing.assert_array_equal(part.fold, fold_ids)
        matrix = np.stack([part.prediction.to_numpy() for part in candidates_cv])
        np.savez_compressed(output / "calibration_predictions.npz", row_ids=row_ids, fold_ids=fold_ids, predictions=matrix)
        rules = calibration_scores(frame, row_ids, fold_ids, matrix, configurations)
        (output / "calibration_search.json").write_text(json.dumps(rules, indent=2))
        selected = rules[0]
    else:
        selected = min(records, key=lambda record: record["mean_cv_rmse"])
    (output / "selection.json").write_text(json.dumps(selected, indent=2))
    residual = selected["configuration"]["mode"] == "residual"
    predictions = []
    for i, (tr, va) in enumerate(folds, 1):
        model = estimator(selected["configuration"]["parameters"], args.jobs).fit(X.iloc[tr], y[tr] - base[tr] if residual else y[tr])
        model.save_model(output / f"fold_{i}.ubj")
        part = frame.iloc[va][["row_id", "event_time", "prediction_time", "observation_cutoff", TARGET]].copy()
        raw_prediction = model.predict(X.iloc[va]) + (base[va] if residual else 0)
        part["raw_prediction"] = raw_prediction
        part["prediction"] = apply_postprocess(frame.iloc[va], raw_prediction, selected.get("postprocess"))
        part["fold"] = i
        part["persistence_prediction"] = base[va]
        predictions.append(part)
    pd.concat(predictions).to_parquet(output / "cv_predictions.parquet", index=False)
    model = estimator(selected["configuration"]["parameters"], args.jobs).fit(X, y - base if residual else y)
    model.save_model(output / "model.ubj")
    pd.DataFrame({"feature": names, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    result = {"model": "stale-30-minute-operational-xgboost", "mean_cv_rmse": selected["mean_cv_rmse"],
              "fold_rmse": selected["fold_rmse"], "target_rmse": 9., "target_met": selected["mean_cv_rmse"] <= 9,
              "development_rows": len(frame), "minimum_input_age_minutes": 30., "holdout_used": False,
              "model_sha256": sha256(output / "model.ubj"), "selection": selected,
              "deployment_not_before": str(pd.to_datetime(frame.event_time, utc=True).max() + pd.Timedelta(minutes=30))}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def verify(args):
    frame = load_data(args.data)
    output = Path(args.output)
    config, result = [json.loads((output / name).read_text()) for name in ["config.json", "result.json"]]
    if config["data_sha256"] != sha256(args.data) or result["model_sha256"] != sha256(output / "model.ubj"):
        raise ValueError("Artifact hash differs")
    records = [json.loads(line) for line in (output / "search.jsonl").read_text().splitlines()]
    if config.get("search_profile") == "calibrated":
        cached = np.load(output / "calibration_predictions.npz", allow_pickle=False)
        rules = calibration_scores(frame, cached["row_ids"], cached["fold_ids"], cached["predictions"], config["configurations"])
        saved_rules = json.loads((output / "calibration_search.json").read_text())
        if rules != saved_rules:
            raise ValueError("CV rule search does not recompute")
        expected = rules[0]
    else:
        expected = min(records, key=lambda row: row["mean_cv_rmse"])
    if len(records) != len(config["configurations"]) or result["selection"] != expected:
        raise ValueError("Development model selection is inconsistent")
    saved = pd.read_parquet(output / "cv_predictions.parquet")
    X = frame[config["features"]].astype(np.float32)
    base = frame.previous_delay.fillna(0).to_numpy(dtype=np.float32)
    residual = result["selection"]["configuration"]["mode"] == "residual"
    scores = []
    for i, (tr, va) in enumerate(folds_for(frame), 1):
        model = XGBRegressor()
        model.load_model(output / f"fold_{i}.ubj")
        prediction = model.predict(X.iloc[va]) + (base[va] if residual else 0)
        prediction = apply_postprocess(frame.iloc[va], prediction, result["selection"].get("postprocess"))
        part = saved.loc[saved.fold == i]
        np.testing.assert_array_equal(part.row_id, frame.iloc[va].row_id)
        np.testing.assert_array_equal(part[TARGET], frame.iloc[va][TARGET])
        np.testing.assert_array_equal(part.prediction, prediction)
        assert frame.iloc[tr].event_time.max() < frame.iloc[va].observation_cutoff.min()
        scores.append(metrics(part[TARGET], prediction)["rmse"])
    np.testing.assert_allclose(scores, result["fold_rmse"], rtol=1e-12)
    np.testing.assert_allclose(np.mean(scores), result["mean_cv_rmse"], rtol=1e-12)
    verification = {"passed": True, "mean_cv_rmse": float(np.mean(scores)), "fold_rmse": scores,
                    "cv_prediction_rows": len(saved), "minimum_input_age_minutes": 30,
                    "all_dynamic_and_static_timestamp_checks_passed": True, "holdout_used": False,
                    "fold_models_reload_exactly": True, "training_label_age_checked": True,
                    "model_sha256": result["model_sha256"]}
    (output / "verification.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prep = subcommands.add_parser("prepare")
    prep.add_argument("--archive", default="data_archive")
    prep.add_argument("--cohort", default="data/oulu_features.parquet")
    prep.add_argument("--output", default="data/oulu_stale30_development.parquet")
    prep.set_defaults(function=prepare)
    augmentation = subcommands.add_parser("augment")
    augmentation.add_argument("--data", default="data/oulu_stale30_development.parquet")
    augmentation.add_argument("--output", default="data/oulu_stale30_service_development.parquet")
    augmentation.set_defaults(function=augment)
    for name, function in [("train", train), ("verify", verify)]:
        sub = subcommands.add_parser(name)
        sub.add_argument("--data", default="data/oulu_stale30_development.parquet")
        sub.add_argument("--output", default="results/stale30-development")
        sub.add_argument("--jobs", type=int, default=2)
        sub.add_argument("--workers", type=int, default=3)
        if name == "train":
            sub.add_argument("--feature-profile", choices=["base", "service"], default="base")
            sub.add_argument("--search-profile", choices=["base", "calibrated"], default="base")
        sub.set_defaults(function=function)
    args = parser.parse_args()
    if getattr(args, "jobs", 1) < 1 or getattr(args, "workers", 1) < 1:
        parser.error("jobs and workers must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
