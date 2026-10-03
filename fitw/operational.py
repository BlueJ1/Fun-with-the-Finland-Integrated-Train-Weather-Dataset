"""Chronological delay model using observations available before scheduled events.

Run preparation, development-only selection, and held-out evaluation separately.
Same-run delay history is an online input, never a training label from the future.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from .experiment import describe_split, metrics, split_indices
from .prepare import sha256
from .spec import TARGET

HISTORY_COLUMNS = ["departureDate", "trainNumber", "stationUICCode", "type",
                   "scheduledTime", "actualTime", TARGET]
FEATURES = ["trainNumber", "trainStopping", "commercialStop", "is_departure",
            "month_sin", "month_cos", "hour_sin", "hour_cos", "weekday_sin",
            "weekday_cos", "day_of_month", "history_count", "previous_delay",
            "previous_age_minutes", "previous_scheduled_gap_minutes",
            "previous_station", "previous_is_departure", "second_previous_delay",
            "delay_change", "history_mean_delay", "history_max_delay",
            "pending_count", "oldest_pending_minutes", "next_expected_gap_minutes",
            "origin_station", "scheduled_run_elapsed_minutes", "scheduled_route_position",
            "scheduled_oulu_dwell_minutes", "previous_is_oulu", "available_oulu_arrival_delay",
            "oulu_recent_mean_delay", "oulu_recent_max_delay", "oulu_recent_count",
            "oulu_pending_count", "oulu_oldest_pending_minutes"]


def history_features(targets, observations, horizon_minutes=0, availability_column=None, audit=False):
    """Only same-run events scheduled earlier AND observed before the cutoff.

    Equality is excluded in both comparisons. This prevents the focal event,
    simultaneous arrivals/departures, and future timetable events from entering.
    Missing history remains missing rather than changing cohort membership.
    """
    if not np.isfinite(horizon_minutes) or horizon_minutes < 0:
        raise ValueError("Prediction horizon must be nonnegative")
    out = pd.DataFrame(index=targets.index)
    scheduled = pd.to_datetime(targets.scheduledTime, utc=True)
    cutoff = scheduled - pd.Timedelta(minutes=horizon_minutes)
    obs = observations.copy()
    obs["observed_time"] = pd.to_datetime(obs.actualTime, utc=True, errors="coerce")
    obs["scheduled_time"] = pd.to_datetime(obs.scheduledTime, utc=True, errors="coerce")
    if availability_column:
        obs["available_time"] = pd.to_datetime(obs[availability_column], utc=True, errors="coerce")
    obs = obs.loc[obs.scheduled_time.notna()].sort_values("observed_time", kind="stable")

    def arrays(frame):
        # Explicit nanosecond conversion avoids pandas' us/ns dtype differences.
        times = lambda name: frame[name].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        result = {"actual": times("observed_time"), "scheduled": times("scheduled_time"),
                "valid_actual": frame.observed_time.notna().to_numpy(),
                "delay": frame[TARGET].to_numpy(dtype=float),
                "station": frame.stationUICCode.to_numpy(), "type": frame.type.to_numpy(),
                "date": frame.departureDate.to_numpy(), "number": frame.trainNumber.to_numpy()}
        if availability_column:
            result["available"] = times("available_time")
            result["valid_available"] = frame.available_time.notna().to_numpy()
        return result

    groups = {key: arrays(frame) for key, frame in obs.groupby(["departureDate", "trainNumber"], sort=False)}
    local = arrays(obs.loc[obs.stationUICCode == 370])
    minute_ns = 60_000_000_000
    records = []
    for key, group in targets.groupby(["departureDate", "trainNumber"], sort=False):
        source = groups.get(key)
        other_run = ~((local["date"] == key[0]) & (local["number"] == key[1]))
        for index in group.index:
            deadline, focal = cutoff.loc[index].value, scheduled.loc[index].value
            record = {"index": index, "history_count": 0}
            used_actual, used_static = [], []
            if source is not None:
                known = (source["valid_available"] & (source["available"] < deadline)) if availability_column else np.ones(len(source["actual"]), dtype=bool)
                planned = (source["scheduled"] < focal) & known
                completed = planned & source["valid_actual"] & (source["actual"] < deadline)
                if completed.any():
                    used_actual.append(source["actual"][completed].max())
                if availability_column and planned.any():
                    used_static.append(source["available"][planned].max())
                eligible = np.flatnonzero(completed & np.isfinite(source["delay"]))
                record["history_count"] = len(eligible)
                planned_indices = np.flatnonzero(planned)
                if len(planned_indices):
                    order = np.lexsort((source["type"][planned_indices],
                                        source["station"][planned_indices],
                                        source["scheduled"][planned_indices]))
                    first = planned_indices[order[0]]
                    record.update(origin_station=source["station"][first],
                                  scheduled_run_elapsed_minutes=(focal - source["scheduled"][first]) / minute_ns,
                                  scheduled_route_position=len(planned_indices))
                arrivals = planned & (source["station"] == 370) & (source["type"] == "ARRIVAL")
                if group.loc[index, "type"] == "DEPARTURE" and arrivals.any():
                    record["scheduled_oulu_dwell_minutes"] = (focal - source["scheduled"][arrivals].max()) / minute_ns
                pending = planned & (source["scheduled"] < deadline) & (~source["valid_actual"] | (source["actual"] >= deadline))
                if completed.any():
                    pending &= source["scheduled"] > source["scheduled"][completed].max()
                record["pending_count"] = int(pending.sum())
                record["oldest_pending_minutes"] = (deadline - source["scheduled"][pending].min()) / minute_ns if pending.any() else 0.
                if len(eligible):
                    last = eligible[-1]
                    values = source["delay"][eligible]
                    record.update(previous_delay=source["delay"][last],
                                  previous_age_minutes=(deadline - source["actual"][last]) / minute_ns,
                                  previous_scheduled_gap_minutes=(focal - source["scheduled"][last]) / minute_ns,
                                  previous_station=source["station"][last],
                                  previous_is_departure=float(source["type"][last] == "DEPARTURE"),
                                  previous_is_oulu=float(source["station"][last] == 370),
                                  history_mean_delay=float(values.mean()), history_max_delay=float(values.max()),
                                  source_actual_time=pd.Timestamp(source["actual"][last], unit="ns", tz="UTC"),
                                  source_scheduled_time=pd.Timestamp(source["scheduled"][last], unit="ns", tz="UTC"),
                                  source_departure_date=key[0], source_train_number=key[1])
                    upcoming = planned & (source["scheduled"] > source["scheduled"][last])
                    record["next_expected_gap_minutes"] = (focal - source["scheduled"][upcoming].min()) / minute_ns if upcoming.any() else 0.
                    available_arrivals = eligible[(source["station"][eligible] == 370) & (source["type"][eligible] == "ARRIVAL")]
                    if len(available_arrivals):
                        record["available_oulu_arrival_delay"] = source["delay"][available_arrivals[-1]]
                    if len(eligible) > 1:
                        record["second_previous_delay"] = source["delay"][eligible[-2]]
                        record["delay_change"] = source["delay"][last] - source["delay"][eligible[-2]]
            beginning = deadline - 180 * minute_ns
            local_known = (local["valid_available"] & (local["available"] < deadline)) if availability_column else np.ones(len(local["actual"]), dtype=bool)
            recent = (other_run & local_known & local["valid_actual"] & (local["actual"] < deadline)
                      & (local["actual"] >= beginning) & np.isfinite(local["delay"]))
            waiting = (other_run & local_known & (local["scheduled"] < deadline) & (local["scheduled"] >= beginning)
                       & (~local["valid_actual"] | (local["actual"] >= deadline)))
            # A later completed event on another train also resolves its earlier
            # missing station observation. Completion needs no finite delay label.
            for pending_index in np.flatnonzero(waiting):
                other = groups[(local["date"][pending_index], local["number"][pending_index])]
                other_known = (other["valid_available"] & (other["available"] < deadline)) if availability_column else np.ones(len(other["actual"]), dtype=bool)
                later_done = (other_known & other["valid_actual"] & (other["actual"] < deadline)
                              & (other["scheduled"] >= local["scheduled"][pending_index]))
                if later_done.any():
                    used_actual.append(other["actual"][later_done].max())
                    if availability_column:
                        used_static.append(other["available"][later_done].max())
                    waiting[pending_index] = False
            if recent.any():
                used_actual.append(local["actual"][recent].max())
            if availability_column and (recent | waiting).any():
                used_static.append(local["available"][recent | waiting].max())
            recent_values = local["delay"][recent]
            record.update(oulu_recent_count=int(recent.sum()),
                          oulu_recent_mean_delay=float(recent_values.mean()) if len(recent_values) else np.nan,
                          oulu_recent_max_delay=float(recent_values.max()) if len(recent_values) else np.nan,
                          oulu_pending_count=int(waiting.sum()),
                          oulu_oldest_pending_minutes=(deadline - local["scheduled"][waiting].min()) / minute_ns if waiting.any() else 0.)
            if audit:
                record["input_latest_actual_time"] = pd.Timestamp(max(used_actual), unit="ns", tz="UTC") if used_actual else pd.NaT
                record["input_latest_timetable_time"] = pd.Timestamp(max(used_static), unit="ns", tz="UTC") if used_static else pd.NaT
            records.append(record)
    history = pd.DataFrame(records).set_index("index").reindex(targets.index)
    for name in FEATURES[11:]:
        out[name] = history[name] if name in history else np.nan
    for name in ["source_actual_time", "source_scheduled_time", "source_departure_date", "source_train_number"]:
        out[name] = history[name] if name in history else pd.NaT if "time" in name else None
    out["prediction_cutoff"] = cutoff
    if audit:
        for name in ["input_latest_actual_time", "input_latest_timetable_time"]:
            out[name] = pd.to_datetime(history[name], utc=True)
    return out


def scheduled_features(frame):
    """Timetable and event metadata only. No focal actual time or weather."""
    event = pd.to_datetime(frame.scheduledTime, utc=True)
    out = frame[["trainNumber", "trainStopping", "commercialStop"]].astype(float).copy()
    out["is_departure"] = (frame.type == "DEPARTURE").astype(float)
    hour = event.dt.hour + event.dt.minute / 60 + event.dt.second / 3600
    for name, value, period in [("month", event.dt.month, 12), ("hour", hour, 24),
                                ("weekday", event.dt.dayofweek, 7)]:
        out[name + "_sin"] = np.sin(2 * np.pi * value / period)
        out[name + "_cos"] = np.cos(2 * np.pi * value / period)
    out["day_of_month"] = event.dt.day.astype(float)
    return out


def prepare(args):
    cohort_path = Path(args.cohort)
    cohort = pd.read_parquet(cohort_path).sort_values("row_id").reset_index(drop=True)
    files = sorted(p for p in Path(args.archive).glob("matched_data_flat_*.parquet")
                   if 2018 <= int(p.stem.split("_")[-2]) <= 2024)
    if len(files) != 84:
        raise ValueError(f"Expected 84 source months; found {len(files)}")
    pieces, sources = [], []
    for i, path in enumerate(files):
        selected = cohort.loc[cohort.source_file == path.name]
        if not len(selected):
            continue
        observations = pq.read_table(path, columns=HISTORY_COLUMNS,
                                     filters=[("trainCategory", "=", "Long-distance"),
                                              ("trainNumber", "in", selected.trainNumber.unique().tolist())]).to_pandas()
        pieces.append(history_features(selected, observations, args.horizon_minutes))
        sources.append({"file": path.name, "sha256": sha256(path), "observations": len(observations)})
        print(f"history {i+1}/{len(files)}: {len(selected)} Oulu events", flush=True)
    history = pd.concat(pieces).reindex(cohort.index)
    metadata = cohort[["row_id", "event_time", "scheduledTime", "departureDate", "trainNumber", "type", TARGET]]
    predictors = pd.concat([scheduled_features(cohort), history], axis=1)
    # trainNumber is both metadata and a predictor; keep exactly one column.
    result = pd.concat([metadata, predictors.drop(columns="trainNumber")], axis=1)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(dest, index=False)
    manifest = {"cohort_sha256": sha256(cohort_path), "features_sha256": sha256(dest),
                "rows": len(result), "features": FEATURES, "sources": sources,
                "horizon_minutes": args.horizon_minutes,
                "availability_policy": "same dated run; source actualTime < scheduledTime minus horizon; source scheduledTime < focal scheduledTime",
                "excluded": ["focal actualTime predictors", "raw/offset target fields", "focal weather", "post-event flags"],
                "history_coverage": float((result.history_count > 0).mean())}
    dest.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2))


def candidates():
    # Fixed before looking at any final holdout scores. Equal regression weights.
    return [{"n_estimators": trees, "max_depth": depth, "learning_rate": rate,
             "min_child_weight": child, "reg_lambda": regularization,
             "subsample": .9, "colsample_bytree": 1., "scale_pos_weight": 1.}
            for trees, depth, rate, child, regularization in [
                (300, 3, .05, 20, 10), (500, 3, .03, 20, 20),
                (300, 4, .05, 30, 20), (500, 4, .03, 30, 30),
                (300, 5, .05, 50, 30), (500, 5, .03, 50, 50),
                (300, 6, .05, 50, 50), (500, 6, .03, 100, 100),
            ]]


def estimator(params, jobs):
    return XGBRegressor(**params, objective="reg:squarederror", tree_method="hist",
                        n_jobs=jobs, random_state=42)


def purge_training(frame, training, validation):
    """Freeze fits before the earliest prediction cutoff in each held-out block."""
    first_cutoff = pd.to_datetime(frame.iloc[validation].prediction_cutoff, utc=True).min()
    available = pd.to_datetime(frame.iloc[training].event_time, utc=True) < first_cutoff
    retained = training[available.to_numpy()]
    if not len(retained):
        raise ValueError("Prediction cutoff leaves no available training labels")
    return retained


def load_data(path):
    frame = pd.read_parquet(path).sort_values("row_id").reset_index(drop=True)
    if frame.row_id.duplicated().any() or not frame.event_time.is_monotonic_increasing:
        raise ValueError("Data must retain unique cohort row IDs and chronological event order")
    observed = pd.to_datetime(frame.source_actual_time, utc=True)
    scheduled = pd.to_datetime(frame.source_scheduled_time, utc=True)
    cutoff = pd.to_datetime(frame.prediction_cutoff, utc=True)
    focal_scheduled = pd.to_datetime(frame.scheduledTime, utc=True)
    available = frame.history_count > 0
    if not ((observed[available] < cutoff[available]).all()
            and (scheduled[available] < focal_scheduled[available]).all()
            and (frame.source_departure_date[available] == frame.departureDate[available]).all()
            and (frame.source_train_number[available] == frame.trainNumber[available]).all()):
        raise ValueError("History provenance violates prediction cutoff or dated run")
    return frame


def train(args):
    frame = load_data(args.data)
    dev, test = split_indices(frame, "chronological")
    folds = list(TimeSeriesSplit(n_splits=5).split(dev))
    split = describe_split(frame, dev, test, folds, "chronological")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    params = [{"mode": mode, "parameters": p} for mode in ["direct", "residual"] for p in candidates()]
    names = FEATURES[:21] if getattr(args, "feature_profile", "context") == "base" else FEATURES
    purged_folds = [(purge_training(frame, dev[tr], dev[va]), dev[va]) for tr, va in folds]
    final_training = purge_training(frame, dev, test)
    manifest_path = Path(args.data).with_suffix(".manifest.json")
    config = {"data_sha256": sha256(args.data), "features": names, "candidates": params,
              "selection": "minimum mean RMSE over five expanding development folds",
              "seed": 42, "jobs": args.jobs, "workers": getattr(args, "workers", 3), "python": platform.python_version(),
              "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__,
              "purged_fold_training_rows": [len(tr) for tr, _ in purged_folds],
              "final_training_rows": len(final_training),
              "input_manifest": json.loads(manifest_path.read_text())}
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output configuration changed; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2))
    (output / "split.json").write_text(json.dumps(split, indent=2))
    pd.DataFrame({"row_id": frame.row_id, "partition": np.where(np.arange(len(frame)) < len(dev), "development", "test")}).to_csv(output / "split_membership.csv", index=False)
    X = frame[names].astype(np.float32)
    y = frame[TARGET].to_numpy(dtype=np.float32)
    base = frame.previous_delay.fillna(0).to_numpy(dtype=np.float32)
    checkpoint = output / "search.jsonl"
    search = [json.loads(line) for line in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
    def score_candidate(i):
        losses = []
        residual = params[i]["mode"] == "residual"
        for tr, va in purged_folds:
            fitted = estimator(params[i]["parameters"], args.jobs).fit(X.iloc[tr], y[tr] - base[tr] if residual else y[tr])
            pred = fitted.predict(X.iloc[va]) + (base[va] if residual else 0)
            losses.append(metrics(y[va], pred)["rmse"])
        record = {"candidate": i+1, "params": params[i], "fold_rmse": losses,
                  "mean_cv_rmse": float(np.mean(losses))}
        return record

    with ThreadPoolExecutor(max_workers=getattr(args, "workers", 3)) as pool:
        # map preserves candidate order, so every checkpoint remains a prefix.
        for record in pool.map(score_candidate, range(len(search), len(params))):
            with checkpoint.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            search.append(record)
            print(json.dumps(record), flush=True)
    best = min(search, key=lambda item: item["mean_cv_rmse"])
    residual = best["params"]["mode"] == "residual"
    tr = final_training
    fitted = estimator(best["params"]["parameters"], args.jobs).fit(X.iloc[tr], y[tr] - base[tr] if residual else y[tr])
    fitted.save_model(output / "model.ubj")
    (output / "selection.json").write_text(json.dumps(best, indent=2))
    pd.DataFrame({"feature": names, "importance": fitted.feature_importances_}).sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    print("Model frozen. Run the evaluate subcommand for the final holdout.", flush=True)


def predict(features, model_directory="results/operational-chronological"):
    """Predict prepared feature rows without requiring focal delay labels.

    Build inputs with scheduled_features() and history_features() using the
    observed feed at prediction time. Input column order follows FEATURES.
    """
    output = Path(model_directory)
    selection = json.loads((output / "selection.json").read_text())
    names = json.loads((output / "config.json").read_text())["features"]
    fitted = XGBRegressor()
    fitted.load_model(output / "model.ubj")
    prediction = fitted.predict(features[names].astype(np.float32))
    if selection["params"]["mode"] == "residual":
        prediction += features.previous_delay.fillna(0).to_numpy(dtype=np.float32)
    return prediction


def evaluate(args):
    frame = load_data(args.data)
    output = Path(args.output)
    config = json.loads((output / "config.json").read_text())
    if config["data_sha256"] != sha256(args.data):
        raise ValueError("Evaluation input differs from frozen training input")
    dev, test = split_indices(frame, "chronological")
    selection = json.loads((output / "selection.json").read_text())
    pred = predict(frame.iloc[test], output)
    y = frame.iloc[test][TARGET].to_numpy()
    persistence = frame.iloc[test].previous_delay.fillna(0).to_numpy()
    result = {"model": "operational-history-xgboost", "test_metrics": metrics(y, pred),
              "persistence_metrics": metrics(y, persistence), "development_rows": len(dev),
              "test_rows": len(test), "selection": selection,
              "final_training_rows": config["final_training_rows"],
              "test_history_coverage": float((frame.iloc[test].history_count > 0).mean()),
              "rmse_target": 12., "rmse_target_met": bool(metrics(y, pred)["rmse"] <= 12),
              "model_sha256": sha256(output / "model.ubj"),
              "by_event_type": {kind: metrics(y[mask], pred[mask]) for kind in ["ARRIVAL", "DEPARTURE"]
                                if (mask := (frame.iloc[test].type == kind).to_numpy()).any()}}
    predictions = frame.iloc[test][["row_id", "event_time", "scheduledTime", "departureDate", "trainNumber", "type", TARGET,
                                   "prediction_cutoff", "source_actual_time", "source_scheduled_time", "previous_delay", "history_count"]].copy()
    predictions["prediction"] = pred
    predictions["persistence_prediction"] = persistence
    predictions.to_parquet(output / "predictions.parquet", index=False)
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def verify(args):
    """Independently check saved predictions, membership, provenance and metrics."""
    frame = load_data(args.data)
    output = Path(args.output)
    config = json.loads((output / "config.json").read_text())
    result = json.loads((output / "result.json").read_text())
    selected = json.loads((output / "selection.json").read_text())
    records = [json.loads(line) for line in (output / "search.jsonl").read_text().splitlines()]
    if len(records) != len(config["candidates"]) or any(len(r["fold_rmse"]) != 5 for r in records):
        raise ValueError("Incomplete development search")
    if selected != min(records, key=lambda record: record["mean_cv_rmse"]):
        raise ValueError("Saved model selection differs from development scores")
    if config["data_sha256"] != sha256(args.data) or result["model_sha256"] != sha256(output / "model.ubj"):
        raise ValueError("Artifact hash mismatch")
    dev, test = split_indices(frame, "chronological")
    saved = pd.read_parquet(output / "predictions.parquet")
    np.testing.assert_array_equal(saved.row_id, frame.iloc[test].row_id)
    np.testing.assert_array_equal(saved[TARGET], frame.iloc[test][TARGET])
    cohort = pd.read_parquet(args.cohort).sort_values("row_id").reset_index(drop=True)
    if sha256(args.cohort) != config["input_manifest"]["cohort_sha256"]:
        raise ValueError("Original cohort hash differs from preparation manifest")
    np.testing.assert_array_equal(frame.row_id, cohort.row_id)
    np.testing.assert_array_equal(frame[TARGET], cohort[TARGET])
    np.testing.assert_array_equal(frame.event_time, cohort.event_time)
    model = XGBRegressor()
    model.load_model(output / "model.ubj")
    prediction = model.predict(frame.iloc[test][config["features"]].astype(np.float32))
    if selected["params"]["mode"] == "residual":
        prediction += frame.iloc[test].previous_delay.fillna(0).to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(prediction, saved.prediction)
    measured = metrics(saved[TARGET], saved.prediction)
    for name, value in measured.items():
        if value is not None:
            np.testing.assert_allclose(value, result["test_metrics"][name], rtol=1e-12)
    folds = list(TimeSeriesSplit(n_splits=5).split(dev))
    absolute_folds = [(dev[tr], dev[va]) for tr, va in folds] + [(dev, test)]
    for training, validation in absolute_folds:
        retained = purge_training(frame, training, validation)
        assert frame.iloc[retained].event_time.max() < frame.iloc[validation].prediction_cutoff.min()
    source_count = 0
    if getattr(args, "archive", None):
        for source in config["input_manifest"]["sources"]:
            if sha256(Path(args.archive) / source["file"]) != source["sha256"]:
                raise ValueError(f"Archive source hash differs: {source['file']}")
            source_count += 1
    verification = {"passed": True, "prediction_rows": len(saved), "rmse": measured["rmse"],
                    "cohort_and_test_membership_unchanged": True,
                    "reload_predictions_exact": True, "metrics_recomputed": True,
                    "history_cutoffs_and_dated_runs_checked": True,
                    "training_label_cutoffs_checked": True,
                    "complete_development_search_checked": True,
                    "archive_source_files_hashed": source_count,
                    "features_sha256": config["data_sha256"], "model_sha256": result["model_sha256"]}
    (output / "verification.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--archive", default="data_archive")
    preparation.add_argument("--cohort", default="data/oulu_features.parquet")
    preparation.add_argument("--output", default="data/oulu_operational.parquet")
    preparation.add_argument("--horizon-minutes", type=float, default=0)
    preparation.set_defaults(function=prepare)
    for command, function in [("train", train), ("evaluate", evaluate), ("verify", verify)]:
        sub = commands.add_parser(command)
        sub.add_argument("--data", default="data/oulu_operational.parquet")
        sub.add_argument("--output", default="results/operational-chronological")
        sub.add_argument("--jobs", type=int, default=2)
        if command == "verify":
            sub.add_argument("--cohort", default="data/oulu_features.parquet")
            sub.add_argument("--archive", help="Also verify every original monthly source hash")
        if command == "train":
            sub.add_argument("--feature-profile", choices=["base", "context"], default="context")
            sub.add_argument("--workers", type=int, default=3)
        sub.set_defaults(function=function)
    args = parser.parse_args()
    if getattr(args, "jobs", 1) < 1 or getattr(args, "workers", 1) < 1:
        parser.error("jobs and workers must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
