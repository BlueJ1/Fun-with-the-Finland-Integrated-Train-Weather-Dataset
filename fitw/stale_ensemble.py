"""Native model and old-snapshot helpers for the final conditional policy.

Optional backends load only when selected. Prediction guards never read labels.
"""

import importlib
import itertools
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from . import stale, stale_enriched
from .prepare import sha256
from .spec import TARGET

BASE_PROVENANCE = ["source_actual_time", "input_latest_actual_time",
                   "input_latest_timetable_time", "focal_timetable_available_time"]
POSTPROCESS_CONTROLS = ["history_count", "is_departure", "oldest_pending_minutes"]
QUERY_IDENTITY = ["prediction_time", "observation_cutoff", "query_train_number", "type", "departureDate"]


def apply_rule(frame, prediction, rule):
    """Apply the frozen cap and floor once, after blending raw predictions."""
    result = np.asarray(prediction, dtype=np.float64).copy()
    cap, threshold = rule["nohistory_departure_cap"], rule["pending_floor_threshold"]
    if cap is not None:
        if not np.isfinite(cap):
            raise ValueError("Departure cap must be finite or null")
        mask = (frame.history_count.to_numpy() <= 0) & (frame.is_departure.to_numpy() == 1)
        result[mask] = np.minimum(result[mask], cap)
    if threshold is not None:
        multiplier = rule["pending_floor_multiplier"]
        if not np.isfinite(threshold) or threshold < 0 or multiplier is None or not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Pending floor requires a finite nonnegative threshold and positive multiplier")
        pending = frame.oldest_pending_minutes.to_numpy()
        mask = (frame.history_count.to_numpy() > 0) & (pending > threshold)
        result[mask] = np.maximum(result[mask], multiplier * pending[mask])
    return result

def _optional_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        runtime = Path(__file__).resolve().parents[1] / "tmp" / f"{name}-runtime"
        if not runtime.is_dir():
            raise ImportError(f"Selected ensemble requires optional backend {name}") from error
        sys.path.insert(0, str(runtime))
        return importlib.import_module(name)

def _family(source):
    family = source["family"]
    if family not in {"xgboost", "catboost", "lightgbm", "ridge", "neural"}:
        raise ValueError(f"Unsupported model backend: {family}")
    return family

def model_matrix(frame, component):
    """Match the source backend's predictor encoding without target statistics."""
    names, family = component["features"], _family(component)
    result = frame[names].copy()
    categorical = component.get("categorical_features", [])
    for name in names:
        numeric = pd.to_numeric(result[name], errors="raise")
        if family == "catboost" and name in categorical:
            result[name] = numeric.map(lambda value: "missing" if pd.isna(value) else str(int(value)))
        elif family == "lightgbm" and name in categorical:
            result[name] = numeric.map(lambda value: None if pd.isna(value) else str(int(value))).astype("category")
        else:
            result[name] = numeric.astype(np.float32)
    return result

def _baseline(frame, component):
    if _family(component) == "xgboost":
        return stale.prediction_baseline(frame, component["configuration"])
    if component["configuration"]["mode"] == "pending_residual":
        return stale.prediction_baseline(frame, component["configuration"]).astype(np.float64)
    return frame.previous_delay.fillna(0).to_numpy(dtype=np.float64)

def _estimator(component, threads):
    family, parameters = _family(component), component["configuration"]["parameters"]
    if family == "xgboost":
        return stale.estimator(parameters, threads)
    if family == "catboost":
        settings = {"loss_function": "RMSE", **parameters}
        return _optional_module("catboost").CatBoostRegressor(
            **settings, cat_features=component["categorical_features"],
            thread_count=threads, random_seed=42, has_time=True, allow_writing_files=False, verbose=False)
    if family == "lightgbm":
        return _optional_module("lightgbm").LGBMRegressor(
            **parameters, objective="regression", n_jobs=threads, random_state=42, verbosity=-1)
    if family == "ridge":
        from .stale_ridge import estimator
        return estimator(parameters, component["features"], component["categorical_features"])
    if family == "neural":
        from .stale_neural import estimator
        return estimator(parameters, component["features"], component["categorical_features"])

def _fit(component, matrix, labels, indices, threads):
    model = _estimator(component, threads)
    training = matrix.iloc[indices].copy()
    if _family(component) == "lightgbm":
        for name in component["categorical_features"]:
            training[name] = training[name].cat.remove_unused_categories()
        model.fit(training, labels[indices], categorical_feature=component["categorical_features"])
        return model.booster_
    with importlib.import_module("threadpoolctl").threadpool_limits(limits=threads):
        model.fit(training, labels[indices])
    return model

def _extension(component):
    return {"xgboost": "ubj", "catboost": "cbm", "lightgbm": "txt",
            "ridge": "joblib", "neural": "joblib"}[_family(component)]

def _save_model(model, component, path):
    if _family(component) in {"ridge", "neural"}:
        importlib.import_module("joblib").dump(model, path)
    else:
        model.save_model(str(path))

def _load_model(component, path, threads):
    family = _family(component)
    if family in {"ridge", "neural"}:
        return importlib.import_module("joblib").load(path)
    if family == "lightgbm":
        return _optional_module("lightgbm").Booster(model_file=str(path))
    model = _estimator(component, threads)
    model.load_model(str(path))
    return model

def _component_prediction(model, matrix, frame, component, threads):
    if _family(component) == "lightgbm":
        prediction = model.predict(matrix, num_threads=threads)
    else:
        with importlib.import_module("threadpoolctl").threadpool_limits(limits=threads):
            prediction = model.predict(matrix)
    # XGBoost's float32 addition must precede conversion to the blend's float64.
    if component["configuration"]["mode"] != "direct":
        prediction = prediction + _baseline(frame, component)
    return np.asarray(prediction, dtype=np.float64)

def _weights(components):
    values = np.asarray([component["weight"] for component in components], dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any() or not np.isclose(values.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError("Blend weights must be positive, finite, and sum to one")
    if len({component["source"] for component in components}) != len(components):
        raise ValueError("Blend components must have unique source identities")

def _load_datasets(components, data_directory):
    frames = {}
    for component in components:
        filename = component["data_file"]
        if Path(filename).name != filename:
            raise ValueError("Artifact data files must be plain filenames")
        path = Path(data_directory) / filename
        if sha256(path) != component["data_sha256"]:
            raise ValueError(f"Development data hash mismatch: {filename}")
        if filename not in frames:
            frames[filename] = stale.load_data(path)
        frame = frames[filename]
        if len(frame) != stale.DEVELOPMENT_ROWS or set(frame.row_id) != set(range(stale.DEVELOPMENT_ROWS)):
            raise ValueError("Final ensemble must retain every frozen development row")
    reference = frames[components[0]["data_file"]]
    for frame in frames.values():
        for name in ["row_id", "prediction_time", "observation_cutoff", "event_time", TARGET]:
            np.testing.assert_array_equal(frame[name].to_numpy(), reference[name].to_numpy())
        _assert_matching_columns(reference, frame, POSTPROCESS_CONTROLS, "postprocess controls")
        for left, right in zip(stale.folds_for(reference, 10), stale.folds_for(frame, 10)):
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_array_equal(left[1], right[1])
    return frames, reference

def _assert_matching_columns(reference, frame, columns, purpose):
    if len(reference) != len(frame):
        raise ValueError(f"Snapshot profiles disagree on {purpose}: row count")
    for name in columns:
        if name not in reference or name not in frame:
            raise ValueError(f"Missing {purpose}: {name}")
        left, right = reference[name].reset_index(drop=True), frame[name].reset_index(drop=True)
        if name in {"prediction_time", "observation_cutoff"}:
            left, right = pd.to_datetime(left, utc=True), pd.to_datetime(right, utc=True)
        if not ((left == right) | (left.isna() & right.isna())).all():
            raise ValueError(f"Snapshot profiles disagree on {purpose}: {name}")

def snapshot_profile_conflicts(components, frames):
    """Record learned predictors whose values differ across aligned profiles."""
    names_by_file = {}
    for component in components:
        names_by_file.setdefault(component["data_file"], set()).update(component["features"])
    conflicts = []
    for left, right in itertools.combinations(sorted(names_by_file), 2):
        different = []
        for name in sorted(names_by_file[left] & names_by_file[right]):
            a = frames[left][name].reset_index(drop=True)
            b = frames[right][name].reset_index(drop=True)
            if not ((a == b) | (a.isna() & b.isna())).all():
                different.append(name)
        if different:
            conflicts.append({"data_files": [left, right], "features": different})
    return conflicts

def _assert_fold(frame, tr, va):
    if not (pd.to_datetime(frame.iloc[tr].event_time, utc=True).max() < pd.to_datetime(frame.iloc[va].observation_cutoff, utc=True).min()):
        raise ValueError("Training labels are newer than the validation snapshot")
    if not (pd.to_datetime(frame.iloc[tr].prediction_time, utc=True).max() < pd.to_datetime(frame.iloc[va].prediction_time, utc=True).min()):
        raise ValueError("Simultaneous or future predictions crossed a training boundary")

def validate_snapshot(frame, config, deployment_not_before):
    """Require all used predictors and old-input provenance without reading labels."""
    names = set().union(*(set(component["features"]) for component in config["components"]))
    for component in config["components"]:
        if component["configuration"]["mode"] != "direct":
            names.add("previous_delay")
        if component["configuration"]["mode"] == "pending_residual":
            names.update(["history_count","oldest_pending_minutes"])
    rule = config["selection"]["rule"]
    if rule["nohistory_departure_cap"] is not None:
        names.update(["history_count","is_departure"])
    if rule["pending_floor_threshold"] is not None:
        names.update(["history_count","oldest_pending_minutes"])
    provenance = set(BASE_PROVENANCE)
    for component in config["components"]:
        provenance.update(component.get("input_manifest", {}).get("additional_provenance_columns", []))
    if names & set(stale.SERVICE_FEATURES):
        provenance.add("service_latest_available_time")
    if names & set(stale_enriched.EXTRA_FEATURES):
        provenance.update(stale_enriched.PROVENANCE_COLUMNS)
    if any(name.startswith("network_") for name in names):
        provenance.update(["network_input_latest_actual_time", "network_input_latest_timetable_time"])
    required = names | provenance | {"prediction_time", "observation_cutoff", "prediction_cutoff", "metadata_known", "history_count"}
    missing = required-set(frame.columns)
    if missing:
        raise ValueError(f"Missing snapshot predictors or provenance: {sorted(missing)}")
    prediction, cutoff = [pd.to_datetime(frame[name], utc=True) for name in ["prediction_time", "observation_cutoff"]]
    if not ((prediction-cutoff)>=pd.Timedelta(minutes=30)).all():
        raise ValueError("Prediction inputs must be at least thirty minutes old")
    if not (pd.to_datetime(frame.prediction_cutoff, utc=True)==cutoff).all():
        raise ValueError("History cutoff disagrees with observation snapshot")
    if not (prediction>=pd.Timestamp(deployment_not_before)).all():
        raise ValueError("Final ensemble was unavailable at this prediction time")
    if "scheduledTime" not in frame:
        if frame.metadata_known.fillna(False).astype(bool).any():
            raise ValueError("Known query metadata requires scheduledTime")
    else:
        scheduled = pd.to_datetime(frame.scheduledTime, utc=True)
        if not (scheduled == prediction).all():
            raise ValueError("Prediction time must equal the scheduled query time")
    for component in config["components"]:
        stale.validate_additional_provenance(frame, component.get("input_manifest", {}), cutoff)
    for name in provenance:
        values = pd.to_datetime(frame[name], utc=True)
        present = values.notna()
        if not (values[present]<cutoff[present]).all():
            raise ValueError(f"Recent input violates old snapshot: {name}")
    if frame.loc[frame.history_count>0,"source_actual_time"].isna().any():
        raise ValueError("Observed history requires source timestamp provenance")
    if frame.loc[frame.history_count>0,"input_latest_actual_time"].isna().any():
        raise ValueError("Observed history requires latest-input timestamp provenance")
    if frame.metadata_known.isna().any():
        raise ValueError("Timetable availability must be explicit")
    if frame.loc[frame.metadata_known.astype(bool),"focal_timetable_available_time"].isna().any():
        raise ValueError("Known focal metadata requires its accepted timestamp")
    service_names = list(names & set(stale.SERVICE_FEATURES))
    if service_names:
        present = frame[service_names].notna().any(axis=1)
        if frame.loc[present,"service_latest_available_time"].isna().any():
            raise ValueError("Present service history requires timestamp provenance")
    if names & set(stale_enriched.EXTRA_FEATURES):
        if "last_completed_scheduled_age_minutes" in frame:
            present = frame.last_completed_scheduled_age_minutes.notna()
            if frame.loc[present,"enriched_input_latest_actual_time"].isna().any():
                raise ValueError("Completed-event features require enriched timestamp provenance")
        if "planned_total_events" in frame:
            present = frame.planned_total_events>0
            if frame.loc[present,"enriched_input_latest_timetable_time"].isna().any():
                raise ValueError("Route features require enriched timetable provenance")
    static_names = list(names & set(stale.FEATURES[:11]+stale.EXTRA_FEATURES[:5]))
    if frame.loc[~frame.metadata_known.astype(bool), static_names].notna().any().any():
        raise ValueError("Unavailable timetable predictors must remain masked")
