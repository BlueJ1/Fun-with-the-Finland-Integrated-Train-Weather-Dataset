"""Fit and deploy a frozen old-snapshot conditional policy on development only.

Inference accepts a filename-to-frame mapping plus a separate ``policy_context``
frame. The context is the exact profile used to select the gates. Gate values
never come from a learned component's potentially conflicting snapshot profile.
"""

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd

from . import stale, stale_ensemble as native
from .prepare import sha256
from .spec import TARGET


DEFAULT_EXPLORATION = "results/stale30-tenfold-policy-model-v4/exploration"
DEFAULT_OUTPUT = "results/stale30-tenfold-policy-model-v4"
CONTEXT_KEY = "policy_context"
RECIPE_FORMAT = "stale30-reproduction-v1"
CALIBRATION_PARAMETERS = ("initial_raw_bias", "negative_branch_scale", "auxiliary_weight",
                          "tail_multiplier", "departure_cap", "pending_multiplier")
RECIPE_TRIAL_FIELDS = {"seed", "event_type_mixtures", "fold_rmse", "mean_cv_rmse", "target_met"} | set(CALIBRATION_PARAMETERS)
FROZEN_FILES = {name: name for name in [
    "selection.json", "definition.json", "search.json", "catalog.json", "raw_component_oof.npz"]}

GATE_FEATURES = ["previous_delay", "oulu_context_arrival_180m_max_delay",
                 "origin_context_720m_max_last_delay", "origin_context_720m_max_projected_delay"]
LIMITATION = ("All policy, mixture, gate and affine choices reuse development CV labels. "
              "This is native reproduction, not independent validation. actualTime and archived "
              "acceptance are availability proxies; FMI publication/revision availability is uncertified.")


def _json(path):
    return json.loads(Path(path).read_text())


def _finite(value, name, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"Policy parameter must be finite: {name}")
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"Invalid policy parameter: {name}")
    return float(value)


def ordered_policy(record, rule, auxiliary_source, auxiliary_weight):
    """Build the supported ordered policy without arbitrary feature expressions."""
    if record["origin_variant"] != "broad":
        raise ValueError("Unsupported origin policy")
    return [
        {"gate": [], "transform": {"kind": "bias", "value": record["initial_raw_bias"]}},
        {"gate": [{"feature": "previous_delay", "operator": "<=", "threshold": 0.}],
         "transform": {"kind": "scale", "value": record["negative_branch_scale"]}},
        {"gate": [{"feature": "oulu_context_arrival_180m_max_delay", "operator": "<=", "threshold": 60.}],
         "transform": {"kind": "blend", "source": auxiliary_source, "weight": auxiliary_weight}},
        {"gate": [{"feature": "previous_delay", "operator": "<=", "threshold": 0.},
                  {"feature": "current_raw_prediction", "operator": ">", "threshold": record["tail_threshold"]}],
         "transform": {"kind": "scale", "value": record["tail_multiplier"]}},
        {"kind": "standard_rule", "rule": rule},
        {"kind": "origin_nohistory_departure_cap_exception", "actual_presence_required": True,
         "feature": "origin_context_720m_max_last_delay",
         "operator": ">", "threshold": 120., "floor_multiplier": None},
    ]


EVENT_TYPE_GATE = {"feature": "is_departure", "operator": "==", "threshold": 1.,
                   "other_and_missing": "arrival_default"}


def _validate_event_type_mixtures(selected):
    """Permit only two old-event-type experts and bind the fitted source union."""
    fields = {"event_type_mixtures", "event_type_mixture_gate"}
    if not fields <= set(selected):
        raise ValueError("Incomplete event-type mixture declaration")
    gate = selected["event_type_mixture_gate"]
    if not isinstance(gate, Mapping) or gate != EVENT_TYPE_GATE:
        raise ValueError("Unsupported event-type mixture gate")
    _finite(gate["threshold"], "event-type threshold")
    mixtures = selected["event_type_mixtures"]
    if not isinstance(mixtures, Mapping) or set(mixtures) != {"arrival_default", "departure"}:
        raise ValueError("Malformed event-type expert inventory")
    average = {}
    for components in mixtures.values():
        native._weights(components)
        for component in components:
            if set(component) != {"source", "weight"} or not isinstance(component["source"], str):
                raise ValueError("Malformed event-type expert source")
            weight = _finite(component["weight"], "event-type weight", positive=True)
            average[component["source"]] = average.get(component["source"], 0.) + weight / 2
    fitted = {component["source"]: component["weight"] for component in selected["components"]}
    if set(fitted) != set(average) or any(not np.isclose(fitted[name], average[name], rtol=0, atol=1e-12)
                                        for name in average):
        raise ValueError("Fitted source union differs from the event-type experts")


def validate_selection(selected):
    required = {"components", "origin_variant", "initial_raw_bias", "negative_branch_scale", "tail_threshold",
                "tail_multiplier", "auxiliary_source", "auxiliary_weight", "rule", "ordered_policy", "holdout_used",
                "event_type_mixtures", "event_type_mixture_gate"}
    if not isinstance(selected, Mapping) or required-set(selected):
        raise ValueError("Incomplete policy selection")
    native._weights(selected["components"])
    _validate_event_type_mixtures(selected)
    for name in ["initial_raw_bias", "negative_branch_scale", "tail_threshold", "tail_multiplier", "auxiliary_weight"]:
        _finite(selected[name], name, positive=name in {"negative_branch_scale", "tail_multiplier"},
                nonnegative=name == "tail_threshold")
    if not 0 < selected["auxiliary_weight"] < 1:
        raise ValueError("Auxiliary weight must lie strictly between zero and one")
    rule = selected["rule"]
    if set(rule) != {"nohistory_departure_cap", "pending_floor_threshold", "pending_floor_multiplier"}:
        raise ValueError("Malformed standard rule")
    for name, value in rule.items():
        _finite(value, name, positive=name == "pending_floor_multiplier", nonnegative=True)
    expected = ordered_policy(selected, rule, selected["auxiliary_source"], selected["auxiliary_weight"])
    if selected.get("ordered_policy") != expected:
        raise ValueError("Unsupported, malformed or reordered policy gates")
    # Only this fixed expression grammar is supported. Identity and focal outcome
    # features cannot be inserted, even if a caller supplies them in its frame.
    if selected.get("holdout_used") is not False:
        raise ValueError("Policy must be development-only")


def _numeric(frame, name, allow_missing=True):
    if name not in frame:
        raise ValueError(f"Missing policy feature: {name}")
    values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
    if np.isinf(values).any() or (not allow_missing and not np.isfinite(values).all()):
        raise ValueError(f"Invalid policy feature: {name}")
    return values


def _count(frame, name):
    values = _numeric(frame, name, False)
    if (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError(f"Provenance count must be a nonnegative integer: {name}")
    return values


def _same_run(frame):
    required = {"query_train_number", "departureDate", "source_train_number", "source_departure_date"}
    if required-set(frame):
        raise ValueError("Missing same-run source provenance")
    history = _count(frame, "history_count") > 0
    source_number = pd.to_numeric(frame.source_train_number, errors="raise")
    query_number = pd.to_numeric(frame.query_train_number, errors="raise")
    source_date = pd.to_datetime(frame.source_departure_date, utc=True).dt.date
    query_date = pd.to_datetime(frame.departureDate, utc=True).dt.date
    if not (source_number[history] == query_number[history]).all() or not (source_date[history] == query_date[history]).all():
        raise ValueError("Observed history belongs to a different dated train run")


def _scheduled_query(frame):
    if "scheduledTime" not in frame:
        if frame.metadata_known.astype(bool).any():
            raise ValueError("Known query metadata requires scheduledTime")
        return
    scheduled, prediction = [pd.to_datetime(frame[name], utc=True) for name in ["scheduledTime", "prediction_time"]]
    if not (scheduled == prediction).all():
        raise ValueError("Prediction time must equal the scheduled query time")


def apply_policy(context, predictions, selected, return_stages=False):
    """Apply bias, old-context gates, tail scale, cap/floor, then cap exception."""
    validate_selection(selected)
    needed = {part["source"] for part in selected["components"]} | {selected["auxiliary_source"]}
    if needed-set(predictions):
        raise ValueError("Missing policy component predictions")
    arrays = {name: np.asarray(predictions[name], dtype=float) for name in needed}
    if any(value.shape != (len(context),) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Policy component predictions must be finite aligned vectors")
    previous = _numeric(context, "previous_delay")
    arrival = _numeric(context, "oulu_context_arrival_180m_max_delay")
    history = _count(context, "history_count") > 0
    # Missing focal metadata masks is_departure. Such a row satisfies no
    # departure gate, just as in the frozen exploration.
    departure_values = _numeric(context, "is_departure")
    if not np.isin(departure_values[np.isfinite(departure_values)], [0., 1.]).all():
        raise ValueError("is_departure must be zero, one or masked missing metadata")
    departures = departure_values == 1
    _numeric(context, "oldest_pending_minutes", False)
    observed = _count(context, "origin_context_actual_presence_count") > 0
    origin_step = selected["ordered_policy"][-1]
    origin = _numeric(context, origin_step["feature"])
    mixture = selected["event_type_mixtures"]
    arrival_raw = sum(part["weight"]*arrays[part["source"]] for part in mixture["arrival_default"])
    departure_raw = sum(part["weight"]*arrays[part["source"]] for part in mixture["departure"])
    raw = np.where(departures, departure_raw, arrival_raw)
    transformed = raw + selected["initial_raw_bias"]
    negative = np.isfinite(previous) & (previous <= 0)
    transformed[negative] *= selected["negative_branch_scale"]
    calm = np.isfinite(arrival) & (arrival <= 60.)
    weight = selected["auxiliary_weight"]
    transformed[calm] = (1-weight)*transformed[calm] + weight*arrays[selected["auxiliary_source"]][calm]
    tail = negative & (transformed > selected["tail_threshold"])
    transformed[tail] *= selected["tail_multiplier"]
    final = native.apply_rule(context, transformed, selected["rule"])
    exception = ~history & departures & observed & np.isfinite(origin) & (origin > origin_step["threshold"])
    final[exception] = transformed[exception]
    if not np.isfinite(final).all():
        raise ValueError("Policy returned nonfinite predictions")
    return (raw, transformed, final) if return_stages else final


def _context_component(config):
    context = config["context"]
    return {"features": context["input_manifest"]["features"], "input_manifest": context["input_manifest"],
            "configuration": {"mode": "direct"}, "data_file": context["data_file"]}


def validate_context(frame, config, deployment_not_before):
    native.validate_snapshot(frame, {"components": [_context_component(config)], "selection": config["selection"]},
                             deployment_not_before)
    _same_run(frame)
    _scheduled_query(frame)
    for names in config["context"]["input_manifest"].get("provenance_count_columns", {}).values():
        for name in names:
            _count(frame, name)
    for name in GATE_FEATURES:
        _numeric(frame, name)


def _all_components(config):
    return config["components"] + [config["auxiliary"]]


def _backend_versions(components):
    versions = {}
    for family in {native._family(component) for component in components}:
        package = "sklearn" if family in {"ridge", "neural"} else family
        versions[package] = native._optional_module(package).__version__
    return versions


def _context_data(config, frames, reference, data_directory):
    context = config["context"]
    path = Path(data_directory)/context["data_file"]
    if sha256(path) != context["data_sha256"] or sha256(path.with_suffix(".manifest.json")) != context["manifest_sha256"]:
        raise ValueError("Policy context data or manifest hash mismatch")
    if _json(path.with_suffix(".manifest.json")) != context["input_manifest"]:
        raise ValueError("Policy context manifest differs")
    frame = frames.get(context["data_file"])
    if frame is None:
        frame = stale.load_data(path)
    for name in ["row_id", "prediction_time", "observation_cutoff", "event_time", TARGET]:
        np.testing.assert_array_equal(frame[name], reference[name])
    native._assert_matching_columns(reference, frame, native.POSTPROCESS_CONTROLS, "policy controls")
    return frame


def replay_frozen(directory, context):
    """Recompute the final selection's three frozen trials from its own OOFs."""
    frozen = Path(directory)
    selected, definition = [_json(frozen/name) for name in ["selection.json", "definition.json"]]
    validate_selection(selected)
    if definition.get("format") != "bounded-event-experts-v1":
        raise ValueError("Unsupported frozen policy search format")
    if sha256(frozen/"catalog.json") != definition["frozen_sha256"]["config.json"]:
        raise ValueError("Frozen source catalog hash mismatch")
    if sha256(frozen/"raw_component_oof.npz") != definition["frozen_sha256"]["raw_component_oof.npz"]:
        raise ValueError("Frozen source OOF hash mismatch")
    expected = {part["source"] for part in selected["components"]} | {selected["auxiliary_source"]}
    catalog = _json(frozen/"catalog.json")
    if set(catalog["sources"]) != expected:
        raise ValueError("Frozen source catalog must contain only the final source inventory")
    with np.load(frozen/"raw_component_oof.npz", allow_pickle=False) as saved:
        ids, folds = saved["row_ids"].copy(), saved["fold_ids"].copy()
        names, matrix = saved["source_names"].tolist(), saved["predictions"].copy()
    if len(names) != len(set(names)) or set(names) != expected or matrix.shape != (len(names), len(ids)) or not np.isfinite(matrix).all():
        raise ValueError("Malformed or unused frozen source prediction matrix")
    splits = stale.folds_for(context, 10)
    np.testing.assert_array_equal(ids, np.concatenate([context.iloc[va].row_id for _, va in splits]))
    np.testing.assert_array_equal(folds, np.concatenate([np.full(len(va), i) for i, (_, va) in enumerate(splits, 1)]))
    for tr, va in splits:
        native._assert_fold(context, tr, va)
    lookup = context.set_index("row_id", drop=False).loc[ids]
    predictions = dict(zip(names, matrix))
    y = lookup[TARGET].to_numpy(dtype=float)
    scores = lambda values: [float(np.sqrt(np.mean((values[folds == i]-y[folds == i])**2))) for i in range(1,11)]
    records = _json(frozen/"search.json")
    return _replay_bounded_calibration(selected, definition, records, lookup, predictions, ids, folds, scores)


def _validate_calibration_records(selected, definition, records, source_names):
    """Check the frozen grammar and inventory before any fresh fitting."""
    required_bounds = {"initial_raw_bias": [-5., 5.], "negative_branch_scale": [.8, 1.5],
                       "auxiliary_weight": [.05, .5], "tail_multiplier": [1., 2.],
                       "departure_cap": [15., 60.], "pending_multiplier": [1., 3.]}
    fixed = {"origin_variant": "broad", "tail_threshold": 240., "auxiliary_source": "fmilight:1",
             "pending_floor_threshold": 360.}
    if definition.get("bounds") != required_bounds or definition.get("fixed_policy") != fixed:
        raise ValueError("Unsupported calibration bounds or fixed policy")
    if definition.get("format") != "bounded-event-experts-v1" or definition.get("event_type_mixture_gate") != EVENT_TYPE_GATE:
        raise ValueError("Frozen event-type gate or format differs")
    _finite(definition["event_type_mixture_gate"]["threshold"], "frozen event-type threshold")
    count = definition.get("trial_count")
    if isinstance(count, bool) or count != 3 or selected.get("trial_count") != count or len(records) != count:
        raise ValueError("Incomplete bounded calibration inventory")
    if {r.get("seed") for r in records} != {1, 2, 3}:
        raise ValueError("Incomplete or duplicate calibration starts")
    if definition.get("context_data_sha256") != selected.get("context_data_sha256"):
        raise ValueError("Calibration context differs from the frozen selection")
    for record in records:
        validate_selection(record)
        if {part["source"] for part in record["components"]} | {record["auxiliary_source"]} != set(source_names):
            raise ValueError("Frozen trial source inventory differs from the final model")
        if (record["origin_variant"] != fixed["origin_variant"] or
                record["tail_threshold"] != fixed["tail_threshold"] or
                record["auxiliary_source"] != fixed["auxiliary_source"] or
                record["rule"]["pending_floor_threshold"] != fixed["pending_floor_threshold"]):
            raise ValueError("Calibration changed a fixed gate")
        for name, (lower, upper) in required_bounds.items():
            value = _finite(record.get(name), name)
            if not lower <= value <= upper:
                raise ValueError("Calibration parameter outside frozen bounds")
        if (record["rule"]["nohistory_departure_cap"] != record["departure_cap"] or
                record["rule"]["pending_floor_multiplier"] != record["pending_multiplier"]):
            raise ValueError("Calibration rule differs from its parameter record")
        if any(record.get(name) != selected.get(name) for name in
               ["context_data_path", "context_data_sha256", "context_input_manifest"]):
            raise ValueError("Calibration trial changed its context snapshot")
        losses = np.asarray(record["fold_rmse"], dtype=float)
        if losses.shape != (10,):
            raise ValueError("Calibration must retain exactly ten fold scores")
        for value in record["fold_rmse"]:
            _finite(value, "fold RMSE", nonnegative=True)
        _finite(record["mean_cv_rmse"], "mean CV RMSE", nonnegative=True)
        np.testing.assert_allclose(np.mean(losses), record["mean_cv_rmse"], rtol=0, atol=1e-10)
        if record.get("target_met") is not bool(np.mean(losses) <= 8.):
            raise ValueError("Calibration target flag differs from its scores")
    if selected != min(records, key=lambda record: record["mean_cv_rmse"]):
        raise ValueError("Selected calibration is not the complete frozen minimum")


def _replay_bounded_calibration(selected, definition, records, context, predictions, ids, folds, scores):
    """Recompute every declared trial after structural preflight."""
    _validate_calibration_records(selected, definition, records, predictions)
    for record in records:
        losses = scores(apply_policy(context, predictions, record))
        np.testing.assert_allclose(losses, record["fold_rmse"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(np.mean(losses), record["mean_cv_rmse"], rtol=0, atol=1e-10)
        if record.get("target_met") != (np.mean(losses) <= 8.):
            raise ValueError("Calibration target flag differs from replay")
    return selected, predictions, ids, folds, len(records)


def _load_recipe(path, data_directory):
    """Hydrate a small reproduction recipe using hash-bound prepared inputs."""
    recipe = _json(path)
    if not isinstance(recipe, Mapping) or set(recipe) != {"format", "selection", "sources", "definition", "search"} or recipe["format"] != RECIPE_FORMAT:
        raise ValueError("Unsupported reproduction recipe format or inventory")
    selected, definition = dict(recipe["selection"]), dict(recipe["definition"])
    validate_selection(selected)
    if ("context_input_manifest" in selected or "frozen_sha256" in definition
            or any(name.startswith("original_") for name in definition)):
        raise ValueError("Compact recipe must omit generated manifests and frozen cache hashes")
    if definition.get("folds") != 10 or definition.get("holdout_used") is not False:
        raise ValueError("Reproduction recipes must retain ten development-only folds")
    context_path = Path(data_directory)/Path(selected["context_data_path"]).name
    manifest_path = context_path.with_suffix(".manifest.json")
    if (selected.get("context_input_manifest_sha256") != definition.get("input_manifest_sha256")
            or sha256(manifest_path) != selected.get("context_input_manifest_sha256")
            or sha256(context_path) != selected["context_data_sha256"]):
        raise ValueError("Recipe context data or manifest hash mismatch")
    selected["context_input_manifest"] = _json(manifest_path)
    expected = {part["source"] for part in selected["components"]} | {selected["auxiliary_source"]}
    sources = recipe["sources"]
    if not isinstance(sources, Mapping) or set(sources) != expected:
        raise ValueError("Recipe source inventory differs from the final model")
    required = {"family", "configuration", "categorical_features", "data_path", "data_sha256", "cv_sha256", "input_manifest_sha256"}
    for source in sources.values():
        if not isinstance(source, Mapping) or required-set(source) or set(source)-required-{"features"}:
            raise ValueError("Malformed reproduction source contract")
        source_path = Path(data_directory)/Path(source["data_path"]).name
        if (sha256(source_path) != source["data_sha256"]
                or sha256(source_path.with_suffix(".manifest.json")) != source["input_manifest_sha256"]):
            raise ValueError("Recipe source data or manifest hash mismatch")
    records = []
    for trial in recipe["search"]:
        if not isinstance(trial, Mapping) or set(trial) != RECIPE_TRIAL_FIELDS:
            raise ValueError("Malformed compact calibration trial")
        record = dict(selected, **trial)
        mixtures = record["event_type_mixtures"]
        if not isinstance(mixtures, Mapping) or set(mixtures) != {"arrival_default", "departure"}:
            raise ValueError("Malformed compact expert inventory")
        average = {}
        for components in mixtures.values():
            native._weights(components)
            for component in components:
                if set(component) != {"source", "weight"} or not isinstance(component["source"], str):
                    raise ValueError("Malformed compact expert source")
                weight = _finite(component["weight"], "compact expert weight", positive=True)
                average[component["source"]] = average.get(component["source"], 0.) + weight/2
        if set(average) != {part["source"] for part in selected["components"]}:
            raise ValueError("Compact trial source union differs from the selected model")
        record["components"] = [{"source": part["source"], "weight": average[part["source"]]} for part in selected["components"]]
        record["rule"] = dict(selected["rule"], nohistory_departure_cap=trial["departure_cap"], pending_floor_multiplier=trial["pending_multiplier"])
        record["ordered_policy"] = ordered_policy(record, record["rule"], record["auxiliary_source"], record["auxiliary_weight"])
        records.append(record)
    _validate_calibration_records(selected, definition, records, sources)
    return selected, {"sources": dict(sources)}, definition, records


def _components(catalog, selected, data_directory):
    components = []
    for number, blend in enumerate(selected["components"] + [{"source": selected["auxiliary_source"], "weight": 1.}], 1):
        source = catalog["sources"][blend["source"]]
        filename = Path(source["data_path"]).name
        manifest = _json((Path(data_directory)/filename).with_suffix(".manifest.json"))
        names = source.get("features", manifest["features"])
        if not set(names) <= set(manifest["features"]) or set(names) & {TARGET,"actualTime","event_time","row_id"}:
            raise ValueError("Unsafe or absent source predictors")
        component = {"source": blend["source"], "weight": blend["weight"], "family": source["family"],
                     "configuration": source["configuration"], "features": names,
                     "categorical_features": [name for name in source.get("categorical_features", []) if name in names],
                     "data_file": filename, "data_sha256": source["data_sha256"], "input_manifest": manifest,
                     "source_cv_sha256": source["cv_sha256"],
                     "model_directory": f"component_{number}"}
        extension = native._extension(component)
        component["model_file"] = f"component_{number}/model.{extension}"
        component["fold_model_files"] = [f"component_{number}/fold_{i}.{extension}" for i in range(1,11)]
        components.append(component)
    if len({part["source"] for part in components}) != len(components) and "event_type_mixtures" not in selected:
        raise ValueError("Auxiliary source must be distinct from the main mixture")
    return components


def train(args):
    output = Path(args.output)
    recipe_path = getattr(args, "recipe", None)
    recipe_records = None
    if recipe_path:
        selected, catalog, recipe_definition, recipe_records = _load_recipe(recipe_path, args.data_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh empty policy artifact directory")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output/"exploration"
    frozen.mkdir()
    if recipe_path:
        for name, value in [("selection.json",selected),("catalog.json",catalog),
                            ("definition.json",recipe_definition),("search.json",recipe_records)]:
            (frozen/name).write_text(json.dumps(value, indent=2))
        shutil.copy2(recipe_path, output/"reproduction_recipe.json")
    else:
        for target, source in FROZEN_FILES.items():
            shutil.copy2(Path(args.exploration)/source, frozen/target)
        selected, catalog = [_json(frozen/name) for name in ["selection.json", "catalog.json"]]
    validate_selection(selected)
    parts = _components(catalog, selected, args.data_directory)
    context_path = Path(args.data_directory)/Path(selected["context_data_path"]).name
    context_manifest = _json(context_path.with_suffix(".manifest.json"))
    if context_manifest != selected["context_input_manifest"] or sha256(context_path) != selected["context_data_sha256"]:
        raise ValueError("Selected gate context does not match its frozen snapshot")
    config = {"model": "stale-30-minute-conditional-policy", "components": parts[:-1], "auxiliary": parts[-1],
              "context": {"profile_key": CONTEXT_KEY, "data_file": context_path.name,
                          "data_sha256": sha256(context_path), "manifest_sha256": sha256(context_path.with_suffix(".manifest.json")),
                          "input_manifest": context_manifest},
              "selection": selected, "folds": 10, "target_rmse": 8., "threads": args.threads,
              "minimum_input_age_minutes": 30., "holdout_used": False, "limitation": LIMITATION,
              "frozen_sha256": {name: sha256(frozen/name) for name in FROZEN_FILES if (frozen/name).exists()},
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    if recipe_path:
        config["reproduction_recipe_sha256"] = sha256(output/"reproduction_recipe.json")
    source_manifests = frozen/"source_manifests"
    source_manifests.mkdir()
    for number, component in enumerate(parts, 1):
        relative = f"source_manifests/component_{number}.json"
        manifest_path = (Path(args.data_directory)/component["data_file"]).with_suffix(".manifest.json")
        shutil.copy2(manifest_path, frozen/relative)
        config["frozen_sha256"][relative] = sha256(frozen/relative)
        component["manifest_file"] = relative
    frames, reference = native._load_datasets(parts, args.data_directory)
    context = _context_data(config, frames, reference, args.data_directory)
    if recipe_path:
        folds = stale.folds_for(context, 10)
        for tr, va in folds:
            native._assert_fold(context, tr, va)
        ids = np.concatenate([context.iloc[va].row_id.to_numpy() for _,va in folds])
        fold_ids = np.concatenate([np.full(len(va),i) for i,(_,va) in enumerate(folds,1)])
        cached = None
    else:
        selected, cached, ids, fold_ids, trials = replay_frozen(frozen, context)
    conflict_frames = dict(frames, **{context_path.name: context})
    config["snapshot_profile_conflicts"] = native.snapshot_profile_conflicts(parts+[_context_component(config)], conflict_frames)
    config["versions"].update(_backend_versions(parts))
    # Check every backend, source matrix and source hash before the first fit.
    matrices = {}
    for component in parts:
        native._estimator(component, args.threads)
        matrices[component["source"]] = native.model_matrix(frames[component["data_file"]], component)
    (output/"config.json").write_text(json.dumps(config, indent=2))
    raw_predictions, model_hashes, fitted_sources = {}, {}, {}
    folds = stale.folds_for(reference, 10)
    for component in parts:
        frame = frames[component["data_file"]]
        matrix = matrices[component["source"]]
        y = frame[TARGET].to_numpy(dtype=np.float32 if native._family(component) == "xgboost" else np.float64)
        labels = y-native._baseline(frame,component) if component["configuration"]["mode"] != "direct" else y
        (output/component["model_directory"]).mkdir()
        if component["source"] in fitted_sources:
            original = fitted_sources[component["source"]]
            binding_keys = ["family", "configuration", "features", "categorical_features", "data_file",
                            "data_sha256", "input_manifest", "source_cv_sha256"]
            if any(component[key] != original[key] for key in binding_keys):
                raise ValueError("Repeated auxiliary role has a different source binding")
            paths = [component["model_file"]] + component["fold_model_files"]
            original_paths = [original["model_file"]] + original["fold_model_files"]
            for relative, original_relative in zip(paths, original_paths):
                shutil.copy2(output/original_relative, output/relative)
                model_hashes[relative] = sha256(output/relative)
                if model_hashes[relative] != model_hashes[original_relative]:
                    raise ValueError("Reused auxiliary model copy differs")
            print(json.dumps({"component":component["source"],"auxiliary_native_model_reused":True}),flush=True)
            continue
        predictions = []
        for i, (tr, va) in enumerate(folds, 1):
            native._assert_fold(frame, tr, va)
            model = native._fit(component, matrix, labels, tr, args.threads)
            relative = component["fold_model_files"][i-1]
            native._save_model(model, component, output/relative)
            model_hashes[relative] = sha256(output/relative)
            predictions.append(native._component_prediction(model, matrix.iloc[va], frame.iloc[va], component, args.threads))
            print(json.dumps({"component":component["source"],"fold":i,"fresh_model_fitted":True}),flush=True)
        raw_predictions[component["source"]] = np.concatenate(predictions)
        if cached is not None:
            np.testing.assert_allclose(raw_predictions[component["source"]], cached[component["source"]], rtol=0, atol=1e-10)
        model = native._fit(component, matrix, labels, np.arange(len(frame)), args.threads)
        native._save_model(model, component, output/component["model_file"])
        model_hashes[component["model_file"]] = sha256(output/component["model_file"])
        fitted_sources[component["source"]] = component
    if recipe_path:
        names = list(catalog["sources"])
        np.savez_compressed(frozen/"raw_component_oof.npz", source_names=np.asarray(names), row_ids=ids,
                            fold_ids=fold_ids, predictions=np.stack([raw_predictions[name] for name in names]))
        recipe_definition["frozen_sha256"] = {"config.json":sha256(frozen/"catalog.json"),
                                                "raw_component_oof.npz":sha256(frozen/"raw_component_oof.npz")}
        (frozen/"definition.json").write_text(json.dumps(recipe_definition, indent=2))
        selected, cached, ids, fold_ids, trials = replay_frozen(frozen, context)
        config["frozen_sha256"].update({name:sha256(frozen/name) for name in FROZEN_FILES})
        (output/"config.json").write_text(json.dumps(config, indent=2))
    lookup = context.set_index("row_id", drop=False).loc[ids]
    raw, transformed, prediction = apply_policy(lookup, raw_predictions, selected, True)
    cv = lookup[["row_id","event_time","prediction_time","observation_cutoff",TARGET]].copy()
    cv["fold"], cv["raw_prediction"], cv["transformed_raw_prediction"], cv["prediction"] = fold_ids, raw, transformed, prediction
    cv.to_parquet(output/"cv_predictions.parquet", index=False)
    losses = [float(np.sqrt(np.mean((part.prediction-part[TARGET])**2))) for _,part in cv.groupby("fold",sort=True)]
    np.testing.assert_allclose(losses, selected["fold_rmse"], rtol=0, atol=1e-10)
    result = {"model":config["model"], "selection":selected, "mean_cv_rmse":float(np.mean(losses)), "fold_rmse":losses,
              "target_rmse":8.,"target_met":bool(np.mean(losses)<=8.),"folds":10,"development_rows":len(context),
              "minimum_input_age_minutes":30.,"holdout_used":False,"model_sha256":model_hashes,
              "config_sha256":sha256(output/"config.json"),"cv_predictions_sha256":sha256(output/"cv_predictions.parquet"),
              "deployment_not_before":str(pd.to_datetime(context.event_time,utc=True).max()+pd.Timedelta(minutes=30)),
              "frozen_affine_scores_replayed":trials,"limitation":LIMITATION}
    result.update(unique_native_sources=len(fitted_sources), fresh_fold_fit_count=10*len(fitted_sources),
                  saved_model_roles=len(parts), auxiliary_model_reused=len(fitted_sources)<len(parts))
    (output/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {"selection","model_sha256"}},indent=2),flush=True)


def _artifact(directory):
    output = Path(directory)
    config, result = [_json(output/name) for name in ["config.json","result.json"]]
    if result["config_sha256"] != sha256(output/"config.json") or config["selection"] != result["selection"]:
        raise ValueError("Policy configuration hash or selection mismatch")
    validate_selection(config["selection"])
    if "reproduction_recipe_sha256" in config and sha256(output/"reproduction_recipe.json") != config["reproduction_recipe_sha256"]:
        raise ValueError("Reproduction recipe hash mismatch")
    frozen = output/"exploration"
    if config["selection"] != _json(frozen/"selection.json"):
        raise ValueError("Policy differs from its frozen selection")
    if any(record.get("holdout_used") is not False or record["folds"] != 10 or record["target_rmse"] != 8. for record in [config,result]):
        raise ValueError("Policy artifact must retain the ten-fold development-only target")
    if not np.isfinite(result["mean_cv_rmse"]) or result["target_met"] != (result["mean_cv_rmse"]<=8.):
        raise ValueError("Policy target flag differs from its score")
    expected = {path for part in _all_components(config) for path in [part["model_file"]]+part["fold_model_files"]}
    if expected != set(result["model_sha256"]) or any(len(part["fold_model_files"]) != 10 for part in _all_components(config)):
        raise ValueError("Policy native model inventory is incomplete")
    for name,digest in result["model_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(output/name) != digest:
            raise ValueError("Policy native model path or hash mismatch")
    manifest_files = {part["manifest_file"] for part in _all_components(config)}
    if set(config["frozen_sha256"]) != set(FROZEN_FILES) | manifest_files:
        raise ValueError("Frozen policy inventory is incomplete")
    for name,digest in config["frozen_sha256"].items():
        if sha256(output/"exploration"/name) != digest:
            raise ValueError("Frozen policy exploration hash mismatch")
    selected, catalog = config["selection"], _json(frozen/"catalog.json")
    recipes = selected["components"] + [{"source": selected["auxiliary_source"], "weight": 1.}]
    parts = _all_components(config)
    if len(parts) != len(recipes):
        raise ValueError("Native component inventory differs from the frozen policy")
    for number, (component, recipe) in enumerate(zip(parts, recipes), 1):
        source = catalog["sources"][recipe["source"]]
        names = source.get("features", component["input_manifest"]["features"])
        expected = {"source":recipe["source"], "weight":recipe["weight"], "family":source["family"],
                    "configuration":source["configuration"], "features":names,
                    "categorical_features":[name for name in source.get("categorical_features",[]) if name in names],
                    "data_file":Path(source["data_path"]).name, "data_sha256":source["data_sha256"],
                    "source_cv_sha256":source["cv_sha256"]}
        extension = native._extension({"family": source["family"]})
        expected.update({"manifest_file": f"source_manifests/component_{number}.json",
                         "model_directory": f"component_{number}",
                         "model_file": f"component_{number}/model.{extension}",
                         "fold_model_files": [f"component_{number}/fold_{i}.{extension}" for i in range(1,11)]})
        if any(component.get(name) != value for name,value in expected.items()):
            raise ValueError("Native component differs from its frozen source catalog")
        manifest_file = Path(component["manifest_file"])
        if manifest_file.is_absolute() or ".." in manifest_file.parts or _json(frozen/manifest_file) != component["input_manifest"]:
            raise ValueError("Native component differs from its frozen source manifest")
        if "input_manifest_sha256" in source and sha256(frozen/manifest_file) != source["input_manifest_sha256"]:
            raise ValueError("Native source manifest differs from the reproduction recipe")
    context = config["context"]
    definition = _json(frozen/"definition.json")
    if (context["profile_key"] != CONTEXT_KEY or context["data_file"] != Path(selected["context_data_path"]).name
            or context["input_manifest"] != selected["context_input_manifest"]
            or context["data_sha256"] != selected["context_data_sha256"]
            or context["manifest_sha256"] != definition["input_manifest_sha256"]):
        raise ValueError("Policy context differs from its frozen snapshot")
    return output,config,result


def _inference_frames(features, config):
    parts = _all_components(config)
    if isinstance(features, Mapping):
        needed = {part["data_file"] for part in parts} | {CONTEXT_KEY}
        if needed-set(features):
            raise ValueError(f"Missing snapshot profiles or policy context: {sorted(needed-set(features))}")
        frames = {name:features[name].copy() for name in needed}
        reference = frames[CONTEXT_KEY]
        for frame in frames.values():
            for name in native.QUERY_IDENTITY:
                if name not in frame or frame[name].isna().any():
                    raise ValueError(f"Missing query identity: {name}")
            native._assert_matching_columns(reference,frame,native.QUERY_IDENTITY,"query identity")
            native._assert_matching_columns(reference,frame,native.POSTPROCESS_CONTROLS,"policy controls")
    else:
        if config.get("snapshot_profile_conflicts"):
            raise ValueError("Conflicting snapshot profiles require a mapping with explicit policy_context")
        frames = {part["data_file"]:features.copy() for part in parts}
        frames[CONTEXT_KEY] = features.copy()
    for frame in frames.values():
        if "feature_train_number" in frame:
            frame["trainNumber"] = frame.feature_train_number
    return frames


def predict(features, model_directory=DEFAULT_OUTPUT):
    """Predict without labels using fresh saved models and explicit gate context."""
    output,config,result = _artifact(model_directory)
    frames = _inference_frames(features,config)
    validate_context(frames[CONTEXT_KEY],config,result["deployment_not_before"])
    predictions = {}
    for component in _all_components(config):
        frame = frames[component["data_file"]]
        native.validate_snapshot(frame,{"components":[component],"selection":config["selection"]},result["deployment_not_before"])
        _same_run(frame)
        _scheduled_query(frame)
        for names in component["input_manifest"].get("provenance_count_columns",{}).values():
            for name in names:
                _count(frame,name)
        model = native._load_model(component,output/component["model_file"],config["threads"])
        predictions[component["source"]] = native._component_prediction(model,native.model_matrix(frame,component),frame,component,config["threads"])
    return apply_policy(frames[CONTEXT_KEY],predictions,config["selection"])


def verify(args):
    output,config,result = _artifact(args.output)
    if sha256(output/"cv_predictions.parquet") != result["cv_predictions_sha256"]:
        raise ValueError("Policy CV prediction hash mismatch")
    parts = _all_components(config)
    frames,reference = native._load_datasets(parts,args.data_directory)
    context = _context_data(config,frames,reference,args.data_directory)
    conflicts = native.snapshot_profile_conflicts(parts+[_context_component(config)],dict(frames,**{config["context"]["data_file"]:context}))
    if conflicts != config["snapshot_profile_conflicts"]:
        raise ValueError("Saved snapshot profile conflicts differ")
    selected,cached,ids,fold_ids,trials = replay_frozen(output/"exploration",context)
    if selected != config["selection"]:
        raise ValueError("Native policy differs from the frozen selection")
    predictions = {}
    for component in parts:
        frame = frames[component["data_file"]]
        matrix = native.model_matrix(frame,component)
        values = []
        for i,(tr,va) in enumerate(stale.folds_for(frame,10),1):
            native._assert_fold(frame,tr,va)
            model = native._load_model(component,output/component["fold_model_files"][i-1],config["threads"])
            values.append(native._component_prediction(model,matrix.iloc[va],frame.iloc[va],component,config["threads"]))
        predictions[component["source"]] = np.concatenate(values)
        np.testing.assert_allclose(predictions[component["source"]],cached[component["source"]],rtol=0,atol=1e-10)
    lookup = context.set_index("row_id",drop=False).loc[ids]
    raw,transformed,prediction = apply_policy(lookup,predictions,selected,True)
    saved = pd.read_parquet(output/"cv_predictions.parquet")
    for name,value in [("row_id",ids),("fold",fold_ids),(TARGET,lookup[TARGET]),("raw_prediction",raw),
                       ("transformed_raw_prediction",transformed),("prediction",prediction)]:
        np.testing.assert_allclose(saved[name],value,rtol=0,atol=1e-10)
    losses = [float(np.sqrt(np.mean((prediction[fold_ids==i]-lookup[TARGET].to_numpy()[fold_ids==i])**2))) for i in range(1,11)]
    np.testing.assert_allclose(losses,result["fold_rmse"],rtol=0,atol=1e-10)
    np.testing.assert_allclose(np.mean(losses),result["mean_cv_rmse"],rtol=0,atol=1e-10)
    deployment = str(pd.to_datetime(context.event_time,utc=True).max()+pd.Timedelta(minutes=30))
    if deployment != result["deployment_not_before"] or result["development_rows"] != len(context):
        raise ValueError("Policy deployment cutoff or row count differs")
    probes = {name:frame.iloc[[-1]].drop(columns=[TARGET]).copy() for name,frame in frames.items()}
    probes[CONTEXT_KEY] = context.iloc[[-1]].drop(columns=[TARGET]).copy()
    for frame in probes.values():
        frame["prediction_time"] = pd.Timestamp(deployment)+pd.Timedelta(days=1)
        frame["scheduledTime"] = frame.prediction_time
        frame["observation_cutoff"] = frame.prediction_time-pd.Timedelta(minutes=30)
        frame["prediction_cutoff"] = frame.observation_cutoff
    if not np.isfinite(predict(probes,output)).all():
        raise ValueError("Native policy unlabeled inference failed")
    rejected_count = 0
    rejection_conditions = []
    def must_reject(altered, condition):
        try:
            predict(altered, output)
        except ValueError:
            rejection_conditions.append(condition)
        else:
            raise ValueError(f"Policy failed to reject {condition}")
    for key,frame in probes.items():
        manifests = [config["context"]["input_manifest"]] if key == CONTEXT_KEY else [part["input_manifest"] for part in parts if part["data_file"] == key]
        names = set(native.BASE_PROVENANCE) | set().union(*(set(m.get("additional_provenance_columns",[])) for m in manifests))
        names |= {name for name in ["service_latest_available_time","enriched_input_latest_actual_time","enriched_input_latest_timetable_time","network_input_latest_actual_time","network_input_latest_timetable_time"] if name in frame}
        for name in names:
            altered = {name:value.copy() for name,value in probes.items()}
            altered[key][name] = altered[key].observation_cutoff
            must_reject(altered, f"recent provenance: {key}/{name}")
            rejected_count += 1
        for manifest in manifests:
            for timestamp,count_names in manifest.get("provenance_count_columns", {}).items():
                altered = {name:value.copy() for name,value in probes.items()}
                altered[key][count_names[0]] = 1.
                altered[key][timestamp] = pd.NaT
                must_reject(altered, f"missing positive-count provenance: {key}/{timestamp}")
                altered = {name:value.copy() for name,value in probes.items()}
                altered[key][count_names[0]] = .5
                must_reject(altered, f"fractional provenance count: {key}/{count_names[0]}")
            for timestamp,lag in manifest.get("additional_provenance_lag_minutes", {}).items():
                altered = {name:value.copy() for name,value in probes.items()}
                altered[key][timestamp] = altered[key].observation_cutoff-pd.Timedelta(minutes=lag)
                must_reject(altered, f"availability lag equality: {key}/{timestamp}")
    altered = {name:value.copy() for name,value in probes.items()}
    for frame in altered.values():
        frame["observation_cutoff"] = frame.prediction_time-pd.Timedelta(minutes=29)
        frame["prediction_cutoff"] = frame.observation_cutoff
    must_reject(altered, "29-minute-old snapshot")
    altered = {name:value.copy() for name,value in probes.items()}
    for frame in altered.values():
        frame["prediction_time"] = pd.Timestamp(deployment)-pd.Timedelta(minutes=1)
        frame["scheduledTime"] = frame.prediction_time
        frame["observation_cutoff"] = frame.prediction_time-pd.Timedelta(minutes=30)
        frame["prediction_cutoff"] = frame.observation_cutoff
    must_reject(altered, "premature deployment")
    altered = {name:value.copy() for name,value in probes.items()}
    altered[CONTEXT_KEY]["scheduledTime"] -= pd.Timedelta(minutes=1)
    must_reject(altered, "scheduled query time mismatch")
    altered = {name:value.copy() for name,value in probes.items()}
    altered[CONTEXT_KEY] = altered[CONTEXT_KEY].drop(columns=["input_latest_actual_time"])
    must_reject(altered, "missing context base provenance")
    altered = {name:value.copy() for name,value in probes.items()}
    altered[CONTEXT_KEY]["query_train_number"] += 1
    must_reject(altered, "context query identity mismatch")
    must_reject({name:value for name,value in probes.items() if name != CONTEXT_KEY}, "missing explicit policy context")
    verification = {"passed":True,"mean_cv_rmse":float(np.mean(losses)),"fold_rmse":losses,"folds":10,"holdout_used":False,
                    "fresh_native_components":len(parts),"fold_models_reloaded":10*len(parts),"full_models_unlabeled_inference_checked":True,
                    "frozen_affine_scores_recomputed":trials,"cv_prediction_rows":len(saved),
                    "training_label_age_and_prediction_ties_checked":True,"all_frozen_data_model_hashes_checked":True,
                    "recent_source_and_context_provenance_rejections":rejected_count,"guard_rejections":rejection_conditions,
                    "prediction_tolerance":1e-10,"limitation":LIMITATION}
    verification["unique_native_sources"] = len({part["source"] for part in parts})
    (output/"verification.json").write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["train","verify"])
    parser.add_argument("--exploration",default=DEFAULT_EXPLORATION)
    parser.add_argument("--recipe",help="Compact reproduction recipe; train from prepared data without an OOF cache")
    parser.add_argument("--data-directory",default="data")
    parser.add_argument("--output",default=DEFAULT_OUTPUT)
    parser.add_argument("--threads",type=int,default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.recipe and args.command != "train":
        parser.error("--recipe is used only when training; verify the generated bundle with --output")
    {"train":train,"verify":verify}[args.command](args)


if __name__ == "__main__":
    main()
