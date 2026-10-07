"""Origin context with disjoint dated runs from the previous archive month.

Current-source route and observation revisions always take precedence. Historical
actualTime and retained timetable acceptance remain availability proxies.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import stale, stale_origin, stale_turnaround
from .prepare import sha256
from .spec import TARGET

RUN_KEY = ["departureDate", "trainNumber"]


def previous_source(name):
    """Resolve only the preceding calendar archive, never a later source."""
    prefix = "matched_data_flat_"
    if not name.startswith(prefix) or not name.endswith(".parquet"):
        raise ValueError(f"Unexpected monthly source name: {name}")
    month = pd.Timestamp(name[len(prefix):-len(".parquet")].replace("_", "-") + "-01", tz="UTC")
    previous = month - pd.offsets.MonthBegin(1)
    return prefix + previous.strftime("%Y_%m") + ".parquet", month


def append_prior_runs(schedules, observed, prior_schedules, prior_observed, month_start):
    """Append complete previous-month dated routes absent from the current feed.

The run-key exclusion applies to the whole route, so a previous archive cannot
fill missing events or replace actual observations on a current retained run.
All prior rows are kept for each selected route so the maximum acceptance date
still controls route eligibility. Pruning uses only scheduled route times.
"""
    current_keys = pd.MultiIndex.from_frame(schedules[RUN_KEY].drop_duplicates())
    prior_keys = pd.MultiIndex.from_frame(prior_schedules[RUN_KEY])
    before_month = pd.to_datetime(prior_schedules.departureDate, utc=True) < month_start
    disjoint = ~prior_keys.isin(current_keys)
    prior = prior_schedules.loc[before_month & disjoint].copy()
    # A terminal before this lower bound cannot enter any focal 720-minute window.
    current_times = pd.to_datetime(schedules.scheduledTime, utc=True, errors="coerce")
    lower = current_times.min() - pd.Timedelta(minutes=720)
    prior_times = pd.to_datetime(prior.scheduledTime, utc=True, errors="coerce")
    end_times = prior.assign(_time=prior_times).groupby(RUN_KEY)._time.max()
    retained_keys = end_times.index[end_times >= lower]
    keep = pd.MultiIndex.from_frame(prior[RUN_KEY]).isin(retained_keys)
    prior = prior.loc[keep]
    prior_obs = prior_observed.loc[pd.MultiIndex.from_frame(prior_observed[RUN_KEY]).isin(retained_keys)]
    result_schedules = pd.concat([schedules, prior], ignore_index=True)
    result_observed = pd.concat([observed, prior_obs], ignore_index=True)
    pd.testing.assert_frame_equal(result_schedules.iloc[:len(schedules)].reset_index(drop=True), schedules.reset_index(drop=True))
    pd.testing.assert_frame_equal(result_observed.iloc[:len(observed)].reset_index(drop=True), observed.reset_index(drop=True))
    stats = {
        "prior_route_rows_appended": len(prior),
        "prior_dated_runs_appended": len(retained_keys),
        "prior_observation_rows_appended": len(prior_obs),
        "prior_overlap_route_rows_excluded": int((before_month & ~disjoint).sum()),
        "prior_schedule_lower_relevance_bound": lower.isoformat(),
    }
    return result_schedules, result_observed, stats


def read_context(archive, name, targets, last_label):
    """Read current and prior source columns with a strict development bound."""
    archive = Path(archive)
    path = archive / name
    bound = min(pd.to_datetime(targets.scheduledTime, utc=True).max() - pd.Timedelta(minutes=30), last_label)
    timestamp = bound.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
    filters = [("departureDate", "<=", str(targets.departureDate.max()))]
    schedules = pq.read_table(path, columns=stale_turnaround.STATIC_COLUMNS, filters=filters).to_pandas()
    observed = pq.read_table(path, columns=stale.KEY + ["actualTime", TARGET],
                             filters=filters + [("actualTime", "<", timestamp)]).to_pandas()
    source = {"file": name, "sha256": sha256(path), "observed_rows": len(observed),
              "observation_read_before": timestamp, "departure_date_read_through": str(targets.departureDate.max())}
    previous_name, month_start = previous_source(name)
    previous_path = archive / previous_name
    prior_source = {"file": previous_name, "available": previous_path.exists(),
                    "departure_date_read_before": month_start.strftime("%Y-%m-%d"),
                    "observation_read_before": timestamp}
    if previous_path.exists():
        prior_filters = [("departureDate", "<", month_start.strftime("%Y-%m-%d"))]
        prior_schedules = pq.read_table(previous_path, columns=stale_turnaround.STATIC_COLUMNS,
                                        filters=prior_filters).to_pandas()
        prior_observed = pq.read_table(previous_path, columns=stale.KEY + ["actualTime", TARGET],
                                       filters=prior_filters + [("actualTime", "<", timestamp)]).to_pandas()
        prior_source.update({"sha256": sha256(previous_path), "schedule_rows_read": len(prior_schedules),
                             "observed_rows_read": len(prior_observed)})
        schedules, observed, stats = append_prior_runs(schedules, observed, prior_schedules, prior_observed, month_start)
        prior_source.update(stats)
    return schedules, observed, {"current": source, "previous": prior_source}


def prepare(args):
    dest = Path(args.output)
    if dest.exists() or dest.with_suffix(".manifest.json").exists():
        raise ValueError("Boundary preparation requires a new separate output")
    cohort = stale.development_cohort(args.cohort)
    if len(cohort) != stale.DEVELOPMENT_ROWS:
        raise ValueError("Incomplete frozen development cohort")
    base = stale.load_data(args.base_data)
    np.testing.assert_array_equal(np.sort(base.row_id), cohort.row_id)
    manifest = json.loads(Path(args.base_data).with_suffix(".manifest.json").read_text())
    if set(stale_origin.FEATURES + stale_origin.REVERSE_FEATURES) & set(base.columns):
        raise ValueError("Use the unchanged turnaround base, before origin predictors")
    last_label = pd.to_datetime(cohort.event_time, utc=True).max()
    parts, sources = [], []
    for name, targets in cohort.groupby("source_file", sort=True):
        schedules, observed, source = read_context(args.archive, name, targets, last_label)
        part = stale_origin.snapshot_features(targets, schedules, observed, reverse_features=True)
        part["row_id"] = targets.row_id
        parts.append(part)
        sources.append(source)
        print(f"Boundary origin {name}: {len(targets)} events, {source['previous'].get('prior_dated_runs_appended', 0)} prior dated runs", flush=True)
    extras = pd.concat(parts).set_index("row_id").reindex(base.row_id).reset_index(drop=True)
    frame = pd.concat([base.reset_index(drop=True), extras], axis=1)
    pd.testing.assert_frame_equal(frame[base.columns], base)
    frame.to_parquet(dest, index=False)
    result = {
        **manifest,
        "features": manifest["features"] + stale_origin.FEATURES + stale_origin.REVERSE_FEATURES,
        "features_sha256": sha256(dest),
        "feature_profile": "origin_reverse_previous_month_disjoint_context",
        "base_data": str(args.base_data), "base_data_sha256": sha256(args.base_data),
        "additional_provenance_columns": manifest.get("additional_provenance_columns", []) + stale_origin.STAMPS,
        "provenance_count_columns": {**manifest.get("provenance_count_columns", {}), **stale_origin.COUNTS},
        "origin_context_sources": sources, "origin_reverse_features": True,
        "holdout_read": False,
        "origin_context_policy": "same v2 origin/reverse summaries; previous-month dated runs only when absent from all current-source run keys; complete appended routes; all route acceptance and actual observations strictly before query minus30min; own focal run excluded",
        "origin_reverse_policy": "scheduled reverse-route proximity, not identified rolling-stock linkage",
        "availability_limitations": "actualTime is an observation availability proxy; retained timetables are unversioned, so acceptance dates do not prove historical revision availability; cancellation flags and live estimates are excluded",
    }
    dest.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2))
    stale.load_data(dest)
    print(f"Saved {len(frame)} development events and {len(result['features'])} predictors", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data_archive")
    parser.add_argument("--cohort", default="data/oulu_features.parquet")
    parser.add_argument("--base-data", default="data/oulu_stale30_turnaround_development.parquet")
    parser.add_argument("--output", default="data/oulu_stale30_origin_boundary_development.parquet")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
