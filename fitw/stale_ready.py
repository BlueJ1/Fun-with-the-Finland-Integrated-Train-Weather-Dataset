"""Timestamped departure-readiness decision summaries on frozen development data.

Decision timestamps and actual movement times are availability proxies. Retained
monthly archives do not record API publication times or historical revisions.
"""

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import stale, stale_enriched, stale_origin_boundary, stale_turnaround
from .prepare import sha256

MINUTE = 60_000_000_000
OWN_SUMMARIES = ["count", "last_delay_minutes", "max_delay_minutes", "mean_delay_minutes",
                 "last_age_minutes", "last_to_actual_minutes", "mean_to_actual_minutes", "max_to_actual_minutes"]
CANDIDATE_SUMMARIES = ["count", "mean_delay_minutes", "max_delay_minutes", "closest_delay_minutes",
                       "closest_age_minutes", "closest_to_actual_minutes", "mean_to_actual_minutes", "max_to_actual_minutes"]
FEATURES = [f"ready_own_{name}" for name in OWN_SUMMARIES]
FEATURES += [f"ready_origin_{window}m_{name}" for window in (180, 720) for name in CANDIDATE_SUMMARIES]
STAMPS = ["ready_latest_decision_time", "ready_latest_actual_time", "ready_latest_timetable_time"]
COUNTS = {name: [name.replace("latest_", "").replace("_time", "_presence_count")] for name in STAMPS}


def parse_readiness(raw, bound):
    """Parse dictionaries safely and retain only accepted old decision records.

Missing records stay unknown. A future retained decision cannot prove an earlier
absence of readiness. Its acceptance flag and source never become predictors.
"""
    records = []
    malformed = 0
    for row in raw.itertuples(index=False):
        value = row.trainReady
        if not isinstance(value, str) or value.strip() in {"", "nan", "None", "null"}:
            continue
        try:
            try:
                message = json.loads(value)
            except json.JSONDecodeError:
                message = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            malformed += 1
            continue
        if not isinstance(message, dict) or message.get("accepted") is not True:
            continue
        timestamp = message.get("timestamp")
        if not isinstance(timestamp, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", timestamp):
            malformed += 1
            continue
        records.append({**{name: getattr(row, name) for name in stale.KEY}, "ready_timestamp": timestamp})
    out = pd.DataFrame(records, columns=stale.KEY + ["ready_timestamp"])
    out["ready_timestamp"] = pd.to_datetime(out.ready_timestamp, utc=True, format="mixed", errors="coerce")
    valid = int(out.ready_timestamp.notna().sum())
    malformed += int(out.ready_timestamp.isna().sum())
    out = out.loc[out.ready_timestamp < bound].reset_index(drop=True)
    return out, {"readiness_rows_read": len(raw), "accepted_timestamp_records": valid,
                 "readiness_decision_rows_retained": len(out), "malformed_readiness_records": malformed,
                 "readiness_decision_before": bound.isoformat()}


def snapshot_features(targets, schedules, observed, readiness):
    """Summarize old nonfocal readiness and already-old actual departures."""
    query = targets.copy()
    if "stationUICCode" not in query:
        query["stationUICCode"] = 370
    feed = schedules[stale_turnaround.STATIC_COLUMNS].drop_duplicates(stale.KEY).merge(
        observed[stale.KEY + ["actualTime"]].drop_duplicates(stale.KEY), on=stale.KEY, how="left", validate="one_to_one")
    feed = feed.merge(readiness[stale.KEY + ["ready_timestamp"]].drop_duplicates(stale.KEY),
                      on=stale.KEY, how="left", validate="one_to_one")
    for name in ["scheduledTime", "actualTime", "timetableAcceptanceDate", "ready_timestamp"]:
        stamp = pd.to_datetime(feed[name], utc=True, errors="coerce", format="mixed")
        feed[name + "_valid"] = stamp.notna()
        feed[name + "_ns"] = stale_enriched._nanoseconds(stamp)
    feed = feed.sort_values(["scheduledTime_ns", "type"], kind="stable").reset_index(drop=True)
    arrays = {name: feed[name].to_numpy() for name in feed}
    routes, terminal = {}, {}
    for key, positions in feed.groupby(["departureDate", "trainNumber"], sort=False).groups.items():
        idx = np.asarray(positions)
        first, last = idx[0], idx[-1]
        route = {"idx": idx, "valid": bool((arrays["scheduledTime_valid"][idx] & arrays["timetableAcceptanceDate_valid"][idx]).all()),
                 "accepted": arrays["timetableAcceptanceDate_ns"][idx].max(),
                 "origin": arrays["stationUICCode"][first], "start": arrays["scheduledTime_ns"][first],
                 "finish": arrays["scheduledTime_ns"][last]}
        routes[key] = route
        if route["valid"] and arrays["type"][last] == "ARRIVAL" and arrays["trainCategory"][last] == "Long-distance":
            terminal.setdefault(arrays["stationUICCode"][last], []).append((route["finish"], key))
    terminals = {}
    for station, entries in terminal.items():
        entries.sort(key=lambda item: item[0])
        terminals[station] = (np.asarray([entry[0] for entry in entries]), [entry[1] for entry in entries])
    records = []
    for q in query.itertuples():
        focal = pd.Timestamp(q.scheduledTime).value
        cutoff = focal - 30 * MINUTE
        own_key = (q.departureDate, q.trainNumber)
        own = routes.get(own_key)
        row = {name: np.nan for name in FEATURES}
        used_ready, used_actual, used_static = [], [], []
        counts = {name: 0 for name in STAMPS}

        def eligible(route, exclude_focal=False):
            idx = route["idx"]
            mask = (arrays["type"][idx] == "DEPARTURE") & arrays["ready_timestamp_valid"][idx]
            mask &= (arrays["ready_timestamp_ns"][idx] < cutoff)
            mask &= arrays["timetableAcceptanceDate_valid"][idx] & (arrays["timetableAcceptanceDate_ns"][idx] < cutoff)
            if exclude_focal:
                mask &= ~((arrays["scheduledTime_ns"][idx] == focal) & (arrays["type"][idx] == q.type)
                          & (arrays["stationUICCode"][idx] == q.stationUICCode))
            return idx[mask]

        def values(indices):
            delay = (arrays["ready_timestamp_ns"][indices] - arrays["scheduledTime_ns"][indices]) / MINUTE
            actual_old = arrays["actualTime_valid"][indices] & (arrays["actualTime_ns"][indices] < cutoff)
            duration = np.full(len(indices), np.nan)
            duration[actual_old] = (arrays["actualTime_ns"][indices[actual_old]] - arrays["ready_timestamp_ns"][indices[actual_old]]) / MINUTE
            return delay, duration, actual_old

        def record(indices, route):
            if not len(indices):
                return
            used_ready.append(arrays["ready_timestamp_ns"][indices].max())
            counts[STAMPS[0]] += len(indices)
            used_static.append(route["accepted"])
            counts[STAMPS[2]] += 1
            old = arrays["actualTime_valid"][indices] & (arrays["actualTime_ns"][indices] < cutoff)
            if old.any():
                used_actual.append(arrays["actualTime_ns"][indices[old]].max())
                counts[STAMPS[1]] += int(old.sum())

        if own and own["valid"] and own["accepted"] < cutoff:
            idx = own["idx"]
            focal_matches = ((arrays["scheduledTime_ns"][idx] == focal) & (arrays["type"][idx] == q.type)
                             & (arrays["stationUICCode"][idx] == q.stationUICCode)).any()
            if focal_matches:
                old = eligible(own, exclude_focal=True)
                if len(old):
                    delay, duration, _ = values(old)
                    last = np.argmax(arrays["ready_timestamp_ns"][old])
                    row.update({"ready_own_count": len(old), "ready_own_last_delay_minutes": delay[last],
                                "ready_own_mean_delay_minutes": float(delay.mean()), "ready_own_max_delay_minutes": float(delay.max()),
                                "ready_own_last_age_minutes": (cutoff - arrays["ready_timestamp_ns"][old[last]]) / MINUTE,
                                "ready_own_last_to_actual_minutes": duration[last]})
                    finite = duration[np.isfinite(duration)]
                    if len(finite):
                        row["ready_own_mean_to_actual_minutes"] = float(finite.mean())
                        row["ready_own_max_to_actual_minutes"] = float(finite.max())
                    record(old, own)
                times, keys = terminals.get(own["origin"], (np.array([], dtype=np.int64), []))
                lo, hi = np.searchsorted(times, [own["start"] - 720 * MINUTE, own["start"] - 15 * MINUTE], side="left")
                candidates = []
                for other_key in keys[lo:hi]:
                    route = routes[other_key]
                    if other_key == own_key or route["accepted"] >= cutoff:
                        continue
                    candidate_ready = eligible(route)
                    if not len(candidate_ready):
                        continue
                    last = candidate_ready[np.argmax(arrays["ready_timestamp_ns"][candidate_ready])]
                    delay, duration, _ = values(np.asarray([last]))
                    candidates.append({"gap": (own["start"] - route["finish"]) / MINUTE,
                                       "delay": delay[0], "age": (cutoff - arrays["ready_timestamp_ns"][last]) / MINUTE,
                                       "duration": duration[0]})
                    record(np.asarray([last]), route)
                for window in (180, 720):
                    selected = [candidate for candidate in candidates if candidate["gap"] <= window]
                    if not selected:
                        continue
                    prefix = f"ready_origin_{window}m_"
                    closest = min(selected, key=lambda candidate: candidate["gap"])
                    row.update({prefix + "count": len(selected),
                                prefix + "mean_delay_minutes": float(np.mean([candidate["delay"] for candidate in selected])),
                                prefix + "max_delay_minutes": max(candidate["delay"] for candidate in selected),
                                prefix + "closest_delay_minutes": closest["delay"],
                                prefix + "closest_age_minutes": closest["age"],
                                prefix + "closest_to_actual_minutes": closest["duration"]})
                    durations = [candidate["duration"] for candidate in selected if np.isfinite(candidate["duration"])]
                    if durations:
                        row[prefix + "mean_to_actual_minutes"] = float(np.mean(durations))
                        row[prefix + "max_to_actual_minutes"] = max(durations)
                if candidates:
                    used_static.append(own["accepted"])
                    counts[STAMPS[2]] += 1
        for name, used in zip(STAMPS, [used_ready, used_actual, used_static]):
            row[name] = pd.Timestamp(max(used), unit="ns", tz="UTC") if used else pd.NaT
            row[COUNTS[name][0]] = counts[name]
        records.append(row)
    out = pd.DataFrame(records, index=targets.index)
    for name in STAMPS:
        out[name] = pd.to_datetime(out[name], utc=True)
    stale.validate_additional_provenance(out, {"additional_provenance_columns": STAMPS, "provenance_count_columns": COUNTS},
                                         pd.to_datetime(targets.scheduledTime, utc=True) - pd.Timedelta(minutes=30))
    return out


def read_context(archive, name, targets, last_label):
    archive = Path(archive)
    previous_name, month = stale_origin_boundary.previous_source(name)
    bound = min(pd.to_datetime(targets.scheduledTime, utc=True).max() - pd.Timedelta(minutes=30), last_label)
    timestamp = bound.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    def read(path, filters):
        schedules = pq.read_table(path, columns=stale_turnaround.STATIC_COLUMNS, filters=filters).to_pandas()
        observed = pq.read_table(path, columns=stale.KEY + ["actualTime"],
                                 filters=filters + [("actualTime", "<", timestamp)]).to_pandas()
        raw = pq.read_table(path, columns=stale.KEY + ["trainReady"], filters=filters).to_pandas()
        readiness, stats = parse_readiness(raw, bound)
        return schedules, observed, readiness, {"file": path.name, "sha256": sha256(path), **stats,
                                               "observation_read_before": timestamp, "observed_rows_read": len(observed)}

    current_filter = [("departureDate", "<=", str(targets.departureDate.max()))]
    schedules, observed, readiness, current = read(archive / name, current_filter)
    current["departure_date_read_through"] = str(targets.departureDate.max())
    previous_path = archive / previous_name
    previous = {"file": previous_name, "available": previous_path.exists(), "departure_date_read_before": month.strftime("%Y-%m-%d")}
    if previous_path.exists():
        prior_schedules, prior_observed, prior_readiness, previous_read = read(previous_path, [("departureDate", "<", month.strftime("%Y-%m-%d"))])
        schedules, observed, stats = stale_origin_boundary.append_prior_runs(schedules, observed, prior_schedules, prior_observed, month)
        kept = pd.MultiIndex.from_frame(schedules.iloc[len(schedules) - stats["prior_route_rows_appended"]:][stale_origin_boundary.RUN_KEY]) if stats["prior_route_rows_appended"] else pd.MultiIndex.from_tuples([], names=stale_origin_boundary.RUN_KEY)
        prior_readiness = prior_readiness.loc[pd.MultiIndex.from_frame(prior_readiness[stale_origin_boundary.RUN_KEY]).isin(kept)]
        readiness = pd.concat([readiness, prior_readiness], ignore_index=True)
        previous.update({**previous_read, **stats, "prior_readiness_rows_appended": len(prior_readiness)})
    return schedules, observed, readiness, {"current": current, "previous": previous}


def prepare(args):
    dest = Path(args.output)
    if dest.exists() or dest.with_suffix(".manifest.json").exists():
        raise ValueError("Readiness preparation requires a new separate output")
    cohort = pq.read_table(args.cohort, columns=["row_id", "event_time", "source_file", "departureDate", "trainNumber", "type", "scheduledTime"],
                           filters=[("row_id", "<", stale.DEVELOPMENT_ROWS)]).to_pandas().sort_values("row_id").reset_index(drop=True)
    if len(cohort) != stale.DEVELOPMENT_ROWS:
        raise ValueError("Incomplete frozen development cohort")
    base = stale.load_data(args.base_data)
    np.testing.assert_array_equal(np.sort(base.row_id), cohort.row_id)
    manifest = json.loads(Path(args.base_data).with_suffix(".manifest.json").read_text())
    if set(FEATURES) & set(base.columns):
        raise ValueError("The base already contains readiness predictors")
    parts, sources = [], []
    last_label = pd.to_datetime(cohort.event_time, utc=True).max()
    for name, targets in cohort.groupby("source_file", sort=True):
        schedules, observed, readiness, source = read_context(args.archive, name, targets, last_label)
        part = snapshot_features(targets, schedules, observed, readiness)
        part["row_id"] = targets.row_id
        parts.append(part)
        sources.append(source)
        print(f"Readiness {name}: {len(targets)} events, {len(readiness)} old decision records", flush=True)
    extras = pd.concat(parts).set_index("row_id").reindex(base.row_id).reset_index(drop=True)
    frame = pd.concat([base.reset_index(drop=True), extras], axis=1)
    pd.testing.assert_frame_equal(frame[base.columns], base)
    frame.to_parquet(dest, index=False)
    result = {**manifest, "features": manifest["features"] + FEATURES, "features_sha256": sha256(dest),
              "feature_profile": "origin_boundary_readiness_decision_context",
              "base_data": str(args.base_data), "base_data_sha256": sha256(args.base_data),
              "additional_provenance_columns": manifest.get("additional_provenance_columns", []) + STAMPS,
              "provenance_count_columns": {**manifest.get("provenance_count_columns", {}), **COUNTS},
              "readiness_sources": sources, "holdout_read": False,
              "readiness_policy": "accepted=True timestamped departure decisions strictly before query minus30min; focal timetable row excluded; complete route maximum acceptance strictly old; only other dated routes in origin candidate context; own-run summaries use all old nonfocal decisions, candidate summaries use latest old decision per route; duration uses only actual departure strictly before same cutoff; missing readiness is unknown, all predictors NaN without old decisions",
              "readiness_target_read": False,
              "readiness_duration_policy": "signed actualTime minus readiness decision timestamp, only when both strictly precede the cutoff; negative gaps mean the decision timestamp follows the recorded movement",
              "readiness_availability_limitations": "readiness timestamp is notification decision time, not proven API publication or ingestion time; actualTime is movement time and may be reported later; retained timetable revisions are unversioned; causes, cancellation/status/live estimate/weather fields excluded; January2018 sample found old own readiness mostly redundant with already-old actual departure",
              "readiness_documentation": "https://www.digitraffic.fi/rautatieliikenne/"}
    dest.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2))
    stale.load_data(dest)
    print(f"Saved {len(frame)} development events and {len(result['features'])} predictors", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data_archive")
    parser.add_argument("--cohort", default="data/oulu_features.parquet")
    parser.add_argument("--base-data", default="data/oulu_stale30_origin_boundary_development.parquet")
    parser.add_argument("--output", default="data/oulu_stale30_readiness_development.parquet")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
