"""Old Oulu station context and terminating-inbound proximity proxies.

Archived acceptance dates are availability proxies. They do not prove that the
retained final timetable revision was available historically. Nearby terminating
services are not identified rolling-stock connections; no vehicle IDs exist in
the source. Every actual observation must precede the per-query snapshot.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import stale, stale_enriched
from .prepare import sha256
from .spec import TARGET


WINDOWS = (60, 180)
STATION_SUMMARIES = ("observed_count", "delay_count", "mean_delay", "max_delay", "last_delay",
                     "pending_count", "resolved_count", "oldest_pending_minutes")
EXTRA_FEATURES = [f"oulu_context_{kind}_{window}m_{name}"
                  for kind in ("arrival", "departure") for window in WINDOWS for name in STATION_SUMMARIES]
EXTRA_FEATURES += ["oulu_origin_departure", "oulu_terminating_recent_count",
                   "oulu_terminating_recent_mean_delay", "oulu_terminating_recent_max_delay",
                   "oulu_terminating_last_delay", "oulu_terminating_last_observation_age_minutes",
                   "oulu_terminating_pending_count", "oulu_terminating_oldest_pending_minutes",
                   "turnaround_candidate_count", "turnaround_candidate_unobserved_count",
                   "turnaround_candidate_scheduled_gap_minutes", "turnaround_candidate_last_known_delay",
                   "turnaround_candidate_observation_age_minutes", "turnaround_candidate_pending_minutes",
                   "turnaround_candidate_projected_departure_delay",
                   "turnaround_worst_projected_departure_delay"]
PROVENANCE_COLUMNS = ["oulu_context_latest_actual_time", "oulu_context_latest_timetable_time"]
PROVENANCE_COUNT_COLUMNS = {PROVENANCE_COLUMNS[0]: ["oulu_context_actual_presence_count"],
                            PROVENANCE_COLUMNS[1]: ["oulu_context_timetable_presence_count"]}
STATIC_COLUMNS = stale.KEY + ["timetableAcceptanceDate", "trainCategory"]
_MINUTE_NS = 60_000_000_000


def snapshot_features(targets, schedules, observed, base_features=None):
    """Return extra features, or append them to an unchanged feature frame.

Terminating long-distance arrivals are candidate context when their accepted
route ends at Oulu and their scheduled arrival precedes an origin departure by
20 to 180 minutes. The candidate closest in scheduled time supplies a persistence
projection with a fixed 30-minute turnaround allowance. This is a proximity
proxy, not evidence that the inbound service supplies the departing train.
"""
    targets = targets.copy()
    if "stationUICCode" not in targets:
        targets["stationUICCode"] = 370
    observed = observed[stale.KEY + ["actualTime", TARGET]].drop_duplicates(stale.KEY, keep="last")
    feed = schedules[STATIC_COLUMNS].merge(observed, on=stale.KEY, how="left", validate="one_to_one")
    for external, internal in [("actualTime", "actual"), ("scheduledTime", "scheduled"),
                               ("timetableAcceptanceDate", "accepted")]:
        stamps = pd.to_datetime(feed[external], utc=True, errors="coerce")
        feed["_" + internal] = stale_enriched._nanoseconds(stamps)
        feed["_" + internal + "_valid"] = stamps.notna()
    feed = feed.sort_values(["_scheduled", "stationUICCode", "type"], kind="stable").reset_index(drop=True)
    arrays = {name: feed[name].to_numpy() for name in feed.columns}
    s, a, av = arrays["_scheduled"], arrays["_actual"], arrays["_accepted"]
    delays = pd.to_numeric(feed[TARGET], errors="coerce").to_numpy(dtype=float)
    run_indices = {key: np.asarray(indices) for key, indices in
                   feed.groupby(["departureDate", "trainNumber"], sort=False).groups.items()}
    run_for = np.empty(len(feed), dtype=object)
    routes = {}
    for key, indices in run_indices.items():
        # A complete retained route must have an age-eligible acceptance date
        # for every row before its endpoint can be used as a terminus proxy.
        valid_route = bool((arrays["_scheduled_valid"][indices] & arrays["_accepted_valid"][indices]).all())
        routes[key] = {"indices": indices, "valid": valid_route,
                       "accepted": av[indices].max() if valid_route else None,
                       "origin": arrays["stationUICCode"][indices[0]],
                       "terminus": arrays["stationUICCode"][indices[-1]]}
        for i in indices:
            run_for[i] = key
    station = np.flatnonzero((arrays["stationUICCode"] == 370) & arrays["_scheduled_valid"])
    station_actual = station[arrays["_actual_valid"][station]]
    station_actual = station_actual[np.argsort(a[station_actual], kind="stable")]
    scheduled_query = pd.to_datetime(targets.scheduledTime, utc=True)
    records = []
    for index, query in targets.iterrows():
        key = (query.departureDate, query.trainNumber)
        focal = scheduled_query.loc[index].value
        cutoff = focal - 30 * _MINUTE_NS
        row = {"index": index}
        used_actual, used_static = [], []
        route = routes.get(key)
        focal_matches = False
        if route:
            own = route["indices"]
            focal_matches = bool(((s[own] == focal) & (arrays["type"][own] == query.type)
                                  & (arrays["stationUICCode"][own] == query.stationUICCode)).any())
        if route and route["valid"] and route["accepted"] < cutoff and focal_matches:
            row["oulu_origin_departure"] = float(query.type == "DEPARTURE" and route["origin"] == 370)
            used_static.append(route["accepted"])
        beginning = cutoff - max(WINDOWS) * _MINUTE_NS
        lo, hi = np.searchsorted(s[station], [beginning, focal], side="left")
        plan_nearby = station[lo:hi]
        lo, hi = np.searchsorted(a[station_actual], [beginning, cutoff], side="left")
        recent_nearby = station_actual[lo:hi]
        nearby = np.unique(np.concatenate([plan_nearby, recent_nearby]))
        same_run = (arrays["departureDate"][nearby] == key[0]) & (arrays["trainNumber"][nearby] == key[1])
        known = arrays["_accepted_valid"][nearby] & (av[nearby] < cutoff)
        nearby = nearby[~same_run & known]
        old = arrays["_actual_valid"][nearby] & (a[nearby] < cutoff)
        completion_cache = {}

        def old_completions(other_key):
            if other_key not in completion_cache:
                other = run_indices[other_key]
                eligible = arrays["_accepted_valid"][other] & (av[other] < cutoff)
                eligible &= arrays["_actual_valid"][other] & (a[other] < cutoff)
                completion_cache[other_key] = other[eligible]
            return completion_cache[other_key]

        def unresolved(indices):
            pending, resolved = [], []
            for i in indices:
                later = old_completions(run_for[i])
                later = later[s[later] >= s[i]]
                if len(later):
                    used_actual.append(a[later].max())
                    used_static.append(av[later].max())
                    resolved.append(i)
                else:
                    pending.append(i)
            return np.asarray(pending, dtype=int), resolved

        for kind in ("ARRIVAL", "DEPARTURE"):
            typed = nearby[arrays["type"][nearby] == kind]
            typed_old = arrays["_actual_valid"][typed] & (a[typed] < cutoff)
            for window in WINDOWS:
                prefix = f"oulu_context_{kind.lower()}_{window}m_"
                beginning = cutoff - window * _MINUTE_NS
                observed_indices = typed[typed_old & (a[typed] >= beginning)]
                finite = observed_indices[np.isfinite(delays[observed_indices])]
                pending_indices = typed[(s[typed] >= beginning) & (s[typed] < cutoff) & ~typed_old]
                pending, resolved = unresolved(pending_indices)
                values = delays[finite]
                row.update({prefix + "observed_count": len(observed_indices), prefix + "delay_count": len(finite),
                            prefix + "mean_delay": float(values.mean()) if len(values) else np.nan,
                            prefix + "max_delay": float(values.max()) if len(values) else np.nan,
                            prefix + "pending_count": len(pending), prefix + "resolved_count": len(resolved),
                            prefix + "oldest_pending_minutes": (cutoff - s[pending].min()) / _MINUTE_NS if len(pending) else 0.})
                if len(finite):
                    row[prefix + "last_delay"] = delays[finite[np.argmax(a[finite])]]
                used = np.unique(np.concatenate([observed_indices, pending_indices]))
                if len(used):
                    used_static.append(av[used].max())
                if len(observed_indices):
                    used_actual.append(a[observed_indices].max())
        terminating = []
        for i in nearby[arrays["type"][nearby] == "ARRIVAL"]:
            other_route = routes[run_for[i]]
            if (other_route["valid"] and other_route["accepted"] < cutoff
                    and other_route["terminus"] == 370 and arrays["trainCategory"][i] == "Long-distance"):
                terminating.append(i)
                used_static.append(other_route["accepted"])
        terminating = np.asarray(terminating, dtype=int)
        terminal_old = arrays["_actual_valid"][terminating] & (a[terminating] < cutoff)
        recent_terminal = terminating[terminal_old & (a[terminating] >= cutoff - 180 * _MINUTE_NS)
                                      & np.isfinite(delays[terminating])]
        terminal_pending, _ = unresolved(terminating[(s[terminating] < cutoff) & ~terminal_old])
        values = delays[recent_terminal]
        row.update(oulu_terminating_recent_count=len(values),
                   oulu_terminating_recent_mean_delay=float(values.mean()) if len(values) else np.nan,
                   oulu_terminating_recent_max_delay=float(values.max()) if len(values) else np.nan,
                   oulu_terminating_pending_count=len(terminal_pending),
                   oulu_terminating_oldest_pending_minutes=(cutoff - s[terminal_pending].min()) / _MINUTE_NS if len(terminal_pending) else 0.)
        if len(recent_terminal):
            latest = recent_terminal[np.argmax(a[recent_terminal])]
            row["oulu_terminating_last_delay"] = delays[latest]
            row["oulu_terminating_last_observation_age_minutes"] = (cutoff - a[latest]) / _MINUTE_NS
            used_actual.append(a[recent_terminal].max())
        if row.get("oulu_origin_departure") == 1.:
            candidates = terminating[(s[terminating] >= focal - 180 * _MINUTE_NS)
                                     & (s[terminating] <= focal - 20 * _MINUTE_NS)]
            row["turnaround_candidate_count"] = len(candidates)
            candidate_old = arrays["_actual_valid"][candidates] & (a[candidates] < cutoff)
            row["turnaround_candidate_unobserved_count"] = int((~candidate_old).sum())
            if len(candidates):
                closest = candidates[np.argmax(s[candidates])]
                row["turnaround_candidate_scheduled_gap_minutes"] = (focal - s[closest]) / _MINUTE_NS
                row["turnaround_candidate_pending_minutes"] = max(0., (cutoff - s[closest]) / _MINUTE_NS) if not (arrays["_actual_valid"][closest] and a[closest] < cutoff) else 0.
                projections = []
                for i in candidates:
                    available = old_completions(run_for[i])
                    available = available[(s[available] <= s[i]) & np.isfinite(delays[available])]
                    if len(available):
                        latest = available[np.argmax(s[available])]
                        projection = max(0., delays[latest] + 30. - (focal - s[i]) / _MINUTE_NS)
                        projections.append(projection)
                        used_actual.append(a[available].max())
                        used_static.append(av[available].max())
                        if i == closest:
                            row.update(turnaround_candidate_last_known_delay=delays[latest],
                                       turnaround_candidate_observation_age_minutes=(cutoff - a[latest]) / _MINUTE_NS,
                                       turnaround_candidate_projected_departure_delay=projection)
                if projections:
                    row["turnaround_worst_projected_departure_delay"] = max(projections)
        if used_actual:
            row[PROVENANCE_COLUMNS[0]] = pd.Timestamp(max(used_actual), unit="ns", tz="UTC")
        if used_static:
            row[PROVENANCE_COLUMNS[1]] = pd.Timestamp(max(used_static), unit="ns", tz="UTC")
        records.append(row)
    extras = pd.DataFrame(records).set_index("index").reindex(targets.index)
    result = pd.DataFrame({name: extras[name] if name in extras else np.nan for name in EXTRA_FEATURES}, index=targets.index)
    for name in PROVENANCE_COLUMNS:
        result[name] = pd.to_datetime(extras[name] if name in extras else pd.Series(pd.NaT, index=targets.index), utc=True)
    result = add_presence_counts(result)
    cutoff = scheduled_query - pd.Timedelta(minutes=30)
    validate_context(result, cutoff)
    if base_features is not None:
        if not base_features.index.equals(targets.index):
            raise ValueError("Base feature index differs from prediction query index")
        return pd.concat([base_features, result], axis=1)
    return result


def add_presence_counts(frame):
    """Add audit counts from feature presence, independently of timestamp values.

The station windows overlap, so these are presence counts rather than counts
of unique observations. They are metadata and are not learned predictors.
"""
    out = frame.copy()
    actual = pd.Series(0., index=frame.index)
    planned = frame.oulu_origin_departure.notna().astype(float)
    for name in EXTRA_FEATURES:
        if ((name.startswith("oulu_context_") and name.endswith("observed_count"))
                or name.endswith(("resolved_count", "recent_count"))):
            actual += frame[name].fillna(0)
        if name.endswith("pending_count") or name == "turnaround_candidate_count":
            planned += frame[name].fillna(0)
        if name.endswith("delay"):
            actual += frame[name].notna().astype(float)
    out[PROVENANCE_COUNT_COLUMNS[PROVENANCE_COLUMNS[0]][0]] = actual
    out[PROVENANCE_COUNT_COLUMNS[PROVENANCE_COLUMNS[1]][0]] = actual + planned
    return out


def validate_context(frame, cutoff=None):
    """Check additional timestamps and require provenance for used contexts."""
    cutoff = pd.to_datetime(frame.observation_cutoff if cutoff is None else cutoff, utc=True)
    for name in PROVENANCE_COLUMNS:
        if name not in frame:
            raise ValueError(f"Context provenance column is absent: {name}")
        values = pd.to_datetime(frame[name], utc=True)
        present = values.notna()
        if not (values[present] < cutoff[present]).all():
            raise ValueError(f"Context violates old snapshot: {name}")
    actual_required = pd.Series(False, index=frame.index)
    static_required = frame.oulu_origin_departure.notna() | frame.turnaround_candidate_count.notna()
    for name in EXTRA_FEATURES:
        if ((name.startswith("oulu_context_") and name.endswith("observed_count"))
                or name.endswith(("resolved_count", "recent_count"))):
            actual_required |= frame[name].fillna(0) > 0
        if name.endswith("pending_count"):
            static_required |= frame[name].fillna(0) > 0
        if name.endswith("delay"):
            actual_required |= frame[name].notna()
    static_required |= actual_required
    if frame.loc[actual_required, PROVENANCE_COLUMNS[0]].isna().any():
        raise ValueError("Observed context lacks actual-time provenance")
    if frame.loc[static_required, PROVENANCE_COLUMNS[1]].isna().any():
        raise ValueError("Planned context lacks timetable-time provenance")


def prepare(args):
    cohort = stale.development_cohort(args.cohort)
    if len(cohort) != stale.DEVELOPMENT_ROWS:
        raise ValueError("Frozen development cohort is incomplete")
    base = stale.load_data(args.base_data)
    np.testing.assert_array_equal(np.sort(base.row_id), cohort.row_id)
    manifest = json.loads(Path(args.base_data).with_suffix(".manifest.json").read_text())
    last_label = pd.to_datetime(cohort.event_time, utc=True).max()
    pieces, sources = [], []
    for name, targets in cohort.groupby("source_file", sort=True):
        path = Path(args.archive) / name
        latest_snapshot = min(pd.to_datetime(targets.scheduledTime, utc=True).max() - pd.Timedelta(minutes=30), last_label)
        timestamp = latest_snapshot.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
        date_filter = [("departureDate", "<=", str(targets.departureDate.max()))]
        schedules = pq.read_table(path, columns=STATIC_COLUMNS, filters=date_filter).to_pandas().drop_duplicates(stale.KEY)
        observed = pq.read_table(path, columns=stale.KEY + ["actualTime", TARGET],
                                filters=date_filter + [("actualTime", "<", timestamp)]).to_pandas()
        extra = snapshot_features(targets[[name for name in stale.KEY if name in targets]], schedules, observed)
        extra["row_id"] = targets.row_id
        pieces.append(extra)
        sources.append({"file": name, "sha256": sha256(path), "schedule_rows": len(schedules),
                        "observed_rows": len(observed), "observation_read_before": timestamp})
        print(f"Oulu context {name}: {len(targets)} queries, {len(observed)} old observations", flush=True)
    extras = pd.concat(pieces).set_index("row_id").reindex(base.row_id).reset_index(drop=True)
    frame = pd.concat([base.reset_index(drop=True), extras], axis=1)
    pd.testing.assert_frame_equal(frame[base.columns], base.reset_index(drop=True))
    validate_context(frame)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    additional = list(dict.fromkeys(manifest.get("additional_provenance_columns", []) + PROVENANCE_COLUMNS))
    result = {**manifest, "features": manifest["features"] + EXTRA_FEATURES,
              "feature_profile": "wide_service_oulu_turnaround", "features_sha256": sha256(destination),
              "base_data": str(Path(args.base_data)), "base_data_sha256": sha256(args.base_data),
              "additional_provenance_columns": additional,
              "provenance_count_columns": {**manifest.get("provenance_count_columns", {}), **PROVENANCE_COUNT_COLUMNS},
              "oulu_context_sources": sources,
              "holdout_read": False, "turnaround_policy": "other dated runs; retained complete route ends at Oulu; long-distance arrival planned 20-180 minutes before an origin departure; fixed 30-minute turnaround projection; no rolling-stock identity",
              "provenance_limit": "actualTime proxies observation availability; archived acceptance dates are availability proxies and do not establish historical availability of retained final timetable revisions; terminating-inbound proximity does not identify shared rolling stock"}
    destination.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2))
    stale.load_data(destination)
    print(f"Saved and age-checked {len(frame)} development rows with {len(EXTRA_FEATURES)} additional features", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data_archive")
    parser.add_argument("--cohort", default="data/oulu_features.parquet")
    parser.add_argument("--base-data", default="data/oulu_stale30_wide_service_development.parquet")
    parser.add_argument("--output", default="data/oulu_stale30_turnaround_development.parquet")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
