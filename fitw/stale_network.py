"""Other-run station context from each prediction's old observation snapshot."""

import numpy as np
import pandas as pd

from . import stale, stale_enriched
from .spec import TARGET


CONTEXTS = ("previous_station", "next_station", "route_ahead")
WINDOWS = (60, 180)
SUMMARY_NAMES = ("mean_delay", "max_delay", "count", "last_delay",
                 "pending_count", "oldest_pending_minutes")
EXTRA_FEATURES = [f"network_{context}_{window}m_{name}" for context in CONTEXTS
                  for window in WINDOWS for name in SUMMARY_NAMES]
NETWORK_FEATURES = stale_enriched.ENRICHED_FEATURES + EXTRA_FEATURES
PROVENANCE_COLUMNS = ["network_input_latest_actual_time", "network_input_latest_timetable_time"]
_MINUTE_NS = 60_000_000_000


def snapshot_features(targets, schedules, observed, *, context_schedules=None, context_observed=None):
    """Append delays and unresolved reports on the accepted route ahead.

    Route stations follow the latest old completed event in scheduled order.
    Context excludes the focal dated run. Every contributing observation and
    accepted timetable precedes scheduledTime minus thirty minutes strictly.
    Later old completions resolve earlier missing reports, even without labels.
    """
    out = stale_enriched.snapshot_features(targets, schedules, observed)
    # A broader feed can supply other-run context while preserving all of the
    # original same-run predictors. Both feeds use the same per-query age gate.
    if context_schedules is not None:
        schedules = context_schedules
    if context_observed is not None:
        observed = context_observed
    observed = observed[stale.KEY + ["actualTime", TARGET]].drop_duplicates(stale.KEY, keep="last")
    feed = schedules.merge(observed, on=stale.KEY, how="left", validate="one_to_one")
    for source, internal in [("actualTime", "actual"), ("scheduledTime", "scheduled"),
                             ("timetableAcceptanceDate", "accepted")]:
        stamps = pd.to_datetime(feed[source], utc=True, errors="coerce")
        feed["_" + internal] = stale_enriched._nanoseconds(stamps)
        feed["_" + internal + "_valid"] = stamps.notna()
    feed = feed.sort_values(["_scheduled", "stationUICCode", "type"], kind="stable").reset_index(drop=True)
    values = {name: feed[name].to_numpy() for name in [
        "_actual", "_scheduled", "_accepted", "_actual_valid", "_scheduled_valid",
        "_accepted_valid", "stationUICCode", "departureDate", "trainNumber", TARGET]}
    run_indices = {key: np.asarray(indices) for key, indices in
                   feed.groupby(["departureDate", "trainNumber"], sort=False).groups.items()}
    station_indices = {key: np.asarray(indices) for key, indices in
                       feed.groupby("stationUICCode", sort=False).groups.items()}
    station_actual_indices = {}
    for station, indices in station_indices.items():
        valid = indices[values["_actual_valid"][indices]]
        station_actual_indices[station] = valid[np.argsort(values["_actual"][valid], kind="stable")]
    records = []
    for key, queries in targets.groupby(["departureDate", "trainNumber"], sort=False):
        source = run_indices.get(key, np.array([], dtype=int))
        for index in queries.index:
            row = {"index": index}
            cutoff, focal = out.loc[index, "observation_cutoff"].value, out.loc[index, "prediction_time"].value
            known = values["_accepted_valid"][source] & values["_scheduled_valid"][source] & (values["_accepted"][source] < cutoff)
            planned = source[known & (values["_scheduled"][source] <= focal)]
            completed = planned[(values["_scheduled"][planned] < focal) & values["_actual_valid"][planned]
                                & (values["_actual"][planned] < cutoff)]
            last = completed[-1] if len(completed) else None
            ahead = planned[values["_scheduled"][planned] > values["_scheduled"][last]] if last is not None else planned
            ahead_stations = list(dict.fromkeys(values["stationUICCode"][ahead]))[:3]
            contexts = {"previous_station": [values["stationUICCode"][last]] if last is not None else [],
                        "next_station": ahead_stations[:1], "route_ahead": ahead_stations}
            used_actual, used_timetable = [], []
            # These accepted plans determine the chosen route context.
            route_used = planned
            if len(route_used):
                used_timetable.append(values["_accepted"][route_used].max())
            if len(completed):
                used_actual.append(values["_actual"][completed].max())
            completion_cache = {}
            for context, stations in contexts.items():
                if not stations:
                    continue
                # The summaries can only use observations or planned events in
                # the longest context window. Index both clocks once rather
                # than scanning a whole station-month for every prediction.
                beginning = cutoff - max(WINDOWS) * _MINUTE_NS
                pieces = []
                for station in stations:
                    planned_indices = station_indices.get(station, np.array([], dtype=int))
                    scheduled_times = values["_scheduled"][planned_indices]
                    lo, hi = np.searchsorted(scheduled_times, [beginning, cutoff], side="left")
                    pieces.append(planned_indices[lo:hi])
                    actual_indices = station_actual_indices.get(station, np.array([], dtype=int))
                    actual_times = values["_actual"][actual_indices]
                    lo, hi = np.searchsorted(actual_times, [beginning, cutoff], side="left")
                    pieces.append(actual_indices[lo:hi])
                nearby = np.unique(np.concatenate(pieces))
                same_run = (values["departureDate"][nearby] == key[0]) & (values["trainNumber"][nearby] == key[1])
                accepted = values["_accepted_valid"][nearby] & values["_scheduled_valid"][nearby] & (values["_accepted"][nearby] < cutoff)
                nearby = nearby[~same_run & accepted]
                old = values["_actual_valid"][nearby] & (values["_actual"][nearby] < cutoff)
                for window in WINDOWS:
                    prefix = f"network_{context}_{window}m_"
                    beginning = cutoff - window * _MINUTE_NS
                    observed_indices = nearby[old & (values["_actual"][nearby] >= beginning)
                                              & np.isfinite(values[TARGET][nearby].astype(float))]
                    pending = nearby[(values["_scheduled"][nearby] >= beginning)
                                     & (values["_scheduled"][nearby] < cutoff) & ~old]
                    completed_status = nearby[(values["_scheduled"][nearby] >= beginning)
                                              & (values["_scheduled"][nearby] < cutoff) & old]
                    if len(completed_status):
                        used_actual.append(values["_actual"][completed_status].max())
                        used_timetable.append(values["_accepted"][completed_status].max())
                    unresolved = []
                    for pending_index in pending:
                        other_key = (values["departureDate"][pending_index], values["trainNumber"][pending_index])
                        if other_key not in completion_cache:
                            other = run_indices[other_key]
                            eligible = values["_accepted_valid"][other] & values["_scheduled_valid"][other] & (values["_accepted"][other] < cutoff)
                            eligible &= values["_actual_valid"][other] & (values["_actual"][other] < cutoff)
                            completion_cache[other_key] = other[eligible]
                        later = completion_cache[other_key]
                        later = later[values["_scheduled"][later] >= values["_scheduled"][pending_index]]
                        if len(later):
                            used_actual.append(values["_actual"][later].max())
                            used_timetable.append(values["_accepted"][later].max())
                        else:
                            unresolved.append(pending_index)
                    delays = values[TARGET][observed_indices].astype(float)
                    row.update({prefix + "count": len(delays),
                                prefix + "mean_delay": float(delays.mean()) if len(delays) else np.nan,
                                prefix + "max_delay": float(delays.max()) if len(delays) else np.nan,
                                prefix + "pending_count": len(unresolved),
                                prefix + "oldest_pending_minutes": (cutoff - values["_scheduled"][unresolved].min()) / _MINUTE_NS if unresolved else 0.})
                    if len(observed_indices):
                        latest = observed_indices[np.argmax(values["_actual"][observed_indices])]
                        row[prefix + "last_delay"] = float(values[TARGET][latest])
                        used_actual.append(values["_actual"][observed_indices].max())
                    used = np.concatenate([observed_indices, pending])
                    if len(used):
                        used_timetable.append(values["_accepted"][used].max())
            if used_actual:
                row[PROVENANCE_COLUMNS[0]] = pd.Timestamp(max(used_actual), unit="ns", tz="UTC")
            if used_timetable:
                row[PROVENANCE_COLUMNS[1]] = pd.Timestamp(max(used_timetable), unit="ns", tz="UTC")
            records.append(row)
    extras = pd.DataFrame(records).set_index("index").reindex(targets.index)
    network = pd.DataFrame({name: extras[name] if name in extras else np.nan
                            for name in EXTRA_FEATURES}, index=targets.index)
    for name in PROVENANCE_COLUMNS:
        network[name] = pd.to_datetime(extras[name] if name in extras else pd.Series(pd.NaT, index=targets.index), utc=True)
    return pd.concat([out, network], axis=1)
