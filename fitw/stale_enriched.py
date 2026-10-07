"""Operational predictors computed from each event's 30-minute-old snapshot.

Missing reports are not treated as completions. A later, already observed event
does resolve earlier missing reports on the same dated run. Pending durations
describe missing observations and are evidence of delay, not guaranteed delay
bounds, because the archived feed does not identify reporting outages.
"""

import numpy as np
import pandas as pd

from . import stale
from .spec import TARGET


EXTRA_FEATURES = [
    "clean_pending_count", "resolved_missing_count", "latest_pending_minutes",
    "latest_pending_station", "latest_pending_is_departure", "clean_oldest_pending_minutes",
    "pending_scheduled_span_minutes", "last_completed_scheduled_age_minutes",
    "last_scheduled_delay", "last_scheduled_station", "last_scheduled_is_departure",
    "last_scheduled_observation_age_minutes", "scheduled_order_delay_change",
    "scheduled_order_delay_slope", "recent3_mean_delay", "recent3_median_delay",
    "recent3_max_delay", "recent3_min_delay", "recent3_std_delay",
    "recent5_mean_delay", "recent5_std_delay", "recent_delay_acceleration",
    "history_departure_mean_delay", "history_arrival_mean_delay",
    "history_departure_last_delay", "history_arrival_last_delay",
    "run_observed_fraction", "planned_total_events", "planned_total_stations",
    "planned_total_duration_minutes", "planned_destination_station",
    "planned_fraction_elapsed", "planned_stopping_events_before",
    "planned_events_after_source", "planned_dwell_after_source_minutes",
    "planned_running_after_source_minutes", "delay_after_planned_dwell",
    "next_scheduled_event_station", "next_scheduled_event_is_departure",
    "minutes_to_next_scheduled_event", "latest_pending_vs_previous_delay",
]
ENRICHED_FEATURES = stale.MODEL_FEATURES + EXTRA_FEATURES
PROVENANCE_COLUMNS = ["enriched_input_latest_actual_time", "enriched_input_latest_timetable_time"]
_MINUTE_NS = 60_000_000_000


def _nanoseconds(series):
    return series.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def snapshot_features(targets, schedules, observed):
    """Append operational features without changing base predictors or metadata.

The target is an unlabeled prediction query. Its scheduledTime is the prediction
time. Every used timetableAcceptanceDate and actualTime must be strictly earlier
than scheduledTime minus 30 minutes. Future scheduled events are known plans,
and their actual observations cannot contribute. Target labels are never read.
"""
    out = stale.snapshot_features(targets, schedules, observed)
    observed = observed[stale.KEY + ["actualTime", TARGET]].drop_duplicates(stale.KEY, keep="last")
    feed = schedules.merge(observed, on=stale.KEY, how="left", validate="one_to_one")
    actual = pd.to_datetime(feed.actualTime, utc=True, errors="coerce")
    scheduled = pd.to_datetime(feed.scheduledTime, utc=True, errors="coerce")
    accepted = pd.to_datetime(feed.timetableAcceptanceDate, utc=True, errors="coerce")
    feed = feed.assign(_actual=_nanoseconds(actual), _scheduled=_nanoseconds(scheduled),
                       _accepted=_nanoseconds(accepted), _actual_valid=actual.notna(),
                       _scheduled_valid=scheduled.notna(), _accepted_valid=accepted.notna())
    groups = {}
    for key, group in feed.groupby(["departureDate", "trainNumber"], sort=False):
        group = group.sort_values(["_scheduled", "stationUICCode", "type"], kind="stable")
        groups[key] = {name: group[name].to_numpy() for name in [
            "_actual", "_scheduled", "_accepted", "_actual_valid", "_scheduled_valid",
            "_accepted_valid", "stationUICCode", "type", "trainStopping", TARGET]}
    records = []
    for key, query in targets.groupby(["departureDate", "trainNumber"], sort=False):
        source = groups.get(key)
        for index in query.index:
            row = {"index": index}
            if source is None:
                records.append(row)
                continue
            focal = out.loc[index, "prediction_time"].value
            cutoff = out.loc[index, "observation_cutoff"].value
            s, a, av = source["_scheduled"], source["_actual"], source["_accepted"]
            known = source["_accepted_valid"] & source["_scheduled_valid"] & (av < cutoff)
            before = known & (s < focal)
            completed = before & source["_actual_valid"] & (a < cutoff)
            finite = completed & np.isfinite(source[TARGET].astype(float))
            observed_indices = np.flatnonzero(finite)
            completed_indices = np.flatnonzero(completed)
            raw_pending = before & (s < cutoff) & ~completed
            pending = raw_pending.copy()
            if len(completed_indices):
                latest_scheduled = s[completed_indices].max()
                pending &= s > latest_scheduled
                row["last_completed_scheduled_age_minutes"] = (cutoff - latest_scheduled) / _MINUTE_NS
            p = np.flatnonzero(pending)
            row.update(clean_pending_count=len(p), resolved_missing_count=int((raw_pending & ~pending).sum()),
                       clean_oldest_pending_minutes=(cutoff - s[p[0]]) / _MINUTE_NS if len(p) else 0.,
                       latest_pending_minutes=(cutoff - s[p[-1]]) / _MINUTE_NS if len(p) else 0.,
                       pending_scheduled_span_minutes=(s[p[-1]] - s[p[0]]) / _MINUTE_NS if len(p) else 0.,
                       run_observed_fraction=float(completed.sum() / before.sum()) if before.any() else np.nan)
            if len(p):
                row.update(latest_pending_station=source["stationUICCode"][p[-1]],
                           latest_pending_is_departure=float(source["type"][p[-1]] == "DEPARTURE"))
            if len(observed_indices):
                last = observed_indices[-1]
                delays = source[TARGET][observed_indices].astype(float)
                row.update(last_scheduled_delay=delays[-1], last_scheduled_station=source["stationUICCode"][last],
                           last_scheduled_is_departure=float(source["type"][last] == "DEPARTURE"),
                           last_scheduled_observation_age_minutes=(cutoff - a[last]) / _MINUTE_NS,
                           latest_pending_vs_previous_delay=row["latest_pending_minutes"] - delays[-1])
                for count in (3, 5):
                    values = delays[-count:]
                    row[f"recent{count}_mean_delay"] = float(values.mean())
                    row[f"recent{count}_std_delay"] = float(values.std())
                row.update(recent3_median_delay=float(np.median(delays[-3:])),
                           recent3_max_delay=float(delays[-3:].max()), recent3_min_delay=float(delays[-3:].min()))
                if len(delays) > 1:
                    change = delays[-1] - delays[-2]
                    gap = (s[last] - s[observed_indices[-2]]) / _MINUTE_NS
                    row.update(scheduled_order_delay_change=change,
                               scheduled_order_delay_slope=change / gap if gap > 0 else np.nan)
                if len(delays) > 2:
                    row["recent_delay_acceleration"] = delays[-1] - 2 * delays[-2] + delays[-3]
                for kind, prefix in [("DEPARTURE", "departure"), ("ARRIVAL", "arrival")]:
                    values = source[TARGET][observed_indices[source["type"][observed_indices] == kind]].astype(float)
                    if len(values):
                        row[f"history_{prefix}_mean_delay"] = float(values.mean())
                        row[f"history_{prefix}_last_delay"] = float(values[-1])
                after_source = before & (s > s[last])
                row["planned_events_after_source"] = int(after_source.sum())
                # Dwell uses matched, accepted arrival/departure plans only.
                dwell = 0.
                for departure in np.flatnonzero(after_source & (source["type"] == "DEPARTURE")):
                    arrivals = known & (source["type"] == "ARRIVAL") & (source["stationUICCode"] == source["stationUICCode"][departure]) & (s <= s[departure]) & (s >= s[last])
                    if arrivals.any():
                        dwell += (s[departure] - s[arrivals].max()) / _MINUTE_NS
                # Include focal departure's scheduled dwell when its plan is old.
                focal_departure = known & (s == focal) & (source["type"] == "DEPARTURE")
                if query.loc[index, "type"] == "DEPARTURE":
                    station = query.loc[index, "stationUICCode"] if "stationUICCode" in query else 370
                    focal_departure &= source["stationUICCode"] == station
                    if focal_departure.any():
                        arrivals = known & (source["type"] == "ARRIVAL") & (source["stationUICCode"] == station) & (s < focal) & (s >= s[last])
                        if arrivals.any():
                            dwell += (focal - s[arrivals].max()) / _MINUTE_NS
                gap = (focal - s[last]) / _MINUTE_NS
                row.update(planned_dwell_after_source_minutes=dwell,
                           planned_running_after_source_minutes=max(0., gap - dwell),
                           delay_after_planned_dwell=max(0., delays[-1] - dwell))
                upcoming = np.flatnonzero(known & (s > s[last]) & (s <= focal))
                if len(upcoming):
                    nxt = upcoming[0]
                    row.update(next_scheduled_event_station=source["stationUICCode"][nxt],
                               next_scheduled_event_is_departure=float(source["type"][nxt] == "DEPARTURE"),
                               minutes_to_next_scheduled_event=(s[nxt] - cutoff) / _MINUTE_NS)
            plan = np.flatnonzero(known)
            if len(plan):
                duration = (s[plan[-1]] - s[plan[0]]) / _MINUTE_NS
                row.update(planned_total_events=len(plan), planned_total_stations=len(np.unique(source["stationUICCode"][plan])),
                           planned_total_duration_minutes=duration,
                           planned_destination_station=source["stationUICCode"][plan[-1]],
                           planned_fraction_elapsed=(focal - s[plan[0]]) / _MINUTE_NS / duration if duration > 0 else np.nan,
                           planned_stopping_events_before=int((before & source["trainStopping"].astype(bool)).sum()),
                           enriched_input_latest_timetable_time=pd.Timestamp(av[plan].max(), unit="ns", tz="UTC"))
            if len(completed_indices):
                row["enriched_input_latest_actual_time"] = pd.Timestamp(a[completed_indices].max(), unit="ns", tz="UTC")
            records.append(row)
    extras = pd.DataFrame(records).set_index("index").reindex(targets.index)
    for name in EXTRA_FEATURES:
        out[name] = extras[name] if name in extras else np.nan
    for name in PROVENANCE_COLUMNS:
        out[name] = pd.to_datetime(extras[name], utc=True) if name in extras else pd.NaT
    return out
