"""Old terminating-service context at the focal route's origin.

Accepted routes indicate possible return-service proximity, not rolling-stock
identity. Every route and observation is checked against prediction minus30m.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from . import stale, stale_enriched, stale_turnaround
from .prepare import sha256
from .spec import TARGET

MINUTE = 60_000_000_000
FEATURES = ["origin_context_route_known", "origin_context_elapsed_minutes"]
SUMMARIES = ["count", "unobserved_count", "reverse_count", "closest_gap_minutes",
             "closest_last_delay", "closest_observation_age_minutes", "closest_pending_minutes",
             "closest_projected_delay", "closest_pending_projection", "max_projected_delay",
             "max_pending_projection", "mean_last_delay", "max_last_delay"]
FEATURES += [f"origin_context_{window}m_{name}" for window in (180, 720) for name in SUMMARIES]
REVERSE_SUMMARIES = ["closest_gap_minutes", "closest_last_delay", "closest_projected_delay",
                     "closest_observation_age_minutes", "closest_pending_minutes",
                     "closest_terminal_completion_age_minutes", "closest_terminal_to_planned_departure_minutes"]
REVERSE_FEATURES = [f"origin_reverse_{window}m_{name}" for window in (180, 720) for name in REVERSE_SUMMARIES]
STAMPS = ["origin_context_latest_actual_time", "origin_context_latest_timetable_time"]
COUNTS = {STAMPS[0]: ["origin_context_actual_presence_count"], STAMPS[1]: ["origin_context_timetable_presence_count"]}


def snapshot_features(targets, schedules, observed, reverse_features=False):
    query = targets.copy()
    if "stationUICCode" not in query:
        query["stationUICCode"] = 370
    feed = schedules[stale_turnaround.STATIC_COLUMNS].drop_duplicates(stale.KEY).merge(
        observed[stale.KEY + ["actualTime", TARGET]].drop_duplicates(stale.KEY), on=stale.KEY, how="left", validate="one_to_one")
    for name in ["scheduledTime", "actualTime", "timetableAcceptanceDate"]:
        stamps = pd.to_datetime(feed[name], utc=True, errors="coerce")
        feed[name+"_valid"] = stamps.notna()
        feed[name+"_ns"] = stale_enriched._nanoseconds(stamps)
    feed = feed.sort_values(["scheduledTime_ns", "type"], kind="stable").reset_index(drop=True)
    a = {name: feed[name].to_numpy() for name in feed}
    delay_values = pd.to_numeric(feed[TARGET], errors="coerce").to_numpy(float)
    routes, terminal_indices = {}, {}
    for key, rows in feed.groupby(["departureDate", "trainNumber"], sort=False).groups.items():
        idx = np.asarray(rows)
        valid = bool((a["scheduledTime_valid"][idx] & a["timetableAcceptanceDate_valid"][idx]).all())
        origin, end = idx[0], idx[-1]
        route = {"idx": idx, "valid": valid, "accepted": a["timetableAcceptanceDate_ns"][idx].max(),
                 "origin": a["stationUICCode"][origin], "end": a["stationUICCode"][end],
                 "start": a["scheduledTime_ns"][origin], "finish": a["scheduledTime_ns"][end], "terminal": end}
        routes[key] = route
        if valid and a["type"][end] == "ARRIVAL" and a["trainCategory"][end] == "Long-distance":
            terminal_indices.setdefault(route["end"], []).append((route["finish"], key))
    terminals = {}
    for station, entries in terminal_indices.items():
        entries.sort(key=lambda x:x[0])
        terminals[station] = (np.asarray([e[0] for e in entries]), [e[1] for e in entries])
    records = []
    for _, q in query.iterrows():
        key = (q.departureDate, q.trainNumber)
        focal = pd.Timestamp(q.scheduledTime).value
        cutoff = focal - 30*MINUTE
        row = {name: np.nan for name in FEATURES + (REVERSE_FEATURES if reverse_features else [])}
        actual_used, static_used = [], []
        actual_presence, static_presence = 0, 0
        own = routes.get(key)
        if own and own["valid"] and own["accepted"] < cutoff:
            idx = own["idx"]
            match = ((a["scheduledTime_ns"][idx] == focal) & (a["type"][idx] == q.type)
                     & (a["stationUICCode"][idx] == q.stationUICCode)).any()
            if match:
                row["origin_context_route_known"] = 1.
                row["origin_context_elapsed_minutes"] = (focal-own["start"])/MINUTE
                static_used.append(own["accepted"]); static_presence += 1
                times, keys = terminals.get(own["origin"], (np.array([], dtype=np.int64), []))
                lo, hi = np.searchsorted(times, [own["start"]-720*MINUTE, own["start"]-15*MINUTE], side="left")
                candidates = []
                for other_key in keys[lo:hi]:
                    other = routes[other_key]
                    if other_key == key or other["accepted"] >= cutoff:
                        continue
                    j = other["idx"]
                    eligible = a["actualTime_valid"][j] & (a["actualTime_ns"][j] < cutoff)
                    old = j[eligible]
                    finite = old[np.isfinite(delay_values[old])]
                    static_used.append(other["accepted"]); static_presence += 1
                    if len(old):
                        actual_used.append(a["actualTime_ns"][old].max()); actual_presence += len(old)
                    observed_terminal = bool(a["actualTime_valid"][other["terminal"]] and a["actualTime_ns"][other["terminal"]] < cutoff)
                    terminal_age = (cutoff-a["actualTime_ns"][other["terminal"]])/MINUTE if observed_terminal else np.nan
                    terminal_gap = (own["start"]-a["actualTime_ns"][other["terminal"]])/MINUTE if observed_terminal else np.nan
                    pending = max(0., (cutoff-other["finish"])/MINUTE) if not observed_terminal else 0.
                    gap = (own["start"]-other["finish"])/MINUTE
                    delay, age, projection = np.nan, np.nan, np.nan
                    if len(finite):
                        last = finite[np.argmax(a["scheduledTime_ns"][finite])]
                        delay = delay_values[last]; age = (cutoff-a["actualTime_ns"][last])/MINUTE
                        projection = max(0., delay+30.-gap)
                    candidates.append({"finish": other["finish"], "gap": gap, "unobserved": int(not observed_terminal),
                                       "reverse": int(other["origin"] == own["end"]), "delay": delay, "age": age,
                                       "pending": pending, "projection": projection,
                                       "terminal_age": terminal_age, "terminal_gap": terminal_gap,
                                       "pending_projection": max(0., pending+30.-gap) if pending > 0 else 0.})
                for window in (180,720):
                    selected = [c for c in candidates if c["gap"] <= window]
                    prefix = f"origin_context_{window}m_"
                    row[prefix+"count"] = len(selected)
                    row[prefix+"unobserved_count"] = sum(c["unobserved"] for c in selected)
                    row[prefix+"reverse_count"] = sum(c["reverse"] for c in selected)
                    if selected:
                        closest = max(selected, key=lambda c:c["finish"])
                        for name, source in [("closest_gap_minutes","gap"),("closest_last_delay","delay"),
                                             ("closest_observation_age_minutes","age"),("closest_pending_minutes","pending"),
                                             ("closest_projected_delay","projection"),("closest_pending_projection","pending_projection")]:
                            row[prefix+name] = closest[source]
                        delays = [c["delay"] for c in selected if np.isfinite(c["delay"])]
                        projections = [c["projection"] for c in selected if np.isfinite(c["projection"])]
                        if delays:
                            row[prefix+"mean_last_delay"] = float(np.mean(delays)); row[prefix+"max_last_delay"] = max(delays)
                        if projections: row[prefix+"max_projected_delay"] = max(projections)
                        row[prefix+"max_pending_projection"] = max(c["pending_projection"] for c in selected)
                        if reverse_features:
                            reverse = [c for c in selected if c["reverse"]]
                            if reverse:
                                closest_reverse = max(reverse, key=lambda c:c["finish"])
                                reverse_prefix = f"origin_reverse_{window}m_"
                                for name, source in [("closest_gap_minutes", "gap"), ("closest_last_delay", "delay"),
                                                     ("closest_projected_delay", "projection"), ("closest_observation_age_minutes", "age"),
                                                     ("closest_pending_minutes", "pending"), ("closest_terminal_completion_age_minutes", "terminal_age"),
                                                     ("closest_terminal_to_planned_departure_minutes", "terminal_gap")]:
                                    row[reverse_prefix+name] = closest_reverse[source]
        row[STAMPS[0]] = pd.Timestamp(max(actual_used), unit="ns", tz="UTC") if actual_used else pd.NaT
        row[STAMPS[1]] = pd.Timestamp(max(static_used), unit="ns", tz="UTC") if static_used else pd.NaT
        row[COUNTS[STAMPS[0]][0]] = actual_presence; row[COUNTS[STAMPS[1]][0]] = static_presence
        records.append(row)
    out = pd.DataFrame(records, index=targets.index)
    for name in STAMPS: out[name] = pd.to_datetime(out[name], utc=True)
    cutoff = pd.to_datetime(targets.scheduledTime, utc=True)-pd.Timedelta(minutes=30)
    stale.validate_additional_provenance(out, {"additional_provenance_columns":STAMPS, "provenance_count_columns":COUNTS}, cutoff)
    return out


def prepare(args):
    cohort = stale.development_cohort(args.cohort)
    if len(cohort) != stale.DEVELOPMENT_ROWS: raise ValueError("Incomplete frozen development cohort")
    base = stale.load_data(args.base_data)
    np.testing.assert_array_equal(np.sort(base.row_id), cohort.row_id)
    manifest = json.loads(Path(args.base_data).with_suffix(".manifest.json").read_text())
    last_label = pd.to_datetime(cohort.event_time, utc=True).max()
    parts, sources = [], []
    for name, targets in cohort.groupby("source_file",sort=True):
        path = Path(args.archive)/name
        bound = min(pd.to_datetime(targets.scheduledTime,utc=True).max()-pd.Timedelta(minutes=30),last_label)
        timestamp = bound.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23]+"Z"
        filters = [("departureDate","<=",str(targets.departureDate.max()))]
        schedules = pq.read_table(path,columns=stale_turnaround.STATIC_COLUMNS,filters=filters).to_pandas()
        observed = pq.read_table(path,columns=stale.KEY+["actualTime",TARGET],filters=filters+[("actualTime","<",timestamp)]).to_pandas()
        part = snapshot_features(targets, schedules, observed);part["row_id"] = targets.row_id
        parts.append(part)
        sources.append({"file":name,"sha256":sha256(path),"observed_rows":len(observed),"observation_read_before":timestamp})
        print(f"Origin context {name}: {len(targets)} development events",flush=True)
    extras = pd.concat(parts).set_index("row_id").reindex(base.row_id).reset_index(drop=True)
    frame = pd.concat([base.reset_index(drop=True),extras],axis=1)
    pd.testing.assert_frame_equal(frame[base.columns],base)
    dest = Path(args.output);frame.to_parquet(dest,index=False)
    result = {**manifest,"features":manifest["features"]+FEATURES,"features_sha256":sha256(dest),
              "feature_profile":"origin_terminating_context","base_data":str(args.base_data),"base_data_sha256":sha256(args.base_data),
              "additional_provenance_columns":manifest.get("additional_provenance_columns",[])+STAMPS,
              "provenance_count_columns":{**manifest.get("provenance_count_columns",{}),**COUNTS},
              "origin_context_sources":sources,"holdout_read":False,
              "origin_context_policy":"other long-distance dated runs ending at focal accepted route origin; arrivals planned15-720min before origin departure;30min turnaround projection; reverse route flag is not rolling-stock identity"}
    dest.with_suffix(".manifest.json").write_text(json.dumps(result,indent=2));stale.load_data(dest)
    print(f"Saved {len(frame)} development events and {len(result['features'])} predictors",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",default="data_archive");p.add_argument("--cohort",default="data/oulu_features.parquet")
    p.add_argument("--base-data",default="data/oulu_stale30_turnaround_development.parquet")
    p.add_argument("--output",default="data/oulu_stale30_origin_development.parquet")
    prepare(p.parse_args())

if __name__ == "__main__": main()
