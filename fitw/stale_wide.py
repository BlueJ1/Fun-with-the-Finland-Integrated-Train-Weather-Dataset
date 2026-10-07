"""Prepare old route context from all train categories without reading holdout labels."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from . import stale, stale_network
from .prepare import sha256
from .spec import TARGET


def prepare(args):
    cohort = stale.development_cohort(args.cohort)
    if len(cohort) != stale.DEVELOPMENT_ROWS:
        raise ValueError("Frozen development cohort is incomplete")
    last_label = pd.to_datetime(cohort.event_time, utc=True).max()
    pieces, sources = [], []
    for name, targets in cohort.groupby("source_file", sort=True):
        path = Path(args.archive) / name
        latest_snapshot = min(pd.to_datetime(targets.scheduledTime, utc=True).max()
                              - pd.Timedelta(minutes=30), last_label)
        timestamp = latest_snapshot.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"
        date_filter = [("departureDate", "<=", str(targets.departureDate.max()))]
        narrow_filter = date_filter + [("trainCategory", "=", "Long-distance"),
                                      ("trainNumber", "in", targets.trainNumber.unique().tolist())]
        schedules = pq.read_table(path, columns=stale.STATIC_COLUMNS,
                                  filters=narrow_filter).to_pandas().drop_duplicates(stale.KEY)
        observed = pq.read_table(path, columns=stale.KEY + ["actualTime", TARGET],
                                filters=narrow_filter + [("actualTime", "<", timestamp)]).to_pandas()
        context_schedules = pq.read_table(path, columns=stale.STATIC_COLUMNS,
                                         filters=date_filter).to_pandas().drop_duplicates(stale.KEY)
        context_observed = pq.read_table(path, columns=stale.KEY + ["actualTime", TARGET],
                                        filters=date_filter + [("actualTime", "<", timestamp)]).to_pandas()
        part = stale_network.snapshot_features(targets, schedules, observed,
                                               context_schedules=context_schedules,
                                               context_observed=context_observed)
        metadata = targets[["row_id", "event_time", "scheduledTime", "departureDate",
                            "trainNumber", "type", TARGET]].copy()
        pieces.append(pd.concat([metadata, part], axis=1))
        sources.append({"file": name, "sha256": sha256(path),
                        "observed_rows": len(observed), "context_observed_rows": len(context_observed),
                        "observation_read_before": timestamp})
        print(f"wide snapshot {name}: {len(targets)} queries, {len(context_observed)} old observations", flush=True)
    frame = pd.concat(pieces).sort_values("row_id").reset_index(drop=True).copy()
    frame["query_train_number"] = frame.trainNumber
    frame["trainNumber"] = frame.pop("feature_train_number")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    manifest = {"rows": len(frame), "development_row_id_upper_exclusive": stale.DEVELOPMENT_ROWS,
                "minimum_input_age_minutes": 30., "features": stale_network.NETWORK_FEATURES,
                "feature_profile": "wide", "features_sha256": sha256(destination),
                "holdout_read": False, "sources": sources,
                "prediction_time": "scheduledTime", "snapshot_time": "scheduledTime minus 30 minutes",
                "context_policy": "all train categories; old accepted plans and strictly old actual observations; focal dated run excluded",
                "timetable_policy": "acceptance strictly before snapshot; unknown or recent focal metadata masked",
                "weather_used": False, "post_snapshot_pending_status_used": False,
                "metadata_masked_rows": int((~frame.metadata_known).sum()),
                "provenance_limit": "actualTime proxies observation availability; archived acceptance does not version timetable revisions"}
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    stale.load_data(destination)
    print(f"Saved and age-checked {len(frame)} development rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data_archive")
    parser.add_argument("--cohort", default="data/oulu_features.parquet")
    parser.add_argument("--output", default="data/oulu_stale30_wide_development.parquet")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
