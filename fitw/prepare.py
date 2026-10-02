"""Extract Oulu data and generate deterministic, timestamped feature tables."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .features import make_features
from .spec import INSTANT, TARGET, CATEGORIES

READ_COLUMNS = [
    "departureDate", "trainNumber", "trainCategory", "trainType", "stationName",
    "stationShortCode", "type", "scheduledTime", "actualTime", TARGET,
    "differenceInMinutes_offset", "differenceInMinutes_eachStation_offset",
    "cancelled", "stop_cancelled", "trainStopping", "commercialStop",
    "Air temperature", "Wind speed", "Gust speed", "Wind direction",
    "Relative humidity", "Dew-point temperature", "Precipitation amount",
    "Precipitation intensity", "Snow depth", "Pressure (msl)",
    "Horizontal visibility", "Cloud amount",
]
DEDUP_COLUMNS = [
    "actualTime", "trainNumber", "trainStopping", "commercialStop", TARGET,
    "differenceInMinutes_offset", "differenceInMinutes_eachStation_offset", "cancelled",
] + INSTANT + ["Gust speed", "Dew-point temperature"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(args.archive).glob("matched_data_flat_*.parquet")
                   if 2018 <= int(p.stem.split("_")[-2]) <= 2024)
    if len(files) != 84:
        raise ValueError(f"Expected 84 months for 2018-2024, found {len(files)}")
    manifest, pieces = [], []
    for path in files:
        table = pq.read_table(path, columns=READ_COLUMNS,
                              filters=[("stationShortCode", "=", "OL"),
                                       ("trainCategory", "=", "Long-distance")])
        frame = table.to_pandas()
        frame["source_file"] = path.name
        pieces.append(frame)
        manifest.append({"path": str(path), "bytes": path.stat().st_size,
                         "sha256": sha256(path), "selected_rows": len(frame)})
    raw = pd.concat(pieces, ignore_index=True)
    raw.to_parquet(dest / "oulu_raw.parquet", index=False)
    audit = {"raw_rows": len(raw), "paper_rows": 101146, "input_files": manifest}
    valid = pd.to_datetime(raw.actualTime, utc=True, errors="coerce").notna()
    valid &= pd.to_datetime(raw.scheduledTime, utc=True, errors="coerce").notna()
    valid &= np.isfinite(raw[TARGET])
    selected = raw.loc[valid].copy()
    audit["rows_after_timestamp_target_filter"] = len(selected)
    duplicates = selected.duplicated(subset=DEDUP_COLUMNS, keep="first")
    audit["projected_duplicate_rows"] = selected.loc[duplicates, [
        "departureDate", "trainNumber", "type", "actualTime", TARGET,
    ]].to_dict(orient="records")
    selected = selected.loc[~duplicates].copy()
    selected["event_time"] = pd.to_datetime(selected[args.timestamp], utc=True)
    selected = selected.sort_values(["event_time", "trainNumber", "type"], kind="stable").reset_index(drop=True)
    selected["row_id"] = np.arange(len(selected))
    features = make_features(selected, timestamp=args.timestamp, temporal=args.temporal,
                             rules=args.rules)
    output_path = dest / "oulu_features.parquet"
    features.to_parquet(output_path, index=False)
    audit.update({"rows": len(features), "matches_paper_count": len(features) == 101146,
                  "timestamp": args.timestamp, "temporal": args.temporal, "category_rules": args.rules,
                  "train_id": "trainNumber", "boundary_policy": "2018-2024 departure-month files; retain overnight events",
                  "first_event": str(features.event_time.min()), "last_event": str(features.event_time.max()),
                  "features_sha256": sha256(output_path),
                  "category_counts": {c: int(features[c].sum()) for c in CATEGORIES},
                  "weather_missing": {c: int(features[c].isna().sum()) for c in INSTANT},
                  "target_description": features[TARGET].describe().to_dict()})
    (dest / "manifest.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k not in ("input_files", "target_description")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data_archive")
    parser.add_argument("--output", default="data")
    parser.add_argument("--timestamp", choices=["actualTime", "scheduledTime"], default="actualTime")
    parser.add_argument("--temporal", choices=["author-legacy", "author-current", "paper"], default="author-legacy")
    parser.add_argument("--rules", choices=["paper", "author-legacy", "author-current"], default="paper")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
