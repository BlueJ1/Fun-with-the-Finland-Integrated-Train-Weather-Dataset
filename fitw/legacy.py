"""Prepare an explicit historical-code diagnostic, not a verified paper run."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .features import make_features
from .prepare import DEDUP_COLUMNS, READ_COLUMNS, sha256
from .spec import TARGET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/oulu_raw.parquet")
    parser.add_argument("--output", default="data/legacy")
    parser.add_argument("--archive", default="data_archive")
    args = parser.parse_args()
    raw = pd.read_parquet(args.raw)
    if "source_file" not in raw:
        raw = raw.rename(columns={"file": "source_file"})
    raw = raw[READ_COLUMNS + ["source_file"]].copy()
    valid = pd.to_datetime(raw.actualTime, utc=True, errors="coerce").notna()
    valid &= pd.to_datetime(raw.scheduledTime, utc=True, errors="coerce").notna()
    valid &= np.isfinite(raw[TARGET])
    selected = raw.loc[valid].drop_duplicates(DEDUP_COLUMNS).copy()
    # Historical extract_nested_data groups observations by train number inside
    # each month before station filtering. Recover the full-network first
    # appearance order from native files, rather than ordering the Oulu subset.
    pieces = []
    mappings = {}
    for filename, group in selected.groupby("source_file", sort=True):
        source = Path(args.archive) / filename
        train_numbers = pq.read_table(source, columns=["trainNumber"]).column(0).to_numpy()
        first_appearance = pd.unique(train_numbers).tolist()
        mappings[filename] = first_appearance
        rank = {number: i for i, number in enumerate(first_appearance)}
        group = group.assign(_rank=group.trainNumber.map(rank))
        if group._rank.isna().any():
            raise ValueError(f"Missing train order in {filename}")
        pieces.append(group.sort_values("_rank", kind="stable").drop(columns="_rank"))
    selected = pd.concat(pieces, ignore_index=True)
    selected["event_time"] = pd.to_datetime(selected.actualTime, utc=True)
    selected["row_id"] = np.arange(len(selected))
    features = make_features(selected, rules="author-legacy", temporal="author-legacy")
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "oulu_features.parquet"
    features.to_parquet(target, index=False)
    (dest / "manifest.json").write_text(json.dumps({
        "rows": len(features), "rules": "author-legacy", "timestamp": "actualTime UTC",
        "row_order": "sorted source month, full-network trainNumber first appearance, native order within group",
        "ordering_is_approximate": True,
        "deduplication": "same timestamp-aware projection as primary for matched cohort",
        "sha256": sha256(target),
    }, indent=2))
    (dest / "global_train_order.json").write_text(json.dumps(mappings, indent=2))
    print(f"Prepared {len(features):,} rows at {target}")


if __name__ == "__main__":
    main()
