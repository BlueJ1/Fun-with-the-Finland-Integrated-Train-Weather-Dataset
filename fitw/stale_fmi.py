"""Separate development experiment with lagged FMI measurement-time proxies."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import stale
from .prepare import sha256

PARAMETERS = ["t2m", "ws_10min", "wg_10min", "wd_10min", "rh", "td", "r_1h", "snow_aws", "p_sea", "vis"]
FMISID = 101799
LAG_MINUTES = 60
MAX_STALE_MINUTES = 120
NS = {"om": "http://www.opengis.net/om/2.0", "gml": "http://www.opengis.net/gml/3.2",
      "wml": "http://www.opengis.net/waterml/2.0", "wfs": "http://www.opengis.net/wfs/2.0"}
XLINK = "{http://www.w3.org/1999/xlink}href"
ROLLING = [("t2m", 6, "mean"), ("t2m", 6, "min"), ("t2m", 6, "max"),
           ("t2m", 24, "min"), ("t2m", 24, "max"), ("ws_10min", 6, "max"),
           ("wg_10min", 6, "max"), ("r_1h", 6, "sum"), ("r_1h", 24, "sum"),
           ("snow_aws", 24, "change")]
EXTRA_FEATURES = ([f"fmi_{p}" for p in PARAMETERS] + [f"fmi_{p}_age_minutes" for p in PARAMETERS]
                  + [f"fmi_{p}_{h}h_{stat}" for p, h, stat in ROLLING]
                  + [f"fmi_{p}_{h}h_count" for p, h in dict.fromkeys((p, h) for p, h, _ in ROLLING)]
                  + ["fmi_freezing_precipitation", "fmi_wind_east", "fmi_wind_north"])
PROVENANCE_COLUMNS = ["fmi_latest_measurement_time"]
PROVENANCE_COUNT_COLUMNS = {name: ["fmi_input_presence_count"] for name in PROVENANCE_COLUMNS}
PROVENANCE_LAG_MINUTES = {"fmi_latest_measurement_time": LAG_MINUTES}


def parse_response(content):
    """Read measurement times, never substitute response/result times."""
    root = ET.fromstring(content)
    if not root.tag.endswith("FeatureCollection"):
        raise ValueError("FMI response is not a feature collection: " + content.decode()[:300])
    records = []
    for member in root.findall("wfs:member", NS):
        observation = member[0]
        station = int(observation.find(".//gml:identifier", NS).text)
        if station != FMISID:
            raise ValueError("Unexpected FMI station")
        prop = observation.find("om:observedProperty", NS).attrib[XLINK]
        parameter = urllib.parse.parse_qs(urllib.parse.urlparse(prop).query)["param"][0]
        if parameter not in PARAMETERS:
            raise ValueError("Unexpected FMI variable: " + parameter)
        location = observation.find(".//gml:pos", NS).text.split()
        name = observation.find(".//gml:name", NS).text
        for point in observation.findall(".//wml:MeasurementTVP", NS):
            timestamp = pd.Timestamp(point.find("wml:time", NS).text)
            if timestamp.tzinfo is None:
                raise ValueError("FMI measurement lacks explicit timezone")
            records.append({"measurement_time": timestamp.tz_convert("UTC"), "parameter": parameter,
                            "value": float(point.find("wml:value", NS).text), "fmisid": station,
                            "station_name": name, "latitude": float(location[0]), "longitude": float(location[1])})
    return pd.DataFrame(records, columns=["measurement_time", "parameter", "value", "fmisid", "station_name", "latitude", "longitude"])


def snapshot_features(targets, observations):
    """Use measurements + assumed one-hour lag strictly before the old snapshot.

    The two-hour carry limit is relative to the lag-adjusted measurement deadline.
    Rolling summaries need >=75% hourly coverage (5/6 or18/24). Counts remain
    informative below that threshold; their contributing times are also audited.
    """
    query = targets.copy()
    predictions = pd.to_datetime(query["prediction_time"] if "prediction_time" in query else query.scheduledTime, utc=True)
    cutoffs = predictions - pd.Timedelta(minutes=30)
    deadlines = (cutoffs - pd.Timedelta(minutes=LAG_MINUTES)).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    weather = observations.copy()
    weather["measurement_time"] = pd.to_datetime(weather.measurement_time, utc=True)
    if weather.measurement_time.isna().any():
        raise ValueError("Weather observation has no measurement time")
    if "fmisid" in weather and (weather.fmisid != FMISID).any():
        raise ValueError("Unexpected FMI station")
    weather["value"] = pd.to_numeric(weather.value, errors="raise")
    # Official guidance defines negative precipitation as no rain. The sampled
    # negative snow reading has no established sentinel meaning: keep missing.
    rain = weather.parameter.eq("r_1h") & weather.value.eq(-1)
    weather.loc[rain, "value"] = 0.
    weather.loc[weather.parameter.isin(["snow_aws", "r_1h", "ws_10min", "wg_10min", "vis"]) & weather.value.lt(0), "value"] = np.nan
    duplicates = weather.duplicated(["measurement_time", "parameter"], keep=False)
    if duplicates.any() and weather[duplicates].groupby(["measurement_time", "parameter"]).value.nunique(dropna=False).gt(1).any():
        raise ValueError("Conflicting duplicate FMI observations")
    weather = weather.drop_duplicates(["measurement_time", "parameter"]).sort_values("measurement_time")
    series = {}
    for parameter in PARAMETERS:
        rows = weather[weather.parameter.eq(parameter) & np.isfinite(weather.value)]
        series[parameter] = (rows.measurement_time.astype("datetime64[ns, UTC]").astype("int64").to_numpy(), rows.value.to_numpy(float))
    minute = 60_000_000_000
    records = []
    for deadline in deadlines:
        record = {name: np.nan for name in EXTRA_FEATURES}
        latest_used, presence = None, 0
        for parameter, (times, values) in series.items():
            stop = np.searchsorted(times, deadline, side="left")
            if stop and deadline - times[stop - 1] <= MAX_STALE_MINUTES * minute:
                record[f"fmi_{parameter}"] = values[stop - 1]
                record[f"fmi_{parameter}_age_minutes"] = (deadline - times[stop - 1]) / minute + LAG_MINUTES + 30
                latest_used = max(latest_used or times[stop - 1], times[stop - 1])
                presence += 1
        for parameter, hours in dict.fromkeys((p, h) for p, h, _ in ROLLING):
            times, values = series[parameter]
            start, stop = np.searchsorted(times, [deadline - hours * 60 * minute, deadline], side="left")
            count = stop - start
            record[f"fmi_{parameter}_{hours}h_count"] = count
            if count:
                latest_used = max(latest_used or times[stop - 1], times[stop - 1])
                presence += count
            enough = count >= int(np.ceil(hours * .75)) and count > 0 and deadline - times[stop - 1] <= MAX_STALE_MINUTES * minute
            if enough:
                window = values[start:stop]
                for p, h, stat in ROLLING:
                    if (p, h) == (parameter, hours):
                        record[f"fmi_{p}_{h}h_{stat}"] = window[-1] - window[0] if stat == "change" else float(getattr(np, stat)(window))
        temperature, rain = record["fmi_t2m"], record["fmi_r_1h"]
        if np.isfinite(temperature) and np.isfinite(rain):
            record["fmi_freezing_precipitation"] = rain if temperature <= 1 else 0.
        speed, direction = record["fmi_ws_10min"], record["fmi_wd_10min"]
        if np.isfinite(speed) and np.isfinite(direction):
            record["fmi_wind_east"] = -speed * np.sin(np.deg2rad(direction))
            record["fmi_wind_north"] = -speed * np.cos(np.deg2rad(direction))
        record["fmi_input_presence_count"] = presence
        record[PROVENANCE_COLUMNS[0]] = pd.Timestamp(latest_used, unit="ns", tz="UTC") if latest_used is not None else pd.NaT
        records.append(record)
    out = pd.DataFrame(records, index=targets.index)
    for name in PROVENANCE_COLUMNS:
        out[name] = pd.to_datetime(out[name], utc=True)
    stale.validate_additional_provenance(out, {"additional_provenance_columns": PROVENANCE_COLUMNS,
                                             "provenance_count_columns": PROVENANCE_COUNT_COLUMNS,
                                             "additional_provenance_lag_minutes": PROVENANCE_LAG_MINUTES}, pd.Series(cutoffs.to_numpy(), index=targets.index))
    if (out.fmi_latest_measurement_time >= pd.Series(cutoffs.to_numpy(), index=targets.index) - pd.Timedelta(minutes=LAG_MINUTES)).any():
        raise ValueError("FMI measurement violates conservative assumed lag")
    return out


def fetch_week(start, end, cache):
    iso = lambda value: value.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"service": "WFS", "version": "2.0.0", "request": "getFeature",
              "storedquery_id": "fmi::observations::weather::timevaluepair", "fmisid": FMISID,
              "starttime": iso(start), "endtime": iso(end), "timestep": 60, "timezone": "UTC", "parameters": ",".join(PARAMETERS)}
    url = "https://opendata.fmi.fi/wfs?" + urllib.parse.urlencode(params)
    path = cache / (start.strftime("%Y%m%dT%H%M") + ".xml")
    sidecar = path.with_suffix(".json")
    if path.exists() and sidecar.exists():
        meta = json.loads(sidecar.read_text())
        if meta.get("url") != url or meta.get("sha256") != sha256(path):
            raise ValueError("FMI cache query/hash mismatch")
        return parse_response(path.read_bytes()), meta
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                content = response.read()
            frame = parse_response(content)
            if len(frame) and not frame.measurement_time.between(start, end).all():
                raise ValueError("FMI returned out-of-request times")
            path.write_bytes(content)
            meta = {"url": url, "sha256": sha256(path), "file": str(path), "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "start": iso(start), "end": iso(end), "parameter_values": len(frame), "failed": False}
            sidecar.write_text(json.dumps(meta, indent=2))
            return frame, meta
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = str(error)
            retry = error.headers.get("Retry-After") if isinstance(error, urllib.error.HTTPError) else None
            if attempt < 3:
                if retry and retry.isdigit():
                    delay = float(retry)
                elif retry:
                    try:
                        delay = max(0., (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds())
                    except (TypeError, ValueError):
                        delay = min(30, 2 ** (attempt + 1))
                else:
                    delay = min(30, 2 ** (attempt + 1))
                time.sleep(delay)
    return parse_response(b'<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0"/>'), {"url": url, "start": iso(start), "end": iso(end), "failed": True, "error": last_error}


def prepare(args):
    base = stale.load_data(args.base_data)
    if len(base) != stale.DEVELOPMENT_ROWS or base.row_id.max() >= stale.DEVELOPMENT_ROWS:
        raise ValueError("Expected complete frozen development data")
    manifest = json.loads(Path(args.base_data).with_suffix(".manifest.json").read_text())
    latest = pd.to_datetime(base.observation_cutoff, utc=True).max()
    end = latest.floor("h")
    start = (pd.to_datetime(base.observation_cutoff, utc=True).min() - pd.Timedelta(hours=25)).floor("h")
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    metadata_sources = []
    for parameter in PARAMETERS:
        url = "https://opendata.fmi.fi/meta?" + urllib.parse.urlencode({"observableProperty": "observation", "param": parameter, "language": "eng"})
        path = cache / ("metadata_" + parameter + ".xml")
        if not path.exists():
            with urllib.request.urlopen(url, timeout=45) as response:
                path.write_bytes(response.read())
        root = ET.parse(path).getroot()
        if not root.tag.endswith("ObservableProperty"):
            raise ValueError("FMI variable metadata is invalid")
        metadata_sources.append({"parameter": parameter, "url": url, "file": str(path), "sha256": sha256(path)})
    windows = []
    while start < end:
        stop = min(start + pd.Timedelta(days=7), end)
        windows.append((start, stop)); start = stop
    pieces, sources = [], []
    with ThreadPoolExecutor(max_workers=min(args.workers, 3)) as executor:
        futures = {executor.submit(fetch_week, start, stop, cache): (start, stop) for start, stop in windows}
        for completed, future in enumerate(as_completed(futures), 1):
            frame, source = future.result()
            pieces.append(frame); sources.append(source)
            print(f"FMI {completed}/{len(windows)} {source['start']} {source.get('parameter_values', 0)} values failed={source['failed']}", flush=True)
    weather = pd.concat(pieces, ignore_index=True)
    if len(weather) and (weather.measurement_time > latest).any():
        raise ValueError("FMI observation after frozen development cutoff")
    weather = weather.sort_values(["measurement_time", "parameter"]).reset_index(drop=True)
    raw_path = cache / "observations.parquet"; weather.to_parquet(raw_path, index=False)
    extras = snapshot_features(base, weather)
    out = pd.concat([base, extras], axis=1)
    pd.testing.assert_frame_equal(out[base.columns], base)
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(destination, index=False)
    coverage = (weather.drop_duplicates(["measurement_time", "parameter"]).assign(month=lambda f: f.measurement_time.dt.strftime("%Y-%m"))
                .groupby(["month", "parameter"]).value.agg(["size", "count"]).reset_index().to_dict("records"))
    updated = {**manifest, "feature_profile": "turnaround_fmi", "features": manifest["features"] + EXTRA_FEATURES,
               "features_sha256": sha256(destination), "base_data": str(args.base_data), "base_data_sha256": sha256(args.base_data),
               "additional_provenance_columns": list(dict.fromkeys(manifest.get("additional_provenance_columns", []) + PROVENANCE_COLUMNS)),
               "provenance_count_columns": {**manifest.get("provenance_count_columns", {}), **PROVENANCE_COUNT_COLUMNS},
               "additional_provenance_lag_minutes": {**manifest.get("additional_provenance_lag_minutes", {}), **PROVENANCE_LAG_MINUTES},
               "fmi": {"attribution": "Finnish Meteorological Institute", "license": "CC BY4.0", "station_fmisid": FMISID,
                       "minimum_snapshot_age_minutes": 30, "additional_assumed_release_lag_minutes": LAG_MINUTES,
                       "release_lag_is_assumption": True, "historical_publication_or_revision_availability_certified": False,
                       "max_point_carry_after_lag_minutes": MAX_STALE_MINUTES, "rolling_minimum_fraction": .75,
                       "latest_permitted_measurement": latest.isoformat(), "raw_observations_sha256": sha256(raw_path),
                       "variable_metadata": metadata_sources,
                       "sources": sorted(sources, key=lambda s: s["start"]), "monthly_coverage": coverage,
                       "missing_policy": "No event dropping or future fill. Negative snow missing; precipitation -1 mapped0; failed weeks missing.",
                       "provenance_limit": "Measurement times plus assumed60min lag are availability proxies. Current archive values may include later quality-control corrections; no historical release/revision versions are available in this interface."},
               "holdout_read": False, "weather_used": True}
    destination.with_suffix(".manifest.json").write_text(json.dumps(updated, indent=2))
    stale.load_data(destination)
    print(f"Saved {len(out)} rows, {len(EXTRA_FEATURES)} weather features; failed weeks={sum(s['failed'] for s in sources)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", default="data/oulu_stale30_turnaround_development.parquet")
    parser.add_argument("--output", default="data/oulu_stale30_fmi_development.parquet")
    parser.add_argument("--cache", default="data/fmi_pellonpaa_development")
    parser.add_argument("--workers", default=3, type=int)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
