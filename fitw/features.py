"""Fixed transformations; no statistic is estimated from test rows."""

import numpy as np
import pandas as pd

from .spec import CATEGORIES, CATEGORY_NAMES


def weather_categories(df, rules="paper"):
    """Apply Table II severity priority. Missing comparisons are False.

    The current author code changes black ice thresholds and priority;
    it is available explicitly for sensitivity analysis, not silently substituted.
    """
    t, p, i, w, g, h, d, s, v = (
        df[c].to_numpy(dtype=float) for c in [
            "Air temperature", "Precipitation amount", "Precipitation intensity",
            "Wind speed", "Gust speed", "Relative humidity", "Dew-point temperature",
            "Snow depth", "Horizontal visibility",
        ]
    )
    near_freezing = (t > -2) & (t < 2)
    wet = (p > 0) | (i > 0)
    criteria = [
        ("Blizzard", (t < 0) & ((i > 1) | (p > 3)) & ((w > 10) | (g > 15)) & (v < 1000)),
        ("Heavy Snow", (t < 0) & ((i > 2) | (p > 5)) & (s > 0)),
        ("Extreme Cold", t < -20),
        ("Heavy Rain", (t > 2) & ((i > 4) | (p > 10))),
        ("Freezing Rain", near_freezing & wet),
        ("Black Ice", near_freezing & (h > 80) & ((d - t) < 2) & (p > 0)),
        ("Dense Fog", (p <= 0.1) & (v < 1000) & (h > 95)),
        ("High Winds", (w > 15) | (g > 20)),
        ("Extreme Heat", t > 30),
    ]
    if rules == "author-current":
        near_freezing = (t >= -2) & (t <= 2)
        criteria[4:6] = [
            ("Black Ice", near_freezing & (h > 80) & (np.abs(d - t) < 2) & (p <= 0.5)),
            ("Freezing Rain", near_freezing & wet),
        ]
    elif rules == "author-legacy":
        criteria[4] = ("Freezing Rain", (t >= -2) & (t <= 2) & wet)
        criteria[5] = ("Black Ice", (t >= -2) & (t <= 2) & (h > 80) & (np.abs(d - t) < 2) & (p > 0))
    elif rules != "paper":
        raise ValueError(f"Unknown category rules: {rules}")
    names = np.full(len(df), "Normal Clear", dtype=object)
    unassigned = np.ones(len(df), dtype=bool)
    for name, mask in criteria:
        selected = mask & unassigned
        names[selected] = name
        unassigned[selected] = False
    encoded = pd.DataFrame({col: (names == name).astype(np.float32)
                            for col, name in zip(CATEGORIES, CATEGORY_NAMES)}, index=df.index)
    assert (encoded.sum(axis=1) == 1).all()
    return encoded


def make_features(df, timestamp="actualTime", temporal="author-legacy", rules="paper", train_id="number"):
    out = df.copy()
    event = pd.to_datetime(out[timestamp], utc=True, errors="coerce")
    if event.isna().any():
        raise ValueError("Missing event timestamps must be removed before feature construction")
    month = event.dt.month.to_numpy()
    weekday = (event.dt.dayofweek.to_numpy() + 1) % 7 + 1  # Sunday=1, Monday=2
    hour = event.dt.hour.to_numpy() + event.dt.minute.to_numpy() / 60
    if temporal == "author-current":
        month = month - 1
        weekday = weekday - 1
    elif temporal not in ("paper", "author-legacy"):
        raise ValueError(temporal)
    for name, values, period in [("month", month, 12), ("hour", hour, 24),
                                  ("day_week", weekday, 7)]:
        out[name + "_sin"] = np.sin(2 * np.pi * values / period)
        out[name + "_cos"] = np.cos(2 * np.pi * values / period)
    out["day_of_month"] = event.dt.day
    out["trainStopping"] = out["trainStopping"].astype(float)
    if train_id == "number":
        out["train_id"] = out["trainNumber"].astype(float)
    else:
        raise ValueError(f"Unsupported train_id scheme: {train_id}")
    return pd.concat([out, weather_categories(out, rules)], axis=1)
