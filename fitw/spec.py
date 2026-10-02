"""Paper feature IDs, mapped to native parquet field names."""

OPERATIONAL = [
    "trainStopping", "month_sin", "month_cos", "hour_sin", "hour_cos",
    "day_week_sin", "day_week_cos", "train_id",
]
INSTANT = [
    "Air temperature", "Wind speed", "Wind direction", "Relative humidity",
    "Precipitation intensity", "Snow depth", "Pressure (msl)",
    "Horizontal visibility", "Cloud amount",
]
CATEGORY_NAMES = [
    "Normal Clear", "Blizzard", "Heavy Snow", "Extreme Cold", "Heavy Rain",
    "Freezing Rain", "Black Ice", "Dense Fog", "High Winds", "Extreme Heat",
]
CATEGORIES = ["weather_scenario_" + name for name in CATEGORY_NAMES]
TARGET = "differenceInMinutes"
PAPER_METRICS = {
    "full": {"r2": 0.70, "rmse": 9.9, "mae": 4.1, "wmape": 56.5},
    "instant": {"r2": 0.70, "rmse": 9.9, "mae": 4.1, "wmape": 56.5},
    "categories": {"r2": 0.78, "rmse": 8.5, "mae": 3.7, "wmape": 50.5},
}


def feature_sets(day_of_month=False, keep_cloud=False):
    base = OPERATIONAL + (["day_of_month"] if day_of_month else [])
    raw = INSTANT if keep_cloud else [f for f in INSTANT if f != "Cloud amount"]
    return {"full": base + raw + CATEGORIES, "instant": base + raw,
            "categories": base + CATEGORIES}
