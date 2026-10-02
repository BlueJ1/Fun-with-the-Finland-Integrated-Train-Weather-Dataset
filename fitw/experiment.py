"""Tune the three scenarios on development data and evaluate a fixed holdout."""

import argparse
import json
import os
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from scipy.stats import randint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterSampler, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from .prepare import sha256
from .scaling import WeatherScaler
from .spec import CATEGORIES, INSTANT, PAPER_METRICS, TARGET, feature_sets


def metrics(y, prediction):
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = np.abs(y - prediction)
    denom = np.abs(y).sum()
    signed = y.sum()
    return {"r2": float(r2_score(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "mae": float(mean_absolute_error(y, prediction)),
            "wmape": float(100 * error.sum() / denom) if denom else None,
            "wmape_signed_denominator": float(100 * error.sum() / signed) if signed else None}


def candidate_parameters(count, seed, legacy=False):
    space = {"n_estimators": randint(100, 400 if legacy else 401),
             "max_depth": randint(4, 8 if legacy else 9),
             "learning_rate": [0.01, 0.05, 0.1],
             "subsample": [0.7, 0.8, 0.9],
             "colsample_bytree": [0.7, 0.8, 1.0],
             "scale_pos_weight": [3.9, 4.9, 5.9]}
    return [{k: v.item() if hasattr(v, "item") else v for k, v in p.items()}
            for p in ParameterSampler(space, n_iter=count, random_state=seed)]


def model(params, seed, jobs):
    # XGBoost's shared regression objective applies this parameter to y == 1.
    # Preserve the paper/source search literally; it is not delay-class balancing.
    return XGBRegressor(**params, objective="reg:squarederror", tree_method="hist",
                        random_state=seed, n_jobs=jobs)


def split_indices(df, protocol):
    indices = np.arange(len(df))
    if protocol == "chronological":
        cutoff = int(0.8 * len(df))
        return indices[:cutoff], indices[cutoff:]
    return train_test_split(indices, test_size=0.2, random_state=42)


def describe_split(df, train, test, folds, protocol):
    def partition(idx):
        return {"rows": len(idx), "first_time": str(df.iloc[idx].event_time.min()),
                "last_time": str(df.iloc[idx].event_time.max()),
                "target_mean": float(df.iloc[idx][TARGET].mean()),
                "target_std": float(df.iloc[idx][TARGET].std(ddof=0)),
                "mean_absolute_target": float(df.iloc[idx][TARGET].abs().mean())}
    out = {"protocol": protocol, "development": partition(train), "test": partition(test), "folds": []}
    if protocol == "chronological":
        assert df.iloc[train].event_time.max() < df.iloc[test].event_time.min()
    for i, (tr, va) in enumerate(folds):
        if protocol == "chronological":
            assert df.iloc[train[tr]].event_time.max() < df.iloc[train[va]].event_time.min()
        out["folds"].append({"fold": i + 1, "train": partition(train[tr]), "validation": partition(train[va])})
    keys = lambda ix: set(zip(df.iloc[ix].departureDate, df.iloc[ix].trainNumber))
    out["dated_runs_shared_by_development_and_test"] = len(keys(train) & keys(test))
    return out


def monthly_impute(df):
    """Legacy author preprocessing, explicitly isolated to the legacy diagnostic.

    It uses all rows in each source month before splitting, as in the historical
    code. Columns with >30% missing are omitted for that month, not filled.
    """
    out = df.copy()
    group = "source_file" if "source_file" in df else "file"
    for _, idx in df.groupby(group, sort=False).groups.items():
        for c in INSTANT:
            values = df.loc[idx, c]
            if values.isna().mean() > 0.3:
                out.loc[idx, c] = np.nan
            elif values.notna().any():
                out.loc[idx, c] = values.fillna(values.median())
    return out


def run(args):
    output = Path(args.output) / args.scenario
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    df = pd.read_parquet(data_path).sort_values("row_id").reset_index(drop=True)
    if args.protocol == "historical-random":
        df = monthly_impute(df)
    features = feature_sets(args.day_of_month, args.keep_cloud)[args.scenario]
    X = df[features].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    dev, test = split_indices(df, args.protocol)
    splitter = TimeSeriesSplit(n_splits=5) if args.protocol == "chronological" else KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(dev))
    split = describe_split(df, dev, test, folds, args.protocol)
    params = candidate_parameters(args.candidates, args.seed, args.protocol == "historical-random")
    scoring = "mae" if args.protocol == "historical-random" else "rmse"
    weather_only_scaling = getattr(args, "legacy_weather_scaling", False)
    if weather_only_scaling and args.protocol != "historical-random":
        raise ValueError("Legacy weather-only scaling belongs only to historical diagnostics")
    weather_indices = [i for i, feature in enumerate(features) if feature in INSTANT]
    make_scaler = lambda: WeatherScaler(weather_indices) if weather_only_scaling else RobustScaler()
    config = {"scenario": args.scenario, "protocol": args.protocol, "features": features,
              "feature_count": len(features), "candidate_count": args.candidates,
              "seed": args.seed, "scoring": scoring, "jobs_per_model": args.jobs,
              "parameter_policy": "literal scale_pos_weight, including target-equals-one weighting",
              "data_sha256": sha256(data_path), "parameters": params,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "scipy": scipy.__version__,
                           "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
              "preprocessing": "fold-fitted RobustScaler; XGBoost native missing values"}
    if args.protocol == "historical-random":
        config["preprocessing"] = "legacy per-month medians, development-fitted RobustScaler"
        if weather_only_scaling:
            config["preprocessing"] = "legacy per-month medians, development-fitted RobustScaler on raw weather only"
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Config mismatch in {output}; choose a new output directory")
    config_path.write_text(json.dumps(config, indent=2))
    (output / "split.json").write_text(json.dumps(split, indent=2))
    pd.DataFrame({"row_id": df.row_id, "partition": np.where(np.isin(np.arange(len(df)), dev), "development", "test")}).to_csv(output / "split_membership.csv", index=False)
    prepared = []
    shared_scaler = make_scaler().fit(X[dev]) if args.protocol == "historical-random" else None
    for tr, va in folds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scaler = shared_scaler or make_scaler().fit(X[dev[tr]])
        prepared.append((scaler.transform(X[dev[tr]]), y[dev[tr]], scaler.transform(X[dev[va]]), y[dev[va]]))
    checkpoint = output / "search.jsonl"
    existing = [json.loads(line) for line in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
    start = time.monotonic()
    for index in range(len(existing), len(params)):
        losses = []
        rmse_losses = []
        for Xt, yt, Xv, yv in prepared:
            fitted = model(params[index], args.seed, args.jobs).fit(Xt, yt)
            pred = fitted.predict(Xv)
            rmse = float(np.sqrt(mean_squared_error(yv, pred)))
            rmse_losses.append(rmse)
            losses.append(float(mean_absolute_error(yv, pred)) if scoring == "mae" else rmse)
        record = {"candidate": index + 1, "params": params[index], "fold_loss": losses,
                  "mean_cv_loss": float(np.mean(losses)), "fold_rmse": rmse_losses,
                  "mean_cv_rmse": float(np.mean(rmse_losses))}
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        existing.append(record)
        if (index + 1) % 5 == 0 or index == 0:
            best = min(existing, key=lambda r: r["mean_cv_loss"])
            print(f"{args.scenario} {index+1}/{len(params)} best CV {scoring}={best['mean_cv_loss']:.4f}; elapsed={time.monotonic()-start:.1f}s", flush=True)
    # Candidate-count curves select solely by CV. Each prefix uses the same
    # sampled sequence, equivalent to fresh RandomizedSearchCV with seed=42.
    curves, fitted_by_candidate = [], {}
    budgets = list(range(10, args.candidates + 1, 10))
    if args.candidates not in budgets:
        budgets.append(args.candidates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        final_scaler = make_scaler().fit(X[dev])
    Xdev, Xtest = final_scaler.transform(X[dev]), final_scaler.transform(X[test])
    for budget in budgets:
        best = min(existing[:budget], key=lambda r: r["mean_cv_loss"])
        chosen = best["candidate"]
        if chosen not in fitted_by_candidate:
            fitted = model(best["params"], args.seed, args.jobs).fit(Xdev, y[dev])
            pred = fitted.predict(Xtest)
            fitted_by_candidate[chosen] = (fitted, pred)
        fitted, pred = fitted_by_candidate[chosen]
        curves.append({"search_candidates": budget, "best_candidate": chosen,
                       "cv_loss": best["mean_cv_loss"], **metrics(y[test], pred)})
    pd.DataFrame(curves).to_csv(output / "search_budget_curve.csv", index=False)
    best = min(existing, key=lambda r: r["mean_cv_loss"])
    fitted, pred = fitted_by_candidate[best["candidate"]]
    fitted.save_model(output / "model.ubj")
    joblib.dump(final_scaler, output / "scaler.joblib")
    predictions = df.iloc[test][["row_id", "event_time", "departureDate", "trainNumber", "type", TARGET]].copy()
    predictions["prediction"] = pred
    predictions.to_parquet(output / "predictions.parquet", index=False)
    pd.DataFrame({"feature": features, "importance": fitted.feature_importances_}).sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    measured = metrics(y[test], pred)
    baselines = {"development_mean": metrics(y[test], np.full(len(test), y[dev].mean())),
                 "development_median": metrics(y[test], np.full(len(test), np.median(y[dev]))),
                 "zero_delay": metrics(y[test], np.zeros(len(test)))}
    result = {"scenario": args.scenario, "protocol": args.protocol,
              "features": features, "feature_count": len(features), "rows": len(df),
              "development_rows": len(dev), "test_rows": len(test), "test_metrics": measured,
              "best_candidate": best, "paper_approximate_metrics": PAPER_METRICS[args.scenario],
              "difference_from_paper": {k: measured[k] - v if measured[k] is not None else None
                                        for k, v in PAPER_METRICS[args.scenario].items()},
              "baselines": baselines, "test_curve_selection_is_diagnostic_only": True}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"scenario": args.scenario, "metrics": measured, "best": best["params"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/oulu_features.parquet")
    parser.add_argument("--output", default="results/chronological")
    parser.add_argument("--scenario", choices=["full", "instant", "categories"], required=True)
    parser.add_argument("--protocol", choices=["chronological", "historical-random"], default="chronological")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--day-of-month", action="store_true")
    parser.add_argument("--keep-cloud", action="store_true")
    parser.add_argument("--legacy-weather-scaling", action="store_true",
                        help="Match author scaling of raw weather only, leaving operational/category inputs unchanged")
    args = parser.parse_args()
    if args.candidates < 1 or args.jobs < 1:
        parser.error("Candidate count and job count must be positive")
    run(args)


if __name__ == "__main__":
    main()
