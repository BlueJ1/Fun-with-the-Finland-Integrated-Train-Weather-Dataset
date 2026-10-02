"""Independently recompute saved results and verify reloadable model artifacts."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .experiment import metrics, monthly_impute
from .prepare import sha256
from .scaling import WeatherScaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Verify one profile using its matching interpreter")
    args = parser.parse_args()
    root = Path("results")
    checked = 0
    cohorts = {}
    delegated = []
    for result_path in sorted(root.glob("*/*/result.json")):
        directory = result_path.parent
        profile = directory.parent.name
        if args.profile is not None and profile != args.profile:
            continue
        if args.profile is None and profile.endswith("-author"):
            if profile not in delegated:
                subprocess.run([str(Path(".venv-author/bin/python").absolute()), "-m", "fitw.verify",
                                "--profile", profile], check=True)
                delegated.append(profile)
            continue
        config = json.loads((directory / "config.json").read_text())
        assert config["parameter_policy"].startswith("literal scale_pos_weight")
        result = json.loads(result_path.read_text())
        path = Path("data/legacy/oulu_features.parquet") if config["protocol"] == "historical-random" else Path("data/oulu_features.parquet")
        if profile not in cohorts:
            frame = pd.read_parquet(path).sort_values("row_id").reset_index(drop=True)
            if config["protocol"] == "historical-random":
                frame = monthly_impute(frame)
            cohorts[profile] = frame
        df = cohorts[profile]
        assert config["data_sha256"] == sha256(path)
        search = [json.loads(line) for line in (directory / "search.jsonl").read_text().splitlines()]
        assert len(search) == config["candidate_count"] == 100
        assert len({r["candidate"] for r in search}) == 100
        assert [r["candidate"] for r in search] == list(range(1, 101))
        assert [r["params"] for r in search] == config["parameters"]
        assert all(len(r["fold_loss"]) == 5 for r in search)
        for record in search:
            np.testing.assert_allclose(np.mean(record["fold_loss"]), record["mean_cv_loss"],
                                       rtol=1e-12, atol=1e-12)
        expected_best = min(search, key=lambda r: r["mean_cv_loss"])
        assert result["best_candidate"] == expected_best
        membership = pd.read_csv(directory / "split_membership.csv")
        assert membership.row_id.is_unique and len(membership) == len(df) == 101146
        test_ids = membership.loc[membership.partition == "test", "row_id"]
        train_ids = membership.loc[membership.partition == "development", "row_id"]
        assert not set(test_ids) & set(train_ids)
        predictions = pd.read_parquet(directory / "predictions.parquet")
        assert set(predictions.row_id) == set(test_ids)
        assert len(predictions) == 20230
        if config["protocol"] == "chronological":
            assert df.iloc[train_ids].event_time.max() < df.iloc[test_ids].event_time.min()
        actual = df.set_index("row_id").loc[predictions.row_id]
        np.testing.assert_array_equal(actual.differenceInMinutes, predictions.differenceInMinutes)
        scaler = joblib.load(directory / "scaler.joblib")
        estimator = XGBRegressor()
        estimator.load_model(directory / "model.ubj")
        saved_objective = json.loads(estimator.get_booster().save_config())["learner"]["objective"]
        np.testing.assert_allclose(float(saved_objective["reg_loss_param"]["scale_pos_weight"]),
                                   result["best_candidate"]["params"]["scale_pos_weight"], rtol=1e-6)
        estimator.set_params(n_jobs=2)
        X = actual[config["features"]].to_numpy(dtype=np.float32)
        reloaded = estimator.predict(scaler.transform(X))
        np.testing.assert_allclose(reloaded, predictions.prediction, rtol=1e-7, atol=1e-6)
        recomputed = metrics(predictions.differenceInMinutes, predictions.prediction)
        for key, value in recomputed.items():
            np.testing.assert_allclose(value, result["test_metrics"][key], rtol=1e-10, atol=1e-10)
        assert (root / "figures" / f"{profile}_search_curve.png").stat().st_size > 10000
        checked += 1
    for profile in delegated:
        child = json.loads((root / "verification" / f"{profile}.json").read_text())
        checked += child["verified_scenarios"]
    if checked < 3:
        raise ValueError("Fewer than three completed scenarios")
    report = {"verified_scenarios": checked, "cohort_rows": 101146, "test_rows": 20230,
              "candidate_configs_per_scenario": 100, "folds_per_candidate": 5,
              "matching_environment_verification": True,
              "checks": ["input hashes", "candidate and fold completeness", "CV-only winner",
                         "split membership", "chronological boundaries", "saved model reload",
                         "literal scale_pos_weight in saved objective", "prediction parity",
                         "metric recomputation", "exported figures"]}
    target = root / "verification.json"
    if args.profile is not None:
        target = root / "verification" / f"{args.profile}.json"
        target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
