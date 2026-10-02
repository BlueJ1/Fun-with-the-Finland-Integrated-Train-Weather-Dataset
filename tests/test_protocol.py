"""Research protocol invariants, using only small synthetic fixtures.

These tests do not estimate or stand in for replication performance. Temporary
experiment outputs are deleted after each test, and no archive row is mutated.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

from fitw.experiment import (candidate_parameters, describe_split, metrics,
                             monthly_impute, model, run, split_indices)
from fitw.features import make_features, weather_categories
from fitw.spec import CATEGORIES, OPERATIONAL, TARGET, feature_sets
from fitw.scaling import WeatherScaler


def weather_rows(*changes):
    ordinary = {
        "Air temperature": 10., "Precipitation amount": 0.,
        "Precipitation intensity": 0., "Wind speed": 1., "Gust speed": 1.,
        "Relative humidity": 60., "Dew-point temperature": 5.,
        "Snow depth": 0., "Horizontal visibility": 10000.,
    }
    return pd.DataFrame([ordinary | row for row in changes],
                        index=pd.Index(range(40, 40 + len(changes)), name="observation"))


def categories(rows, rules="paper"):
    encoded = weather_categories(rows, rules=rules)
    return [column.removeprefix("weather_scenario_")
            for column in encoded.idxmax(axis=1)]


class CategoryProtocolTests(unittest.TestCase):
    def test_compound_blizzard_and_severity_priority(self):
        # All first three categories apply to the first row; severity must win.
        rows = weather_rows(
            {"Air temperature": -25, "Precipitation intensity": 3,
             "Snow depth": 10, "Wind speed": 12, "Horizontal visibility": 500},
            {"Air temperature": -25, "Precipitation intensity": 3, "Snow depth": 10},
            {"Air temperature": -25},
            # Strong wind alone cannot satisfy the compound blizzard definition.
            {"Air temperature": -5, "Wind speed": 16, "Horizontal visibility": 500},
        )
        self.assertEqual(categories(rows), ["Blizzard", "Heavy Snow", "Extreme Cold", "High Winds"])

    def test_printed_strict_boundary_values(self):
        rows = weather_rows(
            {"Air temperature": -20}, {"Air temperature": -20.001},
            {"Air temperature": 30}, {"Air temperature": 30.001},
            {"Wind speed": 15, "Gust speed": 20}, {"Gust speed": 20.001},
            {"Air temperature": -2, "Precipitation intensity": .5},
            {"Air temperature": 2, "Precipitation intensity": .5},
            {"Air temperature": -1.999, "Precipitation intensity": .5},
            {"Air temperature": 1.999, "Precipitation intensity": .5},
        )
        self.assertEqual(categories(rows), [
            "Normal Clear", "Extreme Cold", "Normal Clear", "Extreme Heat",
            "Normal Clear", "High Winds", "Normal Clear", "Normal Clear",
            "Freezing Rain", "Freezing Rain",
        ])
        self.assertEqual(categories(rows, "author-legacy")[-4:], ["Freezing Rain"] * 4)

    def test_fog_uses_amount_and_all_clauses(self):
        rows = weather_rows(
            {"Precipitation amount": .1, "Horizontal visibility": 999, "Relative humidity": 96},
            {"Precipitation amount": .10001, "Horizontal visibility": 999, "Relative humidity": 96},
            {"Horizontal visibility": 1000, "Relative humidity": 96},
            {"Horizontal visibility": 999, "Relative humidity": 95},
        )
        self.assertEqual(categories(rows), ["Dense Fog", "Normal Clear", "Normal Clear", "Normal Clear"])

    def test_literal_hierarchy_makes_black_ice_unreachable(self):
        # Every candidate meets the printed black-ice predicate, and each must
        # already have been consumed by the higher-priority freezing-rain rule.
        rows = weather_rows(*[
            {"Air temperature": t, "Precipitation amount": .01,
             "Relative humidity": 90, "Dew-point temperature": t - 1}
            for t in (-1.9, -1, 0, 1, 1.9)
        ])
        self.assertEqual(categories(rows), ["Freezing Rain"] * len(rows))
        self.assertEqual(float(weather_categories(rows)["weather_scenario_Black Ice"].sum()), 0)

    def test_one_hot_and_missing_measurement_policy_preserve_population(self):
        rows = weather_rows({}, {"Air temperature": np.nan},
                            {"Air temperature": np.nan, "Wind speed": 16})
        encoded = weather_categories(rows)
        self.assertEqual(encoded.index.tolist(), rows.index.tolist())
        self.assertEqual(encoded.columns.tolist(), CATEGORIES)
        np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(len(rows)))
        self.assertEqual(categories(rows), ["Normal Clear", "Normal Clear", "High Winds"])


class FeatureProtocolTests(unittest.TestCase):
    def test_scenarios_change_predictors_without_selecting_different_rows(self):
        rows = weather_rows({}, {"Air temperature": -25}, {"Snow depth": np.nan})
        rows["actualTime"] = ["2024-01-07T00:30:59Z", "2024-02-05T12:00:00Z", "2024-03-05T13:00:00Z"]
        rows["trainNumber"] = [22, 22, 23]
        rows["trainStopping"] = [True, False, True]
        for column in ("Wind direction", "Pressure (msl)", "Cloud amount"):
            rows[column] = np.nan
        frame = make_features(rows)
        for day, cloud, counts in [(False, False, (26, 16, 18)),
                                    (True, False, (27, 17, 19)),
                                    (True, True, (28, 18, 19))]:
            choices = feature_sets(day, cloud)
            self.assertEqual(tuple(len(choices[s]) for s in ("full", "instant", "categories")), counts)
            for columns in choices.values():
                self.assertEqual(len(set(columns)), len(columns))
                self.assertEqual(frame[columns].index.tolist(), rows.index.tolist())
                self.assertNotIn(TARGET, columns)
            self.assertTrue(set(choices["instant"]).issubset(choices["full"]))
            self.assertTrue(set(choices["categories"]).issubset(choices["full"]))
            self.assertEqual(set(choices["instant"]) & set(choices["categories"]),
                             set(OPERATIONAL + (["day_of_month"] if day else [])))

    def test_historical_temporal_conventions_and_train_identifier(self):
        rows = weather_rows({})
        # Sunday January 7, UTC 00:30. Seconds are deliberately ignored.
        rows["actualTime"] = "2024-01-07T00:30:59Z"
        rows["trainNumber"] = 22
        rows["trainStopping"] = True
        frame = make_features(rows)
        self.assertEqual(frame.iloc[0].train_id, 22)
        self.assertAlmostEqual(frame.iloc[0].month_sin, .5)
        self.assertAlmostEqual(frame.iloc[0].day_week_sin, np.sin(2 * np.pi / 7))
        self.assertAlmostEqual(frame.iloc[0].hour_sin, np.sin(2 * np.pi * .5 / 24))
        self.assertEqual(frame.iloc[0].day_of_month, 7)


class HistoricalScalingTests(unittest.TestCase):
    def test_only_designated_weather_predictors_are_scaled(self):
        # Service numbers and binary category indicators must retain their
        # original meanings even when adjacent raw weather columns are scaled.
        training = np.array([[22, 0, 1, 100, 0], [22, 10, 0, 200, 1],
                             [23, 20, 1, 300, 0], [23, 30, 0, 400, 1]], dtype=np.float32)
        original = training.copy()
        scaler = WeatherScaler([1, 3]).fit(training)
        transformed = scaler.transform(training)
        np.testing.assert_array_equal(transformed[:, [0, 2, 4]], training[:, [0, 2, 4]])
        np.testing.assert_array_equal(training, original)
        np.testing.assert_allclose(np.median(transformed[:, [1, 3]], axis=0), [0, 0])
        quartiles = np.percentile(transformed[:, [1, 3]], [25, 75], axis=0)
        np.testing.assert_allclose(quartiles[1] - quartiles[0], [1, 1], atol=1e-6)
        self.assertEqual(transformed.shape, training.shape)
        self.assertEqual(transformed.dtype, training.dtype)

    def test_category_only_input_requires_no_scaling(self):
        categories_only = np.array([[22, 1, 0], [23, 0, 1]], dtype=np.float32)
        scaler = WeatherScaler([]).fit(categories_only)
        transformed = scaler.transform(categories_only)
        np.testing.assert_array_equal(transformed, categories_only)
        self.assertFalse(np.shares_memory(transformed, categories_only))
        # Transforming a new category-only cohort must not depend on the
        # distribution or number of rows in the development cohort.
        heldout = np.array([[1000, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(scaler.transform(heldout), heldout)

    def test_joblib_roundtrip_preserves_heldout_transform(self):
        training = np.array([[22, 0, 1], [23, 10, 0], [24, 20, 1], [25, 30, 0]], dtype=np.float32)
        heldout = np.array([[55, 100, 0], [99, np.nan, 1]], dtype=np.float32)
        scaler = WeatherScaler([1]).fit(training)
        expected = scaler.transform(heldout)
        with tempfile.TemporaryDirectory(prefix="fitw_scaler_test_") as directory:
            saved = Path(directory) / "weather_scaler.joblib"
            joblib.dump(scaler, saved)
            reloaded = joblib.load(saved)
            np.testing.assert_allclose(reloaded.transform(heldout), expected, equal_nan=True)
            self.assertEqual(reloaded.indices, [1])
        # Heldout extremes use development centering and scale, rather than
        # being centered using heldout observations.
        self.assertAlmostEqual(expected[0, 1], (100 - 15) / 15, places=5)
        np.testing.assert_array_equal(expected[:, [0, 2]], heldout[:, [0, 2]])


class EvaluationProtocolTests(unittest.TestCase):
    @staticmethod
    def timeline(count=120):
        return pd.DataFrame({"event_time": pd.date_range("2020-01-01", periods=count, freq="h", tz="UTC"),
                             "departureDate": ["2020-01-01"] * count,
                             "trainNumber": np.arange(count), TARGET: np.arange(count)})

    def test_chronological_holdout_and_expanding_folds(self):
        frame = self.timeline()
        dev, test = split_indices(frame, "chronological")
        self.assertEqual((len(dev), len(test)), (96, 24))
        self.assertEqual(set(dev) & set(test), set())
        self.assertEqual(set(dev) | set(test), set(range(len(frame))))
        self.assertLess(frame.iloc[dev].event_time.max(), frame.iloc[test].event_time.min())
        folds = list(TimeSeriesSplit(n_splits=5).split(dev))
        previous_train = set()
        for tr, va in folds:
            self.assertTrue(previous_train.issubset(set(tr)))
            self.assertLess(frame.iloc[dev[tr]].event_time.max(), frame.iloc[dev[va]].event_time.min())
            self.assertEqual(set(dev[tr]) & set(test), set())
            self.assertEqual(set(dev[va]) & set(test), set())
            previous_train = set(tr)
        description = describe_split(frame, dev, test, folds, "chronological")
        self.assertEqual(len(description["folds"]), 5)
        self.assertEqual(description["dated_runs_shared_by_development_and_test"], 0)

    def test_timestamp_overlap_is_rejected(self):
        frame = self.timeline()
        frame.loc[96, "event_time"] = frame.loc[95, "event_time"]
        dev, test = split_indices(frame, "chronological")
        with self.assertRaises(AssertionError):
            describe_split(frame, dev, test, list(TimeSeriesSplit(n_splits=5).split(dev)), "chronological")

    def test_signed_targets_zero_labels_and_percentage_denominator(self):
        y = np.array([-2., 0., 2., 4.])
        pred = np.array([-1., 1., 1., 3.])
        measured = metrics(y, pred)
        self.assertAlmostEqual(measured["mae"], 1)
        self.assertAlmostEqual(measured["rmse"], 1)
        self.assertAlmostEqual(measured["r2"], .8)
        self.assertAlmostEqual(measured["wmape"], 50)
        self.assertAlmostEqual(measured["wmape_signed_denominator"], 100)
        balanced = metrics(np.array([-1., 1.]), np.zeros(2))
        self.assertEqual(balanced["wmape"], 100)
        self.assertIsNone(balanced["wmape_signed_denominator"])
        zero = metrics(np.zeros(2), np.ones(2))
        self.assertIsNone(zero["wmape"])

    def test_literal_scale_pos_weight_is_preserved_and_changes_regression_fit(self):
        # The historical source passes every sampled parameter to XGBRegressor.
        # In the supported XGBoost implementations, regression observations
        # with target exactly one also respond to scale_pos_weight. Removing it
        # changes the experiment even though the target is continuous delay.
        parameters = {"n_estimators": 10, "max_depth": 2, "learning_rate": .1,
                      "subsample": .9, "colsample_bytree": 1., "scale_pos_weight": 5.9}
        weighted = model(parameters, seed=42, jobs=1)
        self.assertEqual(weighted.get_params()["scale_pos_weight"], 5.9)
        unweighted = model(parameters | {"scale_pos_weight": 1.}, seed=42, jobs=1)
        inputs = np.arange(12, dtype=np.float32).reshape(-1, 1)
        targets = np.array([0, 1, 1, 2, 0, 1, 3, 1, 0, 2, 1, 4], dtype=np.float32)
        weighted.fit(inputs, targets)
        unweighted.fit(inputs, targets)
        difference = np.max(np.abs(weighted.predict(inputs) - unweighted.predict(inputs)))
        self.assertGreater(float(difference), .01)

    def test_seeded_search_budget_prefixes_are_identical(self):
        for legacy in (False, True):
            pool = candidate_parameters(100, 42, legacy)
            for budget in (10, 20, 30, 50, 80):
                self.assertEqual(candidate_parameters(budget, 42, legacy), pool[:budget])
            self.assertNotEqual(candidate_parameters(100, 43, legacy), pool)
            # The scientific discrepancy in integer endpoints must stay explicit.
            large_pool = candidate_parameters(2000, 42, legacy)
            self.assertEqual(max(p["max_depth"] for p in large_pool), 7 if legacy else 8)
            self.assertEqual(max(p["n_estimators"] for p in large_pool), 399 if legacy else 400)
            self.assertTrue(all(100 <= p["n_estimators"] <= 400 for p in pool))
            self.assertTrue(all(4 <= p["max_depth"] <= 8 for p in pool))

    def test_legacy_month_imputation_is_explicit_and_separate(self):
        frame = pd.DataFrame({"source_file": ["month_a"] * 4 + ["month_b"] * 4})
        for column in feature_sets(False, True)["instant"]:
            frame[column] = np.arange(8, dtype=float)
        frame.loc[0, "Air temperature"] = np.nan
        frame.loc[[4, 5], "Air temperature"] = np.nan
        original = frame.copy(deep=True)
        imputed = monthly_impute(frame)
        # One missing of four is within threshold; two missing of four means
        # the month's column is omitted, rather than imputed across months.
        self.assertEqual(imputed.loc[0, "Air temperature"], 2)
        self.assertTrue(imputed.loc[4:7, "Air temperature"].isna().all())
        pd.testing.assert_frame_equal(frame, original)

    def test_run_scaling_is_fold_local_and_test_labels_do_not_select_candidates(self):
        class StubRegressor:
            def __init__(self, offset, features):
                self.offset, self.feature_importances_ = offset, np.zeros(features)

            def fit(self, x, y):
                self.location = float(np.mean(y))
                return self

            def predict(self, x):
                return np.full(len(x), self.location + self.offset)

            def save_model(self, path):
                Path(path).write_text("synthetic unit-test stub, not an XGBoost model")

        frame = self.timeline()
        frame["row_id"] = np.arange(len(frame))
        frame["type"] = "ARRIVAL"
        for name in feature_sets()["full"]:
            frame[name] = np.arange(len(frame), dtype=float)
        original_fit = RobustScaler.fit
        fit_rows = []

        def spy_fit(scaler, x, *args, **kwargs):
            fit_rows.append(np.array(x, copy=True))
            return original_fit(scaler, x, *args, **kwargs)

        def stub_model(parameters, seed, jobs):
            return StubRegressor(parameters["max_depth"], len(feature_sets()["categories"]))

        with tempfile.TemporaryDirectory(prefix="fitw_protocol_test_") as directory:
            directory = Path(directory)
            results = []
            for iteration in range(2):
                changed = frame.copy()
                if iteration:
                    changed.loc[96:, TARGET] += 100000
                path = directory / f"fixture_{iteration}.parquet"
                changed.to_parquet(path, index=False)
                args = SimpleNamespace(data=str(path), output=str(directory / f"run_{iteration}"),
                                       scenario="categories", protocol="chronological",
                                       day_of_month=False, keep_cloud=False, candidates=3,
                                       seed=42, jobs=1)
                with patch("fitw.experiment.model", side_effect=stub_model), \
                     patch.object(RobustScaler, "fit", new=spy_fit), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                results.append(json.loads((Path(args.output) / args.scenario / "result.json").read_text()))
            self.assertEqual(results[0]["best_candidate"], results[1]["best_candidate"])
            self.assertNotEqual(results[0]["test_metrics"]["rmse"], results[1]["test_metrics"]["rmse"])
            # Five expanding development folds, then one complete-development
            # scaler. No fit may see any holdout predictor or validation row.
            self.assertEqual([len(x) for x in fit_rows], [16, 32, 48, 64, 80, 96] * 2)
            for x in fit_rows:
                self.assertLess(float(x.max()), 96)
            self.assertEqual([float(np.median(x[:, 0])) for x in fit_rows[:6]],
                             [7.5, 15.5, 23.5, 31.5, 39.5, 47.5])


if __name__ == "__main__":
    unittest.main()
