"""Adversarial availability and selection checks for the online history model."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fitw import operational
from fitw.spec import TARGET


def observation(scheduled, actual, delay, *, date="2024-01-01", number=23,
                station=10, kind="ARRIVAL"):
    return {"departureDate": date, "trainNumber": number, "stationUICCode": station,
            "type": kind, "scheduledTime": scheduled, "actualTime": actual,
            TARGET: float(delay)}


def minute(value):
    return f"2024-01-01T12:{value:02d}:00.000Z"


class OperationalAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.target = pd.DataFrame([observation(minute(30), minute(35), 5)], index=[17])

    def test_strict_actual_and_schedule_boundaries(self):
        sources = pd.DataFrame([
            observation(minute(10), minute(20), 10, station=1),
            observation(minute(15), minute(30), 999, station=2),  # At cutoff.
            observation(minute(30), minute(25), 998, station=3),  # Focal schedule.
            observation(minute(40), minute(24), 997, station=4),  # Early future event.
            observation(minute(20), minute(31), 996, station=5),  # Observed later.
        ])
        row = operational.history_features(self.target, sources).loc[17]
        self.assertEqual(row.history_count, 1)
        self.assertEqual(row.previous_delay, 10)
        self.assertEqual(row.previous_station, 1)
        self.assertEqual(row.previous_age_minutes, 10)
        self.assertEqual(row.previous_scheduled_gap_minutes, 20)

    def test_horizon_and_same_dated_run(self):
        sources = pd.DataFrame([
            observation(minute(10), minute(20), 10),
            observation(minute(15), minute(25), 100),  # Equal to five-minute cutoff.
            observation(minute(10), minute(20), 200, number=24),
            observation(minute(10), minute(20), 300, date="2023-12-31"),
        ])
        row = operational.history_features(self.target, sources, horizon_minutes=5).loc[17]
        self.assertEqual(row.history_count, 1)
        self.assertEqual(row.previous_delay, 10)
        self.assertEqual(row.previous_age_minutes, 5)
        self.assertEqual(row.prediction_cutoff, pd.Timestamp(minute(25)))
        with self.assertRaises(ValueError):
            operational.history_features(self.target, sources, horizon_minutes=-1)

    def test_future_and_focal_labels_cannot_change_predictors(self):
        sources = pd.DataFrame([
            observation(minute(10), minute(20), 10),
            observation(minute(25), minute(35), 100),
            observation(minute(30), minute(35), 5),
        ])
        expected = operational.history_features(self.target, sources)
        sources.loc[[1, 2], TARGET] = [1e9, -1e9]
        sources.loc[2, "actualTime"] = minute(0)  # Focal timetable row still excluded.
        changed = self.target.copy()
        changed.loc[17, TARGET] = 1e9
        changed.loc[17, "actualTime"] = "2025-01-01T00:00:00Z"
        pd.testing.assert_frame_equal(expected, operational.history_features(changed, sources))

    def test_history_statistics_use_only_available_observations(self):
        sources = pd.DataFrame([
            observation(minute(15), minute(22), 7, station=2, kind="DEPARTURE"),
            observation(minute(10), minute(18), -2, station=1),
            observation(minute(20), minute(40), 10000),
        ])
        row = operational.history_features(self.target, sources).loc[17]
        self.assertEqual(row.history_count, 2)
        self.assertEqual(row.previous_delay, 7)
        self.assertEqual(row.second_previous_delay, -2)
        self.assertEqual(row.delay_change, 9)
        self.assertEqual(row.history_mean_delay, 2.5)
        self.assertEqual(row.history_max_delay, 7)
        self.assertEqual(row.previous_is_departure, 1)

    def test_unavailable_history_preserves_row_and_missing_values(self):
        sources = pd.DataFrame([
            observation(minute(10), None, 10),
            observation(minute(10), minute(20), np.nan),
            observation("invalid", minute(20), 5),
        ])
        result = operational.history_features(self.target, sources)
        self.assertEqual(result.index.tolist(), [17])
        self.assertEqual(result.loc[17, "history_count"], 0)
        self.assertTrue(pd.isna(result.loc[17, "previous_delay"]))
        self.assertTrue(pd.isna(result.loc[17, "source_actual_time"]))

    def test_pending_events_use_unfiltered_timetable_and_skip_completed_positions(self):
        sources = pd.DataFrame([
            observation(minute(5), None, np.nan, station=1),  # Behind a completed event.
            observation(minute(10), minute(15), 5, station=2),
            observation(minute(20), None, np.nan, station=3),
            observation(minute(25), minute(40), 15, station=4),
            observation(minute(30), minute(35), 5),  # Focal schedule is excluded.
        ])
        row = operational.history_features(self.target, sources).loc[17]
        self.assertEqual(row.pending_count, 2)
        self.assertEqual(row.oldest_pending_minutes, 10)
        self.assertEqual(row.origin_station, 1)
        self.assertEqual(row.scheduled_run_elapsed_minutes, 25)
        self.assertEqual(row.scheduled_route_position, 4)
        self.assertEqual(row.next_expected_gap_minutes, 10)
        # The event at the horizon cutoff is not overdue yet.
        row = operational.history_features(self.target, sources, horizon_minutes=5).loc[17]
        self.assertEqual(row.pending_count, 1)
        self.assertEqual(row.oldest_pending_minutes, 5)
        sources.loc[3, TARGET] = 1e9
        self.assertEqual(operational.history_features(self.target, sources).loc[17].pending_count, 2)

    def test_departure_dwell_is_planned_but_arrival_delay_must_be_observed(self):
        target = self.target.copy()
        target["type"] = "DEPARTURE"
        sources = pd.DataFrame([
            observation(minute(10), minute(15), 5, station=100),
            observation(minute(20), minute(25), 5, station=370),
            observation(minute(30), minute(35), 5, station=370, kind="DEPARTURE"),
        ])
        row = operational.history_features(target, sources).loc[17]
        self.assertEqual(row.scheduled_oulu_dwell_minutes, 10)
        self.assertEqual(row.available_oulu_arrival_delay, 5)
        self.assertEqual(row.previous_is_oulu, 1)
        earlier = operational.history_features(target, sources, horizon_minutes=10).loc[17]
        self.assertEqual(earlier.scheduled_oulu_dwell_minutes, 10)
        self.assertTrue(pd.isna(earlier.available_oulu_arrival_delay))
        self.assertEqual(earlier.previous_is_oulu, 0)

    def test_station_context_excludes_same_run_and_future_or_stale_labels(self):
        sources = pd.DataFrame([
            observation(minute(5), minute(10), 5, station=370, number=24),
            observation(minute(20), minute(25), 5, station=370, number=25),
            observation(minute(25), minute(30), 500, station=370, number=26),
            observation(minute(28), None, np.nan, station=370, number=27),
            observation(minute(10), minute(20), 1000, station=370),  # Same dated run.
            observation("2024-01-01T09:10:00Z", "2024-01-01T09:20:00Z", 100,
                        station=370, number=28),  # Before the three-hour window.
            observation(minute(10), minute(20), 2000, station=100, number=29),
        ])
        expected = operational.history_features(self.target, sources)
        row = expected.loc[17]
        self.assertEqual(row.oulu_recent_count, 2)
        self.assertEqual(row.oulu_recent_mean_delay, 5)
        self.assertEqual(row.oulu_recent_max_delay, 5)
        self.assertEqual(row.oulu_pending_count, 2)
        self.assertEqual(row.oulu_oldest_pending_minutes, 5)
        sources.loc[2, TARGET] = -1e9  # At-cutoff observation remains unavailable.
        sources.loc[3, TARGET] = 1e9  # Missing observation remains unavailable.
        pd.testing.assert_frame_equal(expected, operational.history_features(self.target, sources))

    def test_completed_event_without_delay_resolves_earlier_missing_route_event(self):
        sources = pd.DataFrame([
            observation(minute(5), None, np.nan),
            observation(minute(10), minute(20), np.nan),
            observation(minute(25), None, np.nan),
        ])
        row = operational.history_features(self.target, sources).loc[17]
        self.assertEqual(row.history_count, 0)
        self.assertTrue(pd.isna(row.previous_delay))
        self.assertEqual(row.pending_count, 1)
        self.assertEqual(row.oldest_pending_minutes, 5)

    def test_later_other_train_completion_resolves_missing_oulu_arrival(self):
        sources = pd.DataFrame([
            observation(minute(5), None, np.nan, number=24, station=370),
            observation(minute(10), minute(15), np.nan, number=24,
                        station=370, kind="DEPARTURE"),
            observation(minute(20), None, np.nan, number=25, station=370),
        ])
        row = operational.history_features(self.target, sources).loc[17]
        self.assertEqual(row.oulu_recent_count, 0)
        self.assertTrue(pd.isna(row.oulu_recent_mean_delay))
        self.assertEqual(row.oulu_pending_count, 1)
        self.assertEqual(row.oulu_oldest_pending_minutes, 10)
        # Equality to the cutoff does not yet resolve that run's missing arrival.
        sources.loc[1, "actualTime"] = minute(30)
        row = operational.history_features(self.target, sources).loc[17]
        self.assertEqual(row.oulu_pending_count, 3)
        self.assertEqual(row.oulu_oldest_pending_minutes, 25)

    def test_origin_ties_do_not_depend_on_future_observation_order(self):
        sources = pd.DataFrame([
            observation(minute(10), minute(35), 25, station=100),
            observation(minute(10), minute(40), 30, station=200),
        ])
        expected = operational.history_features(self.target, sources)
        sources.loc[0, "actualTime"] = minute(45)
        pd.testing.assert_frame_equal(expected, operational.history_features(self.target, sources))

    def test_explicit_datetime_units_and_timezone_offsets_match_utc_strings(self):
        sources = pd.DataFrame([
            observation(minute(10), minute(20), 10),
            observation(minute(25), minute(30), 5),
        ])
        expected = operational.history_features(self.target, sources, horizon_minutes=2.5)
        target = self.target.copy()
        for unit in ["us", "ns"]:
            with self.subTest(unit=unit):
                changed = sources.copy()
                target["scheduledTime"] = pd.to_datetime(target.scheduledTime, utc=True).astype(f"datetime64[{unit}, UTC]")
                for name in ["scheduledTime", "actualTime"]:
                    changed[name] = pd.to_datetime(changed[name], utc=True).astype(f"datetime64[{unit}, UTC]")
                result = operational.history_features(target, changed, horizon_minutes=2.5)
                pd.testing.assert_frame_equal(expected, result, check_dtype=False)
                self.assertEqual(result.loc[17, "previous_age_minutes"], 7.5)
        sources["scheduledTime"] = ["2024-01-01T14:10:00+02:00", "2024-01-01T14:25:00+02:00"]
        sources["actualTime"] = ["2024-01-01T14:20:00+02:00", "2024-01-01T14:30:00+02:00"]
        pd.testing.assert_frame_equal(expected, operational.history_features(self.target, sources, horizon_minutes=2.5))

    def test_scheduled_predictors_ignore_actual_time_and_offset_labels(self):
        target = self.target.assign(trainStopping=True, commercialStop=True,
                                    differenceInMinutes_offset=5,
                                    differenceInMinutes_eachStation_offset=5)
        expected = operational.scheduled_features(target)
        target["actualTime"] = "2030-01-01T00:00:00Z"
        target[TARGET] = 99999
        target["differenceInMinutes_offset"] = -99999
        target["differenceInMinutes_eachStation_offset"] = 88888
        pd.testing.assert_frame_equal(expected, operational.scheduled_features(target))
        self.assertNotIn(TARGET, operational.FEATURES)
        self.assertNotIn("actualTime", operational.FEATURES)
        self.assertNotIn("differenceInMinutes_offset", operational.FEATURES)
        self.assertNotIn("differenceInMinutes_eachStation_offset", operational.FEATURES)

    def test_loader_rejects_cutoff_or_run_provenance_tampering(self):
        sources = pd.DataFrame([observation(minute(10), minute(20), 10)])
        frame = pd.concat([self.target, operational.history_features(self.target, sources)], axis=1)
        frame["event_time"] = pd.to_datetime(frame.actualTime, utc=True)
        frame["row_id"] = 0
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.parquet"
            frame.to_parquet(path, index=False)
            self.assertEqual(len(operational.load_data(path)), 1)
            for field, value in [("source_actual_time", pd.Timestamp(minute(30))),
                                 ("source_scheduled_time", pd.Timestamp(minute(30))),
                                 ("source_departure_date", "2023-12-31"),
                                 ("source_train_number", 24)]:
                with self.subTest(field=field):
                    changed = frame.copy()
                    changed[field] = value
                    changed.to_parquet(path, index=False)
                    with self.assertRaises(ValueError):
                        operational.load_data(path)


class OperationalSelectionTests(unittest.TestCase):
    def test_training_labels_are_purged_before_earliest_prediction_cutoff(self):
        frame = pd.DataFrame({
            "event_time": pd.to_datetime([minute(10), minute(20), minute(25), minute(30)], utc=True),
            "prediction_cutoff": pd.to_datetime([minute(0), minute(10), minute(20), minute(15)], utc=True),
        })
        retained = operational.purge_training(frame, np.array([0, 1]), np.array([2, 3]))
        np.testing.assert_array_equal(retained, [0])
        # Equality to the earliest cutoff cannot supply an already observed label.
        frame.loc[0, "event_time"] = pd.Timestamp(minute(15))
        with self.assertRaises(ValueError):
            operational.purge_training(frame, np.array([0, 1]), np.array([2, 3]))

    def test_final_holdout_labels_cannot_select_model(self):
        frame = pd.DataFrame({name: np.zeros(120) for name in operational.FEATURES})
        frame["row_id"] = np.arange(120)
        frame["event_time"] = pd.date_range("2020-01-01", periods=120, freq="h", tz="UTC")
        frame["prediction_cutoff"] = frame.event_time - pd.Timedelta(minutes=5)
        frame["departureDate"] = frame.event_time.dt.strftime("%Y-%m-%d")
        frame[TARGET] = (np.arange(120) % 7).astype(float)

        class DummyEstimator:
            def __init__(self, params, jobs):
                self.offset = params["offset"]
                self.feature_importances_ = np.zeros(len(operational.FEATURES))

            def fit(self, x, y):
                self.location = np.mean(y)
                return self

            def predict(self, x):
                return np.full(len(x), self.location + self.offset)

            def save_model(self, path):
                Path(path).write_text(str(self.location))

        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "features.parquet"
            data.write_text("fixed input identity")
            data.with_suffix(".manifest.json").write_text("{}")
            selections = []
            for repeat in range(2):
                changed = frame.copy()
                if repeat:
                    changed.loc[96:, TARGET] = 1e9
                output = Path(temp) / str(repeat)
                args = SimpleNamespace(data=str(data), output=str(output), jobs=1)
                with patch.object(operational, "load_data", return_value=changed), \
                     patch.object(operational, "candidates", return_value=[{"offset": 0}, {"offset": 10}]), \
                     patch.object(operational, "estimator", DummyEstimator), \
                     contextlib.redirect_stdout(io.StringIO()):
                    operational.train(args)
                selections.append(json.loads((output / "selection.json").read_text()))
            self.assertEqual(selections[0], selections[1])


if __name__ == "__main__":
    unittest.main()
