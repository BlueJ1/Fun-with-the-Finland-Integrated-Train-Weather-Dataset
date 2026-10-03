"""Old-snapshot availability checks using synthetic events only."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fitw import stale
from fitw.spec import TARGET


def time(hour, minute=0):
    return f"2024-01-01T{hour:02d}:{minute:02d}:00.000Z"


def event(scheduled, actual, delay, *, number=23, station=100,
          kind="ARRIVAL", accepted=None):
    return {"departureDate": "2024-01-01", "trainNumber": number,
            "stationUICCode": station, "type": kind,
            "scheduledTime": scheduled, "actualTime": actual,
            TARGET: float(delay), "trainStopping": True, "commercialStop": True,
            "trainType": "IC", "timetableType": "REGULAR",
            "timetableAcceptanceDate": accepted or time(10)}


class StaleSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.target = pd.DataFrame([
            event(time(13), time(13, 15), 15, station=370, kind="DEPARTURE")
        ], index=[7])
        self.rows = pd.DataFrame([
            event(time(12, 10), time(12, 20), 10),
            event(time(12, 25), time(12, 40), 15),
            event(time(12, 50), time(12, 55), 5, station=370),
            event(time(12, 15), time(12, 20), 5, station=370, number=24),
            event(time(12, 20), time(12, 40), 20, station=370, number=25),
            self.target.loc[7].to_dict(),
        ])

    def snapshot(self, rows=None, target=None):
        rows = self.rows if rows is None else rows
        return stale.snapshot_features(
            self.target if target is None else target,
            rows[stale.STATIC_COLUMNS], rows[stale.KEY + ["actualTime", TARGET]],
        )

    def prepared(self):
        part = self.snapshot().copy()
        part["query_train_number"] = self.target.trainNumber
        part["trainNumber"] = part.pop("feature_train_number")
        for name in ["scheduledTime", "departureDate", "type", TARGET]:
            part[name] = self.target[name]
        part["row_id"] = 7
        part["event_time"] = pd.to_datetime(self.target.actualTime, utc=True)
        return part

    def test_history_and_context_are_at_least_thirty_minutes_old(self):
        row = self.snapshot().loc[7]
        self.assertEqual(row.prediction_time, pd.Timestamp(time(13)))
        self.assertEqual(row.observation_cutoff, pd.Timestamp(time(12, 30)))
        self.assertEqual(row.previous_delay, 10)
        self.assertEqual(row.history_count, 1)
        self.assertEqual(row.oulu_recent_count, 1)
        self.assertEqual(row.oulu_recent_mean_delay, 5)
        self.assertEqual(row.pending_count, 1)
        self.assertEqual(row.oldest_pending_minutes, 5)
        self.assertEqual(row.oulu_pending_count, 1)
        self.assertEqual(row.oulu_oldest_pending_minutes, 10)
        self.assertGreaterEqual(row.prediction_time - row.input_latest_actual_time,
                                pd.Timedelta(minutes=30))
        self.assertLess(row.input_latest_actual_time, row.observation_cutoff)
        self.assertLess(row.input_latest_timetable_time, row.observation_cutoff)

    def test_post_snapshot_labels_and_completions_cannot_change_features(self):
        expected = self.snapshot()
        changed = self.rows.copy()
        changed.loc[[1, 2, 4, 5], TARGET] = [1e9, -1e9, 2e9, -2e9]
        # These completions are all after the old snapshot, even if before prediction.
        changed.loc[[1, 2, 4, 5], "actualTime"] = [time(12, 31), time(12, 59), time(12, 45), time(13, 59)]
        pd.testing.assert_frame_equal(expected, self.snapshot(changed))

    def test_missing_and_post_snapshot_actual_events_are_equivalent(self):
        expected = self.snapshot()
        changed = self.rows.copy()
        changed.loc[[1, 2, 4, 5], "actualTime"] = None
        changed.loc[[1, 2, 4, 5], TARGET] = np.nan
        pd.testing.assert_frame_equal(expected, self.snapshot(changed))

    def test_exact_snapshot_actual_and_acceptance_are_excluded(self):
        changed = self.rows.copy()
        changed.loc[0, "actualTime"] = time(12, 30)
        changed.loc[3, "timetableAcceptanceDate"] = time(12, 30)
        row = self.snapshot(changed).loc[7]
        self.assertEqual(row.history_count, 0)
        self.assertTrue(pd.isna(row.previous_delay))
        self.assertEqual(row.oulu_recent_count, 0)

    def test_recent_or_missing_focal_timetable_masks_all_static_predictors(self):
        for acceptance in [time(12, 30), time(12, 45), None]:
            with self.subTest(acceptance=acceptance):
                changed = self.rows.copy()
                changed.loc[5, "timetableAcceptanceDate"] = acceptance
                row = self.snapshot(changed).loc[7]
                self.assertFalse(row.metadata_known)
                names = ["feature_train_number", "trainStopping", "commercialStop",
                         "is_departure", "month_sin", "month_cos", "hour_sin", "hour_cos",
                         "weekday_sin", "weekday_cos", "day_of_month",
                         "train_type_ic", "train_type_s", "train_type_p", "train_type_mv",
                         "timetable_adhoc"]
                self.assertTrue(row[names].isna().all())
                self.assertTrue(pd.isna(row.focal_timetable_available_time))

    def test_recent_schedule_cannot_contribute_history_pending_or_context(self):
        changed = self.rows.copy()
        changed.loc[[0, 1, 2, 3, 4], "timetableAcceptanceDate"] = time(12, 45)
        row = self.snapshot(changed).loc[7]
        self.assertEqual(row.history_count, 0)
        self.assertEqual(row.pending_count, 0)
        self.assertEqual(row.oulu_recent_count, 0)
        self.assertEqual(row.oulu_pending_count, 0)
        self.assertTrue(pd.isna(row.origin_station))
        self.assertTrue(pd.isna(row.scheduled_oulu_dwell_minutes))

    def test_recent_completion_cannot_resolve_old_pending_event(self):
        rows = pd.DataFrame([
            event(time(12, 5), None, np.nan, number=24, station=370),
            event(time(12, 10), time(12, 45), np.nan, number=24, station=370,
                  kind="DEPARTURE"),
            self.target.loc[7].to_dict(),
        ])
        row = self.snapshot(rows).loc[7]
        self.assertEqual(row.oulu_pending_count, 2)
        self.assertEqual(row.oulu_oldest_pending_minutes, 25)
        rows.loc[1, "actualTime"] = time(12, 20)
        row = self.snapshot(rows).loc[7]
        self.assertEqual(row.oulu_pending_count, 0)
        self.assertEqual(row.input_latest_actual_time, pd.Timestamp(time(12, 20)))

    def test_loader_rejects_fresh_inputs_and_holdout_membership(self):
        frame = self.prepared()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prepared.parquet"
            frame.to_parquet(path, index=False)
            self.assertEqual(len(stale.load_data(path)), 1)
            changes = [
                ("row_id", stale.DEVELOPMENT_ROWS),
                ("observation_cutoff", pd.Timestamp(time(12, 31))),
                ("prediction_cutoff", pd.Timestamp(time(12, 29))),
                ("source_actual_time", pd.Timestamp(time(12, 30))),
                ("input_latest_actual_time", pd.Timestamp(time(12, 40))),
                ("input_latest_timetable_time", pd.Timestamp(time(12, 30))),
                ("focal_timetable_available_time", pd.Timestamp(time(12, 45))),
                ("source_train_number", 99),
            ]
            for field, value in changes:
                with self.subTest(field=field):
                    changed = frame.copy()
                    changed[field] = value
                    changed.to_parquet(path, index=False)
                    with self.assertRaises(ValueError):
                        stale.load_data(path)


class StaleProtocolTests(unittest.TestCase):
    def test_development_filter_is_applied_in_parquet_io(self):
        table = pa.Table.from_pandas(pd.DataFrame({"row_id": [0, 80915, 80916, 99999]}))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_cohort.parquet"
            pq.write_table(table, path)
            with patch.object(stale.pq, "read_table", wraps=pq.read_table) as reader:
                selected = stale.development_cohort(path)
            self.assertEqual(reader.call_args.kwargs["filters"], [("row_id", "<", 80916)])
            self.assertEqual(selected.row_id.tolist(), [0, 80915])

    def test_all_five_folds_purge_labels_newer_than_oldest_validation_snapshot(self):
        frame = pd.DataFrame({
            "event_time": pd.date_range("2020-01-01", periods=120, freq="h", tz="UTC"),
        })
        frame["prediction_time"] = frame.event_time - pd.Timedelta(hours=2)
        frame["observation_cutoff"] = frame.prediction_time - pd.Timedelta(minutes=30)
        frame["prediction_cutoff"] = frame.observation_cutoff
        folds = stale.folds_for(frame)
        self.assertEqual(len(folds), 5)
        for tr, va in folds:
            self.assertLess(frame.iloc[tr].event_time.max(), frame.iloc[va].observation_cutoff.min())
            self.assertLess(len(tr), va.min())
            self.assertGreaterEqual((frame.iloc[va].prediction_time - frame.iloc[va].observation_cutoff).min(),
                                    pd.Timedelta(minutes=30))


class StaleServiceHistoryTests(unittest.TestCase):
    def frame(self):
        rows = [
            # One eligible earlier dated run.
            ("2024-01-01", "2024-01-01T13:00:00Z", "2024-01-01T13:05:00Z", 5., 23, "ARRIVAL"),
            # Different departure date, but its completion is within thirty minutes.
            ("2024-01-02", "2024-01-03T12:00:00Z", "2024-01-03T12:40:00Z", 500., 23, "ARRIVAL"),
            # Completion is old enough, but the archived prediction was after the cutoff.
            ("2024-01-02", "2024-01-03T12:45:00Z", "2024-01-03T12:20:00Z", 600., 23, "ARRIVAL"),
            # Same dated run is excluded even when an earlier event has completed.
            ("2024-01-03", "2024-01-03T11:00:00Z", "2024-01-03T11:05:00Z", 700., 23, "ARRIVAL"),
            # Identity and event type must both match the prior-run query.
            ("2024-01-01", "2024-01-01T13:00:00Z", "2024-01-01T13:05:00Z", 800., 24, "ARRIVAL"),
            ("2024-01-01", "2024-01-01T13:00:00Z", "2024-01-01T13:05:00Z", 900., 23, "DEPARTURE"),
            # Outside the 28-day observation window.
            ("2023-11-01", "2023-11-01T13:00:00Z", "2023-11-01T13:05:00Z", 1000., 23, "ARRIVAL"),
            ("2024-01-03", "2024-01-03T13:00:00Z", "2024-01-03T13:10:00Z", 10., 23, "ARRIVAL"),
        ]
        frame = pd.DataFrame(rows, columns=["departureDate", "prediction_time", "event_time",
                                           TARGET, "query_train_number", "type"])
        frame["prediction_time"] = pd.to_datetime(frame.prediction_time, utc=True)
        frame["event_time"] = pd.to_datetime(frame.event_time, utc=True)
        frame["observation_cutoff"] = frame.prediction_time - pd.Timedelta(minutes=30)
        frame["metadata_known"] = True
        frame["history_count"] = [2, 3, 4, 5, 6, 7, 8, 0]
        frame["pending_count"] = [1, 20, 30, 40, 50, 60, 70, 0]
        frame["previous_age_minutes"] = [15., 25., 35., 45., 55., 65., 75., np.nan]
        return frame

    def test_service_history_uses_only_older_different_dated_runs(self):
        row = stale.add_service_history(self.frame()).loc[7]
        self.assertEqual(row.service_observed_runs_28d, 1)
        self.assertEqual(row.service_mean_delay_28d, 5)
        self.assertEqual(row.service_last_delay, 5)
        self.assertEqual(row.service_history_coverage_28d, 1)
        self.assertEqual(row.service_mean_pending_28d, 1)
        self.assertEqual(row.service_mean_previous_age_28d, 15)
        self.assertEqual(row.service_latest_available_time, pd.Timestamp("2024-01-01T13:05:00Z"))
        self.assertLess(row.service_latest_available_time, row.observation_cutoff)

    def test_service_features_ignore_future_recent_and_same_date_labels(self):
        frame = self.frame()
        expected = stale.add_service_history(frame).loc[7, stale.SERVICE_FEATURES + ["service_latest_available_time"]]
        changed = frame.copy()
        changed.loc[[1, 2, 3, 4, 5, 6, 7], TARGET] = np.arange(7) * 1e9
        changed.loc[[1, 2, 3], "pending_count"] = 10000
        changed.loc[[1, 2, 3], "history_count"] = 10000
        result = stale.add_service_history(changed).loc[7, stale.SERVICE_FEATURES + ["service_latest_available_time"]]
        pd.testing.assert_series_equal(expected, result)
        # Equality at the old snapshot is still unavailable.
        changed.loc[0, "event_time"] = changed.loc[7, "observation_cutoff"]
        row = stale.add_service_history(changed).loc[7]
        self.assertTrue(row[stale.SERVICE_FEATURES].isna().all())

    def test_unknown_query_metadata_masks_service_profile(self):
        frame = self.frame()
        frame.loc[7, "metadata_known"] = False
        row = stale.add_service_history(frame).loc[7]
        self.assertTrue(row[stale.SERVICE_FEATURES].isna().all())
        self.assertTrue(pd.isna(row.service_latest_available_time))


class StalePostprocessTests(unittest.TestCase):
    rule = {"nohistory_departure_cap": 20., "pending_floor_threshold": 300.,
            "pending_floor_multiplier": 1.5}

    def test_postprocessing_depends_only_on_old_feature_inputs_without_mutation(self):
        frame = pd.DataFrame({"history_count": [0, 0, 1, 1],
                              "is_departure": [1., 0., 1., 0.],
                              "oldest_pending_minutes": [500., 500., 301., 400.],
                              TARGET: [9999., -9999., 1e9, -1e9]})
        raw = np.array([100., 100., 100., 100.], dtype=np.float32)
        original_frame, original_raw = frame.copy(deep=True), raw.copy()
        expected = stale.apply_postprocess(frame, raw, self.rule)
        np.testing.assert_array_equal(expected, [20., 100., 451.5, 600.])
        pd.testing.assert_frame_equal(frame, original_frame)
        np.testing.assert_array_equal(raw, original_raw)
        changed = frame.copy()
        changed[TARGET] = [1., 2., 3., 4.]
        np.testing.assert_array_equal(expected, stale.apply_postprocess(changed, raw, self.rule))
        # The deployment recipe does not require focal labels at all.
        np.testing.assert_array_equal(expected, stale.apply_postprocess(frame.drop(columns=TARGET), raw, self.rule))
        np.testing.assert_array_equal(raw, stale.apply_postprocess(frame, raw, None))

    def test_pending_floor_threshold_is_strict_and_unknown_departure_is_not_capped(self):
        frame = pd.DataFrame({"history_count": [1, 1, 1, 0, 1],
                              "is_departure": [0., 1., 1., np.nan, 0.],
                              "oldest_pending_minutes": [299., 300., 301., 600., np.nan]})
        raw = np.array([50., 50., 50., 100., 50.])
        np.testing.assert_array_equal(stale.apply_postprocess(frame, raw, self.rule),
                                      [50., 50., 451.5, 100., 50.])

    def test_forward_helper_applies_saved_rule_and_rejects_recent_or_unavailable_inputs(self):
        frame = pd.DataFrame({"history_count": [0], "is_departure": [1.],
                              "oldest_pending_minutes": [0.],
                              "prediction_time": [pd.Timestamp("2024-01-03T13:00:00Z")],
                              "observation_cutoff": [pd.Timestamp("2024-01-03T12:30:00Z")],
                              "input_latest_actual_time": [pd.Timestamp("2024-01-03T12:20:00Z")]})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "config.json").write_text(json.dumps({"features": ["history_count"]}))
            (output / "selection.json").write_text(json.dumps({
                "configuration": {"mode": "direct"}, "postprocess": self.rule}))
            (output / "result.json").write_text(json.dumps({"deployment_not_before": "2024-01-03T12:00:00Z"}))
            with patch.object(stale, "XGBRegressor") as constructor:
                constructor.return_value.predict.return_value = np.array([100.])
                np.testing.assert_array_equal(stale.predict(frame, output), [20.])
                changed = frame.copy()
                changed["input_latest_actual_time"] = pd.Timestamp("2024-01-03T12:30:00Z")
                with self.assertRaises(ValueError):
                    stale.predict(changed, output)
                changed = frame.copy()
                changed["observation_cutoff"] = pd.Timestamp("2024-01-03T12:31:00Z")
                with self.assertRaises(ValueError):
                    stale.predict(changed, output)
                (output / "result.json").write_text(json.dumps({"deployment_not_before": "2024-01-03T14:00:00Z"}))
                with self.assertRaises(ValueError):
                    stale.predict(frame, output)


if __name__ == "__main__":
    unittest.main()
