"""Network context availability checks using synthetic events only."""

import unittest
import numpy as np
import pandas as pd

from fitw import stale, stale_enriched, stale_network
from fitw.spec import TARGET


def stamp(hour, minute=0):
    return f"2024-01-01T{hour:02d}:{minute:02d}:00Z"


def event(number, station, scheduled, actual, delay, accepted=None):
    return {"departureDate": "2024-01-01", "trainNumber": number,
            "stationUICCode": station, "type": "ARRIVAL", "scheduledTime": scheduled,
            "actualTime": actual, TARGET: delay, "trainStopping": True,
            "commercialStop": True, "trainType": "IC", "timetableType": "REGULAR",
            "timetableAcceptanceDate": accepted or stamp(9)}


class NetworkSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame([
            event(23, 100, stamp(11), stamp(11, 5), 5.),
            event(23, 200, stamp(12), stamp(12, 45), 45.),
            event(23, 300, stamp(12, 20), stamp(12, 40), 20.),
            event(23, 370, stamp(13), stamp(13, 5), 5.),
            event(24, 100, stamp(12, 10), stamp(12, 20), 10.),
            event(29, 200, stamp(12, 5), stamp(12, 15), 10.),
            event(25, 300, stamp(11, 20), stamp(11, 40), 20.),
            event(26, 200, stamp(10), stamp(10, 20), 30.),
            event(27, 200, stamp(12, 10), None, np.nan),
            # Later old completion at an unrelated station resolves the report.
            event(27, 900, stamp(12, 20), stamp(12, 25), np.nan),
            event(28, 300, stamp(12, 15), stamp(12, 40), 25.),
        ])
        self.target = self.rows.iloc[[3]].copy()
        self.target.index = [17]

    def snapshot(self, rows=None, target=None):
        rows = self.rows if rows is None else rows
        return stale_network.snapshot_features(self.target if target is None else target,
                                               rows[stale.STATIC_COLUMNS],
                                               rows[stale.KEY + ["actualTime", TARGET]])

    def test_station_windows_route_ahead_and_other_dated_run_exclusion(self):
        row = self.snapshot().loc[17]
        self.assertEqual(row.network_previous_station_60m_count, 1)
        self.assertEqual(row.network_previous_station_180m_count, 1)
        self.assertEqual(row.network_previous_station_60m_last_delay, 10.)
        self.assertEqual(row.network_next_station_60m_count, 1)
        self.assertEqual(row.network_next_station_180m_count, 2)
        self.assertEqual(row.network_next_station_180m_mean_delay, 20.)
        self.assertEqual(row.network_route_ahead_60m_count, 2)
        self.assertEqual(row.network_route_ahead_60m_mean_delay, 15.)
        self.assertEqual(row.network_route_ahead_60m_max_delay, 20.)
        self.assertEqual(row.network_route_ahead_60m_last_delay, 10.)
        self.assertEqual(row.network_route_ahead_60m_pending_count, 1)
        self.assertEqual(row.network_route_ahead_60m_oldest_pending_minutes, 15.)

    def test_recent_future_and_focal_observations_cannot_change_output(self):
        expected = self.snapshot()
        changed = self.rows.copy()
        changed.loc[[1, 2, 3, 10], TARGET] = 1e12
        changed.loc[[1, 2, 3, 10], "actualTime"] = [stamp(12, 31), stamp(12, 30), stamp(17), None]
        target = self.target.drop(columns=["actualTime", TARGET])
        pd.testing.assert_frame_equal(expected, self.snapshot(changed, target))
        changed.loc[[1, 2, 3, 10], "actualTime"] = None
        changed.loc[[1, 2, 3, 10], TARGET] = np.nan
        pd.testing.assert_frame_equal(expected, self.snapshot(changed, target))

    def test_broader_context_preserves_base_features_and_ignores_fresh_reports(self):
        narrow = self.rows.loc[self.rows.trainNumber == 23].copy()
        def build(context):
            return stale_network.snapshot_features(
                self.target.drop(columns=["actualTime", TARGET]), narrow[stale.STATIC_COLUMNS],
                narrow[stale.KEY + ["actualTime", TARGET]],
                context_schedules=context[stale.STATIC_COLUMNS],
                context_observed=context[stale.KEY + ["actualTime", TARGET]])
        expected = build(self.rows)
        base = stale_enriched.snapshot_features(self.target, narrow[stale.STATIC_COLUMNS],
                                               narrow[stale.KEY + ["actualTime", TARGET]])
        pd.testing.assert_frame_equal(expected[base.columns], base)
        self.assertEqual(expected.loc[17, "network_next_station_60m_count"], 1)
        changed = self.rows.copy()
        changed.loc[[1, 2, 3, 10], TARGET] = 1e12
        changed.loc[[1, 2, 3, 10], "actualTime"] = [stamp(12, 30), stamp(12, 31), stamp(17), None]
        pd.testing.assert_frame_equal(expected, build(changed))
        changed.loc[[1, 2, 3, 10], "actualTime"] = None
        changed.loc[[1, 2, 3, 10], TARGET] = np.nan
        pd.testing.assert_frame_equal(expected, build(changed))

    def test_strict_actual_acceptance_boundaries_and_pending_cleanup(self):
        changed = self.rows.copy()
        changed.loc[5, "actualTime"] = stamp(12, 30)
        changed.loc[6, "timetableAcceptanceDate"] = stamp(12, 30)
        changed.loc[9, "actualTime"] = stamp(12, 30)
        row = self.snapshot(changed).loc[17]
        self.assertEqual(row.network_route_ahead_60m_count, 0)
        self.assertEqual(row.network_route_ahead_60m_pending_count, 3)
        self.assertEqual(row.network_input_latest_actual_time, pd.Timestamp(stamp(12, 20)))
        # A later observed completion resolves a missing report without a label.
        changed.loc[9, "actualTime"] = stamp(12, 29)
        row = self.snapshot(changed).loc[17]
        self.assertEqual(row.network_route_ahead_60m_pending_count, 2)
        self.assertEqual(row.network_input_latest_actual_time, pd.Timestamp(stamp(12, 29)))

    def test_recent_route_plans_do_not_choose_network_stations(self):
        changed = self.rows.copy()
        changed.loc[1, "timetableAcceptanceDate"] = stamp(12, 30)
        row = self.snapshot(changed).loc[17]
        # Station 300 becomes next because the station-200 plan is unavailable.
        self.assertEqual(row.network_next_station_60m_count, 1)
        self.assertEqual(row.network_next_station_60m_last_delay, 20.)
        self.assertEqual(row.network_route_ahead_180m_count, 1)

    def test_matching_train_number_on_another_date_remains_other_run(self):
        other = event(23, 100, stamp(12, 10), stamp(12, 21), 11.)
        other["departureDate"] = "2023-12-31"
        row = self.snapshot(pd.concat([self.rows, pd.DataFrame([other])], ignore_index=True)).loc[17]
        self.assertEqual(row.network_previous_station_60m_count, 2)
        self.assertEqual(row.network_previous_station_60m_last_delay, 11.)

    def test_window_lower_boundaries_are_inclusive_and_route_uses_three_stations(self):
        extra = pd.DataFrame([
            event(30, 200, stamp(11, 20), stamp(11, 30), 40.),
            event(31, 200, stamp(9, 20), stamp(9, 30), 50.),
            event(32, 200, stamp(9, 19), stamp(9, 29), 1e12),
            # This accepted plan makes station 370 the fourth route-ahead station.
            event(23, 400, stamp(12, 40), stamp(12, 50), 10.),
            event(33, 370, stamp(12), stamp(12, 10), 500.),
        ])
        row = self.snapshot(pd.concat([self.rows, extra], ignore_index=True)).loc[17]
        self.assertEqual(row.network_next_station_60m_count, 2)
        self.assertEqual(row.network_next_station_180m_count, 4)
        self.assertEqual(row.network_next_station_180m_max_delay, 50.)
        self.assertEqual(row.network_route_ahead_60m_max_delay, 40.)

    def test_enriched_fields_index_and_old_provenance_are_preserved(self):
        expected = stale_enriched.snapshot_features(self.target, self.rows[stale.STATIC_COLUMNS],
                                                    self.rows[stale.KEY + ["actualTime", TARGET]])
        result = self.snapshot()
        pd.testing.assert_frame_equal(expected, result[expected.columns])
        self.assertEqual(result.index.tolist(), [17])
        self.assertTrue(set(stale_network.NETWORK_FEATURES).difference({"trainNumber"}).issubset(result.columns))
        for column in stale_network.PROVENANCE_COLUMNS:
            self.assertLess(result.loc[17, column], result.loc[17, "observation_cutoff"])
        self.assertEqual(result.loc[17, "network_input_latest_actual_time"], pd.Timestamp(stamp(12, 25)))


if __name__ == "__main__":
    unittest.main()
