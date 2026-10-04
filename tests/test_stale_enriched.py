"""Synthetic availability and operational-feature checks."""

import unittest

import numpy as np
import pandas as pd

from fitw import stale, stale_enriched
from fitw.spec import TARGET


def stamp(hour, minute=0):
    return f"2024-01-01T{hour:02d}:{minute:02d}:00Z"


def event(hour, minute, actual, delay, station, kind="ARRIVAL", accepted=None):
    return {"departureDate": "2024-01-01", "trainNumber": 23,
            "stationUICCode": station, "type": kind, "scheduledTime": stamp(hour, minute),
            "actualTime": actual, TARGET: delay, "trainStopping": True,
            "commercialStop": True, "trainType": "IC", "timetableType": "REGULAR",
            "timetableAcceptanceDate": accepted or stamp(9)}


class EnrichedSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame([
            event(10, 0, stamp(10, 2), 2., 100),
            event(10, 5, stamp(10, 9), 4., 100, "DEPARTURE"),
            event(11, 0, stamp(11, 8), 8., 200),
            event(11, 5, stamp(11, 17), 12., 200, "DEPARTURE"),
            # Missing report already resolved by a later completion.
            event(11, 15, None, np.nan, 250),
            event(11, 20, stamp(11, 40), 20., 300, "DEPARTURE"),
            event(12, 0, stamp(12, 40), 40., 310),
            event(12, 20, None, np.nan, 320, "DEPARTURE"),
            event(12, 50, stamp(13, 20), 30., 370),
            event(13, 0, stamp(13, 30), 30., 370, "DEPARTURE"),
            event(14, 0, stamp(14, 30), 30., 400),
        ])
        self.target = self.rows.iloc[[9]].copy()
        self.target.index = [17]

    def snapshot(self, rows=None, target=None):
        rows = self.rows if rows is None else rows
        return stale_enriched.snapshot_features(self.target if target is None else target,
                                               rows[stale.STATIC_COLUMNS],
                                               rows[stale.KEY + ["actualTime", TARGET]])

    def test_pending_cleanup_and_latest_pending_are_computed_at_cutoff(self):
        row = self.snapshot().loc[17]
        self.assertEqual(row.clean_pending_count, 2)
        self.assertEqual(row.resolved_missing_count, 1)
        self.assertEqual(row.latest_pending_minutes, 10.)
        self.assertEqual(row.clean_oldest_pending_minutes, 30.)
        self.assertEqual(row.pending_scheduled_span_minutes, 20.)
        self.assertEqual(row.latest_pending_station, 320)
        self.assertEqual(row.latest_pending_is_departure, 1.)
        self.assertEqual(row.latest_pending_vs_previous_delay, -10.)

    def test_recent_history_dynamics_and_route_dwell(self):
        row = self.snapshot().loc[17]
        self.assertEqual(row.last_scheduled_delay, 20.)
        self.assertEqual(row.scheduled_order_delay_change, 8.)
        self.assertAlmostEqual(row.scheduled_order_delay_slope, 8. / 15.)
        self.assertEqual(row.recent3_mean_delay, 40. / 3.)
        self.assertEqual(row.recent3_median_delay, 12.)
        self.assertEqual(row.recent3_min_delay, 8.)
        self.assertEqual(row.recent3_max_delay, 20.)
        self.assertEqual(row.recent_delay_acceleration, 4.)
        self.assertEqual(row.history_arrival_last_delay, 8.)
        self.assertEqual(row.history_departure_last_delay, 20.)
        self.assertEqual(row.planned_total_events, 11)
        self.assertEqual(row.planned_total_duration_minutes, 240.)
        self.assertEqual(row.planned_destination_station, 400)
        self.assertEqual(row.planned_fraction_elapsed, .75)
        self.assertEqual(row.planned_dwell_after_source_minutes, 10.)
        self.assertEqual(row.delay_after_planned_dwell, 10.)
        self.assertEqual(row.next_scheduled_event_station, 310)
        self.assertEqual(row.minutes_to_next_scheduled_event, -30.)

    def test_future_actuals_labels_and_focal_labels_cannot_change_output(self):
        expected = self.snapshot()
        changed = self.rows.copy()
        changed.loc[6:10, TARGET] = 1e12
        changed.loc[6:10, "actualTime"] = [stamp(12, 31), stamp(12, 30), None, stamp(15), stamp(16)]
        target = self.target.copy()
        target[TARGET] = -1e12
        target["actualTime"] = stamp(17)
        pd.testing.assert_frame_equal(expected, self.snapshot(changed, target))
        changed.loc[6:10, "actualTime"] = None
        changed.loc[6:10, TARGET] = np.nan
        pd.testing.assert_frame_equal(expected, self.snapshot(changed, target))

    def test_exact_cutoff_observation_cannot_resolve_missing_report(self):
        changed = self.rows.copy()
        changed.loc[5, "actualTime"] = stamp(12, 30)
        row = self.snapshot(changed).loc[17]
        self.assertEqual(row.resolved_missing_count, 0)
        self.assertEqual(row.clean_pending_count, 4)
        changed.loc[5, "actualTime"] = stamp(12, 29)
        row = self.snapshot(changed).loc[17]
        self.assertEqual(row.resolved_missing_count, 1)
        self.assertEqual(row.clean_pending_count, 2)
        # Missing delay labels still permit already observed events to resolve reports.
        changed.loc[5, TARGET] = np.nan
        row = self.snapshot(changed).loc[17]
        self.assertEqual(row.resolved_missing_count, 1)
        self.assertEqual(row.clean_pending_count, 2)
        self.assertEqual(row.last_scheduled_delay, 12.)

    def test_recent_timetable_cannot_contribute_route_or_observed_features(self):
        changed = self.rows.copy()
        changed.loc[[5, 10], "timetableAcceptanceDate"] = stamp(12, 30)
        row = self.snapshot(changed).loc[17]
        self.assertEqual(row.last_scheduled_delay, 12.)
        self.assertEqual(row.planned_total_events, 9)
        self.assertEqual(row.planned_destination_station, 370)
        self.assertEqual(row.resolved_missing_count, 0)
        self.assertEqual(row.enriched_input_latest_actual_time, pd.Timestamp(stamp(11, 17)))

    def test_base_fields_and_index_are_preserved_and_provenance_is_old(self):
        base = stale.snapshot_features(self.target, self.rows[stale.STATIC_COLUMNS],
                                       self.rows[stale.KEY + ["actualTime", TARGET]])
        result = self.snapshot()
        pd.testing.assert_frame_equal(base, result[base.columns])
        self.assertEqual(result.index.tolist(), [17])
        self.assertTrue(set(stale_enriched.ENRICHED_FEATURES).difference({"trainNumber"}).issubset(result.columns))
        for name in stale_enriched.PROVENANCE_COLUMNS:
            self.assertLess(result.loc[17, name], result.loc[17, "observation_cutoff"])
        self.assertEqual(result.loc[17, "enriched_input_latest_actual_time"], pd.Timestamp(stamp(11, 40)))


if __name__ == "__main__":
    unittest.main()
