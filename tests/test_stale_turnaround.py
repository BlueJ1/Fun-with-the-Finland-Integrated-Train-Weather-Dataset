"""Synthetic Oulu context, snapshot age, and turnaround-proxy checks."""

import unittest
import numpy as np
import pandas as pd

from fitw import stale, stale_turnaround
from fitw.spec import TARGET


def stamp(hour, minute=0):
    return f"2024-01-01T{hour:02d}:{minute:02d}:00Z"


def event(number, station, scheduled, actual, delay, kind="ARRIVAL", category="Long-distance", accepted=None):
    return {"departureDate": "2024-01-01", "trainNumber": number, "stationUICCode": station,
            "type": kind, "scheduledTime": scheduled, "actualTime": actual, TARGET: delay,
            "trainCategory": category, "timetableAcceptanceDate": accepted or stamp(9)}


class OuluContextTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame([
            event(23, 370, stamp(12, 55), stamp(13, 10), 15.),
            event(23, 370, stamp(13), stamp(13, 20), 20., "DEPARTURE"),
            event(23, 400, stamp(14), stamp(14, 20), 20.),
            # Pending inbound with an older report, and a known Oulu terminus.
            event(24, 100, stamp(11), stamp(11, 30), 30.),
            event(24, 370, stamp(12, 15), stamp(12, 50), 35.),
            # Closest plausible inbound is scheduled after the old cutoff.
            event(25, 300, stamp(11, 10), stamp(11, 20), 10.),
            event(25, 370, stamp(12, 35), stamp(13), 25.),
            # Passing service contributes station traffic but not turnaround.
            event(26, 370, stamp(12, 10), stamp(12, 20), 10.),
            event(26, 370, stamp(12, 15), stamp(12, 25), 10., "DEPARTURE"),
            event(26, 400, stamp(13), stamp(13, 10), 10.),
            # Freight contributes station context but is excluded as a stock proxy.
            event(27, 370, stamp(12, 5), stamp(12, 15), 10., category="Freight"),
            # Already observed later event resolves an earlier missing report.
            event(28, 370, stamp(12, 1), None, np.nan),
            event(28, 500, stamp(12, 10), stamp(12, 25), np.nan, "DEPARTURE"),
        ])
        self.target = self.rows.iloc[[1]][stale.KEY].copy()
        self.target.index = [17]

    def snapshot(self, rows=None, targets=None, base=None):
        rows = self.rows if rows is None else rows
        return stale_turnaround.snapshot_features(self.target if targets is None else targets,
                                                  rows[stale_turnaround.STATIC_COLUMNS],
                                                  rows[stale.KEY + ["actualTime", TARGET]], base)

    def test_all_category_station_context_excludes_same_run_and_resolves_missing_reports(self):
        row = self.snapshot().loc[17]
        self.assertEqual(row.oulu_context_arrival_60m_observed_count, 2)
        self.assertEqual(row.oulu_context_arrival_60m_mean_delay, 10.)
        self.assertEqual(row.oulu_context_departure_60m_observed_count, 1)
        self.assertEqual(row.oulu_context_arrival_60m_pending_count, 1)
        self.assertEqual(row.oulu_context_arrival_60m_resolved_count, 1)
        self.assertEqual(row.oulu_context_arrival_60m_oldest_pending_minutes, 15.)
        self.assertEqual(row.oulu_terminating_pending_count, 1)
        self.assertEqual(row.oulu_terminating_recent_count, 0)

    def test_origin_candidate_uses_accepted_terminal_route_and_old_inbound_delay(self):
        row = self.snapshot().loc[17]
        self.assertEqual(row.oulu_origin_departure, 1.)
        self.assertEqual(row.turnaround_candidate_count, 2)
        self.assertEqual(row.turnaround_candidate_unobserved_count, 2)
        self.assertEqual(row.turnaround_candidate_scheduled_gap_minutes, 25.)
        self.assertEqual(row.turnaround_candidate_last_known_delay, 10.)
        self.assertEqual(row.turnaround_candidate_observation_age_minutes, 70.)
        self.assertEqual(row.turnaround_candidate_pending_minutes, 0.)
        self.assertEqual(row.turnaround_candidate_projected_departure_delay, 15.)
        self.assertEqual(row.turnaround_worst_projected_departure_delay, 15.)
        # A through arrival/departure has the same station context but no origin pairing.
        target = self.target.copy()
        target["trainNumber"] = 26
        target["scheduledTime"] = stamp(12, 15)
        rows = self.rows.copy()
        rows = pd.concat([rows, pd.DataFrame([event(26, 100, stamp(10), stamp(10, 5), 5.)])], ignore_index=True)
        row = self.snapshot(rows, target).loc[17]
        self.assertEqual(row.oulu_origin_departure, 0.)
        self.assertTrue(pd.isna(row.turnaround_candidate_count))

    def test_focal_and_post_snapshot_outcomes_do_not_change_any_features(self):
        expected = self.snapshot()
        changed = self.rows.copy()
        changed.loc[[0, 1, 2, 4, 6, 9], "actualTime"] = [stamp(12, 30), stamp(17), None, stamp(12, 31), stamp(18), stamp(19)]
        changed.loc[[0, 1, 2, 4, 6, 9], TARGET] = 1e12
        pd.testing.assert_frame_equal(expected, self.snapshot(changed))
        changed.loc[[0, 1, 2, 4, 6, 9], "actualTime"] = None
        changed.loc[[0, 1, 2, 4, 6, 9], TARGET] = np.nan
        pd.testing.assert_frame_equal(expected, self.snapshot(changed))
        # Even old completions of the focal dated run remain excluded.
        changed = self.rows.copy()
        changed.loc[[0, 1, 2], "actualTime"] = stamp(12)
        changed.loc[[0, 1, 2], TARGET] = -1e12
        pd.testing.assert_frame_equal(expected, self.snapshot(changed))

    def test_exact_acceptance_and_actual_cutoff_are_excluded(self):
        changed = self.rows.copy()
        changed.loc[5, "actualTime"] = stamp(12, 30)
        changed.loc[3, "timetableAcceptanceDate"] = stamp(12, 30)
        changed.loc[12, "actualTime"] = stamp(12, 30)
        row = self.snapshot(changed).loc[17]
        self.assertEqual(row.turnaround_candidate_count, 1)
        self.assertTrue(pd.isna(row.turnaround_candidate_last_known_delay))
        self.assertEqual(row.oulu_context_arrival_60m_resolved_count, 0)
        self.assertEqual(row.oulu_context_arrival_60m_pending_count, 2)
        # Recent final route rows cannot establish a terminal endpoint.
        recent_continuation = event(25, 600, stamp(14), stamp(14, 10), 10., accepted=stamp(12, 30))
        changed = pd.concat([self.rows, pd.DataFrame([recent_continuation])], ignore_index=True)
        row = self.snapshot(changed).loc[17]
        self.assertEqual(row.turnaround_candidate_count, 1)
        self.assertEqual(row.turnaround_candidate_last_known_delay, 30.)

    def test_unknown_origin_timetable_masks_turnaround_but_retains_old_station_context(self):
        changed = self.rows.copy()
        changed.loc[1, "timetableAcceptanceDate"] = stamp(12, 30)
        row = self.snapshot(changed).loc[17]
        self.assertTrue(pd.isna(row.oulu_origin_departure))
        self.assertTrue(pd.isna(row.turnaround_candidate_count))
        self.assertEqual(row.oulu_context_arrival_60m_observed_count, 2)

    def test_preservation_provenance_and_missing_provenance_rejection(self):
        base = pd.DataFrame({"existing_feature": [123.], "existing_metadata": ["unchanged"]}, index=[17])
        result = self.snapshot(base=base)
        pd.testing.assert_frame_equal(result[base.columns], base)
        cutoff = pd.Series(pd.Timestamp(stamp(12, 30)), index=result.index)
        manifest = {"additional_provenance_columns": stale_turnaround.PROVENANCE_COLUMNS,
                    "provenance_count_columns": stale_turnaround.PROVENANCE_COUNT_COLUMNS}
        stale.validate_additional_provenance(result, manifest, cutoff)
        for name in stale_turnaround.PROVENANCE_COLUMNS:
            self.assertLess(result.loc[17, name], cutoff.loc[17])
            changed = result.copy()
            changed[name] = pd.NaT
            with self.assertRaises(ValueError):
                stale_turnaround.validate_context(changed, cutoff)
            with self.assertRaises(ValueError):
                stale.validate_additional_provenance(changed, manifest, cutoff)
            changed[name] = cutoff
            with self.assertRaises(ValueError):
                stale_turnaround.validate_context(changed, cutoff)
        changed = self.rows.copy()
        changed["actualTime"] = None
        changed[TARGET] = np.nan
        row = self.snapshot(changed).loc[17]
        self.assertTrue(pd.isna(row.oulu_context_latest_actual_time))
        self.assertLess(row.oulu_context_latest_timetable_time, pd.Timestamp(stamp(12, 30)))


if __name__ == "__main__":
    unittest.main()
