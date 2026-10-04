import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fitw import stale_origin as origin, stale_origin_boundary as boundary
from fitw.spec import TARGET


class BoundaryContextTests(unittest.TestCase):
    def sample(self):
        records = []
        for date, num, station, kind, time in [
            ('2020-02-01', 2, 900, 'DEPARTURE', '2020-02-01T01:00:00Z'),
            ('2020-02-01', 2, 370, 'ARRIVAL', '2020-02-01T03:00:00Z'),
            ('2020-01-31', 1, 370, 'DEPARTURE', '2020-01-31T21:00:00Z'),
            ('2020-01-31', 1, 900, 'ARRIVAL', '2020-01-31T23:00:00Z'),
        ]:
            records.append(dict(departureDate=date, trainNumber=num, stationUICCode=station,
                                type=kind, scheduledTime=time, trainCategory='Long-distance',
                                timetableAcceptanceDate='2019-12-01T00:00:00Z'))
        schedules = pd.DataFrame(records)
        observed = schedules[origin.stale.KEY].copy()
        observed['actualTime'] = ['2020-02-01T01:05:00Z', '2020-02-01T03:05:00Z',
                                  '2020-01-31T21:40:00Z', '2020-02-01T02:30:00Z']
        observed[TARGET] = [5., 5., 40., 210.]
        return schedules.iloc[[1]].copy(), schedules.iloc[:2].copy(), observed.iloc[:2].copy(), schedules.iloc[2:].copy(), observed.iloc[2:].copy()

    def combine(self, schedules, observed, prior, prior_observed):
        return boundary.append_prior_runs(schedules, observed, prior, prior_observed, pd.Timestamp('2020-02-01', tz='UTC'))

    def test_boundary_candidate_strict_old_observation_and_focal_exclusion(self):
        query, schedules, observed, prior, prior_observed = self.sample()
        without = origin.snapshot_features(query, schedules, observed, reverse_features=True)
        combined, reports, stats = self.combine(schedules, observed, prior, prior_observed)
        with_prior = origin.snapshot_features(query, combined, reports, reverse_features=True)
        self.assertEqual(without.iloc[0].origin_context_720m_count, 0.)
        self.assertEqual(with_prior.iloc[0].origin_context_720m_count, 1.)
        self.assertEqual(with_prior.iloc[0].origin_context_720m_closest_last_delay, 40.)
        self.assertEqual(with_prior.iloc[0].origin_context_720m_unobserved_count, 1.)
        self.assertEqual(stats['prior_dated_runs_appended'], 1)
        changed = reports.copy()
        # An early focal report must still be excluded. Future candidate reports
        # and their target values cannot affect any input or provenance field.
        changed.loc[0:1, 'actualTime'] = '2020-02-01T00:01:00Z'
        changed.loc[0:1, TARGET] = 1000000.
        changed.loc[3, 'actualTime'] = '2020-02-02T00:00:00Z'
        changed.loc[3, TARGET] = 1000000.
        pd.testing.assert_frame_equal(with_prior, origin.snapshot_features(query, combined, changed, reverse_features=True))
        reports.loc[3, 'actualTime'] = '2020-02-01T02:29:59Z'
        old = origin.snapshot_features(query, combined, reports, reverse_features=True)
        self.assertEqual(old.iloc[0].origin_context_720m_closest_last_delay, 210.)
        self.assertEqual(old.iloc[0].origin_context_720m_unobserved_count, 0.)

    def test_route_maximum_acceptance_and_current_revision_precedence(self):
        query, schedules, observed, prior, prior_observed = self.sample()
        # A duplicate prior copy of a current-source run must never fill or
        # replace its route or observation, even if the old copy is corrupted.
        duplicate = schedules.copy()
        duplicate['scheduledTime'] = ['2020-01-31T20:00:00Z', '2020-01-31T22:00:00Z']
        duplicate['stationUICCode'] = [123, 456]
        duplicated_reports = observed.copy()
        duplicated_reports[TARGET] = 1000000.
        combined, reports, stats = self.combine(schedules, observed,
                                                pd.concat([prior, duplicate]),
                                                pd.concat([prior_observed, duplicated_reports]))
        pd.testing.assert_frame_equal(combined.iloc[:2], schedules)
        pd.testing.assert_frame_equal(reports.iloc[:2], observed)
        # Duplicate dated February runs are excluded by the date rule too.
        self.assertEqual(len(combined), 4)
        prior.loc[prior.index[0], 'timetableAcceptanceDate'] = '2020-02-01T02:30:00Z'
        combined, reports, _ = self.combine(schedules, observed, prior, prior_observed)
        row = origin.snapshot_features(query, combined, reports, reverse_features=True).iloc[0]
        self.assertEqual(row.origin_context_720m_count, 0.)
        self.assertTrue(pd.isna(row.origin_context_latest_actual_time))

    def test_prior_source_reads_only_previous_month_and_strict_bounded_reports(self):
        query, schedules, observed, prior, prior_observed = self.sample()
        query['row_id'] = 0
        with tempfile.TemporaryDirectory() as directory:
            current = schedules.merge(observed, on=origin.stale.KEY)
            previous = prior.merge(prior_observed, on=origin.stale.KEY)
            current.to_parquet(Path(directory) / 'matched_data_flat_2020_02.parquet')
            previous.to_parquet(Path(directory) / 'matched_data_flat_2020_01.parquet')
            # A later-month file is deliberately invalid and must not be read.
            (Path(directory) / 'matched_data_flat_2020_03.parquet').write_text('invalid parquet')
            combined, reports, source = boundary.read_context(directory, 'matched_data_flat_2020_02.parquet', query,
                                                              pd.Timestamp('2020-02-01T23:00:00Z'))
            self.assertEqual(source['previous']['file'], 'matched_data_flat_2020_01.parquet')
            self.assertEqual(source['previous']['departure_date_read_before'], '2020-02-01')
            self.assertTrue((pd.to_datetime(reports.actualTime, utc=True) < pd.Timestamp(source['current']['observation_read_before'])).all())
            self.assertEqual(len(combined), 4)
            self.assertEqual(source['previous']['observed_rows_read'], 1)
            self.assertEqual(source['current']['observed_rows'], 1)
            self.assertEqual(boundary.previous_source('matched_data_flat_2020_01.parquet')[0], 'matched_data_flat_2019_12.parquet')

    def test_disjoint_run_exclusion_covers_entire_prior_route(self):
        _, schedules, observed, prior, prior_observed = self.sample()
        schedules = pd.concat([schedules, prior.iloc[[0]]], ignore_index=True)
        observed = pd.concat([observed, prior_observed.iloc[[0]]], ignore_index=True)
        combined, reports, stats = self.combine(schedules, observed, prior, prior_observed)
        pd.testing.assert_frame_equal(combined, schedules)
        pd.testing.assert_frame_equal(reports, observed)
        self.assertEqual(stats['prior_overlap_route_rows_excluded'], 2)
        self.assertEqual(stats['prior_dated_runs_appended'], 0)


if __name__ == '__main__':
    unittest.main()
