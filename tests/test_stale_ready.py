import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fitw import stale, stale_ready as ready


class ReadinessTests(unittest.TestCase):
    def sample(self):
        records = []
        for date, train, station, kind, time in [
            ('2020-02-01', 1, 370, 'DEPARTURE', '2020-02-01T07:00:00Z'),
            ('2020-02-01', 1, 900, 'ARRIVAL', '2020-02-01T09:00:00Z'),
            ('2020-02-01', 2, 900, 'DEPARTURE', '2020-02-01T10:00:00Z'),
            ('2020-02-01', 2, 370, 'ARRIVAL', '2020-02-01T12:00:00Z'),
            ('2020-01-31', 3, 370, 'DEPARTURE', '2020-01-31T21:00:00Z'),
            ('2020-01-31', 3, 900, 'ARRIVAL', '2020-01-31T23:00:00Z'),
        ]:
            records.append(dict(departureDate=date, trainNumber=train, stationUICCode=station, type=kind,
                                scheduledTime=time, timetableAcceptanceDate='2019-12-01T00:00:00Z', trainCategory='Long-distance'))
        schedules = pd.DataFrame(records)
        observed = schedules[stale.KEY].copy()
        observed['actualTime'] = ['2020-02-01T07:20:00Z', '2020-02-01T09:20:00Z',
                                  '2020-02-01T10:10:00Z', '2020-02-01T12:10:00Z',
                                  '2020-01-31T21:15:00Z', '2020-01-31T23:15:00Z']
        decisions = schedules.iloc[[0, 2, 4]][stale.KEY].copy()
        decisions['ready_timestamp'] = pd.to_datetime(['2020-02-01T07:10:00Z', '2020-02-01T10:05:00Z', '2020-01-31T21:05:00Z'], utc=True)
        return schedules.iloc[[3]], schedules, observed, decisions

    def test_summaries_from_nonfocal_decisions_and_old_actuals(self):
        q, s, o, d = self.sample()
        row = ready.snapshot_features(q, s, o, d).iloc[0]
        self.assertEqual(row.ready_own_count, 1.)
        self.assertEqual(row.ready_own_last_delay_minutes, 5.)
        self.assertEqual(row.ready_own_mean_delay_minutes, 5.)
        self.assertEqual(row.ready_own_last_age_minutes, 85.)
        self.assertEqual(row.ready_own_last_to_actual_minutes, 5.)
        self.assertEqual(row.ready_origin_180m_count, 1.)
        self.assertEqual(row.ready_origin_180m_closest_delay_minutes, 10.)
        self.assertEqual(row.ready_origin_720m_count, 2.)
        self.assertEqual(row.ready_origin_720m_mean_delay_minutes, 7.5)
        self.assertEqual(row.ready_origin_720m_mean_to_actual_minutes, 10.)
        self.assertEqual(row.ready_latest_decision_time, pd.Timestamp('2020-02-01T10:05:00Z'))
        self.assertEqual(row.ready_latest_actual_time, pd.Timestamp('2020-02-01T10:10:00Z'))

    def test_focal_and_target_perturbations_do_not_affect_any_input(self):
        q, s, o, d = self.sample()
        baseline = ready.snapshot_features(q, s, o, d)
        o['differenceInMinutes'] = 1e9
        o.loc[3, 'actualTime'] = '2020-02-01T09:00:00Z'
        # Even a fabricated old readiness notification on the focal arrival is
        # excluded; no final delay value is required by this feature builder.
        focal = s.iloc[[3]][stale.KEY].copy()
        focal['ready_timestamp'] = pd.to_datetime(['2020-02-01T09:00:00Z'], utc=True)
        pd.testing.assert_frame_equal(baseline, ready.snapshot_features(q, s, o, pd.concat([d, focal])))
        # Test focal departure exclusion while other origin context is present.
        q = s.iloc[[2]]
        d.loc[2, 'ready_timestamp'] = pd.Timestamp('2020-02-01T09:00:00Z')
        before = ready.snapshot_features(q, s, o, d)
        self.assertTrue(before[[f'ready_own_{name}' for name in ready.OWN_SUMMARIES]].isna().all().all())
        d.loc[2, 'ready_timestamp'] = pd.Timestamp('2020-02-01T08:00:00Z')
        o.loc[2, 'actualTime'] = '2020-02-01T08:01:00Z'
        pd.testing.assert_frame_equal(before, ready.snapshot_features(q, s, o, d))

    def test_strict_readiness_and_actual_cutoffs_and_future_perturbation(self):
        q, s, o, d = self.sample()
        d.loc[0, 'ready_timestamp'] = pd.Timestamp('2020-02-01T11:30:00Z')
        equality = ready.snapshot_features(q, s, o, d)
        self.assertTrue(pd.isna(equality.iloc[0].ready_origin_180m_count))
        d.loc[0, 'ready_timestamp'] = pd.Timestamp('2099-01-01T00:00:00Z')
        pd.testing.assert_frame_equal(equality, ready.snapshot_features(q, s, o, d))
        d.loc[0, 'ready_timestamp'] = pd.Timestamp('2020-02-01T11:29:59Z')
        old = ready.snapshot_features(q, s, o, d)
        self.assertEqual(old.iloc[0].ready_origin_180m_count, 1.)
        o.loc[0, 'actualTime'] = '2020-02-01T11:30:00Z'
        equality_actual = ready.snapshot_features(q, s, o, d)
        self.assertTrue(pd.isna(equality_actual.iloc[0].ready_origin_180m_closest_to_actual_minutes))
        o.loc[0, 'actualTime'] = '2099-01-01T00:00:00.000Z'
        pd.testing.assert_frame_equal(equality_actual, ready.snapshot_features(q, s, o, d))
        o.loc[0, 'actualTime'] = '2020-02-01T11:29:59Z'
        self.assertEqual(ready.snapshot_features(q, s, o, d).iloc[0].ready_origin_180m_closest_to_actual_minutes, 0.)

    def test_maximum_route_acceptance_and_unknown_readiness(self):
        q, s, o, d = self.sample()
        s.loc[1, 'timetableAcceptanceDate'] = '2020-02-01T11:30:00Z'
        row = ready.snapshot_features(q, s, o, d).iloc[0]
        self.assertTrue(pd.isna(row.ready_origin_180m_count))
        self.assertEqual(row.ready_origin_720m_count, 1.)
        s.loc[2, 'timetableAcceptanceDate'] = '2020-02-01T11:30:00Z'
        unknown = ready.snapshot_features(q, s, o, d)
        self.assertTrue(unknown[ready.FEATURES + ready.STAMPS].isna().all().all())
        q, s, o, d = self.sample()
        absent = ready.snapshot_features(q, s, o, d.iloc[:0])
        self.assertTrue(absent[ready.FEATURES + ready.STAMPS].isna().all().all())
        self.assertTrue(absent[[name[0] for name in ready.COUNTS.values()]].eq(0).all().all())
        raw_unknown = s[stale.KEY].copy()
        raw_unknown['trainReady'] = 'nan'
        parsed, _ = ready.parse_readiness(raw_unknown, pd.Timestamp('2020-02-01T11:30:00Z'))
        pd.testing.assert_frame_equal(absent, ready.snapshot_features(q, s, o, parsed))

    def test_safe_parser_accepts_only_true_timestamped_old_decisions(self):
        _, s, _, _ = self.sample()
        raw = pd.concat([s.iloc[[0]]] * 8, ignore_index=True)[stale.KEY]
        raw['trainReady'] = [
            "{'source':'KUPLA','accepted':True,'timestamp':'2020-02-01T08:00:00Z'}",
            json.dumps({'source': 'PHONE', 'accepted': True, 'timestamp': '2020-02-01T11:30:00Z'}),
            json.dumps({'source': 'PHONE', 'accepted': False, 'timestamp': '2020-02-01T08:00:00Z'}),
            'nan', None, 'true', json.dumps({'accepted': True}),
            "__import__('pathlib').Path('/private/tmp/ready-parser-must-not-execute').touch()",
        ]
        parsed, stats = ready.parse_readiness(raw, pd.Timestamp('2020-02-01T11:30:00Z'))
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed.iloc[0].ready_timestamp, pd.Timestamp('2020-02-01T08:00:00Z'))
        self.assertFalse(Path('/private/tmp/ready-parser-must-not-execute').exists())
        self.assertEqual(stats['accepted_timestamp_records'], 2)

    def test_bounded_source_read_needs_no_target_or_excluded_fields(self):
        q, s, o, d = self.sample()
        raw = s.merge(o, on=stale.KEY)
        raw['trainReady'] = 'nan'
        for index, row in d.iterrows():
            raw.loc[index, 'trainReady'] = json.dumps({'accepted': True, 'source': 'KUPLA', 'timestamp': row.ready_timestamp.isoformat()})
        with tempfile.TemporaryDirectory() as directory:
            raw.iloc[:4].to_parquet(Path(directory) / 'matched_data_flat_2020_02.parquet')
            raw.iloc[4:].to_parquet(Path(directory) / 'matched_data_flat_2020_01.parquet')
            (Path(directory) / 'matched_data_flat_2020_03.parquet').write_text('invalid future parquet')
            schedules, observed, decisions, source = ready.read_context(directory, 'matched_data_flat_2020_02.parquet', q,
                                                                      pd.Timestamp('2020-02-01T23:00:00Z'))
            self.assertEqual(set(observed), set(stale.KEY + ['actualTime']))
            self.assertEqual(len(schedules), 6)
            self.assertEqual(len(decisions), 3)
            self.assertTrue((pd.to_datetime(observed.actualTime, utc=True) < pd.Timestamp('2020-02-01T11:30:00Z')).all())
            self.assertTrue(decisions.ready_timestamp.lt(pd.Timestamp('2020-02-01T11:30:00Z')).all())
            self.assertEqual(source['previous']['file'], 'matched_data_flat_2020_01.parquet')
            self.assertEqual(source['previous']['prior_dated_runs_appended'], 1)
            self.assertEqual(source['previous']['prior_readiness_rows_appended'], 1)
            actual = ready.snapshot_features(q, schedules, observed, decisions)
            expected = ready.snapshot_features(q, s, o, d)
            pd.testing.assert_frame_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
