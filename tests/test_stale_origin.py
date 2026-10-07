import unittest
import numpy as np
import pandas as pd
from fitw import stale_origin as origin
from fitw.spec import TARGET


class OriginContextTests(unittest.TestCase):
    def sample(self):
        rows=[]
        for num, station, kind, hour in [(1,370,'DEPARTURE',7),(1,900,'ARRIVAL',9),
                                         (2,900,'DEPARTURE',10),(2,370,'ARRIVAL',12)]:
            rows.append(dict(departureDate='2020-01-01',trainNumber=num,stationUICCode=station,type=kind,
                             scheduledTime=f'2020-01-01T{hour:02d}:00:00Z',timetableAcceptanceDate='2019-12-01T00:00:00Z',
                             trainCategory='Long-distance'))
        schedules=pd.DataFrame(rows)
        observed=schedules[origin.stale.KEY].copy()
        observed['actualTime']=['2020-01-01T07:40:00Z','2020-01-01T13:00:00Z',
                                '2020-01-01T13:30:00Z','2020-01-01T15:30:00Z']
        observed[TARGET]=[40.,240.,210.,210.]
        q=schedules.iloc[[3]].copy()
        return q,schedules,observed

    def test_future_outcome_perturbation_and_focal_exclusion(self):
        q,s,o=self.sample();base=origin.snapshot_features(q,s,o)
        altered=o.copy();altered.loc[1:,'actualTime']='2020-01-02T00:00:00Z';altered.loc[1:,TARGET]=99999.
        pd.testing.assert_frame_equal(base,origin.snapshot_features(q,s,altered))
        self.assertEqual(base.iloc[0].origin_context_720m_closest_last_delay,40.)
        self.assertEqual(base.iloc[0].origin_context_720m_closest_pending_minutes,150.)
        self.assertEqual(base.iloc[0].origin_context_720m_closest_pending_projection,120.)
        # Focal actual could be unusually early: it still must not contribute.
        altered.loc[3,'actualTime']='2020-01-01T11:00:00Z'
        pd.testing.assert_frame_equal(base,origin.snapshot_features(q,s,altered))

    def test_strict_equality_and_old_terminal_completion(self):
        q,s,o=self.sample();o.loc[1,'actualTime']='2020-01-01T11:30:00Z'
        before=origin.snapshot_features(q,s,o)
        self.assertEqual(before.iloc[0].origin_context_720m_unobserved_count,1.)
        o.loc[1,'actualTime']='2020-01-01T11:29:59Z'
        after=origin.snapshot_features(q,s,o)
        self.assertEqual(after.iloc[0].origin_context_720m_unobserved_count,0.)
        self.assertEqual(after.iloc[0].origin_context_720m_closest_pending_minutes,0.)
        self.assertEqual(after.iloc[0].origin_context_720m_closest_last_delay,240.)

    def test_unknown_route_and_recent_candidate_acceptance(self):
        q,s,o=self.sample();s.loc[2:,'timetableAcceptanceDate']='2020-01-01T11:30:00Z'
        unknown=origin.snapshot_features(q,s,o)
        self.assertTrue(unknown[origin.FEATURES].isna().all().all())
        self.assertTrue(unknown[origin.STAMPS].isna().all().all())
        q,s,o=self.sample();s.loc[1,'timetableAcceptanceDate']='2020-01-01T11:30:00Z'
        row=origin.snapshot_features(q,s,o).iloc[0]
        self.assertEqual(row.origin_context_720m_count,0.)
        self.assertTrue(pd.isna(row.origin_context_latest_actual_time))

    def test_reverse_flag_preserves_every_v1_field_and_chooses_reverse(self):
        q,s,o=self.sample()
        # A later terminating service is nearest overall, but it begins at a
        # different station than the focal route ends: it is not reverse context.
        extra=s.iloc[:2].copy();extra['trainNumber']=3
        extra['stationUICCode']=[901,900]
        extra['scheduledTime']=['2020-01-01T08:00:00Z','2020-01-01T09:30:00Z']
        reports=extra[origin.stale.KEY].copy()
        reports['actualTime']=['2020-01-01T08:02:00Z','2020-01-01T09:35:00Z']
        reports[TARGET]=[2.,5.]
        s=pd.concat([s,extra],ignore_index=True);o=pd.concat([o,reports],ignore_index=True)
        v1=origin.snapshot_features(q,s,o)
        v2=origin.snapshot_features(q,s,o,reverse_features=True)
        pd.testing.assert_frame_equal(v1,v2[v1.columns])
        self.assertEqual(set(v2.columns)-set(v1.columns),set(origin.REVERSE_FEATURES))
        self.assertEqual(v1.iloc[0].origin_context_180m_closest_last_delay,5.)
        for window in (180,720):
            prefix=f'origin_reverse_{window}m_'
            row=v2.iloc[0]
            self.assertEqual(row[prefix+'closest_gap_minutes'],60.)
            self.assertEqual(row[prefix+'closest_last_delay'],40.)
            self.assertEqual(row[prefix+'closest_projected_delay'],10.)
            self.assertEqual(row[prefix+'closest_observation_age_minutes'],230.)
            self.assertEqual(row[prefix+'closest_pending_minutes'],150.)
            self.assertTrue(pd.isna(row[prefix+'closest_terminal_completion_age_minutes']))
            self.assertTrue(pd.isna(row[prefix+'closest_terminal_to_planned_departure_minutes']))

    def test_reverse_terminal_age_requires_strict_old_report(self):
        q,s,o=self.sample();o.loc[1,'actualTime']='2020-01-01T11:30:00Z'
        at=origin.snapshot_features(q,s,o,reverse_features=True)
        self.assertTrue(pd.isna(at.iloc[0].origin_reverse_180m_closest_terminal_completion_age_minutes))
        future=o.copy();future.loc[1:,'actualTime']='2026-10-04T00:00:00Z';future.loc[1:,TARGET]=1e9
        pd.testing.assert_frame_equal(at,origin.snapshot_features(q,s,future,reverse_features=True))
        o.loc[1,'actualTime']='2020-01-01T11:29:00Z'
        old=origin.snapshot_features(q,s,o,reverse_features=True)
        self.assertEqual(old.iloc[0].origin_reverse_180m_closest_terminal_completion_age_minutes,1.)
        self.assertEqual(old.iloc[0].origin_reverse_180m_closest_terminal_to_planned_departure_minutes,-89.)
        self.assertEqual(old.iloc[0].origin_reverse_180m_closest_last_delay,240.)
        self.assertEqual(old.iloc[0].origin_reverse_180m_closest_pending_minutes,0.)
        v1=origin.snapshot_features(q,s,o)
        pd.testing.assert_frame_equal(v1,old[v1.columns])

    def test_reverse_candidate_recent_acceptance_or_missing_terminal_masked(self):
        q,s,o=self.sample();s.loc[0,'timetableAcceptanceDate']='2020-01-01T11:30:00Z'
        row=origin.snapshot_features(q,s,o,reverse_features=True).iloc[0]
        self.assertTrue(row[origin.REVERSE_FEATURES].isna().all())
        q,s,o=self.sample();o.loc[1,'actualTime']=None
        row=origin.snapshot_features(q,s,o,reverse_features=True).iloc[0]
        self.assertTrue(pd.isna(row.origin_reverse_720m_closest_terminal_completion_age_minutes))
        self.assertEqual(row.origin_reverse_720m_closest_last_delay,40.)

if __name__=='__main__': unittest.main()
