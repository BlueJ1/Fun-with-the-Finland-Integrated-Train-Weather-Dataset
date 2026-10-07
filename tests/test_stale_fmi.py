import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fitw import stale_fmi as fmi, stale


class WeatherSnapshotTests(unittest.TestCase):
    def xml_observation(self, measurement_time="2020-01-02T10:00:00Z"):
        return f'''<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0"
            xmlns:om="http://www.opengis.net/om/2.0" xmlns:gml="http://www.opengis.net/gml/3.2"
            xmlns:wml="http://www.opengis.net/waterml/2.0" xmlns:xlink="http://www.w3.org/1999/xlink"
            timeStamp="2026-10-03T12:00:00Z"><wfs:member><om:Observation>
            <om:resultTime><gml:TimeInstant><gml:timePosition>2023-09-09T12:00:00Z</gml:timePosition></gml:TimeInstant></om:resultTime>
            <om:observedProperty xlink:href="https://opendata.fmi.fi/meta?param=t2m"/>
            <gml:identifier>{fmi.FMISID}</gml:identifier><gml:pos>64.93698 25.37299</gml:pos>
            <gml:name>Oulu Oulunsalo Pellonpää</gml:name><wml:MeasurementTVP>
            <wml:time>{measurement_time}</wml:time><wml:value>-3.5</wml:value>
            </wml:MeasurementTVP></om:Observation></wfs:member></wfs:FeatureCollection>'''.encode()

    def query(self):
        return pd.DataFrame({"prediction_time": [pd.Timestamp("2020-01-02T12:30Z")]}, index=[23])

    def weather(self):
        times = pd.date_range("2020-01-01T10:00Z", "2020-01-02T11:00Z", freq="h")
        return pd.DataFrame([{"measurement_time": t, "parameter": p, "value": 2., "fmisid": fmi.FMISID}
                             for t in times for p in fmi.PARAMETERS])

    def test_exact_cutoff_one_hour_lag_future_independence(self):
        weather = self.weather()
        original = fmi.snapshot_features(self.query(), weather)
        # Snapshot12:00, one-hour lag means measurements strictly before11:00.
        self.assertEqual(original.loc[23, "fmi_latest_measurement_time"], pd.Timestamp("2020-01-02T10:00Z"))
        self.assertEqual(original.loc[23, "fmi_t2m_age_minutes"], 150.)
        changed = weather.copy()
        changed.loc[changed.measurement_time >= pd.Timestamp("2020-01-02T11:00Z"), "value"] = 10000.
        changed.loc[len(changed)] = [pd.Timestamp("2020-01-02T11:59Z"), "t2m", -10000., fmi.FMISID]
        pd.testing.assert_frame_equal(original, fmi.snapshot_features(self.query(), changed))
        self.assertEqual(original.loc[23, "fmi_r_1h_6h_sum"], 12.)

    def test_parser_ignores_response_and_result_times_and_preserves_timezone(self):
        parsed = fmi.parse_response(self.xml_observation("2020-01-02T12:00:00+02:00"))
        self.assertEqual(parsed.loc[0,"measurement_time"],pd.Timestamp("2020-01-02T10:00Z"))
        self.assertEqual(parsed.loc[0,"value"],-3.5)
        with self.assertRaisesRegex(ValueError,"explicit timezone"):
            fmi.parse_response(self.xml_observation("2020-01-02T10:00:00"))
        with self.assertRaisesRegex(ValueError,"Unexpected FMI station"):
            fmi.parse_response(self.xml_observation().replace(str(fmi.FMISID).encode(),b"101794"))

    def test_only_boundary_or_future_measurements_leave_no_present_weather(self):
        weather = self.weather()
        weather = weather[weather.measurement_time>=pd.Timestamp("2020-01-02T11:00Z")]
        result = fmi.snapshot_features(self.query(),weather)
        self.assertEqual(result.loc[23,"fmi_input_presence_count"],0)
        self.assertTrue(pd.isna(result.loc[23,"fmi_latest_measurement_time"]))
        self.assertTrue(result[[f"fmi_{p}" for p in fmi.PARAMETERS]].isna().all().all())
        self.assertTrue(result[[name for name in fmi.EXTRA_FEATURES if name.endswith("_count")]].eq(0).all().all())

    def test_duplicate_week_endpoints_do_not_inflate_rolling_coverage(self):
        weather = self.weather()
        recent = weather[weather.measurement_time>=pd.Timestamp("2020-01-02T08:00Z")]
        duplicates = pd.concat([recent,recent,recent],ignore_index=True)
        result = fmi.snapshot_features(self.query(),duplicates)
        self.assertEqual(result.loc[23,"fmi_t2m_6h_count"],3)
        self.assertTrue(np.isnan(result.loc[23,"fmi_t2m_6h_mean"]))
        pd.testing.assert_frame_equal(result,fmi.snapshot_features(self.query(),recent))

    def test_missing_coverage_does_not_become_zero_summary(self):
        weather = self.weather()
        weather = weather[weather.measurement_time >= pd.Timestamp("2020-01-02T08:00Z")]
        result = fmi.snapshot_features(self.query(), weather)
        self.assertEqual(result.loc[23, "fmi_r_1h_6h_count"], 3)
        self.assertTrue(np.isnan(result.loc[23, "fmi_r_1h_6h_sum"]))
        self.assertTrue(np.isnan(result.loc[23, "fmi_snow_aws_24h_change"]))

    def test_point_values_expire_without_future_fill(self):
        weather = self.weather()
        weather = weather[weather.measurement_time <= pd.Timestamp("2020-01-02T08:00Z")]
        result = fmi.snapshot_features(self.query(), weather)
        self.assertTrue(np.isnan(result.loc[23, "fmi_t2m"]))
        self.assertTrue(np.isnan(result.loc[23, "fmi_t2m_6h_mean"]))
        self.assertEqual(result.loc[23, "fmi_t2m_6h_count"], 4)
        self.assertGreater(result.loc[23, "fmi_input_presence_count"], 0)

    def test_negative_snow_missing_and_rain_sentinel_zero(self):
        weather = self.weather()
        weather.loc[weather.parameter.eq("snow_aws"), "value"] = -1
        weather.loc[weather.parameter.eq("r_1h"), "value"] = -1
        result = fmi.snapshot_features(self.query(), weather)
        self.assertTrue(np.isnan(result.loc[23, "fmi_snow_aws"]))
        self.assertEqual(result.loc[23, "fmi_r_1h_6h_sum"], 0.)

    def test_missing_timestamp_and_conflicting_duplicates_rejected(self):
        weather = self.weather()
        weather.loc[0, "measurement_time"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "no measurement"):
            fmi.snapshot_features(self.query(), weather)
        weather = self.weather()
        conflict = weather.iloc[[0]].copy(); conflict["value"] = 90
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            fmi.snapshot_features(self.query(), pd.concat([weather, conflict]))

    def test_request_end_clamped_and_parsing_uses_measurement_time(self):
        captured = []
        response = b'''<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0" timeStamp="2026-10-03T12:00:00Z"/>'''
        class Reply:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return response
        def request(url, **kwargs):
            captured.append(url); return Reply()
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder, patch("urllib.request.urlopen", request):
            _, meta = fmi.fetch_week(pd.Timestamp("2023-09-08T12:00Z"), pd.Timestamp("2023-09-09T12:00Z"), Path(folder))
        self.assertIn("endtime=2023-09-09T12%3A00%3A00Z", captured[0])
        self.assertEqual(meta["end"], "2023-09-09T12:00:00Z")
        self.assertEqual(len(fmi.parse_response(response)), 0)
        xml = b'''<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0"
          xmlns:om="http://www.opengis.net/om/2.0" xmlns:gml="http://www.opengis.net/gml/3.2"
          xmlns:wml="http://www.opengis.net/waterml/2.0" xmlns:xlink="http://www.w3.org/1999/xlink"
          timeStamp="2026-10-03T12:00:00Z"><wfs:member><observation>
          <om:resultTime><gml:timePosition>2026-10-03T12:00:00Z</gml:timePosition></om:resultTime>
          <om:observedProperty xlink:href="https://opendata.fmi.fi/meta?param=t2m"/>
          <gml:identifier>101799</gml:identifier><gml:name>Pellonpaa</gml:name><gml:pos>64.9 25.3</gml:pos>
          <wml:MeasurementTVP><wml:time>2020-01-01T02:00:00Z</wml:time><wml:value>-5</wml:value></wml:MeasurementTVP>
          </observation></wfs:member></wfs:FeatureCollection>'''
        parsed = fmi.parse_response(xml)
        self.assertEqual(parsed.loc[0, "measurement_time"], pd.Timestamp("2020-01-01T02:00Z"))
        self.assertEqual(parsed.loc[0, "value"], -5.)

    def test_manifest_enforces_assumed_lag_and_missing_provenance(self):
        result = fmi.snapshot_features(self.query(), self.weather())
        manifest = {"additional_provenance_columns": fmi.PROVENANCE_COLUMNS,
                    "provenance_count_columns": fmi.PROVENANCE_COUNT_COLUMNS,
                    "additional_provenance_lag_minutes": fmi.PROVENANCE_LAG_MINUTES}
        cutoff = self.query().prediction_time - pd.Timedelta(minutes=30)
        stale.validate_additional_provenance(result, manifest, cutoff)
        result["fmi_latest_measurement_time"] = pd.Timestamp("2020-01-02T11:00Z")
        with self.assertRaisesRegex(ValueError, "age violation"):
            stale.validate_additional_provenance(result, manifest, cutoff)
        result["fmi_latest_measurement_time"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "require timestamp"):
            stale.validate_additional_provenance(result, manifest, cutoff)


if __name__ == "__main__":
    unittest.main()
