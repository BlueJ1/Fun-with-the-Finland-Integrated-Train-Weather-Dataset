"""Native backend reloading, snapshot availability, and policy helper checks."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fitw import stale, stale_ensemble
from fitw.prepare import sha256
from fitw.spec import TARGET


def synthetic_frame():
    count = 132
    frame = pd.DataFrame({name: np.zeros(count) for name in stale.MODEL_FEATURES})
    frame["row_id"] = np.arange(count)
    frame["prediction_time"] = pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC")
    frame["scheduledTime"] = frame.prediction_time
    frame["event_time"] = frame.prediction_time + pd.Timedelta(minutes=5)
    frame["observation_cutoff"] = frame.prediction_time - pd.Timedelta(minutes=30)
    frame["prediction_cutoff"] = frame.observation_cutoff
    frame["departureDate"] = frame.prediction_time.dt.strftime("%Y-%m-%d")
    frame["type"] = "ARRIVAL"
    frame["query_train_number"] = 23
    frame["trainNumber"] = 23.
    frame["metadata_known"] = True
    frame["history_count"] = 1.
    frame["previous_delay"] = np.arange(count) % 7
    frame["oldest_pending_minutes"] = np.where(np.arange(count) % 20 == 0, 120., 0.)
    frame[TARGET] = frame.previous_delay + np.arange(count) % 3
    for name in stale_ensemble.BASE_PROVENANCE:
        frame[name] = frame.prediction_time - pd.Timedelta(hours=2)
    frame["source_departure_date"] = frame.departureDate
    frame["source_train_number"] = frame.query_train_number
    return frame


class EnsembleRuleTests(unittest.TestCase):
    def test_only_final_backend_families_are_supported(self):
        for family, extension in [("xgboost","ubj"),("catboost","cbm"),("lightgbm","txt"),
                                  ("ridge","joblib"),("neural","joblib")]:
            self.assertEqual(stale_ensemble._extension({"family":family}),extension)
        for family in ["histogram","extra_trees","unknown"]:
            with self.assertRaises(ValueError):
                stale_ensemble._family({"family":family})

    def test_catboost_robust_loss_is_preserved(self):
        try:
            stale_ensemble._optional_module("catboost")
        except ImportError:
            self.skipTest("Optional CatBoost runtime unavailable")
        frame = pd.DataFrame({"previous_delay":np.arange(20,dtype=np.float32)})
        outcome = np.sin(np.arange(20))*4
        outcome[-1] = 1000.
        for loss in ["Huber:delta=15", "Lq:q=1.5"]:
            choice = {"mode":"direct", "parameters":{"iterations":5,"depth":2,"learning_rate":.1,"loss_function":loss}}
            component = {"family":"catboost","configuration":choice,"features":["previous_delay"],"categorical_features":[]}
            ensemble = stale_ensemble._estimator(component,1).fit(frame,outcome)
            source = stale_ensemble._optional_module("catboost").CatBoostRegressor(
                **choice["parameters"],cat_features=[],thread_count=1,random_seed=42,has_time=True,
                allow_writing_files=False,verbose=False).fit(frame,outcome)
            np.testing.assert_array_equal(ensemble.predict(frame,thread_count=1),source.predict(frame,thread_count=1))
            self.assertEqual(ensemble.get_param("loss_function"),loss)

    def test_pending_baseline_precision_matches_source_backends_without_mutation(self):
        frame = pd.DataFrame({"previous_delay":[-1.25,123.123456789],"history_count":[1,0],
                              "oldest_pending_minutes":[61.123456789,300.],TARGET:[1e12,-1e12]})
        original = frame.copy(deep=True)
        choice = {"mode":"pending_residual","baseline":{"threshold":60.,"multiplier":1.5}}
        expected = stale.prediction_baseline(frame.drop(columns=TARGET),choice)
        self.assertEqual(expected.dtype,np.float32)
        for family in ["xgboost","catboost","lightgbm"]:
            actual = stale_ensemble._baseline(frame.drop(columns=TARGET),{"family":family,"configuration":choice})
            np.testing.assert_array_equal(actual,expected.astype(np.float64) if family != "xgboost" else expected)
            self.assertEqual(actual.dtype,np.float32 if family == "xgboost" else np.float64)
        pd.testing.assert_frame_equal(frame,original)

    def test_null_rules_and_future_target_perturbations_do_not_change_predictions(self):
        frame = pd.DataFrame({"history_count":[0,1,1],"is_departure":[1,0,0],
                              "oldest_pending_minutes":[600.,300.,301.],TARGET:[1.,2.,3.]})
        raw = np.array([50.,50.,50.])
        no_rule = {"nohistory_departure_cap":None,"pending_floor_threshold":None,"pending_floor_multiplier":None}
        np.testing.assert_array_equal(stale_ensemble.apply_rule(frame,raw,no_rule),raw)
        rule = {"nohistory_departure_cap":20.,"pending_floor_threshold":300.,"pending_floor_multiplier":2.}
        expected = np.array([20.,50.,602.])
        np.testing.assert_array_equal(stale_ensemble.apply_rule(frame,raw,rule),expected)
        frame[TARGET] = 1e12
        np.testing.assert_array_equal(stale_ensemble.apply_rule(frame.drop(columns=TARGET),raw,rule),expected)
        np.testing.assert_array_equal(raw,[50.,50.,50.])

    def test_invalid_weights_and_floor_parameters_are_rejected(self):
        for weights in [[.5,.4],[1.,0.],[float("nan"),.5]]:
            with self.assertRaises(ValueError):
                stale_ensemble._weights([{"source":str(i),"weight":value} for i,value in enumerate(weights)])
        with self.assertRaises(ValueError):
            stale_ensemble.apply_rule(pd.DataFrame(),[],{"nohistory_departure_cap":None,"pending_floor_threshold":300.,"pending_floor_multiplier":None})

    def test_optional_backends_preserve_predictions_after_native_reload(self):
        frame = synthetic_frame().iloc[:20].copy()
        frame.loc[frame.index[15:],"trainNumber"] = 24.
        parameters = {"catboost":{"iterations":2,"depth":2,"learning_rate":.1},
                      "lightgbm":{"n_estimators":2,"num_leaves":3,"min_child_samples":2},
                      "xgboost":{"n_estimators":2,"max_depth":2,"learning_rate":.1}}
        names = ["trainNumber","previous_delay","history_count"]
        for family,settings in parameters.items():
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                if family in {"catboost","lightgbm"}:
                    try:
                        stale_ensemble._optional_module(family)
                    except ImportError:
                        continue
                component = {"family":family,"configuration":{"mode":"residual","parameters":settings},
                             "features":names,"categorical_features":["trainNumber"]}
                matrix = stale_ensemble.model_matrix(frame,component)
                labels = frame[TARGET].to_numpy(dtype=float)-stale_ensemble._baseline(frame,component)
                model = stale_ensemble._fit(component,matrix,labels,np.arange(15),1)
                expected = stale_ensemble._component_prediction(model,matrix.iloc[15:],frame.iloc[15:],component,1)
                path = Path(temp)/f"model.{stale_ensemble._extension(component)}"
                stale_ensemble._save_model(model,component,path)
                restored = stale_ensemble._load_model(component,path,1)
                actual = stale_ensemble._component_prediction(restored,matrix.iloc[15:],frame.iloc[15:],component,1)
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)


class SnapshotHelperTests(unittest.TestCase):
    def test_old_snapshot_requires_strict_age_and_positive_presence_provenance(self):
        frame = synthetic_frame().iloc[[-1]].drop(columns=TARGET).copy()
        frame["fmi_timestamp"] = frame.prediction_time-pd.Timedelta(hours=2)
        frame["fmi_count"] = 1.
        manifest = {"additional_provenance_columns":["fmi_timestamp"],
                    "additional_provenance_lag_minutes":{"fmi_timestamp":60.},
                    "provenance_count_columns":{"fmi_timestamp":["fmi_count"]}}
        config = {"components":[{"features":["previous_delay"],"configuration":{"mode":"direct"},
                                 "input_manifest":manifest}],"selection":{"rule":{
                    "nohistory_departure_cap":None,"pending_floor_threshold":None,"pending_floor_multiplier":None}}}
        stale_ensemble.validate_snapshot(frame,config,"2023-01-01T00:00:00Z")
        for column,value in [("input_latest_actual_time",frame.observation_cutoff),
                             ("fmi_timestamp",frame.observation_cutoff-pd.Timedelta(minutes=60)),
                             ("fmi_timestamp",pd.NaT),("source_actual_time",pd.NaT),
                             ("focal_timetable_available_time",pd.NaT)]:
            changed = frame.copy();changed[column] = value
            with self.subTest(column=column,value=str(value)),self.assertRaises(ValueError):
                stale_ensemble.validate_snapshot(changed,config,"2023-01-01T00:00:00Z")
        changed = frame.copy()
        changed["observation_cutoff"] = changed.prediction_time-pd.Timedelta(minutes=29)
        changed["prediction_cutoff"] = changed.observation_cutoff
        with self.assertRaisesRegex(ValueError,"thirty minutes"):
            stale_ensemble.validate_snapshot(changed,config,"2023-01-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError,"unavailable"):
            stale_ensemble.validate_snapshot(frame,config,"2025-01-01T00:00:00Z")

    def test_profile_conflicts_and_query_alignment_are_explicit(self):
        left = synthetic_frame().iloc[:2].copy()
        right = left.copy();right["previous_delay"] += 10
        parts = [{"data_file":name,"features":["previous_delay","history_count"]}
                 for name in ["left.parquet","right.parquet"]]
        conflicts = stale_ensemble.snapshot_profile_conflicts(parts,{"left.parquet":left,"right.parquet":right})
        self.assertEqual(conflicts,[{"data_files":["left.parquet","right.parquet"],"features":["previous_delay"]}])
        stale_ensemble._assert_matching_columns(left,right,stale_ensemble.QUERY_IDENTITY,"query identity")
        right.loc[right.index[0],"query_train_number"] += 1
        with self.assertRaisesRegex(ValueError,"query identity"):
            stale_ensemble._assert_matching_columns(left,right,stale_ensemble.QUERY_IDENTITY,"query identity")

    def test_fold_guard_rejects_labels_at_cutoff_and_timestamp_ties(self):
        frame = synthetic_frame().iloc[:2].copy()
        stale_ensemble._assert_fold(frame,np.array([0]),np.array([1]))
        changed = frame.copy();changed.loc[0,"event_time"] = changed.loc[1,"observation_cutoff"]
        with self.assertRaisesRegex(ValueError,"Training labels"):
            stale_ensemble._assert_fold(changed,np.array([0]),np.array([1]))
        changed = frame.copy();changed.loc[0,"prediction_time"] = changed.loc[1,"prediction_time"]
        with self.assertRaisesRegex(ValueError,"training boundary"):
            stale_ensemble._assert_fold(changed,np.array([0]),np.array([1]))

    def test_prediction_time_is_the_known_or_supplied_scheduled_query(self):
        frame = synthetic_frame().iloc[[-1]].drop(columns=[TARGET]).copy()
        rule = {"nohistory_departure_cap":None,"pending_floor_threshold":None,"pending_floor_multiplier":None}
        config = {"components":[{"features":["previous_delay"],"configuration":{"mode":"direct"},"input_manifest":{}}],
                  "selection":{"rule":rule}}
        stale_ensemble.validate_snapshot(frame,config,"2023-01-01T00:00:00Z")
        changed = frame.copy()
        changed["prediction_time"] += pd.Timedelta(days=1)
        changed["observation_cutoff"] = changed.prediction_time-pd.Timedelta(minutes=30)
        changed["prediction_cutoff"] = changed.observation_cutoff
        with self.assertRaisesRegex(ValueError,"scheduled query"):
            stale_ensemble.validate_snapshot(changed,config,"2023-01-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError,"requires scheduledTime"):
            stale_ensemble.validate_snapshot(frame.drop(columns="scheduledTime"),config,"2023-01-01T00:00:00Z")
        missing_metadata = frame.drop(columns="scheduledTime").copy()
        missing_metadata["metadata_known"] = False
        stale_ensemble.validate_snapshot(missing_metadata,config,"2023-01-01T00:00:00Z")
        supplied = missing_metadata.copy()
        supplied["scheduledTime"] = supplied.prediction_time-pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError,"scheduled query"):
            stale_ensemble.validate_snapshot(supplied,config,"2023-01-01T00:00:00Z")

    def test_development_profiles_require_equal_postprocess_controls(self):
        frame = synthetic_frame()
        other = frame.copy()
        other.loc[0,"oldest_pending_minutes"] += 1.
        with tempfile.TemporaryDirectory() as temp, patch.object(stale,"DEVELOPMENT_ROWS",len(frame)):
            root = Path(temp)
            components = []
            for name,snapshot in [("left",frame),("right",other)]:
                path = root/f"{name}.parquet"
                snapshot.to_parquet(path,index=False)
                components.append({"data_file":path.name,"data_sha256":sha256(path)})
            with self.assertRaisesRegex(ValueError,"postprocess controls"):
                stale_ensemble._load_datasets(components,root)


if __name__ == "__main__":
    unittest.main()
