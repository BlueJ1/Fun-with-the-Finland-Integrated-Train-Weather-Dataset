"""Conditional stage counterexamples, provenance guards and native reproduction."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fitw import stale, stale_ensemble as native, stale_policy as policy
from fitw.prepare import sha256
from fitw.spec import TARGET


def selection():
    result = {"components":[{"source":"a","weight":.3},{"source":"b","weight":.3},{"source":"c","weight":.4}],
              "origin_variant":"broad","initial_raw_bias":0.,"negative_branch_scale":1.1,
              "tail_threshold":240.,"tail_multiplier":1.5,"auxiliary_source":"fmilight:1","auxiliary_weight":.25,
              "rule":{"nohistory_departure_cap":30.,"pending_floor_threshold":360.,"pending_floor_multiplier":2.},
              "holdout_used":False}
    result["ordered_policy"] = policy.ordered_policy(result,result["rule"],"fmilight:1",.25)
    result["event_type_mixture_gate"] = dict(policy.EVENT_TYPE_GATE)
    result["event_type_mixtures"] = {name:copy.deepcopy(result["components"])
                                     for name in ["arrival_default","departure"]}
    return result


def gate_frame():
    return pd.DataFrame({"previous_delay":[0.,0.,np.nan,1.,0.,0.],
        "oulu_context_arrival_180m_max_delay":[60.,np.nan,61.,61.,60.,60.],
        "history_count":[1.,1.,0.,1.,0.,0.],"is_departure":[0.,0.,1.,0.,1.,1.],
        "oldest_pending_minutes":[0.,0.,0.,361.,0.,0.],
        "origin_context_actual_presence_count":[0.,0.,1.,0.,1.,0.],
        "origin_context_720m_max_last_delay":[0.,0.,121.,0.,121.,1000.],
        "origin_context_720m_max_projected_delay":[0.,0.,300.,0.,300.,1000.]})


class BoundedCalibrationReplayTests(unittest.TestCase):
    def test_ridge_and_neural_backends_record_the_installed_package(self):
        import sklearn
        components = [{"family": name} for name in ["ridge", "neural"]]
        versions = policy._backend_versions(components)
        self.assertEqual(versions, {"sklearn": sklearn.__version__})

    def fixture(self):
        frame = gate_frame()
        frame[TARGET] = [200., 400., 30., 650., 60., 30.]
        predictions = {name: np.full(len(frame), 220.) for name in ["a", "b", "c"]}
        predictions["fmilight:1"] = np.full(len(frame), 100.)
        definition = {"format":"bounded-event-experts-v1",
                      "event_type_mixture_gate":dict(policy.EVENT_TYPE_GATE),
                      "bounds": {"initial_raw_bias": [-5., 5.], "negative_branch_scale": [.8, 1.5],
                      "auxiliary_weight": [.05, .5], "tail_multiplier": [1., 2.],
                      "departure_cap": [15., 60.], "pending_multiplier": [1., 3.]},
                      "fixed_policy": {"origin_variant": "broad", "tail_threshold": 240.,
                          "auxiliary_source": "fmilight:1", "pending_floor_threshold": 360.},
                      "trial_count": 3, "context_data_sha256": "context-hash"}
        scores = lambda values: [float(np.sqrt(np.mean((values-frame[TARGET].to_numpy())**2)))] * 10
        records = []
        for seed, bias in enumerate([-1., 0., 1.], 1):
            record = selection()
            record.update(seed=seed, initial_raw_bias=bias, auxiliary_source="fmilight:1",
                          departure_cap=25., pending_multiplier=2.1, trial_count=3,
                          context_data_path="context.parquet", context_data_sha256="context-hash",
                          context_input_manifest={"features": list(gate_frame())})
            record["rule"].update(nohistory_departure_cap=25., pending_floor_multiplier=2.1)
            record["ordered_policy"] = policy.ordered_policy(record, record["rule"], "fmilight:1", .25)
            record["fold_rmse"] = scores(policy.apply_policy(frame, predictions, record))
            record["mean_cv_rmse"] = float(np.mean(record["fold_rmse"]))
            record["target_met"] = record["mean_cv_rmse"] <= 8.
            records.append(record)
        selected = copy.deepcopy(min(records, key=lambda r: r["mean_cv_rmse"]))
        return selected, definition, records, frame, predictions, np.arange(len(frame)), np.ones(len(frame)), scores

    def test_complete_frozen_trials_replay_and_incomplete_inventory_fails(self):
        args = list(self.fixture())
        self.assertEqual(policy._replay_bounded_calibration(*args)[-1], 3)
        args[2] = args[2][:-1]
        with self.assertRaises(ValueError): policy._replay_bounded_calibration(*args)

    def test_bounds_and_fixed_gates_cannot_be_weakened(self):
        args = list(self.fixture()); args[1]["bounds"]["departure_cap"] = [0., 60.]
        with self.assertRaises(ValueError): policy._replay_bounded_calibration(*args)
        args = list(self.fixture()); trial = args[2][0]
        trial["rule"]["pending_floor_threshold"] = 359.
        trial["ordered_policy"] = policy.ordered_policy(trial, trial["rule"], "fmilight:1", .25)
        with self.assertRaises(ValueError): policy._replay_bounded_calibration(*args)

    def test_false_score_and_context_substitution_are_rejected(self):
        args = list(self.fixture()); args[2][0]["mean_cv_rmse"] += .01
        with self.assertRaises(AssertionError): policy._replay_bounded_calibration(*args)
        args = list(self.fixture()); args[2][0]["context_data_sha256"] = "other-context"
        with self.assertRaises(ValueError): policy._replay_bounded_calibration(*args)

    def test_every_trial_must_use_the_frozen_source_inventory(self):
        args = list(self.fixture())
        components = [{"source":"a","weight":.7},{"source":"b","weight":.3}]
        trial = args[2][0];trial["components"] = components
        trial["event_type_mixtures"] = {name:copy.deepcopy(components) for name in ["arrival_default","departure"]}
        policy.validate_selection(trial)
        with self.assertRaisesRegex(ValueError,"source inventory"):policy._replay_bounded_calibration(*args)

    def test_event_experts_require_exact_gate_and_declared_inventory(self):
        args = list(self.fixture())
        args[1]["event_type_mixture_gate"]["threshold"] = True
        with self.assertRaises(ValueError):policy._replay_bounded_calibration(*args)
        for field in ["event_type_mixtures","event_type_mixture_gate"]:
            args = list(self.fixture());del args[2][0][field]
            with self.assertRaises(ValueError):policy._replay_bounded_calibration(*args)


class PolicyGateTests(unittest.TestCase):
    def test_event_type_experts_use_old_departure_gate_and_missing_default(self):
        selected = selection()
        selected["event_type_mixture_gate"] = dict(policy.EVENT_TYPE_GATE)
        selected["event_type_mixtures"] = {
            "arrival_default": [{"source":"a","weight":.6},{"source":"b","weight":.4}],
            "departure": [{"source":"b","weight":.2},{"source":"c","weight":.8}]}
        frame = gate_frame(); frame.loc[2,"is_departure"] = np.nan
        predictions = {name:np.full(len(frame),value) for name,value in [("a",10.),("b",20.),("c",30.),("fmilight:1",5.)]}
        before = frame.copy(deep=True)
        raw, _, result = policy.apply_policy(frame,predictions,selected,True)
        np.testing.assert_allclose(raw,[14.,14.,14.,14.,28.,28.],rtol=0,atol=1e-12)
        pd.testing.assert_frame_equal(frame,before)
        frame[TARGET] = 1e12;frame["row_id"] = 999
        np.testing.assert_array_equal(policy.apply_policy(frame,predictions,selected),result)

    def test_event_type_gate_and_union_tampering_are_rejected(self):
        selected = selection()
        selected["event_type_mixture_gate"] = dict(policy.EVENT_TYPE_GATE)
        selected["event_type_mixtures"] = {name:copy.deepcopy(selected["components"])
                                           for name in ["arrival_default","departure"]}
        policy.validate_selection(selected)
        for key,value in [("feature",TARGET),("threshold",True),("operator",">"),("other_and_missing","departure")]:
            changed = copy.deepcopy(selected);changed["event_type_mixture_gate"][key] = value
            with self.assertRaises(ValueError):policy.validate_selection(changed)
        changed = copy.deepcopy(selected);changed["event_type_mixtures"]["departure"] = [{"source":"a","weight":1.}]
        with self.assertRaisesRegex(ValueError,"source union"):policy.validate_selection(changed)
        changed = copy.deepcopy(selected);del changed["event_type_mixture_gate"]
        with self.assertRaisesRegex(ValueError,"Incomplete"):policy.validate_selection(changed)

    def test_ordered_tail_uses_transformed_prediction_and_exception_restores_precap_value(self):
        frame = gate_frame()
        raw = np.array([220.,220.,50.,50.,50.,50.])
        predictions = {name:raw.copy() for name in ["a","b","c"]}
        predictions["fmilight:1"] = np.array([100.,100.,100.,100.,100.,100.])
        original = frame.copy(deep=True)
        result = policy.apply_policy(frame,predictions,selection())
        # 220*1.1=242 crosses the tail threshold, but calm blending first
        # lowers row zero to 206.5. Row one still receives the tail scale.
        np.testing.assert_allclose(result,[206.5,363.,50.,722.,66.25,30.],rtol=0,atol=1e-12)
        pd.testing.assert_frame_equal(frame,original)
        np.testing.assert_array_equal(predictions["a"],raw)
        frame[TARGET] = 1e12
        np.testing.assert_array_equal(policy.apply_policy(frame,predictions,selection()),result)
        np.testing.assert_array_equal(policy.apply_policy(frame.drop(columns=TARGET),predictions,selection()),result)

    def test_nonfinite_parameters_identity_outcomes_and_reordered_gates_are_rejected(self):
        for name in ["initial_raw_bias","negative_branch_scale","tail_threshold","tail_multiplier","auxiliary_weight"]:
            for invalid in [float("nan"),float("inf"),True]:
                selected = selection();selected[name] = invalid
                with self.assertRaises(ValueError):policy.validate_selection(selected)
        for name in ["row_id","event_time",TARGET,"actualTime","query_train_number"]:
            selected = selection();selected["ordered_policy"][1]["gate"][0]["feature"] = name
            with self.assertRaises(ValueError):policy.validate_selection(selected)
        selected = selection();selected["ordered_policy"][2],selected["ordered_policy"][3] = selected["ordered_policy"][3],selected["ordered_policy"][2]
        with self.assertRaises(ValueError):policy.validate_selection(selected)
        selected = selection();selected["rule"]["pending_floor_multiplier"] = float("nan")
        with self.assertRaises(ValueError):policy.validate_selection(selected)
        predictions = {name:np.ones(6) for name in ["a","b","c","fmilight:1"]};predictions["fmilight:1"][1] = np.nan
        with self.assertRaises(ValueError):policy.apply_policy(gate_frame(),predictions,selection())
        frame = gate_frame();frame.history_count = np.nan
        with self.assertRaises(ValueError):policy.apply_policy(frame,{name:np.ones(6) for name in predictions},selection())

    def test_mapping_keeps_explicit_context_and_rejects_identity_or_control_disagreement(self):
        context = gate_frame().iloc[[0]].copy()
        for name,value in [("prediction_time","2024-01-01T12:00:00Z"),("observation_cutoff","2024-01-01T11:30:00Z"),
                           ("query_train_number",23),("departureDate","2024-01-01"),("type","ARRIVAL")]:context[name] = value
        config = {"components":[{"data_file":"a.parquet"}],"auxiliary":{"data_file":"b.parquet"},"snapshot_profile_conflicts":[{"features":["previous_delay"]}]}
        frames = {"a.parquet":context.copy(),"b.parquet":context.copy(),policy.CONTEXT_KEY:context.copy()}
        frames["a.parquet"]["previous_delay"] = 999.
        result = policy._inference_frames(frames,config)
        self.assertEqual(result[policy.CONTEXT_KEY].previous_delay.iloc[0],0.)
        self.assertEqual(result["a.parquet"].previous_delay.iloc[0],999.)
        with self.assertRaises(ValueError):policy._inference_frames(context,config)
        with self.assertRaises(ValueError):policy._inference_frames({k:v for k,v in frames.items() if k!=policy.CONTEXT_KEY},config)
        for name,value in [("query_train_number",24),("history_count",0)]:
            changed = {k:v.copy() for k,v in frames.items()};changed["a.parquet"][name] = value
            with self.assertRaises(ValueError):policy._inference_frames(changed,config)

    def test_context_timestamp_presence_counts_and_fmi_lag_are_enforced(self):
        frame = gate_frame().iloc[[0]].copy()
        frame["prediction_time"] = pd.Timestamp("2024-01-03T12:00:00Z")
        frame["scheduledTime"] = frame.prediction_time
        frame["observation_cutoff"] = pd.Timestamp("2024-01-03T11:30:00Z")
        frame["prediction_cutoff"] = frame.observation_cutoff
        frame["metadata_known"] = True
        frame["query_train_number"] = 23.;frame["source_train_number"] = 23.
        frame["departureDate"] = "2024-01-03";frame["source_departure_date"] = "2024-01-03"
        frame["origin_presence"] = 1.
        for name in native.BASE_PROVENANCE+ ["origin_timestamp","fmi_timestamp"]:
            frame[name] = pd.Timestamp("2024-01-03T10:00:00Z")
        manifest = {"features":list(gate_frame()),"additional_provenance_columns":["origin_timestamp","fmi_timestamp"],
                    "provenance_count_columns":{"origin_timestamp":["origin_presence"]},"additional_provenance_lag_minutes":{"fmi_timestamp":60.}}
        config = {"context":{"data_file":"c.parquet","input_manifest":manifest},"selection":selection()}
        policy.validate_context(frame,config,"2024-01-02T00:00:00Z")
        for name,value in [("origin_timestamp",pd.NaT),("origin_timestamp",frame.observation_cutoff.iloc[0]),
                           ("fmi_timestamp",pd.Timestamp("2024-01-03T10:30:00Z")),("origin_presence",float("nan")),
                           ("origin_presence",-1.),("origin_presence",.5),
                           ("source_train_number",24.),("source_departure_date","2024-01-02"),
                           ("scheduledTime",pd.Timestamp("2024-01-03T11:59:00Z"))]:
            changed = frame.copy();changed[name] = value
            with self.assertRaises(ValueError):policy.validate_context(changed,config,"2024-01-02T00:00:00Z")
        predictions = {name:np.ones(len(frame)) for name in ["a","b","c","fmilight:1"]}
        for name,value in [("is_departure",2.),("origin_context_actual_presence_count",.5),("history_count",-.5)]:
            changed = frame.copy();changed[name] = value
            with self.assertRaises(ValueError):policy.apply_policy(changed,predictions,selection())


def integration_exploration(directory, alternating_types=False, repeated_auxiliary=False):
    count = 132
    frame = pd.DataFrame({name:np.zeros(count) for name in stale.MODEL_FEATURES})
    frame["row_id"] = np.arange(count)
    frame["prediction_time"] = pd.date_range("2024-01-01",periods=count,freq="h",tz="UTC")
    frame["scheduledTime"] = frame.prediction_time
    frame["event_time"] = frame.prediction_time+pd.Timedelta(minutes=5)
    frame["observation_cutoff"] = frame.prediction_time-pd.Timedelta(minutes=30)
    frame["prediction_cutoff"] = frame.observation_cutoff
    frame["departureDate"] = frame.prediction_time.dt.strftime("%Y-%m-%d")
    frame["type"] = "ARRIVAL";frame["query_train_number"] = 23;frame["trainNumber"] = 23.
    frame["metadata_known"] = True;frame["history_count"] = 1.
    frame["previous_delay"] = np.arange(count)%7
    frame[TARGET] = frame.previous_delay+np.arange(count)%3
    for name in native.BASE_PROVENANCE:frame[name] = frame.prediction_time-pd.Timedelta(hours=2)
    frame["source_departure_date"] = frame.departureDate;frame["source_train_number"] = frame.query_train_number
    frame["fmi_timestamp"] = frame.prediction_time-pd.Timedelta(hours=2)
    frame["fmi_presence"] = 1.
    for name in policy.GATE_FEATURES[1:]:frame[name] = 0.
    frame["origin_context_actual_presence_count"] = 0.
    if alternating_types:
        frame["is_departure"] = np.arange(count)%2
        frame["type"] = np.where(frame.is_departure == 1,"DEPARTURE","ARRIVAL")
    names = list(stale.MODEL_FEATURES)+policy.GATE_FEATURES[1:]+["origin_context_actual_presence_count","fmi_presence"]
    manifest = {"features":names,"holdout_read":False,"additional_provenance_columns":["fmi_timestamp"],
                "provenance_count_columns":{"fmi_timestamp":["fmi_presence"]},"additional_provenance_lag_minutes":{"fmi_timestamp":60.}}
    data = directory/"data";data.mkdir()
    path = data/"synthetic.parquet";frame.to_parquet(path,index=False)
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    root = directory/"exploration";root.mkdir()
    splits = stale.folds_for(frame,10)
    ids = np.concatenate([frame.iloc[va].row_id for _,va in splits])
    folds = np.concatenate([np.full(len(va),i) for i,(_,va) in enumerate(splits,1)])
    sources,predictions = {},{}
    source_names = ["a","b","fmilight:1"] if repeated_auxiliary else ["a","b","c","fmilight:1"]
    for i,name in enumerate(source_names,1):
        source = {"family":"xgboost","features":names,"categorical_features":[],"data_path":str(path),"data_sha256":sha256(path),
                  "cv_sha256":f"{i:064x}",
                  "configuration":{"mode":"direct","parameters":{"n_estimators":i,"max_depth":2,"learning_rate":.1}}}
        matrix = native.model_matrix(frame,source);labels = frame[TARGET].to_numpy(dtype=np.float32)
        predictions[name] = np.concatenate([native._component_prediction(native._fit(source,matrix,labels,tr,1),matrix.iloc[va],frame.iloc[va],source,1) for tr,va in splits])
        sources[name] = source
    catalog = root/"catalog.json";catalog.write_text(json.dumps({"sources":sources}))
    oof = root/"raw_component_oof.npz"
    np.savez_compressed(oof,source_names=np.array(source_names),row_ids=ids,fold_ids=folds,predictions=np.stack(list(predictions.values())))
    records = []
    lookup = frame.set_index("row_id").loc[ids]
    for seed,bias in enumerate([-1.,0.,1.],1):
        selected = selection()
        selected.update(seed=seed,initial_raw_bias=bias,trial_count=3,departure_cap=25.,pending_multiplier=2.1,
                        context_data_path=str(path),context_data_sha256=sha256(path),context_input_manifest=manifest)
        if repeated_auxiliary:
            selected["components"] = [{"source":"a","weight":.3},{"source":"b","weight":.2},{"source":"fmilight:1","weight":.5}]
            selected["event_type_mixtures"] = {"arrival_default":[{"source":"a","weight":.6},{"source":"b","weight":.4}],
                                                "departure":[{"source":"fmilight:1","weight":1.}]}
        selected["rule"].update(nohistory_departure_cap=25.,pending_floor_multiplier=2.1)
        selected["ordered_policy"] = policy.ordered_policy(selected,selected["rule"],"fmilight:1",.25)
        prediction = policy.apply_policy(lookup,predictions,selected)
        losses = [float(np.sqrt(np.mean((prediction[folds==i]-lookup[TARGET].to_numpy()[folds==i])**2))) for i in range(1,11)]
        selected.update(fold_rmse=losses,mean_cv_rmse=float(np.mean(losses)),target_met=bool(np.mean(losses)<=8.))
        records.append(selected)
    best = min(records,key=lambda r:r["mean_cv_rmse"])
    definition = BoundedCalibrationReplayTests().fixture()[1]
    definition.update(context_data_sha256=sha256(path),input_manifest_sha256=sha256(path.with_suffix(".manifest.json")),
                      frozen_sha256={"config.json":sha256(catalog),"raw_component_oof.npz":sha256(oof)})
    for filename,value in [("selection.json",best),("definition.json",definition),("search.json",records)]:
        (root/filename).write_text(json.dumps(value))
    return root,data,path,frame


def compact_recipe(exploration):
    read = lambda name: json.loads((exploration/name).read_text())
    selected = read("selection.json")
    manifest_path = Path(selected["context_data_path"]).with_suffix(".manifest.json")
    selected.pop("context_input_manifest")
    selected["context_input_manifest_sha256"] = sha256(manifest_path)
    sources = read("catalog.json")["sources"]
    for source in sources.values():
        source.pop("features")
        source["input_manifest_sha256"] = sha256(Path(source["data_path"]).with_suffix(".manifest.json"))
    definition = read("definition.json")
    definition.pop("frozen_sha256")
    definition.update(folds=10,holdout_used=False)
    records = [{key: record[key] for key in policy.RECIPE_TRIAL_FIELDS} for record in read("search.json")]
    return {"format":policy.RECIPE_FORMAT,"selection":selected,"sources":sources,"definition":definition,"search":records}


class PolicyNativeIntegrationTests(unittest.TestCase):
    def test_compact_recipe_rebuilds_and_verifies_without_any_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(stale,"DEVELOPMENT_ROWS",132):
            directory = Path(temporary)
            exploration,data,path,frame = integration_exploration(directory,True,True)
            recipe = directory/"recipe.json";recipe.write_text(json.dumps(compact_recipe(exploration)))
            # Removing the entire previous exploration proves regeneration has
            # no dependency on its OOF vectors or expanded metadata.
            import shutil
            shutil.rmtree(exploration)
            output = directory/"native"
            args = SimpleNamespace(recipe=recipe,data_directory=data,output=output,threads=1)
            with patch.object(native,"_fit",wraps=native._fit) as fit,contextlib.redirect_stdout(io.StringIO()):
                policy.train(args);policy.verify(args)
            self.assertEqual(fit.call_count,33)
            verified = json.loads((output/"verification.json").read_text())
            self.assertTrue(verified["passed"])
            self.assertEqual(verified["frozen_affine_scores_recomputed"],3)
            self.assertEqual(verified["fold_models_reloaded"],40)
            config = json.loads((output/"config.json").read_text())
            self.assertEqual(config["reproduction_recipe_sha256"],sha256(recipe))
            for filename in ["model.ubj"]+[f"fold_{i}.ubj" for i in range(1,11)]:
                self.assertEqual(sha256(output/"component_3"/filename),sha256(output/"component_4"/filename))
            with (output/"reproduction_recipe.json").open("a") as saved:saved.write(" ")
            with self.assertRaisesRegex(ValueError,"recipe hash"):
                policy._artifact(output)

    def test_compact_recipe_rejects_inventory_manifest_and_policy_changes_before_fitting(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(stale,"DEVELOPMENT_ROWS",132):
            directory = Path(temporary)
            exploration,data,path,frame = integration_exploration(directory)
            original = compact_recipe(exploration)
            variants = []
            changed = copy.deepcopy(original);changed["format"] = "unknown";variants.append(changed)
            changed = copy.deepcopy(original);changed["sources"]["unused"] = changed["sources"]["a"];variants.append(changed)
            changed = copy.deepcopy(original);changed["sources"]["a"]["input_manifest_sha256"] = "0"*64;variants.append(changed)
            changed = copy.deepcopy(original);changed["selection"]["context_input_manifest_sha256"] = "0"*64;variants.append(changed)
            changed = copy.deepcopy(original);changed["search"][0]["departure_cap"] = 100.;variants.append(changed)
            changed = copy.deepcopy(original);changed["definition"]["folds"] = 5;variants.append(changed)
            changed = copy.deepcopy(original);changed["definition"]["holdout_used"] = True;variants.append(changed)
            recipe = directory/"recipe.json"
            for index,value in enumerate(variants):
                recipe.write_text(json.dumps(value))
                output = directory/f"native_{index}"
                args = SimpleNamespace(recipe=recipe,data_directory=data,output=output,threads=1)
                with patch.object(native,"_fit",wraps=native._fit) as fit:
                    with self.assertRaises(ValueError):policy.train(args)
                    self.assertEqual(fit.call_count,0)
                self.assertFalse(output.exists())

    def test_unsupported_search_formats_are_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(stale,"DEVELOPMENT_ROWS",132):
            directory = Path(temporary)
            exploration,data,path,frame = integration_exploration(directory)
            definition_path = exploration/"definition.json"
            original = json.loads(definition_path.read_text())
            for index,unsupported in enumerate([None,"bounded-global-calibration-v1","unknown"]):
                definition = copy.deepcopy(original);definition["format"] = unsupported
                definition_path.write_text(json.dumps(definition))
                args = SimpleNamespace(exploration=exploration,data_directory=data,output=directory/f"native_{index}",threads=1)
                with patch.object(native,"_fit",wraps=native._fit) as fit:
                    with self.assertRaisesRegex(ValueError,"format"):
                        policy.train(args)
                    self.assertEqual(fit.call_count,0)

    def test_unused_catalog_and_matrix_sources_are_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(stale,"DEVELOPMENT_ROWS",132):
            directory = Path(temporary)
            exploration,data,path,frame = integration_exploration(directory)
            catalog_path = exploration/"catalog.json";oof = exploration/"raw_component_oof.npz"
            original = json.loads(catalog_path.read_text())
            with np.load(oof,allow_pickle=False) as saved:
                ids,folds,matrix = saved["row_ids"].copy(),saved["fold_ids"].copy(),saved["predictions"].copy()
                names = saved["source_names"].tolist()
            for index,(extra_catalog,extra_matrix) in enumerate([(True,False),(False,True),(True,True)]):
                catalog = copy.deepcopy(original)
                if extra_catalog:catalog["sources"]["unused"] = copy.deepcopy(catalog["sources"]["a"])
                catalog_path.write_text(json.dumps(catalog))
                np.savez_compressed(oof,source_names=np.asarray(names+["unused"] if extra_matrix else names),row_ids=ids,fold_ids=folds,
                                    predictions=np.vstack([matrix,matrix[0]]) if extra_matrix else matrix)
                definition_path = exploration/"definition.json";definition = json.loads(definition_path.read_text())
                definition["frozen_sha256"] = {"config.json":sha256(catalog_path),"raw_component_oof.npz":sha256(oof)}
                definition_path.write_text(json.dumps(definition))
                args = SimpleNamespace(exploration=exploration,data_directory=data,output=directory/f"native_{index}",threads=1)
                with patch.object(native,"_fit",wraps=native._fit) as fit:
                    with self.assertRaises(ValueError):policy.train(args)
                    self.assertEqual(fit.call_count,0)

    def test_event_expert_fresh_fit_reuses_auxiliary_and_reloads_every_role(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(stale,"DEVELOPMENT_ROWS",132):
            directory = Path(temporary)
            exploration,data,path,frame = integration_exploration(directory,True,True)
            output = directory/"native"
            args = SimpleNamespace(exploration=exploration,data_directory=data,output=output,threads=1)
            with patch.object(native,"_fit",wraps=native._fit) as fit,contextlib.redirect_stdout(io.StringIO()):
                policy.train(args);policy.verify(args)
            self.assertEqual(fit.call_count,33)
            result = json.loads((output/"result.json").read_text());verified = json.loads((output/"verification.json").read_text())
            self.assertTrue(verified["passed"]);self.assertEqual(verified["fold_models_reloaded"],40)
            self.assertEqual(result["unique_native_sources"],3);self.assertEqual(result["fresh_fold_fit_count"],30)
            self.assertTrue(result["auxiliary_model_reused"])
            for filename in ["model.ubj"]+[f"fold_{i}.ubj" for i in range(1,11)]:
                self.assertEqual(sha256(output/"component_3"/filename),sha256(output/"component_4"/filename))

    def test_fresh_fit_fold_reload_target_free_inference_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(stale,"DEVELOPMENT_ROWS",132):
            directory = Path(temporary)
            exploration,data,path,frame = integration_exploration(directory)
            output = directory/"native"
            args = SimpleNamespace(exploration=exploration,data_directory=data,output=output,threads=1)
            self.assertEqual({p.name for p in exploration.iterdir()},set(policy.FROZEN_FILES))
            with patch.object(native,"_fit",wraps=native._fit) as fit,contextlib.redirect_stdout(io.StringIO()):
                policy.train(args);policy.verify(args)
            self.assertEqual(fit.call_count,44)
            result = json.loads((output/"result.json").read_text());verified = json.loads((output/"verification.json").read_text())
            self.assertTrue(verified["passed"]);self.assertEqual(verified["fold_models_reloaded"],40)
            self.assertEqual(verified["frozen_affine_scores_recomputed"],3)
            probe = frame.iloc[[-1]].drop(columns=TARGET).copy()
            probe.prediction_time = pd.Timestamp(result["deployment_not_before"])+pd.Timedelta(days=1)
            probe.scheduledTime = probe.prediction_time
            probe.observation_cutoff = probe.prediction_time-pd.Timedelta(minutes=30);probe.prediction_cutoff = probe.observation_cutoff
            mapped = {path.name:probe.copy(),policy.CONTEXT_KEY:probe.copy()}
            expected = policy.predict(mapped,output)
            for value in mapped.values():value[TARGET] = 1e12
            np.testing.assert_array_equal(policy.predict(mapped,output),expected)
            for key in mapped:
                changed = {k:v.copy() for k,v in mapped.items()};changed[key].input_latest_actual_time = pd.NaT
                with self.assertRaises(ValueError):policy.predict(changed,output)
            for key in mapped:
                changed = {k:v.copy() for k,v in mapped.items()};changed[key].source_train_number = 24
                with self.assertRaisesRegex(ValueError,"different dated train run"):policy.predict(changed,output)
            config_path = output/"config.json";config = json.loads(config_path.read_text())
            self.assertTrue(all("source_cv_path" not in component for component in policy._all_components(config)))
            catalog = json.loads((exploration/"catalog.json").read_text())
            self.assertEqual([component["source_cv_sha256"] for component in policy._all_components(config)],
                             [catalog["sources"][component["source"]]["cv_sha256"] for component in policy._all_components(config)])
            original_config,original_result = copy.deepcopy(config),copy.deepcopy(result)
            config["selection"]["initial_raw_bias"] = 1.
            config["selection"]["ordered_policy"][0]["transform"]["value"] = 1.
            config_path.write_text(json.dumps(config));result["config_sha256"] = sha256(config_path);result["selection"] = config["selection"]
            (output/"result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,"frozen selection"):policy.predict(mapped,output)
            config = original_config;result = original_result
            config["components"][0]["configuration"]["parameters"]["max_depth"] = 5
            config_path.write_text(json.dumps(config));result["config_sha256"] = sha256(config_path)
            (output/"result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,"frozen source catalog"):policy.predict(mapped,output)
            config["components"][0]["configuration"]["parameters"]["max_depth"] = 2
            config_path.write_text(json.dumps(config));result["config_sha256"] = sha256(config_path)
            (output/"result.json").write_text(json.dumps(result))
            # Consistently rehashing metadata must not permit removing a
            # component's FMI lag while leaving the context and models intact.
            config["components"][0]["input_manifest"]["additional_provenance_lag_minutes"] = {}
            config_path.write_text(json.dumps(config));result["config_sha256"] = sha256(config_path)
            (output/"result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,"frozen source manifest"):policy.predict(mapped,output)
            config["components"][0]["input_manifest"]["additional_provenance_lag_minutes"] = {"fmi_timestamp":60.}
            config_path.write_text(json.dumps(config));result["config_sha256"] = sha256(config_path)
            (output/"result.json").write_text(json.dumps(result))
            for keys in [["manifest_file"],["model_directory","model_file","fold_model_files"]]:
                for key in keys:
                    config["components"][0][key],config["components"][1][key] = config["components"][1][key],config["components"][0][key]
                config_path.write_text(json.dumps(config));result["config_sha256"] = sha256(config_path)
                (output/"result.json").write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError,"frozen source catalog"):policy.predict(mapped,output)
                for key in keys:
                    config["components"][0][key],config["components"][1][key] = config["components"][1][key],config["components"][0][key]
            config_path.write_text(json.dumps(config));result["config_sha256"] = sha256(config_path)
            (output/"result.json").write_text(json.dumps(result))
            model = output/"component_1/model.ubj";model.write_bytes(model.read_bytes()+b"x")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):policy.predict(mapped,output)


if __name__ == "__main__":unittest.main()
