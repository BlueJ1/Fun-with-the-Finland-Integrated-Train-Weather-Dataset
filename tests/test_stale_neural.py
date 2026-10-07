"""Training-only neural transforms, label isolation and native serialization."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fitw import stale_ensemble as native, stale_neural as neural, stale_policy


def component():
    return {"family":"neural","features":["delay","empty_feature","trainNumber"],
            "categorical_features":["trainNumber"],"configuration":{"mode":"pending_residual",
            "baseline":{"threshold":240.,"multiplier":1.5},"parameters":{
                "hidden_layer_sizes":[4],"solver":"adam","learning_rate_init":.001,
                "max_iter":4,"batch_size":16,"random_state":42,"early_stopping":False}}}


class NeuralPipelineTests(unittest.TestCase):
    def test_validation_early_stopping_is_rejected(self):
        parameters = dict(component()["configuration"]["parameters"],early_stopping=True)
        with self.assertRaises(ValueError):
            neural.estimator(parameters,["delay"],[])



    def test_imputation_vocabularies_scaling_and_target_scaling_see_only_training(self):
        training = pd.DataFrame({"delay":[0.,1.,2.,np.nan],"empty_feature":np.nan,"trainNumber":[23.,23.,24.,np.nan]})
        validation = pd.DataFrame({"delay":[1000.,np.nan],"empty_feature":[999.,np.nan],"trainNumber":[999.,23.]})
        matrix = neural.model_matrix(pd.concat([training,validation],ignore_index=True),list(training))
        labels = np.array([-10.,0.,10.,20.,1e12,-1e12])
        model = native._fit(component(),matrix,labels,np.arange(4),1)
        prep = model.regressor_.named_steps["preprocess"]
        numeric = prep.named_transformers_["numeric"]
        np.testing.assert_array_equal(numeric.named_steps["imputer"].statistics_,[1.,0.])
        np.testing.assert_allclose(numeric.named_steps["scaler"].mean_[:2],[1.,0.],rtol=0,atol=1e-12)
        self.assertEqual(numeric.named_steps["imputer"].keep_empty_features,True)
        self.assertEqual(numeric.named_steps["imputer"].add_indicator,True)
        encoder = prep.named_transformers_["categorical"].named_steps["encoder"]
        self.assertNotIn(999.,encoder.categories_[0])
        np.testing.assert_array_equal(model.transformer_.mean_,[5.])
        np.testing.assert_array_equal(model.transformer_.var_,[125.])
        self.assertEqual(model.transformer_.n_samples_seen_,4)
        before = numeric.named_steps["scaler"].scale_.copy()
        with threadpool_limits(limits=1):
            actual = model.predict(matrix.iloc[4:])
            standardized = model.regressor_.predict(matrix.iloc[4:])
        expected = model.transformer_.inverse_transform(standardized.reshape(-1,1)).ravel()
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(numeric.named_steps["scaler"].scale_,before)
        self.assertNotIn(999.,encoder.categories_[0])
        self.assertFalse(model.regressor_.named_steps["mlp"].early_stopping)

    def test_validation_label_and_feature_perturbations_do_not_change_fitted_model(self):
        count = 48
        frame = pd.DataFrame({"delay":np.arange(count)%7,"empty_feature":np.nan,"trainNumber":23.})
        labels = np.sin(np.arange(count))*10
        train = np.arange(40)
        first = native._fit(component(),native.model_matrix(frame,component()),labels,train,1)
        changed = frame.copy();changed.loc[40:,"delay"] = 1e9;changed.loc[40:,"trainNumber"] = 999.
        changed_labels = labels.copy();changed_labels[40:] = 1e12
        second = native._fit(component(),native.model_matrix(changed,component()),changed_labels,train,1)
        with threadpool_limits(limits=1):
            np.testing.assert_array_equal(first.predict(frame),second.predict(frame))
        np.testing.assert_array_equal(first.transformer_.mean_,second.transformer_.mean_)
        np.testing.assert_array_equal(first.regressor_.named_steps["mlp"].coefs_[0],second.regressor_.named_steps["mlp"].coefs_[0])

    def test_native_reload_residual_precision_and_backend_version(self):
        frame = pd.DataFrame({"delay":np.arange(40)%7,"empty_feature":np.nan,"trainNumber":23.,
                              "previous_delay":2.,"history_count":1.,"oldest_pending_minutes":0.})
        choice = component();matrix = native.model_matrix(frame,choice)
        model = native._fit(choice,matrix,np.sin(np.arange(40))*3,np.arange(40),1)
        expected = native._component_prediction(model,matrix,frame,choice,1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"native.joblib"
            native._save_model(model,choice,path)
            restored = native._load_model(choice,path,1)
            np.testing.assert_array_equal(native._component_prediction(restored,matrix,frame,choice,1),expected)
        self.assertEqual(native._extension(choice),"joblib")
        self.assertEqual(expected.dtype,np.float64)
        self.assertIn("sklearn",stale_policy._backend_versions([choice]))



if __name__ == "__main__":
    unittest.main()
