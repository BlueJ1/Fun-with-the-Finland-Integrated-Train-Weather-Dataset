"""Train-only preprocessing and numeric extrapolation for additive baselines."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fitw import stale_ridge, stale_ensemble as native


class RidgePipelineTests(unittest.TestCase):
    def test_imputation_scaling_and_category_vocabularies_fit_training_only(self):
        training = pd.DataFrame({"delay":[0.,1.,2.,np.nan],"trainNumber":[23.,23.,24.,np.nan]})
        validation = pd.DataFrame({"delay":[1000.,np.nan],"trainNumber":[999.,23.]})
        features = list(training)
        model = stale_ridge.estimator({"alpha":1.,"solver":"lsqr"},features,["trainNumber"])
        with threadpool_limits(limits=1):
            model.fit(stale_ridge.model_matrix(training,features),np.array([0.,1.,2.,1.]))
        prep = model.named_steps["preprocess"]
        numeric = prep.named_transformers_["numeric"]
        np.testing.assert_array_equal(numeric.named_steps["imputer"].statistics_,[1.])
        encoder = prep.named_transformers_["categorical"].named_steps["encoder"]
        self.assertNotIn(999.,encoder.categories_[0])
        before = numeric.named_steps["scaler"].scale_.copy()
        with threadpool_limits(limits=1):
            prediction = model.predict(stale_ridge.model_matrix(validation,features))
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_array_equal(numeric.named_steps["scaler"].scale_,before)
        self.assertNotIn(999.,encoder.categories_[0])

    def test_saved_pipeline_extrapolates_numeric_delay_beyond_training_range(self):
        training = pd.DataFrame({"delay":np.arange(20,dtype=float),"empty_feature":np.nan,"trainNumber":23.})
        validation = pd.DataFrame({"delay":[40.],"empty_feature":[np.nan],"trainNumber":[24.]})
        names = list(training)
        model = stale_ridge.estimator({"alpha":.01,"solver":"lsqr","tol":1e-8},names,["trainNumber"])
        with threadpool_limits(limits=1):
            model.fit(stale_ridge.model_matrix(training,names),2*training.delay.to_numpy())
            expected = model.predict(stale_ridge.model_matrix(validation,names))
        self.assertGreater(expected[0],2*training.delay.max())
        self.assertAlmostEqual(expected[0],80.,delta=.1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"pipeline.joblib"
            native._save_model(model,{"family":"ridge"},path)
            restored = native._load_model({"family":"ridge"},path,1)
            with threadpool_limits(limits=1):
                actual = restored.predict(stale_ridge.model_matrix(validation,names))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)

    def test_native_fit_ignores_validation_features_and_labels(self):
        frame = pd.DataFrame({"delay":np.arange(24)%7,"trainNumber":23.})
        component = {"family":"ridge","features":list(frame),"categorical_features":["trainNumber"],
                     "configuration":{"mode":"direct","parameters":{"alpha":1.,"solver":"lsqr"}}}
        labels = np.arange(24,dtype=float)
        first = native._fit(component,native.model_matrix(frame,component),labels,np.arange(20),1)
        changed = frame.copy();changed.loc[20:,"delay"] = 1e9;changed.loc[20:,"trainNumber"] = 999.
        perturbed = labels.copy();perturbed[20:] = 1e12
        second = native._fit(component,native.model_matrix(changed,component),perturbed,np.arange(20),1)
        with threadpool_limits(limits=1):
            np.testing.assert_array_equal(first.predict(frame),second.predict(frame))
        self.assertNotIn(999.,second.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encoder"].categories_[0])


if __name__ == "__main__":
    unittest.main()
