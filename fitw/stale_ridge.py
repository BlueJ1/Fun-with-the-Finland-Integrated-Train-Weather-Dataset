"""Training-only preprocessing and native ridge estimators."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge


def model_matrix(frame, features):
    """Preserve missing values and numeric IDs for the fold-fitted pipeline."""
    return frame[features].apply(pd.to_numeric,errors="raise").astype(np.float32)

def estimator(parameters, features, categorical_features):
    """Return a pipeline without fitting any preprocessing on validation rows."""
    categorical = [name for name in features if name in categorical_features]
    numeric = [name for name in features if name not in categorical]
    transforms = []
    if numeric:
        transforms.append(("numeric",Pipeline([
            ("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),
            ("scaler",StandardScaler(with_mean=False)),
        ]),numeric))
    if categorical:
        transforms.append(("categorical",Pipeline([
            ("imputer",SimpleImputer(strategy="constant",fill_value=-1.,keep_empty_features=True)),
            ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=True,dtype=np.float64)),
        ]),categorical))
    preparation = ColumnTransformer(transforms,sparse_threshold=1.,remainder="drop")
    return Pipeline([("preprocess",preparation),("ridge",Ridge(**parameters))])
