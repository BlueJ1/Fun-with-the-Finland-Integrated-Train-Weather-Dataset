"""Training-only preprocessing and native neural residual estimators."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor


def model_matrix(frame, features):
    """Preserve missing values and numeric IDs for the fold-fitted pipeline."""
    return frame[features].apply(pd.to_numeric,errors="raise").astype(np.float32)

def estimator(parameters, features, categorical_features):
    """Fit preprocessing and target scaling only inside this native estimator."""
    if parameters.get("early_stopping",False):
        raise ValueError("Neural residual models cannot use early stopping")
    categorical = [name for name in features if name in categorical_features]
    numeric = [name for name in features if name not in categorical]
    transforms = []
    if numeric:
        transforms.append(("numeric",Pipeline([
            ("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),
            ("scaler",StandardScaler()),
        ]),numeric))
    if categorical:
        transforms.append(("categorical",Pipeline([
            ("imputer",SimpleImputer(strategy="constant",fill_value=-1.,keep_empty_features=True)),
            ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False,dtype=np.float32)),
        ]),categorical))
    preparation = ColumnTransformer(transforms,sparse_threshold=0.,remainder="drop")
    settings = dict(parameters)
    settings["hidden_layer_sizes"] = tuple(settings["hidden_layer_sizes"])
    settings["early_stopping"] = False
    regressor = Pipeline([("preprocess",preparation),("mlp",MLPRegressor(**settings))])
    return TransformedTargetRegressor(regressor=regressor,transformer=StandardScaler(),check_inverse=True)
