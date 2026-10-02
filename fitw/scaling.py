"""Serializable historical author preprocessing."""

from sklearn.preprocessing import RobustScaler


class WeatherScaler:
    """Scale raw weather only while preserving the complete column order."""

    def __init__(self, indices):
        self.indices = indices

    def fit(self, X):
        self.scaler = RobustScaler().fit(X[:, self.indices]) if self.indices else None
        return self

    def transform(self, X):
        transformed = X.copy()
        if self.scaler is not None:
            transformed[:, self.indices] = self.scaler.transform(X[:, self.indices])
        return transformed
