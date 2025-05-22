from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd


class AnomalyDetector:
    def __init__(self, df):
        self.df = df.copy()

    def detect_all(self):
        self._isolation_forest()
        self._local_outlier_factor()
        self._iqr_filter()
        return self.df

    def _isolation_forest(self, contamination=0.05):
        model = IsolationForest(contamination=contamination, random_state=42)
        self.df['anomaly_if'] = model.fit_predict(self.df[['value']])
        self.df['anomaly_if'] = self.df['anomaly_if'].map({1: 0, -1: 1})

    # ... аналогично остальные методы из второй части ...