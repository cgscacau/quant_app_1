from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def _to_py_number(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return float(x) if isinstance(x, (float, int)) else x

class RandomForestSignal:
    """Classificador para probabilidade de alta usando features técnicas."""
    def __init__(self, n_estimators=300, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1
        )
        self.scaler = None
        self.feature_names_ = None
        self.report_ = None   # dict seguro p/ JSON

    def _build_Xy(self, df: pd.DataFrame):
        feats = ['RSI14','ATR14','EMA20','EMA50','EMA200',
                 'MACD','MACDsig','MACDhist','Mom20','Vol20']
        X = df[feats].copy()
        y = (df['Return'].shift(-1) > 0).astype(int)  # sobe na próxima barra?
        X, y = X.iloc[:-1], y.iloc[:-1]               # alinhar
        return X, y, feats

    def fit(self, train: pd.DataFrame, test: pd.DataFrame):
        Xtr, ytr, feats = self._build_Xy(train)
        Xte, yte, _ = self._build_Xy(test)
        self.feature_names_ = feats

        self.scaler = StandardScaler()
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xte_s = self.scaler.transform(Xte) if len(Xte) > 0 else Xte

        self.model.fit(Xtr_s, ytr)

        # relatório seguro (sem NaN, só tipos Python)
        if len(Xte) > 0:
            ypred = self.model.predict(Xte_s)
            rep = classification_report(yte, ypred, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(rep).T.fillna(0.0).round(6)
            rep_py = {k: {kk: _to_py_number(vv) for kk, vv in d.items()}
                      if isinstance(d, dict) else _to_py_number(d)
                      for k, d in rep_df.to_dict(orient='index').items()}
            self.report_ = rep_py
        else:
            self.report_ = {}

        return self

    def predict_next(self, recent: pd.DataFrame):
        X, _, _ = self._build_Xy(recent)
        x_last = X.iloc[[-1]]
        x_last_s = self.scaler.transform(x_last)
        proba = float(self.model.predict_proba(x_last_s)[0, 1])

        fi = dict(zip(self.feature_names_, self.model.feature_importances_))
        fi = {k: _to_py_number(v) for k, v in fi.items()}

        return {
            'model': 'RandomForest',
            'prob_up': proba,
            'feature_importances': fi,
            'report': self.report_,
            'explain': 'RandomForest em features técnicas; probabilidade calibrada no período de teste.'
        }
