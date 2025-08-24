from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class TrendScoreModel:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=200)
        self.cols_ = None

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        feats['Close_over_EMA50'] = df['Close'] / (df['EMA50'] + 1e-12) - 1
        feats['EMA20_over_EMA50'] = df['EMA20'] / (df['EMA50'] + 1e-12) - 1
        feats['EMA50_over_EMA200'] = df['EMA50'] / (df['EMA200'] + 1e-12) - 1
        feats['MACDhist'] = df['MACDhist']
        feats['RSI14'] = df['RSI14']
        feats['Mom20'] = df['Mom20']
        feats = feats.fillna(0.0)
        return feats

    def fit(self, train: pd.DataFrame, test: pd.DataFrame):
        Xtr = self._features(train).iloc[:-1]
        ytr = (train['Return'].shift(-1) > 0).astype(int).iloc[:-1]
        self.cols_ = Xtr.columns.tolist()
        self.clf.fit(Xtr, ytr)
        return self

    def predict_next(self, recent: pd.DataFrame):
        X = self._features(recent).iloc[[-2]]
        proba = float(self.clf.predict_proba(X)[0,1])
        return {'model': 'TrendScore','prob_up': proba,'explain': 'Momentum/EMAs/RSI via regressão logística.'}