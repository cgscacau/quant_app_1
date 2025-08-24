from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression




def _to_float(x):
try:
return float(x)
except Exception:
return 0.0




class TrendScoreModel:
"""Regressão logística em razões de EMAs/RSI/MACD → probabilidade de alta."""
def __init__(self):
self.clf = LogisticRegression(max_iter=400)
self.cols_ = None


def _features(self, df: pd.DataFrame) -> pd.DataFrame:
feats = pd.DataFrame(index=df.index)
feats['Close_over_EMA50'] = df['Close'] / (df['EMA50'] + 1e-12) - 1
feats['EMA20_over_EMA50'] = df['EMA20'] / (df['EMA50'] + 1e-12) - 1
feats['EMA50_over_EMA200'] = df['EMA50'] / (df['EMA200'] + 1e-12) - 1
feats['MACDhist'] = df['MACDhist']
feats['RSI14'] = df['RSI14']
feats['Mom20'] = df['Mom20']
return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit(self, train: pd.DataFrame, test: pd.DataFrame):
Xtr = self._features(train).iloc[:-1]
ytr = (train['Return'].shift(-1) > 0).astype(int).iloc[:-1]
self.cols_ = Xtr.columns.tolist()
if len(Xtr) >= 20:
self.clf.fit(Xtr, ytr)
return self


def predict_next(self, recent: pd.DataFrame):
F = self._features(recent)
idx = F.dropna().index
last = idx[-1] if len(idx) else F.index[-1]
x = F.loc[[last]]
try:
proba = float(self.clf.predict_proba(x)[0, 1])
except Exception:
proba = 0.5
try:
coefs = {c: _to_float(v) for c, v in zip(self.cols_, self.clf.coef_.ravel())}
intercept = _to_float(self.clf.intercept_[0])
except Exception:
coefs, intercept = {}, 0.0
return {
'model': 'TrendScore',
'prob_up': proba,
'coefficients': coefs,
'intercept': intercept,
'explain': 'Logit sobre razões de EMAs/RSI/MACD → prob. de alta.'
}
