from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm




class ARIMAModel:
"""ARIMA em retornos log, convertendo forecast em P(r>0)."""
def __init__(self, order=(1, 0, 0)):
self.order = order
self.model_ = None
self.res_ = None


def fit(self, train: pd.DataFrame):
y = train['LogRet']
with warnings.catch_warnings():
warnings.simplefilter('ignore')
self.model_ = ARIMA(y, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
self.res_ = self.model_.fit()
return self


def predict_next(self, recent: pd.DataFrame):
f = self.res_.get_forecast(steps=1)
mu = float(f.predicted_mean.iloc[-1])
sigma = float(f.se_mean.iloc[-1])
sigma = sigma if np.isfinite(sigma) and sigma > 0 else 1e-6
p_up = 1.0 - norm.cdf(0.0, loc=mu, scale=sigma)
return {
'model': 'ARIMA',
'order': self.order,
'mu': mu,
'sigma': sigma,
'prob_up': p_up,
'explain': 'ARIMA(returns) → P(r>0) via média e desvio previstos.'
}
