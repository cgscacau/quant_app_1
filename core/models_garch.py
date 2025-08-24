from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from arch import arch_model

class GARCHModel:
    def __init__(self, p=1, q=1):
        self.p, self.q = p, q
        self.res_ = None

    def fit(self, train: pd.DataFrame):
        y = train['LogRet'] * 100.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            am = arch_model(y, mean='Constant', vol='GARCH', p=self.p, q=self.q, dist='normal')
            self.res_ = am.fit(disp='off')
        return self

    def predict_next(self, recent: pd.DataFrame):
        f = self.res_.forecast(horizon=1, reindex=False)
        sigma_next = float(np.sqrt(f.variance.values[-1, -1]) / 100.0)
        return {'model': 'GARCH','sigma_next': sigma_next,'prob_up': 0.5, 'explain': 'GARCH → vol futura (direção neutra).'}