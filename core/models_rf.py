from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

class RandomForestSignal:
    def __init__(self, n_estimators=300, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
        self.scaler = None
        self.feature_names_ = None
        self.report_ = None

    def _build_Xy(self, df: pd.DataFrame):
        feats = ['RSI14','ATR14','EMA20','EMA50','EMA200','MACD','MACDsig','MACDhist','Mom20','Vol20']
        X = df[feats].copy()
        y = (df['Return'].shift(-1) > 0).astype(int)
        X, y = X.iloc[:-1], y.iloc[:-1]
        return X, y, feats

    def fit(self, train: pd.DataFrame, test: pd.DataFrame):
        Xtr, ytr, feats = self._build_Xy(train)
        Xte, yte, _ = self._build_Xy(test)
        self.feature_names_ = feats
        self.scaler = StandardScaler()
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xte_s = self.scaler.transform(Xte) if len(Xte)>0 else Xte
        self.model.fit(Xtr_s, ytr)
        if len(Xte)>0:
            ypred = self.model.predict(Xte_s)
            self.report_ = classification_report(yte, ypred, output_dict=True, zero_division=0)
        return self

    def predict_next(self, recent: pd.DataFrame):
        X, _, _ = self._build_Xy(recent)
        x_last = X.iloc[[-1]]
        x_last_s = self.scaler.transform(x_last)
        proba = float(self.model.predict_proba(x_last_s)[0,1])
        return {'model': 'RandomForest','prob_up': proba,'feature_importances': dict(zip(self.feature_names_, self.model.feature_importances_)),
                'report': self.report_,'explain': 'RandomForest em features técnicas → prob. de alta.'}