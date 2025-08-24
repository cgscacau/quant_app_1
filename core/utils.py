from __future__ import annotations
import numpy as np
import pandas as pd

def safe_pct_change(s: pd.Series, periods:int=1) -> pd.Series:
    try:
        return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    except Exception:
        return pd.Series([0.0]*len(s), index=s.index)

def zscore(x: pd.Series, window:int=20) -> pd.Series:
    r = x.rolling(window)
    return (x - r.mean()) / (r.std().replace(0, np.nan))

def last_valid(series: pd.Series, default=None):
    try:
        return series.dropna().iloc[-1]
    except Exception:
        return default

def train_test_split_by_date(df: pd.DataFrame, train_start, train_end, test_start, test_end):
    train = df.loc[str(train_start):str(train_end)].copy()
    test  = df.loc[str(test_start):str(test_end)].copy()
    return train, test