from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from .indicators import rsi, atr, ema, macd
from .utils import safe_pct_change

def download_prices(ticker:str, start:str, end:str, interval:str='1d') -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].title() for c in df.columns]
    df = df.rename(columns={c: c.title() for c in df.columns})
    df = df.dropna().copy()
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['Return'] = safe_pct_change(out['Close']).fillna(0.0)
    out['LogRet'] = np.log1p(out['Return'])
    out['RSI14'] = rsi(out['Close'], 14)
    out['ATR14'] = atr(out['High'], out['Low'], out['Close'], 14)
    out['EMA20'] = ema(out['Close'], 20)
    out['EMA50'] = ema(out['Close'], 50)
    out['EMA200'] = ema(out['Close'], 200)
    macd_line, signal_line, hist = macd(out['Close'])
    out['MACD'] = macd_line
    out['MACDsig'] = signal_line
    out['MACDhist'] = hist
    out['Mom20'] = out['Close'] / out['Close'].shift(20) - 1
    out['Vol20'] = out['Return'].rolling(20).std()
    out = out.dropna().copy()
    return out