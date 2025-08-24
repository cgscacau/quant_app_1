import numpy as np
import pandas as pd




def ema(series: pd.Series, span: int = 20) -> pd.Series:
return series.ewm(span=span, adjust=False).mean()




def rsi(close: pd.Series, period: int = 14) -> pd.Series:
delta = close.diff()
up, down = delta.clip(lower=0), -delta.clip(upper=0)
roll_up = up.ewm(alpha=1/period, adjust=False).mean()
roll_down = down.ewm(alpha=1/period, adjust=False).mean()
rs = roll_up / (roll_down + 1e-12)
return 100 - (100 / (1 + rs))




def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
prev_close = close.shift(1)
ranges = pd.concat([
high - low,
(high - prev_close).abs(),
(low - prev_close).abs()
], axis=1)
return ranges.max(axis=1)




def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
tr = true_range(high, low, close)
return tr.ewm(alpha=1/period, adjust=False).mean()




def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
ema_fast = ema(close, fast)
ema_slow = ema(close, slow)
macd_line = ema_fast - ema_slow
signal_line = ema(macd_line, signal)
hist = macd_line - signal_line
return macd_line, signal_line, hist
