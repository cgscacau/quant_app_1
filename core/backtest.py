from __future__ import annotations
import numpy as np
import pandas as pd
from .risk import entry_stop_gain, position_size

def simulate_prob_strategy(df: pd.DataFrame, prob_series: pd.Series, threshold_buy=0.55, threshold_sell=0.45,
                            capital0=100000.0, risk_perc=0.01, atr_mult_stop=2.0, rr=2.0):
    eq = capital0
    equity = []
    pos_qty = 0
    direction = None
    entry = stop = take = None

    for i in range(1, len(df)-1):
        px = df['Close'].iloc[i]
        atr = df['ATR14'].iloc[i]
        p = prob_series.iloc[i]

        if pos_qty != 0 and direction is not None:
            if direction == 'BUY':
                if df['Low'].iloc[i] <= stop:
                    eq += pos_qty * (stop - entry)
                    pos_qty = 0; direction = None
                elif df['High'].iloc[i] >= take:
                    eq += pos_qty * (take - entry)
                    pos_qty = 0; direction = None
            else:
                if df['High'].iloc[i] >= stop:
                    eq += pos_qty * (entry - stop)
                    pos_qty = 0; direction = None
                elif df['Low'].iloc[i] <= take:
                    eq += pos_qty * (entry - take)
                    pos_qty = 0; direction = None

        if pos_qty == 0:
            if p >= threshold_buy:
                direction = 'BUY'
                entry, stop, take = entry_stop_gain(px, atr, direction, atr_mult_stop, rr)
                qty = position_size(eq, entry, stop, risk_perc)
                pos_qty = qty
            elif p <= threshold_sell:
                direction = 'SELL'
                entry, stop, take = entry_stop_gain(px, atr, direction, atr_mult_stop, rr)
                qty = position_size(eq, entry, stop, risk_perc)
                pos_qty = qty

        equity.append(eq)

    out = pd.DataFrame({'Equity': equity}, index=df.index[1:len(equity)+1])
    out['Ret'] = out['Equity'].pct_change().fillna(0.0)
    sharpe = np.sqrt(252) * out['Ret'].mean() / (out['Ret'].std() + 1e-12)
    return out, {'final_equity': float(out['Equity'].iloc[-1]), 'sharpe': float(sharpe)}