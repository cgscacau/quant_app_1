# core/backtest.py
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
    entry_date = None

    trades = []  # << novo

    for i in range(1, len(df)-1):
        today = df.index[i]
        px   = float(df['Close'].iloc[i])
        hi   = float(df['High'].iloc[i])
        lo   = float(df['Low'].iloc[i])
        atr  = float(df['ATR14'].iloc[i])
        p    = float(prob_series.iloc[i])

        # fecha posição se bater stop/take (aproxima pelo range da barra)
        if pos_qty != 0 and direction is not None:
            if direction == 'BUY':
                if lo <= stop:
                    eq += pos_qty * (stop - entry)
                    trades.append(dict(side='BUY', entry_date=entry_date, entry=entry,
                                       exit_date=today, exit=stop, reason='STOP',
                                       qty=pos_qty, pnl=pos_qty*(stop-entry)))
                    pos_qty = 0; direction = None
                elif hi >= take:
                    eq += pos_qty * (take - entry)
                    trades.append(dict(side='BUY', entry_date=entry_date, entry=entry,
                                       exit_date=today, exit=take, reason='TAKE',
                                       qty=pos_qty, pnl=pos_qty*(take-entry)))
                    pos_qty = 0; direction = None
            else:
                if hi >= stop:
                    eq += pos_qty * (entry - stop)
                    trades.append(dict(side='SELL', entry_date=entry_date, entry=entry,
                                       exit_date=today, exit=stop, reason='STOP',
                                       qty=pos_qty, pnl=pos_qty*(entry-stop)))
                    pos_qty = 0; direction = None
                elif lo <= take:
                    eq += pos_qty * (entry - take)
                    trades.append(dict(side='SELL', entry_date=entry_date, entry=entry,
                                       exit_date=today, exit=take, reason='TAKE',
                                       qty=pos_qty, pnl=pos_qty*(entry-take)))
                    pos_qty = 0; direction = None

        # abre nova posição
        if pos_qty == 0:
            if p >= threshold_buy:
                direction = 'BUY'
                entry, stop, take = entry_stop_gain(px, atr, direction, atr_mult_stop, rr)
                qty = position_size(eq, entry, stop, risk_perc)
                if qty > 0:
                    pos_qty = qty
                    entry_date = today
            elif p <= threshold_sell:
                direction = 'SELL'
                entry, stop, take = entry_stop_gain(px, atr, direction, atr_mult_stop, rr)
                qty = position_size(eq, entry, stop, risk_perc)
                if qty > 0:
                    pos_qty = qty
                    entry_date = today

        equity.append(eq)

    out = pd.DataFrame({'Equity': equity}, index=df.index[1:len(equity)+1])
    out['Ret'] = out['Equity'].pct_change().fillna(0.0)
    sharpe = np.sqrt(252) * (out['Ret'].mean() / (out['Ret'].std() + 1e-12))

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['cum_pnl'] = trades_df['pnl'].cumsum()

    stats = {
        'final_equity': float(out['Equity'].iloc[-1]),
        'sharpe': float(sharpe),
        'trades': int(len(trades_df)),
        'hit_rate': float((trades_df['pnl']>0).mean()) if not trades_df.empty else 0.0,
        'avg_pnl': float(trades_df['pnl'].mean()) if not trades_df.empty else 0.0,
        'total_pnl': float(trades_df['pnl'].sum()) if not trades_df.empty else 0.0,
    }
    return out, stats, trades_df
