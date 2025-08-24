from __future__ import annotations
exit_date=today, exit=exit_px, reason='STOP',
qty=pos_qty, pnl=pnl))
pos_qty = 0; direction = None
elif hi >= take:
exit_px = take * (1 - slippage_bps/1e4)
pnl = pos_qty * (exit_px - entry) - abs(pos_qty)*exit_px*(cost_bps/1e4)
eq += pnl
trades.append(dict(side='BUY', entry_date=entry_date, entry=entry,
exit_date=today, exit=exit_px, reason='TAKE',
qty=pos_qty, pnl=pnl))
pos_qty = 0; direction = None
else: # SELL
if hi >= stop:
exit_px = stop * (1 + slippage_bps/1e4)
pnl = pos_qty * (entry - exit_px) - abs(pos_qty)*exit_px*(cost_bps/1e4)
eq += pnl
trades.append(dict(side='SELL', entry_date=entry_date, entry=entry,
exit_date=today, exit=exit_px, reason='STOP',
qty=pos_qty, pnl=pnl))
pos_qty = 0; direction = None
elif lo <= take:
exit_px = take * (1 + slippage_bps/1e4)
pnl = pos_qty * (entry - exit_px) - abs(pos_qty)*exit_px*(cost_bps/1e4)
eq += pnl
trades.append(dict(side='SELL', entry_date=entry_date, entry=entry,
exit_date=today, exit=exit_px, reason='TAKE',
qty=pos_qty, pnl=pnl))
pos_qty = 0; direction = None


# abre nova
if pos_qty == 0:
if p >= threshold_buy:
direction = 'BUY'
entry, stop, take = entry_stop_gain(px, atr, direction, atr_mult_stop, rr)
qty = position_size(eq, entry, stop, risk_perc)
if qty > 0:
# custo de entrada
entry_fill = entry * (1 + slippage_bps/1e4)
eq -= abs(qty)*entry_fill*(cost_bps/1e4)
pos_qty = qty
entry = entry_fill
entry_date = today
elif p <= threshold_sell:
direction = 'SELL'
entry, stop, take = entry_stop_gain(px, atr, direction, atr_mult_stop, rr)
qty = position_size(eq, entry, stop, risk_perc)
if qty > 0:
entry_fill = entry * (1 - slippage_bps/1e4)
eq -= abs(qty)*entry_fill*(cost_bps/1e4)
pos_qty = qty
entry = entry_fill
entry_date = today


equity.append(eq)


out = pd.DataFrame({'Equity': equity}, index=df.index[1:len(equity)+1])
out['Ret'] = out['Equity'].pct_change().fillna(0.0)
sharpe = np.sqrt(252) * (out['Ret'].mean() / (out['Ret'].std() + 1e-12))


trades_df = pd.DataFrame(trades)
if not trades_df.empty:
trades_df['cum_pnl'] = trades_df['pnl'].cumsum()


stats = {
'final_equity': float(out['Equity'].iloc[-1]) if len(out) else float(capital0),
'sharpe': float(sharpe) if len(out) else 0.0,
'trades': int(len(trades_df)),
'hit_rate': float((trades_df['pnl']>0).mean()) if not trades_df.empty else 0.0,
'avg_pnl': float(trades_df['pnl'].mean()) if not trades_df.empty else 0.0,
'total_pnl': float(trades_df['pnl'].sum()) if not trades_df.empty else 0.0,
}
return out, stats, trades_df
