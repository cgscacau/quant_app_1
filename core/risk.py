from __future__ import annotations




def entry_stop_gain(last_close: float, atr_value: float, direction: str,
atr_mult_stop: float = 2.0, rr: float = 2.0):
if direction == 'BUY':
entry = last_close
stop = last_close - atr_mult_stop * atr_value
gain = entry + rr * (entry - stop)
elif direction == 'SELL':
entry = last_close
stop = last_close + atr_mult_stop * atr_value
gain = entry - rr * (stop - entry)
else:
return None, None, None
return float(entry), float(stop), float(gain)




def position_size(capital: float, entry: float, stop: float, risk_perc: float = 0.01):
risk_amount = capital * risk_perc
per_unit = abs(entry - stop)
if per_unit <= 1e-8:
return 0
qty = int(risk_amount // per_unit)
return max(qty, 0)




def kelly_fraction(prob_win: float, rr: float = 2.0):
p = prob_win
b = rr
f = (p*(b+1)-1)/b
return max(0.0, min(f, 1.0))
