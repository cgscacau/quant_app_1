# app.py — Quant App (UI moderna com cards) — versão 2
st.dataframe(pd.DataFrame(rep).T.style.format("{:.4f}"))
fi = res.get("feature_importances", {})
if fi:
st.bar_chart(pd.Series(fi, name="importância"))
elif name == "Trend":
render_model_card("TrendScore", {k:v for k,v in res.items() if k != "coefficients"}, "#F06292")
coefs = res.get("coefficients", {})
if coefs:
st.bar_chart(pd.Series(coefs, name="coef"))


with bt_tab:
st.subheader("Backtest — Período de Teste")
probs: List[pd.Series] = []; cols: List[str] = []
if use_arima: probs.append(ps_arima(arima, test)); cols.append("ARIMA")
if use_rf: probs.append(ps_rf(rf, train, test)); cols.append("RandomForest")
if use_trend: probs.append(ps_trend(trend, train, test)); cols.append("TrendScore")
if use_garch: probs.append(_to_series(0.5, test.index)); cols.append("GARCH")
if not probs: probs = [_to_series(0.5, test.index)]; cols = ["Fallback"]


probs_df = pd.concat(probs, axis=1); probs_df.columns = cols
w_cfg = {"ARIMA": w_arima, "GARCH": w_garch, "RandomForest": w_rf, "TrendScore": w_trend}
present = [c for c in probs_df.columns if w_cfg.get(c,0)>0] or probs_df.columns.tolist()
weights = pd.Series({c: w_cfg.get(c,0.0) for c in present}); weights = weights / (weights.sum() or 1.0)
prob_series = (probs_df[present] * weights).sum(axis=1).clip(0,1)


eq, stats, blotter = simulate_prob_strategy(
test, prob_series,
threshold_buy=th_buy, threshold_sell=th_sell,
capital0=capital, risk_perc=risk_perc,
atr_mult_stop=atr_mult_stop, rr=rr,
)


g1, g2 = st.columns(2)
with g1: st.plotly_chart(line_series(eq, 'Equity', title='Equity Curve'), use_container_width=True)
with g2: st.dataframe(eq.tail(12))


st.markdown("<div class='card'><h4>Indicadores</h4>" +
f"<pre class='pretty'>{json.dumps(stats, indent=2, ensure_ascii=False)}</pre></div>",
unsafe_allow_html=True)


st.session_state['prob_series'] = prob_series
st.session_state['probs_df'] = probs_df
st.session_state['blotter'] = blotter


with tr_tab:
st.subheader("Blotter de Trades")
blotter = st.session_state.get('blotter', pd.DataFrame())
if blotter is not None and len(blotter) > 0:
st.dataframe(blotter, use_container_width=True, height=420)
st.download_button("Baixar trades (CSV)", blotter.to_csv(index=False).encode('utf-8'),
file_name='trades.csv', mime='text/csv')
else:
st.info("Sem trades no período/configuração atual. Ajuste thresholds/pesos para gerar sinais.")


with pr_tab:
st.subheader("Série de Probabilidades")
prob_series = st.session_state.get('prob_series')
probs_df = st.session_state.get('probs_df')
if isinstance(prob_series, pd.Series):
st.line_chart(prob_series.rename('prob_up (ensemble)'))
if isinstance(probs_df, pd.DataFrame):
st.dataframe(probs_df.tail(12))


st.caption("Protótipo educacional. Para produção: custos/deslizamento, walk-forward, tuning e validação rigorosa.")
