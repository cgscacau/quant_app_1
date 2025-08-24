import json
from datetime import date, timedelta
import pandas as pd
import numpy as np
import streamlit as st

from core.data import download_prices, add_features
from core.utils import train_test_split_by_date, last_valid
from core.models_arima import ARIMAModel
from core.models_garch import GARCHModel
from core.models_rf import RandomForestSignal
from core.models_trend import TrendScoreModel
from core.ensemble import weighted_ensemble
from core.risk import entry_stop_gain, position_size, kelly_fraction
from core.backtest import simulate_prob_strategy
from core.visual import price_candles, line_series, equity_curve

st.set_page_config(page_title="Quant App — Ensemble & Risco", layout="wide")

st.sidebar.header("Configuração")
with open("assets/tickers.json", "r", encoding="utf-8") as f:
    tickers = json.load(f)

universo = st.sidebar.selectbox("Universo", list(tickers.keys()), index=0)
ticker = st.sidebar.selectbox("Ativo", tickers[universo], index=0)

today = date.today()
d_start = st.sidebar.date_input("Data inicial", value=today - timedelta(days=365*5))
d_end   = st.sidebar.date_input("Data final", value=today)
interval = st.sidebar.selectbox("Intervalo", ["1d","1h","1wk"], index=0)

st.sidebar.subheader("Split Treino/Teste")
train_end_ratio = st.sidebar.slider("Proporção de Treino", 0.5, 0.9, 0.7, 0.05)

st.sidebar.subheader("Modelos e Pesos para Ensemble")
use_arima  = st.sidebar.checkbox("ARIMA", True)
use_garch  = st.sidebar.checkbox("GARCH (vol)", True)
use_rf     = st.sidebar.checkbox("RandomForest (ML)", True)
use_trend  = st.sidebar.checkbox("Trend Score (Logit)", True)

w_arima = st.sidebar.slider("Peso ARIMA", 0.0, 1.0, 0.30, 0.05)
w_garch = st.sidebar.slider("Peso GARCH", 0.0, 1.0, 0.10, 0.05)
w_rf    = st.sidebar.slider("Peso RF",    0.0, 1.0, 0.40, 0.05)
w_trend = st.sidebar.slider("Peso Trend", 0.0, 1.0, 0.20, 0.05)

st.sidebar.subheader("Risco & Trade")
capital = st.sidebar.number_input("Capital (BRL/USD)", 100000.0, step=1000.0)
risk_perc = st.sidebar.slider("Risco por trade (%)", 0.0025, 0.05, 0.01, 0.0025)
atr_mult_stop = st.sidebar.slider("Stop (x ATR)", 1.0, 5.0, 2.0, 0.5)
rr = st.sidebar.slider("Risco:Retorno (R)", 1.0, 5.0, 2.0, 0.5)
th_buy = st.sidebar.slider("Threshold BUY", 0.5, 0.9, 0.55, 0.01)
th_sell = st.sidebar.slider("Threshold SELL", 0.1, 0.5, 0.45, 0.01)

st.title("📈 Quant App — Ensemble de Métodos com Gestão de Risco")

with st.spinner("Baixando dados do Yahoo Finance..."):
    px = download_prices(ticker, str(d_start), str(d_end), interval=interval)
    if len(px) < 250:
        st.warning("Poucos dados retornados — considere ampliar a janela ou usar outro intervalo.")
    df = add_features(px)

split_idx = int(len(df) * train_end_ratio)
train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

st.markdown("### Preço")
st.plotly_chart(price_candles(df), use_container_width=True)

results = []
explanations = []

if use_arima:
    arima = ARIMAModel(order=(1,0,0)).fit(train)
    res_a = arima.predict_next(df)
    results.append(res_a)
    explanations.append(("ARIMA", res_a))

if use_garch:
    garch = GARCHModel().fit(train)
    res_g = garch.predict_next(df)
    results.append(res_g)
    explanations.append(("GARCH", res_g))

if use_rf:
    rf = RandomForestSignal().fit(train, test)
    res_rf = rf.predict_next(df)
    results.append(res_rf)
    explanations.append(("RandomForest", res_rf))

if use_trend:
    trend = TrendScoreModel().fit(train, test)
    res_t = trend.predict_next(df)
    results.append(res_t)
    explanations.append(("Trend", res_t))

weights = {'ARIMA': w_arima, 'GARCH': w_garch, 'RandomForest': w_rf, 'TrendScore': w_trend}
ens = weighted_ensemble(results, weights)
p_up = ens['prob_up']
direction = "BUY" if p_up >= th_buy else ("SELL" if p_up <= th_sell else "NEUTRAL")

last_close = float(df['Close'].iloc[-1])
last_atr = float(df['ATR14'].iloc[-1])

from core.risk import entry_stop_gain, position_size, kelly_fraction
entry = stop = gain = None
qty = 0
kelly_f = 0.0

if direction != "NEUTRAL":
    entry, stop, gain = entry_stop_gain(last_close, last_atr, direction, atr_mult_stop, rr)
    qty = position_size(capital, entry, stop, risk_perc)
    kelly_f = kelly_fraction(p_up if direction=='BUY' else (1-p_up), rr)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prob. de Alta (Ensemble)", f"{p_up:.1%}")
c2.metric("Direção", direction)
c3.metric("Último Close", f"{last_close:,.2f}")
c4.metric("ATR(14)", f"{last_atr:,.2f}")

if direction != "NEUTRAL":
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Entrada", f"{entry:,.2f}")
    c6.metric("Stop", f"{stop:,.2f}")
    c7.metric("Alvo (R)", f"{gain:,.2f}")
    c8.metric("Qtd. sugerida", f"{qty:,}")
    st.caption(f"Kelly fracionado sugerido: **{kelly_f:.2%}** (use fração conservadora, ex. 0.5x).")

import json
st.markdown("### Explicação dos Modelos")
for name, res in explanations:
    with st.expander(f"{name} — detalhes"):
        if name == "RandomForest":
            # Tabela do classification_report
            rep = res.get("report", {})
            if rep:
                rep_df = pd.DataFrame(rep).T
                st.dataframe(rep_df.style.format("{:.4f}"))
            # Importância das features
            fi = res.get("feature_importances", {})
            if fi:
                st.bar_chart(pd.Series(fi, name="importance"))
            # Demais campos do resultado (sem objetos grandes)
            small = {k: v for k, v in res.items() if k not in ("report", "feature_importances")}
            st.code(json.dumps(small, indent=2, ensure_ascii=False))
        else:
            # Modelos simples: renderiza como JSON legível
            st.code(json.dumps(res, indent=2, ensure_ascii=False))

st.markdown("### Backtest — Período de Teste")
prob_series = pd.Series(index=test.index, dtype=float).fillna(0.5)
eq, stats = simulate_prob_strategy(test, prob_series, threshold_buy=th_buy, threshold_sell=th_sell,
                                   capital0=capital, risk_perc=risk_perc, atr_mult_stop=atr_mult_stop, rr=rr)

c9, c10 = st.columns(2)
with c9:
    st.plotly_chart(line_series(eq, 'Equity', title='Equity Curve'), use_container_width=True)
with c10:
    st.dataframe(eq.tail(10))

st.markdown("#### Indicadores do Backtest")
st.write(stats)

st.caption("Protótipo modular; para produção, implementar walk-forward, custos e tuning robusto.")
