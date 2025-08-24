# app.py — Quant App (Ensemble & Risco) — versão corrigida

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from core.data import download_prices, add_features
from core.models_arima import ARIMAModel
from core.models_garch import GARCHModel
from core.models_rf import RandomForestSignal
from core.models_trend import TrendScoreModel
from core.ensemble import weighted_ensemble
from core.risk import entry_stop_gain, position_size, kelly_fraction
from core.backtest import simulate_prob_strategy
from core.visual import price_candles, line_series

# ------------------------------------------------------------------------------
# Config da página
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Quant App — Ensemble & Risco", layout="wide")

# ------------------------------------------------------------------------------
# Sidebar — parâmetros
# ------------------------------------------------------------------------------
st.sidebar.header("Configuração")
with open("assets/tickers.json", "r", encoding="utf-8") as f:
    tickers = json.load(f)

universo = st.sidebar.selectbox("Universo", list(tickers.keys()), index=0)
ticker = st.sidebar.selectbox("Ativo", tickers[universo], index=0)

today = date.today()
d_start = st.sidebar.date_input("Data inicial", value=today - timedelta(days=365 * 5))
d_end = st.sidebar.date_input("Data final", value=today)
interval = st.sidebar.selectbox("Intervalo", ["1d", "1h", "1wk"], index=0)

st.sidebar.subheader("Split Treino/Teste")
train_end_ratio = st.sidebar.slider("Proporção de Treino", 0.5, 0.9, 0.7, 0.05)

st.sidebar.subheader("Modelos e Pesos para Ensemble")
use_arima = st.sidebar.checkbox("ARIMA", True)
use_garch = st.sidebar.checkbox("GARCH (vol)", True)
use_rf = st.sidebar.checkbox("RandomForest (ML)", True)
use_trend = st.sidebar.checkbox("Trend Score (Logit)", True)

w_arima = st.sidebar.slider("Peso ARIMA", 0.0, 1.0, 0.30, 0.05)
w_garch = st.sidebar.slider("Peso GARCH", 0.0, 1.0, 0.10, 0.05)
w_rf = st.sidebar.slider("Peso RF", 0.0, 1.0, 0.40, 0.05)
w_trend = st.sidebar.slider("Peso Trend", 0.0, 1.0, 0.20, 0.05)

st.sidebar.subheader("Risco & Trade")
capital = st.sidebar.number_input("Capital (BRL/USD)", 100000.0, step=1000.0)
risk_perc = st.sidebar.slider("Risco por trade (%)", 0.0025, 0.05, 0.01, 0.0025)
atr_mult_stop = st.sidebar.slider("Stop (x ATR)", 1.0, 5.0, 2.0, 0.5)
rr = st.sidebar.slider("Risco:Retorno (R)", 1.0, 5.0, 2.0, 0.5)
th_buy = st.sidebar.slider("Threshold BUY", 0.5, 0.9, 0.55, 0.01)
th_sell = st.sidebar.slider("Threshold SELL", 0.1, 0.5, 0.45, 0.01)

# ------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------
st.title("📈 Quant App — Ensemble de Métodos com Gestão de Risco")

with st.spinner("Baixando dados do Yahoo Finance..."):
    px = download_prices(ticker, str(d_start), str(d_end), interval=interval)
    if len(px) < 250:
        st.warning("Poucos dados retornados — considere ampliar a janela ou usar outro intervalo.")
    df = add_features(px)

# Split treino/teste
split_idx = max(10, int(len(df) * train_end_ratio))
train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
if len(test) < 40:
    st.warning("Janela de teste muito curta; ajuste a proporção ou datas.")

# Preço
st.markdown("### Preço")
st.plotly_chart(price_candles(df), use_container_width=True)

# ------------------------------------------------------------------------------
# Modelos (ponto atual)
# ------------------------------------------------------------------------------
results = []
explanations = []

if use_arima:
    arima = ARIMAModel(order=(1, 0, 0)).fit(train)
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

# Ensemble e sinal atual
weights = {"ARIMA": w_arima, "GARCH": w_garch, "RandomForest": w_rf, "TrendScore": w_trend}
ens = weighted_ensemble(results, weights)
p_up_now = ens["prob_up"]
direction = "BUY" if p_up_now >= th_buy else ("SELL" if p_up_now <= th_sell else "NEUTRAL")

last_close = float(df["Close"].iloc[-1])
last_atr = float(df["ATR14"].iloc[-1])

entry = stop = gain = None
qty = 0
kelly_f = 0.0

if direction != "NEUTRAL":
    entry, stop, gain = entry_stop_gain(last_close, last_atr, direction, atr_mult_stop, rr)
    qty = position_size(capital, entry, stop, risk_perc)
    kelly_f = kelly_fraction(p_up_now if direction == "BUY" else (1 - p_up_now), rr)

# Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Prob. de Alta (Ensemble)", f"{p_up_now:.1%}")
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

# ------------------------------------------------------------------------------
# Explicação dos modelos (renderização segura)
# ------------------------------------------------------------------------------
st.markdown("### Explicação dos Modelos")
for name, res in explanations:
    with st.expander(f"{name} — detalhes"):
        if name == "RandomForest":
            # Tabela do classification_report
            rep = res.get("report", {})
            if rep:
                rep_df = pd.DataFrame(rep).T
                st.dataframe(rep_df.style.format("{:.4f}"))
            # Importâncias
            fi = res.get("feature_importances", {})
            if fi:
                st.bar_chart(pd.Series(fi, name="importance"))
            # Resumo
            small = {k: v for k, v in res.items() if k not in ("report", "feature_importances")}
            st.code(json.dumps(small, indent=2, ensure_ascii=False))

            # Métricas extras no teste (ROC/AUC + Confusion Matrix)
            try:
                Xte, yte, _ = rf._build_Xy(test)
                if len(Xte) > 0:
                    from sklearn.metrics import roc_auc_score, confusion_matrix

                    Xte_s = rf.scaler.transform(Xte)
                    yprob = rf.model.predict_proba(Xte_s)[:, 1]
                    yhat = (yprob >= 0.5).astype(int)
                    auc = roc_auc_score(yte, yprob)
                    cm = pd.DataFrame(
                        confusion_matrix(yte, yhat),
                        index=["Actual 0", "Actual 1"],
                        columns=["Pred 0", "Pred 1"],
                    )
                    st.metric("ROC AUC (teste)", f"{auc:.3f}")
                    st.dataframe(cm)
            except Exception as e:
                st.caption(f"Obs.: não foi possível calcular métricas extras ({e}).")

        elif name == "Trend":
            coefs = res.get("coefficients", {})
            if coefs:
                st.bar_chart(pd.Series(coefs, name="coef"))
            small = {k: v for k, v in res.items() if k not in ("coefficients",)}
            st.code(json.dumps(small, indent=2, ensure_ascii=False))
        else:
            st.code(json.dumps(res, indent=2, ensure_ascii=False))

# ------------------------------------------------------------------------------
# Backtest — período de teste
# ------------------------------------------------------------------------------
st.markdown("### Backtest — Período de Teste")


def _to_series(x, index, name=None):
    """Garante um pd.Series com o índice correto, a partir de Series/array/escalar."""
    if isinstance(x, pd.Series):
        return x.reindex(index).astype(float)
    arr = np.asarray(x).reshape(-1)
    if arr.size == 1:
        return pd.Series([float(arr[0])] * len(index), index=index, name=name)
    if arr.size != len(index):
        s = pd.Series(arr, index=index[: arr.size], name=name)
        return s.reindex(index).ffill().bfill().astype(float)
    return pd.Series(arr, index=index, name=name).astype(float)


def prob_series_arima(arima_obj, train_df, test_df):
    f = arima_obj.res_.get_forecast(steps=len(test_df))
    mu = pd.Series(f.predicted_mean, index=test_df.index)
    sigma = pd.Series(f.se_mean, index=test_df.index).replace(0, 1e-6)
    p = 1.0 - norm.cdf(0.0, loc=mu, scale=sigma)
    return _to_series(p, test_df.index, "ARIMA")


def prob_series_rf(rf_obj, train_df, test_df):
    X_all, _, _ = rf_obj._build_Xy(pd.concat([train_df, test_df]))
    idx = test_df.index[:-1]
    X_test = X_all.reindex(idx).dropna()
    if len(X_test) == 0:
        return _to_series(0.5, test_df.index, "RandomForest")
    Xs = rf_obj.scaler.transform(X_test)
    p = rf_obj.model.predict_proba(Xs)[:, 1]
    s = pd.Series(p, index=X_test.index, name="RandomForest")
    return s.reindex(test_df.index).ffill().bfill().astype(float)


def prob_series_trend(tr_obj, train_df, test_df):
    F_all = tr_obj._features(pd.concat([train_df, test_df]))
    idx = test_df.index[:-1]
    X_test = F_all.reindex(idx).dropna()
    if len(X_test) == 0:
        return _to_series(0.5, test_df.index, "TrendScore")
    p = tr_obj.clf.predict_proba(X_test)[:, 1]
    s = pd.Series(p, index=X_test.index, name="TrendScore")
    return s.reindex(test_df.index).ffill().bfill().astype(float)


# Probabilidades por modelo (séries seguras)
probs = []
cols = []

if use_arima:
    ps = prob_series_arima(arima, train, test)
    probs.append(ps)
    cols.append("ARIMA")
if use_rf:
    ps = prob_series_rf(rf, train, test)
    probs.append(ps)
    cols.append("RandomForest")
if use_trend:
    ps = prob_series_trend(trend, train, test)
    probs.append(ps)
    cols.append("TrendScore")
if use_garch:
    probs.append(_to_series(0.5, test.index, "GARCH"))
    cols.append("GARCH")

if len(probs) == 0:
    probs = [_to_series(0.5, test.index, "Fallback")]
    cols = ["Fallback"]

probs_df = pd.concat(probs, axis=1)
probs_df.columns = cols

# Ensemble por barra (normalizando apenas modelos presentes)
w_cfg = {"ARIMA": w_arima, "GARCH": w_garch, "RandomForest": w_rf, "TrendScore": w_trend}
present = [c for c in probs_df.columns if w_cfg.get(c, 0) > 0]
if not present:
    present = probs_df.columns.tolist()
weights = pd.Series({c: w_cfg.get(c, 0.0) for c in present})
weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)

prob_series = (probs_df[present] * weights).sum(axis=1).clip(0, 1)

# Simulação (com blotter de trades)
eq, stats, blotter = simulate_prob_strategy(
    test,
    prob_series,
    threshold_buy=th_buy,
    threshold_sell=th_sell,
    capital0=capital,
    risk_perc=risk_perc,
    atr_mult_stop=atr_mult_stop,
    rr=rr,
)

# Visual
c9, c10 = st.columns(2)
with c9:
    st.plotly_chart(line_series(eq, "Equity", title="Equity Curve"), use_container_width=True)
with c10:
    st.dataframe(eq.tail(10))

st.markdown("#### Indicadores do Backtest")
st.write(stats)

st.markdown("#### Blotter de Trades")
if blotter is not None and len(blotter) > 0:
    st.dataframe(blotter.tail(20))
    st.download_button(
        "Baixar trades (CSV)",
        blotter.to_csv(index=False).encode("utf-8"),
        file_name="trades.csv",
        mime="text/csv",
    )
else:
    st.caption("Sem trades no período/configuração atual.")

with st.expander("Ver série de probabilidades do ensemble"):
    st.line_chart(prob_series.rename("prob_up (ensemble)"))
    st.dataframe(probs_df.tail(10))

st.caption(
    "Protótipo modular; para produção, implemente walk-forward, custos/deslizamento e tuning robusto."
)
