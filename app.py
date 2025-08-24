# app.py — Quant App (Ensemble & Risco) — versão UI/UX refinada
# Compatível com os módulos já existentes em core/*
# Autor: você :)  — foco em clareza, explicação e visualização

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Dict, List

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

# =============================================================================
# CONFIGURAÇÃO GERAL
# =============================================================================
st.set_page_config(page_title="Quant App — Ensemble & Risco", layout="wide")

# Paleta/tema geral do Streamlit (modo escuro recomendado nas configs da conta)
PRIMARY = "#7BD389"   # verde suave para destaques
ACCENT  = "#8FA3FF"   # azul p/ gráficos auxiliares

# =============================================================================
# SIDEBAR — ENTRADA DO USUÁRIO
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configurações")
    st.caption("Escolha o ativo, o período de dados e parâmetros dos modelos e do risco.")

    with open("assets/tickers.json", "r", encoding="utf-8") as f:
        tickers = json.load(f)

    col_univ, col_asset = st.columns(2)
    with col_univ:
        universo = st.selectbox("Universo", list(tickers.keys()), index=0)
    with col_asset:
        ticker = st.selectbox("Ativo", tickers[universo], index=0)

    today = date.today()
    d_start = st.date_input("Data inicial", value=today - timedelta(days=365 * 5))
    d_end = st.date_input("Data final", value=today)
    interval = st.selectbox("Intervalo", ["1d", "1h", "1wk"], index=0, help="Frequência dos candles.")

    st.subheader("Treino vs. Teste")
    train_end_ratio = st.slider(
        "Proporção de Treino",
        0.5, 0.9, 0.7, 0.05,
        help="Percentual do histórico usado para TREINAR os modelos; o restante é TESTE."
    )

    st.subheader("Modelos do Ensemble")
    use_arima = st.checkbox("ARIMA (time-series)", True)
    use_garch = st.checkbox("GARCH (volatilidade)", True, help="Não dá direção; ajuda só no risco.")
    use_rf    = st.checkbox("RandomForest (ML)", True)
    use_trend = st.checkbox("Trend Score (Logit)", True)

    st.caption("Pesos determinam a influência de cada modelo na probabilidade combinada.")
    w_arima = st.slider("Peso ARIMA", 0.0, 1.0, 0.30, 0.05)
    w_garch = st.slider("Peso GARCH", 0.0, 1.0, 0.10, 0.05)
    w_rf    = st.slider("Peso RF",    0.0, 1.0, 0.40, 0.05)
    w_trend = st.slider("Peso Trend", 0.0, 1.0, 0.20, 0.05)

    st.subheader("Gestão de Risco")
    capital       = st.number_input("Capital (BRL/USD)", 100000.0, step=1000.0)
    risk_perc     = st.slider("Risco por trade (%)", 0.0025, 0.05, 0.01, 0.0025,
                              help="Parcela do capital arriscada por operação.")
    atr_mult_stop = st.slider("Stop (x ATR14)", 1.0, 5.0, 2.0, 0.5)
    rr            = st.slider("Risco:Retorno (R)", 1.0, 5.0, 2.0, 0.5)
    th_buy        = st.slider("Threshold BUY", 0.50, 0.90, 0.55, 0.01)
    th_sell       = st.slider("Threshold SELL", 0.10, 0.50, 0.45, 0.01)

# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================
st.title("📊 Quant App — Ensemble de Métodos com Gestão de Risco")
st.caption(
    "App educacional para análise quantitativa. Combine modelos, veja a probabilidade de alta, "
    "sinal de compra/venda e gestão de risco. Faça backtest no período de teste."
)

with st.spinner("Baixando dados do Yahoo Finance e preparando features..."):
    px = download_prices(ticker, str(d_start), str(d_end), interval=interval)
    if len(px) < 250:
        st.warning("Poucos dados retornados — considere ampliar a janela ou usar outro intervalo.")
    df = add_features(px)

# Split
split_idx = max(10, int(len(df) * train_end_ratio))
train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

# Preço
st.subheader("Preço")
st.plotly_chart(price_candles(df), use_container_width=True)

# =============================================================================
# TREINO + PREDIÇÃO PONTUAL (AGORA)
# =============================================================================
results: List[Dict] = []
explain: List = []

if use_arima:
    arima = ARIMAModel(order=(1, 0, 0)).fit(train)
    res_a = arima.predict_next(df)
    results.append(res_a); explain.append(("ARIMA", res_a))

if use_garch:
    garch = GARCHModel().fit(train)
    res_g = garch.predict_next(df)
    results.append(res_g); explain.append(("GARCH", res_g))

if use_rf:
    rf = RandomForestSignal().fit(train, test)
    res_rf = rf.predict_next(df)
    results.append(res_rf); explain.append(("RandomForest", res_rf))

if use_trend:
    trend = TrendScoreModel().fit(train, test)
    res_t = trend.predict_next(df)
    results.append(res_t); explain.append(("Trend", res_t))

# Ensemble no ponto atual
weights = {"ARIMA": w_arima, "GARCH": w_garch, "RandomForest": w_rf, "TrendScore": w_trend}
ens_now = weighted_ensemble(results, weights)
p_up_now = ens_now["prob_up"]
signal_now = "BUY" if p_up_now >= th_buy else ("SELL" if p_up_now <= th_sell else "NEUTRAL")

# Cards principais
last_close = float(df["Close"].iloc[-1])
last_atr   = float(df["ATR14"].iloc[-1])

entry = stop = gain = None
qty   = 0
kelly_f = 0.0
if signal_now != "NEUTRAL":
    entry, stop, gain = entry_stop_gain(last_close, last_atr, signal_now, atr_mult_stop, rr)
    qty = position_size(capital, entry, stop, risk_perc)
    kelly_f = kelly_fraction(p_up_now if signal_now == "BUY" else (1 - p_up_now), rr)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prob. de Alta (Ensemble)", f"{p_up_now:.1%}")
c2.metric("Sinal", signal_now)
c3.metric("Último Close", f"{last_close:,.2f}")
c4.metric("ATR(14)", f"{last_atr:,.2f}")

if signal_now != "NEUTRAL":
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Entrada", f"{entry:,.2f}")
    c6.metric("Stop",    f"{stop:,.2f}")
    c7.metric("Alvo (R)",f"{gain:,.2f}")
    c8.metric("Qtd. sugerida", f"{qty:,}")
    st.caption(f"Kelly fracionado sugerido: **{kelly_f:.2%}** (use fração conservadora, ex. 0.5×).")

# =============================================================================
# FUNÇÕES AUXILIARES DO BACKTEST (probabilidade por barra)
# =============================================================================
def _to_series(x, index, name=None) -> pd.Series:
    """Converte array/escalar para Series com índice alinhado."""
    if isinstance(x, pd.Series):
        return x.reindex(index).astype(float)
    arr = np.asarray(x).reshape(-1)
    if arr.size == 1:
        return pd.Series([float(arr[0])] * len(index), index=index, name=name)
    if arr.size != len(index):
        s = pd.Series(arr, index=index[:arr.size], name=name)
        return s.reindex(index).ffill().bfill().astype(float)
    return pd.Series(arr, index=index, name=name).astype(float)

def ps_arima(arima_obj, test_df) -> pd.Series:
    f = arima_obj.res_.get_forecast(steps=len(test_df))
    mu = pd.Series(f.predicted_mean, index=test_df.index)
    sigma = pd.Series(f.se_mean, index=test_df.index).replace(0, 1e-6)
    p = 1.0 - norm.cdf(0.0, loc=mu, scale=sigma)
    return _to_series(p, test_df.index, "ARIMA")

def ps_rf(rf_obj, train_df, test_df) -> pd.Series:
    X_all, _, _ = rf_obj._build_Xy(pd.concat([train_df, test_df]))
    idx = test_df.index[:-1]
    X_test = X_all.reindex(idx).dropna()
    if len(X_test) == 0:
        return _to_series(0.5, test_df.index, "RandomForest")
    Xs = rf_obj.scaler.transform(X_test)
    p = rf_obj.model.predict_proba(Xs)[:, 1]
    s = pd.Series(p, index=X_test.index, name="RandomForest")
    return s.reindex(test_df.index).ffill().bfill().astype(float)

def ps_trend(tr_obj, train_df, test_df) -> pd.Series:
    F_all = tr_obj._features(pd.concat([train_df, test_df]))
    idx = test_df.index[:-1]
    X_test = F_all.reindex(idx).dropna()
    if len(X_test) == 0:
        return _to_series(0.5, test_df.index, "TrendScore")
    p = tr_obj.clf.predict_proba(X_test)[:, 1]
    s = pd.Series(p, index=X_test.index, name="TrendScore")
    return s.reindex(test_df.index).ffill().bfill().astype(float)

# =============================================================================
# ABAS — Overview | Modelos | Backtest | Trades | Probabilidades
# =============================================================================
tab_over, tab_models, tab_bt, tab_trades, tab_probs = st.tabs(
    ["📌 Overview", "🧠 Modelos", "🔁 Backtest", "🧾 Trades", "📈 Probabilidades"]
)

# -- OVERVIEW: só um texto de ajuda curto
with tab_over:
    st.markdown("""
**Como ler:**
- A **probabilidade de alta (Ensemble)** combina os modelos pelos **pesos** definidos na sidebar.
- Se `P ≥ BUY` ⇒ sinal **BUY**; se `P ≤ SELL` ⇒ **SELL**; entre eles ⇒ **NEUTRAL**.
- A gestão de risco usa **ATR(14)** para stop/take e **risk%** para dimensionar a posição.
- O **Backtest** usa a série de probabilidades ao longo do período de **teste**.
""")

# -- MODELOS: explicações ricas
with tab_models:
    st.subheader("Explicação dos Modelos")
    import json as _json  # evitar conflito com json da stdlib no dump abaixo

    for name, res in explain:
        with st.expander(f"{name} — detalhes"):
            if name == "RandomForest":
                rep = res.get("report", {})
                if rep:
                    rep_df = pd.DataFrame(rep).T
                    st.dataframe(rep_df.style.format("{:.4f}"))
                fi = res.get("feature_importances", {})
                if fi:
                    st.bar_chart(pd.Series(fi, name="importance"))
                small = {k: v for k, v in res.items() if k not in ("report", "feature_importances")}
                st.code(_json.dumps(small, indent=2, ensure_ascii=False))
                # métricas extras
                try:
                    Xte, yte, _ = rf._build_Xy(test)
                    if len(Xte) > 0:
                        from sklearn.metrics import roc_auc_score, confusion_matrix
                        Xte_s = rf.scaler.transform(Xte)
                        yprob = rf.model.predict_proba(Xte_s)[:, 1]
                        yhat  = (yprob >= 0.5).astype(int)
                        auc   = roc_auc_score(yte, yprob)
                        cm = pd.DataFrame(confusion_matrix(yte, yhat),
                                          index=["Actual 0","Actual 1"],
                                          columns=["Pred 0","Pred 1"])
                        st.metric("ROC AUC (teste)", f"{auc:.3f}")
                        st.dataframe(cm)
                except Exception as e:
                    st.caption(f"Obs.: não foi possível calcular métricas extras ({e}).")

            elif name == "Trend":
                coefs = res.get("coefficients", {})
                if coefs:
                    st.bar_chart(pd.Series(coefs, name="coef"))
                small = {k: v for k, v in res.items() if k not in ("coefficients",)}
                st.code(_json.dumps(small, indent=2, ensure_ascii=False))

            else:
                st.code(_json.dumps(res, indent=2, ensure_ascii=False))

# -- BACKTEST: calcula probs por barra, faz ensemble e simula
with tab_bt:
    st.subheader("Backtest — Período de Teste")
    probs: List[pd.Series] = []
    cols:  List[str] = []

    if len(test) < 10:
        st.warning("Janela de teste muito curta; ajuste a proporção ou datas.")
    else:
        if use_arima:
            s = ps_arima(arima, test); probs.append(s); cols.append("ARIMA")
        if use_rf:
            s = ps_rf(rf, train, test); probs.append(s); cols.append("RandomForest")
        if use_trend:
            s = ps_trend(trend, train, test); probs.append(s); cols.append("TrendScore")
        if use_garch:
            probs.append(_to_series(0.5, test.index, "GARCH")); cols.append("GARCH")

        if not probs:
            probs = [_to_series(0.5, test.index, "Fallback")]; cols = ["Fallback"]

        probs_df = pd.concat(probs, axis=1); probs_df.columns = cols

        # ensemble por barra (normaliza pesos só dos presentes)
        w_cfg = {"ARIMA": w_arima, "GARCH": w_garch, "RandomForest": w_rf, "TrendScore": w_trend}
        present = [c for c in probs_df.columns if w_cfg.get(c, 0) > 0]
        if not present:
            present = probs_df.columns.tolist()
        weights = pd.Series({c: w_cfg.get(c, 0.0) for c in present})
        weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)

        prob_series = (probs_df[present] * weights).sum(axis=1).clip(0, 1)

        # simulação + resultados
        eq, stats, blotter = simulate_prob_strategy(
            test, prob_series,
            threshold_buy=th_buy, threshold_sell=th_sell,
            capital0=capital, risk_perc=risk_perc,
            atr_mult_stop=atr_mult_stop, rr=rr
        )

        c9, c10 = st.columns(2)
        with c9:
            st.plotly_chart(line_series(eq, "Equity", title="Equity Curve"), use_container_width=True)
        with c10:
            st.dataframe(eq.tail(12))

        st.markdown("##### Indicadores do Backtest")
        st.json(stats, expanded=True)

        # guarda os objetos no session_state para as outras abas
        st.session_state["prob_series"] = prob_series
        st.session_state["probs_df"]    = probs_df
        st.session_state["blotter"]     = blotter

# -- TRADES: mostra blotter e exporta CSV
with tab_trades:
    st.subheader("Blotter de Trades")
    blotter = st.session_state.get("blotter", pd.DataFrame())
    if blotter is not None and len(blotter) > 0:
        st.dataframe(blotter, use_container_width=True, height=420)
        st.download_button(
            "⬇️ Baixar trades (CSV)",
            blotter.to_csv(index=False).encode("utf-8"),
            file_name="trades.csv", mime="text/csv"
        )
    else:
        st.info("Sem trades no período/parametrização atual. Ajuste thresholds e pesos para testar.")

# -- PROBABILIDADES: séries por modelo + ensemble
with tab_probs:
    st.subheader("Série de Probabilidades")
    prob_series = st.session_state.get("prob_series", None)
    probs_df    = st.session_state.get("probs_df", None)
    if isinstance(prob_series, pd.Series):
        st.line_chart(prob_series.rename("prob_up (ensemble)"))
    if isinstance(probs_df, pd.DataFrame):
        st.dataframe(probs_df.tail(12))

st.caption(
    "⚠️ Protótipo educacional. Para produção: custos/deslizamento, walk-forward, "
    "regularização e tuning, validação fora da amostra e gestão de risco institucional."
)
