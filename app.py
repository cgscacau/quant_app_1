# ============================================================================ #
# app.py — Quant App (UI moderna, blocos bem separados e robustos)
# ============================================================================ #

from __future__ import annotations

# [BLOCO 0] — IMPORTS (não deve haver NADA antes destas linhas)
import json
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

# Módulos do projeto (pasta core/)
from core.data import download_prices, add_features
from core.models_arima import ARIMAModel
from core.models_garch import GARCHModel
from core.models_rf import RandomForestSignal
from core.models_trend import TrendScoreModel
from core.ensemble import weighted_ensemble
from core.risk import entry_stop_gain, position_size, kelly_fraction
from core.backtest import simulate_prob_strategy
from core.visual import price_candles, line_series


# [BLOCO 1] — CONFIG VISUAL
st.set_page_config(page_title="Quant App — Ensemble & Risco", layout="wide")

CARD_BG = "#0f1116"
CARD_BD = "#2a2f3a"

st.markdown(
    """
    <style>
      .card { background:#0f1116; border:1px solid #2a2f3a; border-radius:16px;
              padding:16px; margin:8px 0; }
      .card h4 { margin:0 0 10px 0; font-weight:700; }
      .kpi  { font-size:28px; font-weight:700; }
      .sub  { opacity:.8; font-size:12px; }
      pre.pretty { background:#131622; border-radius:12px; padding:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# [BLOCO 2] — SIDEBAR (parâmetros do usuário)
with st.sidebar:
    st.header("⚙️ Configurações")

    # Universo / ticker
    with open("assets/tickers.json", "r", encoding="utf-8") as f:
        TICKERS = json.load(f)
    c1, c2 = st.columns(2)
    with c1:
        universo = st.selectbox("Universo", list(TICKERS.keys()), index=0)
    with c2:
        ticker = st.selectbox("Ativo", TICKERS[universo], index=0)

    # Janela temporal
    today = date.today()
    d_start = st.date_input("Data inicial", value=today - timedelta(days=365 * 5))
    d_end = st.date_input("Data final", value=today)
    interval = st.selectbox("Intervalo", ["1d", "1h", "1wk"], index=0)

    # Split treino/teste
    st.subheader("Treino vs. Teste")
    train_end_ratio = st.slider("Proporção de Treino", 0.5, 0.9, 0.7, 0.05)

    # Modelos & pesos
    st.subheader("Modelos do Ensemble")
    use_arima = st.checkbox("ARIMA", True)
    use_garch = st.checkbox("GARCH (volatilidade)", True)
    use_rf = st.checkbox("RandomForest (ML)", True)
    use_trend = st.checkbox("TrendScore (Logit)", True)

    st.caption("Pesos controlam a influência de cada modelo no ensemble.")
    w_arima = st.slider("Peso ARIMA", 0.0, 1.0, 0.30, 0.05)
    w_garch = st.slider("Peso GARCH", 0.0, 1.0, 0.10, 0.05)
    w_rf = st.slider("Peso RF", 0.0, 1.0, 0.40, 0.05)
    w_trend = st.slider("Peso Trend", 0.0, 1.0, 0.20, 0.05)

    # Risco
    st.subheader("Gestão de Risco")
    capital = st.number_input("Capital (BRL/USD)", 100000.0, step=1000.0)
    risk_perc = st.slider("Risco por trade (%)", 0.0025, 0.05, 0.01, 0.0025)
    atr_mult_stop = st.slider("Stop (x ATR14)", 1.0, 5.0, 2.0, 0.5)
    rr = st.slider("Risco:Retorno (R)", 1.0, 5.0, 2.0, 0.5)
    th_buy = st.slider("Threshold BUY", 0.50, 0.90, 0.55, 0.01)
    th_sell = st.slider("Threshold SELL", 0.10, 0.50, 0.45, 0.01)
    st.caption("Dica: 0.52/0.48 = agressivo; 0.60/0.40 = conservador.")


# [BLOCO 3] — DADOS & FEATURES
st.title("📈 Quant App — Ensemble de Métodos com Gestão de Risco")
st.caption("Combine modelos para gerar probabilidade, direção e plano de trade. Faça backtest no período de teste.")

with st.spinner("Baixando dados e calculando features..."):
    px = download_prices(ticker, str(d_start), str(d_end), interval=interval)
    if len(px) < 250:
        st.warning("Poucos dados — amplie o período ou troque o intervalo.")
    df = add_features(px)

split_idx = max(10, int(len(df) * train_end_ratio))
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

st.subheader("Preço")
st.plotly_chart(price_candles(df), use_container_width=True)


# [BLOCO 4] — HELPERS ROBUSTOS (evitam concat de ndarray e NaN)
def _to_series(x, index: pd.Index, name: str) -> pd.Series:
    """
    Garante saída como Series alinhada a `index`.
    - escalar → série constante
    - ndarray/list → série alinhada (corta/preenche)
    - Series → reindex
    """
    if isinstance(x, pd.Series):
        return x.reindex(index).astype(float)

    arr = np.asarray(x).reshape(-1)
    if arr.size == 1:
        return pd.Series([float(arr[0])] * len(index), index=index, name=name)

    if arr.size != len(index):
        s = pd.Series(arr, index=index[: arr.size], name=name).astype(float)
        return s.reindex(index).ffill().bfill()
    return pd.Series(arr, index=index, name=name).astype(float)


def ps_arima_series(model: ARIMAModel, test_df: pd.DataFrame) -> pd.Series:
    """Prob(Up) diária a partir do ARIMA (mu, sigma → P(r>0))."""
    f = model.res_.get_forecast(steps=len(test_df))
    mu = pd.Series(f.predicted_mean, index=test_df.index)
    sigma = pd.Series(f.se_mean, index=test_df.index).replace(0, 1e-6)
    p = 1.0 - norm.cdf(0.0, loc=mu, scale=sigma)
    return _to_series(p, test_df.index, "ARIMA")


def ps_rf_series(model: RandomForestSignal, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """Prob(Up) do RandomForest para o período de teste (alinhado)."""
    X_all, _, _ = model._build_Xy(pd.concat([train_df, test_df]))
    idx = test_df.index[:-1]  # previne look-ahead
    X_test = X_all.reindex(idx).dropna()
    if len(X_test) == 0:
        return _to_series(0.5, test_df.index, "RandomForest")
    Xs = model.scaler.transform(X_test)
    p = model.model.predict_proba(Xs)[:, 1]
    s = pd.Series(p, index=X_test.index, name="RandomForest")
    return _to_series(s, test_df.index, "RandomForest")


def ps_trend_series(model: TrendScoreModel, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """Prob(Up) do TrendScore (logit) para o período de teste."""
    F_all = model._features(pd.concat([train_df, test_df]))
    idx = test_df.index[:-1]
    X_test = F_all.reindex(idx).dropna()
    if len(X_test) == 0:
        return _to_series(0.5, test_df.index, "TrendScore")
    p = model.clf.predict_proba(X_test)[:, 1]
    s = pd.Series(p, index=X_test.index, name="TrendScore")
    return _to_series(s, test_df.index, "TrendScore")


# [BLOCO 5] — TREINO/PREVISÃO (ponto atual) + KPI CARDS
results_now: List[Dict] = []   # para ensemble instantâneo (última barra)
explain: List[Tuple[str, Dict]] = []
models: Dict[str, object] = {}

if use_arima:
    arima = ARIMAModel(order=(1, 0, 0)).fit(train)
    res_a = arima.predict_next(df)  # dict resumido (mu/sigma/prob_up)
    results_now.append(res_a); explain.append(("ARIMA", res_a)); models["ARIMA"] = arima

if use_garch:
    garch = GARCHModel().fit(train)
    res_g = garch.predict_next(df)
    results_now.append(res_g); explain.append(("GARCH", res_g)); models["GARCH"] = garch

if use_rf:
    rf = RandomForestSignal().fit(train, test)
    res_rf = rf.predict_next(df)
    results_now.append(res_rf); explain.append(("RandomForest", res_rf)); models["RandomForest"] = rf

if use_trend:
    trend = TrendScoreModel().fit(train, test)
    res_t = trend.predict_next(df)
    results_now.append(res_t); explain.append(("Trend", res_t)); models["TrendScore"] = trend

weights_cfg = {"ARIMA": w_arima, "GARCH": w_garch, "RandomForest": w_rf, "TrendScore": w_trend}
ens_now = weighted_ensemble(results_now, weights_cfg)  # usa os dicts “now”
p_up_now = float(ens_now.get("prob_up", 0.5))
signal_now = "BUY" if p_up_now >= th_buy else ("SELL" if p_up_now <= th_sell else "NEUTRAL")

last_close = float(df["Close"].iloc[-1])
last_atr = float(df["ATR14"].iloc[-1])

entry = stop = gain = None
qty = 0
kelly_f = 0.0
if signal_now != "NEUTRAL":
    entry, stop, gain = entry_stop_gain(last_close, last_atr, signal_now, atr_mult_stop, rr)
    qty = position_size(capital, entry, stop, risk_perc)
    kelly_f = kelly_fraction(p_up_now if signal_now == "BUY" else (1 - p_up_now), rr)

# KPIs
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='card'><div class='sub'>Prob. de Alta</div><div class='kpi'>{p_up_now:.1%}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='card'><div class='sub'>Sinal</div><div class='kpi'>{signal_now}</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='card'><div class='sub'>Último Close</div><div class='kpi'>{last_close:,.2f}</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='card'><div class='sub'>ATR(14)</div><div class='kpi'>{last_atr:,.2f}</div></div>", unsafe_allow_html=True)

if signal_now != "NEUTRAL":
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"<div class='card'><div class='sub'>Entrada</div><div class='kpi'>{entry:,.2f}</div></div>", unsafe_allow_html=True)
    with d2:
        st.markdown(f"<div class='card'><div class='sub'>Stop</div><div class='kpi'>{stop:,.2f}</div></div>", unsafe_allow_html=True)
    with d3:
        st.markdown(f"<div class='card'><div class='sub'>Alvo (R)</div><div class='kpi'>{gain:,.2f}</div></div>", unsafe_allow_html=True)
    with d4:
        st.markdown(f"<div class='card'><div class='sub'>Qtd. sugerida</div><div class='kpi'>{qty:,}</div></div>", unsafe_allow_html=True)
    st.caption(f"Kelly fracionado sugerido: **{kelly_f:.2%}** (use fração conservadora, ex. 0.5×).")


# [BLOCO 6] — ABAS
tab_models, tab_bt, tab_trades, tab_probs = st.tabs(
    ["🧠 Modelos", "🔁 Backtest", "🧾 Trades", "📈 Probabilidades"]
)

# 6.1 — MODELOS (cards bonitos)
with tab_models:
    st.subheader("Explicação dos Modelos — Cards Modernos")

    def render_card(name: str, payload: dict, color: str) -> None:
        st.markdown(
            f"""
            <div class='card' style='border-color:{color};'>
              <h4 style='color:{color}'>🧠 {name}</h4>
              <pre class='pretty'>{json.dumps(payload, indent=2, ensure_ascii=False)}</pre>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for name, res in explain:
        if name == "ARIMA":
            render_card("ARIMA", res, "#7BD389")

        elif name == "GARCH":
            render_card("GARCH", res, "#FFB347")

        elif name == "RandomForest":
            body = {k: v for k, v in res.items() if k not in ("report", "feature_importances")}
            render_card("RandomForest", body, "#8FA3FF")

            rep = res.get("report") or {}
            if isinstance(rep, dict) and rep:
                rep_df = pd.DataFrame(rep).T
                st.dataframe(rep_df.style.format("{:.4f}"))

            fi = res.get("feature_importances") or {}
            if isinstance(fi, dict) and fi:
                st.bar_chart(pd.Series(fi, name="Importância"))

        elif name == "Trend":
            body = {k: v for k, v in res.items() if k != "coefficients"}
            render_card("TrendScore", body, "#F06292")

            coefs = res.get("coefficients") or {}
            if isinstance(coefs, dict) and coefs:
                st.bar_chart(pd.Series(coefs, name="Coeficientes"))


# 6.2 — BACKTEST (gera séries de prob. SEMPRE como Series)
with tab_bt:
    st.subheader("Backtest — Período de Teste")

    probs_list: List[pd.Series] = []
    cols: List[str] = []

    if use_arima and "ARIMA" in models:
        probs_list.append(ps_arima_series(models["ARIMA"], test)); cols.append("ARIMA")
    if use_rf and "RandomForest" in models:
        probs_list.append(ps_rf_series(models["RandomForest"], train, test)); cols.append("RandomForest")
    if use_trend and "TrendScore" in models:
        probs_list.append(ps_trend_series(models["TrendScore"], train, test)); cols.append("TrendScore")
    if use_garch and "GARCH" in models:
        # direção neutra → 0.5 ajuda a ponderar risco
        probs_list.append(_to_series(0.5, test.index, "GARCH")); cols.append("GARCH")

    if not probs_list:
        probs_list = [_to_series(0.5, test.index, "Fallback")]
        cols = ["Fallback"]

    probs_df = pd.concat(probs_list, axis=1)
    probs_df.columns = cols

    w_cfg = {"ARIMA": w_arima, "GARCH": w_garch, "RandomForest": w_rf, "TrendScore": w_trend}
    present = [c for c in probs_df.columns if w_cfg.get(c, 0) > 0] or probs_df.columns.tolist()
    weights = pd.Series({c: w_cfg.get(c, 0.0) for c in present})
    weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)

    prob_series = (probs_df[present] * weights).sum(axis=1).clip(0, 1)

    equity_df, stats, blotter = simulate_prob_strategy(
        test,
        prob_series,
        threshold_buy=th_buy,
        threshold_sell=th_sell,
        capital0=capital,
        risk_perc=risk_perc,
        atr_mult_stop=atr_mult_stop,
        rr=rr,
    )

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(line_series(equity_df, "Equity", title="Equity Curve"), use_container_width=True)
    with g2:
        st.dataframe(equity_df.tail(12))

    st.markdown(
        "<div class='card'><h4>Indicadores do Backtest</h4>"
        + f"<pre class='pretty'>{json.dumps(stats, indent=2, ensure_ascii=False)}</pre></div>",
        unsafe_allow_html=True,
    )

    # guarda p/ outras abas
    st.session_state["prob_series"] = prob_series
    st.session_state["probs_df"] = probs_df
    st.session_state["blotter"] = blotter


# 6.3 — TRADES
with tab_trades:
    st.subheader("Blotter de Trades")
    blotter_df = st.session_state.get("blotter", pd.DataFrame())
    if blotter_df is not None and len(blotter_df) > 0:
        st.dataframe(blotter_df, use_container_width=True, height=420)
        st.download_button(
            "Baixar trades (CSV)",
            blotter_df.to_csv(index=False).encode("utf-8"),
            file_name="trades.csv",
            mime="text/csv",
        )
    else:
        st.info("Sem trades para o período/configuração atual.")


# 6.4 — PROBABILIDADES
with tab_probs:
    st.subheader("Série de Probabilidades")
    prob_series = st.session_state.get("prob_series", None)
    probs_df = st.session_state.get("probs_df", None)
    if isinstance(prob_series, pd.Series):
        st.line_chart(prob_series.rename("prob_up (ensemble)"))
    if isinstance(probs_df, pd.DataFrame):
        st.dataframe(probs_df.tail(12))


# [BLOCO 7] — RODAPÉ
st.caption(
    "Protótipo educacional. Para produção: custos/slippage, walk-forward, tuning e validação rigorosa."
)
