# app.py — Quant App (UI moderna com cards)

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

# =============================== Aparência global ===============================
st.set_page_config(page_title="Quant App — Ensemble & Risco", layout="wide")

CARD_BG = "#0f1116"
CARD_BD = "#2a2f3a"

st.markdown(
    f"""
    <style>
    .card {{
      background: {CARD_BG};
      border: 1px solid {CARD_BD};
      border-radius: 16px;
      padding: 16px;
      margin: 8px 0;
    }}
    .card h4 {{ margin: 0 0 10px 0; font-weight: 700; }}
    .kpi  {{ font-size: 28px; font-weight: 700; }}
    .sub  {{ opacity: .8; font-size: 12px; }}
    pre.pretty {{ background:#131622; border-radius:12px; padding:12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)



# =============================== Sidebar ===============================
with st.sidebar:
    st.header("⚙️ Configurações")
    with open("assets/tickers.json", "r", encoding="utf-8") as f:
        tickers = json.load(f)

    c1, c2 = st.columns(2)
    with c1:
        universo = st.selectbox("Universo", list(tickers.keys()), index=0)
    with c2:
        ticker = st.selectbox("Ativo", tickers[universo], index=0)

    today = date.today()
    d_start = st.date_input("Data inicial", value=today - timedelta(days=365 * 5))
    d_end = st.date_input("Data final", value=today)
    interval = st.selectbox("Intervalo", ["1d", "1h", "1wk"], index=0)

    st.subheader("Treino vs. Teste")
    train_end_ratio = st.slider("Proporção de Treino", 0.5, 0.9, 0.7, 0.05)

    st.subheader("Modelos do Ensemble")
    use_arima = st.checkbox("ARIMA (time-series)", True)
    use_garch = st.checkbox("GARCH (volatilidade)", True, help="Direção neutra; ajuda no risco.")
    use_rf = st.checkbox("RandomForest (ML)", True)
    use_trend = st.checkbox("Trend Score (Logit)", True)

    st.caption("Pesos definem a influência de cada modelo no ensemble.")
    w_arima = st.slider("Peso ARIMA", 0.0, 1.0, 0.30, 0.05)
    w_garch = st.slider("Peso GARCH", 0.0, 1.0, 0.10, 0.05)
    w_rf = st.slider("Peso RF", 0.0, 1.0, 0.40, 0.05)
    w_trend = st.slider("Peso Trend", 0.0, 1.0, 0.20, 0.05)

    st.subheader("Gestão de Risco")
    capital = st.number_input("Capital (BRL/USD)", 100000.0, step=1000.0)
    risk_perc = st.slider("Risco por trade (%)", 0.0025, 0.05, 0.01, 0.0025)
    atr_mult_stop = st.slider("Stop (x ATR14)", 1.0, 5.0, 2.0, 0.5)
    rr = st.slider("Risco:Retorno (R)", 1.0, 5.0, 2.0, 0.5)
    th_buy = st.slider("Threshold BUY", 0.50, 0.90, 0.55, 0.01)
    th_sell = st.slider("Threshold SELL", 0.10, 0.50, 0.45, 0.01)
    st.caption("Dica: 0.52/0.48 cria mais trades; 0.60/0.40 é mais conservador.")


# =============================== Dados & Preço ===============================
st.title("📈 Quant App — Ensemble de Métodos com Gestão de Risco")
st.caption(
    "Combine modelos para gerar probabilidade de alta, sinal e plano de trade. "
    "Faça backtest no período de teste."
)

with st.spinner("Baixando dados e calculando features..."):
    px = download_prices(ticker, str(d_start), str(d_end), interval=interval)
    if len(px) < 250:
        st.warning("Poucos dados retornados — amplie o período ou mude o intervalo.")
    df = add_features(px)

split_idx = max(10, int(len(df) * train_end_ratio))
train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

st.subheader("Preço")
st.plotly_chart(price_candles(df), use_container_width=True)


# =============================== Modelos (ponto atual) ===============================
results: List[Dict] = []
explain: List = []

if use_arima:
    arima = ARIMAModel(order=(1, 0, 0)).fit(train)
    res_a = arima.predict_next(df)
    results.append(res_a)
    explain.append(("ARIMA", res_a))

if use_garch:
    garch = GARCHModel().fit(train)
    res_g = garch.predict_next(df)
    results.append(res_g)
    explain.append(("GARCH", res_g))

if use_rf:
    rf = RandomForestSignal().fit(train, test)
    res_rf = rf.predict_next(df)
    results.append(res_rf)
    explain.append(("RandomForest", res_rf))

if use_trend:
    trend = TrendScoreModel().fit(train, test)
    res_t = trend.predict_next(df)
    results.append(res_t)
    explain.append(("Trend", res_t))

weights = {"ARIMA": w_arima, "GARCH": w_garch, "RandomForest": w_rf, "TrendScore": w_trend}
ens_now = weighted_ensemble(results, weights)
p_up_now = ens_now["prob_up"]
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

# KPI Cards
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"<div class='card'><div class='sub'>Prob. de Alta</div>"
        f"<div class='kpi'>{p_up_now:.1%}</div></div>",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"<div class='card'><div class='sub'>Sinal</div>"
        f"<div class='kpi'>{signal_now}</div></div>",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"<div class='card'><div class='sub'>Último Close</div>"
        f"<div class='kpi'>{last_close:,.2f}</div></div>",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"<div class='card'><div class='sub'>ATR(14)</div>"
        f"<div class='kpi'>{last_atr:,.2f}</div></div>",
        unsafe_allow_html=True,
    )

if signal_now != "NEUTRAL":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='card'><div class='sub'>Entrada</div>"
            f"<div class='kpi'>{entry:,.2f}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='card'><div class='sub'>Stop</div>"
            f"<div class='kpi'>{stop:,.2f}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='card'><div class='sub'>Alvo (R)</div>"
            f"<div class='kpi'>{gain:,.2f}</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='card'><div class='sub'>Qtd. sugerida</div>"
            f"<div class='kpi'>{qty:,}</div></div>",
            unsafe_allow_html=True,
        )
    st.caption(f"Kelly fracionado sugerido: **{kelly_f:.2%}** (use fração conservadora, ex. 0.5×).")


# =============================== Helpers p/ probs por barra ===============================
def _to_series(x, index, name=None) -> pd.Series:
    """Converte array/escalar para Series alinhada ao índice."""
    if isinstance(x, pd.Series):
        return x.reindex(index).astype(float)
    arr = np.asarray(x).reshape(-1)
    if arr.size == 1:
        return pd.Series([float(arr[0])] * len(index), index=index, name=name)
    if arr.size != len(index):
        s = pd.Series(arr, index=index[: arr.size], name=name)
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


# =============================== Abas ===============================
mod_tab, bt_tab, tr_tab, pr_tab = st.tabs(
    ["🧠 Modelos", "🔁 Backtest", "🧾 Trades", "📈 Probabilidades"]
)

# ---------- MODELOS ----------
with mod_tab:
    st.subheader("Explicação dos Modelos — Cards Modernos")

    def render_model_card(name: str, res: dict, color: str = "#7BD389") -> None:
        st.markdown(
            f"""
            <div class='card' style='border-color:{color};'>
              <h4 style='color:{color}'>🧠 {name}</h4>
              <pre class='pretty'>{json.dumps(res, indent=2, ensure_ascii=False)}</pre>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for name, res in explain:
        if name == "ARIMA":
            render_model_card("ARIMA", res, "#7BD389")

        elif name == "GARCH":
            render_model_card("GARCH", res, "#FFB347")

        elif name == "RandomForest":
            # Card com resumo (sem tabelas grandes dentro do JSON)
            body = {k: v for k, v in res.items() if k not in ("report", "feature_importances")}
            render_model_card("RandomForest", body, "#8FA3FF")

            # Tabela do classification_report (se existir)
            rep = res.get("report") or {}
            if isinstance(rep, dict) and rep:
                rep_df = pd.DataFrame(rep).T
                st.dataframe(rep_df.style.format("{:.4f}"))

            # Importância das features
            fi = res.get("feature_importances") or {}
            if isinstance(fi, dict) and fi:
                st.bar_chart(pd.Series(fi, name="Importância"))
                st.caption("Importância das features no modelo RandomForest.")

            # Métricas extras no teste (ROC AUC e Matriz de Confusão)
            try:
                if "rf" in locals():
                    Xte, yte, _ = rf._build_Xy(test)
                    if len(Xte) > 0:
                        from sklearn.metrics import roc_auc_score, confusion_matrix

                        Xte_s = rf.scaler.transform(Xte)
                        yprob = rf.model.predict_proba(Xte_s)[:, 1]
                        yhat = (yprob >= 0.5).astype(int)
                        auc = roc_auc_score(yte, yprob)

                        st.metric("ROC AUC (teste)", f"{auc:.3f}")

                        cm = pd.DataFrame(
                            confusion_matrix(yte, yhat),
                            index=["Actual 0", "Actual 1"],
                            columns=["Pred 0", "Pred 1"],
                        )
                        st.dataframe(cm)
            except Exception as e:
                st.caption(f"Obs.: não foi possível calcular métricas extras ({e}).")

        elif name == "Trend":
            body = {k: v for k, v in res.items() if k != "coefficients"}
            render_model_card("TrendScore", body, "#F06292")

            coefs = res.get("coefficients") or {}
            if isinstance(coefs, dict) and coefs:
                st.bar_chart(pd.Series(coefs, name="Coeficientes"))

# ---------- BACKTEST ----------
with bt_tab:
    st.subheader("Backtest — Período de Teste")

    probs: List[pd.Series] = []
    cols: List[str] = []

    if use_arima:
        probs.append(ps_arima(arima, test))
        cols.append("ARIMA")
    if use_rf:
        probs.append(ps_rf(rf, train, test))
        cols.append("RandomForest")
    if use_trend:
        probs.append(ps_trend(trend, train, test))
        cols.append("TrendScore")
    if use_garch:
        probs.append(_to_series(0.5, test.index, "GARCH"))
        cols.append("GARCH")
    if not probs:
        probs = [_to_series(0.5, test.index, "Fallback")]
        cols = ["Fallback"]

    probs_df = pd.concat(probs, axis=1)
    probs_df.columns = cols

    w_cfg = {"ARIMA": w_arima, "GARCH": w_garch, "RandomForest": w_rf, "TrendScore": w_trend}
    present = [c for c in probs_df.columns if w_cfg.get(c, 0) > 0] or probs_df.columns.tolist()
    weights = pd.Series({c: w_cfg.get(c, 0.0) for c in present})
    weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)

    prob_series = (probs_df[present] * weights).sum(axis=1).clip(0, 1)

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

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(line_series(eq, "Equity", title="Equity Curve"), use_container_width=True)
    with g2:
        st.dataframe(eq.tail(12))

    st.markdown(
        "<div class='card'><h4>Indicadores do Backtest</h4>"
        + f"<pre class='pretty'>{json.dumps(stats, indent=2, ensure_ascii=False)}</pre></div>",
        unsafe_allow_html=True,
    )

    # guarda objetos p/ outras abas
    st.session_state["prob_series"] = prob_series
    st.session_state["probs_df"] = probs_df
    st.session_state["blotter"] = blotter

# ---------- TRADES ----------
with tr_tab:
    st.subheader("Blotter de Trades")
    blotter = st.session_state.get("blotter", pd.DataFrame())
    if blotter is not None and len(blotter) > 0:
        st.dataframe(blotter, use_container_width=True, height=420)
        st.download_button(
            "Baixar trades (CSV)",
            blotter.to_csv(index=False).encode("utf-8"),
            file_name="trades.csv",
            mime="text/csv",
        )
    else:
        st.info("Sem trades no período/configuração atual. Ajuste thresholds/pesos para gerar sinais.")

# ---------- PROBABILIDADES ----------
with pr_tab:
    st.subheader("Série de Probabilidades")
    prob_series = st.session_state.get("prob_series", None)
    probs_df = st.session_state.get("probs_df", None)
    if isinstance(prob_series, pd.Series):
        st.line_chart(prob_series.rename("prob_up (ensemble)"))
    if isinstance(probs_df, pd.DataFrame):
        st.dataframe(probs_df.tail(12))

st.caption(
    "Protótipo educacional. Para produção: custos/deslizamento, walk-forward, tuning e validação rigorosa."
)
