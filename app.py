# =============================================================================
# app.py — Quant App (single-file) • Ensemble + Gestão de Risco + Backtest
# =============================================================================
from __future__ import annotations

# ------------------------- [BLOCO 0] Imports “flat” --------------------------
import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

# Modelagem
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from statsmodels.tsa.arima.model import ARIMA

# Dados
import yfinance as yf

# Opcional: GARCH via arch (app segue funcionando se faltar)
try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False


# ------------------------- [BLOCO 1] Aparência/UI ----------------------------
st.set_page_config(page_title="Quant App — Ensemble & Risco (single-file)", layout="wide")
st.markdown("""
<style>
  :root {
    --bg:#0f1116; --bd:#222733; --muted:#9aa4b2;
    --ok:#7BD389; --warn:#FFB347; --info:#8FA3FF; --trend:#F06292;
  }
  .card {background:var(--bg); border:1px solid var(--bd); border-radius:16px; padding:16px; margin:10px 0}
  .kpi-title{font-size:12px; color:var(--muted); margin-bottom:6px}
  .kpi-value{font-size:28px; font-weight:800}
  .pill{display:inline-block; padding:2px 10px; border:1px solid var(--bd); border-radius:999px; font-size:12px; color:var(--muted); margin-right:6px}
  pre.pretty{background:#121622; border:1px solid var(--bd); border-radius:12px; padding:12px; font-size:12px}
</style>
""", unsafe_allow_html=True)

def metric_card(title:str, value:str, sub:str="", color:str=None):
    c = "" if not color else f"style='color:{color}'"
    st.markdown(
        f"<div class='card'>"
        f"<div class='kpi-title'>{title}</div>"
        f"<div class='kpi-value' {c}>{value}</div>"
        f"<div style='color:var(--muted);font-size:12px;margin-top:8px'>{sub}</div>"
        f"</div>", unsafe_allow_html=True
    )

def section_card(title:str, color:str, body_html:str):
    st.markdown(
        f"<div class='card' style='border-color:{color}'>"
        f"<h4 style='margin:0 0 10px 0; color:{color}'>🧠 {title}</h4>"
        f"{body_html}"
        f"</div>", unsafe_allow_html=True
    )


# ------------------------- [BLOCO 2] Utilidades de Dados/Gráficos ------------

def download_prices(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Baixa preços do Yahoo e GARANTE colunas planas:
    Open, High, Low, Close, Volume (e Adj Close se existir).
    Lida com MultiIndex (('Open','AAPL'), ...) e nomes estranhos.
    """
    df = yf.download(
        ticker, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False
    )

    # índice de datas consistente
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # MultiIndex → achata
    if isinstance(df.columns, pd.MultiIndex):
        tickers_level = df.columns.get_level_values(-1)
        if ticker in tickers_level:
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)

    # Normaliza nomes
    cols_map = {c: str(c).title() for c in df.columns}
    df = df.rename(columns=cols_map)

    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(how="any")


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    ret = close.diff()
    up = ret.clip(lower=0.0)
    down = -ret.clip(upper=0.0)
    ma_up = up.rolling(n).mean()
    ma_down = down.rolling(n).mean().replace(0, 1e-9)
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enriquece o preço com indicadores e sinais, garantindo alinhamento e tipos."""
    out = df.copy()

    for c in ("Open", "High", "Low", "Close"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["Ret"]   = out["Close"].pct_change()
    out["EMA20"] = _ema(out["Close"], 20)
    out["EMA50"] = _ema(out["Close"], 50)
    out["EMA200"]= _ema(out["Close"], 200)
    out["RSI14"] = _rsi(out["Close"], 14)
    out["ATR14"] = _atr(out, 14)
    out["Vol20"] = out["Ret"].rolling(20).std()
    out["MACD"]  = _ema(out["Close"], 12) - _ema(out["Close"], 26)

    def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        r = (a.astype(float) / b.replace(0, np.nan).astype(float) - 1.0)
        return r.reindex(out.index).astype(float)

    out["Close_over_EMA50"]  = safe_ratio(out["Close"], out["EMA50"]).fillna(0.0)
    out["EMA20_over_EMA50"]  = safe_ratio(out["EMA20"], out["EMA50"]).fillna(0.0)
    out["EMA50_over_EMA200"] = safe_ratio(out["EMA50"], out["EMA200"]).fillna(0.0)

    need = [
        "Close","EMA20","EMA50","EMA200","RSI14","ATR14","Vol20","MACD",
        "Close_over_EMA50","EMA20_over_EMA50","EMA50_over_EMA200"
    ]
    need = [c for c in need if c in out.columns]
    out = out.dropna(subset=need)
    out["Ret"] = out["Ret"].fillna(0.0)

    return out

def price_chart(df: pd.DataFrame, title: str = "Preço"):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Preço"
    )
    fig.update_layout(height=420, title=title, template="plotly_dark", margin=dict(l=0,r=0,t=40,b=0))
    return fig

def line_chart(s: pd.Series | pd.DataFrame, title: str = ""):
    import plotly.express as px
    df = s if isinstance(s, pd.DataFrame) else s.to_frame("value")
    fig = px.line(df, x=df.index, y=df.columns)
    fig.update_layout(height=340, title=title, template="plotly_dark", margin=dict(l=0,r=0,t=40,b=0))
    return fig

def to_series(x, index: pd.Index, name: str) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reindex(index).astype(float)
    arr = np.asarray(x).reshape(-1)
    if arr.size == 1:
        return pd.Series([float(arr[0])] * len(index), index=index, name=name)
    if arr.size != len(index):
        s = pd.Series(arr, index=index[:arr.size], name=name).astype(float)
        return s.reindex(index).ffill().bfill()
    return pd.Series(arr, index=index, name=name).astype(float)


# ------------------------- [BLOCO 3] Modelos ---------------------------------
@dataclass
class ArimaOut:
    mu: float
    sigma: float
    prob_up: float
    order: Tuple[int,int,int]

def arima_now_and_series(returns_train: pd.Series, test_len: int, index_future: pd.Index) -> Tuple[ArimaOut, pd.Series]:
    model = ARIMA(returns_train.values, order=(1,0,0)).fit(method_kwargs={"warn_convergence":False})
    f1 = model.get_forecast(steps=1)
    mu = float(f1.predicted_mean[0])
    sigma = float(np.sqrt(f1.var_pred_mean[0]))
    prob_up_now = 1.0 - norm.cdf(0.0, loc=mu, scale=sigma)
    out = ArimaOut(mu=mu, sigma=sigma, prob_up=prob_up_now, order=(1,0,0))

    f = model.get_forecast(steps=test_len)
    mu_s = pd.Series(f.predicted_mean, index=index_future)
    sigma_s = pd.Series(np.sqrt(f.var_pred_mean), index=index_future).replace(0, 1e-6)
    prob_s = 1.0 - norm.cdf(0.0, loc=mu_s, scale=sigma_s)
    return out, to_series(prob_s, index_future, "ARIMA")


@dataclass
class GarchOut:
    sigma_next: float
    prob_up: float

def garch_vol_series(returns_train: pd.Series, test_len: int, index_future: pd.Index) -> Tuple[GarchOut, pd.Series]:
    """Se houver 'arch', usa GARCH(1,1) pra vol; direção neutra (0.5). Senão, usa rolling std."""
    if HAVE_ARCH:
        am = arch_model(returns_train*100, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
        res = am.fit(disp="off")
        f = res.forecast(horizon=test_len)
        sigma = (f.variance.values[-1, :] ** 0.5) / 100.0
        sigma_next = float(sigma[0])
    else:
        vol = returns_train.rolling(20).std().fillna(returns_train.std())
        sigma_next = float(vol.iloc[-1])
    out = GarchOut(sigma_next=sigma_next, prob_up=0.5)
    prob_neutral = to_series(0.5, index_future, "GARCH")
    return out, prob_neutral


@dataclass
class RFOut:
    prob_up: float
    report: Dict
    feature_importances: Dict[str, float]

def random_forest(train: pd.DataFrame, test_index: pd.Index) -> Tuple[RFOut, pd.Series, StandardScaler, CalibratedClassifierCV]:
    feats = ["EMA20","EMA50","EMA200","RSI14","ATR14","Vol20","MACD",
             "Close_over_EMA50","EMA20_over_EMA50","EMA50_over_EMA200"]
    X = train[feats].fillna(method="ffill").dropna()
    y = (train["Ret"].reindex(X.index) > 0).astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    base = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")
    model.fit(Xs, y)

    p_now = float(model.predict_proba(scaler.transform(X.iloc[[-1]]))[:,1][0])

    rep = classification_report(y, model.predict(Xs), output_dict=True, zero_division=0)
    fi = dict(zip(feats, base.fit(Xs, y).feature_importances_.round(4)))

    Xf = train.reindex(test_index).dropna()[feats]
    if len(Xf) == 0:
        prob_s = to_series(0.5, test_index, "RandomForest")
    else:
        prob_s = pd.Series(model.predict_proba(scaler.transform(Xf))[:,1], index=Xf.index, name="RandomForest")
        prob_s = to_series(prob_s, test_index, "RandomForest")

    out = RFOut(prob_up=p_now, report=rep, feature_importances=fi)
    return out, prob_s, scaler, model


@dataclass
class TrendOut:
    prob_up: float
    intercept: float
    coefficients: Dict[str, float]

def trend_score(train: pd.DataFrame, test_index: pd.Index) -> Tuple[TrendOut, pd.Series, StandardScaler, LogisticRegression]:
    feats = ["Close_over_EMA50","EMA20_over_EMA50","EMA50_over_EMA200","MACD","RSI14"]
    X = train[feats].fillna(method="ffill").dropna()
    y = (train["Ret"].reindex(X.index) > 0).astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xs, y)

    p_now = float(clf.predict_proba(scaler.transform(X.iloc[[-1]]))[:,1][0])
    coefs = dict(zip(feats, clf.coef_[0].round(4)))

    Xf = train.reindex(test_index).dropna()[feats]
    if len(Xf) == 0:
        prob_s = to_series(0.5, test_index, "TrendScore")
    else:
        prob_s = pd.Series(clf.predict_proba(scaler.transform(Xf))[:,1], index=Xf.index, name="TrendScore")
        prob_s = to_series(prob_s, test_index, "TrendScore")

    out = TrendOut(prob_up=p_now, intercept=float(clf.intercept_[0]), coefficients=coefs)
    return out, prob_s, scaler, clf


# ------------------------- [BLOCO 4] Ensemble & Risco ------------------------
def weighted_prob(probs: Dict[str, float], weights: Dict[str, float]) -> float:
    ws = {k: weights.get(k, 0.0) for k in probs.keys()}
    s = sum(ws.values())
    if s == 0:
        return float(np.mean(list(probs.values())))
    return float(sum(probs[k]*ws[k] for k in probs)/s)

def entry_stop_gain(last_close: float, atr: float, side: str, atr_mult: float, rr: float) -> Tuple[float,float,float]:
    if side == "BUY":
        entry = last_close
        stop  = last_close - atr_mult*atr
        gain  = last_close + rr*(last_close - stop)
    else:
        entry = last_close
        stop  = last_close + atr_mult*atr
        gain  = last_close - rr*(stop - last_close)
    return entry, stop, gain

def position_size(capital: float, entry: float, stop: float, risk_perc: float) -> int:
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0
    risk_total = capital * risk_perc
    return int(max(0, np.floor(risk_total / risk_per_unit)))

def kelly_fraction(p: float, rr: float) -> float:
    q = 1 - p
    edge = (rr*p - q)
    denom = rr
    return max(0.0, min(1.0, edge/denom if denom>0 else 0.0))


# ------------------------- [BLOCO 5] Backtest --------------------------------
def backtest_prob_strategy(df_test: pd.DataFrame,
                           prob_series: pd.Series,
                           th_buy: float, th_sell: float,
                           capital0: float = 100_000.0,
                           risk_perc: float = 0.01,
                           atr_mult_stop: float = 2.0,
                           rr: float = 2.0) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """Backtest simplificado: entra no fechamento quando há sinal; sai por stop/target/flip."""
    equity = []
    blotter = []

    capital = capital0
    pos = 0
    entry_px = stop_px = tgt_px = 0.0

    for dt, p_up in prob_series.items():
        px = float(df_test.at[dt, "Close"])
        atr = float(df_test.at[dt, "ATR14"])
        signal = "BUY" if p_up >= th_buy else ("SELL" if p_up <= th_sell else "FLAT")

        # saída por stop/target/flip
        if pos != 0:
            if pos > 0:
                if px <= stop_px or px >= tgt_px or signal == "SELL":
                    pnl = (px - entry_px) * pos
                    capital += pnl
                    blotter.append({"date": dt, "side":"EXIT-L", "qty": pos, "price": px, "pnl": pnl})
                    pos = 0
            else:
                if px >= stop_px or px <= tgt_px or signal == "BUY":
                    pnl = (entry_px - px) * (-pos)
                    capital += pnl
                    blotter.append({"date": dt, "side":"EXIT-S", "qty": -pos, "price": px, "pnl": pnl})
                    pos = 0

        # entrada
        if pos == 0 and signal in ("BUY","SELL"):
            entry_px, stop_px, tgt_px = entry_stop_gain(px, atr, signal, atr_mult_stop, rr)
            qty = position_size(capital, entry_px, stop_px, risk_perc)
            if qty > 0:
                if signal == "BUY":
                    pos = qty
                    blotter.append({"date": dt, "side":"BUY", "qty": qty, "price": px, "pnl": 0.0})
                else:
                    pos = -qty
                    blotter.append({"date": dt, "side":"SELL", "qty": qty, "price": px, "pnl": 0.0})

        equity.append({"Date": dt, "Equity": capital})

    eq = pd.DataFrame(equity).set_index("Date")
    rets = eq["Equity"].pct_change().fillna(0)
    sharpe = (rets.mean() / (rets.std()+1e-9)) * np.sqrt(252)
    stats = {
        "final_equity": float(eq["Equity"].iloc[-1]),
        "sharpe": float(sharpe),
        "trades": int(len([b for b in blotter if b["side"] in ("BUY","SELL")])),
        "hit_rate": float(pd.Series([x["pnl"] for x in blotter if "EXIT" in x["side"]]).gt(0).mean() if blotter else 0),
        "avg_pnl": float(pd.Series([x["pnl"] for x in blotter if "EXIT" in x["side"]]).mean() if blotter else 0),
        "total_pnl": float(pd.Series([x["pnl"] for x in blotter if "EXIT" in x["side"]]).sum() if blotter else 0),
    }
    blotter_df = pd.DataFrame(blotter)
    return eq, stats, blotter_df


# ------------------------- [BLOCO 6] Sidebar ---------------------------------
with st.sidebar:
    st.header("⚙️ Configurações")

    try:
        with open("assets/tickers.json", "r", encoding="utf-8") as f:
            TICKERS = json.load(f)
    except Exception:
        TICKERS = {"US": ["AAPL", "MSFT", "GOOG", "SPY"],
                   "BR": ["PETR4.SA","VALE3.SA","BOVA11.SA"]}

    c1, c2 = st.columns(2)
    with c1: universo = st.selectbox("Universo", list(TICKERS.keys()), index=0)
    with c2: ticker   = st.selectbox("Ativo", TICKERS[universo], index=0)

    today = date.today()
    d_start = st.date_input("Data inicial", value=today - timedelta(days=365*5))
    d_end   = st.date_input("Data final", value=today)
    interval = st.selectbox("Intervalo", ["1d","1h","1wk"], index=0)

    st.subheader("Treino vs. Teste")
    train_ratio = st.slider("Proporção Treino", 0.5, 0.9, 0.7, 0.05)

    st.subheader("Modelos do Ensemble")
    use_arima = st.checkbox("ARIMA", True)
    use_garch = st.checkbox("GARCH (vol)", True, help="Direção neutra 0.5; informa volatilidade.")
    use_rf    = st.checkbox("RandomForest", True)
    use_trend = st.checkbox("TrendScore (Logit)", True)

    w_arima = st.slider("Peso ARIMA", 0.0, 1.0, 0.30, 0.05)
    w_garch = st.slider("Peso GARCH", 0.0, 1.0, 0.10, 0.05)
    w_rf    = st.slider("Peso RF", 0.0, 1.0, 0.40, 0.05)
    w_trend = st.slider("Peso Trend", 0.0, 1.0, 0.20, 0.05)

    st.subheader("Risco")
    capital   = st.number_input("Capital", 100000.0, step=1000.0)
    risk_perc = st.slider("Risco por trade (%)", 0.0025, 0.05, 0.01, 0.0025)
    atr_mult  = st.slider("Stop (x ATR14)", 1.0, 5.0, 2.0, 0.5)
    rr        = st.slider("Risco:Retorno (R)", 1.0, 5.0, 2.0, 0.5)
    th_buy    = st.slider("Threshold BUY", 0.50, 0.90, 0.55, 0.01)
    th_sell   = st.slider("Threshold SELL", 0.10, 0.50, 0.45, 0.01)


# ------------------------- [BLOCO 7] Dados & Features ------------------------
st.title("📈 Quant App — Ensemble de Métodos com Gestão de Risco (single-file)")
with st.spinner("Baixando dados..."):
    px = download_prices(ticker, str(d_start), str(d_end), interval)

df = add_features(px)
split = max(10, int(len(df)*train_ratio))
train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

st.plotly_chart(price_chart(df, f"Preço — {ticker}"), use_container_width=True)


# ------------------------- [BLOCO 8] Rodadas de modelos ----------------------
explain_cards: List[Tuple[str, Dict]] = []
probs_now: Dict[str, float] = {}
probs_series: Dict[str, pd.Series] = {}

if use_arima:
    a_out, a_series = arima_now_and_series(train["Ret"], len(test), test.index)
    explain_cards.append(("ARIMA", {"model":"ARIMA","order":list(a_out.order),
                                    "mu":a_out.mu,"sigma":a_out.sigma,"prob_up":a_out.prob_up}))
    probs_now["ARIMA"] = a_out.prob_up
    probs_series["ARIMA"] = a_series

if use_garch:
    g_out, g_series = garch_vol_series(train["Ret"], len(test), test.index)
    explain_cards.append(("GARCH", {"model":"GARCH","sigma_next":g_out.sigma_next,"prob_up":0.5}))
    probs_now["GARCH"] = 0.5
    probs_series["GARCH"] = g_series

if use_rf:
    rf_out, rf_series, rf_scaler, rf_model = random_forest(train, test.index)
    explain_cards.append(("RandomForest", {"model":"RandomForest","prob_up":rf_out.prob_up,
                                           "report":rf_out.report, "feature_importances":rf_out.feature_importances}))
    probs_now["RandomForest"] = rf_out.prob_up
    probs_series["RandomForest"] = rf_series

if use_trend:
    tr_out, tr_series, tr_scaler, tr_clf = trend_score(train, test.index)
    explain_cards.append(("Trend", {"model":"TrendScore","prob_up":tr_out.prob_up,
                                    "intercept":tr_out.intercept,"coefficients":tr_out.coefficients}))
    probs_now["TrendScore"] = tr_out.prob_up
    probs_series["TrendScore"] = tr_series

weights = {"ARIMA": w_arima, "GARCH": w_garch, "RandomForest": w_rf, "TrendScore": w_trend}
p_up_now = weighted_prob(probs_now, weights)
signal_now = "BUY" if p_up_now >= th_buy else ("SELL" if p_up_now <= th_sell else "NEUTRAL")

last_close = float(df["Close"].iloc[-1])
last_atr   = float(df["ATR14"].iloc[-1])
entry = stop = tgt = None
qty = 0
if signal_now != "NEUTRAL":
    entry, stop, tgt = entry_stop_gain(last_close, last_atr, signal_now, atr_mult, rr)
    qty = position_size(capital, entry, stop, risk_perc)
kelly_f = kelly_fraction(p_up_now if signal_now=="BUY" else (1-p_up_now), rr)

# KPIs
k1, k2, k3, k4 = st.columns(4)
with k1: metric_card("Prob. de Alta (Ensemble)", f"{p_up_now:.1%}")
with k2:
    cor = "#7BD389" if signal_now=="BUY" else ("#F06292" if signal_now=="SELL" else None)
    metric_card("Direção", signal_now, color=cor)
with k3: metric_card("Último Close", f"{last_close:,.2f}")
with k4: metric_card("ATR(14)", f"{last_atr:,.2f}")

if signal_now!="NEUTRAL":
    d1,d2,d3,d4 = st.columns(4)
    with d1: metric_card("Entrada", f"{entry:,.2f}")
    with d2: metric_card("Stop", f"{stop:,.2f}")
    with d3: metric_card("Alvo (R)", f"{tgt:,.2f}")
    with d4: metric_card("Qtd. sugerida", f"{qty:,}")
st.markdown(f"<span class='pill'>Kelly: {kelly_f:.2%}</span> "
            f"<span class='pill'>TH Buy: {th_buy:.2f}</span> "
            f"<span class='pill'>TH Sell: {th_sell:.2f}</span>", unsafe_allow_html=True)


# ------------------------- [BLOCO 9] Abas ------------------------------------
tab_models, tab_bt, tab_trades, tab_probs = st.tabs(
    ["🧠 Modelos","🔁 Backtest","🧾 Trades","📈 Probabilidades"]
)

with tab_models:
    st.subheader("Explicação dos Modelos — Cards")

    # ARIMA
    for name, payload in explain_cards:
        if name == "ARIMA":
            body = f"<pre class='pretty'>{json.dumps(payload, indent=2, ensure_ascii=False)}</pre>"
            section_card("ARIMA", "var(--ok)", body)

    # GARCH
    for name, payload in explain_cards:
        if name == "GARCH":
            body = f"<pre class='pretty'>{json.dumps(payload, indent=2, ensure_ascii=False)}</pre>"
            section_card("GARCH", "var(--warn)", body)

    # RandomForest
    for name, payload in explain_cards:
        if name == "RandomForest":
            body = f"<pre class='pretty'>{json.dumps({k:v for k,v in payload.items() if k not in ('report','feature_importances')}, indent=2, ensure_ascii=False)}</pre>"
            section_card("RandomForest", "var(--info)", body)

            rep = payload.get("report", {})
            if isinstance(rep, dict) and rep:
                st.markdown("**Relatório de Classificação (treino/calibração)**")
                rep_df = pd.DataFrame(rep).T
                st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)

            fi = payload.get("feature_importances", {})
            if fi:
                st.markdown("**Importância de Features**")
                st.bar_chart(pd.Series(fi, name="Importância"))

    # TrendScore
    for name, payload in explain_cards:
        if name == "Trend":
            body = f"<pre class='pretty'>{json.dumps({k:v for k,v in payload.items() if k!='coefficients'}, indent=2, ensure_ascii=False)}</pre>"
            section_card("TrendScore (Logit)", "var(--trend)", body)

            coefs = payload.get("coefficients", {})
            if coefs:
                st.markdown("**Coeficientes (sinal e magnitude)**")
                st.bar_chart(pd.Series(coefs, name="Coeficientes"))

with tab_bt:
    st.subheader("Backtest — Período de Teste")

    present = [k for k in probs_series.keys() if weights.get(k,0)>0] or list(probs_series.keys())
    if not present:
        present = ["Fallback"]; probs_series["Fallback"] = to_series(0.5, test.index, "Fallback"); weights["Fallback"] = 1.0

    w_norm = pd.Series({k:weights.get(k,0.0) for k in present})
    w_norm = w_norm / (w_norm.sum() if w_norm.sum()>0 else 1.0)

    probs_df = pd.concat([probs_series[k] for k in present], axis=1)
    probs_df.columns = present
    prob_ens = (probs_df * w_norm).sum(axis=1).clip(0,1)

    eq, stats, blotter = backtest_prob_strategy(
        test, prob_ens, th_buy, th_sell,
        capital0=capital, risk_perc=risk_perc,
        atr_mult_stop=atr_mult, rr=rr
    )

    c1, c2 = st.columns([2,1])
    with c1: st.plotly_chart(line_chart(eq["Equity"], "Equity Curve"), use_container_width=True)
    with c2: st.dataframe(eq.tail(12), use_container_width=True)

    g1, g2, g3, g4, g5 = st.columns(5)
    with g1: metric_card("Equity Final", f"{stats['final_equity']:,.2f}")
    with g2: metric_card("Sharpe", f"{stats['sharpe']:.2f}")
    with g3: metric_card("Trades", str(stats["trades"]))
    with g4: metric_card("Hit Rate", f"{stats['hit_rate']:.1%}")
    with g5: metric_card("P/L Médio", f"{stats['avg_pnl']:,.2f}")

    st.download_button(
        "Baixar Equity (CSV)", eq.to_csv().encode("utf-8"),
        file_name="equity_curve.csv", mime="text/csv"
    )

    st.session_state["prob_series"] = prob_ens
    st.session_state["probs_df"] = probs_df
    st.session_state["blotter"] = blotter

with tab_trades:
    st.subheader("Blotter de Trades")
    bt = st.session_state.get("blotter", pd.DataFrame())
    if isinstance(bt, pd.DataFrame) and len(bt)>0:
        st.dataframe(bt, use_container_width=True, height=420)
        st.download_button("Baixar trades (CSV)", bt.to_csv(index=False).encode("utf-8"),
                           file_name="trades.csv", mime="text/csv")
    else:
        st.info("Sem trades no período.")

with tab_probs:
    st.subheader("Série de Probabilidades")
    ps  = st.session_state.get("prob_series", None)
    pdf = st.session_state.get("probs_df", None)

    if isinstance(ps, pd.Series):
        st.line_chart(ps.rename("prob_up (ensemble)"))
        st.download_button(
            "Baixar Ensemble (CSV)", ps.to_frame("prob_up").to_csv().encode("utf-8"),
            file_name="prob_ensemble.csv", mime="text/csv"
        )
    if isinstance(pdf, pd.DataFrame):
        st.markdown("**Componentes do Ensemble (últimas 20 barras)**")
        st.dataframe(pdf.tail(20), use_container_width=True)
        st.download_button(
            "Baixar Componentes (CSV)", pdf.to_csv().encode("utf-8"),
            file_name="prob_components.csv", mime="text/csv"
        )

# ------------------------- [BLOCO 10] Rodapé ---------------------------------
st.caption("Protótipo educacional (single-file). Para produção: custos/slippage, walk-forward, tuning e validação rigorosa.")
