from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# ZIA RESEARCH TERMINAL — consolidated build
# Merges dashboard_v6 (silent 1s live engine) with the visual
# polish of v3/v5, and adds a new Multi-Market Scanner tab that
# ranks every tracked symbol by live signal strength at once.
# ============================================================

st.set_page_config(page_title="ZIA Research Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "xgboost_obi_model.pkl"
SIGNAL_FILE = ROOT / "saved_signals.csv"
TRADE_FILE = ROOT / "trade_history.csv"

SIGNAL_VALIDITY_MINUTES = {"15M": 30, "1H": 120, "4H": 180}
AUTO_SCAN_TFS = ("15M", "1H", "4H")
AUTO_SCAN_INTERVAL_SECONDS = 10

FUTURES = ["https://fapi.binance.com", "https://fapi1.binance.com", "https://fapi2.binance.com"]
SPOT = ["https://api.binance.com", "https://api1.binance.com"]
DATA = ["https://data-api.binance.vision"]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
           "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "TRXUSDT", "LTCUSDT"]
TFS = {"1MO": "1M", "1W": "1w", "1D": "1d", "4H": "4h", "1H": "1h",
       "30M": "30m", "15M": "15m", "5M": "5m", "3M": "3m", "1M": "1m"}

# ------------------------------------------------------------
# STYLE — dark quant-terminal theme, violet/green/red signal accents
# ------------------------------------------------------------
st.markdown("""
<style>
:root{--bg:#05070b;--panel:#0b1119;--panel2:#101925;--line:#1d2a39;--txt:#edf3fb;--muted:#7f8da1;
--green:#42dda0;--red:#ff7184;--amber:#f3c86a;--cyan:#65d7ff;--violet:#969eff}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--txt)}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1920px;padding:10px clamp(8px,1.5vw,30px) 32px}
.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding:4px 2px 12px;margin-bottom:10px}
.brand{font-size:clamp(25px,2.7vw,42px);font-weight:950;letter-spacing:-2px}.brand b{color:var(--violet)}
.micro{color:var(--muted);font-size:9px;letter-spacing:1.5px}
.live{border:1px solid #245d45;background:#071810;color:#6ce3a5;border-radius:999px;padding:7px 12px;font-size:10px;font-weight:900}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);margin-right:6px}
.panel{background:linear-gradient(145deg,#0e151f,#080d14);border:1px solid var(--line);border-radius:15px;padding:14px;margin-bottom:10px}
.card{background:linear-gradient(145deg,#111a26,#0a1018);border:1px solid var(--line);border-radius:14px;padding:11px;min-height:76px}
.label{font-size:9px;color:var(--muted);font-weight:900;letter-spacing:1.1px}
.value{font-size:19px;font-weight:950;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sub{font-size:10px;color:#8794a8;margin-top:3px}
.good{color:var(--green)}.bad{color:var(--red)}.amber{color:var(--amber)}.violet{color:var(--violet)}.cyan{color:var(--cyan)}
.signalbox{border-radius:18px;padding:16px 20px;border:1px solid var(--line);background:linear-gradient(145deg,#111a26,#080d14);text-align:center;margin-bottom:10px}
.signal-long{border-color:#277b59;box-shadow:0 0 28px rgba(66,221,160,.08)}
.signal-short{border-color:#843a4a;box-shadow:0 0 28px rgba(255,113,132,.08)}
.signal-wait{border-color:#705e30}
.signal-main{font-size:clamp(34px,4vw,58px);font-weight:1000;letter-spacing:-2px;line-height:1}
.signal-meta{font-size:10px;color:var(--muted);margin-top:7px;letter-spacing:1px}
.signal-timer{display:inline-block;margin-top:9px;padding:6px 12px;border:1px solid #2b4054;border-radius:999px;background:#071019;color:#65d7ff;font-size:11px;font-weight:950;letter-spacing:.8px}
.signal-timer.live{border-color:#277b59;background:#071810;color:#42dda0;box-shadow:0 0 16px rgba(66,221,160,.08)}
.signal-timer.expired{border-color:#705e30;color:#f3c86a}
.tri-strip{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 9px}
.tri-chip{border:1px solid var(--line);background:#0a1119;border-radius:9px;padding:6px 9px;font-size:9px;font-weight:900;letter-spacing:.5px}
.tri-chip span{color:var(--cyan)}
.section-title{font-size:16px;font-weight:950;letter-spacing:-.2px;margin:12px 0 6px}
.section-sub{font-size:10px;color:var(--muted);margin-bottom:9px}
.scan-row{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:9px 4px;font-size:12.5px}
.scan-row:last-child{border-bottom:none}
.pill{border-radius:999px;padding:3px 10px;font-size:10px;font-weight:900;letter-spacing:.4px}
.pill-long{background:#0c2018;color:var(--green);border:1px solid #1f5c42}
.pill-short{background:#26101a;color:var(--red);border:1px solid #6e2c3b}
.pill-wait{background:#22190a;color:var(--amber);border:1px solid #6b551f}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:900}
div[data-testid="stTabs"] button{font-weight:900;font-size:11px}
[data-testid="stSelectbox"] label{font-size:9px;font-weight:900;letter-spacing:1px;color:var(--muted)}
@media(max-width:700px){
  .block-container{padding:6px 7px 22px}.brand{font-size:21px}.micro{font-size:7px}
  .live{font-size:8px;padding:5px 8px}.panel{padding:9px;border-radius:12px}
  .card{min-height:62px;padding:9px}.value{font-size:16px}
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# DATA HELPERS
# ------------------------------------------------------------

def num(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def api(hosts, path, params):
    err = "network"
    for host in hosts:
        try:
            r = requests.get(host + path, params=params, timeout=2.2, headers={"User-Agent": "ZIA-Research"})
            if r.ok:
                return r.json(), host, "OK"
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = type(e).__name__
    return None, None, err


@st.cache_data(ttl=1.0, show_spinner=False)
def candles(symbol, interval, limit=650):
    raw, host, status = api(FUTURES, "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)})
    source = "Futures"
    if not isinstance(raw, list):
        raw, host, status = api(SPOT, "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)})
        source = "Spot"
    if not isinstance(raw, list):
        raw, host, status = api(DATA, "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)})
        source = "Data API"
    rows = []
    for c in raw or []:
        try:
            rows.append([pd.to_datetime(int(c[0]), unit="ms", utc=True), num(c[1]), num(c[2]), num(c[3]), num(c[4]), num(c[5]), num(c[9])])
        except Exception:
            pass
    return pd.DataFrame(rows, columns=["Time", "Open", "High", "Low", "Close", "Volume", "TakerBuy"]), source, status, host


@st.cache_data(ttl=1.0, show_spinner=False)
def orderbook(symbol):
    raw, host, status = api(FUTURES, "/fapi/v1/depth", {"symbol": symbol, "limit": 100})
    source = "Futures"
    if not isinstance(raw, dict) or not raw.get("bids"):
        raw, host, status = api(SPOT, "/api/v3/depth", {"symbol": symbol, "limit": 100})
        source = "Spot"
    if not isinstance(raw, dict) or not raw.get("bids"):
        raw, host, status = api(DATA, "/api/v3/depth", {"symbol": symbol, "limit": 100})
        source = "Data API"
    try:
        return np.asarray(raw.get("bids", []), float), np.asarray(raw.get("asks", []), float), source, status, host
    except Exception:
        return np.empty((0, 2)), np.empty((0, 2)), source, status, host


@st.cache_data(ttl=0.8, show_spinner=False)
def live_prices():
    raw, _, _ = api(FUTURES, "/fapi/v1/ticker/price", {})
    if not isinstance(raw, list):
        raw, _, _ = api(SPOT, "/api/v3/ticker/price", {})
    if not isinstance(raw, list):
        return {}
    return {str(x.get("symbol")): num(x.get("price")) for x in raw if isinstance(x, dict) and x.get("symbol")}


def obi(bids, asks, k):
    if len(bids) == 0 or len(asks) == 0:
        return 0., 0., 0.
    k = min(k, len(bids), len(asks))
    bv = float(bids[:k, 1].sum())
    av = float(asks[:k, 1].sum())
    return ((bv - av) / (bv + av) if bv + av else 0.), bv, av


def features(df, b, a):
    f = {k: 0. for k in ["top20_bid_sum", "top20_ask_sum", "obi_5", "obi_10", "obi_20", "obi_50", "spread",
                          "spread_pct", "bid_ask_ratio_20", "bid_ask_ratio_50", "top20_total_depth",
                          "top50_total_depth", "taker_buy_volume", "taker_sell_volume", "taker_flow",
                          "taker_flow_ratio", "price_return", "price_change", "sma_distance",
                          "realized_volatility", "BOOK_IMB", "QUANT_IMPLY", "ADAPT_CONF", "BAYESIAN", "FOURIER_TREND"]}
    (o5, b5, a5), (o10, b10, a10), (o20, b20, a20), (o50, b50, a50) = [obi(b, a, k) for k in (5, 10, 20, 50)]
    f.update(top20_bid_sum=b20, top20_ask_sum=a20, obi_5=o5, obi_10=o10, obi_20=o20, obi_50=o50,
              top20_total_depth=b20 + a20, top50_total_depth=b50 + a50)
    if df.empty:
        return f
    c = df.Close
    last = num(c.iloc[-1])
    prev = num(c.iloc[-2] if len(c) > 1 else last)
    sma = num(c.rolling(20).mean().iloc[-1], last)
    total = num(df.Volume.tail(20).sum())
    buy = num(df.TakerBuy.tail(20).sum())
    sell = max(total - buy, 0)
    flow = buy - sell
    spread = num(a[0, 0] - b[0, 0]) if len(a) and len(b) else 0
    trend = np.tanh((last / sma - 1) * 100) if sma else 0
    rv = num(c.pct_change().tail(30).std())
    four = np.tanh(c.pct_change().tail(16).mean() * 1000)
    f.update(spread=spread, spread_pct=spread / last if last else 0,
              bid_ask_ratio_20=b20 / a20 if a20 else 1, bid_ask_ratio_50=b50 / a50 if a50 else 1,
              taker_buy_volume=buy, taker_sell_volume=sell, taker_flow=flow,
              taker_flow_ratio=flow / total if total else 0, price_return=last / prev - 1 if prev else 0,
              price_change=last - prev, sma_distance=last / sma - 1 if sma else 0, realized_volatility=rv,
              BOOK_IMB=o20, QUANT_IMPLY=float(np.tanh((o20 + o50 + trend) / 3)),
              ADAPT_CONF=float(np.clip(.5 + (abs(o20) + abs(trend)) / 2, 0, 1)),
              BAYESIAN=float(np.clip(.5 + (o20 + trend) / 4, 0, 1)), FOURIER_TREND=float(four))
    return f


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
    except Exception:
        return None


def ml_predict(f):
    m = load_model()
    if m is None:
        return None, None, "MODEL NOT FOUND", 0
    try:
        names = list(m.get_booster().feature_names or []) if hasattr(m, "get_booster") else []
        count = int(getattr(m, "n_features_in_", len(names) or 25))
        legacy = ["top20_bid_sum", "top20_ask_sum", "obi_top20", "spread", "bid_ask_ratio", "total_depth", "trend_signal"]
        cols = names if names else (legacy if count == 7 else list(f.keys()))
        row = dict(f, obi_top20=f["obi_20"], bid_ask_ratio=f["bid_ask_ratio_20"],
                   total_depth=f["top20_total_depth"], trend_signal=f["sma_distance"])
        x = pd.DataFrame([[row.get(c, 0.) for c in cols]], columns=cols)
        pred = int(m.predict(x)[0])
        proba = float(m.predict_proba(x)[0][-1]) if hasattr(m, "predict_proba") else None
        return pred, proba, "OK", len(cols)
    except Exception as e:
        return None, None, "ML ERROR: " + type(e).__name__, 0


def research(f):
    scores = {
        "OBI 20": np.clip(f["obi_20"] * 2, -1, 1),
        "OBI 20+50": np.clip((f["obi_20"] + f["obi_50"]) / 1.5, -1, 1),
        "OFI / Taker": np.clip(f["taker_flow_ratio"] * 2, -1, 1),
        "Trend / SMA": np.clip(np.tanh(f["sma_distance"] * 100), -1, 1),
        "Fourier": np.clip(f["FOURIER_TREND"], -1, 1),
        "Bayesian": np.clip((f["BAYESIAN"] - .5) * 2, -1, 1),
        "Quant Imply": np.clip(f["QUANT_IMPLY"], -1, 1),
        "Adaptive": np.clip((f["ADAPT_CONF"] - .5) * 2, -1, 1),
    }
    weights = {"OBI 20": .22, "OBI 20+50": .14, "OFI / Taker": .20, "Trend / SMA": .14,
               "Fourier": .10, "Bayesian": .08, "Quant Imply": .07, "Adaptive": .05}
    return scores, weights, float(sum(scores[k] * weights[k] for k in scores))


def final_state(f, p, pr, threshold=0.45, strict=False):
    scores, weights, rscore = research(f)
    mlscore = (pr - .5) * 2 if pr is not None else (1 if p == 1 else -1 if p == 0 else 0)
    combined = .6 * rscore + .4 * mlscore if p is not None else rscore
    effective_threshold = max(float(threshold), 0.35) if strict else float(threshold)
    signal = "LONG" if combined >= effective_threshold else "SHORT" if combined <= -effective_threshold else "WAIT"
    if strict and p is not None:
        research_dir = 1 if rscore > 0 else -1 if rscore < 0 else 0
        ml_dir = 1 if mlscore > 0 else -1 if mlscore < 0 else 0
        probability_ok = pr is not None and max(pr, 1.0 - pr) >= 0.60
        agreement_ok = research_dir == ml_dir and research_dir != 0
        if signal != "WAIT" and not (agreement_ok and probability_ok):
            signal = "WAIT"
    confidence = float(np.clip(50 + abs(combined) * 49, 1, 99))
    return signal, confidence, combined, scores, weights, rscore, mlscore


def visible_tri_timeframes(tf):
    if tf in ("1M", "3M", "5M", "15M", "30M"):
        return [("1H", "1 HOUR"), ("4H", "4 HOUR")]
    if tf in ("1H", "4H"):
        return [("1D", "DAILY"), ("1W", "WEEKLY")]
    if tf == "1D":
        return [("1W", "WEEKLY"), ("1MO", "MONTHLY")]
    if tf == "1W":
        return [("1MO", "MONTHLY")]
    return []


@st.cache_data(ttl=20, show_spinner=False)
def tri_levels(symbol, interval):
    df, _, _, _ = candles(symbol, interval, 8)
    if len(df) < 2:
        return None
    c = df.iloc[-2]
    o, h, l, cl = map(num, [c.Open, c.High, c.Low, c.Close])
    bh, bl = max(o, cl), min(o, cl)
    return {"body": (bh + bl) / 2, "upper": (h + bh) / 2, "lower": (l + bl) / 2}


def make_chart(df, symbol, tf, future):
    fig = go.Figure()
    if df.empty:
        return fig
    view = df.tail(650).copy()
    fig.add_trace(go.Candlestick(x=view.Time, open=view.Open, high=view.High, low=view.Low, close=view.Close,
                                  name="PRICE", increasing_line_color="#42dda0", increasing_fillcolor="#176d4f",
                                  decreasing_line_color="#ff7184", decreasing_fillcolor="#8e3448"))
    for span in (10, 20, 50, 200):
        if len(view) >= span:
            fig.add_trace(go.Scatter(x=view.Time, y=view.Close.ewm(span=span, adjust=False).mean(),
                                      mode="lines", name=f"EMA {span}", line={"width": 1.05}))
    line_colors = {"1MO": "#b29cff", "1W": "#8e98ff", "1D": "#65d7ff", "1H": "#42dda0", "4H": "#f3c86a"}
    for label, _ in visible_tri_timeframes(tf):
        lv = tri_levels(symbol, TFS[label])
        if not lv:
            continue
        col = line_colors[label]
        for kind, key, dash, width, pos in [("BODY 50", "body", "solid", 1.9, "top right"),
                                             ("UPPER 50", "upper", "dot", 1.05, "top left"),
                                             ("LOWER 50", "lower", "dot", 1.05, "bottom left")]:
            fig.add_hline(y=lv[key], line_color=col, line_width=width, line_dash=dash,
                          annotation_text=f"TRI {label} • {kind}", annotation_position=pos, annotation_font_size=9)
    step = view.Time.iloc[-1] - view.Time.iloc[-2] if len(view) > 1 else pd.Timedelta(minutes=5)
    fig.update_xaxes(range=[view.Time.iloc[0], view.Time.iloc[-1] + step * future], rangeslider_visible=False,
                     showgrid=True, gridcolor="#172230", showspikes=True, spikemode="across", fixedrange=False)
    fig.update_yaxes(side="right", showgrid=True, gridcolor="#172230", fixedrange=False, automargin=True)
    fig.update_layout(height=680, margin=dict(l=4, r=4, t=12, b=8), paper_bgcolor="#080d14", plot_bgcolor="#080d14",
                      font=dict(color="#cbd5e1"), hovermode="x unified", dragmode="pan",
                      uirevision=f"ZIA-{symbol}-{tf}", legend=dict(orientation="h", y=1.02, x=0),
                      hoverlabel=dict(font_size=11))
    return fig


def cards(items):
    cs = st.columns(len(items))
    for c, (lab, val, sub, cl) in zip(cs, items):
        with c:
            st.markdown(f'<div class="card"><div class="label">{lab}</div><div class="value {cl}">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def read_csv(path):
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def signal_validity_minutes(tf):
    return SIGNAL_VALIDITY_MINUTES.get(tf, 0)


def load_signal_history():
    try:
        if not SIGNAL_FILE.exists():
            return pd.DataFrame()
        return pd.read_csv(SIGNAL_FILE)
    except Exception:
        return pd.DataFrame()


def save_signal(symbol, tf, price, sig, conf, pr, f, rs, ml_status="", ml_pred=None, ml_score=0.0):
    if sig not in ("LONG", "SHORT"):
        return False
    now = datetime.now(timezone.utc)
    hist = load_signal_history()
    if not hist.empty:
        try:
            h = hist[(hist["symbol"].astype(str) == symbol) & (hist["timeframe"].astype(str) == tf)]
            if not h.empty:
                last = h.iloc[-1]
                last_ts = pd.to_datetime(last["timestamp"], utc=True, errors="coerce")
                if pd.notna(last_ts) and last.get("signal") == sig:
                    validity = signal_validity_minutes(tf)
                    if validity and (now - last_ts.to_pydatetime()) < timedelta(minutes=validity):
                        return False
        except Exception:
            pass
    row = {"timestamp": now.isoformat(), "symbol": symbol, "timeframe": tf, "price": price,
           "signal": sig, "confidence": conf, "ml_probability": pr if pr is not None else "",
           "ml_status": ml_status, "ml_prediction": ml_pred if ml_pred is not None else "",
           "ml_score": ml_score, "obi5": f.get("obi_5", 0.0), "obi10": f.get("obi_10", 0.0),
           "obi20": f["obi_20"], "obi50": f["obi_50"], "ofi": f["taker_flow_ratio"],
           "research_score": rs, "validity_minutes": signal_validity_minutes(tf)}
    pd.DataFrame([row]).to_csv(SIGNAL_FILE, mode="a", header=not SIGNAL_FILE.exists(), index=False)
    return True


def trade_pnl_stats(t):
    if t is None or t.empty or "result" not in t.columns:
        return {"wins": 0, "losses": 0, "win_pnl": 0.0, "loss_pnl": 0.0, "net_pnl": 0.0}
    df = t.copy()
    result = df["result"].astype(str).str.upper().str.strip()
    pct_col = next((c for c in ["pnl_pct", "pnl_percent", "profit_pct", "profit_percent", "return_pct", "pnl_percentage", "P&L %", "P&L%"] if c in df.columns), None)
    pct = pd.Series(np.nan, index=df.index, dtype=float)
    if pct_col:
        pct = pd.to_numeric(df[pct_col], errors="coerce")
        if pct.abs().dropna().median() <= 1.0:
            pct = pct * 100.0
    else:
        entry_col = next((c for c in ["entry_price", "entry", "open_price", "Entry Price"] if c in df.columns), None)
        exit_col = next((c for c in ["exit_price", "exit", "close_price", "Exit Price"] if c in df.columns), None)
        side_col = next((c for c in ["side", "direction", "signal", "Side"] if c in df.columns), None)
        if entry_col and exit_col:
            entry = pd.to_numeric(df[entry_col], errors="coerce")
            exit_ = pd.to_numeric(df[exit_col], errors="coerce")
            raw = (exit_ - entry) / entry.replace(0, np.nan) * 100.0
            if side_col:
                side = df[side_col].astype(str).str.upper()
                pct = np.where(side.str.contains("SHORT"), -raw, raw)
                pct = pd.Series(pct, index=df.index, dtype=float)
            else:
                pct = raw
    pct = pd.to_numeric(pct, errors="coerce")
    win_mask = result.eq("WIN") & pct.notna()
    loss_mask = result.eq("LOSS") & pct.notna()
    win_pnl = float(pct[win_mask].abs().sum())
    loss_pnl = -float(pct[loss_mask].abs().sum())
    return {"wins": int(result.eq("WIN").sum()), "losses": int(result.eq("LOSS").sum()), "win_pnl": win_pnl, "loss_pnl": loss_pnl, "net_pnl": win_pnl + loss_pnl}


def recover_active_signal(symbol, tf):
    validity = signal_validity_minutes(tf)
    if not validity:
        return None
    hist = load_signal_history()
    if hist.empty:
        return None
    try:
        h = hist[(hist["symbol"].astype(str) == symbol) & (hist["timeframe"].astype(str) == tf)]
        if h.empty:
            return None
        row = h.iloc[-1]
        if row.get("signal") not in ("LONG", "SHORT"):
            return None
        ts = pd.to_datetime(row["timestamp"], utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        age = datetime.now(timezone.utc) - ts.to_pydatetime()
        if age < timedelta(minutes=validity):
            return {"signal": row["signal"], "started": ts.to_pydatetime(), "confidence": num(row.get("confidence"), 0), "combined": 0.0}
    except Exception:
        return None
    return None


@st.cache_data(ttl=3.0, show_spinner=False)
def scan_symbol(symbol, tf_key, threshold=0.20):
    df, source, _, _ = candles(symbol, TFS[tf_key], 120)
    bids, asks, *_ = orderbook(symbol)
    f = features(df, bids, asks)
    pred, prob, mlstat, feature_count = ml_predict(f)
    strict = tf_key == "15M"
    signal, confidence, combined, rs, rw, rscore, mlscore = final_state(f, pred, prob, threshold, strict=strict)
    price = num(df.Close.iloc[-1]) if not df.empty else 0
    prev = num(df.Close.iloc[-2]) if len(df) > 1 else price
    change = (price / prev - 1) * 100 if prev else 0
    return {"symbol": symbol, "price": price, "change": change, "signal": signal, "confidence": confidence, "combined": combined,
            "obi20": f["obi_20"], "obi5": f["obi_5"], "obi10": f["obi_10"], "obi50": f["obi_50"], "ofi": f["taker_flow_ratio"],
            "ml_pred": pred, "ml_probability": prob, "ml_status": mlstat, "ml_features": feature_count, "research_score": rscore,
            "ml_score": mlscore, "research_scores": rs, "features": f}


def latest_saved_signal(symbol, tf):
    hist = load_signal_history()
    if hist.empty:
        return None
    try:
        h = hist[(hist["symbol"].astype(str) == symbol) & (hist["timeframe"].astype(str) == tf)]
        if h.empty:
            return None
        row = h.iloc[-1]
        if row.get("signal") not in ("LONG", "SHORT"):
            return None
        ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return {"signal": str(row.get("signal")), "entry_price": num(row.get("price")), "started": ts.to_pydatetime(), "confidence": num(row.get("confidence"))}
    except Exception:
        return None


def live_signal_pnl(entry_price, current_price, signal):
    entry = num(entry_price)
    current = num(current_price)
    if entry <= 0 or current <= 0 or signal not in ("LONG", "SHORT"):
        return 0.0
    raw = (current - entry) / entry * 100.0
    return raw if signal == "LONG" else -raw


def auto_scan_once(threshold=0.20):
    rows = []
    for tf_key in AUTO_SCAN_TFS:
        for symbol_key in SYMBOLS:
            snap = scan_symbol(symbol_key, tf_key, threshold)
            active = recover_active_signal(symbol_key, tf_key)
            if active:
                signal = active["signal"]
                confidence = active["confidence"]
                started = active["started"]
                rec = latest_saved_signal(symbol_key, tf_key)
                entry_price = rec["entry_price"] if rec else snap["price"]
            elif snap["signal"] in ("LONG", "SHORT"):
                signal = snap["signal"]
                confidence = snap["confidence"]
                started = datetime.now(timezone.utc)
                save_signal(symbol_key, tf_key, snap["price"], signal, confidence, snap.get("ml_probability"), snap["features"], snap.get("research_score", 0.0), snap.get("ml_status", ""), snap.get("ml_pred"), snap.get("ml_score", 0.0))
                rec = latest_saved_signal(symbol_key, tf_key)
                entry_price = rec["entry_price"] if rec else snap["price"]
            else:
                signal = "WAIT"
                confidence = snap["confidence"]
                started = None
                entry_price = 0.0
            pnl_pct = live_signal_pnl(entry_price, snap["price"], signal)
            validity = signal_validity_minutes(tf_key)
            remaining = 0
            if started and validity:
                remaining = max(0, int((started + timedelta(minutes=validity) - datetime.now(timezone.utc)).total_seconds()))
            rows.append({"symbol": symbol_key, "timeframe": tf_key, "price": snap["price"], "signal": signal, "confidence": confidence, "entry_price": entry_price, "pnl_pct": pnl_pct, "remaining_sec": remaining, "obi20": snap["obi20"]})
    return rows


# ------------------------------------------------------------
# STATE
# ------------------------------------------------------------
if "symbol" not in st.session_state:
    st.session_state.symbol = "BTCUSDT"
if "tf" not in st.session_state:
    st.session_state.tf = "15M"
if "future" not in st.session_state:
    st.session_state.future = 30
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.20
if "active_signal" not in st.session_state:
    st.session_state.active_signal = None
if "active_signal_started" not in st.session_state:
    st.session_state.active_signal_started = None
if "active_signal_key" not in st.session_state:
    st.session_state.active_signal_key = None
if "active_signal_confidence" not in st.session_state:
    st.session_state.active_signal_confidence = None
if "active_signal_combined" not in st.session_state:
    st.session_state.active_signal_combined = None

st.markdown('<div class="hero"><div><div class="brand">ZIA <b>RESEARCH</b></div>'
            '<div class="micro">QUANT MARKET INTELLIGENCE • LIVE ML • ORDER FLOW • MULTI-MARKET SCANNER</div></div>'
            '<div class="live"><span class="dot"></span>LIVE • SILENT</div></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([2, 1, 1, 1.3])
with c1:
    symbol = st.selectbox("MARKET", SYMBOLS, index=SYMBOLS.index(st.session_state.symbol), key="symbol")
with c2:
    tf = st.selectbox("TIMEFRAME", list(TFS.keys()), index=list(TFS.keys()).index(st.session_state.tf), key="tf")
with c3:
    future = st.selectbox("FUTURE SPACE", [12, 20, 30, 45, 60], index=2, key="future", format_func=lambda x: f"{x} bars")
with c4:
    threshold = st.slider("SIGNAL SENSITIVITY", min_value=0.10, max_value=0.60, step=0.05, value=st.session_state.threshold, key="threshold", help="Lower = more LONG/SHORT signals (less strict). Higher = fewer, stronger-conviction signals.")

visible = visible_tri_timeframes(tf)
tri_text = " + ".join(x[0] for x in visible) if visible else "NONE"
st.markdown(f'<div class="tri-strip"><div class="tri-chip">AUTO TRI</div><div class="tri-chip">CHART <span>{tf}</span></div><div class="tri-chip">DISPLAY <span>{tri_text}</span></div><div class="tri-chip">ZOOM <span>ON</span></div><div class="tri-chip">PAN <span>ON</span></div></div>', unsafe_allow_html=True)


@st.fragment(run_every="10s")
def live_engine():
    cycle_started = time.perf_counter()

    scan_clock = time.time()
    auto_rows_map = st.session_state.get("auto_scan_rows_map", {})
    cursor = int(st.session_state.get("auto_scan_cursor", 0))
    if not auto_rows_map:
        st.session_state.auto_scan_rows_map = {}
    if not st.session_state.get("auto_scan_initialized", False):
        st.session_state.auto_scan_initialized = True
        st.session_state.auto_scan_last = scan_clock
    elif scan_clock - st.session_state.get("auto_scan_last", 0.0) >= 1.0:
        try:
            pairs = [(tf_key, symbol_key) for tf_key in AUTO_SCAN_TFS for symbol_key in SYMBOLS]
            tf_key, symbol_key = pairs[cursor % len(pairs)]
            snap = scan_symbol(symbol_key, tf_key, threshold)
            active = recover_active_signal(symbol_key, tf_key)
            if active:
                signal = active["signal"]
                confidence = active["confidence"]
                started = active["started"]
                rec = latest_saved_signal(symbol_key, tf_key)
                entry_price = rec["entry_price"] if rec else snap["price"]
            elif snap["signal"] in ("LONG", "SHORT"):
                signal = snap["signal"]
                confidence = snap["confidence"]
                started = datetime.now(timezone.utc)
                save_signal(symbol_key, tf_key, snap["price"], signal, confidence, snap.get("ml_probability"), snap["features"], snap.get("research_score", 0.0), snap.get("ml_status", ""), snap.get("ml_pred"), snap.get("ml_score", 0.0))
                rec = latest_saved_signal(symbol_key, tf_key)
                entry_price = rec["entry_price"] if rec else snap["price"]
            else:
                signal = "WAIT"
                confidence = snap["confidence"]
                started = None
                entry_price = 0.0
            pnl_pct = live_signal_pnl(entry_price, snap["price"], signal)
            validity = signal_validity_minutes(tf_key)
            remaining = 0
            if started and validity:
                remaining = max(0, int((started + timedelta(minutes=validity) - datetime.now(timezone.utc)).total_seconds()))
            auto_rows_map[f"{symbol_key}:{tf_key}"] = {"symbol": symbol_key, "timeframe": tf_key, "price": snap["price"], "signal": signal, "confidence": confidence, "entry_price": entry_price, "pnl_pct": pnl_pct, "remaining_sec": remaining, "obi20": snap["obi20"], "obi5": snap.get("obi5", 0.0), "obi10": snap.get("obi10", 0.0), "obi50": snap.get("obi50", 0.0), "ofi": snap.get("ofi", 0.0), "ml_pred": snap.get("ml_pred"), "ml_probability": snap.get("ml_probability"), "ml_status": snap.get("ml_status", "—"), "ml_features": snap.get("ml_features", 0), "research_score": snap.get("research_score", 0.0), "ml_score": snap.get("ml_score", 0.0)}
            st.session_state.auto_scan_rows_map = auto_rows_map
            st.session_state.auto_scan_cursor = cursor + 1
            st.session_state.auto_scan_last = scan_clock
        except Exception:
            st.session_state.auto_scan_last = scan_clock
    auto_rows = list(st.session_state.get("auto_scan_rows_map", {}).values())
    prices_live = live_prices()
    for row in auto_rows:
        current = prices_live.get(row.get("symbol"))
        if current and row.get("signal") in ("LONG", "SHORT"):
            row["price"] = current
            row["pnl_pct"] = live_signal_pnl(row.get("entry_price", 0.0), current, row.get("signal"))

    df, source, cstat, _ = candles(symbol, TFS[tf], 650)
    bids, asks, bsrc, bstat, _ = orderbook(symbol)
    f = features(df, bids, asks)
    pred, prob, mlstat, feature_count = ml_predict(f)
    strict_15m = tf == "15M"
    raw_signal, raw_confidence, raw_combined, rs, rw, rscore, mlscore = final_state(f, pred, prob, threshold, strict=strict_15m)

    now = datetime.now(timezone.utc)
    signal = raw_signal
    confidence = raw_confidence
    combined = raw_combined

    validity = signal_validity_minutes(tf)
    if validity:
        key = f"{symbol}:{tf}"
        started_at = st.session_state.get("active_signal_started")
        active = st.session_state.get("active_signal")
        same_key = st.session_state.get("active_signal_key") == key
        expired = (not started_at) or ((now - started_at) >= timedelta(minutes=validity))
        if not same_key:
            recovered = recover_active_signal(symbol, tf)
            if recovered:
                active = recovered["signal"]
                started_at = recovered["started"]
                st.session_state.active_signal_confidence = recovered["confidence"]
                st.session_state.active_signal_combined = recovered["combined"]
            else:
                active = None
                started_at = None
            st.session_state.active_signal_key = key
            st.session_state.active_signal = active
            st.session_state.active_signal_started = started_at
            expired = (not started_at) or ((now - started_at) >= timedelta(minutes=validity))
        if expired:
            active = None
            st.session_state.active_signal = None
            st.session_state.active_signal_started = None
        if active in ("LONG", "SHORT") and not expired:
            signal = active
            confidence = float(st.session_state.get("active_signal_confidence") or raw_confidence)
            combined = float(st.session_state.get("active_signal_combined") or raw_combined)
        elif raw_signal in ("LONG", "SHORT"):
            st.session_state.active_signal = raw_signal
            st.session_state.active_signal_started = now
            st.session_state.active_signal_key = key
            st.session_state.active_signal_confidence = raw_confidence
            st.session_state.active_signal_combined = raw_combined
            signal = raw_signal
            confidence = raw_confidence
            combined = raw_combined
        save_signal(symbol, tf, num(df.Close.iloc[-1]) if not df.empty else 0, signal, confidence, prob, f, rscore)

    price = num(df.Close.iloc[-1]) if not df.empty else 0
    prev = num(df.Close.iloc[-2]) if len(df) > 1 else price
    change = (price / prev - 1) * 100 if prev else 0
    elapsed = (time.perf_counter() - cycle_started) * 1000
    cls = "signal-long" if signal == "LONG" else "signal-short" if signal == "SHORT" else "signal-wait"
    sigcolor = "good" if signal == "LONG" else "bad" if signal == "SHORT" else "amber"
    mltext = f"{prob * 100:.2f}%" if prob is not None else "—"
    selected_rec = latest_saved_signal(symbol, tf) if signal in ("LONG", "SHORT") else None
    selected_entry = selected_rec["entry_price"] if selected_rec else 0.0
    live_pnl_pct = live_signal_pnl(selected_entry, price, signal) if selected_entry else 0.0
    live_pnl_text = f"{live_pnl_pct:+.2f}%" if selected_entry else "—"

    lock_started = st.session_state.get("active_signal_started")
    lock_remaining = 0
    timer_text = "NO ACTIVE TIMER"
    timer_class = ""
    if validity and signal in ("LONG", "SHORT") and lock_started:
        lock_remaining = max(0, int((lock_started + timedelta(minutes=validity) - now).total_seconds()))
        mm, ss = divmod(lock_remaining, 60)
        hh, mm = divmod(mm, 60)
        countdown = f"{hh:02d}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"
        timer_text = f"⏱ SIGNAL LOCK • {countdown} REMAINING • FIXED {validity} MIN"
        timer_class = "live" if lock_remaining > 0 else "expired"
    elif validity:
        timer_text = f"⏱ NO ACTIVE LOCK • NEXT SIGNAL WINDOW {validity} MIN"

    st.markdown(f'<div class="signalbox {cls}"><div class="label">MAIN AI + RESEARCH SIGNAL</div><div class="signal-main {sigcolor}">{signal}</div><div class="signal-meta">CONFIDENCE {confidence:.1f}% • ML {mltext} ({mlstat}) • OBI5 {f["obi_5"]:+.3f} • OBI20 {f["obi_20"]:+.3f} • OBI50 {f["obi_50"]:+.3f} • OFI {f["taker_flow_ratio"]:+.3f} • LIVE P&L {live_pnl_text} • RESEARCH {rscore:+.3f} • COMPOSITE {combined:+.3f}</div><div class="signal-timer {timer_class}">{timer_text}</div></div>', unsafe_allow_html=True)

    cards([("PRICE", f"${price:,.2f}", f"{change:+.2f}% • {tf}", "good" if change >= 0 else "bad"),
           ("SIGNAL", signal, f"strength {confidence:.1f}%", sigcolor), ("ML", mltext, mlstat, "violet"),
           ("OBI 20", f"{f['obi_20']:+.3f}", "top 20 depth", "good" if f['obi_20'] >= 0 else "bad"),
           ("OBI 50", f"{f['obi_50']:+.3f}", "top 50 depth", "good" if f['obi_50'] >= 0 else "bad"),
           ("OFI", f"{f['taker_flow_ratio']:+.3f}", "live taker flow", "good" if f['taker_flow_ratio'] >= 0 else "bad"),
           ("LIVE P&L", live_pnl_text, "mark-to-market from signal entry", "good" if live_pnl_pct >= 0 else "bad"),
           ("DATA", source, f"book {bsrc}", "cyan")])

    trade_df = read_csv(TRADE_FILE)
    pnl = trade_pnl_stats(trade_df)
    cards([("WIN P&L", f"+{pnl['win_pnl']:.2f}%", f"{pnl['wins']} winning trades", "good"),
           ("LOSS P&L", f"{pnl['loss_pnl']:.2f}%", f"{pnl['losses']} losing trades", "bad"),
           ("NET P&L", f"{pnl['net_pnl']:+.2f}%", "closed trades total", "good" if pnl['net_pnl'] >= 0 else "bad"),
           ("CLOSED", str(pnl['wins'] + pnl['losses']), "resolved trades", "cyan")])

    tabs = st.tabs(["⌂ OVERVIEW", "◈ CHART", "◌ ORDER FLOW", "🧠 ML LAB", "🔬 RESEARCH LAB", "▣ SIGNALS", "⌖ SCANNER"])

    with tabs[0]:
        l, r = st.columns([2, 1])
        with l:
            st.markdown('<div class="panel"><b>MARKET REGIME</b>', unsafe_allow_html=True)
            regime = "BULLISH FLOW" if combined > .25 else "BEARISH FLOW" if combined < -.25 else "BALANCED / WAIT"
            st.markdown(f"## {regime}")
            st.progress(min(max(confidence / 100, 0), 1), text=f"Signal strength {confidence:.1f}%")
            st.write(f"Research **{rscore:+.3f}** • ML **{mlscore:+.3f}** • Composite **{combined:+.3f}**")
            st.markdown('</div>', unsafe_allow_html=True)
        with r:
            st.markdown('<div class="panel"><b>LIVE STATUS</b>', unsafe_allow_html=True)
            st.write(f"Candles: `{source}`")
            st.write(f"Order book: `{bsrc}`")
            st.write(f"Connection: `{cstat} / {bstat}`")
            st.write(f"Engine: `{elapsed:.0f} ms` • silent live cycle")
            st.write(f"Updated: `{datetime.now().strftime('%H:%M:%S')}`")
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown(f'<div class="panel"><b>TRADINGVIEW-STYLE MARKET CHART</b><div class="section-sub">Automatic TRI only • {tf} → {tri_text} • scroll zoom • mouse pan • crosshair • future space</div>', unsafe_allow_html=True)
        st.plotly_chart(make_chart(df, symbol, tf, future), use_container_width=True, config={"scrollZoom": True, "displaylogo": False, "responsive": True, "doubleClick": "reset", "modeBarButtonsToRemove": ["lasso2d", "select2d"]}, key="main_market_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        vals = [obi(bids, asks, k) for k in (5, 10, 20, 50)]
        cards([(f"OBI {k}", f"{v[0]:+.3f}", f"B {v[1]:,.1f} / A {v[2]:,.1f}", "good" if v[0] >= 0 else "bad") for k, v in zip((5, 10, 20, 50), vals)])
        l, r = st.columns(2)
        l.dataframe(pd.DataFrame(bids[:20], columns=["Bid Price", "Bid Qty"]), use_container_width=True, hide_index=True)
        r.dataframe(pd.DataFrame(asks[:20], columns=["Ask Price", "Ask Qty"]), use_container_width=True, hide_index=True)

    with tabs[3]:
        cards([("MODEL", mlstat, "xgboost_obi_model.pkl", "violet"), ("PREDICTION", "LONG" if pred == 1 else "SHORT" if pred == 0 else "—", f"class {pred}", "good" if pred == 1 else "bad" if pred == 0 else "amber"), ("PROBABILITY", mltext, "model probability", "violet"), ("FEATURES", str(feature_count), "supplied to model", "cyan")])
        st.markdown('<div class="panel"><b>LIVE MODEL INPUTS</b>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Feature": ["OBI 5", "OBI 10", "OBI 20", "OBI 50", "Spread", "Taker Flow", "Trend/SMA", "Volatility"], "Value": [f["obi_5"], f["obi_10"], f["obi_20"], f["obi_50"], f["spread"], f["taker_flow_ratio"], f["sma_distance"], f["realized_volatility"]]}), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[4]:
        rd = pd.DataFrame([{"Formula": k, "Live Score": round(float(v), 4), "Weight %": round(rw[k] * 100, 1), "Contribution": round(float(v * rw[k]), 4), "Direction": "BULL" if v > 0 else "BEAR" if v < 0 else "NEUTRAL"} for k, v in rs.items()]).sort_values("Contribution", ascending=False)
        st.markdown('<div class="panel"><b>RESEARCH FORMULA SCOREBOARD</b>', unsafe_allow_html=True)
        st.dataframe(rd, use_container_width=True, hide_index=True)
        st.write(f"Strongest contributor: **{rd.iloc[0]['Formula'] if not rd.empty else '—'}** • Composite **{rscore:+.3f}**")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[5]:
        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            if st.button("💾 SAVE CURRENT SIGNAL", use_container_width=True, key="save_signal_btn"):
                save_signal(symbol, tf, price, signal, confidence, prob, f, rscore, mlstat, pred, mlscore)
                st.success("Signal saved")
        with hcol2:
            if st.button("🗑 CLEAR HISTORY", use_container_width=True, key="clear_signal_history_btn"):
                try:
                    if SIGNAL_FILE.exists(): SIGNAL_FILE.unlink()
                    st.session_state.auto_scan_rows_map = {}
                    st.session_state.auto_scan_cursor = 0
                    st.session_state.active_signal = None
                    st.session_state.active_signal_started = None
                    st.session_state.active_signal_key = None
                    st.session_state.active_signal_confidence = None
                    st.session_state.active_signal_combined = None
                    st.success("Signal history cleared")
                except Exception as exc:
                    st.error(f"Could not clear signal history: {exc}")
        h = read_csv(SIGNAL_FILE)
        if not h.empty and "timeframe" in h.columns: h = h[h["timeframe"].astype(str).isin(AUTO_SCAN_TFS)]
        t = read_csv(TRADE_FILE)
        if not h.empty:
            st.dataframe(h.tail(80).iloc[::-1], use_container_width=True, hide_index=True)
            st.download_button("⬇ Download Signal Journal", h.to_csv(index=False), "zia_saved_signals.csv", "text/csv", use_container_width=True)
        else: st.info("No saved signals yet.")
        if not t.empty and "result" in t.columns:
            wins, losses = pnl["wins"], pnl["losses"]
            total = wins + losses
            wr = wins / total * 100 if total else 0
            cards([("CLOSED", str(total), "resolved trades", "cyan"), ("WINS", str(wins), "winning trades", "good"), ("LOSSES", str(losses), "losing trades", "bad"), ("WIN RATE", f"{wr:.1f}%", "closed trade rate", "violet")])
            cards([("WIN P&L", f"+{pnl['win_pnl']:.2f}%", "sum of winning trade P&L", "good"), ("LOSS P&L", f"{pnl['loss_pnl']:.2f}%", "sum of losing trade P&L", "bad"), ("NET P&L", f"{pnl['net_pnl']:+.2f}%", "wins + losses", "good" if pnl['net_pnl'] >= 0 else "bad")])

    with tabs[6]:
        st.markdown('<div class="panel"><b>MULTI-MARKET SCANNER</b><div class="section-sub">Automatic all-coin scanner • ONLY 15M / 1H / 4H • signals are saved automatically • live P&L</div>', unsafe_allow_html=True)
        rows = list(auto_rows)
        rows.sort(key=lambda r: (r["signal"] not in ("LONG", "SHORT"), -r["pnl_pct"]))
        longs = sum(1 for r in rows if r["signal"] == "LONG")
        shorts = sum(1 for r in rows if r["signal"] == "SHORT")
        cards([("LONG SIGNALS", str(longs), "active across all coins", "good"), ("SHORT SIGNALS", str(shorts), "active across all coins", "bad"), ("ACTIVE", str(len(rows)), "automatic signals", "cyan")])
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        for r in rows:
            pill_cls = "pill-long" if r["signal"] == "LONG" else "pill-short" if r["signal"] == "SHORT" else ""
            pnl_cls = "good" if r["pnl_pct"] >= 0 else "bad"
            mins, secs = divmod(max(0, int(r["remaining_sec"])), 60)
            timer = f"{mins:02d}:{secs:02d}" if mins < 60 else f"{mins // 60:02d}:{mins % 60:02d}"
            st.markdown(f'<div class="scan-row"><div style="width:13%;font-weight:900">{r["symbol"]}</div><div style="width:9%;font-weight:900">{r["timeframe"]}</div><div style="width:13%">${r["price"]:,.4f}</div><div style="width:13%" class="{pnl_cls}">{r["pnl_pct"]:+.2f}% P&L</div><div style="width:10%">OBI20 {r["obi20"]:+.3f}</div><div style="width:10%">OBI50 {r.get("obi50", 0.0):+.3f}</div><div style="width:10%">OFI {r.get("ofi", 0.0):+.3f}</div><div style="width:19%">ML {((r.get("ml_probability") or 0) * 100):.1f}% • {r.get("ml_status", "—")}</div><div style="width:13%">RESEARCH {r.get("research_score", 0.0):+.3f}</div><div style="width:10%">conf {r["confidence"]:.1f}%</div><div style="width:9%">lock {timer}</div><div style="width:13%;text-align:right"><span class="pill {pill_cls}">{r["signal"]}</span></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Every scanner row is live: ML probability/status, OBI20/OBI50, OFI, Research score, confidence and signal lock. WAIT rows are also shown so every tracked coin/timeframe is visible.")

    st.caption(f"ZIA Research • {symbol} • {tf} • silent live engine • {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")


live_engine()
