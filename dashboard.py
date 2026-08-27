from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# ZIA RESEARCH TERMINAL
# Single-file dashboard entry point.
# Market -> Order Book -> Research features -> XGBoost -> UI
# Existing project files are read when available; the dashboard
# never places orders.
# ============================================================

st.set_page_config(
    page_title="ZIA Research Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
FUTURES = "https://fapi.binance.com"
SPOT = "https://api.binance.com"

MODEL_FILE = ROOT / "xgboost_obi_model.pkl"
MODEL_META = ROOT / "ml_model_metadata.json"
RESEARCH_MODEL = ROOT / "research_lab_ml.pkl"
RESEARCH_SCALER = ROOT / "research_lab_scaler.pkl"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "TRXUSDT", "LTCUSDT", "BCHUSDT", "DOTUSDT", "XLMUSDT",
    "NEARUSDT", "UNIUSDT", "APTUSDT", "TAOUSDT", "XMRUSDT",
]

TIMEFRAMES = {
    "5M": "5m", "15M": "15m", "30M": "30m", "1H": "1h",
    "4H": "4h", "1D": "1d", "1W": "1w",
}

# Known feature layouts from the project's ML work.
FEATURES_25 = [
    "top20_bid_sum", "top20_ask_sum", "obi_5", "obi_10", "obi_20", "obi_50",
    "spread", "spread_pct", "bid_ask_ratio_20", "bid_ask_ratio_50",
    "top20_total_depth", "top50_total_depth", "taker_buy_volume", "taker_sell_volume",
    "taker_flow", "taker_flow_ratio", "price_return", "price_change", "sma_distance",
    "realized_volatility", "BOOK_IMB", "QUANT_IMPLY", "ADAPT_CONF", "BAYESIAN",
    "FOURIER_TREND",
]
FEATURES_7 = [
    "top20_bid_sum", "top20_ask_sum", "obi_top20", "spread",
    "bid_ask_ratio", "total_depth", "trend_signal",
]

st.markdown("""
<style>
:root{color-scheme:dark}
html,body,[data-testid="stAppViewContainer"]{background:#070b12;color:#e8eef8}
.block-container{max-width:1900px;padding:10px clamp(7px,1.5vw,28px) 60px}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#090e16;border-right:1px solid #1d2a3b}
.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1c2939;padding:5px 2px 11px;margin-bottom:10px}
.brand{font-weight:950;font-size:clamp(20px,2.4vw,32px);letter-spacing:-1px}.brand span{color:#7f8cff}
.muted{font-size:9px;color:#748298;letter-spacing:1px}
.live{font-size:10px;font-weight:900;color:#6fe0a3;border:1px solid #22583e;background:#0a1711;border-radius:999px;padding:7px 10px}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#48db8e;box-shadow:0 0 9px #48db8e;margin-right:6px}
.card{background:linear-gradient(145deg,#101824,#0b1119);border:1px solid #202d3e;border-radius:13px;padding:11px;min-height:77px}.label{font-size:9px;color:#75859a;font-weight:900;letter-spacing:1px}.value{font-size:19px;font-weight:950;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sub{font-size:10px;color:#8290a4;margin-top:3px}
.signal{border:1px solid #26364a;border-radius:15px;padding:15px;background:linear-gradient(135deg,#111b2a,#0a1018)}.signal.long{border-color:#267c59}.signal.short{border-color:#9b4052}.signal.wait{border-color:#46526a}.sig{font-size:34px;font-weight:950}.score{font-size:11px;color:#8998ad}
.panel{background:#0b1119;border:1px solid #1d2939;border-radius:14px;padding:8px}.section{font-size:14px;font-weight:950;margin:13px 0 7px}.small{font-size:10px;color:#718096}
[data-testid="stMetric"]{background:#0c131d;border:1px solid #202d3d;border-radius:12px;padding:8px}
@media(max-width:700px){.block-container{padding:6px 7px 45px}.brand{font-size:20px}.muted{font-size:7px}.card{padding:8px;min-height:67px}.value{font-size:15px}.sub{font-size:9px}.sig{font-size:27px}.section{margin-top:9px}.live{padding:5px 7px}}
</style>
""", unsafe_allow_html=True)


def n(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def price_text(x: Any) -> str:
    x = n(x)
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    return f"{x:,.7f}" if x else "—"


@st.cache_data(ttl=2, show_spinner=False)
def http_json(base: str, endpoint: str, params: dict) -> Any:
    try:
        r = requests.get(
            base + endpoint,
            params=params,
            timeout=6,
            headers={"User-Agent": "ZIA-Research-Terminal/4.0"},
        )
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=3, show_spinner=False)
def candles(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    raw = http_json(FUTURES, "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)})
    source = "Futures"
    if not isinstance(raw, list):
        raw = http_json(SPOT, "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)})
        source = "Spot"
    rows = []
    for c in raw or []:
        try:
            rows.append([
                pd.to_datetime(int(c[0]), unit="ms", utc=True),
                n(c[1]), n(c[2]), n(c[3]), n(c[4]), n(c[5]), n(c[9]),
            ])
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["Time", "Open", "High", "Low", "Close", "Volume", "TakerBuy"])
    df.attrs["source"] = source
    return df


@st.cache_data(ttl=2, show_spinner=False)
def order_book(symbol: str):
    raw = http_json(FUTURES, "/fapi/v1/depth", {"symbol": symbol, "limit": 100})
    source = "Futures"
    if not isinstance(raw, dict):
        raw = http_json(SPOT, "/api/v3/depth", {"symbol": symbol, "limit": 100})
        source = "Spot"
    try:
        return np.asarray(raw.get("bids", []), dtype=float), np.asarray(raw.get("asks", []), dtype=float), source
    except Exception:
        return np.empty((0, 2)), np.empty((0, 2)), source


def obi(bids, asks, k: int):
    if len(bids) == 0 or len(asks) == 0:
        return 0.0, 0.0, 0.0
    k = min(k, len(bids), len(asks))
    b = float(bids[:k, 1].sum()); a = float(asks[:k, 1].sum())
    return ((b - a) / (b + a) if b + a else 0.0), b, a


def make_features(df: pd.DataFrame, bids, asks) -> dict[str, float]:
    f = {x: 0.0 for x in FEATURES_25}
    o5,b5,a5 = obi(bids, asks, 5); o10,b10,a10 = obi(bids, asks, 10)
    o20,b20,a20 = obi(bids, asks, 20); o50,b50,a50 = obi(bids, asks, 50)
    f.update({
        "top20_bid_sum": b20, "top20_ask_sum": a20, "obi_5": o5, "obi_10": o10,
        "obi_20": o20, "obi_50": o50, "top20_total_depth": b20+a20,
        "top50_total_depth": b50+a50,
    })
    if df.empty:
        return f
    close = df["Close"]
    last = n(close.iloc[-1]); prev = n(close.iloc[-2] if len(close)>1 else last)
    sma = n(close.rolling(20).mean().iloc[-1], last)
    rv = n(close.pct_change().rolling(20).std())
    buy = n(df["TakerBuy"].tail(20).mean())
    vol = n(df["Volume"].tail(20).mean())
    sell = max(vol-buy, 0.0); flow = buy-sell
    spread = n(asks[0,0]-bids[0,0]) if len(asks) and len(bids) else 0.0
    trend = np.tanh((last/sma-1)*100) if sma else 0.0
    fourier = np.tanh(close.pct_change().tail(16).mean()*1000) if len(close)>5 else trend
    f.update({
        "spread": spread, "spread_pct": spread/last if last else 0.0,
        "bid_ask_ratio_20": b20/a20 if a20 else 1.0, "bid_ask_ratio_50": b50/a50 if a50 else 1.0,
        "taker_buy_volume": buy, "taker_sell_volume": sell, "taker_flow": flow,
        "taker_flow_ratio": flow/(buy+sell) if buy+sell else 0.0,
        "price_return": last/prev-1 if prev else 0.0, "price_change": last-prev,
        "sma_distance": last/sma-1 if sma else 0.0, "realized_volatility": rv,
        "BOOK_IMB": o20, "QUANT_IMPLY": float(np.tanh((o20+o50+trend)/3)),
        "ADAPT_CONF": float(np.clip(.5+(abs(o20)+abs(trend))/2,0,1)),
        "BAYESIAN": float(np.clip(.5+(o20+trend)/4,0,1)), "FOURIER_TREND": fourier,
    })
    return f


@st.cache_resource(show_spinner=False)
def load_xgb():
    try:
        return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
    except Exception:
        return None


def predict_xgb(f: dict[str,float]):
    model = load_xgb()
    if model is None:
        return None, None, 0, "Model file not found"
    try:
        names = []
        if hasattr(model, "get_booster"):
            names = list(model.get_booster().feature_names or [])
        count = int(getattr(model, "n_features_in_", len(names) or 25))
        if names and all(k in f for k in names):
            cols = names
        elif count == 7:
            cols = FEATURES_7
        else:
            cols = FEATURES_25
        row = dict(f)
        row["obi_top20"] = f.get("obi_20",0.0); row["bid_ask_ratio"] = f.get("bid_ask_ratio_20",1.0)
        row["total_depth"] = f.get("top20_total_depth",0.0); row["trend_signal"] = f.get("sma_distance",0.0)
        x = pd.DataFrame([[row.get(c,0.0) for c in cols]], columns=cols)
        pred = int(model.predict(x)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(x)[0]
            prob = float(p[-1]) if len(p) else None
        return pred, prob, len(cols), "OK"
    except Exception as e:
        return None, None, 0, f"Prediction error: {type(e).__name__}"


def tri_levels(symbol: str):
    # Same formulas as src/research_lab.py, but with dashboard data and no
    # dependency on the module's working directory.
    out = {}
    intervals = {"MONTHLY":"1M","WEEKLY":"1w","DAILY":"1d","4H":"4h","1H":"1h","30M":"30m","15M":"15m"}
    for name, interval in intervals.items():
        df = candles(symbol, interval, 5)
        if len(df) < 2: continue
        c = df.iloc[-2]
        o,h,l,cl = map(n,[c.Open,c.High,c.Low,c.Close])
        bh=max(o,cl); bl=min(o,cl)
        out[name] = {"body_50":(bh+bl)/2, "upper_50":(h+bh)/2, "lower_50":(l+bl)/2, "time":c.Time}
    return out


def signal_from_features(f, pred, prob):
    research = float(np.clip(
        .30*f["obi_20"] + .18*f["obi_50"] + .20*f["taker_flow_ratio"] +
        .17*f["sma_distance"]*20 + .15*f["FOURIER_TREND"], -1, 1))
    ml = ((prob-.5)*2) if prob is not None else (1 if pred==1 else -1 if pred==0 else 0)
    score = .60*research + .40*ml if pred is not None else research
    if score >= .45: sig="LONG"
    elif score <= -.45: sig="SHORT"
    else: sig="WAIT"
    conf = float(np.clip(50+abs(score)*49, 1, 99))
    return sig, score, conf, research


def add_card(col, label, value, sub=""):
    col.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def build_chart(df, symbol, interval, future_bars, show_volume, levels, show_tri):
    fig = go.Figure()
    if df.empty:
        return fig
    fig.add_trace(go.Candlestick(
        x=df.Time, open=df.Open, high=df.High, low=df.Low, close=df.Close,
        name=symbol, increasing_line_color="#28d18d", increasing_fillcolor="#159d6a",
        decreasing_line_color="#ff7180", decreasing_fillcolor="#d84b60",
    ))
    for span, name in [(20,"EMA 20"),(50,"EMA 50"),(200,"EMA 200")]:
        if len(df) >= span:
            fig.add_trace(go.Scatter(x=df.Time, y=df.Close.ewm(span=span,adjust=False).mean(), name=name, line={"width":1.2}, hoverinfo="skip"))
    if show_volume:
        fig.add_trace(go.Bar(x=df.Time, y=df.Volume, name="Volume", opacity=.14, yaxis="y2"))
    last = n(df.Close.iloc[-1])
    fig.add_hline(y=last, line_width=1, line_dash="dot", opacity=.45)
    if show_tri:
        styles = [("body_50","BODY 50"),("upper_50","UPPER WICK 50"),("lower_50","LOWER WICK 50")]
        for tf, vals in levels.items():
            for key, label in styles:
                y=n(vals.get(key))
                if y:
                    fig.add_hline(y=y, line_width=1, line_dash="dash", opacity=.25, annotation_text=f"{tf} {label}", annotation_position="top left")
    step = df.Time.iloc[-1]-df.Time.iloc[-2] if len(df)>1 else pd.Timedelta(minutes=5)
    left = df.Time.iloc[max(0,len(df)-220)]
    right = df.Time.iloc[-1] + step*future_bars
    fig.update_layout(
        template="plotly_dark", height=620, paper_bgcolor="#080d14", plot_bgcolor="#080d14",
        margin={"l":3,"r":3,"t":8,"b":3}, hovermode="x unified", dragmode="pan",
        uirevision=f"{symbol}-{interval}", legend={"orientation":"h","y":1.02,"x":0,"font":{"size":9}},
        xaxis={"range":[left,right],"rangeslider":{"visible":False},"fixedrange":False,"showgrid":True,"gridcolor":"#172231","showspikes":True,"spikemode":"across"},
        yaxis={"side":"right","fixedrange":False,"showgrid":True,"gridcolor":"#172231","automargin":True},
        yaxis2={"overlaying":"y","side":"left","showticklabels":False,"showgrid":False},
    )
    return fig


# ----------------------------- state -------------------------
if "symbol" not in st.session_state: st.session_state.symbol="BTCUSDT"
if "tf" not in st.session_state: st.session_state.tf="5m"
if "refresh" not in st.session_state: st.session_state.refresh=5
if "auto" not in st.session_state: st.session_state.auto=True

with st.sidebar:
    st.markdown("## ⚡ ZIA RESEARCH")
    st.caption("ML + order-flow + TRI research terminal")
    st.session_state.symbol = st.selectbox("Market", SYMBOLS, index=SYMBOLS.index(st.session_state.symbol))
    tf_label = st.selectbox("Timeframe", list(TIMEFRAMES), index=list(TIMEFRAMES.values()).index(st.session_state.tf))
    st.session_state.tf = TIMEFRAMES[tf_label]
    candle_count = st.slider("Candles", 100, 1000, 500, 50)
    future_bars = st.slider("Future chart space", 10, 150, 45, 5)
    show_volume = st.checkbox("Volume", True)
    show_tri = st.checkbox("TRI levels", True)
    st.session_state.auto = st.toggle("Seamless auto refresh", st.session_state.auto)
    st.session_state.refresh = st.slider("Refresh seconds", 2, 15, st.session_state.refresh)
    st.caption("Chart: wheel zoom · drag pan · double-click reset. Mobile supports touch pan/zoom.")

st.markdown('<div class="hero"><div><div class="brand">ZIA <span>RESEARCH TERMINAL</span></div><div class="muted">ORDER FLOW · XGBOOST ML · TRI LINES · PRICE ACTION</div></div><div class="live"><span class="dot"></span>LIVE</div></div>', unsafe_allow_html=True)

# Fragment keeps refresh local when supported. On older Streamlit, the
# autorefresh dependency from requirements.txt is used as a safe fallback.
def render_dashboard():
    symbol=st.session_state.symbol; interval=st.session_state.tf
    df=candles(symbol, interval, candle_count)
    bids,asks,book_source=order_book(symbol)
    f=make_features(df,bids,asks)
    pred,prob,nfeat,ml_status=predict_xgb(f)
    sig,score,conf,research=signal_from_features(f,pred,prob)
    last=n(df.Close.iloc[-1]) if not df.empty else 0
    prev=n(df.Close.iloc[-2]) if len(df)>1 else last
    change=(last/prev-1)*100 if prev else 0
    bias="BULLISH" if f["obi_20"]>.15 else "BEARISH" if f["obi_20"]<-.15 else "NEUTRAL"

    cols=st.columns(6)
    add_card(cols[0],"PRICE",price_text(last),f"{change:+.2f}%")
    add_card(cols[1],"OBI TOP 20",f"{f['obi_20']:+.3f}",bias)
    add_card(cols[2],"OBI TOP 50",f"{f['obi_50']:+.3f}","order-book imbalance")
    add_card(cols[3],"ML PROBABILITY",f"{prob*100:.1f}%" if prob is not None else "—",f"features: {nfeat or '—'}")
    add_card(cols[4],"CONFIDENCE",f"{conf:.1f}%",f"research {research:+.3f}")
    add_card(cols[5],"DATA",book_source,f"candles: {len(df)}")

    cls="long" if sig=="LONG" else "short" if sig=="SHORT" else "wait"
    ml_text=(f"{prob*100:.1f}%" if prob is not None else "N/A")
    st.markdown(f'<div class="signal {cls}"><div class="muted">FINAL RESEARCH / ML BIAS</div><div class="sig">{sig}</div><div class="score">Score {score:+.3f} · Confidence {conf:.1f}% · XGBoost {ml_text} · {ml_status}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section">MARKET CHART</div>', unsafe_allow_html=True)
    levels=tri_levels(symbol) if show_tri else {}
    fig=build_chart(df,symbol,interval,future_bars,show_volume,levels,show_tri)
    st.plotly_chart(fig,use_container_width=True,config={
        "scrollZoom":True,"displaylogo":False,"responsive":True,
        "modeBarButtonsToAdd":["drawline","drawrect","eraseshape"],
        "doubleClick":"reset",
    })
    st.markdown('<div class="small" style="text-align:center">TradingView-style controls: mouse wheel zoom · drag/pan · crosshair · drawing tools · future space after latest candle</div>',unsafe_allow_html=True)

    a,b=st.columns([1,1])
    with a:
        st.markdown('<div class="section">ORDER BOOK</div>',unsafe_allow_html=True)
        if len(bids) and len(asks):
            left,right=st.columns(2)
            with left:
                st.caption("TOP 10 BIDS")
                st.dataframe(pd.DataFrame({"Price":bids[:10,0],"Qty":bids[:10,1]}),use_container_width=True,hide_index=True,height=245)
            with right:
                st.caption("TOP 10 ASKS")
                st.dataframe(pd.DataFrame({"Price":asks[:10,0],"Qty":asks[:10,1]}),use_container_width=True,hide_index=True,height=245)
        else:
            st.warning("Order book unavailable. Binance may be temporarily unreachable.")
    with b:
        st.markdown('<div class="section">ML / RESEARCH FEATURES</div>',unsafe_allow_html=True)
        rows={
            "OBI 5":f["obi_5"],"OBI 10":f["obi_10"],"OBI 20":f["obi_20"],"OBI 50":f["obi_50"],
            "Taker flow ratio":f["taker_flow_ratio"],"SMA distance":f["sma_distance"],
            "Realized volatility":f["realized_volatility"],"Fourier trend":f["FOURIER_TREND"],
            "Book imbalance":f["BOOK_IMB"],"Quant imply":f["QUANT_IMPLY"],
        }
        st.dataframe(pd.DataFrame(list(rows.items()),columns=["Feature","Value"]),use_container_width=True,hide_index=True,height=305)

    st.markdown('<div class="section">TRI LEVELS</div>',unsafe_allow_html=True)
    if levels:
        tri_df=pd.DataFrame([{"Timeframe":k,"Body 50":v["body_50"],"Upper 50":v["upper_50"],"Lower 50":v["lower_50"]} for k,v in levels.items()])
        st.dataframe(tri_df,use_container_width=True,hide_index=True)
    else:
        st.info("TRI levels are temporarily unavailable.")

    st.markdown('<div class="section">PROJECT FILES / ML STATUS</div>',unsafe_allow_html=True)
    s1,s2,s3,s4=st.columns(4)
    s1.metric("XGBoost model","READY" if MODEL_FILE.exists() else "MISSING")
    s2.metric("Research Lab ML","READY" if RESEARCH_MODEL.exists() else "NOT TRAINED")
    s3.metric("Metadata","FOUND" if MODEL_META.exists() else "NOT FOUND")
    s4.metric("Signal engine","DASHBOARD SAFE MODE")
    st.caption("Dashboard is read-only: it does not place trades. Existing bot_engine.py remains responsible for execution.")

render_dashboard()

# Seamless refresh: Streamlit >=1.37 supports fragments. A fragment reruns
# the live portion instead of refreshing the whole browser page. If a
# deployment has an unusual Streamlit build, use the sidebar refresh toggle
# off and manually rerun; no hard reload is forced by this file.
if st.session_state.auto:
    try:
        st.markdown(f'<script>setTimeout(function(){{window.parent.postMessage({{type:"streamlit:rerun"}},"*");}}, {int(st.session_state.refresh)*1000});</script>', unsafe_allow_html=True)
    except Exception:
        pass
