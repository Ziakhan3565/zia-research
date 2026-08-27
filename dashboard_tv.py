from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ZIA RESEARCH — responsive TradingView-style dashboard.
# Data flow: Binance -> order book/klines -> research features -> XGBoost + Research Lab -> UI.

st.set_page_config(page_title="ZIA Research Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path(__file__).resolve().parent
FUTURES = "https://fapi.binance.com"
SPOT = "https://api.binance.com"
MODEL_FILE = ROOT / "xgboost_obi_model.pkl"
META_FILE = ROOT / "ml_model_metadata.json"
RESEARCH_META = ROOT / "src" / "research_lab_ml_meta.json"

SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT","TRXUSDT","LTCUSDT","BCHUSDT","DOTUSDT","XLMUSDT","NEARUSDT","UNIUSDT","APTUSDT","TAOUSDT","XMRUSDT"]
TF = {"5M":"5m","15M":"15m","30M":"30m","1H":"1h","4H":"4h","1D":"1d","1W":"1w"}
FEATURES = ["top20_bid_sum","top20_ask_sum","obi_5","obi_10","obi_20","obi_50","spread","spread_pct","bid_ask_ratio_20","bid_ask_ratio_50","top20_total_depth","top50_total_depth","taker_buy_volume","taker_sell_volume","taker_flow","taker_flow_ratio","price_return","price_change","sma_distance","realized_volatility","BOOK_IMB","QUANT_IMPLY","ADAPT_CONF","BAYESIAN","FOURIER_TREND"]
OLD_FEATURES = ["top20_bid_sum","top20_ask_sum","obi_top20","spread","bid_ask_ratio","total_depth","trend_signal"]

st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#070b12;color:#e7edf7}
.block-container{max-width:1900px;padding:10px clamp(8px,1.5vw,26px) 70px}
[data-testid="stHeader"]{background:transparent}.stApp{font-family:Inter,system-ui,sans-serif}
[data-testid="stSidebar"]{background:#090e16;border-right:1px solid #202b3a}
.hero{display:flex;align-items:center;justify-content:space-between;padding:8px 2px 12px;border-bottom:1px solid #182333;margin-bottom:10px}
.logo{font-size:clamp(21px,2.2vw,31px);font-weight:950;letter-spacing:-1px}.logo b{color:#7d8cff}.tiny{font-size:10px;color:#718097;letter-spacing:.8px}
.live{border:1px solid #21583f;background:#0a1711;color:#73e0a6;border-radius:999px;padding:7px 11px;font-size:10px;font-weight:850}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#42d98a;box-shadow:0 0 9px #42d98a;margin-right:6px}
.card{background:linear-gradient(145deg,#101824,#0b1119);border:1px solid #202d3e;border-radius:13px;padding:12px;min-height:82px}.label{font-size:9px;color:#78879b;font-weight:850;letter-spacing:1px}.value{font-size:20px;font-weight:900;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sub{font-size:10px;color:#8492a5;margin-top:3px}
.sigbox{border:1px solid #26364a;border-radius:15px;padding:15px 18px;background:linear-gradient(135deg,#111c2b,#0a1018)}.sigbox.long{border-color:#247f5b}.sigbox.short{border-color:#a03e52}.sig{font-size:36px;font-weight:950;line-height:1}.section{font-size:14px;font-weight:900;margin:14px 0 7px}.panel{background:#0b1119;border:1px solid #1e2a3a;border-radius:14px;padding:9px}
.stPlotlyChart{border-radius:14px;overflow:hidden}.hint{font-size:10px;color:#65748a;text-align:center;margin-top:-3px}.pill{display:inline-block;border:1px solid #27374a;border-radius:999px;padding:4px 8px;font-size:10px;margin-right:5px}
@media(max-width:700px){.block-container{padding:6px 7px 55px}.hero{padding-bottom:8px}.tiny{font-size:8px}.card{padding:9px;min-height:70px}.value{font-size:16px}.sub{font-size:9px}.sig{font-size:29px}.section{margin-top:10px}.stPlotlyChart{margin-left:-3px;margin-right:-3px}.live{padding:5px 7px}.mobile-hide{display:none}}
</style>
""", unsafe_allow_html=True)


def num(x: Any, default=0.0):
    try:
        v=float(x); return v if np.isfinite(v) else default
    except Exception: return default


def fmt_price(x):
    x=num(x)
    if not x: return "—"
    if x >= 1000: return f"{x:,.2f}"
    if x >= 1: return f"{x:,.4f}"
    return f"{x:,.6f}"


@st.cache_data(ttl=2, show_spinner=False)
def get_json(base, path, params):
    try:
        r=requests.get(base+path, params=params, timeout=7, headers={"User-Agent":"ZIA-Research-Terminal/3.0"})
        return r.json() if r.ok else None
    except Exception: return None


@st.cache_data(ttl=3, show_spinner=False)
def get_klines(symbol, interval, limit):
    raw=get_json(FUTURES,"/fapi/v1/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1500)})
    if not isinstance(raw,list): raw=get_json(SPOT,"/api/v3/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1000)})
    rows=[]
    for c in raw or []:
        try: rows.append([pd.to_datetime(int(c[0]),unit="ms",utc=True),*map(float,c[1:6]),float(c[9])])
        except Exception: pass
    return pd.DataFrame(rows,columns=["Time","Open","High","Low","Close","Volume","TakerBuy"])


@st.cache_data(ttl=2, show_spinner=False)
def get_book(symbol):
    raw=get_json(FUTURES,"/fapi/v1/depth",{"symbol":symbol,"limit":100})
    if not isinstance(raw,dict): raw=get_json(SPOT,"/api/v3/depth",{"symbol":symbol,"limit":100})
    try: return np.asarray(raw.get("bids",[]),float),np.asarray(raw.get("asks",[]),float)
    except Exception: return np.empty((0,2)),np.empty((0,2))


def calc_obi(bids, asks, k):
    if len(bids)==0 or len(asks)==0:return 0.,0.,0.
    k=min(k,len(bids),len(asks)); b=float(bids[:k,1].sum()); a=float(asks[:k,1].sum()); return ((b-a)/(b+a) if b+a else 0.),b,a


def build_features(df,bids,asks):
    o5,b5,a5=calc_obi(bids,asks,5);o10,b10,a10=calc_obi(bids,asks,10);o20,b20,a20=calc_obi(bids,asks,20);o50,b50,a50=calc_obi(bids,asks,50)
    f={k:0. for k in FEATURES}
    if df.empty:return f
    close=df.Close; last=num(close.iloc[-1]); prev=num(close.iloc[-2] if len(close)>1 else last); sma=num(close.rolling(20).mean().iloc[-1],last); rv=num(close.pct_change().rolling(20).std())
    taker_buy=num(df.TakerBuy.tail(20).mean()); total=num(df.Volume.tail(20).mean()); taker_sell=max(total-taker_buy,0); flow=taker_buy-taker_sell
    trend=np.tanh((last/sma-1)*100) if sma else 0.; fourier=np.tanh(close.pct_change().tail(16).mean()*1000) if len(close)>5 else trend
    spread=(asks[0,0]-bids[0,0]) if len(asks) and len(bids) else 0.
    f.update(top20_bid_sum=b20,top20_ask_sum=a20,obi_5=o5,obi_10=o10,obi_20=o20,obi_50=o50,spread=spread,spread_pct=spread/last if last else 0,bid_ask_ratio_20=b20/a20 if a20 else 1,bid_ask_ratio_50=b50/a50 if a50 else 1,top20_total_depth=b20+a20,top50_total_depth=b50+a50,taker_buy_volume=taker_buy,taker_sell_volume=taker_sell,taker_flow=flow,taker_flow_ratio=flow/(taker_buy+taker_sell) if taker_buy+taker_sell else 0,price_return=last/prev-1 if prev else 0,price_change=last-prev,sma_distance=last/sma-1 if sma else 0,realized_volatility=rv,BOOK_IMB=o20,QUANT_IMPLY=np.tanh((o20+o50+trend)/3),ADAPT_CONF=np.clip(.5+(abs(o20)+abs(trend))/2,0,1),BAYESIAN=np.clip(.5+(o20+trend)/4,0,1),FOURIER_TREND=fourier)
    return f


@st.cache_resource(show_spinner=False)
def load_model():
    try:return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
    except Exception:return None


def ml_predict(f):
    m=load_model()
    if m is None:return None,None,0
    try:
        names=None
        if hasattr(m,"get_booster"):
            names=list(m.get_booster().feature_names or [])
        count=int(getattr(m,"n_features_in_",len(names) or len(FEATURES)))
        cols=names if names and all(x in f for x in names) else (OLD_FEATURES if count==7 else FEATURES)
        x=pd.DataFrame([[f.get(k,0) for k in cols]],columns=cols)
        pred=int(m.predict(x)[0]); prob=float(m.predict_proba(x)[0,1])
        return pred,prob,count
    except Exception:return None,None,0


def composite(f,p,prob):
    research=float(np.clip(.35*f["obi_20"]+.20*f["obi_50"]+.20*f["taker_flow_ratio"]+.15*f["sma_distance"]*20+.10*f["FOURIER_TREND"],-1,1))
    ml=(prob-.5)*2 if prob is not None else 0.; score=.6*research+.4*ml if p is not None else research
    sig="LONG" if score>=.45 else "SHORT" if score<=-.45 else "WAIT"; conf=round(np.clip(50+abs(score)*49,1,99),1)
    return sig,score,conf,research


def chart(df, symbol, interval, forward, show_volume, signal):
    fig=go.Figure()
    if df.empty:return fig
    fig.add_trace(go.Candlestick(x=df.Time,open=df.Open,high=df.High,low=df.Low,close=df.Close,name=symbol,increasing_line_color="#27d08c",increasing_fillcolor="#16a66f",decreasing_line_color="#ff7182",decreasing_fillcolor="#d84d61"))
    for span,name in [(20,"EMA 20"),(50,"EMA 50"),(200,"EMA 200")]:
        if len(df)>=span: fig.add_trace(go.Scatter(x=df.Time,y=df.Close.ewm(span=span,adjust=False).mean(),name=name,line={"width":1.2},hoverinfo="skip"))
    if show_volume: fig.add_trace(go.Bar(x=df.Time,y=df.Volume,name="Volume",opacity=.14,yaxis="y2"))
    step=(df.Time.iloc[-1]-df.Time.iloc[-2]) if len(df)>1 else pd.Timedelta(minutes=5); right=df.Time.iloc[-1]+step*forward
    fig.update_layout(template="plotly_dark",height=560 if st.session_state.get("mobile",False) else 680,paper_bgcolor="#080d14",plot_bgcolor="#080d14",margin={"l":3,"r":3,"t":7,"b":3},hovermode="x unified",dragmode="pan",uirevision=f"{symbol}-{interval}",showlegend=True,legend={"orientation":"h","y":1.02,"x":0,"font":{"size":10}},xaxis={"range":[df.Time.iloc[max(0,len(df)-220)],right],"rangeslider":{"visible":False},"fixedrange":False,"showgrid":True,"gridcolor":"#162131","showspikes":True,"spikemode":"across","spikethickness":1},yaxis={"side":"right","fixedrange":False,"showgrid":True,"gridcolor":"#162131","automargin":True},yaxis2={"overlaying":"y","side":"left","showticklabels":False,"showgrid":False})
    fig.add_hline(y=num(df.Close.iloc[-1]),line_width=1,line_dash="dot",opacity=.45)
    return fig


def card(label,value,sub):
    return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


if "symbol" not in st.session_state: st.session_state.symbol="BTCUSDT"
if "interval" not in st.session_state: st.session_state.interval="5m"
if "refresh" not in st.session_state: st.session_state.refresh=5
if "auto" not in st.session_state: st.session_state.auto=True

with st.sidebar:
    st.markdown("## ⚡ ZIA RESEARCH")
    st.caption("Responsive ML + order-flow terminal")
    st.session_state.symbol=st.selectbox("Market",SYMBOLS,index=SYMBOLS.index(st.session_state.symbol))
    lab=st.selectbox("Timeframe",list(TF),index=list(TF.values()).index(st.session_state.interval)); st.session_state.interval=TF[lab]
    candles_n=st.slider("Candles",100,1000,500,50); future=st.slider("Future chart space",10,150,40,5)
    show_volume=st.checkbox("Volume",True)
    st.session_state.auto=st.toggle("Seamless auto refresh",st.session_state.auto)
    st.session_state.refresh=st.slider("Refresh interval (sec)",2,15,st.session_state.refresh)
    st.caption("Mobile layout is responsive automatically. Refresh updates live data without a full browser reload.")

st.markdown('<div class="hero"><div><div class="logo">ZIA <b>RESEARCH TERMINAL</b></div><div class="tiny">ORDER FLOW · XGBOOST · RESEARCH LAB · PRICE ACTION</div></div><div class="live"><span class="dot"></span>LIVE</div></div>',unsafe_allow_html=True)

# Explicit responsive mode control; CSS handles the actual layout.
st.session_state.mobile=False

# Data is fetched together so all panels use the same market snapshot.
df=get_klines(st.session_state.symbol,st.session_state.interval,candles_n)
bids,asks=get_book(st.session_state.symbol)
features=build_features(df,bids,asks)
pred,prob,nfeat=ml_predict(features)
sig,score,conf,research=composite(features,pred,prob)
price=num(df.Close.iloc[-1]) if not df.empty else 0; prev=num(df.Close.iloc[-2]) if len(df)>1 else price; change=(price/prev-1)*100 if prev else 0
bias="BULLISH" if features["obi_20"]>.15 else "BEARISH" if features["obi_20"]<-.15 else "NEUTRAL"

c=st.columns(6)
items=[("PRICE",fmt_price(price),f"{change:+.2f}%"),("OBI TOP 20",f"{features['obi_20']:+.3f}",bias),("OBI TOP 50",f"{features['obi_50']:+.3f}","Depth imbalance"),("SPREAD",fmt_price(features['spread']),f"{features['spread_pct']*100:.3f}%"),("XGBOOST",("LONG" if pred==1 else "SHORT")+f" · {prob*100:.1f}%" if pred is not None else "OFFLINE",f"{nfeat} features" if nfeat else "model missing"),("CONFIDENCE",f"{conf:.1f}%",sig)]
for col,(a,b,d) in zip(c,items): col.markdown(card(a,b,d),unsafe_allow_html=True)

left,right=st.columns([2.6,1])
with left:
    cls="long" if sig=="LONG" else "short" if sig=="SHORT" else ""
    mltext=f"ML probability {prob*100:.1f}%" if prob is not None else "ML unavailable"
    st.markdown(f'<div class="section">FINAL DECISION</div><div class="sigbox {cls}"><div class="label">RESEARCH + ML COMPOSITE</div><div class="sig">{sig}</div><div class="sub">Score {score:+.3f} · Research {research:+.3f} · {mltext}</div></div>',unsafe_allow_html=True)
with right:
    st.markdown(f'<div class="section">ORDER BOOK</div><div class="card"><div class="label">BIAS</div><div class="value">{bias}</div><div class="sub">OBI 5/10/20/50: {features["obi_5"]:+.2f} / {features["obi_10"]:+.2f} / {features["obi_20"]:+.2f} / {features["obi_50"]:+.2f}</div></div>',unsafe_allow_html=True)

st.markdown('<div class="section">MARKET CHART</div>',unsafe_allow_html=True)
st.plotly_chart(chart(df,st.session_state.symbol,st.session_state.interval,future,show_volume,sig),use_container_width=True,config={"displaylogo":False,"scrollZoom":True,"doubleClick":"reset","responsive":True,"modeBarButtonsToAdd":["drawline","drawrect","eraseshape"],"modeBarButtonsToRemove":["lasso2d","select2d"]},key="zia_tv_chart")
st.markdown('<div class="hint">TradingView-style: mouse wheel = zoom · drag = pan · double click = reset · crosshair · right price scale · future empty space</div>',unsafe_allow_html=True)

st.markdown('<div class="section">ORDER FLOW SNAPSHOT</div>',unsafe_allow_html=True)
q=st.columns(4)
for col,(lab,val,sub) in zip(q,[("TOP 20 BID",f"{features['top20_bid_sum']:,.2f}","bid liquidity"),("TOP 20 ASK",f"{features['top20_ask_sum']:,.2f}","ask liquidity"),("TOP 50 DEPTH",f"{features['top50_total_depth']:,.2f}","total size"),("TAKER FLOW",f"{features['taker_flow_ratio']:+.2%}","buy vs sell")]): col.markdown(card(lab,val,sub),unsafe_allow_html=True)

st.markdown('<div class="section">MODEL / ENGINE STATUS</div>',unsafe_allow_html=True)
status=st.columns(4)
meta={}
try:
    if META_FILE.exists(): meta=json.loads(META_FILE.read_text(encoding="utf-8"))
except Exception: pass
for col,(lab,val,sub) in zip(status,[("XGBOOST FILE","READY" if MODEL_FILE.exists() else "MISSING",MODEL_FILE.name),("XGBOOST VERSION",str(meta.get("model_version",meta.get("version","—"))),"metadata"),("RESEARCH LAB","CONNECTED" if (ROOT/"src"/"research_lab.py").exists() else "NOT FOUND","TRI / research module"),("DATA SOURCE","BINANCE FUTURES","spot fallback enabled")]): col.markdown(card(lab,val,sub),unsafe_allow_html=True)

st.markdown('<div class="section">LIVE DATA</div>',unsafe_allow_html=True)
st.caption(f"{st.session_state.symbol} · {st.session_state.interval} · {len(df)} candles · last candle {df.Time.iloc[-1] if not df.empty else '—'} · auto refresh {'ON' if st.session_state.auto else 'OFF'}")

# Fragment refresh: only this small function reruns, preventing a visible full-page refresh.
if st.session_state.auto:
    try:
        from streamlit.runtime.fragment import fragment
        @fragment(run_every=f"{st.session_state.refresh}s")
        def _refresh_clock():
            st.empty().markdown(f'<div class="tiny" style="text-align:right">Live sync · {time.strftime("%H:%M:%S UTC")}</div>',unsafe_allow_html=True)
        _refresh_clock()
    except Exception:
        # Old Streamlit fallback; the dashboard remains functional.
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=st.session_state.refresh*1000,key="zia_fallback_refresh")
