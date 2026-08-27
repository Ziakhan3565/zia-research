from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="ZIA Research Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "xgboost_obi_model.pkl"

# Public Binance endpoints. We try several hosts because a single Binance host can
# be blocked/rate-limited from a cloud region.
FUTURES_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]
SPOT_HOSTS = ["https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com", "https://api3.binance.com"]
DATA_HOST = "https://data-api.binance.vision"

TIMEFRAMES = {"5M":"5m", "15M":"15m", "30M":"30m", "1H":"1h", "4H":"4h", "1D":"1d", "1W":"1w"}
SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT","TRXUSDT","LTCUSDT","BCHUSDT","DOTUSDT","XLMUSDT"]

F25 = ["top20_bid_sum","top20_ask_sum","obi_5","obi_10","obi_20","obi_50","spread","spread_pct","bid_ask_ratio_20","bid_ask_ratio_50","top20_total_depth","top50_total_depth","taker_buy_volume","taker_sell_volume","taker_flow","taker_flow_ratio","price_return","price_change","sma_distance","realized_volatility","BOOK_IMB","QUANT_IMPLY","ADAPT_CONF","BAYESIAN","FOURIER_TREND"]
F7 = ["top20_bid_sum","top20_ask_sum","obi_top20","spread","bid_ask_ratio","total_depth","trend_signal"]

st.markdown("""
<style>
html,body,[data-testid=stAppViewContainer]{background:#070b12;color:#e9eef7}
.block-container{max-width:1900px;padding:10px clamp(7px,1.5vw,26px) 45px}
[data-testid=stHeader]{background:transparent}
.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1c2938;padding:4px 2px 10px;margin-bottom:10px}
.brand{font-size:clamp(20px,2.3vw,31px);font-weight:950;letter-spacing:-1px}.brand span{color:#8490ff}.live{border:1px solid #225a40;background:#0a1711;color:#6fe2a5;border-radius:999px;padding:6px 10px;font-size:10px;font-weight:900}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#48dc8f;box-shadow:0 0 9px #48dc8f;margin-right:6px}
.card{background:linear-gradient(145deg,#101824,#0b1119);border:1px solid #202d3e;border-radius:13px;padding:10px;min-height:72px}.label{font-size:9px;color:#77879d;font-weight:900;letter-spacing:1px}.value{font-size:18px;font-weight:950;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sub{font-size:10px;color:#8491a5;margin-top:3px}.panel{background:#0b1119;border:1px solid #1d2939;border-radius:14px;padding:8px}.section{font-size:14px;font-weight:950;margin:12px 0 7px}.sig{font-size:32px;font-weight:950}.small{font-size:10px;color:#748399}
@media(max-width:700px){.block-container{padding:6px 7px 40px}.brand{font-size:20px}.live{font-size:8px;padding:5px 7px}.card{min-height:62px;padding:8px}.value{font-size:15px}.sub{font-size:9px}.sig{font-size:27px}}
</style>
""", unsafe_allow_html=True)


def num(x: Any, default=0.0):
    try:
        v=float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def fmt_price(x):
    x=num(x)
    if x >= 1000: return f"{x:,.2f}"
    if x >= 1: return f"{x:,.4f}"
    return f"{x:,.7f}" if x else "—"


def request_json(hosts, path, params):
    last_error=""
    for host in hosts:
        try:
            r=requests.get(host+path, params=params, timeout=5, headers={"User-Agent":"ZIA-Research/5.0"})
            if r.ok:
                return r.json(), host, "OK"
            last_error=f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_error=type(e).__name__
    return None, None, last_error


@st.cache_data(ttl=2, show_spinner=False)
def get_candles(symbol, interval, limit=500):
    raw,host,status=request_json(FUTURES_HOSTS,"/fapi/v1/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1500)})
    source="Futures"
    if not isinstance(raw,list):
        raw,host,status=request_json(SPOT_HOSTS,"/api/v3/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1000)})
        source="Spot"
    if not isinstance(raw,list):
        raw,host,status=request_json([DATA_HOST],"/api/v3/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1000)})
        source="Data API"
    rows=[]
    for c in raw or []:
        try:
            rows.append([pd.to_datetime(int(c[0]),unit="ms",utc=True),num(c[1]),num(c[2]),num(c[3]),num(c[4]),num(c[5]),num(c[9])])
        except Exception: pass
    df=pd.DataFrame(rows,columns=["Time","Open","High","Low","Close","Volume","TakerBuy"])
    return df,source,status,host


@st.cache_data(ttl=2, show_spinner=False)
def get_order_book(symbol):
    # Futures first, then several Binance mirrors, then Binance public data API.
    raw,host,status=request_json(FUTURES_HOSTS,"/fapi/v1/depth",{"symbol":symbol,"limit":100})
    source="Futures"
    if not isinstance(raw,dict) or not raw.get("bids") or not raw.get("asks"):
        raw,host,status=request_json(SPOT_HOSTS,"/api/v3/depth",{"symbol":symbol,"limit":100})
        source="Spot"
    if not isinstance(raw,dict) or not raw.get("bids") or not raw.get("asks"):
        raw,host,status=request_json([DATA_HOST],"/api/v3/depth",{"symbol":symbol,"limit":100})
        source="Data API"
    try:
        bids=np.asarray(raw.get("bids",[]),dtype=float)
        asks=np.asarray(raw.get("asks",[]),dtype=float)
        if bids.ndim!=2 or asks.ndim!=2 or len(bids)==0 or len(asks)==0: raise ValueError("empty book")
        return bids,asks,source,"OK",host
    except Exception:
        return np.empty((0,2)),np.empty((0,2)),source,status,host


def ob(bids,asks,k):
    if len(bids)==0 or len(asks)==0:return 0.0,0.0,0.0
    k=min(k,len(bids),len(asks)); b=float(bids[:k,1].sum()); a=float(asks[:k,1].sum())
    return ((b-a)/(b+a) if b+a else 0),b,a


def features(df,bids,asks):
    f={x:0.0 for x in F25}
    o5,b5,a5=ob(bids,asks,5);o10,b10,a10=ob(bids,asks,10);o20,b20,a20=ob(bids,asks,20);o50,b50,a50=ob(bids,asks,50)
    f.update(top20_bid_sum=b20,top20_ask_sum=a20,obi_5=o5,obi_10=o10,obi_20=o20,obi_50=o50,top20_total_depth=b20+a20,top50_total_depth=b50+a50)
    if df.empty:return f
    close=df.Close; last=num(close.iloc[-1]); prev=num(close.iloc[-2] if len(close)>1 else last); sma=num(close.rolling(20).mean().iloc[-1],last)
    spread=num(asks[0,0]-bids[0,0]) if len(asks) and len(bids) else 0
    buy=num(df.TakerBuy.tail(20).sum()); total=num(df.Volume.tail(20).sum()); sell=max(total-buy,0); flow=buy-sell
    ret=last/prev-1 if prev else 0; rv=num(close.pct_change().tail(30).std()); trend=last/sma-1 if sma else 0
    f.update(spread=spread,spread_pct=spread/last if last else 0,bid_ask_ratio_20=b20/a20 if a20 else 1,bid_ask_ratio_50=b50/a50 if a50 else 1,taker_buy_volume=buy,taker_sell_volume=sell,taker_flow=flow,taker_flow_ratio=flow/total if total else 0,price_return=ret,price_change=last-prev,sma_distance=trend,realized_volatility=rv,BOOK_IMB=o20,QUANT_IMPLY=float(np.tanh((o20+o50+np.tanh(trend*100))/3)),ADAPT_CONF=float(np.clip(.5+(abs(o20)+abs(np.tanh(trend*100)))/2,0,1)),BAYESIAN=float(np.clip(.5+(o20+np.tanh(trend*100))/4,0,1)),FOURIER_TREND=float(np.tanh(close.pct_change().tail(16).mean()*1000)))
    return f


@st.cache_resource(show_spinner=False)
def load_model():
    try:return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
    except Exception:return None


def ml_predict(f):
    model=load_model()
    if model is None:return None,None,"MODEL FILE NOT FOUND"
    try:
        names=[]
        if hasattr(model,"get_booster"):
            names=list(model.get_booster().feature_names or [])
        count=int(getattr(model,"n_features_in_",len(names) or 25))
        cols=names if names and all(x in f or x in ("obi_top20","bid_ask_ratio","total_depth","trend_signal") for x in names) else (F7 if count==7 else F25)
        row=dict(f,obi_top20=f["obi_20"],bid_ask_ratio=f["bid_ask_ratio_20"],total_depth=f["top20_total_depth"],trend_signal=f["sma_distance"])
        x=pd.DataFrame([[row.get(c,0.0) for c in cols]],columns=cols)
        pred=int(model.predict(x)[0]); prob=None
        if hasattr(model,"predict_proba"):
            p=model.predict_proba(x)[0]; prob=float(p[-1])
        return pred,prob,f"OK • {len(cols)} features"
    except Exception as e:return None,None,f"ML ERROR: {type(e).__name__}"


def signal(f,pred,prob):
    research=float(np.clip(.35*f["obi_20"]+.20*f["obi_50"]+.20*f["taker_flow_ratio"]+.15*np.tanh(f["sma_distance"]*100)+.10*f["FOURIER_TREND"],-1,1))
    ml=((prob-.5)*2) if prob is not None else (1 if pred==1 else -1 if pred==0 else 0)
    score=.6*research+.4*ml if pred is not None else research
    s="LONG" if score>=.45 else "SHORT" if score<=-.45 else "WAIT"
    return s,score,float(np.clip(50+abs(score)*49,1,99))


def chart(df,symbol,interval,future):
    fig=go.Figure()
    if df.empty:return fig
    fig.add_trace(go.Candlestick(x=df.Time,open=df.Open,high=df.High,low=df.Low,close=df.Close,name=symbol,increasing_line_color="#28d18d",increasing_fillcolor="#159d6a",decreasing_line_color="#ff7180",decreasing_fillcolor="#d84b60"))
    for span in (20,50,200):
        if len(df)>=span:fig.add_trace(go.Scatter(x=df.Time,y=df.Close.ewm(span=span,adjust=False).mean(),name=f"EMA {span}",mode="lines",line={"width":1}))
    last=df.Time.iloc[-1]
    if len(df)>1:
        step=df.Time.iloc[-1]-df.Time.iloc[-2]
        end=last+step*future
        fig.update_xaxes(range=[df.Time.iloc[max(0,len(df)-220)],end])
    fig.update_layout(template="plotly_dark",height=620,margin=dict(l=5,r=5,t=25,b=5),xaxis_rangeslider_visible=False,dragmode="pan",hovermode="x unified",paper_bgcolor="#0b1119",plot_bgcolor="#0b1119",font=dict(size=11),legend=dict(orientation="h",y=1.02,x=0))
    fig.update_xaxes(showgrid=True,gridcolor="#172230",showspikes=True,spikemode="across",spikesnap="cursor")
    fig.update_yaxes(showgrid=True,gridcolor="#172230",side="right",fixedrange=False)
    return fig


def card(col,label,value,sub=""):
    col.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)


# ---------------- UI ----------------
st.markdown('<div class="hero"><div><div class="brand">ZIA <span>RESEARCH</span> TERMINAL</div><div class="muted">ORDER FLOW • RESEARCH LAB • XGBOOST • LIVE MARKET</div></div><div class="live"><span class="dot"></span>LIVE</div></div>',unsafe_allow_html=True)

with st.sidebar:
    st.header("Terminal")
    symbol=st.selectbox("Symbol",SYMBOLS,index=0)
    tf=st.selectbox("Timeframe",list(TIMEFRAMES),index=3)
    refresh=st.slider("Refresh seconds",2,15,4)
    bars=st.slider("Candles",100,1000,500,50)
    future=st.slider("Future chart space",5,80,25)
    st.caption("Data is public/read-only. No orders are placed by this dashboard.")

# Keep the whole UI refresh seamless when supported. On older Streamlit, the app still works.
def render():
    df,market_source,market_status,market_host=get_candles(symbol,TIMEFRAMES[tf],bars)
    bids,asks,book_source,book_status,book_host=get_order_book(symbol)
    f=features(df,bids,asks)
    pred,prob,ml_status=ml_predict(f)
    sig,score,conf=signal(f,pred,prob)
    price=num(df.Close.iloc[-1]) if not df.empty else 0
    spread=num(asks[0,0]-bids[0,0]) if len(asks) and len(bids) else 0
    o20,b20,a20=ob(bids,asks,20);o50,b50,a50=ob(bids,asks,50)

    controls=st.columns([1.3,1,1,1,1,1])
    card(controls[0],"PRICE",fmt_price(price),f"{symbol} • {market_source}")
    card(controls[1],"SIGNAL",sig,f"Confidence {conf:.1f}%")
    card(controls[2],"ML",f"{prob*100:.1f}%" if prob is not None else "N/A",ml_status)
    card(controls[3],"OBI 20",f"{o20:+.3f}",f"B {b20:,.2f} / A {a20:,.2f}")
    card(controls[4],"OBI 50",f"{o50:+.3f}",f"B {b50:,.2f} / A {a50:,.2f}")
    card(controls[5],"SPREAD",f"{spread:.4f}",f"Book: {book_source}")

    if market_status!="OK":st.warning(f"Market API issue: {market_status}. Trying Binance fallback endpoints automatically.")
    if book_status!="OK":st.error(f"Order book unavailable after all Binance endpoints. Status: {book_status}. Check Streamlit Cloud network access.")

    c1,c2=st.columns([5,1])
    with c1:
        st.markdown(f'<div class="section">{symbol} / {tf} — LIVE CHART</div>',unsafe_allow_html=True)
        st.plotly_chart(chart(df,symbol,tf,future),use_container_width=True,config={"scrollZoom":True,"displaylogo":False,"responsive":True,"modeBarButtonsToAdd":["drawline","drawrect","eraseshape"]})
    with c2:
        st.markdown('<div class="section">ORDER BOOK</div>',unsafe_allow_html=True)
        if len(bids):
            for i in range(min(10,len(bids))):
                st.markdown(f"`{bids[i,0]:,.2f}`  **{bids[i,1]:.4f}**")
            st.divider()
            for i in range(min(10,len(asks))):
                st.markdown(f"`{asks[i,0]:,.2f}`  **{asks[i,1]:.4f}**")
        else: st.info("Waiting for order book data…")

    st.markdown('<div class="section">RESEARCH SNAPSHOT</div>',unsafe_allow_html=True)
    a=st.columns(6)
    card(a[0],"OBI 5",f"{f['obi_5']:+.3f}")
    card(a[1],"OBI 10",f"{f['obi_10']:+.3f}")
    card(a[2],"OBI 20",f"{f['obi_20']:+.3f}")
    card(a[3],"OBI 50",f"{f['obi_50']:+.3f}")
    card(a[4],"TAKER FLOW",f"{f['taker_flow_ratio']:+.3f}")
    card(a[5],"VOLATILITY",f"{f['realized_volatility']:.4%}")

    with st.expander("ML / ENGINE DIAGNOSTICS"):
        st.write({"model_file":str(MODEL_FILE),"model_exists":MODEL_FILE.exists(),"ml":ml_status,"market_source":market_source,"market_host":market_host,"book_source":book_source,"book_host":book_host,"score":round(score,4),"signal":sig})

try:
    fragment=st.fragment
except AttributeError:
    fragment=None

if fragment:
    @fragment(run_every=refresh)
    def live_dashboard():
        render()
    live_dashboard()
else:
    render()
    time.sleep(refresh)
    st.rerun()
