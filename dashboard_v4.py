from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="ZIA Research Terminal", page_icon="", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "xgboost_obi_model.pkl"
FUTURES = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT","TRXUSDT","LTCUSDT"]
TFS = {"5M":"5m","15M":"15m","30M":"30m","1H":"1h","4H":"4h","1D":"1d","1W":"1w"}
TRI_TFS = {"4H":"4h", "1H":"1h", "DAY":"1d", "WEEK":"1w", "MONTH":"1M"}

# Plain monochrome UI: no colored cards, badges, dots or decorative gradients.
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#000;color:#fff}
[data-testid="stHeader"]{background:#000}
.block-container{max-width:1900px;padding:10px 18px 40px}
.card,.panel{background:#000;border:1px solid #303030;border-radius:8px;padding:12px}
.brand{font-size:28px;font-weight:800;letter-spacing:-.5px}.muted{color:#999;font-size:11px}
.value{font-size:20px;font-weight:700}.section-title{font-size:15px;font-weight:800;margin:12px 0 7px}
div[data-testid="stTabs"] button{font-weight:700}.stButton>button,.stSelectbox>div>div,.stMultiSelect>div>div{background:#000!important;color:#fff!important;border-color:#444!important}
[data-testid="stMetric"]{background:#000!important;border:1px solid #303030;padding:8px;border-radius:8px}
</style>
""", unsafe_allow_html=True)


def api(path: str, params: dict[str, Any]):
    try:
        r = requests.get(FUTURES + path, params=params, timeout=4)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=3, show_spinner=False)
def candles(symbol: str, interval: str, limit: int = 1000):
    raw = api("/fapi/v1/klines", {"symbol":symbol,"interval":interval,"limit":min(limit,1500)})
    rows=[]
    for c in raw or []:
        try: rows.append([pd.to_datetime(int(c[0]),unit="ms",utc=True),float(c[1]),float(c[2]),float(c[3]),float(c[4]),float(c[5])])
        except (TypeError,ValueError,IndexError): pass
    return pd.DataFrame(rows, columns=["Time","Open","High","Low","Close","Volume"])


def tri_levels(symbol: str, selected: list[str]):
    out=[]
    for tf_name in selected:
        interval=TRI_TFS[tf_name]
        d=candles(symbol,interval,5)
        if len(d)<2: continue
        c=d.iloc[-2]
        o,h,l,cl=map(float,[c.Open,c.High,c.Low,c.Close])
        body_hi=max(o,cl); body_lo=min(o,cl)
        out.extend([(f"TRI {tf_name} • BODY 50",(body_hi+body_lo)/2,"solid"),(f"TRI {tf_name} • UPPER 50",(h+body_hi)/2,"dot"),(f"TRI {tf_name} • LOWER 50",(l+body_lo)/2,"dot")])
    return out


def chart(df: pd.DataFrame, symbol: str, selected_tri: list[str], future: int):
    fig=go.Figure()
    if df.empty:return fig
    fig.add_trace(go.Candlestick(x=df.Time,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="PRICE",increasing_line_color="#fff",increasing_fillcolor="#fff",decreasing_line_color="#777",decreasing_fillcolor="#777"))
    for span in (20,50,200):
        if len(df)>=span:
            fig.add_trace(go.Scatter(x=df.Time,y=df.Close.ewm(span=span,adjust=False).mean(),name=f"EMA {span}",mode="lines",line={"width":1,"color":"#aaa"}))
    for label,level,dash in tri_levels(symbol,selected_tri):
        fig.add_hline(y=level,line_dash=dash,line_width=1,line_color="#aaa",annotation_text=label,annotation_position="top right")
    step=df.Time.iloc[-1]-df.Time.iloc[-2] if len(df)>1 else pd.Timedelta(minutes=5)
    start=max(0,len(df)-500)
    fig.update_xaxes(range=[df.Time.iloc[start],df.Time.iloc[-1]+step*future],rangeslider_visible=False,showgrid=True,gridcolor="#1b1b1b",showspikes=True,spikemode="across",fixedrange=False)
    fig.update_yaxes(side="right",showgrid=True,gridcolor="#1b1b1b",fixedrange=False)
    fig.update_layout(template="plotly_dark",paper_bgcolor="#000",plot_bgcolor="#000",font={"color":"#fff"},height=690,margin=dict(l=5,r=55,t=25,b=10),hovermode="x unified",uirevision=f"{symbol}-{st.session_state.get('tf','5M')}" )
    return fig


@st.cache_resource(show_spinner=False)
def model():
    try:return joblib.load(MODEL) if MODEL.exists() else None
    except Exception:return None


# Controls stay outside the auto-refresh fragment so changing them does not fight refresh state.
if "symbol" not in st.session_state: st.session_state.symbol="BTCUSDT"
if "tf" not in st.session_state: st.session_state.tf="5M"
if "tri" not in st.session_state: st.session_state.tri=[]

st.markdown('<div class="brand">ZIA RESEARCH TERMINAL</div><div class="muted">Live market research • monochrome interface</div>',unsafe_allow_html=True)

c1,c2,c3,c4=st.columns([1.4,1,1.8,1])
with c1: symbol=st.selectbox("Symbol",SYMBOLS,index=SYMBOLS.index(st.session_state.symbol))
with c2: tf=st.selectbox("Chart Timeframe",list(TFS),index=list(TFS).index(st.session_state.tf))
with c3: tri=st.multiselect("TRI line timeframes",list(TRI_TFS),default=st.session_state.tri,help="Select only the TRI timeframes whose lines you want on the chart.")
with c4: future=st.slider("Future space",10,300,80)
st.session_state.symbol=symbol; st.session_state.tf=tf; st.session_state.tri=tri

st.markdown('<div class="section-title">TRI LINE TIMEFRAMES</div>',unsafe_allow_html=True)
st.caption("Tick a timeframe and its TRI Body 50 / Upper 50 / Lower 50 lines will appear. No other TRI controls.")

# Seamless polling: update only this live area. No st.rerun(), no sleep(), no page-level refresh.
@st.fragment(run_every=3)
def live_panel():
    df=candles(symbol,TFS[tf],1000)
    if df.empty:
        st.warning("Market data unavailable")
        return
    last=float(df.Close.iloc[-1]); prev=float(df.Close.iloc[-2]) if len(df)>1 else last
    delta=last-prev
    a,b,c=st.columns(3)
    a.metric("PRICE",f"{last:,.4f}",f"{delta:+,.4f}")
    b.metric("CANDLES",len(df))
    c.metric("LAST UPDATE","LIVE")
    st.plotly_chart(chart(df,symbol,tri,future),use_container_width=True,config={"scrollZoom":True,"displaylogo":False,"modeBarButtonsToRemove":["lasso2d","select2d"]})

live_panel()

st.markdown('<div class="section-title">RESEARCH</div>',unsafe_allow_html=True)
with st.expander("ML model status",expanded=False):
    m=model()
    st.write("XGBoost model:", "loaded" if m is not None else "not found")
    st.write("Model file:", MODEL.name)
