from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# ZIA RESEARCH TERMINAL v8
# UX: static shell + isolated live fragment. No time.sleep/rerun.
# ============================================================
st.set_page_config(page_title="ZIA Research Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "xgboost_obi_model.pkl"
SIGNALS = ROOT / "saved_signals.csv"
SCORES = ROOT / "research_scores.csv"

FUTURES = ["https://fapi.binance.com", "https://fapi1.binance.com", "https://fapi2.binance.com", "https://fapi3.binance.com", "https://fapi4.binance.com"]
SPOT = ["https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com", "https://api3.binance.com"]
DATA = ["https://data-api.binance.vision"]
SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT","TRXUSDT","LTCUSDT"]
TFS = {"5M":"5m","15M":"15m","30M":"30m","1H":"1h","4H":"4h","1D":"1d","1W":"1w"}
F25 = ["top20_bid_sum","top20_ask_sum","obi_5","obi_10","obi_20","obi_50","spread","spread_pct","bid_ask_ratio_20","bid_ask_ratio_50","top20_total_depth","top50_total_depth","taker_buy_volume","taker_sell_volume","taker_flow","taker_flow_ratio","price_return","price_change","sma_distance","realized_volatility","BOOK_IMB","QUANT_IMPLY","ADAPT_CONF","BAYESIAN","FOURIER_TREND"]
F7 = ["top20_bid_sum","top20_ask_sum","obi_top20","spread","bid_ask_ratio","total_depth","trend_signal"]

st.markdown("""
<style>
:root{--bg:#05080d;--panel:#0b1119;--panel2:#101925;--line:#1c2938;--text:#e9eff8;--muted:#738196;--violet:#8b96ff;--green:#47dda0;--red:#ff6f82;--amber:#f4c76a}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(5,8,13,.92)}
.block-container{max-width:1900px;padding:10px clamp(8px,1.7vw,30px) 50px}
.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding:5px 2px 12px;margin-bottom:9px}.brand{font-size:clamp(21px,2.4vw,34px);font-weight:950;letter-spacing:-1.2px}.brand b{color:var(--violet)}.micro{color:var(--muted);font-size:9px;letter-spacing:1.3px;margin-top:2px}.live{border:1px solid #21573e;background:#071810;color:#6ce1a2;border-radius:999px;padding:6px 11px;font-size:10px;font-weight:900}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green);margin-right:6px}
.card{background:linear-gradient(145deg,#111b28,#0a1018);border:1px solid var(--line);border-radius:14px;padding:11px;min-height:78px}.label{font-size:9px;color:var(--muted);font-weight:900;letter-spacing:1.1px}.value{font-size:20px;font-weight:950;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sub{font-size:10px;color:#8592a7;margin-top:3px}.section-title{font-size:16px;font-weight:950;letter-spacing:-.2px;margin:12px 0 6px}.section-sub{font-size:10px;color:var(--muted);margin-bottom:9px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:12px}.signal{font-size:34px;font-weight:1000;letter-spacing:-1px}.good{color:var(--green)}.bad{color:var(--red)}.amber{color:var(--amber)}.violet{color:var(--violet)}
div[data-testid="stTabs"] button{font-weight:900;font-size:11px}.stButton>button{border-radius:10px;font-weight:900}.stDownloadButton>button{border-radius:10px}.stDataFrame{border:1px solid var(--line);border-radius:10px;overflow:hidden}
@media(max-width:700px){.block-container{padding:6px 7px 35px}.brand{font-size:21px}.micro{font-size:7px}.live{font-size:8px;padding:5px 7px}.card{min-height:64px;padding:8px}.value{font-size:16px}.section-title{font-size:14px}.signal{font-size:29px}.stTabs [data-baseweb="tab"]{font-size:9px;padding-left:7px;padding-right:7px}}
</style>
""", unsafe_allow_html=True)


def num(x, default=0.0):
    try:
        v=float(x); return v if np.isfinite(v) else default
    except Exception: return default


def fmt(x):
    x=num(x)
    if abs(x)>=1000:return f"{x:,.2f}"
    if abs(x)>=1:return f"{x:,.4f}"
    return f"{x:,.7f}" if x else "—"


def api(hosts: list[str], path: str, params: dict[str, Any]):
    err="network"
    for host in hosts:
        try:
            r=requests.get(host+path,params=params,timeout=4,headers={"User-Agent":"ZIA-Research/8.0"})
            if r.ok:return r.json(),host,"OK"
            err=f"HTTP {r.status_code}"
        except requests.RequestException as e:err=type(e).__name__
    return None,None,err


@st.cache_data(ttl=2,show_spinner=False)
def get_candles(symbol: str, interval: str, limit: int):
    raw,host,status=api(FUTURES,"/fapi/v1/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1500)}); source="Futures"
    if not isinstance(raw,list): raw,host,status=api(SPOT,"/api/v3/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1000)}); source="Spot"
    if not isinstance(raw,list): raw,host,status=api(DATA,"/api/v3/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1000)}); source="Data API"
    rows=[]
    for c in raw or []:
        try: rows.append([pd.to_datetime(int(c[0]),unit="ms",utc=True),num(c[1]),num(c[2]),num(c[3]),num(c[4]),num(c[5]),num(c[9])])
        except Exception: pass
    return pd.DataFrame(rows,columns=["Time","Open","High","Low","Close","Volume","TakerBuy"]),source,status,host


@st.cache_data(ttl=2,show_spinner=False)
def get_book(symbol: str):
    raw,host,status=api(FUTURES,"/fapi/v1/depth",{"symbol":symbol,"limit":100}); source="Futures"
    if not isinstance(raw,dict) or not raw.get("bids") or not raw.get("asks"): raw,host,status=api(SPOT,"/api/v3/depth",{"symbol":symbol,"limit":100}); source="Spot"
    if not isinstance(raw,dict) or not raw.get("bids") or not raw.get("asks"): raw,host,status=api(DATA,"/api/v3/depth",{"symbol":symbol,"limit":100}); source="Data API"
    try:
        b=np.asarray(raw.get("bids",[]),float); a=np.asarray(raw.get("asks",[]),float)
        if len(b)==0 or len(a)==0: raise ValueError
        return b,a,source,status,host
    except Exception: return np.empty((0,2)),np.empty((0,2)),source,status,host


def obi(b,a,k):
    if len(b)==0 or len(a)==0:return 0.,0.,0.
    k=min(k,len(b),len(a)); bv=float(b[:k,1].sum()); av=float(a[:k,1].sum()); return ((bv-av)/(bv+av) if bv+av else 0.),bv,av


def features(df,b,a):
    f={x:0. for x in F25}
    vals=[obi(b,a,k) for k in (5,10,20,50)]
    (o5,b5,a5),(o10,b10,a10),(o20,b20,a20),(o50,b50,a50)=vals
    f.update(top20_bid_sum=b20,top20_ask_sum=a20,obi_5=o5,obi_10=o10,obi_20=o20,obi_50=o50,top20_total_depth=b20+a20,top50_total_depth=b50+a50)
    if df.empty:return f
    c=df.Close; last=num(c.iloc[-1]); prev=num(c.iloc[-2] if len(c)>1 else last); sma=num(c.rolling(20).mean().iloc[-1],last); ret=last/prev-1 if prev else 0
    spread=num(a[0,0]-b[0,0]) if len(a) and len(b) else 0
    total=num(df.Volume.tail(20).sum()); buy=num(df.TakerBuy.tail(20).sum()); sell=max(total-buy,0); flow=buy-sell
    trend=np.tanh((last/sma-1)*100) if sma else 0; rv=num(c.pct_change().tail(30).std()); four=np.tanh(c.pct_change().tail(16).mean()*1000)
    f.update(spread=spread,spread_pct=spread/last if last else 0,bid_ask_ratio_20=b20/a20 if a20 else 1,bid_ask_ratio_50=b50/a50 if a50 else 1,taker_buy_volume=buy,taker_sell_volume=sell,taker_flow=flow,taker_flow_ratio=flow/total if total else 0,price_return=ret,price_change=last-prev,sma_distance=last/sma-1 if sma else 0,realized_volatility=rv,BOOK_IMB=o20,QUANT_IMPLY=float(np.tanh((o20+o50+trend)/3)),ADAPT_CONF=float(np.clip(.5+(abs(o20)+abs(trend))/2,0,1)),BAYESIAN=float(np.clip(.5+(o20+trend)/4,0,1)),FOURIER_TREND=four)
    return f


@st.cache_resource(show_spinner=False)
def load_model():
    try:return joblib.load(MODEL) if MODEL.exists() else None
    except Exception:return None


def predict(f):
    m=load_model()
    if m is None:return None,None,"MODEL NOT FOUND",0
    try:
        names=list(m.get_booster().feature_names or []) if hasattr(m,"get_booster") else []
        n=int(getattr(m,"n_features_in_",len(names) or 25)); cols=names if names else (F7 if n==7 else F25)
        row=dict(f,obi_top20=f["obi_20"],bid_ask_ratio=f["bid_ask_ratio_20"],total_depth=f["top20_total_depth"],trend_signal=f["sma_distance"])
        x=pd.DataFrame([[row.get(k,0.) for k in cols]],columns=cols)
        p=int(m.predict(x)[0]); pr=float(m.predict_proba(x)[0][-1]) if hasattr(m,"predict_proba") else None
        return p,pr,"OK",len(cols)
    except Exception as e:return None,None,"ML ERROR: "+type(e).__name__,0


def formula_scores(f):
    return {"OBI 20":float(np.clip(f["obi_20"]*2,-1,1)),"OBI 20+50":float(np.clip(f["obi_20"]+f["obi_50"],-1,1)),"OFI":float(np.clip(f["taker_flow_ratio"]*2,-1,1)),"Trend / SMA":float(np.clip(np.tanh(f["sma_distance"]*100),-1,1)),"Fourier":float(np.clip(f["FOURIER_TREND"],-1,1)),"Bayesian":float(np.clip((f["BAYESIAN"]-.5)*2,-1,1)),"Quant Imply":float(np.clip(f["QUANT_IMPLY"],-1,1)),"Adaptive":float(np.clip((f["ADAPT_CONF"]-.5)*2,-1,1))}


def decision(f,pred,prob):
    scores=formula_scores(f); weights={"OBI 20":.22,"OBI 20+50":.14,"OFI":.20,"Trend / SMA":.14,"Fourier":.10,"Bayesian":.08,"Quant Imply":.07,"Adaptive":.05}
    research=sum(scores[k]*weights[k] for k in weights); ml=((prob-.5)*2) if prob is not None else (1 if pred==1 else -1 if pred==0 else 0); score=.60*research+.40*ml if pred is not None else research
    signal="LONG" if score>=.45 else "SHORT" if score<=-.45 else "WAIT"; confidence=float(np.clip(50+abs(score)*49,1,99))
    return signal,score,confidence,scores,research,ml,weights


def tri_periods(tf):
    if tf=="15M":return {"4H":"4h","1H":"1h"}
    if tf in ("1H","4H"):return {"DAY":"1d","WEEK":"1w","MONTH":"1M"}
    return {}


@st.cache_data(ttl=30,show_spinner=False)
def tri_levels(symbol,tf):
    out={}
    for name,interval in tri_periods(tf).items():
        d,_,_,_=get_candles(symbol,interval,5)
        if len(d)<2:continue
        c=d.iloc[-2];o,h,l,cl=map(num,[c.Open,c.High,c.Low,c.Close]); hi=max(o,cl); lo=min(o,cl)
        out[name]={"BODY 50":(hi+lo)/2,"UPPER 50":(h+hi)/2,"LOWER 50":(l+lo)/2}
    return out


def make_chart(df,symbol,tf,future,show_tri):
    fig=go.Figure()
    if df.empty:return fig
    fig.add_trace(go.Candlestick(x=df.Time,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="PRICE",increasing_line_color="#47dda0",increasing_fillcolor="#198e64",decreasing_line_color="#ff6f82",decreasing_fillcolor="#bd465b"))
    for span in (20,50,200):
        if len(df)>=span:fig.add_trace(go.Scatter(x=df.Time,y=df.Close.ewm(span=span,adjust=False).mean(),name=f"EMA {span}",mode="lines",line={"width":1}))
    if show_tri:
        for period,levels in tri_levels(symbol,tf).items():
            for label,dash in (("BODY 50","solid"),("UPPER 50","dot"),("LOWER 50","dot")):
                if label in levels:fig.add_hline(y=levels[label],line_dash=dash,line_width=1,opacity=.75,annotation_text=f"TRI {period} • {label}",annotation_position="top right")
    step=df.Time.iloc[-1]-df.Time.iloc[-2] if len(df)>1 else pd.Timedelta(minutes=5)
    start=max(0,len(df)-300); fig.update_xaxes(range=[df.Time.iloc[start],df.Time.iloc[-1]+step*future],showgrid=True,gridcolor="#172230",showspikes=True,spikemode="across")
    fig.update_yaxes(side="right",showgrid=True,gridcolor="#172230")
    fig.update_layout(template="plotly_dark",height=660,margin=dict(l=4,r=4,t=25,b=4),xaxis_rangeslider_visible=False,dragmode="pan",hovermode="x unified",paper_bgcolor="#0b1119",plot_bgcolor="#0b1119",legend=dict(orientation="h",y=1.02,x=0))
    return fig


def card(c,label,value,sub=""):
    c.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)


def append_csv(path,row):
    exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as fp:
        w=csv.DictWriter(fp,fieldnames=list(row.keys()))
        if not exists:w.writeheader()
        w.writerow(row)


def read_signals():
    if not SIGNALS.exists():return pd.DataFrame()
    try:return pd.read_csv(SIGNALS)
    except Exception:return pd.DataFrame()


def history_stats(d):
    if d.empty or "result" not in d:return 0,0,0
    r=d.result.astype(str).str.upper(); wins=int((r=="WIN").sum()); losses=int((r=="LOSS").sum()); total=wins+losses
    return wins,losses,(wins/total*100 if total else 0)


# ---------- persistent UI state ----------
if "symbol" not in st.session_state: st.session_state.symbol="BTCUSDT"
if "tf" not in st.session_state: st.session_state.tf="15M"
if "refresh" not in st.session_state: st.session_state.refresh=4

with st.sidebar:
    st.markdown("## ⚡ ZIA RESEARCH")
    st.selectbox("Market",SYMBOLS,key="symbol")
    st.selectbox("Timeframe",list(TFS),key="tf")
    st.slider("Live interval",2,15,key="refresh")
    st.caption("Auto-update runs only inside the live fragment. The full page is not periodically rerun.")
    if st.button("Clear Streamlit cache",use_container_width=True):
        st.cache_data.clear(); st.toast("Live data cache cleared")

symbol=st.session_state.symbol; tf=st.session_state.tf; refresh=st.session_state.refresh

st.markdown('<div class="hero"><div><div class="brand">ZIA <b>RESEARCH</b> TERMINAL</div><div class="micro">QUANT MARKET INTELLIGENCE • ML • ORDER FLOW • RESEARCH LAB</div></div><div class="live"><span class="dot"></span>LIVE</div></div>',unsafe_allow_html=True)

# A lightweight live strip. This is the only automatically rerunning area.
@st.fragment(run_every=f"{refresh}s")
def live_strip():
    df,ms,msv,mh=get_candles(symbol,TFS[tf],160); b,a,bs,bsv,bh=get_book(symbol); f=features(df,b,a); pred,prob,mls,n=predict(f); sig,score,conf,_,_,_,_=decision(f,pred,prob); price=num(df.Close.iloc[-1]) if not df.empty else 0; o20,b20,a20=obi(b,a,20); o50,b50,a50=obi(b,a,50)
    cols=st.columns(7)
    card(cols[0],"PRICE",fmt(price),symbol); card(cols[1],"SIGNAL",sig,f"Confidence {conf:.1f}%"); card(cols[2],"ML",f"{prob*100:.1f}%" if prob is not None else "—",mls); card(cols[3],"OBI 20",f"{o20:+.3f}","top 20"); card(cols[4],"OBI 50",f"{o50:+.3f}","top 50"); card(cols[5],"DEPTH",f"{b20+a20:,.0f}",bs); card(cols[6],"ENGINE",msv,f"{ms} • {mh or '—'}")

live_strip()

st.markdown('<div class="section-title">WORKSPACE</div><div class="section-sub">Each section has one job. Live updates are isolated from navigation and the rest of the terminal.</div>',unsafe_allow_html=True)

t_over, t_chart, t_flow, t_ml, t_research, t_trades = st.tabs(["⌂ OVERVIEW","◈ CHART","◌ ORDER FLOW","🧠 ML LAB","🔬 RESEARCH LAB","▣ TRADE JOURNAL"])

# ---------------- OVERVIEW ----------------
with t_over:
    df,ms,msv,mh=get_candles(symbol,TFS[tf],300); b,a,bs,bsv,bh=get_book(symbol); f=features(df,b,a); pred,prob,mls,n=predict(f); sig,score,conf,fs,research,mlscore,weights=decision(f,pred,prob); price=num(df.Close.iloc[-1]) if not df.empty else 0; ofi=float((f["taker_flow"] if f else 0));
    a1,a2=st.columns([1.35,1])
    with a1:
        st.markdown("### Market Command Center")
        q=st.columns(4); card(q[0],"FINAL BIAS",sig,f"score {score:+.3f}"); card(q[1],"CONFIDENCE",f"{conf:.1f}%","research + ML blend"); card(q[2],"VOLATILITY",f"{f['realized_volatility']*100:.3f}%","30-candle realized"); card(q[3],"TAKER FLOW",f"{f['taker_flow_ratio']:+.3f}","buy/sell proxy")
        st.markdown("#### Market Readout")
        regime="TRENDING" if abs(f["sma_distance"])>.003 else "RANGING"; flow="BUY PRESSURE" if f["taker_flow_ratio"]>.05 else "SELL PRESSURE" if f["taker_flow_ratio"]<-.05 else "BALANCED"
        st.markdown(f'<div class="panel"><b>{regime}</b> • {flow}<br><span style="color:#8290a5">OBI20 {f["obi_20"]:+.3f} • OBI50 {f["obi_50"]:+.3f} • spread {fmt(f["spread"])} • SMA distance {f["sma_distance"]*100:+.3f}%</span></div>',unsafe_allow_html=True)
    with a2:
        st.markdown("### Signal Strength")
        gauge=go.Figure(go.Indicator(mode="gauge+number",value=conf,gauge={"axis":{"range":[0,100]},"bar":{"color":"#8b96ff"},"steps":[{"range":[0,45],"color":"#29151a"},{"range":[45,65],"color":"#29261b"},{"range":[65,100],"color":"#10271e"}]}))
        gauge.update_layout(height=230,margin=dict(l=15,r=15,t=20,b=10),paper_bgcolor="#0b1119",font={"color":"#e9eff8"})
        st.plotly_chart(gauge,use_container_width=True,config={"displayModeBar":False})

# ---------------- CHART ----------------
with t_chart:
    c1,c2,c3=st.columns([1,1,1]); bars=c1.selectbox("History",[200,300,500,800,1200],index=2); future=c2.slider("Future space",5,100,25); tri=c3.toggle("TRI lines",True)
    df,ms,msv,mh=get_candles(symbol,TFS[tf],bars)
    st.plotly_chart(make_chart(df,symbol,tf,future,tri),use_container_width=True,config={"scrollZoom":True,"displaylogo":False,"modeBarButtonsToAdd":["drawline","drawrect","eraseshape"]},key="zia_chart")
    st.caption("TRI mapping: 15M → 4H + 1H. 1H/4H → Day + Week + Month. BODY 50 = candle-body midpoint; UPPER/LOWER 50 = wick midpoints.")
    levels=tri_levels(symbol,tf)
    if levels:
        rows=[]
        for p,lv in levels.items(): rows.append([p,lv.get("BODY 50"),lv.get("UPPER 50"),lv.get("LOWER 50")])
        st.dataframe(pd.DataFrame(rows,columns=["Reference","BODY 50","UPPER 50","LOWER 50"]).style.format({"BODY 50":"{:,.4f}","UPPER 50":"{:,.4f}","LOWER 50":"{:,.4f}"}),use_container_width=True,hide_index=True)

# ---------------- ORDER FLOW ----------------
with t_flow:
    df,_,_,_=get_candles(symbol,TFS[tf],200); b,a,src,status,host=get_book(symbol); f=features(df,b,a)
    st.markdown("### Order Book Intelligence")
    cols=st.columns(4)
    for i,k in enumerate((5,10,20,50)):
        o,bb,aa=obi(b,a,k); card(cols[i],f"OBI TOP {k}",f"{o:+.3f}",f"bid {bb:,.0f} • ask {aa:,.0f}")
    left,right=st.columns([1.2,1])
    with left:
        levels=[]
        for i in range(min(15,len(b),len(a))): levels.append([i+1,b[i,0],b[i,1],a[i,0],a[i,1]])
        st.dataframe(pd.DataFrame(levels,columns=["Level","Bid price","Bid qty","Ask price","Ask qty"]).style.format({"Bid price":"{:,.4f}","Bid qty":"{:,.2f}","Ask price":"{:,.4f}","Ask qty":"{:,.2f}"}),use_container_width=True,hide_index=True)
    with right:
        st.markdown("#### Flow Diagnostics")
        st.metric("Taker flow ratio",f"{f['taker_flow_ratio']:+.4f}"); st.metric("Top 20 depth",f"{f['top20_total_depth']:,.0f}"); st.metric("Top 50 depth",f"{f['top50_total_depth']:,.0f}"); st.metric("Spread",fmt(f['spread']))
        st.caption(f"Source: {src} • {status} • {host or 'unavailable'}")

# ---------------- ML LAB ----------------
with t_ml:
    df,_,_,_=get_candles(symbol,TFS[tf],300); b,a,_,_,_=get_book(symbol); f=features(df,b,a); pred,prob,status,n=predict(f)
    st.markdown("### Live Machine Learning Lab")
    q=st.columns(4); card(q[0],"MODEL",MODEL.name if MODEL.exists() else "MISSING",status); card(q[1],"PREDICTION",str(pred) if pred is not None else "—","0=down • 1=up for binary model"); card(q[2],"PROBABILITY",f"{prob*100:.2f}%" if prob is not None else "—","predict_proba"); card(q[3],"FEATURES",str(n),"model input count")
    row=dict(f,obi_top20=f["obi_20"],bid_ask_ratio=f["bid_ask_ratio_20"],total_depth=f["top20_total_depth"],trend_signal=f["sma_distance"])
    st.markdown("#### Live Feature Vector")
    vals=pd.DataFrame({"Feature":list(row.keys()),"Value":[num(v) for v in row.values()]})
    st.dataframe(vals.style.format({"Value":"{:+.6f}"}),use_container_width=True,hide_index=True)

# ---------------- RESEARCH LAB ----------------
with t_research:
    df,_,_,_=get_candles(symbol,TFS[tf],300); b,a,_,_,_=get_book(symbol); f=features(df,b,a); pred,prob,_,_=predict(f); sig,score,conf,fs,research,mlscore,weights=decision(f,pred,prob)
    st.markdown("### Research Lab • Formula Observatory")
    rows=[[k,v,weights[k]*100,v*weights[k]] for k,v in fs.items()]; rdf=pd.DataFrame(rows,columns=["Formula","Live score","Weight %","Contribution"]).sort_values("Contribution",ascending=False)
    left,right=st.columns([1.25,1])
    with left: st.dataframe(rdf.style.format({"Live score":"{:+.3f}","Weight %":"{:.1f}%","Contribution":"{:+.3f}"}),use_container_width=True,hide_index=True)
    with right:
        fig=go.Figure(go.Bar(x=rdf["Live score"],y=rdf["Formula"],orientation="h")); fig.update_layout(template="plotly_dark",height=330,margin=dict(l=5,r=5,t=20,b=5),paper_bgcolor="#0b1119",plot_bgcolor="#0b1119",xaxis=dict(range=[-1,1],zeroline=True,zerolinecolor="#526070"),showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    best=rdf.iloc[0]["Formula"] if not rdf.empty else "—"
    st.markdown(f'<div class="panel"><b>Current strongest contributor:</b> <span class="violet">{best}</span> • Research composite <b>{research:+.3f}</b> • ML contribution <b>{mlscore:+.3f}</b> • Final <b>{sig}</b> at <b>{conf:.1f}%</b><br><span style="color:#77869a">Weights are research-engine defaults. Historical accuracy should be used before treating a formula as superior.</span></div>',unsafe_allow_html=True)

# ---------------- TRADE JOURNAL ----------------
with t_trades:
    st.markdown("### Trade Journal")
    hist=read_signals(); wins,losses,wr=history_stats(hist); total=wins+losses
    q=st.columns(4); card(q[0],"TOTAL CLOSED",str(total),"resolved trades"); card(q[1],"WINS",str(wins),"marked WIN"); card(q[2],"LOSSES",str(losses),"marked LOSS"); card(q[3],"WIN RATE",f"{wr:.1f}%","closed trades only")
    act1,act2,act3=st.columns([1,1,2])
    with act1:
        if st.button("✅ MARK LAST PENDING WIN",use_container_width=True):
            if not hist.empty:
                idx=hist.index[hist.get("result",pd.Series("PENDING",index=hist.index)).astype(str).str.upper().eq("PENDING")]
                if len(idx): hist.loc[idx[-1],"result"]="WIN"; hist.to_csv(SIGNALS,index=False); st.rerun()
    with act2:
        if st.button("❌ MARK LAST PENDING LOSS",use_container_width=True):
            if not hist.empty:
                idx=hist.index[hist.get("result",pd.Series("PENDING",index=hist.index)).astype(str).str.upper().eq("PENDING")]
                if len(idx): hist.loc[idx[-1],"result"]="LOSS"; hist.to_csv(SIGNALS,index=False); st.rerun()
    with act3:
        st.caption("Signals are saved as journal records. Marking WIN/LOSS resolves them for the win-rate panel.")
    if not hist.empty:
        cols=[c for c in ["timestamp","symbol","timeframe","signal","price","confidence","ml_probability","obi20","obi50","ofi","research_score","result"] if c in hist.columns]
        st.dataframe(hist[cols].iloc[::-1].head(200),use_container_width=True,hide_index=True)
        d1,d2=st.columns(2)
        with d1: st.download_button("⬇️ DOWNLOAD TRADE HISTORY",hist.to_csv(index=False),"trade_history.csv","text/csv",use_container_width=True)
        with d2:
            if st.button("🗑️ CLEAR TRADE HISTORY",use_container_width=True):
                SIGNALS.unlink(missing_ok=True); st.toast("Trade history cleared"); st.rerun()
    else: st.info("No saved signals yet. Save a signal from the Overview section when you want to create a journal record.")

st.markdown('<div class="micro" style="text-align:center;margin-top:18px">ZIA RESEARCH • READ-ONLY MARKET ANALYTICS • LIVE DATA UPDATES ARE ISOLATED FROM THE STATIC UI</div>',unsafe_allow_html=True)
