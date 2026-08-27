from __future__ import annotations

import csv
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

st.set_page_config(page_title="ZIA Research Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "xgboost_obi_model.pkl"
SIGNAL_FILE = ROOT / "saved_signals.csv"
SCORE_FILE = ROOT / "research_scores.csv"

FUTURES = ["https://fapi.binance.com","https://fapi1.binance.com","https://fapi2.binance.com","https://fapi3.binance.com","https://fapi4.binance.com"]
SPOT = ["https://api.binance.com","https://api1.binance.com","https://api2.binance.com","https://api3.binance.com"]
DATA = ["https://data-api.binance.vision"]
SYMBOLS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT"]
TFS={"5M":"5m","15M":"15m","30M":"30m","1H":"1h","4H":"4h","1D":"1d","1W":"1w"}
F25=["top20_bid_sum","top20_ask_sum","obi_5","obi_10","obi_20","obi_50","spread","spread_pct","bid_ask_ratio_20","bid_ask_ratio_50","top20_total_depth","top50_total_depth","taker_buy_volume","taker_sell_volume","taker_flow","taker_flow_ratio","price_return","price_change","sma_distance","realized_volatility","BOOK_IMB","QUANT_IMPLY","ADAPT_CONF","BAYESIAN","FOURIER_TREND"]
F7=["top20_bid_sum","top20_ask_sum","obi_top20","spread","bid_ask_ratio","total_depth","trend_signal"]

st.markdown("""
<style>
html,body,[data-testid=stAppViewContainer]{background:#060a10;color:#e9eef7}
.block-container{max-width:1900px;padding:9px clamp(7px,1.5vw,28px) 55px}
[data-testid=stHeader]{background:transparent}.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1d2938;padding:4px 2px 11px;margin-bottom:10px}.brand{font-size:clamp(20px,2.5vw,32px);font-weight:950;letter-spacing:-1px}.brand span{color:#8792ff}.muted{font-size:9px;color:#748298;letter-spacing:1px}.live{border:1px solid #22583f;background:#091711;color:#70e0a5;border-radius:999px;padding:6px 10px;font-size:10px;font-weight:900}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#48dc8e;box-shadow:0 0 9px #48dc8e;margin-right:6px}
.card{background:linear-gradient(145deg,#101824,#0b1119);border:1px solid #202d3e;border-radius:13px;padding:10px;min-height:72px}.label{font-size:9px;color:#77879d;font-weight:900;letter-spacing:1px}.value{font-size:18px;font-weight:950;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sub{font-size:10px;color:#8491a5;margin-top:3px}.panel{background:#0b1119;border:1px solid #1d2939;border-radius:14px;padding:10px}.section{font-size:14px;font-weight:950;margin:12px 0 7px}.small{font-size:10px;color:#748399}.good{color:#65dfa0}.bad{color:#ff7787}.accent{color:#8d98ff}.sig{font-size:31px;font-weight:950}
@media(max-width:700px){.block-container{padding:6px 7px 40px}.brand{font-size:20px}.muted{font-size:7px}.live{font-size:8px;padding:5px 7px}.card{min-height:62px;padding:8px}.value{font-size:15px}.sub{font-size:9px}.sig{font-size:27px}}
</style>
""",unsafe_allow_html=True)

def num(x,d=0.0):
    try:
        x=float(x); return x if np.isfinite(x) else d
    except Exception:return d

def fmt(x):
    x=num(x)
    return f"{x:,.2f}" if x>=1000 else f"{x:,.4f}" if x>=1 else (f"{x:,.7f}" if x else "—")

def request_json(hosts,path,params):
    err="network"
    for h in hosts:
        try:
            r=requests.get(h+path,params=params,timeout=5,headers={"User-Agent":"ZIA-Research/6.0"})
            if r.ok:return r.json(),h,"OK"
            err=f"HTTP {r.status_code}"
        except requests.RequestException as e:err=type(e).__name__
    return None,None,err

@st.cache_data(ttl=2,show_spinner=False)
def candles(symbol,interval,limit):
    raw,h,s=request_json(FUTURES,"/fapi/v1/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1500)})
    source="Futures"
    if not isinstance(raw,list):raw,h,s=request_json(SPOT,"/api/v3/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1000)});source="Spot"
    if not isinstance(raw,list):raw,h,s=request_json(DATA,"/api/v3/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1000)});source="Data API"
    rows=[]
    for c in raw or []:
        try:rows.append([pd.to_datetime(int(c[0]),unit="ms",utc=True),num(c[1]),num(c[2]),num(c[3]),num(c[4]),num(c[5]),num(c[9])])
        except Exception:pass
    return pd.DataFrame(rows,columns=["Time","Open","High","Low","Close","Volume","TakerBuy"]),source,s,h

@st.cache_data(ttl=2,show_spinner=False)
def book(symbol):
    raw,h,s=request_json(FUTURES,"/fapi/v1/depth",{"symbol":symbol,"limit":100});source="Futures"
    if not isinstance(raw,dict) or not raw.get("bids") or not raw.get("asks"):raw,h,s=request_json(SPOT,"/api/v3/depth",{"symbol":symbol,"limit":100});source="Spot"
    if not isinstance(raw,dict) or not raw.get("bids") or not raw.get("asks"):raw,h,s=request_json(DATA,"/api/v3/depth",{"symbol":symbol,"limit":100});source="Data API"
    try:
        b=np.asarray(raw.get("bids",[]),float);a=np.asarray(raw.get("asks",[]),float)
        if len(b)==0 or len(a)==0:raise ValueError()
        return b,a,source,"OK",h
    except Exception:return np.empty((0,2)),np.empty((0,2)),source,s,h

def obi(b,a,k):
    if len(b)==0 or len(a)==0:return 0.,0.,0.
    k=min(k,len(b),len(a));bs=float(b[:k,1].sum());as_=float(a[:k,1].sum());return ((bs-as_)/(bs+as_) if bs+as_ else 0.),bs,as_

def get_features(df,b,a):
    f={x:0. for x in F25};o5,b5,a5=obi(b,a,5);o10,b10,a10=obi(b,a,10);o20,b20,a20=obi(b,a,20);o50,b50,a50=obi(b,a,50)
    f.update(top20_bid_sum=b20,top20_ask_sum=a20,obi_5=o5,obi_10=o10,obi_20=o20,obi_50=o50,top20_total_depth=b20+a20,top50_total_depth=b50+a50)
    if df.empty:return f
    c=df.Close;last=num(c.iloc[-1]);prev=num(c.iloc[-2] if len(c)>1 else last);sma=num(c.rolling(20).mean().iloc[-1],last);ret=last/prev-1 if prev else 0
    spread=num(a[0,0]-b[0,0]) if len(a) and len(b) else 0
    total=num(df.Volume.tail(20).sum());buy=num(df.TakerBuy.tail(20).sum());sell=max(total-buy,0);flow=buy-sell
    trend=np.tanh((last/sma-1)*100) if sma else 0;rv=num(c.pct_change().tail(30).std());four=np.tanh(c.pct_change().tail(16).mean()*1000)
    f.update(spread=spread,spread_pct=spread/last if last else 0,bid_ask_ratio_20=b20/a20 if a20 else 1,bid_ask_ratio_50=b50/a50 if a50 else 1,taker_buy_volume=buy,taker_sell_volume=sell,taker_flow=flow,taker_flow_ratio=flow/total if total else 0,price_return=ret,price_change=last-prev,sma_distance=last/sma-1 if sma else 0,realized_volatility=rv,BOOK_IMB=o20,QUANT_IMPLY=float(np.tanh((o20+o50+trend)/3)),ADAPT_CONF=float(np.clip(.5+(abs(o20)+abs(trend))/2,0,1)),BAYESIAN=float(np.clip(.5+(o20+trend)/4,0,1)),FOURIER_TREND=four)
    return f

@st.cache_resource(show_spinner=False)
def model():
    try:return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
    except Exception:return None

def ml(f):
    m=model()
    if m is None:return None,None,"MODEL NOT FOUND",0
    try:
        names=list(m.get_booster().feature_names or []) if hasattr(m,"get_booster") else []
        count=int(getattr(m,"n_features_in_",len(names) or 25));cols=names if names and all(k in f or k in {"obi_top20","bid_ask_ratio","total_depth","trend_signal"} for k in names) else (F7 if count==7 else F25)
        row=dict(f,obi_top20=f["obi_20"],bid_ask_ratio=f["bid_ask_ratio_20"],total_depth=f["top20_total_depth"],trend_signal=f["sma_distance"])
        x=pd.DataFrame([[row.get(k,0.) for k in cols]],columns=cols);p=int(m.predict(x)[0]);prob=None
        if hasattr(m,"predict_proba"):prob=float(m.predict_proba(x)[0][-1])
        return p,prob,"OK",len(cols)
    except Exception as e:return None,None,"ML ERROR: "+type(e).__name__,0

def formula_scores(f):
    # Independent normalized scores. These are research diagnostics, not guarantees.
    return {
        "OBI":float(np.clip(f["obi_20"]*2,-1,1)),
        "OBI+50":float(np.clip((f["obi_20"]+f["obi_50"]),-1,1)),
        "OFI":float(np.clip(f["taker_flow_ratio"]*2,-1,1)),
        "Trend":float(np.clip(np.tanh(f["sma_distance"]*100),-1,1)),
        "Fourier":float(np.clip(f["FOURIER_TREND"],-1,1)),
        "Bayesian":float(np.clip((f["BAYESIAN"]-.5)*2,-1,1)),
        "Quant Imply":float(np.clip(f["QUANT_IMPLY"],-1,1)),
        "Adaptive":float(np.clip((f["ADAPT_CONF"]-.5)*2,-1,1)),
    }

def combined(f,pred,prob):
    s=formula_scores(f);research=.25*s["OBI"]+.18*s["OBI+50"]+.20*s["OFI"]+.15*s["Trend"]+.10*s["Fourier"]+.12*s["Quant Imply"]
    mlscore=((prob-.5)*2) if prob is not None else (1 if pred==1 else -1 if pred==0 else 0)
    score=.60*research+.40*mlscore if pred is not None else research
    sig="LONG" if score>=.45 else "SHORT" if score<=-.45 else "WAIT"
    return sig,float(score),float(np.clip(50+abs(score)*49,1,99)),s,research,mlscore

def ofi_now(b,a):
    prev=st.session_state.get("prev_book")
    st.session_state["prev_book"]=(b.copy(),a.copy())
    if prev is None:return 0.
    pb,pa=prev;_,bc,ac=obi(b,a,20);_,pbc,pac=obi(pb,pa,20)
    return (bc-pbc)-(ac-pac)

def save_signal(row):
    exists=SIGNAL_FILE.exists();
    with SIGNAL_FILE.open("a",newline="",encoding="utf-8") as fp:
        w=csv.DictWriter(fp,fieldnames=list(row));
        if not exists:w.writeheader()
        w.writerow(row)

def save_scores(row):
    exists=SCORE_FILE.exists();
    with SCORE_FILE.open("a",newline="",encoding="utf-8") as fp:
        w=csv.DictWriter(fp,fieldnames=list(row));
        if not exists:w.writeheader()
        w.writerow(row)

def card(c,label,value,sub=""):
    c.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)

def chart(df,future):
    fig=go.Figure()
    if df.empty:return fig
    fig.add_trace(go.Candlestick(x=df.Time,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="Price",increasing_line_color="#2bd28f",increasing_fillcolor="#159d6a",decreasing_line_color="#ff7180",decreasing_fillcolor="#d84b60"))
    for span in (20,50,200):
        if len(df)>=span:fig.add_trace(go.Scatter(x=df.Time,y=df.Close.ewm(span=span,adjust=False).mean(),name=f"EMA {span}",mode="lines",line={"width":1}))
    step=df.Time.iloc[-1]-df.Time.iloc[-2] if len(df)>1 else pd.Timedelta(minutes=5)
    fig.update_xaxes(range=[df.Time.iloc[max(0,len(df)-220)],df.Time.iloc[-1]+step*future],showgrid=True,gridcolor="#172230",showspikes=True,spikemode="across",spikesnap="cursor")
    fig.update_yaxes(side="right",showgrid=True,gridcolor="#172230")
    fig.update_layout(template="plotly_dark",height=620,margin=dict(l=4,r=4,t=28,b=4),xaxis_rangeslider_visible=False,dragmode="pan",hovermode="x unified",paper_bgcolor="#0b1119",plot_bgcolor="#0b1119",legend=dict(orientation="h",y=1.02,x=0))
    return fig

def leaderboard():
    if not SCORE_FILE.exists():return pd.DataFrame()
    try:
        d=pd.read_csv(SCORE_FILE);d["close"]=pd.to_numeric(d["close"],errors="coerce");d["future_close"]=d.groupby("symbol")["close"].shift(-5);d["ret"]=d["future_close"]/d["close"]-1
        out=[]
        names=["OBI","OBI+50","OFI","Trend","Fourier","Bayesian","Quant Imply","Adaptive"]
        for x in names:
            sc=pd.to_numeric(d[x],errors="coerce");direction=np.sign(sc);actual=np.sign(d.ret);mask=(sc.abs()>=.15)&d.ret.notna();n=int(mask.sum());acc=float((direction[mask]==actual[mask]).mean()*100) if n else np.nan;out.append([x,n,acc])
        return pd.DataFrame(out,columns=["Formula","Samples","Directional accuracy %"]).sort_values("Directional accuracy %",ascending=False,na_position="last")
    except Exception:return pd.DataFrame()

with st.sidebar:
    st.header("ZIA Terminal")
    symbol=st.selectbox("Symbol",SYMBOLS)
    tf=st.selectbox("Timeframe",list(TFS),index=0)
    bars=st.slider("Candles",100,1000,500,50)
    future=st.slider("Future space",5,80,25)
    refresh=st.slider("Live refresh",2,15,4)
    st.caption("Public read-only market data. Save Signal writes a local CSV; it does not place orders.")

st.markdown('<div class="hero"><div><div class="brand">ZIA <span>RESEARCH</span> TERMINAL</div><div class="muted">LIVE ML • ORDER FLOW • OFI • RESEARCH LAB • SIGNAL LAB</div></div><div class="live"><span class="dot"></span>LIVE ENGINE</div></div>',unsafe_allow_html=True)

def render():
    df,msrc,mstat,mhost=candles(symbol,TFS[tf],bars);b,a,bsrc,bstat,bhost=book(symbol);f=get_features(df,b,a);pred,prob,mlstat,nfeat=ml(f);sig,score,conf,fs,research,mlscore=combined(f,pred,prob);ofi=ofi_now(b,a);price=num(df.Close.iloc[-1]) if not df.empty else 0;o20,b20,a20=obi(b,a,20);o50,b50,a50=obi(b,a,50)
    top=st.columns(6);card(top[0],"PRICE",fmt(price),msrc);card(top[1],"FINAL SIGNAL",sig,f"Confidence {conf:.1f}%");card(top[2],"LIVE ML",f"{prob*100:.1f}%" if prob is not None else "—",mlstat);card(top[3],"OBI 20",f"{o20:+.3f}",f"Bid {b20:,.0f} / Ask {a20:,.0f}");card(top[4],"OBI 50",f"{o50:+.3f}","Top 50 levels");card(top[5],"OFI",f"{ofi:+,.1f}","20-level depth delta")
    left,right=st.columns([2.15,1])
    with left:
        st.markdown('<div class="section">PRICE ACTION</div>',unsafe_allow_html=True);st.plotly_chart(chart(df,future),use_container_width=True,config={"scrollZoom":True,"displaylogo":False,"modeBarButtonsToAdd":["drawline","drawrect","eraseshape"]},key="main_chart")
    with right:
        st.markdown('<div class="section">SIGNAL CONTROL</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="panel"><div class="sig">{sig}</div><div class="small">Score <b>{score:+.3f}</b> • Research <b>{research:+.3f}</b> • ML <b>{mlscore:+.3f}</b></div></div>',unsafe_allow_html=True)
        if st.button("💾 SAVE SIGNAL",use_container_width=True,type="primary"):
            save_signal({"timestamp":pd.Timestamp.now(tz="UTC").isoformat(),"symbol":symbol,"timeframe":tf,"signal":sig,"price":price,"confidence":conf,"score":score,"ml_probability":prob if prob is not None else "","obi20":o20,"obi50":o50,"ofi":ofi,"research_score":research})
            st.success("Signal saved")
        st.caption(f"Book: {bsrc} / {bstat} • {bhost or 'no host'}")
        st.caption(f"Market: {msrc} / {mstat} • {mhost or 'no host'}")
    st.markdown('<div class="section">ORDER FLOW & RESEARCH LAB</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown('<div class="panel"><b>OBI MATRIX</b></div>',unsafe_allow_html=True)
        vals=[]
        for k in (5,10,20,50):
            o,bb,aa=obi(b,a,k);vals.append([f"Top {k}",o,bb,aa])
        st.dataframe(pd.DataFrame(vals,columns=["Depth","OBI","Bid volume","Ask volume"]).style.format({"OBI":"{:+.3f}","Bid volume":"{:,.0f}","Ask volume":"{:,.0f}"}),use_container_width=True,hide_index=True)
    with c2:
        st.markdown('<div class="panel"><b>RESEARCH FORMULAS</b></div>',unsafe_allow_html=True)
        rr=pd.DataFrame([[k,v,"LONG" if v>.15 else "SHORT" if v<-.15 else "NEUTRAL"] for k,v in fs.items()],columns=["Formula","Score","Bias"])
        st.dataframe(rr.style.format({"Score":"{:+.3f}"}),use_container_width=True,hide_index=True)
    with c3:
        st.markdown('<div class="panel"><b>LIVE ML INPUTS</b></div>',unsafe_allow_html=True)
        mlrows=[["Model","XGBoost"],["Status",mlstat],["Features used",nfeat],["Prediction",pred if pred is not None else "—"],["Probability",f"{prob*100:.2f}%" if prob is not None else "—"],["Model file",MODEL_FILE.name if MODEL_FILE.exists() else "Missing"]]
        st.dataframe(pd.DataFrame(mlrows,columns=["Item","Value"]),use_container_width=True,hide_index=True)
    save_scores({"timestamp":time.time(),"symbol":symbol,"timeframe":tf,"close":price,**fs})
    st.markdown('<div class="section">FORMULA PERFORMANCE — HISTORICAL SNAPSHOTS</div>',unsafe_allow_html=True)
    lb=leaderboard()
    if lb.empty:st.info("Save/live snapshots first. Accuracy becomes meaningful after enough later price observations; this is research statistics, not a guarantee.")
    else:
        st.dataframe(lb.style.format({"Directional accuracy %":"{:.1f}%"}),use_container_width=True,hide_index=True)
        best=lb.dropna(subset=["Directional accuracy %"])
        if not best.empty and int(best.iloc[0]["Samples"])>=20:st.success(f"Current research leader: {best.iloc[0]['Formula']} — {best.iloc[0]['Directional accuracy %']:.1f}% directional accuracy over {int(best.iloc[0]['Samples'])} resolved snapshots.")
        else:st.caption("Leader is provisional until at least 20 resolved samples are available.")
    if st.session_state.get("last_refresh"):
        st.caption("Last live update: "+st.session_state.last_refresh)
    st.session_state.last_refresh=pd.Timestamp.now(tz="UTC").strftime("%H:%M:%S UTC")

# Fragment refresh updates the live area without rebuilding the browser page. Fallback keeps older Streamlit versions working.
if hasattr(st,"fragment"):
    @st.fragment(run_every=f"{refresh}s")
    def live_app():render()
    live_app()
else:
    render()
    time.sleep(refresh)
    st.rerun()
