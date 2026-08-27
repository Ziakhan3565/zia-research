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

FUTURES = ["https://fapi.binance.com", "https://fapi1.binance.com", "https://fapi2.binance.com", "https://fapi3.binance.com", "https://fapi4.binance.com"]
SPOT = ["https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com", "https://api3.binance.com"]
DATA = ["https://data-api.binance.vision"]
SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT","TRXUSDT","LTCUSDT"]
TFS = {"5M":"5m","15M":"15m","30M":"30m","1H":"1h","4H":"4h","1D":"1d","1W":"1w"}
F25 = ["top20_bid_sum","top20_ask_sum","obi_5","obi_10","obi_20","obi_50","spread","spread_pct","bid_ask_ratio_20","bid_ask_ratio_50","top20_total_depth","top50_total_depth","taker_buy_volume","taker_sell_volume","taker_flow","taker_flow_ratio","price_return","price_change","sma_distance","realized_volatility","BOOK_IMB","QUANT_IMPLY","ADAPT_CONF","BAYESIAN","FOURIER_TREND"]
F7 = ["top20_bid_sum","top20_ask_sum","obi_top20","spread","bid_ask_ratio","total_depth","trend_signal"]

st.markdown("""
<style>
html,body,[data-testid=stAppViewContainer]{background:#060a10;color:#e9eef7}
.block-container{max-width:1900px;padding:9px clamp(7px,1.5vw,28px) 55px}
[data-testid=stHeader]{background:transparent}
.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1d2938;padding:4px 2px 11px;margin-bottom:10px}.brand{font-size:clamp(20px,2.5vw,32px);font-weight:950;letter-spacing:-1px}.brand span{color:#8792ff}.muted{font-size:9px;color:#748298;letter-spacing:1px}.live{border:1px solid #22583f;background:#091711;color:#70e0a5;border-radius:999px;padding:6px 10px;font-size:10px;font-weight:900}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#48dc8e;box-shadow:0 0 9px #48dc8e;margin-right:6px}
.card{background:linear-gradient(145deg,#101824,#0b1119);border:1px solid #202d3e;border-radius:13px;padding:10px;min-height:72px}.label{font-size:9px;color:#77879d;font-weight:900;letter-spacing:1px}.value{font-size:18px;font-weight:950;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sub{font-size:10px;color:#8491a5;margin-top:3px}.panel{background:#0b1119;border:1px solid #1d2939;border-radius:14px;padding:10px}.section{font-size:14px;font-weight:950;margin:13px 0 7px}.small{font-size:10px;color:#748399}.good{color:#65dfa0}.bad{color:#ff7787}.accent{color:#8d98ff}.sig{font-size:31px;font-weight:950}
@media(max-width:700px){.block-container{padding:6px 7px 40px}.brand{font-size:20px}.muted{font-size:7px}.live{font-size:8px;padding:5px 7px}.card{min-height:62px;padding:8px}.value{font-size:15px}.sub{font-size:9px}.sig{font-size:27px}}
</style>
""", unsafe_allow_html=True)

def num(x,d=0.0):
    try:
        x=float(x); return x if np.isfinite(x) else d
    except Exception:return d

def fmt(x):
    x=num(x); return f"{x:,.2f}" if x>=1000 else f"{x:,.4f}" if x>=1 else (f"{x:,.7f}" if x else "—")

def request_json(hosts,path,params):
    err="network"
    for h in hosts:
        try:
            r=requests.get(h+path,params=params,timeout=5,headers={"User-Agent":"ZIA-Research/7.0"})
            if r.ok:return r.json(),h,"OK"
            err=f"HTTP {r.status_code}"
        except requests.RequestException as e:err=type(e).__name__
    return None,None,err

@st.cache_data(ttl=2,show_spinner=False)
def candles(symbol,interval,limit):
    raw,h,s=request_json(FUTURES,"/fapi/v1/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1500)});source="Futures"
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
    return {
        "OBI 20":float(np.clip(f["obi_20"]*2,-1,1)),"OBI 20+50":float(np.clip(f["obi_20"]+f["obi_50"],-1,1)),
        "OFI":float(np.clip(f["taker_flow_ratio"]*2,-1,1)),"Trend / SMA":float(np.clip(np.tanh(f["sma_distance"]*100),-1,1)),
        "Fourier":float(np.clip(f["FOURIER_TREND"],-1,1)),"Bayesian":float(np.clip((f["BAYESIAN"]-.5)*2,-1,1)),
        "Quant Imply":float(np.clip(f["QUANT_IMPLY"],-1,1)),"Adaptive":float(np.clip((f["ADAPT_CONF"]-.5)*2,-1,1)),
    }

def combined(f,pred,prob):
    s=formula_scores(f);weights={"OBI 20":.22,"OBI 20+50":.14,"OFI":.20,"Trend / SMA":.14,"Fourier":.10,"Bayesian":.08,"Quant Imply":.07,"Adaptive":.05}
    research=sum(s[k]*weights[k] for k in weights);mlscore=((prob-.5)*2) if prob is not None else (1 if pred==1 else -1 if pred==0 else 0)
    score=.60*research+.40*mlscore if pred is not None else research;sig="LONG" if score>=.45 else "SHORT" if score<=-.45 else "WAIT";conf=float(np.clip(50+abs(score)*49,1,99))
    return sig,float(score),conf,s,research,mlscore,weights

def ofi_now(b,a):
    prev=st.session_state.get("prev_book");st.session_state["prev_book"]=(b.copy(),a.copy())
    if prev is None:return 0.
    pb,pa=prev;_,bc,ac=obi(b,a,20);_,pbc,pac=obi(pb,pa,20);return (bc-pbc)-(ac-pac)

def save_row(path,row):
    exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as fp:
        w=csv.DictWriter(fp,fieldnames=list(row));
        if not exists:w.writeheader()
        w.writerow(row)

def saved_signals():
    if not SIGNAL_FILE.exists():return pd.DataFrame()
    try:return pd.read_csv(SIGNAL_FILE).tail(100).iloc[::-1]
    except Exception:return pd.DataFrame()

def save_signal(row):save_row(SIGNAL_FILE,row)
def save_scores(row):save_row(SCORE_FILE,row)

def tri_set_for_chart(tf):
    if tf=="15M":return {"4H":"4h","1H":"1h"}
    if tf in ("1H","4H"):return {"DAY":"1d","WEEK":"1w","MONTH":"1M"}
    return {}

@st.cache_data(ttl=20,show_spinner=False)
def tri_levels(symbol,tf):
    result={}
    for name,interval in tri_set_for_chart(tf).items():
        d,_,_,_=candles(symbol,interval,5)
        if len(d)<2:continue
        c=d.iloc[-2];o,h,l,cl=map(num,[c.Open,c.High,c.Low,c.Close]);body_hi=max(o,cl);body_lo=min(o,cl)
        result[name]={"BODY 50":(body_hi+body_lo)/2,"UPPER 50":(h+body_hi)/2,"LOWER 50":(l+body_lo)/2}
    return result

def chart(df,future,tf,symbol,show_tri):
    fig=go.Figure()
    if df.empty:return fig
    fig.add_trace(go.Candlestick(x=df.Time,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="PRICE",increasing_line_color="#2bd28f",increasing_fillcolor="#159d6a",decreasing_line_color="#ff7180",decreasing_fillcolor="#d84b60"))
    for span in (20,50,200):
        if len(df)>=span:fig.add_trace(go.Scatter(x=df.Time,y=df.Close.ewm(span=span,adjust=False).mean(),name=f"EMA {span}",mode="lines",line={"width":1}))
    if show_tri:
        levels=tri_levels(symbol,tf);line_specs=[("BODY 50",0.9,"solid"),("UPPER 50",0.7,"dot"),("LOWER 50",0.7,"dot")]
        for period,lv in levels.items():
            for label,opacity,dash in line_specs:
                if label in lv:
                    fig.add_hline(y=lv[label],line_dash=dash,line_width=1,opacity=opacity,annotation_text=f"TRI {period} {label}",annotation_position="top right")
    step=df.Time.iloc[-1]-df.Time.iloc[-2] if len(df)>1 else pd.Timedelta(minutes=5)
    fig.update_xaxes(range=[df.Time.iloc[max(0,len(df)-260)],df.Time.iloc[-1]+step*future],showgrid=True,gridcolor="#172230",showspikes=True,spikemode="across",spikesnap="cursor")
    fig.update_yaxes(side="right",showgrid=True,gridcolor="#172230")
    fig.update_layout(template="plotly_dark",height=650,margin=dict(l=4,r=4,t=30,b=4),xaxis_rangeslider_visible=False,dragmode="pan",hovermode="x unified",paper_bgcolor="#0b1119",plot_bgcolor="#0b1119",legend=dict(orientation="h",y=1.02,x=0))
    return fig

def card(c,label,value,sub=""):c.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)

def leaderboard():
    if not SCORE_FILE.exists():return pd.DataFrame()
    try:
        d=pd.read_csv(SCORE_FILE);d["close"]=pd.to_numeric(d["close"],errors="coerce");d["future_close"]=d.groupby(["symbol","timeframe"])["close"].shift(-5);d["ret"]=d["future_close"]/d["close"]-1
        out=[]
        names=list(formula_scores({k:0 for k in F25}).keys())
        for x in names:
            if x not in d:continue
            sc=pd.to_numeric(d[x],errors="coerce");mask=sc.abs()>=.15;actual=np.sign(d.ret);n=int((mask&d.ret.notna()).sum());acc=float((np.sign(sc[mask&d.ret.notna()])==actual[mask&d.ret.notna()]).mean()*100) if n else np.nan;out.append([x,n,acc])
        return pd.DataFrame(out,columns=["Formula","Resolved samples","Directional accuracy %"]).sort_values("Directional accuracy %",ascending=False,na_position="last")
    except Exception:return pd.DataFrame()

with st.sidebar:
    st.header("ZIA Research")
    symbol=st.selectbox("Symbol",SYMBOLS)
    tf=st.selectbox("Chart timeframe",list(TFS),index=1)
    bars=st.slider("Candles",100,1000,500,50)
    future=st.slider("Future space",5,100,25)
    refresh=st.slider("Live refresh (sec)",2,15,4)
    show_tri=st.checkbox("Show TRI lines",True)
    st.caption("15M chart → 4H + 1H TRI. 1H/4H chart → Day + Week + Month TRI.")

st.markdown('<div class="hero"><div><div class="brand">ZIA <span>RESEARCH</span> TERMINAL</div><div class="muted">LIVE ML • OBI • OFI • RESEARCH LAB • TRI LINES • SAVED SIGNALS</div></div><div class="live"><span class="dot"></span>LIVE ENGINE</div></div>',unsafe_allow_html=True)

def render():
    df,msrc,mstat,mhost=candles(symbol,TFS[tf],bars);b,a,bsrc,bstat,bhost=book(symbol);f=get_features(df,b,a);pred,prob,mlstat,nfeat=ml(f);sig,score,conf,fs,research,mlscore,weights=combined(f,pred,prob);ofi=ofi_now(b,a);price=num(df.Close.iloc[-1]) if not df.empty else 0;o20,b20,a20=obi(b,a,20);o50,b50,a50=obi(b,a,50)
    top=st.columns(6);card(top[0],"PRICE",fmt(price),msrc);card(top[1],"FINAL SIGNAL",sig,f"Confidence {conf:.1f}%");card(top[2],"LIVE ML",f"{prob*100:.1f}%" if prob is not None else "—",mlstat);card(top[3],"OBI 20",f"{o20:+.3f}",f"Bid {b20:,.0f} / Ask {a20:,.0f}");card(top[4],"OBI 50",f"{o50:+.3f}","Top 50 levels");card(top[5],"OFI",f"{ofi:+,.1f}","20-level flow delta")
    left,right=st.columns([2.25,1])
    with left:
        st.markdown(f'<div class="section">PRICE ACTION • {symbol} • {tf}</div>',unsafe_allow_html=True)
        st.plotly_chart(chart(df,future,tf,symbol,show_tri),use_container_width=True,config={"scrollZoom":True,"displaylogo":False,"modeBarButtonsToAdd":["drawline","drawrect","eraseshape"]},key="main_chart")
        if show_tri:
            st.caption("TRI mapping: 15M = 4H + 1H • 1H/4H = Day + Week + Month. BODY 50 is the body midpoint; UPPER/LOWER 50 are wick midpoints.")
    with right:
        st.markdown('<div class="section">SIGNAL CONTROL</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="panel"><div class="sig">{sig}</div><div class="small">Score <b>{score:+.3f}</b> • Research <b>{research:+.3f}</b> • ML <b>{mlscore:+.3f}</b></div></div>',unsafe_allow_html=True)
        if st.button("💾 SAVE CURRENT SIGNAL",use_container_width=True,type="primary"):
            save_signal({"timestamp":pd.Timestamp.now(tz="UTC").isoformat(),"symbol":symbol,"timeframe":tf,"signal":sig,"price":price,"confidence":conf,"score":score,"ml_probability":prob if prob is not None else "","obi20":o20,"obi50":o50,"ofi":ofi,"research_score":research,**fs});st.success("Signal saved to saved_signals.csv")
        st.caption(f"Book: {bsrc} / {bstat} • {bhost or '—'}")
        st.caption(f"Market: {msrc} / {mstat} • {mhost or '—'}")
        st.caption(f"ML model: {MODEL_FILE.name if MODEL_FILE.exists() else 'MISSING'} • features: {nfeat}")

    st.markdown('<div class="section">LIVE ORDER FLOW • OBI / OFI</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        vals=[]
        for k in (5,10,20,50):
            o,bb,aa=obi(b,a,k);vals.append([f"Top {k}",o,bb,aa])
        st.dataframe(pd.DataFrame(vals,columns=["Depth","OBI","Bid volume","Ask volume"]).style.format({"OBI":"{:+.3f}","Bid volume":"{:,.0f}","Ask volume":"{:,.0f}"}),use_container_width=True,hide_index=True)
    with c2:
        st.markdown('<div class="panel"><b>OFI SNAPSHOT</b></div>',unsafe_allow_html=True)
        st.metric("20-level OFI",f"{ofi:+,.2f}")
        st.metric("Taker-flow ratio",f"{f['taker_flow_ratio']:+.3f}")
        st.caption("OFI is the live change in 20-level displayed depth between dashboard samples. Taker flow uses the available kline taker-buy volume proxy.")
    with c3:
        st.markdown('<div class="panel"><b>DEPTH BALANCE</b></div>',unsafe_allow_html=True)
        st.metric("Top 20 depth",f"{b20+a20:,.0f}");st.metric("Top 50 depth",f"{b50+a50:,.0f}");st.metric("Spread",fmt(f["spread"]))

    st.markdown('<div class="section">RESEARCH LAB • FORMULA SCORE + CONTRIBUTION</div>',unsafe_allow_html=True)
    r1,r2=st.columns([1.35,1])
    with r1:
        rows=[]
        for name,val in fs.items():rows.append([name,val,weights.get(name,0)*100,val*weights.get(name,0)])
        rdf=pd.DataFrame(rows,columns=["Formula","Live score","Weight %","Weighted contribution"]).sort_values("Weighted contribution",ascending=False)
        st.dataframe(rdf.style.format({"Live score":"{:+.3f}","Weight %":"{:.1f}","Weighted contribution":"{:+.3f}"}),use_container_width=True,hide_index=True)
    with r2:
        best=max(fs,key=lambda k:abs(fs[k])) if fs else "—";leader=max(fs,key=lambda k:fs[k]) if fs else "—"
        st.markdown(f'<div class="panel"><b>FORMULA LEADER</b><br><br><span class="accent" style="font-size:24px;font-weight:900">{best}</span><br><span class="small">Strongest current normalized reading: {fs.get(best,0):+.3f}</span><br><br><b>Directional leader</b><br>{leader} • {fs.get(leader,0):+.3f}<br><br><b>Research composite</b><br>{research:+.3f}</div>',unsafe_allow_html=True)

    st.markdown('<div class="section">LIVE ML ENGINE</div>',unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4);card(m1,"MODEL",MODEL_FILE.name if MODEL_FILE.exists() else "MISSING",mlstat);card(m2,"FEATURES",str(nfeat),"model input columns");card(m3,"PREDICTION",str(pred) if pred is not None else "—","0=down • 1=up when model uses binary labels");card(m4,"PROBABILITY",f"{prob*100:.2f}%" if prob is not None else "—","XGBoost predict_proba")

    save_scores({"timestamp":time.time(),"symbol":symbol,"timeframe":tf,"close":price,**fs})
    st.markdown('<div class="section">FORMULA PERFORMANCE • HISTORICAL RESOLVED SNAPSHOTS</div>',unsafe_allow_html=True)
    lb=leaderboard()
    if lb.empty:st.info("Historical leaderboard will populate after saved/live snapshots have enough future candles to resolve.")
    else:
        st.dataframe(lb.style.format({"Directional accuracy %":"{:.1f}%"}),use_container_width=True,hide_index=True)
        bestrow=lb.dropna(subset=["Directional accuracy %"])
        if not bestrow.empty:
            st.caption(f"Best historical formula: {bestrow.iloc[0]['Formula']} • {bestrow.iloc[0]['Directional accuracy %']:.1f}% directional accuracy • {int(bestrow.iloc[0]['Resolved samples'])} resolved samples. Research statistic, not a guarantee.")

    st.markdown('<div class="section">SAVED SIGNALS</div>',unsafe_allow_html=True)
    saved=saved_signals()
    if saved.empty:st.info("No saved signals yet. Click SAVE CURRENT SIGNAL when you want to record the current ML/OBI/OFI state.")
    else:
        cols=[x for x in ["timestamp","symbol","timeframe","signal","price","confidence","score","ml_probability","obi20","obi50","ofi","research_score"] if x in saved.columns]
        st.dataframe(saved[cols],use_container_width=True,hide_index=True)
        st.download_button("⬇️ DOWNLOAD SAVED SIGNALS CSV",saved.to_csv(index=False),file_name="saved_signals.csv",mime="text/csv",use_container_width=True)

    st.caption(f"Live update: {pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S UTC')} • Data: {msrc} • Order book: {bsrc} • Dashboard is read-only and does not place orders.")

if hasattr(st,"fragment"):
    @st.fragment(run_every=f"{refresh}s")
    def live_app():render()
    live_app()
else:
    render()
    time.sleep(refresh)
    st.rerun()
