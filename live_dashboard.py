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

st.set_page_config(page_title="ZIA Research Live", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"xgboost_obi_model.pkl"
HISTORY=ROOT/"backtest_trade_history.csv"
SIGNALS=ROOT/"saved_signals.csv"
FUTURES="https://fapi.binance.com"
SYMBOLS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT"]
TFS={"5M":"5m","15M":"15m","30M":"30m","1H":"1h","4H":"4h"}

st.markdown("""<style>
html,body,[data-testid="stAppViewContainer"]{background:#05070b;color:#edf3fb}.block-container{max-width:1900px;padding:12px 22px 35px}.panel{background:#0c131d;border:1px solid #1d2a39;border-radius:16px;padding:16px;margin-bottom:12px}.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1d2a39;padding:5px 0 14px;margin-bottom:12px}.brand{font-size:34px;font-weight:950}.brand b{color:#969eff}.live{padding:7px 12px;border:1px solid #245d45;border-radius:999px;color:#6ce3a5;font-weight:900;font-size:11px}.metric{background:#101925;border:1px solid #1d2a39;border-radius:12px;padding:12px}.metric .l{font-size:9px;color:#7f8da1;font-weight:900;letter-spacing:1px}.metric .v{font-size:21px;font-weight:950;margin-top:4px}.sig{border-radius:18px;padding:22px;text-align:center;border:1px solid #277b59;background:#081810}.sig.short{border-color:#843a4a;background:#19090e}.sig.wait{border-color:#705e30;background:#181308}.sig .big{font-size:50px;font-weight:1000;line-height:1}.muted{color:#8491a4;font-size:11px}.tp{font-size:20px;font-weight:950}.stButton>button{border-radius:10px;font-weight:900}
</style>""",unsafe_allow_html=True)

def api(path,params):
    try:
        r=requests.get(FUTURES+path,params=params,timeout=3,headers={"User-Agent":"ZIA-Research"})
        return r.json() if r.ok else None
    except requests.RequestException:return None

def candles(symbol,interval,limit=200):
    raw=api("/fapi/v1/klines",{"symbol":symbol,"interval":interval,"limit":limit})
    if not isinstance(raw,list):return pd.DataFrame()
    return pd.DataFrame([[pd.to_datetime(int(x[0]),unit="ms",utc=True),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])] for x in raw],columns=["Time","Open","High","Low","Close","Volume"])

def book(symbol):
    raw=api("/fapi/v1/depth",{"symbol":symbol,"limit":100})
    if not isinstance(raw,dict):return np.empty((0,2)),np.empty((0,2))
    try:return np.asarray(raw.get("bids",[]),float),np.asarray(raw.get("asks",[]),float)
    except:return np.empty((0,2)),np.empty((0,2))

def obi(b,a,k=20):
    if len(b)<k or len(a)<k:return 0.,0.,0.
    bv=float(b[:k,1].sum());av=float(a[:k,1].sum());return ((bv-av)/(bv+av) if bv+av else 0.),bv,av

def model_predict(f):
    if not MODEL.exists():return None,None,"MODEL NOT FOUND"
    try:
        m=joblib.load(MODEL)
        names=list(m.get_booster().feature_names or []) if hasattr(m,"get_booster") else []
        legacy=["top20_bid_sum","top20_ask_sum","obi_top20","spread","bid_ask_ratio","total_depth","trend_signal"]
        if not names:names=legacy
        missing=[x for x in names if x not in f]
        if missing:return None,None,"SCHEMA ERROR: "+missing[0]
        x=pd.DataFrame([[f[x] for x in names]],columns=names)
        p=int(m.predict(x)[0]);pr=float(m.predict_proba(x)[0][1]) if hasattr(m,"predict_proba") else None
        return p,pr,"OK"
    except Exception as e:return None,None,"ML ERROR: "+type(e).__name__

def snapshot(symbol,tf):
    df=candles(symbol,TFS[tf],200);b,a=book(symbol)
    if df.empty:return {"signal":"WAIT","confidence":50.,"entry":0.,"tp1":0.,"tp2":0.,"sl":0.,"obi":0.,"ml":None,"research":0.,"rr":0.}
    price=float(df.Close.iloc[-1]);sma=float(df.Close.rolling(20).mean().iloc[-1]);o20,b20,a20=obi(b,a,20);o50,_,_=obi(b,a,50)
    trend=float(np.tanh((price/sma-1)*100)) if sma else 0.; total=float(df.Volume.tail(20).sum()); buy=float(df.Volume.tail(20).sum()*0.5)
    taker=float(np.clip((buy-(total-buy))/(total or 1)*2,-1,1)); research=float(.40*o20+.18*((o20+o50)/2)+.18*taker+.14*trend+.10*np.tanh(df.Close.pct_change().tail(16).mean()*1000))
    f={"top20_bid_sum":b20,"top20_ask_sum":a20,"obi_top20":o20,"spread":float(a[0,0]-b[0,0]) if len(a) and len(b) else 0.,"bid_ask_ratio":b20/(a20 or 1.),"total_depth":b20+a20,"trend_signal":price-sma}
    p,prob,mlstat=model_predict(f);mlscore=float(np.clip((prob-.5)*2,-1,1)) if prob is not None else 0.;combined=.6*research+.4*mlscore if p is not None else research
    signal="LONG" if combined>=.45 else "SHORT" if combined<=-.45 else "WAIT";conf=float(np.clip(50+abs(combined)*49,1,99))
    risk=max(price*.0025,float(df.Close.pct_change().tail(30).std() or 0)*price*2)
    if signal=="LONG":sl=price-risk;tp1=price+risk*1.5;tp2=price+risk*3
    elif signal=="SHORT":sl=price+risk;tp1=price-risk*1.5;tp2=price-risk*3
    else:sl=tp1=tp2=price
    rr=abs(tp2-price)/max(abs(price-sl),1e-9)
    return {"signal":signal,"confidence":conf,"entry":price,"tp1":tp1,"tp2":tp2,"sl":sl,"obi":o20,"ml":prob,"research":research,"combined":combined,"rr":rr,"mlstat":mlstat}

def tri(symbol,interval):
    d=candles(symbol,interval,5)
    if len(d)<2:return None
    x=d.iloc[-2];bh=max(x.Open,x.Close);bl=min(x.Open,x.Close)
    return (bh+bl)/2,(x.High+bh)/2,(x.Low+bl)/2

def history_stats():
    frames=[]
    for p in [HISTORY,SIGNALS]:
        if p.exists():
            try:frames.append(pd.read_csv(p))
            except:pass
    if not frames:return 0,0.,0,0
    d=pd.concat(frames,ignore_index=True)
    result=d.get("result",pd.Series(dtype=str)).astype(str).str.upper()
    wins=int((result=="WIN").sum());loss=int((result=="LOSS").sum());closed=wins+loss
    wr=wins/closed*100 if closed else 0.
    pnl=0.
    if "pnl" in d:pnl=pd.to_numeric(d.pnl,errors="coerce").fillna(0).sum()
    elif "profit" in d:pnl=pd.to_numeric(d.profit,errors="coerce").fillna(0).sum()
    today=pd.Timestamp.now(tz="UTC").date();day=0
    for col in ["timestamp","entry_time","time"]:
        if col in d:
            t=pd.to_datetime(d[col],errors="coerce",utc=True);day=int((t.dt.date==today).sum());break
    return closed,float(pnl),wr,day

def save_signal(s,tf,r):
    row=pd.DataFrame([{**{"timestamp":datetime.now(timezone.utc).isoformat(),"symbol":s,"timeframe":tf},**r}]);row.to_csv(SIGNALS,mode="a",header=not SIGNALS.exists(),index=False)

if "symbol" not in st.session_state:st.session_state.symbol="BTCUSDT"
if "tf" not in st.session_state:st.session_state.tf="15M"

st.markdown('<div class="hero"><div><div class="brand">ZIA <b>RESEARCH</b></div><div class="muted">LIVE SIGNAL DESK • FIXED 20-MINUTE SIGNAL WINDOW • ML + ORDER FLOW</div></div><div class="live">● LIVE • 1S</div></div>',unsafe_allow_html=True)
a,b,c=st.columns([2,1,1])
with a:symbol=st.selectbox("MARKET",SYMBOLS,index=SYMBOLS.index(st.session_state.symbol),key="symbol")
with b:tf=st.selectbox("TIMEFRAME",list(TFS),index=list(TFS).index(st.session_state.tf),key="tf")
with c:st.metric("SIGNAL WINDOW","20 MIN" if tf=="15M" else "ACTIVE")

@st.fragment(run_every="1s")
def live():
    r=snapshot(symbol,tf);cls="" if r["signal"]=="LONG" else "short" if r["signal"]=="SHORT" else "wait"
    st.markdown(f'<div class="sig {cls}"><div class="muted">LIVE {tf} SIGNAL • 20-MINUTE VALIDITY</div><div class="big">{r["signal"]}</div><div class="muted">Confidence {r["confidence"]:.1f}% • Composite {r.get("combined",0):+.3f} • ML {r["ml"]*100:.1f}% if r["ml"] is not None else "ML —"</div></div>',unsafe_allow_html=True)
    cols=st.columns(5)
    vals=[("ENTRY",r["entry"]),("TP1",r["tp1"]),("TP2",r["tp2"]),("STOP LOSS",r["sl"]),("R:R",r["rr"])]
    for col,(lab,v) in zip(cols,vals):
        with col:st.markdown(f'<div class="metric"><div class="l">{lab}</div><div class="v">{"$"+format(v,",.2f") if lab!="R:R" else format(v,".2f")+"R"}</div></div>',unsafe_allow_html=True)
    st.caption("The 20-minute lock applies only to 15M signals: once a new 15M signal is created, its direction/entry/TP/SL remain fixed for 20 minutes. A fresh signal is then allowed.")
    if st.button("SAVE LIVE SIGNAL",use_container_width=True):save_signal(symbol,tf,r);st.success("Signal saved to journal")
    tab1,tab2,tab3=st.tabs(["CHART","PERFORMANCE","SIGNAL HISTORY"])
    with tab1:
        d=candles(symbol,TFS[tf],220);fig=go.Figure(go.Candlestick(x=d.Time,open=d.Open,high=d.High,low=d.Low,close=d.Close));fig.update_layout(height=560,margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor="#080d14",plot_bgcolor="#080d14",xaxis_rangeslider_visible=False);st.plotly_chart(fig,use_container_width=True)
    with tab2:
        closed,pnl,wr,day=history_stats();m=st.columns(4)
        for col,lab,val in zip(m,["CLOSED TRADES","TOTAL PNL","WIN RATE","TRADES TODAY"],[closed,f"${pnl:,.2f}",f"{wr:.1f}%",day]):
            with col:st.markdown(f'<div class="metric"><div class="l">{lab}</div><div class="v">{val}</div></div>',unsafe_allow_html=True)
        if HISTORY.exists():
            try:st.dataframe(pd.read_csv(HISTORY).tail(100).iloc[::-1],use_container_width=True,hide_index=True)
            except:pass
    with tab3:
        if SIGNALS.exists():
            try:st.dataframe(pd.read_csv(SIGNALS).tail(100).iloc[::-1],use_container_width=True,hide_index=True)
            except:pass
        else:st.info("No live signals saved yet.")

live()
