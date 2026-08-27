from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

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
TRADE_FILE = ROOT / "trade_history.csv"
FUTURES = ["https://fapi.binance.com", "https://fapi1.binance.com", "https://fapi2.binance.com"]
SPOT = ["https://api.binance.com", "https://api1.binance.com"]
DATA = ["https://data-api.binance.vision"]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT"]
TFS = {"1MO":"1M", "1W":"1w", "1D":"1d", "4H":"4h", "1H":"1h", "30M":"30m", "15M":"15m", "5M":"5m", "3M":"3m", "1M":"1m"}

st.markdown("""
<style>
:root{--bg:#05070b;--panel:#0b1119;--panel2:#101925;--line:#1d2a39;--txt:#edf3fb;--muted:#7f8da1;--green:#42dda0;--red:#ff7184;--amber:#f3c86a;--cyan:#65d7ff;--violet:#969eff}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--txt)}
[data-testid="stHeader"]{background:transparent}.block-container{max-width:1920px;padding:10px clamp(8px,1.5vw,30px) 28px}
.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding:4px 2px 12px;margin-bottom:10px}.brand{font-size:clamp(25px,2.7vw,42px);font-weight:950;letter-spacing:-2px}.brand b{color:var(--violet)}.micro{color:var(--muted);font-size:9px;letter-spacing:1.5px}.live{border:1px solid #245d45;background:#071810;color:#6ce3a5;border-radius:999px;padding:7px 12px;font-size:10px;font-weight:900}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);margin-right:6px}
.panel{background:linear-gradient(145deg,#0e151f,#080d14);border:1px solid var(--line);border-radius:15px;padding:13px;margin-bottom:10px}.card{background:linear-gradient(145deg,#111a26,#0a1018);border:1px solid var(--line);border-radius:14px;padding:11px;min-height:76px}.label{font-size:9px;color:var(--muted);font-weight:900;letter-spacing:1.1px}.value{font-size:19px;font-weight:950;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sub{font-size:10px;color:#8794a8;margin-top:3px}.good{color:var(--green)}.bad{color:var(--red)}.amber{color:var(--amber)}.violet{color:var(--violet)}.cyan{color:var(--cyan)}
.signalbox{border-radius:18px;padding:16px 20px;border:1px solid var(--line);background:linear-gradient(145deg,#111a26,#080d14);text-align:center;margin-bottom:10px}.signal-long{border-color:#277b59;box-shadow:0 0 28px rgba(66,221,160,.08)}.signal-short{border-color:#843a4a;box-shadow:0 0 28px rgba(255,113,132,.08)}.signal-wait{border-color:#705e30}.signal-main{font-size:clamp(34px,4vw,58px);font-weight:1000;letter-spacing:-2px;line-height:1}.signal-meta{font-size:10px;color:var(--muted);margin-top:7px;letter-spacing:1px}
.tri-strip{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 9px}.tri-chip{border:1px solid var(--line);background:#0a1119;border-radius:9px;padding:6px 9px;font-size:9px;font-weight:900;letter-spacing:.5px}.tri-chip span{color:var(--cyan)}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:900}div[data-testid="stTabs"] button{font-weight:900;font-size:11px}
[data-testid="stSelectbox"] label{font-size:9px;font-weight:900;letter-spacing:1px;color:var(--muted)}
@media(max-width:700px){.block-container{padding:6px 7px 22px}.brand{font-size:21px}.micro{font-size:7px}.live{font-size:8px;padding:5px 8px}.panel{padding:9px;border-radius:12px}.card{min-height:62px;padding:9px}.value{font-size:16px}}
</style>
""", unsafe_allow_html=True)

def num(x, default=0.0):
    try:
        v=float(x); return v if np.isfinite(v) else default
    except Exception: return default

def api(hosts, path, params):
    err="network"
    for host in hosts:
        try:
            r=requests.get(host+path, params=params, timeout=2.2, headers={"User-Agent":"ZIA-Research"})
            if r.ok:return r.json(),host,"OK"
            err=f"HTTP {r.status_code}"
        except requests.RequestException as e:err=type(e).__name__
    return None,None,err

@st.cache_data(ttl=1.0, show_spinner=False)
def candles(symbol, interval, limit=650):
    raw,host,status=api(FUTURES,"/fapi/v1/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1500)}); source="Futures"
    if not isinstance(raw,list):raw,host,status=api(SPOT,"/api/v3/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1000)}); source="Spot"
    if not isinstance(raw,list):raw,host,status=api(DATA,"/api/v3/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1000)}); source="Data API"
    rows=[]
    for c in raw or []:
        try:rows.append([pd.to_datetime(int(c[0]),unit="ms",utc=True),num(c[1]),num(c[2]),num(c[3]),num(c[4]),num(c[5]),num(c[9])])
        except Exception:pass
    return pd.DataFrame(rows,columns=["Time","Open","High","Low","Close","Volume","TakerBuy"]),source,status,host

@st.cache_data(ttl=1.0, show_spinner=False)
def orderbook(symbol):
    raw,host,status=api(FUTURES,"/fapi/v1/depth",{"symbol":symbol,"limit":100}); source="Futures"
    if not isinstance(raw,dict) or not raw.get("bids"):raw,host,status=api(SPOT,"/api/v3/depth",{"symbol":symbol,"limit":100}); source="Spot"
    if not isinstance(raw,dict) or not raw.get("bids"):raw,host,status=api(DATA,"/api/v3/depth",{"symbol":symbol,"limit":100}); source="Data API"
    try:return np.asarray(raw.get("bids",[]),float),np.asarray(raw.get("asks",[]),float),source,status,host
    except Exception:return np.empty((0,2)),np.empty((0,2)),source,status,host

def obi(bids,asks,k):
    if len(bids)==0 or len(asks)==0:return 0.,0.,0.
    k=min(k,len(bids),len(asks)); bv=float(bids[:k,1].sum()); av=float(asks[:k,1].sum()); return ((bv-av)/(bv+av) if bv+av else 0.),bv,av

def features(df,b,a):
    f={k:0. for k in ["top20_bid_sum","top20_ask_sum","obi_5","obi_10","obi_20","obi_50","spread","spread_pct","bid_ask_ratio_20","bid_ask_ratio_50","top20_total_depth","top50_total_depth","taker_buy_volume","taker_sell_volume","taker_flow","taker_flow_ratio","price_return","price_change","sma_distance","realized_volatility","BOOK_IMB","QUANT_IMPLY","ADAPT_CONF","BAYESIAN","FOURIER_TREND"]}
    (o5,b5,a5),(o10,b10,a10),(o20,b20,a20),(o50,b50,a50)=[obi(b,a,k) for k in (5,10,20,50)]
    f.update(top20_bid_sum=b20,top20_ask_sum=a20,obi_5=o5,obi_10=o10,obi_20=o20,obi_50=o50,top20_total_depth=b20+a20,top50_total_depth=b50+a50)
    if df.empty:return f
    c=df.Close; last=num(c.iloc[-1]); prev=num(c.iloc[-2] if len(c)>1 else last); sma=num(c.rolling(20).mean().iloc[-1],last); total=num(df.Volume.tail(20).sum()); buy=num(df.TakerBuy.tail(20).sum()); sell=max(total-buy,0); flow=buy-sell; spread=num(a[0,0]-b[0,0]) if len(a) and len(b) else 0; trend=np.tanh((last/sma-1)*100) if sma else 0; rv=num(c.pct_change().tail(30).std()); four=np.tanh(c.pct_change().tail(16).mean()*1000)
    f.update(spread=spread,spread_pct=spread/last if last else 0,bid_ask_ratio_20=b20/a20 if a20 else 1,bid_ask_ratio_50=b50/a50 if a50 else 1,taker_buy_volume=buy,taker_sell_volume=sell,taker_flow=flow,taker_flow_ratio=flow/total if total else 0,price_return=last/prev-1 if prev else 0,price_change=last-prev,sma_distance=last/sma-1 if sma else 0,realized_volatility=rv,BOOK_IMB=o20,QUANT_IMPLY=float(np.tanh((o20+o50+trend)/3)),ADAPT_CONF=float(np.clip(.5+(abs(o20)+abs(trend))/2,0,1)),BAYESIAN=float(np.clip(.5+(o20+trend)/4,0,1)),FOURIER_TREND=float(four)); return f

@st.cache_resource(show_spinner=False)
def load_model():
    try:return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
    except Exception:return None

def ml_predict(f):
    m=load_model()
    if m is None:return None,None,"MODEL NOT FOUND",0
    try:
        names=list(m.get_booster().feature_names or []) if hasattr(m,"get_booster") else []; count=int(getattr(m,"n_features_in_",len(names) or 25)); legacy=["top20_bid_sum","top20_ask_sum","obi_top20","spread","bid_ask_ratio","total_depth","trend_signal"]; cols=names if names else (legacy if count==7 else list(f.keys())); row=dict(f,obi_top20=f["obi_20"],bid_ask_ratio=f["bid_ask_ratio_20"],total_depth=f["top20_total_depth"],trend_signal=f["sma_distance"]); x=pd.DataFrame([[row.get(c,0.) for c in cols]],columns=cols); pred=int(m.predict(x)[0]); proba=float(m.predict_proba(x)[0][-1]) if hasattr(m,"predict_proba") else None; return pred,proba,"OK",len(cols)
    except Exception as e:return None,None,"ML ERROR: "+type(e).__name__,0

def research(f):
    scores={"OBI 20":np.clip(f["obi_20"]*2,-1,1),"OBI 20+50":np.clip((f["obi_20"]+f["obi_50"])/1.5,-1,1),"OFI / Taker":np.clip(f["taker_flow_ratio"]*2,-1,1),"Trend / SMA":np.clip(np.tanh(f["sma_distance"]*100),-1,1),"Fourier":np.clip(f["FOURIER_TREND"],-1,1),"Bayesian":np.clip((f["BAYESIAN"]-.5)*2,-1,1),"Quant Imply":np.clip(f["QUANT_IMPLY"],-1,1),"Adaptive":np.clip((f["ADAPT_CONF"]-.5)*2,-1,1)}
    weights={"OBI 20":.22,"OBI 20+50":.14,"OFI / Taker":.20,"Trend / SMA":.14,"Fourier":.10,"Bayesian":.08,"Quant Imply":.07,"Adaptive":.05}
    return scores,weights,float(sum(scores[k]*weights[k] for k in scores))

def final_state(f,p,pr):
    scores,weights,rscore=research(f); mlscore=(pr-.5)*2 if pr is not None else (1 if p==1 else -1 if p==0 else 0); combined=.6*rscore+.4*mlscore if p is not None else rscore; signal="LONG" if combined>=.45 else "SHORT" if combined<=-.45 else "WAIT"; confidence=float(np.clip(50+abs(combined)*49,1,99)); return signal,confidence,combined,scores,weights,rscore,mlscore

# TRI controls are intentionally removed. The chart decides which source timeframes are visible.
def visible_tri_timeframes(tf):
    if tf in ("1M","3M","5M","15M","30M"): return [("1H","1 HOUR"),("4H","4 HOUR")]
    if tf in ("1H","4H"): return [("1D","DAILY"),("1W","WEEKLY")]
    if tf=="1D": return [("1W","WEEKLY"),("1MO","MONTHLY")]
    if tf=="1W": return [("1MO","MONTHLY")]
    return []

@st.cache_data(ttl=20,show_spinner=False)
def tri_levels(symbol,interval):
    df,_,_,_=candles(symbol,interval,8)
    if len(df)<2:return None
    c=df.iloc[-2]; o,h,l,cl=map(num,[c.Open,c.High,c.Low,c.Close]); bh,bl=max(o,cl),min(o,cl)
    return {"body":(bh+bl)/2,"upper":(h+bh)/2,"lower":(l+bl)/2}

def make_chart(df,symbol,tf,future):
    fig=go.Figure()
    if df.empty:return fig
    view=df.tail(650).copy()
    fig.add_trace(go.Candlestick(x=view.Time,open=view.Open,high=view.High,low=view.Low,close=view.Close,name="PRICE",increasing_line_color="#42dda0",increasing_fillcolor="#176d4f",decreasing_line_color="#ff7184",decreasing_fillcolor="#8e3448"))
    for span in (10,20,50,200):
        if len(view)>=span:fig.add_trace(go.Scatter(x=view.Time,y=view.Close.ewm(span=span,adjust=False).mean(),mode="lines",name=f"EMA {span}",line={"width":1.05}))
    line_colors={"1MO":"#b29cff","1W":"#8e98ff","1D":"#65d7ff","1H":"#42dda0","4H":"#f3c86a"}
    for label,_ in visible_tri_timeframes(tf):
        lv=tri_levels(symbol,TFS[label])
        if not lv:continue
        col=line_colors[label]
        for kind,key,dash,width,pos in [("BODY 50","body","solid",1.9,"top right"),("UPPER 50","upper","dot",1.05,"top left"),("LOWER 50","lower","dot",1.05,"bottom left")]:
            fig.add_hline(y=lv[key],line_color=col,line_width=width,line_dash=dash,annotation_text=f"TRI {label} • {kind}",annotation_position=pos,annotation_font_size=9)
    step=view.Time.iloc[-1]-view.Time.iloc[-2] if len(view)>1 else pd.Timedelta(minutes=5)
    fig.update_xaxes(range=[view.Time.iloc[0],view.Time.iloc[-1]+step*future],rangeslider_visible=False,showgrid=True,gridcolor="#172230",showspikes=True,spikemode="across",fixedrange=False)
    fig.update_yaxes(side="right",showgrid=True,gridcolor="#172230",fixedrange=False,automargin=True)
    fig.update_layout(height=680,margin=dict(l=4,r=4,t=12,b=8),paper_bgcolor="#080d14",plot_bgcolor="#080d14",font=dict(color="#cbd5e1"),hovermode="x unified",dragmode="pan",uirevision=f"ZIA-{symbol}-{tf}",legend=dict(orientation="h",y=1.02,x=0),hoverlabel=dict(font_size=11))
    return fig

def cards(items):
    cs=st.columns(len(items))
    for c,(lab,val,sub,cl) in zip(cs,items):
        with c:st.markdown(f'<div class="card"><div class="label">{lab}</div><div class="value {cl}">{val}</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)

def read_csv(path):
    try:return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:return pd.DataFrame()

def save_signal(symbol,tf,price,sig,conf,pr,f,rs):
    row={"timestamp":datetime.now(timezone.utc).isoformat(),"symbol":symbol,"timeframe":tf,"price":price,"signal":sig,"confidence":conf,"ml_probability":pr if pr is not None else "","obi20":f["obi_20"],"obi50":f["obi_50"],"ofi":f["taker_flow_ratio"],"research_score":rs}
    pd.DataFrame([row]).to_csv(SIGNAL_FILE,mode="a",header=not SIGNAL_FILE.exists(),index=False)

if "symbol" not in st.session_state:st.session_state.symbol="BTCUSDT"
if "tf" not in st.session_state:st.session_state.tf="15M"
if "future" not in st.session_state:st.session_state.future=30

st.markdown('<div class="hero"><div><div class="brand">ZIA <b>RESEARCH</b></div><div class="micro">QUANT MARKET INTELLIGENCE • LIVE ML • ORDER FLOW • RESEARCH LAB</div></div><div class="live"><span class="dot"></span>LIVE • SILENT 1S</div></div>',unsafe_allow_html=True)
c1,c2,c3=st.columns([2.2,1.15,1])
with c1:symbol=st.selectbox("MARKET",SYMBOLS,index=SYMBOLS.index(st.session_state.symbol),key="symbol")
with c2:tf=st.selectbox("TIMEFRAME",list(TFS.keys()),index=list(TFS.keys()).index(st.session_state.tf),key="tf")
with c3:future=st.selectbox("FUTURE SPACE",[12,20,30,45,60],index=2,key="future",format_func=lambda x:f"{x} bars")

visible=visible_tri_timeframes(tf)
tri_text=" + ".join(x[0] for x in visible) if visible else "NONE"
st.markdown(f'<div class="tri-strip"><div class="tri-chip">AUTO TRI</div><div class="tri-chip">CHART <span>{tf}</span></div><div class="tri-chip">DISPLAY <span>{tri_text}</span></div><div class="tri-chip">CONTROLS <span>OFF</span></div><div class="tri-chip">ZOOM <span>ON</span></div><div class="tri-chip">PAN <span>ON</span></div></div>',unsafe_allow_html=True)

@st.fragment(run_every="1s")
def live_engine():
    started=time.perf_counter()
    df,source,cstat,_=candles(symbol,TFS[tf],650); bids,asks,bsrc,bstat,_=orderbook(symbol); f=features(df,bids,asks)
    pred,prob,mlstat,feature_count=ml_predict(f); signal,confidence,combined,rs,rw,rscore,mlscore=final_state(f,pred,prob)
    price=num(df.Close.iloc[-1]) if not df.empty else 0; prev=num(df.Close.iloc[-2]) if len(df)>1 else price; change=(price/prev-1)*100 if prev else 0
    elapsed=(time.perf_counter()-started)*1000; cls="signal-long" if signal=="LONG" else "signal-short" if signal=="SHORT" else "signal-wait"; sigcolor="good" if signal=="LONG" else "bad" if signal=="SHORT" else "amber"; mltext=f"{prob*100:.2f}%" if prob is not None else "—"
    st.markdown(f'<div class="signalbox {cls}"><div class="label">MAIN AI + RESEARCH SIGNAL</div><div class="signal-main {sigcolor}">{signal}</div><div class="signal-meta">CONFIDENCE {confidence:.1f}% • ML {mltext} • RESEARCH {rscore:+.3f} • COMPOSITE {combined:+.3f}</div></div>',unsafe_allow_html=True)
    cards([("PRICE",f"${price:,.2f}",f"{change:+.2f}% • {tf}","good" if change>=0 else "bad"),("SIGNAL",signal,f"strength {confidence:.1f}%",sigcolor),("ML",mltext,mlstat,"violet"),("OBI 20",f"{f['obi_20']:+.3f}","top 20 depth","good" if f['obi_20']>=0 else "bad"),("OBI 50",f"{f['obi_50']:+.3f}","top 50 depth","good" if f['obi_50']>=0 else "bad"),("DATA",source,f"book {bsrc}","cyan")])
    tabs=st.tabs(["⌂ OVERVIEW","◈ CHART","◌ ORDER FLOW","🧠 ML LAB","🔬 RESEARCH LAB","▣ SIGNALS"])
    with tabs[0]:
        l,r=st.columns([2,1])
        with l:
            st.markdown('<div class="panel"><b>MARKET REGIME</b>',unsafe_allow_html=True); regime="BULLISH FLOW" if combined>.25 else "BEARISH FLOW" if combined<-.25 else "BALANCED / WAIT"; st.markdown(f"## {regime}"); st.progress(min(max(confidence/100,0),1),text=f"Signal strength {confidence:.1f}%"); st.write(f"Research **{rscore:+.3f}** • ML **{mlscore:+.3f}** • Composite **{combined:+.3f}**"); st.markdown('</div>',unsafe_allow_html=True)
        with r:
            st.markdown('<div class="panel"><b>LIVE STATUS</b>',unsafe_allow_html=True); st.write(f"Candles: `{source}`"); st.write(f"Order book: `{bsrc}`"); st.write(f"Connection: `{cstat} / {bstat}`"); st.write(f"Engine: `{elapsed:.0f} ms` • silent 1s cycle"); st.write(f"Updated: `{datetime.now().strftime('%H:%M:%S')}`"); st.markdown('</div>',unsafe_allow_html=True)
    with tabs[1]:
        st.markdown(f'<div class="panel"><b>TRADINGVIEW-STYLE MARKET CHART</b><div class="section-sub">Automatic TRI only • {tf} → {tri_text} • scroll zoom • mouse pan • crosshair • future space • chart state preserved</div>',unsafe_allow_html=True)
        st.plotly_chart(make_chart(df,symbol,tf,future),use_container_width=True,config={"scrollZoom":True,"displaylogo":False,"responsive":True,"doubleClick":"reset","modeBarButtonsToRemove":["lasso2d","select2d"]},key="main_market_chart")
        st.markdown('</div>',unsafe_allow_html=True)
    with tabs[2]:
        vals=[obi(bids,asks,k) for k in (5,10,20,50)]; cards([(f"OBI {k}",f"{v[0]:+.3f}",f"B {v[1]:,.1f} / A {v[2]:,.1f}","good" if v[0]>=0 else "bad") for k,v in zip((5,10,20,50),vals)])
        l,r=st.columns(2); l.dataframe(pd.DataFrame(bids[:20],columns=["Bid Price","Bid Qty"]),use_container_width=True,hide_index=True); r.dataframe(pd.DataFrame(asks[:20],columns=["Ask Price","Ask Qty"]),use_container_width=True,hide_index=True)
    with tabs[3]:
        cards([("MODEL",mlstat,"xgboost_obi_model.pkl","violet"),("PREDICTION","LONG" if pred==1 else "SHORT" if pred==0 else "—",f"class {pred}","good" if pred==1 else "bad" if pred==0 else "amber"),("PROBABILITY",mltext,"model probability","violet"),("FEATURES",str(feature_count),"supplied to model","cyan")])
        st.markdown('<div class="panel"><b>LIVE MODEL INPUTS</b>',unsafe_allow_html=True); st.dataframe(pd.DataFrame({"Feature":["OBI 5","OBI 10","OBI 20","OBI 50","Spread","Taker Flow","Trend/SMA","Volatility"],"Value":[f["obi_5"],f["obi_10"],f["obi_20"],f["obi_50"],f["spread"],f["taker_flow_ratio"],f["sma_distance"],f["realized_volatility"]]}),use_container_width=True,hide_index=True); st.markdown('</div>',unsafe_allow_html=True)
    with tabs[4]:
        rd=pd.DataFrame([{"Formula":k,"Live Score":round(float(v),4),"Weight %":round(rw[k]*100,1),"Contribution":round(float(v*rw[k]),4),"Direction":"BULL" if v>0 else "BEAR" if v<0 else "NEUTRAL"} for k,v in rs.items()]).sort_values("Contribution",ascending=False)
        st.markdown('<div class="panel"><b>RESEARCH FORMULA SCOREBOARD</b>',unsafe_allow_html=True); st.dataframe(rd,use_container_width=True,hide_index=True); st.write(f"Strongest contributor: **{rd.iloc[0]['Formula'] if not rd.empty else '—'}** • Composite **{rscore:+.3f}**"); st.markdown('</div>',unsafe_allow_html=True)
    with tabs[5]:
        if st.button("💾 SAVE CURRENT SIGNAL",use_container_width=True,key="save_signal_btn"):save_signal(symbol,tf,price,signal,confidence,prob,f,rscore); st.success("Signal saved")
        h=read_csv(SIGNAL_FILE); t=read_csv(TRADE_FILE)
        if not h.empty:st.dataframe(h.tail(80).iloc[::-1],use_container_width=True,hide_index=True); st.download_button("⬇ Download Signal Journal",h.to_csv(index=False),"zia_saved_signals.csv","text/csv",use_container_width=True)
        else:st.info("No saved signals yet.")
        if not t.empty and "result" in t.columns:
            rr=t.result.astype(str).str.upper(); wins=int((rr=="WIN").sum()); losses=int((rr=="LOSS").sum()); total=wins+losses; wr=wins/total*100 if total else 0; cards([("CLOSED",str(total),"resolved trades","cyan"),("WINS",str(wins),"winning trades","good"),("LOSSES",str(losses),"losing trades","bad"),("WIN RATE",f"{wr:.1f}%","closed trade rate","violet")])
    st.caption(f"ZIA Research • {symbol} • {tf} • silent live engine • {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

live_engine()
