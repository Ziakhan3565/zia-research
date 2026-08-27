from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ============================================================
# ZIA RESEARCH LAB — ML MARKET TERMINAL
# Dashboard only: existing research/training files are untouched.
# Existing XGBoost 7-feature schema is preserved.
# ============================================================

st.set_page_config(page_title="ZIA Research Lab", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "xgboost_obi_model.pkl"
HISTORY_FILE = ROOT / "signal_history.csv"
BINANCE_BASE = "https://fapi.binance.com"
KLINES_URL = f"{BINANCE_BASE}/fapi/v1/klines"
DEPTH_URL = f"{BINANCE_BASE}/fapi/v1/depth"
TICKER_URL = f"{BINANCE_BASE}/fapi/v1/ticker/24hr"
AGG_TRADES_URL = f"{BINANCE_BASE}/fapi/v1/aggTrades"
REQUEST_TIMEOUT = 8

COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT","TRXUSDT","LTCUSDT","BCHUSDT","DOTUSDT","XLMUSDT","NEARUSDT","UNIUSDT","APTUSDT","TAOUSDT","XMRUSDT"]
MODES = {
    "SCALPING": {"label":"5M / SCALP", "tf":"5m", "hold":"5–30 min", "refs":["15m","1h","4h"]},
    "15M": {"label":"15M", "tf":"15m", "hold":"30–120 min", "refs":["1h","4h"]},
    "1H": {"label":"1H", "tf":"1h", "hold":"2–24 hours", "refs":["4h","1d"]},
    "4H": {"label":"4H", "tf":"4h", "hold":"12–72 hours", "refs":["1d","1w"]},
}
MODEL_FEATURES = ["top20_bid_sum","top20_ask_sum","obi_top20","spread","bid_ask_ratio","total_depth","trend_signal"]

for key, value in {"symbol":"BTCUSDT","mode":"SCALPING","auto_refresh":True,"refresh_seconds":5,"last_signal":None,"last_saved_key":""}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""
<style>
.block-container{max-width:1700px;padding:1.15rem 2rem 2rem}
[data-testid="stSidebar"]{background:#080d14;border-right:1px solid #182231}
[data-testid="stSidebar"] *{color:#e9eef6}
.stApp{background:radial-gradient(circle at 75% -10%,#172235 0%,#080c12 42%)}
.hero{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:18px}
.brand{font-size:30px;font-weight:900}.brand span{color:#7c8cff}.subtitle{color:#8190a5;font-size:13px;margin-top:3px}
.live{display:inline-flex;gap:7px;align-items:center;padding:6px 10px;border:1px solid #284236;border-radius:999px;background:#0c1914;color:#8fe0b2;font-size:12px;font-weight:700}
.dot{width:7px;height:7px;border-radius:50%;background:#4bd58b;box-shadow:0 0 12px #4bd58b}
.signal{border-radius:18px;padding:22px 24px;background:linear-gradient(135deg,#111b2a,#0b111a);border:1px solid #263448;margin-bottom:14px}
.signal.long{border-color:#1c8f60;box-shadow:0 0 35px rgba(35,170,111,.08)}.signal.short{border-color:#a23c4d;box-shadow:0 0 35px rgba(220,67,91,.08)}.signal.wait{border-color:#344153}
.signal-label{color:#8794a6;font-size:11px;font-weight:800;letter-spacing:1.1px}.signal-name{font-size:42px;line-height:1.05;font-weight:950;margin:6px 0 5px}.signal-meta{color:#9ba7b8;font-size:13px}.big-price{font-size:29px;font-weight:850}
.kpi{background:#0d141e;border:1px solid #1d2939;border-radius:13px;padding:15px 16px;min-height:88px}.kpi-label{color:#7f8c9f;font-size:10px;text-transform:uppercase;letter-spacing:1px;font-weight:800}.kpi-value{color:#f1f5fa;font-size:22px;font-weight:850;margin-top:5px}.kpi-sub{color:#7f8c9f;font-size:11px;margin-top:2px}
.trade-row{background:#0c131d;border:1px solid #1d2938;border-radius:12px;padding:13px;text-align:center}.trade-label{color:#7f8c9e;font-size:10px;font-weight:800;letter-spacing:.8px}.trade-value{font-size:18px;font-weight:850;margin-top:4px}
.section{font-size:16px;font-weight:850;margin:20px 0 10px}.model-box{background:linear-gradient(145deg,#101a29,#0b1119);border:1px solid #29374a;border-radius:15px;padding:17px}.panel-title{color:#8c99aa;font-size:11px;font-weight:800;letter-spacing:1.25px;text-transform:uppercase}.progress{height:8px;border-radius:99px;background:#1b2635;overflow:hidden;margin-top:8px}.progress>div{height:100%;border-radius:99px;background:#7586ff}.small{color:#7e8b9d;font-size:11px}.good{color:#62d69b!important}.bad{color:#f17c8d!important}.neutral{color:#aab5c4!important}
div[data-testid="stMetric"]{background:#0d141e;border:1px solid #1d2939;padding:12px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

# --------------------------- helpers -------------------------
def f(v: Any, default: float=0.0) -> float:
    try:
        x=float(v); return x if np.isfinite(x) else default
    except Exception: return default

def clamp(v: Any, low: float=-1.0, high: float=1.0) -> float: return float(np.clip(f(v),low,high))
def price(v: Any) -> str:
    x=f(v)
    if x<=0:return "—"
    if x>=1000:return f"{x:,.2f}"
    if x>=1:return f"{x:,.4f}"
    return f"{x:.6f}"
def now_utc(): return dt.datetime.now(dt.timezone.utc)
def signal_class(d): return "long" if "LONG" in str(d).upper() else "short" if "SHORT" in str(d).upper() else "wait"
def bias(v):
    x=f(v)
    return ("BULLISH","good") if x>.25 else ("BEARISH","bad") if x<-.25 else ("NEUTRAL","neutral")
def pred_direction(p):
    try:return "LONG" if int(p)==1 else "SHORT"
    except:return "UNKNOWN"

# --------------------------- Binance --------------------------
def api_get(url, params=None):
    try:
        r=requests.get(url,params=params,timeout=REQUEST_TIMEOUT,headers={"User-Agent":"ZIA-RESEARCH-LAB"})
        r.raise_for_status(); return r.json()
    except Exception:return None

@st.cache_data(ttl=7,show_spinner=False)
def klines(symbol,interval,limit=260):
    raw=api_get(KLINES_URL,{"symbol":symbol,"interval":interval,"limit":limit})
    if not isinstance(raw,list):return pd.DataFrame()
    rows=[]
    for c in raw:
        try:rows.append({"Time":pd.to_datetime(int(c[0]),unit="ms",utc=True),"Open":float(c[1]),"High":float(c[2]),"Low":float(c[3]),"Close":float(c[4]),"Volume":float(c[5]),"Trades":int(c[8]),"TakerBuy":float(c[9])})
        except Exception:pass
    return pd.DataFrame(rows).dropna().reset_index(drop=True) if rows else pd.DataFrame()

@st.cache_data(ttl=2,show_spinner=False)
def orderbook(symbol,limit=100):
    raw=api_get(DEPTH_URL,{"symbol":symbol,"limit":limit})
    if not isinstance(raw,dict):return np.empty((0,2)),np.empty((0,2))
    try:
        b=np.asarray(raw.get("bids",[]),dtype=float); a=np.asarray(raw.get("asks",[]),dtype=float)
        if b.ndim!=2:b=np.empty((0,2))
        if a.ndim!=2:a=np.empty((0,2))
        return b,a
    except Exception:return np.empty((0,2)),np.empty((0,2))

@st.cache_data(ttl=4,show_spinner=False)
def ticker(symbol):
    raw=api_get(TICKER_URL,{"symbol":symbol}); return raw if isinstance(raw,dict) else {}

@st.cache_data(ttl=3,show_spinner=False)
def agg_trades(symbol,limit=1000):
    raw=api_get(AGG_TRADES_URL,{"symbol":symbol,"limit":limit}); return raw if isinstance(raw,list) else []

# --------------------------- research math --------------------
def obi(bids,asks,levels):
    n=min(len(bids),len(asks),levels)
    if n<=0:return 0.0
    bv=max(0.0,float(bids[:n,1].sum())); av=max(0.0,float(asks[:n,1].sum())); total=bv+av
    return clamp((bv-av)/total) if total else 0.0

def weighted_obi(bids,asks,levels=20):
    n=min(len(bids),len(asks),levels)
    if n<=0:return 0.0
    w=1/(np.arange(n)+1.0); bv=float((bids[:n,1]*w).sum()); av=float((asks[:n,1]*w).sum())
    return clamp((bv-av)/(bv+av)) if bv+av else 0.0

def depth(bids,asks,levels):
    n=min(len(bids),len(asks),levels)
    return (float(bids[:n,1].sum()),float(asks[:n,1].sum())) if n else (0.0,0.0)

def taker_flow(trades):
    buy=sell=0.0; count=0
    for t in trades:
        try:
            q=float(t["q"])
            if bool(t["m"]):sell+=q
            else:buy+=q
            count+=1
        except Exception:pass
    total=buy+sell
    return {"buy":buy,"sell":sell,"flow":buy-sell,"ratio":clamp((buy-sell)/total) if total else 0.0,"count":count}

def technical(df):
    if df.empty:return {}
    c=df["Close"]; e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean()
    m5=f(c.iloc[-1]/c.iloc[-6]-1) if len(c)>=6 else 0.0; m20=f(c.iloc[-1]/c.iloc[-21]-1) if len(c)>=21 else 0.0
    p=f(c.iloc[-1]); trend=(.30 if p>f(e20.iloc[-1]) else -.30)+(.25 if p>f(e50.iloc[-1]) else -.25)+(.20 if p>f(e200.iloc[-1]) else -.20)+clamp(m20*20,-.25,.25)
    return {"price":p,"ema20":f(e20.iloc[-1]),"ema50":f(e50.iloc[-1]),"ema200":f(e200.iloc[-1]),"momentum5":m5,"momentum20":m20,"volatility":f(c.pct_change().rolling(20).std().iloc[-1]),"trend":clamp(trend)}

def calc_atr(df,period=14):
    if len(df)<2:return 0.0
    prev=df["Close"].shift(1); tr=pd.concat([df["High"]-df["Low"],(df["High"]-prev).abs(),(df["Low"]-prev).abs()],axis=1).max(axis=1)
    return max(0.0,f(tr.ewm(alpha=1/period,adjust=False).mean().iloc[-1]))

@st.cache_data(ttl=12,show_spinner=False)
def htf_bias(symbol):
    out={}
    for tf in ["1h","4h","1d"]:
        d=klines(symbol,tf,120)
        if d.empty:out[tf]=0.0; continue
        c=d["Close"]; e20=c.ewm(span=20,adjust=False).mean().iloc[-1]; e50=c.ewm(span=50,adjust=False).mean().iloc[-1]; p=f(c.iloc[-1])
        out[tf]=(.5 if p>f(e20) else -.5)+(.5 if p>f(e50) else -.5)
    return out

# --------------------------- ML connection -------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_FILE.exists():return None,"Model file not found"
    try:return joblib.load(MODEL_FILE),"Loaded"
    except Exception as e:return None,f"Load error: {type(e).__name__}"

def predict_ml(model,features):
    if model is None:return None
    try:
        expected=getattr(model,"n_features_in_",None)
        if expected is not None and int(expected)!=len(features):return {"error":f"Model expects {expected} features; dashboard supplied {len(features)}"}
        X=np.asarray([features],dtype=float); p=int(model.predict(X)[0]); out={"prediction":p,"direction":pred_direction(p)}
        if hasattr(model,"predict_proba"):
            probs=np.asarray(model.predict_proba(X)[0],dtype=float); out["probabilities"]=probs.tolist(); out["confidence"]=float(np.max(probs)); out["classes"]=np.asarray(getattr(model,"classes_",range(len(probs)))).tolist()
        else:out["confidence"]=.5
        return out
    except Exception as e:return {"error":f"Prediction error: {type(e).__name__}"}

def build_signal(df,bids,asks,symbol,mode_key):
    tech=technical(df)
    if not tech:return None
    flow=taker_flow(agg_trades(symbol)); o5,o10,o20,o50=[obi(bids,asks,n) for n in (5,10,20,50)]; wobi=weighted_obi(bids,asks,20)
    multi=clamp(o5*.15+o10*.20+o20*.35+o50*.30); bid20,ask20=depth(bids,asks,20); bid50,ask50=depth(bids,asks,50); spread=f(asks[0,0]-bids[0,0]) if len(bids) and len(asks) else 0.0; ratio=bid20/ask20 if ask20>0 else 0.0
    ml_features=[bid20,ask20,o20,spread,ratio,bid20+ask20,tech["trend"]]
    model,model_status=load_model(); ml=predict_ml(model,ml_features)
    score=multi*.30+flow["ratio"]*.25+tech["trend"]*.25+clamp(tech["momentum5"]*30)*.10+clamp(tech["momentum20"]*15)*.10
    htf=htf_bias(symbol); hscore=htf.get("1h",0)*.45+htf.get("4h",0)*.35+htf.get("1d",0)*.20; score=clamp(score+hscore*.20)
    ml_conf=f(ml.get("confidence",.5),.5) if ml and "error" not in ml else .5
    if ml and "error" not in ml:score=clamp(score+(1 if ml["prediction"]==1 else -1)*min(.25,ml_conf*.25))
    conf=abs(score)*55+abs(multi)*20+abs(flow["ratio"])*15+abs(hscore)*10
    if ml and "error" not in ml:conf=conf*.75+ml_conf*100*.25
    confidence=float(np.clip(conf,0,99))
    if score>=.70 and confidence>=70:direction="STRONG LONG"
    elif score>=.42 and confidence>=55:direction="LONG"
    elif score<=-.70 and confidence>=70:direction="STRONG SHORT"
    elif score<=-.42 and confidence>=55:direction="SHORT"
    else:direction="WAIT"
    a=calc_atr(df) or tech["price"]*.005; sd=min(max(a*1.15,tech["price"]*.0025),tech["price"]*.006); entry=tech["price"]
    if "LONG" in direction:sl,tp1,tp2=entry-sd,entry+sd*2,entry+sd*3
    elif "SHORT" in direction:sl,tp1,tp2=entry+sd,entry-sd*2,entry-sd*3
    else:sl=tp1=tp2=entry
    tk=ticker(symbol)
    return {"timestamp":now_utc().isoformat(),"symbol":symbol,"mode":mode_key,"direction":direction,"score":score,"confidence":confidence,"price":entry,"entry":entry,"stop_loss":sl,"target1":tp1,"target2":tp2,"atr":a,"obi5":o5,"obi10":o10,"obi20":o20,"obi50":o50,"weighted_obi":wobi,"multi_obi":multi,"bid20":bid20,"ask20":ask20,"bid50":bid50,"ask50":ask50,"spread":spread,"taker_buy":flow["buy"],"taker_sell":flow["sell"],"taker_flow":flow["flow"],"taker_flow_ratio":flow["ratio"],"trade_count":flow["count"],"trend":tech["trend"],"momentum5":tech["momentum5"],"momentum20":tech["momentum20"],"ema20":tech["ema20"],"ema50":tech["ema50"],"ema200":tech["ema200"],"volatility":tech["volatility"],"htf_1h":htf.get("1h",0),"htf_4h":htf.get("4h",0),"htf_1d":htf.get("1d",0),"ml":ml,"ml_available":ml is not None and "error" not in ml,"ml_confidence":ml_conf,"ml_features":ml_features,"model_status":model_status,"change24":f(tk.get("priceChangePercent")),"volume24":f(tk.get("quoteVolume"))}

# --------------------------- persistence ----------------------
def save_signal(s):
    if s["direction"]=="WAIT":return
    row={k:s.get(k) for k in ["timestamp","symbol","mode","direction","score","confidence","entry","stop_loss","target1","target2","obi20","obi50","taker_flow_ratio"]}
    try:pd.DataFrame([row]).to_csv(HISTORY_FILE,mode="a",header=not HISTORY_FILE.exists(),index=False)
    except Exception:pass

def read_history():
    try:return pd.read_csv(HISTORY_FILE).tail(300) if HISTORY_FILE.exists() else pd.DataFrame()
    except Exception:return pd.DataFrame()

# --------------------------- charts ---------------------------
def price_chart(df,s):
    d=df.tail(180); fig=go.Figure(); fig.add_trace(go.Candlestick(x=d["Time"],open=d["Open"],high=d["High"],low=d["Low"],close=d["Close"],name="Price"))
    for span,name in [(20,"EMA 20"),(50,"EMA 50"),(200,"EMA 200")]:fig.add_trace(go.Scatter(x=d["Time"],y=d["Close"].ewm(span=span,adjust=False).mean(),name=name,line=dict(width=1.3)))
    fig.add_hline(y=s["entry"],annotation_text="ENTRY",line_dash="solid")
    if s["direction"]!="WAIT":
        fig.add_hline(y=s["stop_loss"],annotation_text="SL",line_dash="dot"); fig.add_hline(y=s["target1"],annotation_text="TP1 1:2",line_dash="dash"); fig.add_hline(y=s["target2"],annotation_text="TP2 1:3",line_dash="dash")
    fig.update_layout(template="plotly_dark",height=560,xaxis_rangeslider_visible=False,margin=dict(l=5,r=5,t=25,b=5),legend=dict(orientation="h",y=1.02),hovermode="x unified",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    return fig

def obi_chart(s):
    vals=[s["obi5"],s["obi10"],s["obi20"],s["obi50"]]; fig=go.Figure(go.Bar(x=["Top 5","Top 10","Top 20","Top 50"],y=vals,text=[f"{v:+.3f}" for v in vals],textposition="outside")); fig.add_hline(y=0); fig.update_layout(template="plotly_dark",height=310,yaxis=dict(range=[-1,1]),margin=dict(l=5,r=5,t=20,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)"); return fig

# --------------------------- sidebar --------------------------
with st.sidebar:
    st.markdown("### ⚡ ZIA RESEARCH"); st.caption("ML-powered Binance Futures research terminal"); st.divider()
    st.session_state.symbol=st.selectbox("MARKET",COINS,index=COINS.index(st.session_state.symbol) if st.session_state.symbol in COINS else 0)
    keys=list(MODES); st.session_state.mode=st.selectbox("ANALYSIS MODE",keys,index=keys.index(st.session_state.mode),format_func=lambda x:MODES[x]["label"])
    st.divider(); st.markdown("**ENGINE**")
    st.session_state.auto_refresh=st.toggle("Live refresh",value=st.session_state.auto_refresh)
    if st.session_state.auto_refresh:st.session_state.refresh_seconds=st.slider("Refresh interval",3,30,int(st.session_state.refresh_seconds),1)
    if st.button("↻ Refresh now",use_container_width=True):st.cache_data.clear();st.rerun()
    st.divider(); m=MODES[st.session_state.mode]; st.markdown("**CURRENT HORIZON**"); st.info(f"**{m['tf'].upper()}** analysis\n\nHolding: {m['hold']}\n\nHTF: {', '.join(m['refs']).upper()}"); st.caption("Research / signal generation only — no order execution.")
if st.session_state.auto_refresh:st_autorefresh(interval=int(st.session_state.refresh_seconds*1000),key="zia_live_refresh")

# --------------------------- live state -----------------------
symbol=st.session_state.symbol; mode_key=st.session_state.mode; mode=MODES[mode_key]; df=klines(symbol,mode["tf"]); bids,asks=orderbook(symbol)
if df.empty or len(df)<30:st.error(f"Market data unavailable for {symbol}. Check Binance Futures connectivity and refresh.");st.stop()
if len(bids)<20 or len(asks)<20:st.warning("Order-book depth is temporarily unavailable. Retrying on the next refresh.");st.stop()
s=build_signal(df,bids,asks,symbol,mode_key)
if s is None:st.error("Signal engine could not calculate the current market state.");st.stop()
st.session_state.last_signal=s
save_key=f"{symbol}:{mode_key}:{now_utc().strftime('%Y-%m-%d-%H-%M')}"
if st.session_state.last_saved_key!=save_key:save_signal(s);st.session_state.last_saved_key=save_key

# --------------------------- header ----------------------------
st.markdown(f"<div class='hero'><div><div class='brand'><span>⚡</span> ZIA RESEARCH LAB</div><div class='subtitle'>Binance USDⓈ-M Futures · Order Flow · OBI · Quant Trend · XGBoost ML</div></div><div class='live'><span class='dot'></span> LIVE · {now_utc().strftime('%H:%M:%S UTC')}</div></div>",unsafe_allow_html=True)
sc=signal_class(s["direction"]); ml_label="CONNECTED" if s["ml_available"] else "OFFLINE"; ml_cls="good" if s["ml_available"] else "bad"
st.markdown(f"<div class='signal {sc}'><div style='display:flex;justify-content:space-between;gap:20px;align-items:center'><div><div class='signal-label'>FINAL RESEARCH SIGNAL · {symbol} · {mode['label']}</div><div class='signal-name'>{s['direction']}</div><div class='signal-meta'>Quant score <b>{s['score']:+.3f}</b> · Confidence <b>{s['confidence']:.1f}%</b> · ML <b class='{ml_cls}'>{ml_label}</b></div></div><div style='text-align:right'><div class='signal-label'>LAST PRICE</div><div class='big-price'>${price(s['price'])}</div><div class='signal-meta'>24H {s['change24']:+.2f}%</div></div></div></div>",unsafe_allow_html=True)

cols=st.columns(6); kpis=[("Confidence",f"{s['confidence']:.1f}%","signal confidence"),("OBI 20",f"{s['obi20']:+.4f}","order-book imbalance"),("OBI 50",f"{s['obi50']:+.4f}","deep liquidity"),("Taker Flow",f"{s['taker_flow_ratio']:+.4f}","aggressive flow"),("Trend",f"{s['trend']:+.3f}","quant trend score"),("XGBoost",f"{s['ml_confidence']*100:.1f}%" if s['ml_available'] else "OFF","model probability")]
for c,(lab,val,sub) in zip(cols,kpis):
    with c:st.markdown(f"<div class='kpi'><div class='kpi-label'>{lab}</div><div class='kpi-value'>{val}</div><div class='kpi-sub'>{sub}</div></div>",unsafe_allow_html=True)

# --------------------------- tabs -----------------------------
tab1,tab2,tab3,tab4=st.tabs(["▣ Overview","◈ Order Flow","◎ ML Engine","▤ Signal History"])
with tab1:
    st.markdown('<div class="section">TRADE PLAN</div>',unsafe_allow_html=True); pc=st.columns(4)
    for c,(lab,val,sub) in zip(pc,[("ENTRY",s["entry"],"market reference"),("STOP LOSS",s["stop_loss"],"volatility adjusted"),("TARGET 1",s["target1"],"1 : 2 risk / reward"),("TARGET 2",s["target2"],"1 : 3 risk / reward")]):
        with c:st.markdown(f"<div class='trade-row'><div class='trade-label'>{lab}</div><div class='trade-value'>{price(val) if s['direction']!='WAIT' or lab=='ENTRY' else 'WAIT'}</div><div class='small'>{sub}</div></div>",unsafe_allow_html=True)
    st.markdown('<div class="section">PRICE ACTION</div>',unsafe_allow_html=True); st.plotly_chart(price_chart(df,s),use_container_width=True,config={"displaylogo":False,"responsive":True})
    a,b,c=st.columns(3)
    with a:
        st.markdown("**Higher-timeframe bias**")
        for tf in ["1h","4h","1d"]:
            lab,cls=bias(s[f"htf_{tf}"]);st.markdown(f"`{tf.upper()}` &nbsp; <span class='{cls}'><b>{lab}</b></span> &nbsp; {s[f'htf_{tf}']:+.2f}",unsafe_allow_html=True)
    with b:
        st.markdown("**Momentum & volatility**");st.metric("5-candle momentum",f"{s['momentum5']*100:+.2f}%");st.metric("20-candle momentum",f"{s['momentum20']*100:+.2f}%");st.metric("ATR",price(s["atr"]))
    with c:
        st.markdown("**Moving averages**");st.metric("EMA 20",price(s["ema20"]));st.metric("EMA 50",price(s["ema50"]));st.metric("EMA 200",price(s["ema200"]))

with tab2:
    st.markdown('<div class="section">ORDER BOOK IMBALANCE</div>',unsafe_allow_html=True);x,y=st.columns([1.25,1])
    with x:st.plotly_chart(obi_chart(s),use_container_width=True,config={"displaylogo":False})
    with y:st.metric("Top 20 bid volume",f"{s['bid20']:,.3f}");st.metric("Top 20 ask volume",f"{s['ask20']:,.3f}");st.metric("Top 50 bid volume",f"{s['bid50']:,.3f}");st.metric("Top 50 ask volume",f"{s['ask50']:,.3f}")
    st.markdown('<div class="section">TAKER / AGGRESSIVE FLOW</div>',unsafe_allow_html=True);fc=st.columns(4)
    for c,lab,val in zip(fc,["Taker Buy","Taker Sell","Flow Ratio","Trades"],[f"{s['taker_buy']:,.3f}",f"{s['taker_sell']:,.3f}",f"{s['taker_flow_ratio']:+.4f}",f"{s['trade_count']:,}"]):
        with c:st.metric(lab,val)
    st.markdown('<div class="section">LIVE ORDER BOOK · TOP 20</div>',unsafe_allow_html=True);l,r=st.columns(2)
    with l:bd=pd.DataFrame(bids[:20],columns=["Price","Quantity"]);bd["Price"]=bd["Price"].map(price);st.dataframe(bd,use_container_width=True,hide_index=True,height=470)
    with r:ad=pd.DataFrame(asks[:20],columns=["Price","Quantity"]);ad["Price"]=ad["Price"].map(price);st.dataframe(ad,use_container_width=True,hide_index=True,height=470)

with tab3:
    st.markdown('<div class="section">XGBOOST DECISION CENTER</div>',unsafe_allow_html=True);ml=s["ml"];l,r=st.columns([1.05,1.45]);status_cls="good" if s["ml_available"] else "bad";status_txt="CONNECTED" if s["ml_available"] else "OFFLINE"
    with l:
        st.markdown(f"<div class='model-box'><div class='panel-title'>MODEL STATUS</div><div style='font-size:24px;font-weight:900'>XGBoost <span class='{status_cls}'>{status_txt}</span></div><div class='small' style='margin-top:8px'>{s['model_status']}</div></div>",unsafe_allow_html=True)
        if ml and "error" in ml:st.error(ml["error"])
        elif ml:
            st.metric("ML direction",ml["direction"]);st.metric("ML confidence",f"{s['ml_confidence']*100:.2f}%")
            probs=ml.get("probabilities",[]);classes=ml.get("classes",list(range(len(probs))))
            for cl,p in zip(classes,probs):
                st.markdown(f"**{pred_direction(cl)}** · {p*100:.2f}%");st.markdown(f"<div class='progress'><div style='width:{max(0,min(100,p*100)):.2f}%'></div></div>",unsafe_allow_html=True)
    with r:
        st.markdown("**Exact 7 features sent to the trained model**");rows=[{"Feature":n,"Live value":v} for n,v in zip(MODEL_FEATURES,s["ml_features"])];st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True);st.info("Dashboard uses the existing 7-feature XGBoost schema and does not retrain or reorder the model.")

with tab4:
    st.markdown('<div class="section">RECENT RESEARCH SIGNALS</div>',unsafe_allow_html=True);h=read_history()
    if h.empty:st.info("No non-WAIT signals have been recorded yet.")
    else:
        cols=[c for c in ["timestamp","symbol","mode","direction","score","confidence","entry","stop_loss","target1","target2"] if c in h.columns];d=h[cols].tail(50).iloc[::-1].copy()
        for c in ["entry","stop_loss","target1","target2"]:
            if c in d:d[c]=d[c].map(price)
        if "score" in d:d["score"]=d["score"].astype(float).round(3)
        if "confidence" in d:d["confidence"]=d["confidence"].astype(float).round(1)
        st.dataframe(d,use_container_width=True,hide_index=True,height=520);st.caption(f"Stored history: {len(h):,} rows")

st.divider();a,b,c,d=st.columns(4)
with a:st.caption(f"● Binance Futures · {symbol}")
with b:st.caption(f"● Analysis · {mode['tf'].upper()}")
with c:st.caption(f"● ML · {'Connected' if s['ml_available'] else 'Unavailable'}")
with d:st.caption(f"● Updated · {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}")
