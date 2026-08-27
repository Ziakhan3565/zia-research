from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
import json, time
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="ZIA Research Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
ROOT=Path(__file__).resolve().parent
MODEL_FILE=ROOT/"xgboost_obi_model.pkl"
HISTORY=ROOT/"trade_history.csv"
SAVED=ROOT/"saved_signals.csv"
LOCKS=ROOT/".signal_locks.json"
HOST="https://fapi.binance.com"
SYMBOLS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT","TRXUSDT","LTCUSDT"]
TFS={"5M":"5m","15M":"15m","30M":"30m","1H":"1h","4H":"4h"}
LOCK_MIN={"15M":20,"1H":150}

st.markdown("""
<style>
.stApp{background:#05070b;color:#edf3fb}.block-container{max-width:1900px;padding:14px 22px 40px}
.hero{padding:10px 0 14px;border-bottom:1px solid #202b3a;margin-bottom:14px}.brand{font-size:34px;font-weight:900;letter-spacing:-1.5px}.muted{color:#8190a5;font-size:11px}
.panel{background:#0b1119;border:1px solid #1d2938;border-radius:16px;padding:15px;margin:10px 0}.title{font-size:19px;font-weight:900;margin-bottom:3px}.sub{font-size:10px;color:#8190a5}
.sig{border-radius:16px;padding:14px;border:1px solid #263344;background:#0d151f}.long{border-color:#2c805d}.short{border-color:#8b4050}.wait{border-color:#80682c}
.direction{font-size:31px;font-weight:1000}.long .direction{color:#4be0a2}.short .direction{color:#ff7185}.wait .direction{color:#f3c86a}
.metric{background:#101925;border:1px solid #202d3e;border-radius:10px;padding:8px;margin-top:7px}.mlabel{font-size:8px;color:#7f8da1;font-weight:900}.mvalue{font-size:14px;font-weight:900;margin-top:3px}
.lock{font-size:10px;color:#65d7ff;font-weight:900;margin-top:7px}.table-head{font-size:10px;color:#8492a7;font-weight:900}
</style>
""",unsafe_allow_html=True)


def api(path,params):
    try:
        r=requests.get(HOST+path,params=params,timeout=4,headers={"User-Agent":"ZIA-Research"}); r.raise_for_status(); return r.json()
    except Exception:return None

@st.cache_data(ttl=2,show_spinner=False)
def candles(symbol,interval,limit=250):
    raw=api("/fapi/v1/klines",{"symbol":symbol,"interval":interval,"limit":limit})
    if not isinstance(raw,list): return pd.DataFrame()
    return pd.DataFrame([[pd.to_datetime(x[0],unit="ms",utc=True),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5]),float(x[9])] for x in raw],columns=["Time","Open","High","Low","Close","Volume","TakerBuy"])

@st.cache_data(ttl=2,show_spinner=False)
def book(symbol):
    raw=api("/fapi/v1/depth",{"symbol":symbol,"limit":50})
    try:return np.asarray(raw["bids"],float),np.asarray(raw["asks"],float)
    except Exception:return np.empty((0,2)),np.empty((0,2))

@st.cache_resource(show_spinner=False)
def model():
    try:return joblib.load(MODEL_FILE)
    except Exception:return None


def obi(b,a,k):
    k=min(k,len(b),len(a))
    if not k:return 0.,0.,0.
    bv=float(b[:k,1].sum());av=float(a[:k,1].sum());return ((bv-av)/(bv+av) if bv+av else 0),bv,av


def raw_signal(symbol,tf):
    df=candles(symbol,TFS[tf]); b,a=book(symbol)
    if df.empty:return None
    price=float(df.Close.iloc[-1]); sma=float(df.Close.rolling(20).mean().iloc[-1]);
    o5,_,_=obi(b,a,5);o20,b20,a20=obi(b,a,20);o50,_,_=obi(b,a,50)
    ret=float(df.Close.pct_change().tail(8).mean()); trend=float(np.tanh((price/sma-1)*100)) if sma else 0
    taker=float((df.TakerBuy.tail(20).sum()-(df.Volume.tail(20).sum()-df.TakerBuy.tail(20).sum()))/max(df.Volume.tail(20).sum(),1e-9))
    score=float(np.clip(.38*o20+.17*o50+.20*trend+.15*taker+.10*ret*100,-1,1))
    mlp=None; m=model()
    if m is not None:
        try:
            row={"top20_bid_sum":b20,"top20_ask_sum":a20,"obi_5":o5,"obi_10":obi(b,a,10)[0],"obi_20":o20,"obi_50":o50,
                 "spread":float(a[0,0]-b[0,0]) if len(a) and len(b) else 0,"spread_pct":0,"bid_ask_ratio_20":b20/max(a20,1e-9),
                 "bid_ask_ratio_50":1,"top20_total_depth":b20+a20,"top50_total_depth":float(b[:50,1].sum()+a[:50,1].sum()),
                 "taker_buy_volume":float(df.TakerBuy.tail(20).sum()),"taker_sell_volume":float(df.Volume.tail(20).sum()-df.TakerBuy.tail(20).sum()),
                 "taker_flow":taker,"taker_flow_ratio":taker,"price_return":float(df.Close.pct_change().iloc[-1]),"price_change":float(df.Close.iloc[-1]-df.Close.iloc[-2]),
                 "sma_distance":price/sma-1 if sma else 0,"realized_volatility":float(df.Close.pct_change().tail(30).std()),
                 "BOOK_IMB":o20,"QUANT_IMPLY":float(np.tanh((o20+o50+trend)/3)),"ADAPT_CONF":float(np.clip(.5+(abs(o20)+abs(trend))/2,0,1)),
                 "BAYESIAN":float(np.clip(.5+(o20+trend)/4,0,1)),"FOURIER_TREND":float(np.tanh(df.Close.pct_change().tail(16).mean()*1000))}
            names=list(m.get_booster().feature_names or []) if hasattr(m,"get_booster") else []
            if not names:names=list(row)[:int(getattr(m,"n_features_in_",len(row)))]
            x=pd.DataFrame([[row.get(c,0.) for c in names]],columns=names); mlp=float(m.predict_proba(x)[0][-1]) if hasattr(m,"predict_proba") else None
        except Exception:mlp=None
    if mlp is not None:score=float(np.clip(.65*score+.35*((mlp-.5)*2),-1,1))
    direction="LONG" if score>=.30 else "SHORT" if score<=-.30 else "WAIT"
    conf=float(np.clip(50+abs(score)*49,1,99))
    atr=float((df.High-df.Low).tail(14).mean())
    if direction=="LONG":tp1=price+atr;tp2=price+2*atr;sl=price-atr*.75
    elif direction=="SHORT":tp1=price-atr;tp2=price-2*atr;sl=price+atr*.75
    else:tp1=price+atr;tp2=price+2*atr;sl=price-atr*.75
    return {"symbol":symbol,"tf":tf,"direction":direction,"confidence":conf,"ml":mlp,"entry":price,"tp1":tp1,"tp2":tp2,"sl":sl,"score":score,"created":datetime.now(timezone.utc).isoformat()}


def read_locks():
    try:return json.loads(LOCKS.read_text())
    except Exception:return {}

def write_locks(x):
    try:LOCKS.write_text(json.dumps(x))
    except Exception:pass

def locked_signal(symbol,tf):
    locks=read_locks();key=f"{symbol}:{tf}";now=datetime.now(timezone.utc)
    old=locks.get(key)
    if old and tf in LOCK_MIN:
        try:
            until=datetime.fromisoformat(old["until"])
            if now<until:
                old["remaining"]=int((until-now).total_seconds());return old
        except Exception:pass
    s=raw_signal(symbol,tf)
    if s is None:return None
    if tf in LOCK_MIN and s["direction"] in ("LONG","SHORT"):
        until=now+timedelta(minutes=LOCK_MIN[tf]);s["until"]=until.isoformat();s["remaining"]=LOCK_MIN[tf]*60;locks[key]=s;write_locks(locks)
    return s


def money(x):
    return f"${x:,.2f}" if abs(x)<100000 else f"{x:,.2f}"

st.markdown('<div class="hero"><div class="brand">⚡ ZIA <b>RESEARCH</b> TERMINAL</div><div class="muted">LIVE MULTI-CRYPTO SIGNAL ENGINE • AUTO LOOP</div></div>',unsafe_allow_html=True)

with st.sidebar:
    st.header("Engine")
    tf=st.selectbox("Selected timeframe",list(TFS),index=1)
    refresh=st.slider("Refresh seconds",5,60,10)
    if st.button("Refresh now",use_container_width=True):st.cache_data.clear();st.rerun()

# ---- ALL CRYPTO LIVE BOARD ----
st.markdown('<div class="panel"><div class="title">🔴 LIVE LONG / SHORT — ALL CRYPTOCURRENCIES</div><div class="sub">Every loop scans the tracked markets. 15M LONG/SHORT stays locked for 20 minutes; 1H LONG/SHORT stays locked for 2.5 hours.</div></div>',unsafe_allow_html=True)
cols=st.columns(3)
board=[]
for i,sym in enumerate(SYMBOLS):
    s=locked_signal(sym,tf)
    board.append(s)
    if not s:continue
    cls="long" if s["direction"]=="LONG" else "short" if s["direction"]=="SHORT" else "wait"
    rem=s.get("remaining",0);lock=f"LOCK {rem//60:02d}:{rem%60:02d}" if rem>0 else "LIVE"
    with cols[i%3]:
        st.markdown(f'''<div class="sig {cls}"><div style="display:flex;justify-content:space-between"><b>{sym}</b><span class="muted">{tf}</span></div><div class="direction">{s["direction"]}</div><div class="lock">{lock} • {s["confidence"]:.0f}% confidence</div><div class="metric"><span class="mlabel">ENTRY</span><div class="mvalue">{money(s["entry"])}</div></div><div class="metric"><span class="mlabel">TP1 / TP2</span><div class="mvalue">{money(s["tp1"])} / {money(s["tp2"])}</div></div><div class="metric"><span class="mlabel">STOP LOSS</span><div class="mvalue">{money(s["sl"])}</div></div></div>''',unsafe_allow_html=True)

# ---- SELECTED DETAIL ----
st.markdown('<div class="panel"><div class="title">🎯 SELECTED SIGNAL DETAIL</div><div class="sub">Locked trade levels remain unchanged during the lock period.</div></div>',unsafe_allow_html=True)
selected=locked_signal(st.selectbox("Crypto",SYMBOLS,index=0),tf)
if selected:
    a,b,c,d,e=st.columns(5)
    a.metric("SIGNAL",selected["direction"]);b.metric("BTC / RATE",money(selected["entry"]));c.metric("ENTRY",money(selected["entry"]));d.metric("TP1",money(selected["tp1"]));e.metric("TP2",money(selected["tp2"]))
    st.info(f"Stop Loss: {money(selected['sl'])}  |  Confidence: {selected['confidence']:.1f}%  |  ML probability: {selected['ml']*100:.1f}%" if selected['ml'] is not None else f"Stop Loss: {money(selected['sl'])} | Confidence: {selected['confidence']:.1f}%")

# ---- HISTORY / PERFORMANCE ----
st.markdown('<div class="panel"><div class="title">📊 TRADE & BACKTEST PERFORMANCE</div><div class="sub">Reads the repository trade history when available and calculates PNL, trades today and win ratio.</div></div>',unsafe_allow_html=True)
frames=[]
for p in [HISTORY,ROOT/"backtest_trade_history.csv",SAVED]:
    if p.exists():
        try:
            q=pd.read_csv(p)
            q["__source"]=p.name;frames.append(q)
        except Exception:pass
hist=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
if not hist.empty:
    pnlcol=next((c for c in hist.columns if c.lower() in {"pnl","profit","profit_loss","net_pnl"}),None)
    resultcol=next((c for c in hist.columns if c.lower() in {"result","outcome","status","win_loss"}),None)
    datecol=next((c for c in hist.columns if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower()),None)
    pnl=pd.to_numeric(hist[pnlcol],errors="coerce").fillna(0).sum() if pnlcol else 0.0
    wins=int(hist[resultcol].astype(str).str.upper().isin(["WIN","WON","TP1","TP2","PROFIT"]).sum()) if resultcol else 0
    total=len(hist);ratio=(wins/total*100) if total else 0
    today=len(hist) if not datecol else int(pd.to_datetime(hist[datecol],errors="coerce",utc=True).dt.date.eq(datetime.now(timezone.utc).date()).sum())
    x1,x2,x3,x4=st.columns(4);x1.metric("TOTAL TRADES",total);x2.metric("TOTAL PNL",money(pnl));x3.metric("WIN RATIO",f"{ratio:.1f}%");x4.metric("TRADES TODAY",today)
    st.dataframe(hist.tail(100).iloc[::-1],use_container_width=True,hide_index=True)
else:
    st.warning("No trade/backtest history file found yet. Live signals will still run.")

st.caption(f"Last loop: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} • Auto refresh: {refresh}s")
time.sleep(refresh)
st.rerun()
