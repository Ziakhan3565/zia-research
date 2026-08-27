from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="ZIA Research Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "xgboost_obi_model.pkl"
LOCKS = ROOT / ".signal_locks.json"
HOST = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT","TRXUSDT","LTCUSDT"]
TFS = {"5M":"5m", "15M":"15m", "30M":"30m", "1H":"1h", "4H":"4h"}
# User requested fixed signal windows: 15M = 20 minutes, 1H = 2.5 hours.
LOCK_MINUTES = {"15M": 20, "1H": 150}

st.markdown("""
<style>
.stApp{background:#05070b;color:#edf3fb}.block-container{max-width:1900px;padding:16px 24px 50px}
.hero{padding:8px 0 16px;border-bottom:1px solid #202b3a;margin-bottom:14px}.brand{font-size:34px;font-weight:900}.muted{color:#8190a5;font-size:11px}
.panel{background:#0b1119;border:1px solid #1d2938;border-radius:16px;padding:15px;margin:10px 0}.title{font-size:19px;font-weight:900}.sub{font-size:10px;color:#8190a5;margin-top:3px}
.sig{border-radius:16px;padding:14px;border:1px solid #263344;background:#0d151f;margin-bottom:12px}.long{border-color:#267653}.short{border-color:#913d50}.wait{border-color:#66572a}
.direction{font-size:28px;font-weight:1000;margin:4px 0}.long .direction{color:#4be0a2}.short .direction{color:#ff7185}.wait .direction{color:#f3c86a}
.metric{background:#101925;border:1px solid #202d3e;border-radius:10px;padding:7px;margin-top:7px}.mlabel{font-size:8px;color:#7f8da1;font-weight:900}.mvalue{font-size:13px;font-weight:900;margin-top:2px}.lock{font-size:10px;color:#65d7ff;font-weight:900;margin-top:7px}
.badge{display:inline-block;padding:3px 7px;border-radius:8px;background:#172231;font-size:9px;font-weight:900}.good{color:#4be0a2}.danger{color:#ff7185}
</style>
""", unsafe_allow_html=True)


def api(path, params):
    try:
        r = requests.get(HOST + path, params=params, timeout=5, headers={"User-Agent":"ZIA-Research"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=3, show_spinner=False)
def candles(symbol, interval, limit=250):
    raw = api("/fapi/v1/klines", {"symbol":symbol,"interval":interval,"limit":limit})
    if not isinstance(raw, list): return pd.DataFrame()
    return pd.DataFrame([[pd.to_datetime(x[0],unit="ms",utc=True),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5]),float(x[9])] for x in raw], columns=["Time","Open","High","Low","Close","Volume","TakerBuy"])


@st.cache_data(ttl=3, show_spinner=False)
def book(symbol):
    raw = api("/fapi/v1/depth", {"symbol":symbol,"limit":50})
    try: return np.asarray(raw["bids"],float), np.asarray(raw["asks"],float)
    except Exception: return np.empty((0,2)), np.empty((0,2))


@st.cache_resource(show_spinner=False)
def load_model():
    try: return joblib.load(MODEL_FILE)
    except Exception: return None


def obi(b,a,k):
    k=min(k,len(b),len(a))
    if not k:return 0.,0.,0.
    bv=float(b[:k,1].sum()); av=float(a[:k,1].sum())
    return ((bv-av)/(bv+av) if bv+av else 0.),bv,av


def model_feature_names(m):
    names=list(m.get_booster().feature_names or []) if hasattr(m,"get_booster") else []
    if names:return names
    return ["top20_bid_sum","top20_ask_sum","obi_5","obi_10","obi_20","obi_50","spread","spread_pct","bid_ask_ratio_20","bid_ask_ratio_50","top20_total_depth","top50_total_depth","taker_buy_volume","taker_sell_volume","taker_flow","taker_flow_ratio","price_return","price_change","sma_distance","realized_volatility","BOOK_IMB","QUANT_IMPLY","ADAPT_CONF","BAYESIAN","FOURIER_TREND"]


def calculate(symbol, tf):
    df=candles(symbol,TFS[tf]); b,a=book(symbol)
    if df.empty or len(df)<25:return None
    price=float(df.Close.iloc[-1]); sma=float(df.Close.rolling(20).mean().iloc[-1])
    o5,_,_=obi(b,a,5); o10,_,_=obi(b,a,10); o20,b20,a20=obi(b,a,20); o50,b50,a50=obi(b,a,50)
    spread=float(a[0,0]-b[0,0]) if len(a) and len(b) else 0.
    total=float(df.Volume.tail(20).sum()); buy=float(df.TakerBuy.tail(20).sum()); sell=max(total-buy,0.)
    taker_flow=buy-sell; taker_ratio=taker_flow/max(total,1e-9)
    ret=float(df.Close.pct_change().tail(8).mean()); trend=float(np.tanh((price/sma-1)*100)) if sma else 0.
    rv=float(df.Close.pct_change().tail(30).std()); four=float(np.tanh(df.Close.pct_change().tail(16).mean()*1000))
    # Research score.
    research=float(np.clip(.38*o20+.17*o50+.20*trend+.15*taker_ratio+.10*ret*100,-1,1))
    row={"top20_bid_sum":b20,"top20_ask_sum":a20,"obi_5":o5,"obi_10":o10,"obi_20":o20,"obi_50":o50,
         "spread":spread,"spread_pct":spread/max(price,1e-9),"bid_ask_ratio_20":b20/max(a20,1e-9),"bid_ask_ratio_50":b50/max(a50,1e-9),
         "top20_total_depth":b20+a20,"top50_total_depth":b50+a50,"taker_buy_volume":buy,"taker_sell_volume":sell,
         "taker_flow":taker_flow,"taker_flow_ratio":taker_ratio,"price_return":float(df.Close.pct_change().iloc[-1]),
         "price_change":float(df.Close.iloc[-1]-df.Close.iloc[-2]),"sma_distance":price/sma-1 if sma else 0.,"realized_volatility":rv,
         "BOOK_IMB":o20,"QUANT_IMPLY":float(np.tanh((o20+o50+trend)/3)),"ADAPT_CONF":float(np.clip(.5+(abs(o20)+abs(trend))/2,0,1)),
         "BAYESIAN":float(np.clip(.5+(o20+trend)/4,0,1)),"FOURIER_TREND":four}
    m=load_model(); ml_prob=None; ml_status="MODEL OFF"
    if m is not None:
        try:
            names=model_feature_names(m); expected=int(getattr(m,"n_features_in_",len(names)))
            if len(names)!=expected: raise ValueError(f"model expects {expected}, metadata has {len(names)}")
            missing=[x for x in names if x not in row]
            if missing: raise ValueError("missing feature: "+missing[0])
            x=pd.DataFrame([[row[x] for x in names]],columns=names)
            if not np.isfinite(x.to_numpy(dtype=float)).all(): raise ValueError("non-finite feature")
            if hasattr(m,"predict_proba"): ml_prob=float(m.predict_proba(x)[0][-1])
            ml_status="ML OK"
        except Exception as e: ml_status="ML ERROR: "+str(e)[:55]
    combined=research if ml_prob is None else float(np.clip(.65*research+.35*((ml_prob-.5)*2),-1,1))
    direction="LONG" if combined>=.30 else "SHORT" if combined<=-.30 else "WAIT"
    confidence=float(np.clip(50+abs(combined)*49,1,99))
    atr=float((df.High-df.Low).tail(14).mean())
    if direction=="LONG": tp1,tp2,sl=price+atr,price+2*atr,price-.75*atr
    elif direction=="SHORT": tp1,tp2,sl=price-atr,price-2*atr,price+.75*atr
    else: tp1,tp2,sl=price+atr,price+2*atr,price-.75*atr
    return {"symbol":symbol,"tf":tf,"direction":direction,"confidence":confidence,"ml":ml_prob,"ml_status":ml_status,"entry":price,"tp1":tp1,"tp2":tp2,"sl":sl,"score":combined,"research":research,"created":datetime.now(timezone.utc).isoformat()}


def read_locks():
    try:return json.loads(LOCKS.read_text())
    except Exception:return {}


def write_locks(x):
    try: LOCKS.write_text(json.dumps(x,indent=2))
    except Exception: pass


def locked_signal(symbol,tf):
    locks=read_locks(); key=f"{symbol}:{tf}"; now=datetime.now(timezone.utc); old=locks.get(key)
    if old and tf in LOCK_MINUTES:
        try:
            until=datetime.fromisoformat(old["until"])
            if now<until:
                old["remaining"]=max(0,int((until-now).total_seconds())); return old
        except Exception: pass
    s=calculate(symbol,tf)
    if s is None:return None
    # Only actionable LONG/SHORT signals are locked. WAIT remains live.
    if tf in LOCK_MINUTES and s["direction"] in ("LONG","SHORT"):
        until=now+timedelta(minutes=LOCK_MINUTES[tf]); s["until"]=until.isoformat(); s["remaining"]=LOCK_MINUTES[tf]*60; locks[key]=s; write_locks(locks)
    return s


def fmt(x):
    if x is None:return "—"
    return f"{x:,.2f}" if abs(float(x))>=1 else f"{x:,.6f}"


def render_card(s):
    cls="long" if s["direction"]=="LONG" else "short" if s["direction"]=="SHORT" else "wait"
    rem=int(s.get("remaining",0)); lock=f"LOCKED • {rem//3600:02d}:{(rem%3600)//60:02d}:{rem%60:02d}" if rem>0 else "LIVE"
    ml="—" if s.get("ml") is None else f"{s['ml']*100:.1f}%"
    st.markdown(f'''<div class="sig {cls}"><div style="display:flex;justify-content:space-between"><b>{s['symbol']}</b><span class="badge">{s['tf']}</span></div><div class="direction">{s['direction']}</div><div class="lock">{lock} • {s['confidence']:.0f}% CONFIDENCE</div><div class="metric"><span class="mlabel">ENTRY / RATE</span><div class="mvalue">{fmt(s['entry'])}</div></div><div class="metric"><span class="mlabel">TP1</span><div class="mvalue">{fmt(s['tp1'])}</div></div><div class="metric"><span class="mlabel">TP2</span><div class="mvalue">{fmt(s['tp2'])}</div></div><div class="metric"><span class="mlabel">STOP LOSS</span><div class="mvalue">{fmt(s['sl'])}</div></div><div class="sub">ML: {ml} • Score: {s['score']:.3f}</div></div>''',unsafe_allow_html=True)


st.markdown('<div class="hero"><div class="brand">⚡ ZIA RESEARCH TERMINAL</div><div class="muted">LIVE MULTI-CRYPTO SIGNAL ENGINE • 15M / 1H LOCKED SIGNALS</div></div>',unsafe_allow_html=True)
with st.sidebar:
    st.header("Controls")
    refresh=st.slider("Auto refresh (seconds)",5,60,10)
    selected_tf=st.selectbox("Detailed timeframe",list(TFS),index=1)
    selected_symbol=st.selectbox("Detailed crypto",SYMBOLS,index=0)
    if st.button("Refresh now",use_container_width=True): st.cache_data.clear(); st.rerun()

# Always scan both requested actionable timeframes. This is independent of the detail selector.
all15={s:locked_signal(s,"15M") for s in SYMBOLS}
all1h={s:locked_signal(s,"1H") for s in SYMBOLS}

st.markdown('<div class="panel"><div class="title">🔴 LIVE SIGNALS — ALL CRYPTOCURRENCIES</div><div class="sub">Every loop scans every tracked crypto. 15M LONG/SHORT is fixed for 20 minutes. 1H LONG/SHORT is fixed for 2 hours 30 minutes. WAIT is not artificially locked.</div></div>',unsafe_allow_html=True)
tab15,tab1h=st.tabs(["15M • 20 MIN LOCK","1H • 2H 30M LOCK"])
for tab,data in ((tab15,all15),(tab1h,all1h)):
    with tab:
        cols=st.columns(3)
        for i,s in enumerate(data.values()):
            if s:
                with cols[i%3]: render_card(s)

st.markdown('<div class="panel"><div class="title">🎯 SELECTED SIGNAL</div><div class="sub">The selected market/timeframe uses the same locked signal object as the live board.</div></div>',unsafe_allow_html=True)
detail=locked_signal(selected_symbol,selected_tf)
if detail:
    a,b,c,d,e,f=st.columns(6)
    a.metric("SIGNAL",detail["direction"]); b.metric("ENTRY / RATE",fmt(detail["entry"])); c.metric("TP1",fmt(detail["tp1"])); d.metric("TP2",fmt(detail["tp2"])); e.metric("STOP",fmt(detail["sl"])); f.metric("CONFIDENCE",f"{detail['confidence']:.1f}%")
    lock_seconds=int(detail.get("remaining",0)); lock_text=f"{lock_seconds//3600:02d}:{(lock_seconds%3600)//60:02d}:{lock_seconds%60:02d}" if lock_seconds else "Not locked"
    st.info(f"Lock remaining: {lock_text}  |  ML: {('N/A' if detail['ml'] is None else f'{detail['ml']*100:.1f}%')}  |  {detail['ml_status']}  |  Score: {detail['score']:.3f}")
else: st.warning("No market data available for this selection.")

st.markdown('<div class="panel"><div class="title">📊 TRADE HISTORY / PNL</div><div class="sub">Loads existing trade-history CSVs without mixing saved signal snapshots into closed-trade PNL.</div></div>',unsafe_allow_html=True)
paths=[ROOT/"trade_history.csv",ROOT/"backtest_trade_history.csv"]
frames=[]
for p in paths:
    if p.exists():
        try:
            q=pd.read_csv(p); q["Source"]=p.name; frames.append(q)
        except Exception: pass
hist=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
if not hist.empty:
    low={c.lower():c for c in hist.columns}
    pnlcol=next((low[x] for x in ["pnl","profit","profit_loss","net_pnl"] if x in low),None)
    resultcol=next((low[x] for x in ["result","outcome","status","win_loss"] if x in low),None)
    datecol=next((low[x] for x in ["timestamp","time","date","created"] if x in low),None)
    pnl=pd.to_numeric(hist[pnlcol],errors="coerce").fillna(0).sum() if pnlcol else 0.
    wins=0; closed=0
    if resultcol:
        r=hist[resultcol].astype(str).str.upper(); wins=int(r.isin(["WIN","WON","TP1","TP2","PROFIT"]).sum()); closed=int(r.isin(["WIN","WON","TP1","TP2","PROFIT","LOSS","LOST","SL"]).sum())
    total=len(hist); denom=closed or total; winrate=(wins/denom*100) if denom else 0.
    today=total
    if datecol:
        dtv=pd.to_datetime(hist[datecol],errors="coerce",utc=True); today=int(dtv.dt.date.eq(datetime.now(timezone.utc).date()).sum())
    x1,x2,x3,x4=st.columns(4); x1.metric("TOTAL RECORDS",total); x2.metric("PNL",fmt(pnl)); x3.metric("WIN RATE",f"{winrate:.1f}%"); x4.metric("TRADES TODAY",today)
    st.dataframe(hist.tail(150).iloc[::-1],use_container_width=True,hide_index=True)
else: st.warning("No closed-trade/backtest history CSV found yet.")

st.caption(f"Last loop: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} • Next refresh: {refresh}s")
time.sleep(refresh)
st.rerun()
