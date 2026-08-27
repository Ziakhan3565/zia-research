from __future__ import annotations
import json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='ZIA Research Terminal', page_icon='⚡', layout='wide', initial_sidebar_state='collapsed')
ROOT=Path(__file__).resolve().parent
HOST='https://fapi.binance.com'
MODEL_FILE=ROOT/'xgboost_obi_model.pkl'
LOCK_FILE=ROOT/'.signal_locks.json'
SYMBOLS=['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','SUIUSDT','TRXUSDT','LTCUSDT']
TFS={'5M':'5m','15M':'15m','30M':'30m','1H':'1h','4H':'4h'}
FEATURES=['top20_bid_sum','top20_ask_sum','obi_5','obi_10','obi_20','obi_50','spread','spread_pct','bid_ask_ratio_20','bid_ask_ratio_50','top20_total_depth','top50_total_depth','taker_buy_volume','taker_sell_volume','taker_flow','taker_flow_ratio','price_return','price_change','sma_distance','realized_volatility','BOOK_IMB','QUANT_IMPLY','ADAPT_CONF','BAYESIAN','FOURIER_TREND']

st.markdown('''<style>
.stApp{background:#060910;color:#edf3fb}.block-container{max-width:1800px;padding:18px 28px 60px}.hero{padding:8px 0 18px;border-bottom:1px solid #202a38;margin-bottom:16px}.brand{font-size:34px;font-weight:900}.muted{color:#8290a4;font-size:11px}.panel{background:#0b111a;border:1px solid #202c3b;border-radius:16px;padding:16px;margin:10px 0}.section{font-size:19px;font-weight:900}.sub{font-size:11px;color:#7f8da1}.trade{background:#0d151f;border:1px solid #263444;border-radius:14px;padding:14px;margin:7px 0}.long{border-color:#247b59}.short{border-color:#9a4053}.sig{font-size:26px;font-weight:950}.long .sig{color:#4be0a2}.short .sig{color:#ff7185}.pill{display:inline-block;background:#172231;border:1px solid #29384a;border-radius:999px;padding:3px 8px;font-size:10px;font-weight:800}.metric{font-size:12px;color:#8a98aa}.big{font-size:20px;font-weight:900;color:#eef5ff}.lock{color:#65d7ff;font-weight:800;font-size:11px}
</style>''',unsafe_allow_html=True)

def api(path,params):
    try:
        r=requests.get(HOST+path,params=params,timeout=6,headers={'User-Agent':'ZIA-Research'}); r.raise_for_status(); return r.json()
    except Exception:return None

@st.cache_data(ttl=5,show_spinner=False)
def candles(symbol,interval,limit=250):
    raw=api('/fapi/v1/klines',{'symbol':symbol,'interval':interval,'limit':limit})
    if not isinstance(raw,list):return pd.DataFrame()
    return pd.DataFrame([[pd.to_datetime(x[0],unit='ms',utc=True),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5]),float(x[9])] for x in raw],columns=['Time','Open','High','Low','Close','Volume','TakerBuy'])

@st.cache_data(ttl=3,show_spinner=False)
def orderbook(symbol):
    raw=api('/fapi/v1/depth',{'symbol':symbol,'limit':50})
    try:return np.asarray(raw['bids'],float),np.asarray(raw['asks'],float)
    except Exception:return np.empty((0,2)),np.empty((0,2))

@st.cache_resource(show_spinner=False)
def model():
    try:return joblib.load(MODEL_FILE)
    except Exception:return None

def obi(b,a,k):
    k=min(k,len(b),len(a));
    if not k:return 0.,0.,0.
    bv=float(b[:k,1].sum());av=float(a[:k,1].sum());return ((bv-av)/(bv+av) if bv+av else 0.),bv,av

def make_signal(symbol,tf):
    df=candles(symbol,TFS[tf]);b,a=orderbook(symbol)
    if df.empty or len(df)<35:return None
    p=float(df.Close.iloc[-1]);sma=float(df.Close.rolling(20).mean().iloc[-1])
    o5,_,_=obi(b,a,5);o10,_,_=obi(b,a,10);o20,b20,a20=obi(b,a,20);o50,b50,a50=obi(b,a,50)
    spread=float(a[0,0]-b[0,0]) if len(a) and len(b) else 0
    vol=float(df.Volume.tail(20).sum());buy=float(df.TakerBuy.tail(20).sum());sell=max(vol-buy,0);flow=buy-sell;flowr=flow/max(vol,1e-9)
    ret=float(df.Close.pct_change().tail(8).mean());trend=float(np.tanh((p/sma-1)*100)) if sma else 0;rv=float(df.Close.pct_change().tail(30).std())
    four=float(np.tanh(df.Close.pct_change().tail(16).mean()*1000))
    research=float(np.clip(.38*o20+.17*o50+.20*trend+.15*flowr+.10*ret*100,-1,1))
    row=dict(zip(FEATURES,[b20,a20,o5,o10,o20,o50,spread,spread/max(p,1e-9),b20/max(a20,1e-9),b50/max(a50,1e-9),b20+a20,b50+a50,buy,sell,flow,flowr,float(df.Close.pct_change().iloc[-1]),float(df.Close.iloc[-1]-df.Close.iloc[-2]),p/sma-1 if sma else 0,rv,o20,float(np.tanh((o20+o50+trend)/3)),float(np.clip(.5+(abs(o20)+abs(trend))/2,0,1)),float(np.clip(.5+(o20+trend)/4,0,1)),four]))
    m=model();ml=None;ml_status='MODEL OFF'
    if m is not None:
        try:
            names=list(m.get_booster().feature_names or []) if hasattr(m,'get_booster') else []
            if not names:names=FEATURES
            if any(x not in row for x in names):raise ValueError('missing model feature')
            x=pd.DataFrame([[row[x] for x in names]],columns=names)
            if not np.isfinite(x.to_numpy(float)).all():raise ValueError('non-finite feature')
            ml=float(m.predict_proba(x)[0][-1]) if hasattr(m,'predict_proba') else None;ml_status='ML OK'
        except Exception as e:ml_status='ML ERROR: '+str(e)[:45]
    score=research if ml is None else float(np.clip(.65*research+.35*(ml-.5)*2,-1,1))
    direction='LONG' if score>=.30 else 'SHORT' if score<=-.30 else 'WAIT'
    atr=float((df.High-df.Low).tail(14).mean())
    if direction=='LONG':tp1,tp2,sl=p+atr,p+2*atr,p-.75*atr
    elif direction=='SHORT':tp1,tp2,sl=p-atr,p-2*atr,p+.75*atr
    else:tp1,tp2,sl=p+atr,p+2*atr,p-.75*atr
    return {'symbol':symbol,'tf':tf,'direction':direction,'confidence':float(np.clip(50+abs(score)*49,1,99)),'entry':p,'tp1':tp1,'tp2':tp2,'sl':sl,'score':score,'research':research,'ml':ml,'ml_status':ml_status,'created':datetime.now(timezone.utc).isoformat(),'atr':atr}

def locks():
    try:return json.loads(LOCK_FILE.read_text())
    except Exception:return {}

def save_locks(x):
    try:LOCK_FILE.write_text(json.dumps(x,indent=2))
    except Exception:pass

def locked_15m(symbol):
    d=locks();key=symbol;now=datetime.now(timezone.utc);old=d.get(key)
    if old:
        try:
            until=datetime.fromisoformat(old['until'])
            if now<until:
                old['remaining']=int((until-now).total_seconds());return old
            d.pop(key,None);save_locks(d)
        except Exception:d.pop(key,None);save_locks(d)
    s=make_signal(symbol,'15M')
    if s and s['direction'] in ('LONG','SHORT'):
        until=now+timedelta(minutes=20);s['until']=until.isoformat();s['remaining']=1200;d[key]=s;save_locks(d)
    return s

def price(x):return '—' if x is None else (f'{x:,.2f}' if abs(float(x))>=1 else f'{x:,.6f}')

def card(s):
    if not s:return
    c='long' if s['direction']=='LONG' else 'short' if s['direction']=='SHORT' else ''
    rem=max(0,int(s.get('remaining',0)));lock=f'{rem//60:02d}:{rem%60:02d}' if rem else 'LIVE'
    st.markdown(f'''<div class="trade {c}"><div><b>{s['symbol']}</b> <span class="pill">{s['tf']}</span></div><div class="sig">{s['direction']}</div><div class="lock">{lock} {'LOCKED' if rem else ''} • {s['confidence']:.0f}% confidence</div><div class="metric">ENTRY <span class="big">{price(s['entry'])}</span> &nbsp; TP1 <span class="big">{price(s['tp1'])}</span> &nbsp; TP2 <span class="big">{price(s['tp2'])}</span> &nbsp; SL <span class="big">{price(s['sl'])}</span></div><div class="sub">ML: {'—' if s.get('ml') is None else f"{s['ml']*100:.1f}%"} • Score {s['score']:.3f}</div></div>''',unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="brand">⚡ ZIA RESEARCH TERMINAL</div><div class="muted">LIVE MARKET • ML • ORDER FLOW • TRADE MONITOR</div></div>',unsafe_allow_html=True)
with st.sidebar:
    st.header('Controls');refresh=st.slider('Auto refresh',5,60,10);sym=st.selectbox('Chart / analysis',SYMBOLS);tf=st.selectbox('Timeframe',list(TFS),index=1)
    if st.button('Refresh now',use_container_width=True):st.cache_data.clear();st.rerun()

# KPI strip
active=[]
for x in SYMBOLS:
    s=locked_15m(x)
    if s and s['direction'] in ('LONG','SHORT'):active.append(s)
cols=st.columns(5);cols[0].metric('ACTIVE 15M TRADES',len(active));cols[1].metric('LONG',sum(x['direction']=='LONG' for x in active));cols[2].metric('SHORT',sum(x['direction']=='SHORT' for x in active));cols[3].metric('ML MODEL','ONLINE' if model() is not None else 'OFF');cols[4].metric('MARKET','BINANCE FUTURES')

st.markdown('<div class="panel"><div class="section">🔥 LIVE TRADES</div><div class="sub">Every loop scans the tracked markets. 15M LONG/SHORT signals remain fixed for 20 minutes.</div></div>',unsafe_allow_html=True)
if active:
    cs=st.columns(3)
    for i,s in enumerate(active):
        with cs[i%3]:card(s)
else:st.info('No active 15M LONG/SHORT trade right now.')

st.markdown('<div class="panel"><div class="section">📈 TRADINGVIEW-STYLE MARKET CHART</div><div class="sub">Candles, pan/zoom, EMA trend and forward projection zone.</div></div>',unsafe_allow_html=True)
df=candles(sym,TFS[tf])
if not df.empty:
    fig=go.Figure();fig.add_trace(go.Candlestick(x=df.Time,open=df.Open,high=df.High,low=df.Low,close=df.Close,name='Price'))
    for n in (10,20,50,200):
        if len(df)>=n:fig.add_trace(go.Scatter(x=df.Time,y=df.Close.rolling(n).mean(),name=f'EMA {n}',mode='lines'))
    last=float(df.Close.iloc[-1]);atr=float((df.High-df.Low).tail(14).mean());future_x=pd.date_range(df.Time.iloc[-1],periods=9,freq=pd.Timedelta(TFS[tf]));
    fig.add_trace(go.Scatter(x=future_x,y=np.linspace(last,last+atr*2,9),name='Forward upside guide',mode='lines',line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=future_x,y=np.linspace(last,last-atr*2,9),name='Forward downside guide',mode='lines',line=dict(dash='dash')))
    fig.update_layout(height=620,template='plotly_dark',xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=35,b=10),hovermode='x unified')
    st.plotly_chart(fig,use_container_width=True,config={'scrollZoom':True,'displaylogo':False})
else:st.warning('Market data unavailable.')

st.markdown('<div class="panel"><div class="section">🧠 ML / MODEL MONITOR</div><div class="sub">Live feature construction and model status for the selected market.</div></div>',unsafe_allow_html=True)
selected=make_signal(sym,tf)
if selected:
    a,b,c,d,e=st.columns(5);a.metric('SIGNAL',selected['direction']);b.metric('CONFIDENCE',f"{selected['confidence']:.1f}%");c.metric('ML PROB', '—' if selected['ml'] is None else f"{selected['ml']*100:.1f}%");d.metric('RESEARCH SCORE',f"{selected['research']:.3f}");e.metric('MODEL STATUS',selected['ml_status'])

st.markdown('<div class="panel"><div class="section">📚 TRADE HISTORY & PERFORMANCE</div><div class="sub">Closed-trade CSVs are read separately from active signal locks.</div></div>',unsafe_allow_html=True)
frames=[]
for p in (ROOT/'trade_history.csv',ROOT/'backtest_trade_history.csv'):
    if p.exists():
        try:q=pd.read_csv(p);q['Source']=p.name;frames.append(q)
        except Exception:pass
hist=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
if not hist.empty:
    low={c.lower():c for c in hist.columns};pc=next((low[x] for x in ('pnl','profit','profit_loss','net_pnl') if x in low),None);rc=next((low[x] for x in ('result','outcome','status','win_loss') if x in low),None);dc=next((low[x] for x in ('timestamp','time','date','created') if x in low),None)
    pnl=float(pd.to_numeric(hist[pc],errors='coerce').fillna(0).sum()) if pc else 0;wins=closed=0
    if rc:
        r=hist[rc].astype(str).str.upper();wins=int(r.isin(['WIN','WON','TP1','TP2','PROFIT']).sum());closed=int(r.isin(['WIN','WON','TP1','TP2','PROFIT','LOSS','LOST','SL']).sum())
    total=len(hist);wr=wins/(closed or total)*100 if total else 0;today=total
    if dc:
        z=pd.to_datetime(hist[dc],errors='coerce',utc=True);today=int(z.dt.date.eq(datetime.now(timezone.utc).date()).sum())
    a,b,c,d=st.columns(4);a.metric('TOTAL TRADES',total);b.metric('PNL',price(pnl));c.metric('WIN RATE',f'{wr:.1f}%');d.metric('TRADES TODAY',today);st.dataframe(hist.tail(200).iloc[::-1],use_container_width=True,hide_index=True)
else:st.info('No trade-history CSV found yet.')

st.caption(f'Last update: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")} • Auto refresh {refresh}s')
time.sleep(refresh);st.rerun()
