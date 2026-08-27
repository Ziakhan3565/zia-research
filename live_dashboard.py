from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import joblib, numpy as np, pandas as pd, requests, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="ZIA Research Live", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
ROOT=Path(__file__).resolve().parent; MODEL=ROOT/"xgboost_obi_model.pkl"; HISTORY=ROOT/"backtest_trade_history.csv"; SIGNALS=ROOT/"saved_signals.csv"
BASE="https://fapi.binance.com"; SYMBOLS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT"]
TFS={"5M":"5m","15M":"15m","30M":"30m","1H":"1h","4H":"4h"}
st.markdown('''<style>html,body,[data-testid="stAppViewContainer"]{background:#05070b;color:#edf3fb}.block-container{max-width:1900px;padding:12px 22px 35px}.panel,.metric{background:#0c131d;border:1px solid #1d2a39;border-radius:15px;padding:15px;margin-bottom:10px}.hero{display:flex;justify-content:space-between;border-bottom:1px solid #1d2a39;padding:5px 0 13px;margin-bottom:12px}.brand{font-size:34px;font-weight:950}.brand b{color:#969eff}.live{padding:7px 12px;border:1px solid #245d45;border-radius:999px;color:#6ce3a5;font-weight:900}.label{font-size:9px;color:#7f8da1;font-weight:900;letter-spacing:1px}.value{font-size:21px;font-weight:950;margin-top:4px}.muted{color:#8491a4;font-size:11px}.signal{border-radius:18px;padding:22px;text-align:center;border:1px solid #277b59;background:#081810;margin-bottom:10px}.signal.short{border-color:#843a4a;background:#19090e}.signal.wait{border-color:#705e30;background:#181308}.big{font-size:52px;font-weight:1000}.stButton>button{border-radius:10px;font-weight:900}</style>''',unsafe_allow_html=True)

def get(path,params):
    try:
        r=requests.get(BASE+path,params=params,timeout=3); return r.json() if r.ok else None
    except requests.RequestException:return None

def candles(symbol,interval,limit=220):
    x=get('/fapi/v1/klines',{'symbol':symbol,'interval':interval,'limit':limit})
    if not isinstance(x,list):return pd.DataFrame()
    return pd.DataFrame([[pd.to_datetime(int(a[0]),unit='ms',utc=True),*map(float,a[1:6])] for a in x],columns=['Time','Open','High','Low','Close','Volume'])

def book(symbol):
    x=get('/fapi/v1/depth',{'symbol':symbol,'limit':100})
    try:return np.asarray(x['bids'],float),np.asarray(x['asks'],float)
    except:return np.empty((0,2)),np.empty((0,2))

def obi(b,a,k=20):
    if len(b)<k or len(a)<k:return 0.,0.,0.
    bv=float(b[:k,1].sum());av=float(a[:k,1].sum());return (bv-av)/(bv+av) if bv+av else 0.,bv,av

def calc(symbol,tf):
    d=candles(symbol,TFS[tf]);b,a=book(symbol)
    if d.empty:return {'signal':'WAIT','confidence':50.,'entry':0.,'tp1':0.,'tp2':0.,'sl':0.,'rr':0.,'ml':None,'composite':0.}
    price=float(d.Close.iloc[-1]); sma=float(d.Close.rolling(20).mean().iloc[-1]); o20,b20,a20=obi(b,a,20); o50,_,_=obi(b,a,50)
    trend=np.tanh((price/sma-1)*100) if sma else 0.; taker=np.tanh(d.Close.pct_change().tail(5).mean()*500); research=.42*o20+.18*((o20+o50)/2)+.20*taker+.20*trend
    f={'top20_bid_sum':b20,'top20_ask_sum':a20,'obi_top20':o20,'spread':float(a[0,0]-b[0,0]) if len(a) and len(b) else 0.,'bid_ask_ratio':b20/(a20 or 1.),'total_depth':b20+a20,'trend_signal':price-sma}
    prob=None; pred=None
    if MODEL.exists():
        try:
            m=joblib.load(MODEL); names=list(m.get_booster().feature_names or []) if hasattr(m,'get_booster') else ['top20_bid_sum','top20_ask_sum','obi_top20','spread','bid_ask_ratio','total_depth','trend_signal']
            if all(x in f for x in names): pred=int(m.predict(pd.DataFrame([[f[x] for x in names]],columns=names))[0]); prob=float(m.predict_proba(pd.DataFrame([[f[x] for x in names]],columns=names))[0][1]) if hasattr(m,'predict_proba') else None
        except: pred=None
    ml=(prob-.5)*2 if prob is not None else 0.; composite=.6*research+.4*ml if pred is not None else research
    sig='LONG' if composite>=.45 else 'SHORT' if composite<=-.45 else 'WAIT'; conf=float(np.clip(50+abs(composite)*49,1,99)); risk=max(price*.0025,float(d.Close.pct_change().tail(30).std() or 0)*price*2)
    if sig=='LONG': sl,tp1,tp2=price-risk,price+risk*1.5,price+risk*3
    elif sig=='SHORT': sl,tp1,tp2=price+risk,price-risk*1.5,price-risk*3
    else: sl=tp1=tp2=price
    return {'signal':sig,'confidence':conf,'entry':price,'tp1':tp1,'tp2':tp2,'sl':sl,'rr':abs(tp2-price)/max(abs(price-sl),1e-9),'ml':prob,'composite':composite,'obi':o20}

def stats():
    frames=[]
    for p in (HISTORY,SIGNALS):
        if p.exists():
            try:frames.append(pd.read_csv(p))
            except:pass
    if not frames:return 0,0.,0.,0
    d=pd.concat(frames,ignore_index=True); result=d.get('result',pd.Series('',index=d.index)).astype(str).str.upper(); closed=int(((result=='WIN')|(result=='LOSS')).sum()); wins=int((result=='WIN').sum()); wr=wins/closed*100 if closed else 0
    pnl=0.
    for c in ('pnl','profit','PNL'):
        if c in d:pnl=float(pd.to_numeric(d[c],errors='coerce').fillna(0).sum());break
    today=pd.Timestamp.now(tz='UTC').date(); day=0
    for c in ('timestamp','entry_time','time'):
        if c in d:
            t=pd.to_datetime(d[c],errors='coerce',utc=True);day=int((t.dt.date==today).sum());break
    return closed,pnl,wr,day

def save(symbol,tf,r):
    pd.DataFrame([{**{'timestamp':datetime.now(timezone.utc).isoformat(),'symbol':symbol,'timeframe':tf},**r}]).to_csv(SIGNALS,mode='a',header=not SIGNALS.exists(),index=False)

if 'symbol' not in st.session_state:st.session_state.symbol='BTCUSDT'
if 'tf' not in st.session_state:st.session_state.tf='15M'
st.markdown('<div class="hero"><div><div class="brand">ZIA <b>RESEARCH</b></div><div class="muted">LIVE SIGNAL DESK • ML + ORDER FLOW • 15M SIGNAL LOCK</div></div><div class="live">● LIVE • 1S</div></div>',unsafe_allow_html=True)
c1,c2=st.columns([2,1]);
with c1:symbol=st.selectbox('MARKET',SYMBOLS,index=SYMBOLS.index(st.session_state.symbol),key='symbol')
with c2:tf=st.selectbox('TIMEFRAME',list(TFS),index=list(TFS).index(st.session_state.tf),key='tf')

@st.fragment(run_every='1s')
def live():
    now=datetime.now(timezone.utc)
    if tf=='15M':
        lock=st.session_state.get('lock15')
        if not lock or lock['symbol']!=symbol or lock['expires']<=now:
            r=calc(symbol,tf); st.session_state.lock15={'symbol':symbol,'data':r,'created':now,'expires':now+timedelta(minutes=20)}
        r=st.session_state.lock15['data']; expires=st.session_state.lock15['expires']; remaining=max(0,int((expires-now).total_seconds())); window=f"LOCKED • {remaining//60:02d}:{remaining%60:02d} REMAINING"
    else:r=calc(symbol,tf);window='LIVE • NO LOCK'
    mltext=f"{r['ml']*100:.1f}%" if r.get('ml') is not None else '—'; cls='' if r['signal']=='LONG' else 'short' if r['signal']=='SHORT' else 'wait'
    st.markdown(f'<div class="signal {cls}"><div class="muted">{tf} SIGNAL • {window}</div><div class="big">{r["signal"]}</div><div class="muted">Confidence {r["confidence"]:.1f}% • ML {mltext} • Composite {r["composite"]:+.3f}</div></div>',unsafe_allow_html=True)
    cols=st.columns(5)
    for col,(lab,val) in zip(cols,[('ENTRY / BTC RATE',r['entry']),('TP1',r['tp1']),('TP2',r['tp2']),('STOP LOSS',r['sl']),('R:R',r['rr'])]):
        with col:st.markdown(f'<div class="metric"><div class="label">{lab}</div><div class="value">{"$"+format(val,",.2f") if lab!="R:R" else format(val,".2f")+"R"}</div></div>',unsafe_allow_html=True)
    if st.button('💾 SAVE CURRENT SIGNAL',use_container_width=True):save(symbol,tf,r);st.success('Signal saved')
    a,b,c=st.tabs(['LIVE CHART','PERFORMANCE','HISTORY'])
    with a:
        d=candles(symbol,TFS[tf]);fig=go.Figure(go.Candlestick(x=d.Time,open=d.Open,high=d.High,low=d.Low,close=d.Close));fig.update_layout(height=560,margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor='#080d14',plot_bgcolor='#080d14',xaxis_rangeslider_visible=False);st.plotly_chart(fig,use_container_width=True)
    with b:
        closed,pnl,wr,day=stats();cols=st.columns(4)
        for col,lab,val in zip(cols,['TOTAL TRADES','TOTAL PNL','WIN RATE','TRADES TODAY'],[closed,f'${pnl:,.2f}',f'{wr:.1f}%',day]):
            with col:st.markdown(f'<div class="metric"><div class="label">{lab}</div><div class="value">{val}</div></div>',unsafe_allow_html=True)
        if HISTORY.exists():
            try:st.dataframe(pd.read_csv(HISTORY).tail(150).iloc[::-1],use_container_width=True,hide_index=True)
            except:pass
    with c:
        if SIGNALS.exists():
            try:st.dataframe(pd.read_csv(SIGNALS).tail(150).iloc[::-1],use_container_width=True,hide_index=True)
            except:pass
        else:st.info('No saved live signals yet.')
live()
