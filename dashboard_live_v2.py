from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import joblib, numpy as np, pandas as pd, requests, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="ZIA Research • Live Signals", page_icon="⚡", layout="wide")
ROOT=Path(__file__).resolve().parent; MODEL=ROOT/"xgboost_obi_model.pkl"; HISTORY=ROOT/"backtest_trade_history.csv"; LOG=ROOT/"saved_signals.csv"
BASE="https://fapi.binance.com"
SYMBOLS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT"]
TFS={"5M":"5m","15M":"15m","30M":"30m","1H":"1h","4H":"4h"}
LOCKS={"15M":20,"1H":150}

st.markdown('''<style>
html,body,[data-testid="stAppViewContainer"]{background:#05070b;color:#edf3fb}.block-container{max-width:1900px;padding:14px 24px 40px}.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #202c3b;padding:4px 0 14px;margin-bottom:14px}.brand{font-size:34px;font-weight:950}.brand b{color:#929cff}.live{padding:7px 12px;border:1px solid #27664b;border-radius:999px;color:#67dfa1;font-weight:900}.card{background:#0c131d;border:1px solid #1c2a38;border-radius:15px;padding:14px}.signal{border-radius:18px;padding:22px;text-align:center;border:1px solid #2b805c;background:#081811;margin-bottom:10px}.signal.short{border-color:#8b3d4d;background:#19090e}.signal.wait{border-color:#76602d;background:#191409}.big{font-size:52px;font-weight:1000}.label{font-size:9px;color:#8190a5;font-weight:900;letter-spacing:1px}.value{font-size:21px;font-weight:950;margin-top:4px}.row{background:#0b121b;border:1px solid #1b2938;border-radius:13px;padding:12px 14px;margin:5px 0}.row.long{border-left:5px solid #39d98a}.row.short{border-left:5px solid #e85b70}.row.wait{border-left:5px solid #d2a84b}.muted{color:#8491a4;font-size:11px}
</style>''',unsafe_allow_html=True)

def api(path,params):
    try:
        r=requests.get(BASE+path,params=params,timeout=4); return r.json() if r.ok else None
    except Exception:return None

def candles(sym,tf,limit=220):
    raw=api('/fapi/v1/klines',{'symbol':sym,'interval':TFS[tf],'limit':limit})
    if not isinstance(raw,list):return pd.DataFrame()
    return pd.DataFrame([[pd.to_datetime(int(x[0]),unit='ms',utc=True),*map(float,x[1:6])] for x in raw],columns=['Time','Open','High','Low','Close','Volume'])

def book(sym):
    raw=api('/fapi/v1/depth',{'symbol':sym,'limit':100})
    try:return np.asarray(raw['bids'],float),np.asarray(raw['asks'],float)
    except Exception:return np.empty((0,2)),np.empty((0,2))

def obi(b,a,k):
    if len(b)<k or len(a)<k:return 0.,0.,0.
    bv=float(b[:k,1].sum()); av=float(a[:k,1].sum()); return ((bv-av)/(bv+av) if bv+av else 0.),bv,av

def raw_signal(sym,tf):
    d=candles(sym,tf); b,a=book(sym)
    if d.empty:return dict(signal='WAIT',confidence=50.,entry=0.,tp1=0.,tp2=0.,sl=0.,rr=0.,ml=None,composite=0.,status='NO DATA')
    price=float(d.Close.iloc[-1]); sma=float(d.Close.rolling(20).mean().iloc[-1]); o20,b20,a20=obi(b,a,20); o50,_,_=obi(b,a,50)
    trend=float(np.tanh((price/sma-1)*100)) if sma else 0.; flow=float(np.tanh(d.Close.pct_change().tail(5).mean()*500)); research=.42*o20+.18*((o20+o50)/2)+.20*flow+.20*trend
    f={'top20_bid_sum':b20,'top20_ask_sum':a20,'obi_top20':o20,'spread':float(a[0,0]-b[0,0]) if len(a) and len(b) else 0.,'bid_ask_ratio':b20/(a20 or 1.),'total_depth':b20+a20,'trend_signal':price-sma}
    pred=None; prob=None; status='MODEL NOT FOUND'
    if MODEL.exists():
        try:
            m=joblib.load(MODEL); names=list(m.get_booster().feature_names or []) if hasattr(m,'get_booster') else ['top20_bid_sum','top20_ask_sum','obi_top20','spread','bid_ask_ratio','total_depth','trend_signal']; expected=int(getattr(m,'n_features_in_',len(names)))
            if len(names)!=expected or any(n not in f for n in names): status='ML SCHEMA ERROR'
            else:
                x=pd.DataFrame([[f[n] for n in names]],columns=names); pred=int(m.predict(x)[0]); prob=float(m.predict_proba(x)[0][1]) if hasattr(m,'predict_proba') else None; status='OK'
        except Exception as e:status='ML ERROR'
    ml=float(np.clip((prob-.5)*2,-1,1)) if prob is not None else 0.; composite=float(.6*research+.4*ml) if pred is not None else float(research)
    sig='LONG' if composite>=.45 else 'SHORT' if composite<=-.45 else 'WAIT'; conf=float(np.clip(50+abs(composite)*49,1,99)); risk=max(price*.0025,float(d.Close.pct_change().tail(30).std() or 0)*price*2)
    if sig=='LONG':sl,tp1,tp2=price-risk,price+risk*1.5,price+risk*3
    elif sig=='SHORT':sl,tp1,tp2=price+risk,price-risk*1.5,price-risk*3
    else:sl=tp1=tp2=price
    return dict(signal=sig,confidence=conf,entry=price,tp1=tp1,tp2=tp2,sl=sl,rr=abs(tp2-price)/max(abs(price-sl),1e-9),ml=prob,composite=composite,status=status)

def locked(sym,tf):
    mins=LOCKS.get(tf); now=datetime.now(timezone.utc); key=f'lock_{sym}_{tf}'
    if not mins:return raw_signal(sym,tf),None
    state=st.session_state.get(key)
    if state is None or state['expires']<=now:
        state={'data':raw_signal(sym,tf),'created':now,'expires':now+timedelta(minutes=mins)}; st.session_state[key]=state
    return state['data'],state['expires']

def history_stats():
    frames=[]
    for p in (HISTORY,LOG):
        if p.exists():
            try:frames.append(pd.read_csv(p))
            except Exception:pass
    if not frames:return 0,0.,0.,0
    d=pd.concat(frames,ignore_index=True); result=d.get('result',pd.Series('',index=d.index)).astype(str).str.upper(); closed=int(result.isin(['WIN','LOSS']).sum()); wins=int((result=='WIN').sum()); wr=wins/closed*100 if closed else 0.; pnl=0.
    for c in ('pnl','PNL','profit'):
        if c in d:pnl=float(pd.to_numeric(d[c],errors='coerce').fillna(0).sum());break
    today=datetime.now(timezone.utc).date(); trades_today=0
    for c in ('timestamp','entry_time','time'):
        if c in d:
            t=pd.to_datetime(d[c],errors='coerce',utc=True); trades_today=int((t.dt.date==today).sum());break
    return closed,pnl,wr,trades_today

def save_signal(sym,tf,r):
    row={"timestamp":datetime.now(timezone.utc).isoformat(),"symbol":sym,"timeframe":tf,**r}; pd.DataFrame([row]).to_csv(LOG,mode='a',header=not LOG.exists(),index=False)

st.markdown('<div class="hero"><div><div class="brand">ZIA <b>RESEARCH</b></div><div class="muted">MULTI-CRYPTO LIVE SIGNAL DESK • AUTO LOOP • ML + ORDER FLOW</div></div><div class="live">● LIVE • 1 SEC</div></div>',unsafe_allow_html=True)

@st.fragment(run_every='1s')
def board():
    st.markdown('### 🔴 Live LONG / SHORT Board — All Cryptocurrencies')
    for sym in SYMBOLS:
        r,exp=locked(sym,'15M'); remain=max(0,int((exp-datetime.now(timezone.utc)).total_seconds())) if exp else 0; timer=f'{remain//60:02d}:{remain%60:02d}' if exp else 'LIVE'; cls='long' if r['signal']=='LONG' else 'short' if r['signal']=='SHORT' else 'wait'; ml=f'{r["ml"]*100:.1f}%' if r.get('ml') is not None else '—'
        st.markdown(f'<div class="row {cls}"><div style="display:grid;grid-template-columns:1.2fr .8fr 1fr 1fr 1fr 1fr 1fr 1fr;gap:10px;align-items:center"><b>{sym}</b><b>{r["signal"]}</b><span>Conf {r["confidence"]:.1f}%</span><span>Entry ${r["entry"]:,.2f}</span><span>TP1 ${r["tp1"]:,.2f}</span><span>TP2 ${r["tp2"]:,.2f}</span><span>SL ${r["sl"]:,.2f}</span><span>ML {ml} • {timer}</span></div></div>',unsafe_allow_html=True)

board()
st.markdown('---')
c1,c2=st.columns([2,1])
with c1:symbol=st.selectbox('SELECT MARKET',SYMBOLS,key='selected_market')
with c2:tf=st.selectbox('SELECT TIMEFRAME',list(TFS),index=1,key='selected_tf')
r,exp=locked(symbol,tf); remain=max(0,int((exp-datetime.now(timezone.utc)).total_seconds())) if exp else 0; locktext=f'LOCKED • {remain//3600:02d}:{(remain%3600)//60:02d}:{remain%60:02d}' if exp else 'LIVE'; cls='' if r['signal']=='LONG' else 'short' if r['signal']=='SHORT' else 'wait'; ml=f'{r["ml"]*100:.1f}%' if r.get('ml') is not None else '—'
st.markdown(f'<div class="signal {cls}"><div class="muted">{symbol} • {tf} • {locktext}</div><div class="big">{r["signal"]}</div><div class="muted">Confidence {r["confidence"]:.1f}% • ML {ml} • Composite {r["composite"]:+.3f}</div></div>',unsafe_allow_html=True)
cols=st.columns(5)
for col,(lab,val) in zip(cols,[('ENTRY / RATE',r['entry']),('TP1',r['tp1']),('TP2',r['tp2']),('STOP LOSS',r['sl']),('R:R',r['rr'])]):
    with col:st.markdown(f'<div class="card"><div class="label">{lab}</div><div class="value">{"$"+format(val,",.2f") if lab!="R:R" else format(val,".2f")+"R"}</div></div>',unsafe_allow_html=True)
if st.button('💾 SAVE CURRENT SIGNAL',use_container_width=True):save_signal(symbol,tf,r);st.success('Saved to signal journal')

t1,t2,t3=st.tabs(['LIVE CHART','PERFORMANCE','SIGNAL HISTORY'])
with t1:
    d=candles(symbol,tf); fig=go.Figure(go.Candlestick(x=d.Time,open=d.Open,high=d.High,low=d.Low,close=d.Close)); fig.update_layout(height=560,margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor='#080d14',plot_bgcolor='#080d14',xaxis_rangeslider_visible=False); st.plotly_chart(fig,use_container_width=True)
with t2:
    total,pnl,wr,today=history_stats(); cols=st.columns(4)
    for col,lab,val in zip(cols,['TOTAL CLOSED','TOTAL PNL','WIN RATE','TRADES TODAY'],[total,f'${pnl:,.2f}',f'{wr:.1f}%',today]):
        with col:st.markdown(f'<div class="card"><div class="label">{lab}</div><div class="value">{val}</div></div>',unsafe_allow_html=True)
    if HISTORY.exists():
        try:st.dataframe(pd.read_csv(HISTORY).tail(200).iloc[::-1],use_container_width=True,hide_index=True)
        except Exception:pass
with t3:
    if LOG.exists():
        try:st.dataframe(pd.read_csv(LOG).tail(200).iloc[::-1],use_container_width=True,hide_index=True)
        except Exception:pass
    else:st.info('No live signal journal entries yet.')
