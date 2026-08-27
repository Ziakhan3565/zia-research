from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title='ZIA Research Terminal',page_icon='⚡',layout='wide',initial_sidebar_state='collapsed')
ROOT=Path(__file__).resolve().parent
MODEL_FILE=ROOT/'xgboost_obi_model.pkl'; SIGNAL_FILE=ROOT/'saved_signals.csv'; TRADE_FILE=ROOT/'trade_history.csv'
SYMBOLS=['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','SUIUSDT']
TFS={'1M':'1m','3M':'3m','5M':'5m','15M':'15m','30M':'30m','1H':'1h','2H':'2h','4H':'4h','6H':'6h','8H':'8h','12H':'12h','1D':'1d','3D':'3d','1W':'1w','1MO':'1M'}
TRI_COLORS={'BODY 50':'#8E98FF','UPPER 50':'#65D7FF','LOWER 50':'#F3C86A'}
FUT=['https://fapi.binance.com','https://fapi1.binance.com','https://fapi2.binance.com','https://fapi3.binance.com']; SPOT=['https://api.binance.com','https://api1.binance.com']; DATA=['https://data-api.binance.vision']

st.markdown('''<style>
:root{--bg:#05070b;--p:#0b1119;--p2:#101925;--line:#1d2a39;--txt:#edf3fb;--muted:#7d8ba0;--g:#42dda0;--r:#ff7184;--b:#8e98ff;--c:#65d7ff;--y:#f3c86a}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--txt)}[data-testid="stHeader"]{background:rgba(5,7,11,.78)}.block-container{max-width:1920px;padding:12px clamp(8px,1.8vw,32px) 40px}
.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding:5px 2px 13px;margin-bottom:11px}.brand{font-size:clamp(24px,2.8vw,38px);font-weight:950;letter-spacing:-1.7px}.brand b{color:var(--b)}.micro{color:var(--muted);font-size:9px;letter-spacing:1.5px;margin-top:5px}.live{border:1px solid #235c43;background:#071810;color:#6ce3a5;border-radius:999px;padding:8px 12px;font-size:10px;font-weight:900}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--g);box-shadow:0 0 12px var(--g);margin-right:6px}
.panel{background:linear-gradient(145deg,#111a26,#080e15);border:1px solid var(--line);border-radius:15px;padding:13px}.card{background:linear-gradient(145deg,#101925,#090f17);border:1px solid var(--line);border-radius:13px;padding:11px;min-height:80px}.label{font-size:9px;color:var(--muted);font-weight:900;letter-spacing:1.1px}.value{font-size:20px;font-weight:950;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sub{font-size:10px;color:#8794a8;margin-top:3px}.title{font-size:18px;font-weight:950;margin:8px 0 2px}.subtitle{font-size:10px;color:var(--muted);margin-bottom:9px}.signal{border:1px solid var(--line);border-radius:18px;padding:17px;background:radial-gradient(circle at 85% 15%,rgba(142,152,255,.12),transparent 38%),linear-gradient(145deg,#111b28,#080e15)}.signal-label{font-size:9px;color:var(--muted);font-weight:950;letter-spacing:1.5px}.signal-main{font-size:clamp(42px,5vw,70px);font-weight:1000;letter-spacing:-3px;line-height:.95;margin:7px 0}.long{color:var(--g)}.short{color:var(--r)}.wait{color:var(--y)}.small{font-size:9px;color:var(--muted)}.tri{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:4px 8px;margin:2px;color:#cbd5e1;background:#0a111a;font-size:9px;font-weight:800}div[data-testid="stTabs"] button{font-size:11px;font-weight:900}div[data-testid="stTabs"] [aria-selected="true"]{color:var(--b)}.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:900}
@media(max-width:700px){.block-container{padding:7px}.brand{font-size:22px}.micro{font-size:7px}.live{font-size:8px;padding:6px 8px}.panel{padding:9px}.card{padding:8px;min-height:65px}.value{font-size:16px}.title{font-size:14px}}
</style>''',unsafe_allow_html=True)

def num(x,d=0.):
    try:
        v=float(x); return v if np.isfinite(v) else d
    except Exception:return d

def api(hosts,path,params):
    err='NETWORK'
    for h in hosts:
        try:
            r=requests.get(h+path,params=params,timeout=2.6,headers={'User-Agent':'ZIA-Research'})
            if r.ok:return r.json(),h,'OK'
            err=f'HTTP {r.status_code}'
        except requests.RequestException as e:err=type(e).__name__
    return None,None,err

@st.cache_data(ttl=2,show_spinner=False)
def candles(symbol,interval,limit=500):
    raw,host,status=api(FUT,'/fapi/v1/klines',{'symbol':symbol,'interval':interval,'limit':min(limit,1500)});source='Futures'
    if not isinstance(raw,list):raw,host,status=api(SPOT,'/api/v3/klines',{'symbol':symbol,'interval':interval,'limit':min(limit,1000)});source='Spot'
    if not isinstance(raw,list):raw,host,status=api(DATA,'/api/v3/klines',{'symbol':symbol,'interval':interval,'limit':min(limit,1000)});source='Data API'
    rows=[]
    for c in raw or []:
        try:rows.append([pd.to_datetime(int(c[0]),unit='ms',utc=True),num(c[1]),num(c[2]),num(c[3]),num(c[4]),num(c[5]),num(c[9])])
        except Exception:pass
    return pd.DataFrame(rows,columns=['Time','Open','High','Low','Close','Volume','TakerBuy']),source,status,host

@st.cache_data(ttl=2,show_spinner=False)
def orderbook(symbol):
    raw,host,status=api(FUT,'/fapi/v1/depth',{'symbol':symbol,'limit':100});source='Futures'
    if not isinstance(raw,dict) or not raw.get('bids'):raw,host,status=api(SPOT,'/api/v3/depth',{'symbol':symbol,'limit':100});source='Spot'
    if not isinstance(raw,dict) or not raw.get('bids'):raw,host,status=api(DATA,'/api/v3/depth',{'symbol':symbol,'limit':100});source='Data API'
    try:return np.asarray(raw.get('bids',[]),float),np.asarray(raw.get('asks',[]),float),source,status,host
    except Exception:return np.empty((0,2)),np.empty((0,2)),source,status,host

def obi(b,a,k):
    if not len(b) or not len(a):return 0.,0.,0.
    k=min(k,len(b),len(a));bv=float(b[:k,1].sum());av=float(a[:k,1].sum());t=bv+av
    return (bv-av)/t if t else 0.,bv,av

def features(df,b,a):
    f={k:0. for k in ['top20_bid_sum','top20_ask_sum','obi_5','obi_10','obi_20','obi_50','spread','spread_pct','bid_ask_ratio_20','bid_ask_ratio_50','top20_total_depth','top50_total_depth','taker_flow_ratio','price_return','price_change','sma_distance','realized_volatility','BOOK_IMB','QUANT_IMPLY','ADAPT_CONF','BAYESIAN','FOURIER_TREND']};f['bid_ask_ratio_20']=f['bid_ask_ratio_50']=1.
    o5,_,_=obi(b,a,5);o10,_,_=obi(b,a,10);o20,b20,a20=obi(b,a,20);o50,b50,a50=obi(b,a,50);f.update(obi_5=o5,obi_10=o10,obi_20=o20,obi_50=o50,top20_bid_sum=b20,top20_ask_sum=a20,top20_total_depth=b20+a20,top50_total_depth=b50+a50)
    if df.empty:return f
    c=df.Close;last=num(c.iloc[-1]);prev=num(c.iloc[-2] if len(c)>1 else last);sma=num(c.rolling(20).mean().iloc[-1],last);vol=num(df.Volume.tail(20).sum());buy=num(df.TakerBuy.tail(20).sum());sell=max(vol-buy,0.);spread=num(a[0,0]-b[0,0]) if len(a) and len(b) else 0.;trend=np.tanh((last/sma-1)*100) if sma else 0.;rv=num(c.pct_change().tail(30).std());four=np.tanh(num(c.pct_change().tail(16).mean())*1000)
    f.update(spread=spread,spread_pct=spread/last if last else 0,bid_ask_ratio_20=b20/a20 if a20 else 1,bid_ask_ratio_50=b50/a50 if a50 else 1,taker_flow_ratio=(buy-sell)/vol if vol else 0,price_return=last/prev-1 if prev else 0,price_change=last-prev,sma_distance=last/sma-1 if sma else 0,realized_volatility=rv,BOOK_IMB=o20,QUANT_IMPLY=float(np.tanh((o20+o50+trend)/3)),ADAPT_CONF=float(np.clip(.5+(abs(o20)+abs(trend))/2,0,1)),BAYESIAN=float(np.clip(.5+(o20+trend)/4,0,1)),FOURIER_TREND=float(four));return f

@st.cache_resource(show_spinner=False)
def model():
    try:return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
    except Exception:return None

def ml(f):
    m=model()
    if m is None:return None,None,'MODEL NOT FOUND',0
    try:
        names=list(m.get_booster().feature_names or []) if hasattr(m,'get_booster') else []
        n=int(getattr(m,'n_features_in_',len(names) or 7));known=['top20_bid_sum','top20_ask_sum','obi_top20','spread','bid_ask_ratio','total_depth','trend_signal'];row=dict(f,obi_top20=f['obi_20'],bid_ask_ratio=f['bid_ask_ratio_20'],total_depth=f['top20_total_depth'],trend_signal=f['sma_distance']);cols=names or (known if n==7 else list(f.keys()));x=pd.DataFrame([[num(row.get(c,0)) for c in cols]],columns=cols);p=int(m.predict(x)[0]);pr=float(m.predict_proba(x)[0][-1]) if hasattr(m,'predict_proba') else None;return p,pr,'OK',len(cols)
    except Exception as e:return None,None,'ML ERROR: '+type(e).__name__,0

def research(f):
    s={'OBI 20':np.clip(f['obi_20']*2,-1,1),'OBI 20 + 50':np.clip((f['obi_20']+f['obi_50'])/1.5,-1,1),'Taker Flow':np.clip(f['taker_flow_ratio']*2,-1,1),'Trend / SMA':np.clip(np.tanh(f['sma_distance']*100),-1,1),'Fourier':np.clip(f['FOURIER_TREND'],-1,1),'Bayesian':np.clip((f['BAYESIAN']-.5)*2,-1,1),'Quant Imply':np.clip(f['QUANT_IMPLY'],-1,1),'Adaptive':np.clip((f['ADAPT_CONF']-.5)*2,-1,1)};w={'OBI 20':.22,'OBI 20 + 50':.14,'Taker Flow':.20,'Trend / SMA':.14,'Fourier':.10,'Bayesian':.08,'Quant Imply':.07,'Adaptive':.05};return s,w,float(sum(s[k]*w[k] for k in s))

def signal(f,p,pr):
    rs,rw,score=research(f);ms=(pr-.5)*2 if pr is not None else (1 if p==1 else -1 if p==0 else 0);comp=.6*score+.4*ms if p is not None else score;sig='LONG' if comp>=.45 else 'SHORT' if comp<=-.45 else 'WAIT';conf=float(np.clip(50+abs(comp)*49,1,99));return sig,conf,comp,rs,rw,score,ms

@st.cache_data(ttl=20,show_spinner=False)
def tri(symbol,interval):
    df,_,_,_=candles(symbol,interval,4)
    if len(df)<2:return None
    c=df.iloc[-2];o,h,l,cl=map(num,[c.Open,c.High,c.Low,c.Close]);bh=max(o,cl);bl=min(o,cl)
    return {'BODY 50':(bh+bl)/2,'UPPER 50':(h+bh)/2,'LOWER 50':(l+bl)/2}

def tri_lines(symbol,selected):
    out=[]
    for label in selected:
        lv=tri(symbol,TFS[label])
        if lv:out.append((label,lv))
    return out

def make_chart(df,symbol,selected,body,upper,lower,future):
    fig=go.Figure()
    if df.empty:return fig
    fig.add_trace(go.Candlestick(x=df.Time,open=df.Open,high=df.High,low=df.Low,close=df.Close,name='PRICE',increasing_line_color='#42dda0',increasing_fillcolor='#176d4f',decreasing_line_color='#ff7184',decreasing_fillcolor='#8e3448'))
    for n,col in [(10,'#8E98FF'),(20,'#65D7FF'),(50,'#F3C86A'),(200,'#B58CFF')]:
        if len(df)>=n:fig.add_trace(go.Scatter(x=df.Time,y=df.Close.ewm(span=n,adjust=False).mean(),mode='lines',name=f'EMA {n}',line={'width':1.2,'color':col}))
    flags={'BODY 50':body,'UPPER 50':upper,'LOWER 50':lower}
    for label,levels in tri_lines(symbol,selected):
        for name,on in flags.items():
            if on:fig.add_hline(y=levels[name],line_color=TRI_COLORS[name],line_width=1.4,line_dash='solid' if name=='BODY 50' else 'dot',opacity=.9,annotation_text=f'TRI {label} • {name}',annotation_position='top right')
    step=df.Time.iloc[-1]-df.Time.iloc[-2] if len(df)>1 else pd.Timedelta(minutes=5);start=max(0,len(df)-350)
    fig.update_xaxes(range=[df.Time.iloc[start],df.Time.iloc[-1]+step*future],rangeslider_visible=False,showgrid=True,gridcolor='#172230',showspikes=True,spikemode='across',spikesnap='cursor')
    fig.update_yaxes(side='right',showgrid=True,gridcolor='#172230');fig.update_layout(height=620,margin=dict(l=4,r=4,t=8,b=8),paper_bgcolor='#080d14',plot_bgcolor='#080d14',font={'color':'#cbd5e1'},hovermode='x unified',dragmode='pan',legend={'orientation':'h','y':1.02,'x':0,'bgcolor':'rgba(0,0,0,0)'})
    return fig

def cards(items):
    cs=st.columns(len(items))
    for c,(lab,val,sub,col) in zip(cs,items):
        with c:st.markdown(f'<div class="card"><div class="label">{lab}</div><div class="value" style="color:{col}">{val}</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)

def read(p):
    try:return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception:return pd.DataFrame()

def save(symbol,tf,price,sig,conf,pr,f,rs):
    row={'timestamp':datetime.now(timezone.utc).isoformat(),'symbol':symbol,'timeframe':tf,'price':price,'signal':sig,'confidence':conf,'ml_probability':pr if pr is not None else '','obi20':f['obi_20'],'obi50':f['obi_50'],'ofi':f['taker_flow_ratio'],'research_score':rs};pd.DataFrame([row]).to_csv(SIGNAL_FILE,mode='a',header=not SIGNAL_FILE.exists(),index=False)

# UI state is independent of the 3-second live data cycle.
for k,v in {'symbol':'BTCUSDT','tf':'15M','tri_selected':['1D','1W','1MO'],'tri_body':True,'tri_upper':True,'tri_lower':True,'future':24}.items():
    if k not in st.session_state:st.session_state[k]=v

st.markdown('<div class="hero"><div><div class="brand">ZIA <b>RESEARCH</b></div><div class="micro">QUANT MARKET INTELLIGENCE · LIVE ML · ORDER FLOW · RESEARCH LAB</div></div><div class="live"><span class="dot"></span>LIVE ENGINE · SILENT 3s</div></div>',unsafe_allow_html=True)

c=st.columns([1.4,1,1.2,1])
with c[0]:symbol=st.selectbox('MARKET',SYMBOLS,index=SYMBOLS.index(st.session_state.symbol),key='symbol')
with c[1]:tf=st.selectbox('TIMEFRAME',list(TFS),index=list(TFS).index(st.session_state.tf),key='tf')
with c[2]:future=st.selectbox('FUTURE SPACE',[12,18,24,32,48],index=[12,18,24,32,48].index(st.session_state.future),format_func=lambda x:f'{x} BARS',key='future')
with c[3]:st.caption('AUTO REFRESH');st.markdown('**3 seconds · silent cycle**')

with st.expander('△ TRI LINE CONTROL',expanded=True):
    a,b,c=st.columns([2.4,1,1])
    with a:selected=st.multiselect('TIMEFRAMES ON CHART',list(TFS),key='tri_selected')
    with b:body=st.toggle('BODY 50',key='tri_body');upper=st.toggle('UPPER 50',key='tri_upper')
    with c:lower=st.toggle('LOWER 50',key='tri_lower')
    badges=''.join(f'<span class="tri">{x}</span>' for x in selected);st.markdown(f'<div class="small">ACTIVE TRI: {badges or "NONE"} · Colors are applied directly to chart lines.</div>',unsafe_allow_html=True)

@st.fragment(run_every='3s')
def live():
    # Only this fragment reruns. Tabs, controls and page shell are not timer-rerun.
    df,source,cstat,_=candles(symbol,TFS[tf],500);b,a,bsrc,bstat,_=orderbook(symbol);f=features(df,b,a);p,pr,mstat,fc=ml(f);sig,conf,comp,rs,rw,rscore,mscore=signal(f,p,pr);price=num(df.Close.iloc[-1]) if not df.empty else 0.;prev=num(df.Close.iloc[-2]) if len(df)>1 else price;chg=(price/prev-1)*100 if prev else 0.;sclass='long' if sig=='LONG' else 'short' if sig=='SHORT' else 'wait'
    st.markdown('<div class="title">⚡ COMMAND CENTER</div><div class="subtitle">Live composite signal · ML prediction · order-flow context</div>',unsafe_allow_html=True)
    left,right=st.columns([1.1,3.2])
    with left:st.markdown(f'<div class="signal"><div class="signal-label">MAIN SIGNAL · {symbol} · {tf}</div><div class="signal-main {sclass}">{sig}</div><b>Strength {conf:.1f}%</b><div class="sub">Composite {comp:+.3f} · Research {rscore:+.3f} · ML {mscore:+.3f}</div></div>',unsafe_allow_html=True)
    with right:cards([('PRICE',f'${price:,.2f}',f'{chg:+.2f}% · {tf}','#42dda0' if chg>=0 else '#ff7184'),('ML',f'{pr*100:.1f}%' if pr is not None else '—',mstat,'#8e98ff'),('OBI 20',f'{f["obi_20"]:+.3f}','top 20 levels','#42dda0' if f['obi_20']>=0 else '#ff7184'),('OBI 50',f'{f["obi_50"]:+.3f}','top 50 levels','#42dda0' if f['obi_50']>=0 else '#ff7184'),('FLOW',f'{f["taker_flow_ratio"]:+.3f}','taker ratio','#42dda0' if f['taker_flow_ratio']>=0 else '#ff7184')])
    tabs=st.tabs(['⌂ OVERVIEW','◈ CHART','◌ ORDER FLOW','🧠 ML LAB','🔬 RESEARCH LAB','▣ SIGNALS'])
    with tabs[0]:
        x,y=st.columns([2,1]);reg='BULLISH FLOW' if comp>.25 else 'BEARISH FLOW' if comp<-.25 else 'BALANCED / WAIT'
        with x:st.markdown(f'<div class="panel"><b>MARKET REGIME</b><h3>{reg}</h3>',unsafe_allow_html=True);st.progress(min(max(conf/100,0),1),text=f'Signal strength {conf:.1f}%');st.write(f'Research **{rscore:+.3f}** · ML **{mscore:+.3f}** · Composite **{comp:+.3f}**');st.markdown('</div>',unsafe_allow_html=True)
        with y:st.markdown(f'<div class="panel"><b>CONNECTION</b><br>Candles: `{source}`<br>Order book: `{bsrc}`<br>Status: `{bstat}`<br>Update: `{datetime.now(timezone.utc).strftime("%H:%M:%S UTC")}`</div>',unsafe_allow_html=True)
    with tabs[1]:
        st.markdown('<div class="panel"><b>PRICE ACTION + TRI STRUCTURE</b><div class="small">TRI uses the last completed candle of each selected timeframe.</div>',unsafe_allow_html=True);st.plotly_chart(make_chart(df,symbol,selected,body,upper,lower,future),use_container_width=True,config={'scrollZoom':True,'displaylogo':False,'responsive':True,'doubleClick':'reset','modeBarButtonsToAdd':['drawline','drawrect','eraseshape']});st.markdown('</div>',unsafe_allow_html=True)
    with tabs[2]:
        if len(b) and len(a):
            vals=[obi(b,a,k) for k in (5,10,20,50)];cards([(f'OBI {k}',f'{v[0]:+.3f}',f'B {v[1]:,.1f} / A {v[2]:,.1f}','#42dda0' if v[0]>=0 else '#ff7184') for k,v in zip((5,10,20,50),vals)]);l,r=st.columns(2);l.dataframe(pd.DataFrame(b[:20],columns=['Bid Price','Bid Qty']),use_container_width=True,hide_index=True);r.dataframe(pd.DataFrame(a[:20],columns=['Ask Price','Ask Qty']),use_container_width=True,hide_index=True)
        else:st.warning(f'Order book unavailable · {bstat}')
    with tabs[3]:
        cards([('MODEL',mstat,'xgboost_obi_model.pkl','#8e98ff'),('PREDICTION','LONG' if p==1 else 'SHORT' if p==0 else '—',f'class {p}' if p is not None else 'no prediction','#42dda0' if p==1 else '#ff7184' if p==0 else '#f3c86a'),('PROBABILITY',f'{pr*100:.2f}%' if pr is not None else '—','model probability','#8e98ff'),('FEATURES',str(fc),'supplied to model','#65d7ff')]);st.markdown('<div class="panel"><b>LIVE MODEL INPUTS</b>',unsafe_allow_html=True);st.dataframe(pd.DataFrame({'Feature':['OBI 5','OBI 10','OBI 20','OBI 50','Spread','Taker Flow','Trend / SMA','Volatility'],'Value':[f['obi_5'],f['obi_10'],f['obi_20'],f['obi_50'],f['spread'],f['taker_flow_ratio'],f['sma_distance'],f['realized_volatility']]}),use_container_width=True,hide_index=True);st.markdown('</div>',unsafe_allow_html=True)
    with tabs[4]:
        rd=pd.DataFrame([{'Formula':k,'Live Score':round(float(v),4),'Weight %':round(rw[k]*100,1),'Contribution':round(float(v*rw[k]),4),'Direction':'BULL' if v>0 else 'BEAR' if v<0 else 'NEUTRAL'} for k,v in rs.items()]).sort_values('Contribution',ascending=False);st.markdown('<div class="panel"><b>RESEARCH FORMULA SCOREBOARD</b>',unsafe_allow_html=True);st.dataframe(rd,use_container_width=True,hide_index=True);st.write(f'Strongest contributor: **{rd.iloc[0]["Formula"] if not rd.empty else "—"}** · Composite **{rscore:+.3f}**');st.markdown('</div>',unsafe_allow_html=True)
    with tabs[5]:
        if st.button('💾 SAVE CURRENT SIGNAL',use_container_width=True):save(symbol,tf,price,sig,conf,pr,f,rscore);st.success('Signal saved.')
        h=read(SIGNAL_FILE);t=read(TRADE_FILE)
        if not h.empty:st.dataframe(h.tail(60).iloc[::-1],use_container_width=True,hide_index=True);st.download_button('⬇ DOWNLOAD SIGNAL JOURNAL',h.to_csv(index=False),'zia_saved_signals.csv','text/csv',use_container_width=True)
        else:st.info('No saved signals yet.')
        if not t.empty and 'result' in t.columns:
            rr=t.result.astype(str).str.upper();wins=int((rr=='WIN').sum());loss=int((rr=='LOSS').sum());tot=wins+loss;wr=wins/tot*100 if tot else 0.;cards([('CLOSED',str(tot),'resolved trades','#65d7ff'),('WINS',str(wins),'winning trades','#42dda0'),('LOSSES',str(loss),'losing trades','#ff7184'),('WIN RATE',f'{wr:.1f}%','closed trade rate','#8e98ff')])
    st.caption(f'ZIA Research · {symbol} · {tf} · silent 3s live cycle · {datetime.now(timezone.utc).strftime("%H:%M:%S UTC")}')

live()
