from __future__ import annotations
import csv,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,plotly.graph_objects as go,requests,streamlit as st

st.set_page_config(page_title='ZIA Research Terminal',page_icon='⚡',layout='wide',initial_sidebar_state='collapsed')
ROOT=Path(__file__).resolve().parent; MODEL=ROOT/'xgboost_obi_model.pkl'; SIGNALS=ROOT/'saved_signals.csv'; SCORES=ROOT/'research_scores.csv'
FUT=['https://fapi.binance.com','https://fapi1.binance.com','https://fapi2.binance.com','https://fapi3.binance.com','https://fapi4.binance.com']; SPOT=['https://api.binance.com','https://api1.binance.com','https://api2.binance.com','https://api3.binance.com']; DATA=['https://data-api.binance.vision']
SYMBOLS=['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','SUIUSDT','TRXUSDT','LTCUSDT']; TFS={'5M':'5m','15M':'15m','30M':'30m','1H':'1h','4H':'4h','1D':'1d','1W':'1w'}
F25=['top20_bid_sum','top20_ask_sum','obi_5','obi_10','obi_20','obi_50','spread','spread_pct','bid_ask_ratio_20','bid_ask_ratio_50','top20_total_depth','top50_total_depth','taker_buy_volume','taker_sell_volume','taker_flow','taker_flow_ratio','price_return','price_change','sma_distance','realized_volatility','BOOK_IMB','QUANT_IMPLY','ADAPT_CONF','BAYESIAN','FOURIER_TREND']; F7=['top20_bid_sum','top20_ask_sum','obi_top20','spread','bid_ask_ratio','total_depth','trend_signal']

st.markdown('''<style>
html,body,[data-testid="stAppViewContainer"]{background:#070b11;color:#e8edf5}.block-container{max-width:1900px;padding:12px clamp(7px,1.6vw,30px) 45px}.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #202b3b;padding:3px 2px 13px;margin-bottom:12px}.brand{font-size:clamp(20px,2.6vw,34px);font-weight:950}.brand span{color:#8d98ff}.muted{font-size:9px;color:#718097;letter-spacing:1px}.live{color:#70e0a5;border:1px solid #245b42;background:#091710;border-radius:99px;padding:6px 10px;font-size:10px;font-weight:900}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#4ce096;margin-right:6px;box-shadow:0 0 9px #4ce096}.card{background:linear-gradient(145deg,#111a25,#0b1119);border:1px solid #202d3e;border-radius:14px;padding:11px;min-height:74px}.label{font-size:9px;color:#78879b;font-weight:900;letter-spacing:1px}.value{font-size:19px;font-weight:950;margin-top:5px}.sub{font-size:10px;color:#8190a5;margin-top:3px}.panel{background:#0b1119;border:1px solid #1e2a3a;border-radius:14px;padding:13px}.section{font-size:15px;font-weight:950;margin:12px 0 8px}.sig{font-size:30px;font-weight:950}.hint{font-size:10px;color:#77869b}@media(max-width:700px){.block-container{padding:6px 7px 35px}.hero{align-items:flex-start}.brand{font-size:20px}.muted{font-size:7px}.live{font-size:8px;padding:5px 7px}.card{min-height:62px;padding:8px}.value{font-size:15px}.sub{font-size:9px}.sig{font-size:26px}}
</style>''',unsafe_allow_html=True)

def n(x,d=0.):
    try:x=float(x);return x if np.isfinite(x) else d
    except:return d

def pricefmt(x):
    x=n(x);return f'{x:,.2f}' if x>=1000 else f'{x:,.5f}' if x else '—'

def req(hosts,path,params):
    err='network'
    for h in hosts:
        try:
            r=requests.get(h+path,params=params,timeout=4,headers={'User-Agent':'ZIA-Research'}); 
            if r.ok:return r.json(),h,'OK'
            err=f'HTTP {r.status_code}'
        except requests.RequestException as e:err=type(e).__name__
    return None,None,err

@st.cache_data(ttl=2,show_spinner=False)
def candles(symbol,interval,limit):
    raw,h,s=req(FUT,'/fapi/v1/klines',{'symbol':symbol,'interval':interval,'limit':min(limit,1500)});src='Futures'
    if not isinstance(raw,list):raw,h,s=req(SPOT,'/api/v3/klines',{'symbol':symbol,'interval':interval,'limit':min(limit,1000)});src='Spot'
    if not isinstance(raw,list):raw,h,s=req(DATA,'/api/v3/klines',{'symbol':symbol,'interval':interval,'limit':min(limit,1000)});src='Data API'
    rows=[]
    for x in raw or []:
        try:rows.append([pd.to_datetime(int(x[0]),unit='ms',utc=True),n(x[1]),n(x[2]),n(x[3]),n(x[4]),n(x[5]),n(x[9])])
        except:pass
    return pd.DataFrame(rows,columns=['Time','Open','High','Low','Close','Volume','TakerBuy']),src,s,h

@st.cache_data(ttl=2,show_spinner=False)
def orderbook(symbol):
    raw,h,s=req(FUT,'/fapi/v1/depth',{'symbol':symbol,'limit':100});src='Futures'
    if not isinstance(raw,dict) or not raw.get('bids') or not raw.get('asks'):raw,h,s=req(SPOT,'/api/v3/depth',{'symbol':symbol,'limit':100});src='Spot'
    if not isinstance(raw,dict) or not raw.get('bids') or not raw.get('asks'):raw,h,s=req(DATA,'/api/v3/depth',{'symbol':symbol,'limit':100});src='Data API'
    try:return np.asarray(raw.get('bids',[]),float),np.asarray(raw.get('asks',[]),float),src,s,h
    except:return np.empty((0,2)),np.empty((0,2)),src,s,h

def obi(b,a,k):
    if len(b)==0 or len(a)==0:return 0.,0.,0.
    k=min(k,len(b),len(a));bv=float(b[:k,1].sum());av=float(a[:k,1].sum());return (bv-av)/(bv+av) if bv+av else 0.,bv,av

def features(df,b,a):
    o5,b5,a5=obi(b,a,5);o10,b10,a10=obi(b,a,10);o20,b20,a20=obi(b,a,20);o50,b50,a50=obi(b,a,50);f={x:0. for x in F25};f.update(top20_bid_sum=b20,top20_ask_sum=a20,obi_5=o5,obi_10=o10,obi_20=o20,obi_50=o50,top20_total_depth=b20+a20,top50_total_depth=b50+a50)
    if df.empty:return f
    c=df.Close;last=n(c.iloc[-1]);prev=n(c.iloc[-2] if len(c)>1 else last);sma=n(c.rolling(20).mean().iloc[-1],last);spread=n(a[0,0]-b[0,0]) if len(a) and len(b) else 0;vol=n(df.Volume.tail(20).sum());buy=n(df.TakerBuy.tail(20).sum());sell=max(vol-buy,0);flow=buy-sell;trend=np.tanh((last/sma-1)*100) if sma else 0;rv=n(c.pct_change().tail(30).std());four=np.tanh(c.pct_change().tail(16).mean()*1000)
    f.update(spread=spread,spread_pct=spread/last if last else 0,bid_ask_ratio_20=b20/a20 if a20 else 1,bid_ask_ratio_50=b50/a50 if a50 else 1,taker_buy_volume=buy,taker_sell_volume=sell,taker_flow=flow,taker_flow_ratio=flow/vol if vol else 0,price_return=last/prev-1 if prev else 0,price_change=last-prev,sma_distance=last/sma-1 if sma else 0,realized_volatility=rv,BOOK_IMB=o20,QUANT_IMPLY=np.tanh((o20+o50+trend)/3),ADAPT_CONF=np.clip(.5+(abs(o20)+abs(trend))/2,0,1),BAYESIAN=np.clip(.5+(o20+trend)/4,0,1),FOURIER_TREND=four);return f

@st.cache_resource(show_spinner=False)
def load_model():
    try:return joblib.load(MODEL)
    except:return None

def ml_predict(f):
    m=load_model()
    if m is None:return None,None,'MODEL NOT FOUND',0
    try:
        aliases={'obi_top20':f['obi_20'],'bid_ask_ratio':f['bid_ask_ratio_20'],'total_depth':f['top20_total_depth'],'trend_signal':f['sma_distance']};names=list(m.get_booster().feature_names or []) if hasattr(m,'get_booster') else [];count=int(getattr(m,'n_features_in_',len(names) or 25));cols=names if names else (F7 if count==7 else F25);row=dict(f,**aliases);x=pd.DataFrame([[row.get(k,0.) for k in cols]],columns=cols);p=int(m.predict(x)[0]);prob=float(m.predict_proba(x)[0][-1]) if hasattr(m,'predict_proba') else None;return p,prob,'OK',len(cols)
    except Exception as e:return None,None,'ML ERROR: '+type(e).__name__,0

def research_scores(f):
    return {'OBI 20':float(np.clip(f['obi_20']*2,-1,1)),'OBI 20+50':float(np.clip(f['obi_20']+f['obi_50'],-1,1)),'OFI':float(np.clip(f['taker_flow_ratio']*2,-1,1)),'Trend / SMA':float(np.clip(np.tanh(f['sma_distance']*100),-1,1)),'Fourier':float(f['FOURIER_TREND']),'Bayesian':float((f['BAYESIAN']-.5)*2),'Quant Imply':float(f['QUANT_IMPLY']),'Adaptive':float((f['ADAPT_CONF']-.5)*2)}

def signal_engine(f,p,prob):
    s=research_scores(f);w={'OBI 20':.22,'OBI 20+50':.14,'OFI':.20,'Trend / SMA':.14,'Fourier':.10,'Bayesian':.08,'Quant Imply':.07,'Adaptive':.05};r=sum(s[k]*w[k] for k in w);m=((prob-.5)*2) if prob is not None else (1 if p==1 else -1 if p==0 else 0);score=.6*r+.4*m if p is not None else r;sig='LONG' if score>=.45 else 'SHORT' if score<=-.45 else 'WAIT';return sig,score,float(np.clip(50+abs(score)*49,1,99)),s,r,m,w

def tri_map(tf):return {'4H':'4h','1H':'1h'} if tf=='15M' else ({'DAY':'1d','WEEK':'1w','MONTH':'1M'} if tf in ('1H','4H') else {})
@st.cache_data(ttl=20,show_spinner=False)
def tri(symbol,tf):
    out={}
    for name,it in tri_map(tf).items():
        d,_,_,_=candles(symbol,it,5)
        if len(d)<2:continue
        c=d.iloc[-2];o,h,l,cl=map(n,[c.Open,c.High,c.Low,c.Close]);bh=max(o,cl);bl=min(o,cl);out[name]={'BODY 50':(bh+bl)/2,'UPPER 50':(h+bh)/2,'LOWER 50':(l+bl)/2}
    return out

def make_chart(df,symbol,tf,future,show_tri):
    fig=go.Figure()
    if df.empty:return fig
    fig.add_trace(go.Candlestick(x=df.Time,open=df.Open,high=df.High,low=df.Low,close=df.Close,name='PRICE'))
    for span in (20,50,200):
        if len(df)>=span:fig.add_trace(go.Scatter(x=df.Time,y=df.Close.ewm(span=span,adjust=False).mean(),name=f'EMA {span}',mode='lines',line={'width':1}))
    if show_tri:
        for period,lv in tri(symbol,tf).items():
            for key,dash in [('BODY 50','solid'),('UPPER 50','dot'),('LOWER 50','dot')]:fig.add_hline(y=lv[key],line_dash=dash,line_width=1,annotation_text=f'TRI {period} {key}',annotation_position='top right')
    step=df.Time.iloc[-1]-df.Time.iloc[-2] if len(df)>1 else pd.Timedelta(minutes=5);fig.update_xaxes(range=[df.Time.iloc[max(0,len(df)-260)],df.Time.iloc[-1]+step*future],showspikes=True,spikemode='across',showgrid=True);fig.update_yaxes(side='right',showgrid=True);fig.update_layout(template='plotly_dark',height=650,margin=dict(l=4,r=4,t=25,b=4),xaxis_rangeslider_visible=False,dragmode='pan',hovermode='x unified',paper_bgcolor='#0b1119',plot_bgcolor='#0b1119',legend=dict(orientation='h',y=1.02,x=0));return fig

def write_signal(row):
    exists=SIGNALS.exists()
    with SIGNALS.open('a',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=list(row));
        if not exists:w.writeheader()
        w.writerow(row)

def saved():
    if not SIGNALS.exists():return pd.DataFrame()
    try:return pd.read_csv(SIGNALS).tail(200).iloc[::-1]
    except:return pd.DataFrame()

def write_score(row):
    exists=SCORES.exists()
    with SCORES.open('a',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=list(row));
        if not exists:w.writeheader()
        w.writerow(row)

def perf():
    if not SCORES.exists():return pd.DataFrame()
    try:
        d=pd.read_csv(SCORES);d['close']=pd.to_numeric(d['close'],errors='coerce');d['future']=d.groupby(['symbol','timeframe'])['close'].shift(-5);ret=d['future']/d['close']-1;out=[]
        for col in ['OBI 20','OBI 20+50','OFI','Trend / SMA','Fourier','Bayesian','Quant Imply','Adaptive']:
            x=pd.to_numeric(d[col],errors='coerce');mask=x.abs()>=.15;mask&=ret.notna();n=int(mask.sum());acc=float((np.sign(x[mask])==np.sign(ret[mask])).mean()*100) if n else np.nan;out.append([col,n,acc])
        return pd.DataFrame(out,columns=['Formula','Resolved samples','Directional accuracy %']).sort_values('Directional accuracy %',ascending=False,na_position='last')
    except:return pd.DataFrame()

with st.sidebar:
    st.header('ZIA RESEARCH')
    symbol=st.selectbox('Symbol',SYMBOLS,index=0);tf=st.selectbox('Timeframe',list(TFS),index=1);bars=st.slider('Candles',100,1000,500,50);future=st.slider('Future space',5,100,25);refresh=st.slider('Live refresh (seconds)',2,15,4);show_tri=st.checkbox('TRI Lines',True);st.caption('15M → 4H + 1H TRI | 1H/4H → Day + Week + Month TRI')

st.markdown('<div class="hero"><div><div class="brand">ZIA <span>RESEARCH</span></div><div class="muted">QUANT TERMINAL • LIVE ML • ORDER FLOW • RESEARCH LAB</div></div><div class="live"><span class="dot"></span>LIVE</div></div>',unsafe_allow_html=True)
# Navigation is outside the live fragment. Only the selected section is refreshed, so the browser/page does not flash.
section=st.segmented_control('SECTION',['Overview','Chart','Order Flow','ML Lab','Research Lab','Signals'],default='Overview',key='nav') if hasattr(st,'segmented_control') else st.radio('SECTION',['Overview','Chart','Order Flow','ML Lab','Research Lab','Signals'],horizontal=True,key='nav_old')

def render():
    df,msrc,mstat,mhost=candles(symbol,TFS[tf],bars);b,a,bsrc,bstat,bhost=orderbook(symbol);f=features(df,b,a);p,prob,mlstat,nfeat=ml_predict(f);sig,score,conf,rs,research,mlscore,weights=signal_engine(f,p,prob);price=n(df.Close.iloc[-1]) if not df.empty else 0;o20,b20,a20=obi(b,a,20);o50,b50,a50=obi(b,a,50);prev=st.session_state.get('book');ofi=0.;st.session_state.book=(b.copy(),a.copy());
    if prev is not None:ofi=(b[:20,1].sum()-prev[0][:20,1].sum())-(a[:20,1].sum()-prev[1][:20,1].sum())
    if section=='Overview':
        c=st.columns(6);vals=[('PRICE',pricefmt(price),msrc),('SIGNAL',sig,f'Confidence {conf:.1f}%'),('LIVE ML',f'{prob*100:.1f}%' if prob is not None else '—',mlstat),('OBI 20',f'{o20:+.3f}',f'Bid {b20:,.0f} / Ask {a20:,.0f}'),('OBI 50',f'{o50:+.3f}','Top 50 levels'),('OFI',f'{ofi:+,.1f}','20-level delta')]
        for cc,(l,v,s) in zip(c,vals):cc.markdown(f'<div class="card"><div class="label">{l}</div><div class="value">{v}</div><div class="sub">{s}</div></div>',unsafe_allow_html=True)
        st.markdown('### Signal Monitor');a1,a2=st.columns([1.2,1]);a1.markdown(f'<div class="panel"><div class="sig">{sig}</div><div class="hint">Composite {score:+.3f} • Research {research:+.3f} • ML {mlscore:+.3f}</div></div>',unsafe_allow_html=True)
        if a2.button('💾 SAVE CURRENT SIGNAL',use_container_width=True,type='primary'):
            write_signal({'timestamp':pd.Timestamp.now(tz='UTC').isoformat(),'symbol':symbol,'timeframe':tf,'signal':sig,'price':price,'confidence':conf,'score':score,'ml_probability':prob if prob is not None else '','obi20':o20,'obi50':o50,'ofi':ofi,'research_score':research,**rs});st.success('Signal saved')
        st.markdown('### Connection');st.info(f'Market: {msrc} / {mstat} • Order Book: {bsrc} / {bstat} • ML: {mlstat}')
    elif section=='Chart':
        st.markdown(f'### {symbol} • {tf} PRICE ACTION');st.plotly_chart(make_chart(df,symbol,tf,future,show_tri),use_container_width=True,config={'scrollZoom':True,'displaylogo':False,'modeBarButtonsToAdd':['drawline','drawrect','eraseshape']},key='chart');st.caption('TRI: 15M uses 4H + 1H. 1H/4H use Day + Week + Month. Future space remains after the latest candle.')
    elif section=='Order Flow':
        st.markdown('### Order Flow');c=st.columns(3)
        rows=[]
        for k in (5,10,20,50):o,bb,aa=obi(b,a,k);rows.append([f'Top {k}',o,bb,aa])
        c[0].dataframe(pd.DataFrame(rows,columns=['Depth','OBI','Bid volume','Ask volume']).style.format({'OBI':'{:+.3f}','Bid volume':'{:,.0f}','Ask volume':'{:,.0f}'}),use_container_width=True,hide_index=True)
        c[1].metric('OFI',f'{ofi:+,.2f}');c[1].metric('Taker Flow Ratio',f'{f["taker_flow_ratio"]:+.3f}');c[1].metric('Spread',pricefmt(f['spread']))
        c[2].metric('Top 20 Depth',f'{b20+a20:,.0f}');c[2].metric('Top 50 Depth',f'{o50*0+b50+a50:,.0f}');c[2].caption('OFI = displayed 20-level depth change between live samples.')
    elif section=='ML Lab':
        st.markdown('### XGBoost ML Lab');c=st.columns(4);items=[('MODEL',MODEL.name if MODEL.exists() else 'MISSING',mlstat),('FEATURES',nfeat,'input columns'),('PREDICTION',p if p is not None else '—','model output'),('PROBABILITY',f'{prob*100:.2f}%' if prob is not None else '—','predict_proba')]
        for cc,(l,v,s) in zip(c,items):cc.markdown(f'<div class="card"><div class="label">{l}</div><div class="value">{v}</div><div class="sub">{s}</div></div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({'Feature':F25,'Live value':[f.get(x,0) for x in F25]}),use_container_width=True,hide_index=True)
    elif section=='Research Lab':
        st.markdown('### Research Lab • Formula Breakdown');rows=[[k,rs[k],weights[k]*100,rs[k]*weights[k]] for k in rs];rdf=pd.DataFrame(rows,columns=['Formula','Live Score','Weight %','Contribution']).sort_values('Contribution',ascending=False);st.dataframe(rdf.style.format({'Live Score':'{:+.3f}','Weight %':'{:.1f}','Contribution':'{:+.3f}'}),use_container_width=True,hide_index=True);st.metric('Research Composite',f'{research:+.3f}');best=max(rs,key=lambda k:abs(rs[k]));st.info(f'Current strongest reading: {best} ({rs[best]:+.3f}). Historical performance is shown below when enough resolved samples exist.');lb=perf();
        if not lb.empty:st.dataframe(lb.style.format({'Directional accuracy %':'{:.1f}%'}),use_container_width=True,hide_index=True)
    elif section=='Signals':
        st.markdown('### Saved Signals');d=saved();
        if d.empty:st.info('No saved signals yet. Use SAVE CURRENT SIGNAL in Overview.')
        else:
            st.dataframe(d,use_container_width=True,hide_index=True);st.download_button('⬇️ DOWNLOAD CSV',d.to_csv(index=False),'saved_signals.csv','text/csv',use_container_width=True)
    write_score({'timestamp':time.time(),'symbol':symbol,'timeframe':tf,'close':price,**rs});st.caption(f'Live update {pd.Timestamp.now(tz="UTC").strftime("%H:%M:%S UTC")} • {msrc} • {bsrc}')

if hasattr(st,'fragment'):
    @st.fragment(run_every=f'{refresh}s')
    def live():render()
    live()
else:
    render()
    st.warning('Your Streamlit version does not support fragment refresh; upgrade Streamlit to avoid full-page refresh.')
