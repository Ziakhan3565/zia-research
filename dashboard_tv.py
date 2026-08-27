from __future__ import annotations
from pathlib import Path
import joblib, numpy as np, pandas as pd, requests, streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="ZIA Research Lab",page_icon="⚡",layout="wide")
ROOT=Path(__file__).resolve().parent; MODEL=ROOT/"xgboost_obi_model.pkl"
FUT="https://fapi.binance.com"; SPOT="https://api.binance.com"
TFS={"5M":"5m","15M":"15m","1H":"1h","4H":"4h","1D":"1d","1W":"1w"}
COINS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT","TRXUSDT","LTCUSDT","BCHUSDT","DOTUSDT","XLMUSDT","NEARUSDT","UNIUSDT","APTUSDT","TAOUSDT","XMRUSDT"]
FEATURES=["top20_bid_sum","top20_ask_sum","obi_5","obi_10","obi_20","obi_50","spread","spread_pct","bid_ask_ratio_20","bid_ask_ratio_50","top20_total_depth","top50_total_depth","taker_buy_volume","taker_sell_volume","taker_flow","taker_flow_ratio","price_return","price_change","sma_distance","realized_volatility","BOOK_IMB","QUANT_IMPLY","ADAPT_CONF","BAYESIAN","FOURIER_TREND"]
OLD=["top20_bid_sum","top20_ask_sum","obi_top20","spread","bid_ask_ratio","total_depth","trend_signal"]

st.markdown('''<style>
.block-container{max-width:1800px;padding:1rem 1.2rem 2rem}.stApp{background:#070b11;color:#e8eef7}[data-testid="stSidebar"]{background:#090e16;border-right:1px solid #202c3c}[data-testid="stSidebar"] *{color:#dce5f2}.hero{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}.brand{font-size:30px;font-weight:950}.brand span{color:#7d8cff}.sub{color:#7c899d;font-size:11px}.live{border:1px solid #24523d;background:#0b1712;border-radius:999px;padding:7px 12px;color:#7fe0aa;font-size:11px;font-weight:800}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#48d58b;box-shadow:0 0 10px #48d58b;margin-right:7px}.card{background:linear-gradient(145deg,#101824,#0b1119);border:1px solid #202d3e;border-radius:14px;padding:13px}.lab{font-size:9px;color:#7d899a;font-weight:800;letter-spacing:1px}.val{font-size:20px;font-weight:900;margin-top:5px}.muted{font-size:10px;color:#8190a3}.section{font-size:15px;font-weight:900;margin:15px 0 8px}.signal{border:1px solid #344256;border-radius:16px;padding:18px;background:linear-gradient(135deg,#111b2a,#0a1018)}.signal.long{border-color:#23875f}.signal.short{border-color:#a43d51}.sig{font-size:40px;font-weight:950}.chart{border:1px solid #1f2c3d;border-radius:15px;overflow:hidden;background:#080d14}
</style>''',unsafe_allow_html=True)

def n(x,d=0.):
    try:x=float(x);return x if np.isfinite(x) else d
    except:return d

def fmt(x):
    x=n(x)
    return "—" if x<=0 else f"{x:,.2f}" if x>=1000 else f"{x:,.5f}"

@st.cache_data(ttl=2,show_spinner=False)
def api(base,path,params):
    try:
        r=requests.get(base+path,params=params,timeout=8,headers={"User-Agent":"ZIA-Research-Lab"});return r.json() if r.ok else None
    except:return None

@st.cache_data(ttl=4,show_spinner=False)
def candles(symbol,tf,limit):
    raw=api(FUT,"/fapi/v1/klines",{"symbol":symbol,"interval":tf,"limit":min(limit,1500)})
    if not isinstance(raw,list):raw=api(SPOT,"/api/v3/klines",{"symbol":symbol,"interval":tf,"limit":min(limit,1000)})
    rows=[]
    for c in raw or []:
        try:rows.append([pd.to_datetime(int(c[0]),unit="ms",utc=True),*map(float,c[1:6]),float(c[9])])
        except:pass
    return pd.DataFrame(rows,columns=["Time","Open","High","Low","Close","Volume","TakerBuy"])

@st.cache_data(ttl=2,show_spinner=False)
def book(symbol):
    raw=api(FUT,"/fapi/v1/depth",{"symbol":symbol,"limit":100})
    if not isinstance(raw,dict):raw=api(SPOT,"/api/v3/depth",{"symbol":symbol,"limit":100})
    try:return np.asarray(raw.get("bids",[]),float),np.asarray(raw.get("asks",[]),float)
    except:return np.empty((0,2)),np.empty((0,2))

def obi(b,a,k):
    if len(b)==0 or len(a)==0:return 0.,0.,0.
    k=min(k,len(b),len(a));bs=b[:k,1].sum();ase=a[:k,1].sum();return ((bs-ase)/(bs+ase) if bs+ase else 0.),bs,ase

def make_features(df,b,a):
    o5,b5,a5=obi(b,a,5);o10,b10,a10=obi(b,a,10);o20,b20,a20=obi(b,a,20);o50,b50,a50=obi(b,a,50);z={x:0. for x in FEATURES}
    if df.empty:return z
    c=df.Close;last=n(c.iloc[-1]);prev=n(c.iloc[-2] if len(c)>1 else last);sma=n(c.rolling(20).mean().iloc[-1],last);vol=n(c.pct_change().rolling(20).std());buy=n(df.TakerBuy.tail(20).mean());tv=n(df.Volume.tail(20).mean());sell=max(tv-buy,0);flow=buy-sell;trend=np.tanh((last/sma-1)*100) if sma else 0.;fourier=np.tanh(c.pct_change().tail(16).mean()*1000) if len(c)>5 else trend;spread=(a[0,0]-b[0,0]) if len(a) and len(b) else 0
    z.update(top20_bid_sum=b20,top20_ask_sum=a20,obi_5=o5,obi_10=o10,obi_20=o20,obi_50=o50,spread=spread,spread_pct=spread/last if last else 0,bid_ask_ratio_20=b20/a20 if a20 else 1,bid_ask_ratio_50=b50/a50 if a50 else 1,top20_total_depth=b20+a20,top50_total_depth=b50+a50,taker_buy_volume=buy,taker_sell_volume=sell,taker_flow=flow,taker_flow_ratio=flow/(buy+sell) if buy+sell else 0,price_return=last/prev-1 if prev else 0,price_change=last-prev,sma_distance=last/sma-1 if sma else 0,realized_volatility=vol,BOOK_IMB=o20,QUANT_IMPLY=np.tanh((o20+o50+trend)/3),ADAPT_CONF=float(np.clip(.5+(abs(o20)+abs(trend))/2,0,1)),BAYESIAN=float(np.clip(.5+(o20+trend)/4,0,1)),FOURIER_TREND=fourier)
    return z

@st.cache_resource(show_spinner=False)
def load_model():
    try:return joblib.load(MODEL) if MODEL.exists() else None
    except:return None

def prediction(f):
    m=load_model()
    if m is None:return None,None,0
    try:
        names=None
        try:names=list(m.get_booster().feature_names)
        except:pass
        count=int(getattr(m,"n_features_in_",len(FEATURES)));cols=names if names and all(k in f for k in names) else (OLD if count==7 else FEATURES);x=pd.DataFrame([[f.get(k,0) for k in cols]],columns=cols);p=int(m.predict(x)[0]);pr=float(m.predict_proba(x)[0,1]);return p,pr,count
    except:return None,None,0

def signal(f,p,pr):
    research=float(np.clip(f["obi_20"]*.35+f["obi_50"]*.2+f["taker_flow_ratio"]*.2+f["sma_distance"]*20*.15+f["FOURIER_TREND"]*.1,-1,1));ml=(pr-.5)*2 if pr is not None else 0.;score=.6*research+.4*ml if p is not None else research;s="LONG" if score>=.45 else "SHORT" if score<=-.45 else "WAIT";conf=round(min(99,max(1,50+abs(score)*49)));return s,score,conf,research

def chart(df,s,tf,show,forward):
    fig=go.Figure()
    if df.empty:return fig
    fig.add_trace(go.Candlestick(x=df.Time,open=df.Open,high=df.High,low=df.Low,close=df.Close,name=s,increasing_fillcolor="#18b77a",decreasing_fillcolor="#e25569",increasing_line_color="#2bd397",decreasing_line_color="#ff7182"))
    for span,name in [(20,"EMA 20"),(50,"EMA 50"),(200,"EMA 200")]:
        if len(df)>=span:fig.add_trace(go.Scatter(x=df.Time,y=df.Close.ewm(span=span,adjust=False).mean(),name=name,mode="lines",line={"width":1.2},hoverinfo="skip"))
    if show:fig.add_trace(go.Bar(x=df.Time,y=df.Volume,name="Volume",opacity=.16,yaxis="y2"))
    step=df.Time.iloc[-1]-df.Time.iloc[-2] if len(df)>1 else pd.Timedelta(minutes=5);right=df.Time.iloc[-1]+step*forward
    fig.update_layout(height=690,template="plotly_dark",paper_bgcolor="#080d14",plot_bgcolor="#080d14",margin={"l":8,"r":8,"t":8,"b":8},hovermode="x unified",dragmode="pan",uirevision=f"{s}-{tf}",xaxis={"range":[df.Time.iloc[max(0,len(df)-180)],right],"rangeslider":{"visible":False},"fixedrange":False,"showgrid":True,"gridcolor":"#172131","showspikes":True,"spikemode":"across"},yaxis={"side":"right","fixedrange":False,"showgrid":True,"gridcolor":"#172131"},yaxis2={"overlaying":"y","side":"left","showticklabels":False,"showgrid":False},legend={"orientation":"h","y":1.02,"x":0})
    fig.add_hline(y=n(df.Close.iloc[-1]),line_width=1,line_dash="dot",opacity=.4);return fig

if "symbol" not in st.session_state:st.session_state.symbol="BTCUSDT"
if "tf" not in st.session_state:st.session_state.tf="5m"
if "auto" not in st.session_state:st.session_state.auto=True
if "secs" not in st.session_state:st.session_state.secs=5

with st.sidebar:
    st.markdown("### ⚡ ZIA RESEARCH");st.caption("Live ML market terminal")
    st.session_state.symbol=st.selectbox("Market",COINS,index=COINS.index(st.session_state.symbol))
    label=st.selectbox("Timeframe",list(TFS),index=list(TFS.values()).index(st.session_state.tf));st.session_state.tf=TFS[label]
    st.markdown("---");limit=st.slider("Candles",100,1000,500,50);forward=st.slider("Future space",10,150,45,5);show=st.checkbox("Volume",True)
    st.markdown("---");st.session_state.auto=st.toggle("Seamless auto refresh",st.session_state.auto);st.session_state.secs=st.slider("Refresh seconds",2,15,st.session_state.secs);st.caption("Only the live panel reruns; the page stays visually stable.")

st.markdown('<div class="hero"><div><div class="brand">ZIA <span>RESEARCH LAB</span></div><div class="sub">ML + ORDER FLOW + PRICE ACTION · LIVE TERMINAL</div></div><div class="live"><span class="dot"></span>LIVE ENGINE</div></div>',unsafe_allow_html=True)

def live():
    s=st.session_state.symbol;tf=st.session_state.tf;df=candles(s,tf,limit);b,a=book(s);f=make_features(df,b,a);p,pr,count=prediction(f);sig,score,conf,research=signal(f,p,pr);cur=n(df.Close.iloc[-1]) if not df.empty else 0;prev=n(df.Close.iloc[-2]) if len(df)>1 else cur;move=(cur/prev-1)*100 if prev else 0
    cs=st.columns(6);vals=[("PRICE",fmt(cur),f"{move:+.2f}%"),("OBI TOP 20",f"{f['obi_20']:+.3f}","BULLISH" if f["obi_20"]>.15 else "BEARISH" if f["obi_20"]<-.15 else "NEUTRAL"),("TOP 20 DEPTH",f"{f['top20_total_depth']:,.2f}","ORDER BOOK"),("SPREAD",fmt(f["spread"]),f"{f['spread_pct']*100:.3f}%"),("ML",("LONG" if p==1 else "SHORT")+f" · {pr*100:.1f}%" if p is not None else "OFFLINE",f"{count} features" if count else "model unavailable"),("CONFIDENCE",f"{conf}%",sig)]
    for c,(l,v,sub) in zip(cs,vals):c.markdown(f'<div class="card"><div class="lab">{l}</div><div class="val">{v}</div><div class="muted">{sub}</div></div>',unsafe_allow_html=True)
    a1,a2=st.columns([2.5,1]);st.markdown('<div class="section">LIVE SIGNAL</div>',unsafe_allow_html=True)
    with a1:st.markdown(f'<div class="signal {"long" if sig=="LONG" else "short" if sig=="SHORT" else ""}"><div class="lab">ZIA ML / ORDER-FLOW DECISION</div><div class="sig">{sig}</div><div class="muted">Composite {score:+.3f} · Research {research:+.3f} · ML {pr*100:.1f}%</div></div>',unsafe_allow_html=True)
    with a2:st.markdown(f'<div class="card"><div class="lab">ORDER BOOK BIAS</div><div class="val">{"BULLISH" if f["obi_20"]>.15 else "BEARISH" if f["obi_20"]<-.15 else "NEUTRAL"}</div><div class="muted">OBI 5/10/20/50<br>{f["obi_5"]:+.2f} / {f["obi_10"]:+.2f} / {f["obi_20"]:+.2f} / {f["obi_50"]:+.2f}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">MARKET CHART</div><div class="chart">',unsafe_allow_html=True)
    st.plotly_chart(chart(df,s,tf,show,forward),use_container_width=True,config={"displaylogo":False,"scrollZoom":True,"doubleClick":"reset","modeBarButtonsToAdd":["drawline","drawrect","eraseshape"],"modeBarButtonsToRemove":["lasso2d","select2d"]},key="tv_chart")
    st.markdown('</div>',unsafe_allow_html=True);st.caption("TradingView-style: mouse wheel zoom · drag pan · double-click reset · crosshair · right price axis · future empty space after latest candle.")
    st.markdown('<div class="section">ORDER FLOW SNAPSHOT</div>',unsafe_allow_html=True);q=st.columns(4)
    for c,(l,v) in zip(q,[("TOP 20 BID",f["top20_bid_sum"]),("TOP 20 ASK",f["top20_ask_sum"]),("TAKER FLOW",f["taker_flow"]),("TOP 50 OBI",f["obi_50"])]):c.markdown(f'<div class="card"><div class="lab">{l}</div><div class="val">{v:,.3f}</div></div>',unsafe_allow_html=True)

if hasattr(st,"fragment"):
    if st.session_state.auto:
        @st.fragment(run_every=f"{st.session_state.secs}s")
        def _run():live()
    else:
        @st.fragment
        def _run():live()
    _run()
else:live()
