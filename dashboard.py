from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import joblib, numpy as np, pandas as pd, plotly.graph_objects as go, requests, streamlit as st

st.set_page_config(page_title="ZIA Research Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
ROOT=Path(__file__).resolve().parent; MODEL_FILE=ROOT/"xgboost_obi_model.pkl"; SIGNAL_FILE=ROOT/"saved_signals.csv"; TRADE_FILE=ROOT/"trade_history.csv"
FUTURES=["https://fapi.binance.com","https://fapi1.binance.com","https://fapi2.binance.com","https://fapi3.binance.com","https://fapi4.binance.com"]; SPOT=["https://api.binance.com","https://api1.binance.com","https://api2.binance.com","https://api3.binance.com"]; DATA=["https://data-api.binance.vision"]
SYMBOLS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT"]; TFS={"5M":"5m","15M":"15m","30M":"30m","1H":"1h","4H":"4h","1D":"1d","1W":"1w"}

st.markdown('''<style>
:root{--bg:#05070b;--p:#0b1119;--p2:#101925;--line:#1d2a39;--txt:#edf3fb;--muted:#7d8ba0;--violet:#8e98ff;--green:#42dda0;--red:#ff7184;--amber:#f3c86a;--cyan:#65d7ff}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--txt)} [data-testid="stHeader"]{background:rgba(5,7,11,.88)}
.block-container{max-width:1920px;padding:12px clamp(8px,1.8vw,34px) 45px}.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding:4px 2px 12px;margin-bottom:10px}.brand{font-size:clamp(22px,2.6vw,36px);font-weight:950;letter-spacing:-1.5px}.brand b{color:var(--violet)}.micro{color:var(--muted);font-size:9px;letter-spacing:1.5px}.live{border:1px solid #235c43;background:#071810;color:#6ce3a5;border-radius:999px;padding:7px 12px;font-size:10px;font-weight:900}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);margin-right:6px}
.panel{background:linear-gradient(145deg,#0d141e,#080d14);border:1px solid var(--line);border-radius:15px;padding:13px}.card{background:linear-gradient(145deg,#111a26,#0a1018);border:1px solid var(--line);border-radius:14px;padding:12px;min-height:82px}.label{font-size:9px;color:var(--muted);font-weight:900;letter-spacing:1.1px}.value{font-size:21px;font-weight:950;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sub{font-size:10px;color:#8794a8;margin-top:3px}.section-title{font-size:18px;font-weight:950;margin:12px 0 2px}.section-sub{font-size:10px;color:var(--muted);margin-bottom:9px}.good{color:var(--green)}.bad{color:var(--red)}.amber{color:var(--amber)}.violet{color:var(--violet)}.cyan{color:var(--cyan)}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:900} div[data-testid="stTabs"] button{font-weight:900;font-size:11px}
@media(max-width:700px){.block-container{padding:6px 7px 30px}.brand{font-size:21px}.micro{font-size:7px}.live{font-size:8px;padding:5px 8px}.panel{padding:9px;border-radius:12px}.card{min-height:65px;padding:9px}.value{font-size:16px}.section-title{font-size:14px}.stTabs [data-baseweb="tab"]{font-size:9px;padding:6px}.stTabs [data-baseweb="tab-list"]{overflow-x:auto}}
</style>''',unsafe_allow_html=True)

def num(x,d=0.):
    try:
        v=float(x); return v if np.isfinite(v) else d
    except: return d

def api(hosts,path,params):
    err="network"
    for h in hosts:
        try:
            r=requests.get(h+path,params=params,timeout=3.5,headers={"User-Agent":"ZIA-Research"})
            if r.ok:return r.json(),h,"OK"
            err=f"HTTP {r.status_code}"
        except requests.RequestException as e:err=type(e).__name__
    return None,None,err

@st.cache_data(ttl=2,show_spinner=False)
def candles(symbol,interval,limit=500):
    raw,host,status=api(FUTURES,"/fapi/v1/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1500)}); source="Futures"
    if not isinstance(raw,list):raw,host,status=api(SPOT,"/api/v3/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1000)});source="Spot"
    if not isinstance(raw,list):raw,host,status=api(DATA,"/api/v3/klines",{"symbol":symbol,"interval":interval,"limit":min(limit,1000)});source="Data API"
    rows=[]
    for c in raw or []:
        try:rows.append([pd.to_datetime(int(c[0]),unit="ms",utc=True),num(c[1]),num(c[2]),num(c[3]),num(c[4]),num(c[5]),num(c[9])])
        except:pass
    return pd.DataFrame(rows,columns=["Time","Open","High","Low","Close","Volume","TakerBuy"]),source,status,host

@st.cache_data(ttl=2,show_spinner=False)
def orderbook(symbol):
    raw,host,status=api(FUTURES,"/fapi/v1/depth",{"symbol":symbol,"limit":100});source="Futures"
    if not isinstance(raw,dict) or not raw.get("bids"):raw,host,status=api(SPOT,"/api/v3/depth",{"symbol":symbol,"limit":100});source="Spot"
    if not isinstance(raw,dict) or not raw.get("bids"):raw,host,status=api(DATA,"/api/v3/depth",{"symbol":symbol,"limit":100});source="Data API"
    try:return np.asarray(raw.get("bids",[]),float),np.asarray(raw.get("asks",[]),float),source,status,host
    except:return np.empty((0,2)),np.empty((0,2)),source,status,host

def obi(b,a,k):
    if len(b)==0 or len(a)==0:return 0.,0.,0.
    k=min(k,len(b),len(a));bv=float(b[:k,1].sum());av=float(a[:k,1].sum());return ((bv-av)/(bv+av) if bv+av else 0.),bv,av

def features(df,b,a):
    f={k:0. for k in ["top20_bid_sum","top20_ask_sum","obi_5","obi_10","obi_20","obi_50","spread","spread_pct","bid_ask_ratio_20","bid_ask_ratio_50","top20_total_depth","top50_total_depth","taker_buy_volume","taker_sell_volume","taker_flow","taker_flow_ratio","price_return","price_change","sma_distance","realized_volatility","BOOK_IMB","QUANT_IMPLY","ADAPT_CONF","BAYESIAN","FOURIER_TREND"]}
    (o5,b5,a5),(o10,b10,a10),(o20,b20,a20),(o50,b50,a50)=[obi(b,a,k) for k in (5,10,20,50)];f.update(top20_bid_sum=b20,top20_ask_sum=a20,obi_5=o5,obi_10=o10,obi_20=o20,obi_50=o50,top20_total_depth=b20+a20,top50_total_depth=b50+a50)
    if df.empty:return f
    c=df.Close;last=num(c.iloc[-1]);prev=num(c.iloc[-2] if len(c)>1 else last);sma=num(c.rolling(20).mean().iloc[-1],last);total=num(df.Volume.tail(20).sum());buy=num(df.TakerBuy.tail(20).sum());sell=max(total-buy,0);flow=buy-sell;spread=num(a[0,0]-b[0,0]) if len(a) and len(b) else 0;trend=np.tanh((last/sma-1)*100) if sma else 0;rv=num(c.pct_change().tail(30).std());four=np.tanh(c.pct_change().tail(16).mean()*1000)
    f.update(spread=spread,spread_pct=spread/last if last else 0,bid_ask_ratio_20=b20/a20 if a20 else 1,bid_ask_ratio_50=b50/a50 if a50 else 1,taker_buy_volume=buy,taker_sell_volume=sell,taker_flow=flow,taker_flow_ratio=flow/total if total else 0,price_return=last/prev-1 if prev else 0,price_change=last-prev,sma_distance=last/sma-1 if sma else 0,realized_volatility=rv,BOOK_IMB=o20,QUANT_IMPLY=float(np.tanh((o20+o50+trend)/3)),ADAPT_CONF=float(np.clip(.5+(abs(o20)+abs(trend))/2,0,1)),BAYESIAN=float(np.clip(.5+(o20+trend)/4,0,1)),FOURIER_TREND=float(four));return f

@st.cache_resource(show_spinner=False)
def load_model():
    try:return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None
    except:return None

def ml(f):
    m=load_model()
    if m is None:return None,None,"MODEL NOT FOUND",0
    try:
        names=list(m.get_booster().feature_names or []) if hasattr(m,"get_booster") else []; count=int(getattr(m,"n_features_in_",len(names) or 25)); f7=["top20_bid_sum","top20_ask_sum","obi_top20","spread","bid_ask_ratio","total_depth","trend_signal"];cols=names if names else (f7 if count==7 else list(f.keys()));row=dict(f,obi_top20=f["obi_20"],bid_ask_ratio=f["bid_ask_ratio_20"],total_depth=f["top20_total_depth"],trend_signal=f["sma_distance"]);x=pd.DataFrame([[row.get(c,0.) for c in cols]],columns=cols);p=int(m.predict(x)[0]);pr=float(m.predict_proba(x)[0][-1]) if hasattr(m,"predict_proba") else None;return p,pr,"OK",len(cols)
    except Exception as e:return None,None,"ML ERROR: "+type(e).__name__,0

def research(f):
    s={"OBI 20":np.clip(f["obi_20"]*2,-1,1),"OBI 20+50":np.clip((f["obi_20"]+f["obi_50"])/1.5,-1,1),"OFI / Taker":np.clip(f["taker_flow_ratio"]*2,-1,1),"Trend / SMA":np.clip(np.tanh(f["sma_distance"]*100),-1,1),"Fourier":np.clip(f["FOURIER_TREND"],-1,1),"Bayesian":np.clip((f["BAYESIAN"]-.5)*2,-1,1),"Quant Imply":np.clip(f["QUANT_IMPLY"],-1,1),"Adaptive":np.clip((f["ADAPT_CONF"]-.5)*2,-1,1)};w={"OBI 20":.22,"OBI 20+50":.14,"OFI / Taker":.20,"Trend / SMA":.14,"Fourier":.10,"Bayesian":.08,"Quant Imply":.07,"Adaptive":.05};return s,w,float(sum(s[k]*w[k] for k in s))

def state(f,p,pr):
    s,w,rs=research(f);m=(pr-.5)*2 if pr is not None else (1 if p==1 else -1 if p==0 else 0);combined=.6*rs+.4*m if p is not None else rs;sig="LONG" if combined>=.45 else "SHORT" if combined<=-.45 else "WAIT";return sig,float(np.clip(50+abs(combined)*49,1,99)),combined,s,w,rs,m

def tri_periods(tf):return [("4H","4h"),("1H","1h")] if tf=="15M" else [("DAY","1d"),("WEEK","1w"),("MONTH","1M")] if tf in ("1H","4H") else []
@st.cache_data(ttl=30,show_spinner=False)
def tri(symbol,tf):
    out=[]
    for label,interval in tri_periods(tf):
        d,_,_,_=candles(symbol,interval,5)
        if len(d)<2:continue
        c=d.iloc[-2];o,h,l,cl=map(num,[c.Open,c.High,c.Low,c.Close]);hi=max(o,cl);lo=min(o,cl);out.append((label,{"BODY 50":(hi+lo)/2,"UPPER 50":(h+hi)/2,"LOWER 50":(l+lo)/2}))
    return out

def make_chart(df,symbol,tf,future,show):
    fig=go.Figure()
    if df.empty:return fig
    fig.add_trace(go.Candlestick(x=df.Time,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="PRICE",increasing_line_color="#42dda0",increasing_fillcolor="#176d4f",decreasing_line_color="#ff7184",decreasing_fillcolor="#8e3448"))
    for span in (20,50,200):
        if len(df)>=span:fig.add_trace(go.Scatter(x=df.Time,y=df.Close.ewm(span=span,adjust=False).mean(),mode="lines",name=f"EMA {span}",line={"width":1}))
    if show:
        for period,levels in tri(symbol,tf):
            for lab,dash in (("BODY 50","solid"),("UPPER 50","dot"),("LOWER 50","dot")):fig.add_hline(y=levels[lab],line_dash=dash,line_width=1,opacity=.72,annotation_text=f"TRI {period} • {lab}",annotation_position="top right")
    step=df.Time.iloc[-1]-df.Time.iloc[-2] if len(df)>1 else pd.Timedelta(minutes=5);start=max(0,len(df)-350);fig.update_xaxes(range=[df.Time.iloc[start],df.Time.iloc[-1]+step*future],rangeslider_visible=False,showgrid=True,gridcolor="#172230",showspikes=True,spikemode="across");fig.update_yaxes(side="right",showgrid=True,gridcolor="#172230");fig.update_layout(height=610,margin=dict(l=5,r=5,t=8,b=8),paper_bgcolor="#080d14",plot_bgcolor="#080d14",font=dict(color="#cbd5e1"),hovermode="x unified",dragmode="pan",legend=dict(orientation="h",y=1.02,x=0));return fig

def cards(items):
    cs=st.columns(len(items))
    for c,(lab,val,sub,cl) in zip(cs,items):
        with c:st.markdown(f'<div class="card"><div class="label">{lab}</div><div class="value {cl}">{val}</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)

def read(path):
    try:return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except:return pd.DataFrame()

def save_signal(symbol,tf,price,sig,conf,pr,f,rs):
    row={"timestamp":datetime.now(timezone.utc).isoformat(),"symbol":symbol,"timeframe":tf,"price":price,"signal":sig,"confidence":conf,"ml_probability":pr if pr is not None else "","obi20":f["obi_20"],"obi50":f["obi_50"],"ofi":f["taker_flow_ratio"],"research_score":rs};pd.DataFrame([row]).to_csv(SIGNAL_FILE,mode="a",header=not SIGNAL_FILE.exists(),index=False)

if "symbol" not in st.session_state:st.session_state.symbol="BTCUSDT"
if "tf" not in st.session_state:st.session_state.tf="15M"
if "refresh" not in st.session_state:st.session_state.refresh=5
st.markdown('<div class="hero"><div><div class="brand">ZIA <b>RESEARCH</b></div><div class="micro">QUANT MARKET INTELLIGENCE • LIVE ML • ORDER FLOW • RESEARCH LAB</div></div><div class="live"><span class="dot"></span>LIVE ENGINE</div></div>',unsafe_allow_html=True)
c=st.columns([2.1,1.2,1,1,1]);
with c[0]:symbol=st.selectbox("MARKET",SYMBOLS,index=SYMBOLS.index(st.session_state.symbol) if st.session_state.symbol in SYMBOLS else 0,key="symbol")
with c[1]:tf=st.selectbox("TIMEFRAME",list(TFS),index=list(TFS).index(st.session_state.tf),key="tf")
with c[2]:refresh=st.selectbox("UPDATE",[2,3,5,10,15],index=2,key="refresh")
with c[3]:show=st.toggle("TRI LINES",True)
with c[4]:future=st.selectbox("FUTURE",[12,20,28,40],index=2,format_func=lambda x:f"{x} bars")

@st.fragment(run_every="5s")
def live():
    # The fragment owns live data only. Navigation and page shell are not timer-rerun.
    df,source,cstat,chost=candles(symbol,TFS[tf],500);b,a,bsrc,bstat,bhost=orderbook(symbol);f=features(df,b,a);p,pr,mstat,fc=ml(f);sig,conf,combined,rs,rw,rscore,mlscore=state(f,p,pr);price=num(df.Close.iloc[-1]) if not df.empty else 0;prev=num(df.Close.iloc[-2]) if len(df)>1 else price;chg=(price/prev-1)*100 if prev else 0
    st.markdown('<div class="section-title">⚡ Command Center</div><div class="section-sub">Live values update in-place; the page itself is not timer-rerun.</div>',unsafe_allow_html=True)
    cards([("PRICE",f"${price:,.2f}",f"{chg:+.2f}% • {tf}","good" if chg>=0 else "bad"),("FINAL SIGNAL",sig,f"strength {conf:.1f}%","good" if sig=="LONG" else "bad" if sig=="SHORT" else "amber"),("ML",f"{pr*100:.1f}%" if pr is not None else "—",mstat,"violet"),("OBI 20",f"{f['obi_20']:+.3f}","top 20 levels","good" if f['obi_20']>=0 else "bad"),("OFI",f"{f['taker_flow_ratio']:+.3f}","taker flow ratio","good" if f['taker_flow_ratio']>=0 else "bad"),("DATA",source,f"book: {bsrc}","cyan")])
    tabs=st.tabs(["⌂ OVERVIEW","◈ CHART","◌ ORDER FLOW","🧠 ML LAB","🔬 RESEARCH LAB","▣ SIGNALS"])
    with tabs[0]:
        x,y=st.columns([2,1]);
        with x:
            st.markdown('<div class="panel"><b>MARKET REGIME</b>',unsafe_allow_html=True);reg="BULLISH FLOW" if combined>.25 else "BEARISH FLOW" if combined<-.25 else "BALANCED / WAIT";st.markdown(f"## {reg}");st.progress(min(max(conf/100,0),1),text=f"Signal strength {conf:.1f}%");st.write(f"Research **{rscore:+.3f}** • ML **{mlscore:+.3f}** • Composite **{combined:+.3f}**");st.markdown('</div>',unsafe_allow_html=True)
        with y:
            st.markdown('<div class="panel"><b>CONNECTION</b>',unsafe_allow_html=True);st.write(f"Candles: `{source}`");st.write(f"Order book: `{bsrc}`");st.write(f"Book status: `{bstat}`");st.write(f"Updated: `{datetime.now().strftime('%H:%M:%S')}`");st.markdown('</div>',unsafe_allow_html=True)
    with tabs[1]:
        st.markdown('<div class="panel">',unsafe_allow_html=True);st.plotly_chart(make_chart(df,symbol,tf,future,show),use_container_width=True,config={"scrollZoom":True,"displaylogo":False,"responsive":True,"modeBarButtonsToAdd":["drawline","drawrect","eraseshape"]});st.markdown('</div>',unsafe_allow_html=True)
    with tabs[2]:
        if len(b) and len(a):
            vals=[obi(b,a,k) for k in (5,10,20,50)];cards([(f"OBI {k}",f"{v[0]:+.3f}",f"B {v[1]:,.1f} / A {v[2]:,.1f}","good" if v[0]>=0 else "bad") for k,v in zip((5,10,20,50),vals)]);l,r=st.columns(2);l.dataframe(pd.DataFrame(b[:15],columns=["Bid Price","Bid Qty"]),use_container_width=True,hide_index=True);r.dataframe(pd.DataFrame(a[:15],columns=["Ask Price","Ask Qty"]),use_container_width=True,hide_index=True)
        else:st.warning(f"Order book unavailable • {bstat}")
    with tabs[3]:
        cards([("MODEL",mstat,"xgboost_obi_model.pkl","violet"),("PREDICTION","LONG" if p==1 else "SHORT" if p==0 else "—",f"class {p}","good" if p==1 else "bad" if p==0 else "amber"),("PROBABILITY",f"{pr*100:.2f}%" if pr is not None else "—","model probability","violet"),("FEATURES",str(fc),"supplied to model","cyan")]);st.markdown('<div class="panel"><b>LIVE MODEL INPUTS</b>',unsafe_allow_html=True);st.dataframe(pd.DataFrame({"Feature":["OBI 5","OBI 10","OBI 20","OBI 50","Spread","Taker Flow","Trend/SMA","Volatility"],"Value":[f["obi_5"],f["obi_10"],f["obi_20"],f["obi_50"],f["spread"],f["taker_flow_ratio"],f["sma_distance"],f["realized_volatility"]]}),use_container_width=True,hide_index=True);st.markdown('</div>',unsafe_allow_html=True)
    with tabs[4]:
        rd=pd.DataFrame([{"Formula":k,"Live Score":round(float(v),4),"Weight %":round(rw[k]*100,1),"Contribution":round(float(v*rw[k]),4),"Direction":"BULL" if v>0 else "BEAR" if v<0 else "NEUTRAL"} for k,v in rs.items()]).sort_values("Contribution",ascending=False);st.markdown('<div class="panel"><b>RESEARCH FORMULA SCOREBOARD</b>',unsafe_allow_html=True);st.dataframe(rd,use_container_width=True,hide_index=True);st.write(f"Strongest current contributor: **{rd.iloc[0]['Formula'] if not rd.empty else '—'}** • Composite **{rscore:+.3f}**");st.markdown('</div>',unsafe_allow_html=True)
    with tabs[5]:
        if st.button("💾 SAVE CURRENT SIGNAL",use_container_width=True):save_signal(symbol,tf,price,sig,conf,pr,f,rscore);st.success("Signal saved")
        h=read(SIGNAL_FILE);t=read(TRADE_FILE)
        if not h.empty:st.dataframe(h.tail(60).iloc[::-1],use_container_width=True,hide_index=True);st.download_button("⬇ Download Signal Journal",h.to_csv(index=False),"zia_saved_signals.csv","text/csv",use_container_width=True)
        else:st.info("No saved signals yet.")
        if not t.empty and "result" in t.columns:
            rr=t.result.astype(str).str.upper();wins=int((rr=="WIN").sum());loss=int((rr=="LOSS").sum());tot=wins+loss;wr=wins/tot*100 if tot else 0;cards([("CLOSED",str(tot),"resolved trades","cyan"),("WINS",str(wins),"winning trades","good"),("LOSSES",str(loss),"losing trades","bad"),("WIN RATE",f"{wr:.1f}%","closed trade rate","violet")])
        else:st.caption("Trade history will appear when trade_history.csv is available.")
    st.caption(f"ZIA Research • {symbol} • {tf} • live fragment • {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

live()
