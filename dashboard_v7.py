from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="ZIA Research Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","SUIUSDT"]
TFS = {"1MO":"1M","1W":"1w","1D":"1d","4H":"4h","1H":"1h","30M":"30m","15M":"15m","5M":"5m","3M":"3m","1M":"1m"}

try:
    from auto_collector import fetch_futures_ohlcv, fetch_futures_order_book, fetch_futures_trades
    from src.research_lab import TenPaperResearchLab, TRILineEngine
    IMPORT_STATUS = "CONNECTED"
except Exception as exc:
    IMPORT_STATUS = f"IMPORT ERROR: {type(exc).__name__}"
    fetch_futures_ohlcv = fetch_futures_order_book = fetch_futures_trades = None
    TenPaperResearchLab = TRILineEngine = None

st.markdown("""
<style>
:root{--bg:#05070b;--panel:#0b1119;--line:#1d2a39;--txt:#edf3fb;--muted:#7f8da1;--green:#42dda0;--red:#ff7184;--amber:#f3c86a;--cyan:#65d7ff;--violet:#969eff}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--txt)}
[data-testid="stHeader"]{background:transparent}.block-container{max-width:1920px;padding:10px clamp(8px,1.5vw,30px) 30px}
.hero{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding:4px 2px 12px;margin-bottom:10px}.brand{font-size:clamp(25px,2.7vw,42px);font-weight:950;letter-spacing:-2px}.brand b{color:var(--violet)}.micro{color:var(--muted);font-size:9px;letter-spacing:1.5px}.live{border:1px solid #245d45;background:#071810;color:#6ce3a5;border-radius:999px;padding:7px 12px;font-size:10px;font-weight:900}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);margin-right:6px}
.panel{background:linear-gradient(145deg,#0e151f,#080d14);border:1px solid var(--line);border-radius:15px;padding:13px;margin-bottom:10px}.card{background:linear-gradient(145deg,#111a26,#0a1018);border:1px solid var(--line);border-radius:14px;padding:11px;min-height:76px}.label{font-size:9px;color:var(--muted);font-weight:900;letter-spacing:1.1px}.value{font-size:19px;font-weight:950;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sub{font-size:10px;color:#8794a8;margin-top:3px}.good{color:var(--green)}.bad{color:var(--red)}.amber{color:var(--amber)}.violet{color:var(--violet)}.cyan{color:var(--cyan)}
.signalbox{border-radius:18px;padding:17px 20px;border:1px solid var(--line);background:linear-gradient(145deg,#111a26,#080d14);text-align:center;margin-bottom:10px}.signal-long{border-color:#277b59}.signal-short{border-color:#843a4a}.signal-wait{border-color:#705e30}.signal-main{font-size:clamp(34px,4vw,58px);font-weight:1000;letter-spacing:-2px;line-height:1}.signal-meta{font-size:10px;color:var(--muted);margin-top:7px;letter-spacing:1px}
.pipeline{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin:7px 0 11px}.stage{background:#0a1119;border:1px solid var(--line);border-radius:9px;padding:7px 9px;font-size:9px;font-weight:900}.arrow{color:var(--muted);font-size:12px}.scorebar{height:9px;border-radius:99px;background:#111a26;border:1px solid var(--line);overflow:hidden}.scorefill{height:100%;border-radius:99px;background:var(--violet)}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:900}div[data-testid="stTabs"] button{font-weight:900;font-size:11px}
@media(max-width:700px){.block-container{padding:6px 7px 22px}.brand{font-size:21px}.micro{font-size:7px}.live{font-size:8px;padding:5px 8px}.panel{padding:9px;border-radius:12px}.card{min-height:62px;padding:9px}.value{font-size:16px}}
</style>
""", unsafe_allow_html=True)

def num(x,d=0.0):
    try:
        v=float(x); return v if np.isfinite(v) else d
    except Exception:return d

def cards(items):
    cs=st.columns(len(items))
    for c,(a,b,d,cl) in zip(cs,items):
        with c: st.markdown(f'<div class="card"><div class="label">{a}</div><div class="value {cl}">{b}</div><div class="sub">{d}</div></div>',unsafe_allow_html=True)

def tri_visible(tf):
    if tf in ("1M","3M","5M","15M","30M"): return [("1H","1H"),("4H","4H")]
    if tf in ("1H","4H"): return [("1D","DAILY"),("1W","WEEKLY")]
    if tf=="1D": return [("1W","WEEKLY"),("1MO","MONTHLY")]
    if tf=="1W": return [("1MO","MONTHLY")]
    return []

@st.cache_resource(show_spinner=False)
def get_lab(symbol):
    if TenPaperResearchLab is None:return None
    try:
        tri=TRILineEngine(symbol=symbol)
        return TenPaperResearchLab(tri_engine=tri)
    except Exception:return None

@st.cache_data(ttl=1,show_spinner=False)
def get_market(symbol,interval):
    if fetch_futures_ohlcv is None:return pd.DataFrame()
    try:return fetch_futures_ohlcv(symbol,interval,650) or pd.DataFrame()
    except Exception:return pd.DataFrame()

@st.cache_data(ttl=1,show_spinner=False)
def get_book(symbol):
    if fetch_futures_order_book is None:return None,None
    try:return fetch_futures_order_book(symbol,100)
    except Exception:return None,None

@st.cache_data(ttl=1,show_spinner=False)
def get_trades(symbol):
    if fetch_futures_trades is None:return []
    try:return fetch_futures_trades(symbol,1000)
    except Exception:return []

def ofi_score(bids,asks):
    if bids is None or asks is None:return 0.0
    now=(float(np.asarray(bids)[:20,1].sum()),float(np.asarray(asks)[:20,1].sum()))
    prev=st.session_state.get("ofi_book")
    st.session_state.ofi_book=now
    if prev is None:return 0.0
    delta_bid=now[0]-prev[0]; delta_ask=now[1]-prev[1]; denom=abs(delta_bid)+abs(delta_ask)+1e-9
    return float(np.clip((delta_bid-delta_ask)/denom,-1,1))

def tri_lines(symbol,tf):
    out=[]
    lab=get_lab(symbol)
    for label,_ in tri_visible(tf):
        if lab is None:continue
        try:
            x=lab.tri_engine.calculate_levels({"1H":"1H","4H":"4H","1D":"DAILY","1W":"WEEKLY","1MO":"MONTHLY"}[label])
            out.append((label,x))
        except Exception:pass
    return out

def chart(df,symbol,tf,future):
    fig=go.Figure()
    if df is None or df.empty:return fig
    v=df.tail(650).copy(); x=v["open_time_utc"]
    fig.add_trace(go.Candlestick(x=x,open=v.open,high=v.high,low=v.low,close=v.close,name="PRICE",increasing_line_color="#42dda0",increasing_fillcolor="#176d4f",decreasing_line_color="#ff7184",decreasing_fillcolor="#8e3448"))
    for p in (20,50,200):
        if len(v)>=p:fig.add_trace(go.Scatter(x=x,y=v.close.ewm(span=p,adjust=False).mean(),mode="lines",name=f"EMA {p}",line={"width":1.05}))
    colors={"1MO":"#b29cff","1W":"#8e98ff","1D":"#65d7ff","1H":"#42dda0","4H":"#f3c86a"}
    for label,lev in tri_lines(symbol,tf):
        for key,text,dash,w,pos in [("body_50","BODY 50","solid",1.9,"top right"),("upper_50","UPPER 50","dot",1.05,"top left"),("lower_50","LOWER 50","dot",1.05,"bottom left")]:
            fig.add_hline(y=float(getattr(lev,key)),line_color=colors[label],line_width=w,line_dash=dash,annotation_text=f"TRI {label} • {text}",annotation_position=pos,annotation_font_size=9)
    step=x.iloc[-1]-x.iloc[-2] if len(x)>1 else pd.Timedelta(minutes=5)
    fig.update_xaxes(range=[x.iloc[0],x.iloc[-1]+step*future],rangeslider_visible=False,fixedrange=False,showgrid=True,gridcolor="#172230",showspikes=True,spikemode="across")
    fig.update_yaxes(side="right",fixedrange=False,showgrid=True,gridcolor="#172230",automargin=True)
    fig.update_layout(height=680,margin=dict(l=4,r=4,t=12,b=8),paper_bgcolor="#080d14",plot_bgcolor="#080d14",font=dict(color="#cbd5e1"),hovermode="x unified",dragmode="pan",uirevision=f"{symbol}:{tf}",legend=dict(orientation="h",y=1.02,x=0))
    return fig

def pipeline_result(symbol,tf,df,bids,asks,trades):
    lab=get_lab(symbol)
    if lab is None:return {"signal":"WAIT","score":0.0,"confidence":0.0,"features":{},"obi":{},"ml":{}}
    # IMPORTANT: convert numpy arrays to lists because research_lab checks list truthiness.
    bl=np.asarray(bids).tolist() if bids is not None else []
    al=np.asarray(asks).tolist() if asks is not None else []
    history=pd.DataFrame()
    try:
        result=lab.calculate_all_signals(df,bl,al,performance_history=history,trades=trades,trade_mode="15M" if tf in ("1M","3M","5M","15M","30M") else tf)
    except Exception as exc:
        return {"signal":"WAIT","score":0.0,"confidence":0.0,"features":{},"obi":{},"ml":{},"error":type(exc).__name__}
    obi={"5":num(result.get("obi_top5")),"10":num(result.get("obi_top10")),"20":num(result.get("obi_top20")),"50":num(result.get("obi_top50")),"final":num(result.get("obi_final"))}
    ofi=ofi_score(bids,asks)
    base=num(result.get("final_score")); data_score=float(np.clip(base*0.85+ofi*0.15,-1,1))
    # Final dashboard signal is driven by the complete data score, while TRI remains the research confirmation/filter.
    tri=result.get("tri_signal","NEUTRAL")
    if data_score>=0.45 and (tri in ("NEUTRAL","LONG")):signal="LONG"
    elif data_score<=-0.45 and (tri in ("NEUTRAL","SHORT")):signal="SHORT"
    else:signal="WAIT"
    conf=float(np.clip(abs(data_score)*100,0,100))
    return {"signal":signal,"score":data_score,"research_score":base,"confidence":conf,"features":result,"obi":obi,"ofi":ofi,"ml":result.get("ml_probability",.5),"tri":tri}

if "symbol" not in st.session_state:st.session_state.symbol="BTCUSDT"
if "tf" not in st.session_state:st.session_state.tf="15M"
if "future" not in st.session_state:st.session_state.future=30

st.markdown('<div class="hero"><div><div class="brand">ZIA <b>RESEARCH</b></div><div class="micro">AUTO COLLECTOR • RESEARCH LAB • OBI • OFI • DATA SCORE • ML</div></div><div class="live"><span class="dot"></span>LIVE • 1S</div></div>',unsafe_allow_html=True)
a,b,c=st.columns([2.2,1.15,1])
with a:symbol=st.selectbox("MARKET",SYMBOLS,index=SYMBOLS.index(st.session_state.symbol),key="symbol")
with b:tf=st.selectbox("TIMEFRAME",list(TFS),index=list(TFS).index(st.session_state.tf),key="tf")
with c:future=st.selectbox("FUTURE SPACE",[12,20,30,45,60],index=2,key="future",format_func=lambda n:f"{n} bars")

visible=tri_visible(tf); names=" + ".join(x[0] for x in visible) if visible else "NONE"
st.markdown(f'<div class="pipeline"><div class="stage">01 AUTO COLLECTOR <span class="cyan">{IMPORT_STATUS}</span></div><div class="arrow">→</div><div class="stage">02 RESEARCH LAB</div><div class="arrow">→</div><div class="stage">03 OBI</div><div class="arrow">→</div><div class="stage">04 OFI</div><div class="arrow">→</div><div class="stage">05 DATA SCORE</div><div class="arrow">→</div><div class="stage">06 FINAL SIGNAL</div><div class="stage">TRI {names}</div></div>',unsafe_allow_html=True)

@st.fragment(run_every="1s")
def live():
    started=time.perf_counter(); df=get_market(symbol,TFS[tf]); bids,asks=get_book(symbol); trades=get_trades(symbol)
    r=pipeline_result(symbol,tf,df,bids,asks,trades); price=num(df.close.iloc[-1]) if not df.empty else 0; prev=num(df.close.iloc[-2]) if len(df)>1 else price; pct=(price/prev-1)*100 if prev else 0
    sig=r["signal"]; cls="signal-long" if sig=="LONG" else "signal-short" if sig=="SHORT" else "signal-wait"; col="good" if sig=="LONG" else "bad" if sig=="SHORT" else "amber"
    st.markdown(f'<div class="signalbox {cls}"><div class="label">FINAL DATA-PIPELINE SIGNAL</div><div class="signal-main {col}">{sig}</div><div class="signal-meta">DATA SCORE {r["score"]:+.3f} • RESEARCH {r.get("research_score",0):+.3f} • OBI {r.get("obi",{}).get("final",0):+.3f} • OFI {r.get("ofi",0):+.3f} • CONFIDENCE {r["confidence"]:.1f}%</div></div>',unsafe_allow_html=True)
    cards([("PRICE",f"${price:,.2f}",f"{pct:+.2f}% • {tf}","good" if pct>=0 else "bad"),("FINAL SIGNAL",sig,f"data score {r['score']:+.3f}",col),("RESEARCH",f"{r.get('research_score',0):+.3f}","Research Lab output","violet"),("OBI",f"{r.get('obi',{}).get('final',0):+.3f}","5/10/20/50 weighted","good" if r.get('obi',{}).get('final',0)>=0 else "bad"),("OFI",f"{r.get('ofi',0):+.3f}","book-flow delta","cyan" if r.get('ofi',0)>=0 else "bad"),("ML",f"{num(r.get('ml',.5))*100:.1f}%","up probability","violet")])
    tabs=st.tabs(["⌂ OVERVIEW","◈ CHART","◌ ORDER FLOW","🧠 ML LAB","🔬 RESEARCH LAB","▣ SIGNALS"])
    with tabs[0]:
        l,rr=st.columns([1.6,1])
        with l:
            st.markdown('<div class="panel"><b>DATA PIPELINE</b><div class="section-sub">The overview now displays the signal produced after the complete collector → research → OBI → OFI → score pipeline.</div>',unsafe_allow_html=True)
            st.write(f"**Auto Collector:** {IMPORT_STATUS}")
            st.write("**Research Lab:** formulas evaluated on the live candle/order-book/trade snapshot")
            st.write(f"**OBI:** Top 5 `{r.get('obi',{}).get('5',0):+.3f}` • Top 10 `{r.get('obi',{}).get('10',0):+.3f}` • Top 20 `{r.get('obi',{}).get('20',0):+.3f}` • Top 50 `{r.get('obi',{}).get('50',0):+.3f}`")
            st.write(f"**OFI:** `{r.get('ofi',0):+.3f}`")
            st.write(f"**DATA SCORE:** `{r['score']:+.3f}` → **{sig}**")
            st.markdown('</div>',unsafe_allow_html=True)
        with rr:
            st.markdown('<div class="panel"><b>LIVE STATUS</b>',unsafe_allow_html=True); st.write(f"Collector: `{IMPORT_STATUS}`"); st.write(f"Candles: `{len(df)}` • Trades: `{len(trades)}`"); st.write(f"Engine: `{(time.perf_counter()-started)*1000:.0f} ms`"); st.write(f"Updated: `{datetime.now().strftime('%H:%M:%S')}`"); st.write(f"TRI references: `{names}`"); st.markdown('</div>',unsafe_allow_html=True)
    with tabs[1]:
        st.markdown(f'<div class="panel"><b>TRADINGVIEW-STYLE MARKET CHART</b><div class="section-sub">{tf} chart • automatic TRI {names} • zoom • pan • crosshair • future space • no manual TRI controls</div>',unsafe_allow_html=True)
        try: st.plotly_chart(chart(df,symbol,tf,future),use_container_width=True,config={"scrollZoom":True,"displaylogo":False,"responsive":True,"doubleClick":"reset","modeBarButtonsToRemove":["lasso2d","select2d"]},key=f"chart_{symbol}_{tf}")
        except Exception as exc: st.error(f"Chart render recovered from error: {type(exc).__name__}")
        st.markdown('</div>',unsafe_allow_html=True)
    with tabs[2]:
        o=r.get("obi",{}); cards([("OBI 5",f"{o.get('5',0):+.3f}","top 5","good" if o.get('5',0)>=0 else "bad"),("OBI 10",f"{o.get('10',0):+.3f}","top 10","good" if o.get('10',0)>=0 else "bad"),("OBI 20",f"{o.get('20',0):+.3f}","top 20","good" if o.get('20',0)>=0 else "bad"),("OBI 50",f"{o.get('50',0):+.3f}","top 50","good" if o.get('50',0)>=0 else "bad")]); l,rr=st.columns(2); l.dataframe(pd.DataFrame(bids[:20],columns=["Bid Price","Bid Qty"]) if bids is not None else pd.DataFrame(),use_container_width=True,hide_index=True); rr.dataframe(pd.DataFrame(asks[:20],columns=["Ask Price","Ask Qty"]) if asks is not None else pd.DataFrame(),use_container_width=True,hide_index=True); st.info(f"OFI score: {r.get('ofi',0):+.3f}")
    with tabs[3]:
        f=r.get("features",{}); cards([("ML",f.get("ml_status","NOT_TRAINED"),"Research Lab model","violet"),("UP",f"{num(r.get('ml',.5))*100:.1f}%","probability up","good"),("DOWN",f"{(1-num(r.get('ml',.5)))*100:.1f}%","probability down","bad"),("ML SCORE",f"{num(f.get('ml_score',0)):+.3f}","model contribution","cyan")]); st.json({k:v for k,v in f.items() if k in ("ml_direction","ml_strength","ml_accuracy","ml_samples","ml_trained")})
    with tabs[4]:
        f=r.get("features",{}); rows=[]
        for k in ("BOOK_IMB","TAKER_FLOW","QUANT_IMPLY","ADAPT_CONF","BAYESIAN","FOURIER_TREND"):
            rows.append({"Formula":k,"Live Score":num(f.get(k,0))})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True); st.write(f"Research Lab score: **{r.get('research_score',0):+.3f}** • TRI confirmation: **{r.get('tri','NEUTRAL')}**")
    with tabs[5]:
        p=ROOT/"saved_signals.csv"
        if p.exists():
            h=pd.read_csv(p); st.dataframe(h.tail(80).iloc[::-1],use_container_width=True,hide_index=True)
        else:st.info("No saved signals yet.")
    st.caption(f"ZIA Research • {symbol} • {tf} • live pipeline • {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

live()
