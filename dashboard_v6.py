from __future__ import annotations

from pathlib import Path
import datetime as dt
import streamlit as st

ROOT = Path(__file__).resolve().parent
TV = ROOT / "dashboard_tv.py"

# Keep the existing dashboard UI. Only inject the requested 15M lock + LIVE TRADES.
source = TV.read_text(encoding="utf-8")

needle = 'sig,score,conf,research=composite(features,pred,prob)\nprice=num(df.Close.iloc[-1]) if not df.empty else 0; prev=num(df.Close.iloc[-2] if len(df)>1 else price); change=(price/prev-1)*100 if prev else 0\n'
patch = r'''sig,score,conf,research=composite(features,pred,prob)

# ============================================================
# ZIA MINIMAL ADDITION: 15M SIGNAL LOCK + LIVE TRADES
# Existing dashboard/UI is intentionally preserved.
# ============================================================
_ZIA_LOCK_MINUTES = 20
if "_zia_15m_trade" not in st.session_state:
    st.session_state._zia_15m_trade = None

_now = dt.datetime.now(dt.timezone.utc)
_zia_is_15m = st.session_state.interval == "15m"
_zia_trade = st.session_state._zia_15m_trade

# Expire the old 15M trade only after its fixed 20-minute window.
if _zia_trade is not None:
    try:
        _until = dt.datetime.fromisoformat(_zia_trade["until"])
        if _now >= _until:
            st.session_state._zia_15m_trade = None
            _zia_trade = None
    except Exception:
        st.session_state._zia_15m_trade = None
        _zia_trade = None

# While locked, NEVER allow a new 15M LONG/SHORT to replace the active signal.
if _zia_is_15m and _zia_trade is not None:
    sig = _zia_trade["signal"]
    score = float(_zia_trade["score"])
    conf = float(_zia_trade["confidence"])
    research = float(_zia_trade["research"])

# Create a new locked trade only when a fresh 15M actionable signal appears.
if _zia_is_15m and _zia_trade is None and sig in ("LONG", "SHORT"):
    _entry = float(price) if 'price' in locals() else (num(df.Close.iloc[-1]) if not df.empty else 0.0)
    _atr = float((df.High - df.Low).tail(14).mean()) if not df.empty else 0.0
    _atr = max(_atr, _entry * 0.001) if _entry else _atr
    if sig == "LONG":
        _tp1, _tp2, _sl = _entry + _atr, _entry + 2*_atr, _entry - 0.75*_atr
    else:
        _tp1, _tp2, _sl = _entry - _atr, _entry - 2*_atr, _entry + 0.75*_atr
    _until = _now + dt.timedelta(minutes=_ZIA_LOCK_MINUTES)
    st.session_state._zia_15m_trade = {
        "symbol": st.session_state.symbol,
        "timeframe": "15M",
        "signal": sig,
        "confidence": float(conf),
        "score": float(score),
        "research": float(research),
        "entry": _entry,
        "tp1": _tp1,
        "tp2": _tp2,
        "sl": _sl,
        "created": _now.isoformat(),
        "until": _until.isoformat(),
    }
    _zia_trade = st.session_state._zia_15m_trade

# Price variables used by the original dashboard remain unchanged.
price=num(df.Close.iloc[-1]) if not df.empty else 0; prev=num(df.Close.iloc[-2] if len(df)>1 else price); change=(price/prev-1)*100 if prev else 0
'''

if needle not in source:
    raise RuntimeError("Expected dashboard_tv signal marker was not found; dashboard left untouched.")
source = source.replace(needle, patch, 1)

# Insert the LIVE TRADES section before the existing Market Chart. This does not alter existing panels.
chart_marker = 'st.markdown(\'<div class="section">MARKET CHART</div>\',unsafe_allow_html=True)\n'
trade_panel = r'''# Existing dashboard + one additional LIVE TRADES section.
st.markdown('<div class="section">LIVE TRADES</div>',unsafe_allow_html=True)
if _zia_trade is not None and _zia_trade.get("signal") in ("LONG","SHORT"):
    try:
        _remaining=max(0,int((dt.datetime.fromisoformat(_zia_trade["until"])-dt.datetime.now(dt.timezone.utc)).total_seconds()))
    except Exception:
        _remaining=0
    _mins,_secs=divmod(_remaining,60)
    _hours,_mins=divmod(_mins,60)
    _lock=f"{_hours:02d}:{_mins:02d}:{_secs:02d}"
    _sigcls="long" if _zia_trade["signal"]=="LONG" else "short"
    _entry=float(_zia_trade["entry"]); _tp1=float(_zia_trade["tp1"]); _tp2=float(_zia_trade["tp2"]); _sl=float(_zia_trade["sl"])
    _html=(
        f'<div class="sigbox {_sigcls}"><div class="label">15M LIVE TRADE • LOCKED</div>'
        f'<div class="sig">{_zia_trade["symbol"]} · {_zia_trade["signal"]}</div>'
        f'<div class="sub">Entry {fmt_price(_entry)} · TP1 {fmt_price(_tp1)} · TP2 {fmt_price(_tp2)} · SL {fmt_price(_sl)} · Confidence {_zia_trade["confidence"]:.1f}% · Lock {_lock}</div>'
        f'</div>'
    )
    st.markdown(_html,unsafe_allow_html=True)
else:
    st.markdown('<div class="panel"><div class="sub">No active 15M LONG/SHORT trade. WAIT signals are not locked.</div></div>',unsafe_allow_html=True)

''' + chart_marker
if chart_marker not in source:
    raise RuntimeError("Expected market chart marker was not found; dashboard left untouched.")
source = source.replace(chart_marker, trade_panel, 1)

exec(compile(source, str(TV), "exec"), globals(), globals())
