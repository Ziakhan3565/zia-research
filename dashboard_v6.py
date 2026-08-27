"""ZIA Research Terminal v6 launcher.

Keeps the original v6 dashboard UI/engine intact and applies only two small
runtime additions:
1) 15M LONG/SHORT signals are locked for 20 minutes.
2) A LIVE TRADES section lists currently locked actionable signals.

The original v6 source is loaded from the immutable pre-patch commit so the
existing dashboard layout, chart, TRI lines, ML lab and research lab are not
redesigned.
"""
from __future__ import annotations

import requests
import streamlit as st

ORIGINAL = "https://raw.githubusercontent.com/Ziakhan3565/zia-research/2c02c995fbb6e30790da8c9644fe7e9ebb29cf16/dashboard_v6.py"

# Load the exact original dashboard source, then make surgical text patches.
r = requests.get(ORIGINAL, timeout=12, headers={"User-Agent": "ZIA-Research"})
r.raise_for_status()
source = r.text

# Rename the original signal function so we can put the 15M lock in front of it.
source = source.replace(
    "def final_state(f,p,pr):",
    "def _zia_raw_final_state(f,p,pr):",
    1,
)

# Inject the lock layer immediately before the TRI mapping function.
marker = "\n# TRI controls are intentionally removed. The chart decides which source timeframes are visible."
lock_code = r'''

# ============================================================
# ZIA ADDITION: 15M SIGNAL LOCK + LIVE TRADES
# Existing dashboard/UI remains unchanged.
# ============================================================
from datetime import timedelta as _zia_timedelta

_ZIA_LOCK_MINUTES = 20

if "_zia_15m_locks" not in st.session_state:
    st.session_state._zia_15m_locks = {}
if "_zia_live_trades" not in st.session_state:
    st.session_state._zia_live_trades = {}

def final_state(f,p,pr):
    """Return the original signal, except 15M LONG/SHORT is frozen 20m."""
    raw = _zia_raw_final_state(f,p,pr)
    # Only 15M is locked. Every other timeframe behaves exactly as before.
    if str(tf) != "15M":
        return raw

    key = f"{symbol}:15M"
    now = datetime.now(timezone.utc)
    old = st.session_state._zia_15m_locks.get(key)

    if old is not None:
        try:
            until = datetime.fromisoformat(old["until"])
            if now < until:
                return tuple(old["result"])
            # Expired: remove the old live trade so a fresh signal can replace it.
            st.session_state._zia_15m_locks.pop(key, None)
            st.session_state._zia_live_trades.pop(key, None)
        except Exception:
            st.session_state._zia_15m_locks.pop(key, None)
            st.session_state._zia_live_trades.pop(key, None)

    signal, confidence, combined, scores, weights, rscore, mlscore = raw
    if signal not in ("LONG", "SHORT"):
        return raw

    until = now + _zia_timedelta(minutes=_ZIA_LOCK_MINUTES)
    result = [signal, confidence, combined, scores, weights, rscore, mlscore]
    st.session_state._zia_15m_locks[key] = {
        "symbol": str(symbol),
        "timeframe": "15M",
        "signal": signal,
        "confidence": float(confidence),
        "combined": float(combined),
        "created": now.isoformat(),
        "until": until.isoformat(),
        "result": result,
    }
    st.session_state._zia_live_trades[key] = {
        "symbol": str(symbol),
        "timeframe": "15M",
        "signal": signal,
        "confidence": float(confidence),
        "entry_time": now.isoformat(),
        "until": until.isoformat(),
        "status": "LOCKED / LIVE",
    }
    return raw

def _zia_live_trade_frame():
    now = datetime.now(timezone.utc)
    rows = []
    expired = []
    for key, trade in list(st.session_state._zia_live_trades.items()):
        try:
            until = datetime.fromisoformat(trade["until"])
            remaining = max(0, int((until-now).total_seconds()))
            if remaining <= 0:
                expired.append(key)
                continue
            rows.append({
                "Crypto": trade["symbol"],
                "TF": trade["timeframe"],
                "Signal": trade["signal"],
                "Confidence": f"{trade['confidence']:.1f}%",
                "Entry Time": pd.to_datetime(trade["entry_time"]).strftime("%H:%M:%S UTC"),
                "Lock Remaining": f"{remaining//60:02d}:{remaining%60:02d}",
                "Status": "LONG LOCKED" if trade["signal"] == "LONG" else "SHORT LOCKED",
            })
        except Exception:
            expired.append(key)
    for key in expired:
        st.session_state._zia_live_trades.pop(key, None)
        st.session_state._zia_15m_locks.pop(key, None)
    return pd.DataFrame(rows)
'''
source = source.replace(marker, lock_code + marker, 1)

# Add the live-trades section inside the EXISTING Signals tab. No new dashboard/layout.
needle = 'with tabs[5]:\n        if st.button("💾 SAVE CURRENT SIGNAL"'
replacement = '''with tabs[5]:
        # LIVE TRADES: only currently active 15M LONG/SHORT locked signals.
        _zia_live = _zia_live_trade_frame()
        st.markdown('<div class="panel"><b>LIVE TRADES</b><div class="section-sub">15M LONG/SHORT signals remain fixed for 20 minutes. No replacement signal is accepted during the lock.</div>',unsafe_allow_html=True)
        if not _zia_live.empty:
            st.dataframe(_zia_live,use_container_width=True,hide_index=True)
        else:
            st.info("No active 15M LONG/SHORT trade right now.")
        st.markdown('</div>',unsafe_allow_html=True)
        if st.button("💾 SAVE CURRENT SIGNAL"'''
if needle not in source:
    raise RuntimeError("Original Signals tab anchor not found; refusing to alter dashboard UI.")
source = source.replace(needle, replacement, 1)

# Execute the original dashboard after the surgical patches.
exec(compile(source, ORIGINAL, "exec"), globals(), globals())
