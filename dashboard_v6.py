"""ZIA Research Terminal v6.

Keeps the original v6 dashboard as the source of truth. The launcher only
adds the requested 15M signal lock and LIVE TRADES section when the original
source is available. It never replaces the dashboard UI with a new design.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
ORIGINAL = "https://raw.githubusercontent.com/Ziakhan3565/zia-research/2c02c995fbb6e30790da8c9644fe7e9ebb29cf16/dashboard_v6.py"


def _load_original():
    """Load the original v6 source; fall back to the local TV dashboard."""
    try:
        r = requests.get(ORIGINAL, timeout=8, headers={"User-Agent": "ZIA-Research"})
        r.raise_for_status()
        return r.text
    except Exception:
        # Streamlit Cloud/network failure must not make the app crash.
        fallback = ROOT / "dashboard_tv.py"
        if fallback.exists():
            return fallback.read_text(encoding="utf-8")
        raise RuntimeError("Unable to load the original dashboard source.")


def _patch(source: str) -> str:
    """Apply only the two requested additions to the original source."""
    if "def final_state(f,p,pr):" not in source:
        return source

    source = source.replace(
        "def final_state(f,p,pr):",
        "def _zia_raw_final_state(f,p,pr):",
        1,
    )

    marker = "\n# TRI controls are intentionally removed. The chart decides which source timeframes are visible."
    if marker not in source:
        return source

    lock_code = r'''

# ============================================================
# ZIA ADDITION: 15M LONG/SHORT LOCK + LIVE TRADES
# ============================================================
from datetime import timedelta as _zia_timedelta

_ZIA_LOCK_MINUTES = 20
if "_zia_15m_locks" not in st.session_state:
    st.session_state._zia_15m_locks = {}
if "_zia_live_trades" not in st.session_state:
    st.session_state._zia_live_trades = {}

def final_state(f,p,pr):
    raw = _zia_raw_final_state(f,p,pr)
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
        except Exception:
            pass
        st.session_state._zia_15m_locks.pop(key, None)
        st.session_state._zia_live_trades.pop(key, None)
    signal, confidence, combined, scores, weights, rscore, mlscore = raw
    if signal not in ("LONG", "SHORT"):
        return raw
    until = now + _zia_timedelta(minutes=_ZIA_LOCK_MINUTES)
    result = [signal, confidence, combined, scores, weights, rscore, mlscore]
    st.session_state._zia_15m_locks[key] = {
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
    }
    return raw

def _zia_live_trade_frame():
    now = datetime.now(timezone.utc)
    rows = []
    for key, trade in list(st.session_state._zia_live_trades.items()):
        try:
            until = datetime.fromisoformat(trade["until"])
            remaining = max(0, int((until-now).total_seconds()))
            if remaining <= 0:
                st.session_state._zia_live_trades.pop(key, None)
                st.session_state._zia_15m_locks.pop(key, None)
                continue
            rows.append({
                "Crypto": trade["symbol"],
                "TF": "15M",
                "Signal": trade["signal"],
                "Confidence": f"{trade['confidence']:.1f}%",
                "Entry Time": pd.to_datetime(trade["entry_time"]).strftime("%H:%M:%S UTC"),
                "Lock Remaining": f"{remaining//60:02d}:{remaining%60:02d}",
                "Status": "LONG LOCKED" if trade["signal"] == "LONG" else "SHORT LOCKED",
            })
        except Exception:
            st.session_state._zia_live_trades.pop(key, None)
            st.session_state._zia_15m_locks.pop(key, None)
    return pd.DataFrame(rows)
'''
    source = source.replace(marker, lock_code + marker, 1)

    needle = 'with tabs[5]:\n        if st.button("💾 SAVE CURRENT SIGNAL"'
    if needle in source:
        replacement = '''with tabs[5]:
        _zia_live = _zia_live_trade_frame()
        st.markdown('<div class="panel"><b>LIVE TRADES</b><div class="section-sub">15M LONG/SHORT signals are locked for 20 minutes. No new 15M signal replaces an active one.</div>',unsafe_allow_html=True)
        if not _zia_live.empty:
            st.dataframe(_zia_live,use_container_width=True,hide_index=True)
        else:
            st.info("No active 15M LONG/SHORT trade right now.")
        st.markdown('</div>',unsafe_allow_html=True)
        if st.button("💾 SAVE CURRENT SIGNAL"'''
        source = source.replace(needle, replacement, 1)
    return source


source = _patch(_load_original())

# The original source already calls st.set_page_config once; execute it directly.
exec(compile(source, "dashboard_v6_original.py", "exec"), globals(), globals())
