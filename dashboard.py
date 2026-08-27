from dashboard_tv import *

# ============================================================
# LIVE TRADES — additive layer only
# Keeps dashboard_tv UI/chart/ML/research unchanged.
# 15M actionable LONG/SHORT signals are frozen for 20 minutes.
# ============================================================
from datetime import datetime, timezone, timedelta
import json as _json

_LIVE_LOCK_FILE = ROOT / ".live_15m_locks.json"
_LIVE_LOCK_MINUTES = 20


def _read_live_locks():
    try:
        data = _json.loads(_LIVE_LOCK_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_live_locks(data):
    try:
        _LIVE_LOCK_FILE.write_text(_json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _atr14(df):
    if df is None or df.empty:
        return 0.0
    try:
        return float((df["High"] - df["Low"]).tail(14).mean())
    except Exception:
        return 0.0


def _fresh_15m_signal(symbol):
    df15 = get_klines(symbol, "15m", 120)
    bids15, asks15 = get_book(symbol)
    if df15 is None or df15.empty or len(df15) < 25:
        return None
    f15 = build_features(df15, bids15, asks15)
    pred15, prob15, _ = ml_predict(f15)
    direction, score15, conf15, research15 = composite(f15, pred15, prob15)
    if direction not in ("LONG", "SHORT"):
        return None
    entry = float(df15.Close.iloc[-1])
    atr = _atr14(df15)
    if not np.isfinite(entry) or atr <= 0:
        return None
    if direction == "LONG":
        tp1, tp2, sl = entry + atr, entry + 2 * atr, entry - 0.75 * atr
    else:
        tp1, tp2, sl = entry - atr, entry - 2 * atr, entry + 0.75 * atr
    now = datetime.now(timezone.utc)
    return {
        "symbol": symbol,
        "tf": "15M",
        "direction": direction,
        "entry": entry,
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "confidence": float(conf15),
        "score": float(score15),
        "ml": None if prob15 is None else float(prob15),
        "created": now.isoformat(),
        "until": (now + timedelta(minutes=_LIVE_LOCK_MINUTES)).isoformat(),
    }


def _locked_15m_signal(symbol):
    locks = _read_live_locks()
    now = datetime.now(timezone.utc)
    old = locks.get(symbol)
    if isinstance(old, dict):
        try:
            until = datetime.fromisoformat(old["until"])
            if now < until and old.get("direction") in ("LONG", "SHORT"):
                old["remaining"] = max(0, int((until - now).total_seconds()))
                return old
        except Exception:
            pass
        locks.pop(symbol, None)
    signal = _fresh_15m_signal(symbol)
    if signal is not None:
        locks[symbol] = signal
        _write_live_locks(locks)
    else:
        _write_live_locks(locks)
    return signal


# A small independent fragment keeps the original dashboard untouched while
# refreshing only this new live-trades section.
try:
    from streamlit.runtime.fragment import fragment as _fragment
except Exception:
    _fragment = None


def _render_live_trades():
    st.markdown('<div class="section">LIVE TRADES</div>', unsafe_allow_html=True)
    st.caption("15M LONG/SHORT signals only • each actionable signal stays locked for 20 minutes • no replacement signal during the lock")

    rows = []
    for symbol in SYMBOLS:
        try:
            s = _locked_15m_signal(symbol)
            if not s:
                continue
            until = datetime.fromisoformat(s["until"])
            remaining = max(0, int((until - datetime.now(timezone.utc)).total_seconds()))
            rows.append({
                "CRYPTO": s["symbol"],
                "TF": "15M",
                "SIGNAL": s["direction"],
                "ENTRY": fmt_price(s["entry"]),
                "TP1": fmt_price(s["tp1"]),
                "TP2": fmt_price(s["tp2"]),
                "SL": fmt_price(s["sl"]),
                "CONFIDENCE": f"{s['confidence']:.1f}%",
                "LOCK LEFT": f"{remaining//60:02d}:{remaining%60:02d}",
            })

    if not rows:
        st.info("No active 15M LONG/SHORT trade is currently locked.")
        return

    # Keep the visual language of the original dashboard; this is an additive table.
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.caption(f"Active locked trades: {len(rows)} • Last scan: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")


if _fragment is not None:
    @_fragment(run_every="5s")
    def _live_trades_fragment():
        _render_live_trades()
    _live_trades_fragment()
else:
    _render_live_trades()
