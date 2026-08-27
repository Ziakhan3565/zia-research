from __future__ import annotations

import datetime as _dt
import time
from typing import Callable, Iterable

import pandas as pd
import streamlit as st


# This module is intentionally isolated: it does not change the existing
# dashboard layout, signal formula, ML model, risk engine, or chart.
# It only provides a small loop that can collect the existing dashboard's
# signal calculation for every configured coin and display actionable trades.


def scan_all_coins(
    coins: Iterable[str],
    signal_fn: Callable[[str], dict | None],
) -> list[dict]:
    """Run the existing signal function once for every coin.

    signal_fn must return the dashboard's normal signal dictionary.  No new
    signal logic is introduced here.
    """
    trades: list[dict] = []
    for symbol in coins:
        try:
            result = signal_fn(symbol)
        except Exception:
            result = None
        if not result:
            continue
        direction = str(result.get("direction", "NEUTRAL")).upper()
        if direction not in {"LONG", "SHORT"}:
            continue
        item = dict(result)
        item["scanned_at"] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trades.append(item)
    return trades


def render_live_trades(trades: list[dict]) -> None:
    """Render a compact live-trades section without altering other UI."""
    st.markdown("---")
    st.subheader("🔥 Live Trades — All Coins")
    st.caption("Auto-scans every configured cryptocurrency each refresh cycle. Existing signal logic is used unchanged.")

    if not trades:
        st.info("No active LONG/SHORT signal across the configured coins.")
        return

    rows = []
    for t in trades:
        rows.append({
            "Coin": t.get("symbol", "—"),
            "Timeframe": t.get("timeframe", "—"),
            "Signal": t.get("direction", "—"),
            "Entry": t.get("entry_price", "—"),
            "TP1": t.get("tp1", "—"),
            "TP2": t.get("tp2", "—"),
            "Stop Loss": t.get("stop_loss", "—"),
            "Confidence": t.get("confidence", "—"),
            "Score": t.get("final_score", "—"),
            "Status": "LIVE",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
