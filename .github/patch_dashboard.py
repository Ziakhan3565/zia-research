from pathlib import Path

p = Path("dashboard.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    'SIGNAL_VALIDITY_MINUTES = {"15M": 30, "1H": 120, "4H": 180}\n',
    'SIGNAL_VALIDITY_MINUTES = {"15M": 30, "1H": 120, "4H": 180}\n'
    'AUTO_SCAN_TFS = ("15M", "1H", "4H")\n'
    'AUTO_SCAN_INTERVAL_SECONDS = 10\n'
)

marker = '\n\n# ------------------------------------------------------------\n# STATE\n# ------------------------------------------------------------\n'
helpers = r'''


def latest_saved_signal(symbol, tf):
    """Return the latest saved LONG/SHORT signal for a symbol/timeframe."""
    hist = load_signal_history()
    if hist.empty:
        return None
    try:
        h = hist[(hist["symbol"].astype(str) == symbol) & (hist["timeframe"].astype(str) == tf)]
        if h.empty:
            return None
        row = h.iloc[-1]
        if row.get("signal") not in ("LONG", "SHORT"):
            return None
        ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return {"signal": str(row.get("signal")), "entry_price": num(row.get("price")),
                "started": ts.to_pydatetime(), "confidence": num(row.get("confidence"))}
    except Exception:
        return None


def live_signal_pnl(entry_price, current_price, signal):
    """Mark-to-market P&L percentage for an active LONG/SHORT signal."""
    entry = num(entry_price)
    current = num(current_price)
    if entry <= 0 or current <= 0 or signal not in ("LONG", "SHORT"):
        return 0.0
    raw = (current - entry) / entry * 100.0
    return raw if signal == "LONG" else -raw


def auto_scan_once(threshold=0.20):
    """Scan every tracked coin on 15M/1H/4H and auto-journal new signals."""
    rows = []
    for tf_key in AUTO_SCAN_TFS:
        for symbol_key in SYMBOLS:
            snap = scan_symbol(symbol_key, tf_key, threshold)
            active = recover_active_signal(symbol_key, tf_key)
            if active:
                signal = active["signal"]
                confidence = active["confidence"]
                started = active["started"]
                rec = latest_saved_signal(symbol_key, tf_key)
                entry_price = rec["entry_price"] if rec else snap["price"]
            elif snap["signal"] in ("LONG", "SHORT"):
                signal = snap["signal"]
                confidence = snap["confidence"]
                started = datetime.now(timezone.utc)
                save_signal(symbol_key, tf_key, snap["price"], signal, confidence, None,
                            {"obi_20": snap["obi20"], "obi_50": 0.0, "taker_flow_ratio": 0.0}, 0.0)
                rec = latest_saved_signal(symbol_key, tf_key)
                entry_price = rec["entry_price"] if rec else snap["price"]
            else:
                signal = "WAIT"
                confidence = snap["confidence"]
                started = None
                entry_price = 0.0
            pnl_pct = live_signal_pnl(entry_price, snap["price"], signal)
            validity = signal_validity_minutes(tf_key)
            remaining = 0
            if started and validity:
                remaining = max(0, int((started + timedelta(minutes=validity) - datetime.now(timezone.utc)).total_seconds()))
            rows.append({"symbol": symbol_key, "timeframe": tf_key, "price": snap["price"],
                         "signal": signal, "confidence": confidence, "entry_price": entry_price,
                         "pnl_pct": pnl_pct, "remaining_sec": remaining, "obi20": snap["obi20"]})
    return rows
'''
if marker not in s:
    raise SystemExit("STATE marker not found")
if "def latest_saved_signal(" not in s:
    s = s.replace(marker, helpers + marker, 1)

old = '    started = time.perf_counter()\n    df, source, cstat, _ = candles(symbol, TFS[tf], 650)'
new = '''    started = time.perf_counter()

    # Background multi-market scanner: every 10 seconds it checks ALL tracked
    # coins on ONLY 15M, 1H and 4H and automatically journals fresh signals.
    scan_clock = time.time()
    last_scan = st.session_state.get("auto_scan_last", 0.0)
    if scan_clock - last_scan >= AUTO_SCAN_INTERVAL_SECONDS:
        try:
            st.session_state.auto_scan_rows = auto_scan_once(threshold)
            st.session_state.auto_scan_last = scan_clock
        except Exception:
            st.session_state.auto_scan_rows = st.session_state.get("auto_scan_rows", [])
    auto_rows = st.session_state.get("auto_scan_rows", [])

    df, source, cstat, _ = candles(symbol, TFS[tf], 650)'''
if old not in s:
    raise SystemExit("live_engine marker not found")
s = s.replace(old, new, 1)

start = '''    with tabs[6]:
        st.markdown('<div class="panel"><b>MULTI-MARKET SCANNER</b>'
                    f'<div class="section-sub">Every tracked symbol scored on the {tf} timeframe, refreshed every few seconds • click a row\\'s symbol above to jump in</div>',
                    unsafe_allow_html=True)
        rows = [scan_symbol(s, tf, threshold) for s in SYMBOLS]
        rows.sort(key=lambda r: r["combined"], reverse=True)'''
replacement = '''    with tabs[6]:
        st.markdown('<div class="panel"><b>MULTI-MARKET SCANNER</b>'
                    '<div class="section-sub">Automatic all-coin scanner • ONLY 15M / 1H / 4H • new signals are saved automatically • live P&L</div>',
                    unsafe_allow_html=True)
        rows = [r for r in auto_rows if r["signal"] in ("LONG", "SHORT")]
        rows.sort(key=lambda r: r["pnl_pct"], reverse=True)
        longs = sum(1 for r in rows if r["signal"] == "LONG")
        shorts = sum(1 for r in rows if r["signal"] == "SHORT")'''
if start not in s:
    raise SystemExit("scanner marker not found")
s = s.replace(start, replacement, 1)

old_counts = '''        longs = sum(1 for r in rows if r["signal"] == "LONG")
        shorts = sum(1 for r in rows if r["signal"] == "SHORT")
        waits = sum(1 for r in rows if r["signal"] == "WAIT")
        cards([("LONG SIGNALS", str(longs), "bullish across scan", "good"),'''
new_counts = '''        cards([("LONG SIGNALS", str(longs), "bullish across all coins", "good"),'''
if old_counts in s:
    s = s.replace(old_counts, new_counts, 1)

old_row = '''                f'<div style="width:16%;font-weight:900">{r["symbol"]}</div>'
                f'<div style="width:16%">${r["price"]:,.4f}</div>'
                f'<div style="width:14%" class="{chg_cls}">{r["change"]:+.2f}%</div>'
                f'<div style="width:14%">OBI {r["obi20"]:+.3f}</div>'
                f'<div style="width:18%">conf {r["confidence"]:.1f}%</div>'
                f'<div style="width:12%;text-align:right"><span class="pill {pill_cls}">{r["signal"]}</span></div>'
'''
new_row = '''                f'<div style="width:13%;font-weight:900">{r["symbol"]}</div>'
                f'<div style="width:9%;font-weight:900">{r["timeframe"]}</div>'
                f'<div style="width:13%">${r["price"]:,.4f}</div>'
                f'<div style="width:13%" class="{"good" if r["pnl_pct"] >= 0 else "bad"}">{r["pnl_pct"]:+.2f}% P&L</div>'
                f'<div style="width:12%">OBI {r["obi20"]:+.3f}</div>'
                f'<div style="width:14%">conf {r["confidence"]:.1f}%</div>'
                f'<div style="width:13%;text-align:right"><span class="pill {pill_cls}">{r["signal"]}</span></div>'
'''
if old_row not in s:
    raise SystemExit("scanner row marker not found")
s = s.replace(old_row, new_row, 1)

# Make the Signals tab show only the automatically tracked timeframes.
s = s.replace(
    '        h = read_csv(SIGNAL_FILE)\n        t = read_csv(TRADE_FILE)',
    '        h = read_csv(SIGNAL_FILE)\n        if not h.empty and "timeframe" in h.columns:\n            h = h[h["timeframe"].astype(str).isin(AUTO_SCAN_TFS)]\n        t = read_csv(TRADE_FILE)',
    1
)

# Ensure the auto scanner list is signal-only and never references the old price-change field.
s = s.replace('        rows = auto_rows\n', '        rows = [r for r in auto_rows if r["signal"] in ("LONG", "SHORT")]\n', 1)
s = s.replace('            chg_cls = "good" if r["change"] >= 0 else "bad"\n', '            chg_cls = "good" if r["pnl_pct"] >= 0 else "bad"\n', 1)

p.write_text(s, encoding="utf-8")
