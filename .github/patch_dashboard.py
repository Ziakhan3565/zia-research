from pathlib import Path

p = Path("dashboard.py")
s = p.read_text(encoding="utf-8")

# The background scanner is already installed. Make its visible list contain
# only real LONG/SHORT signals and use the live P&L field for styling.
s = s.replace(
    '        rows = auto_rows\n        longs = sum(1 for r in rows if r["signal"] == "LONG")',
    '        rows = [r for r in auto_rows if r["signal"] in ("LONG", "SHORT")]\n        longs = sum(1 for r in rows if r["signal"] == "LONG")',
    1,
)
s = s.replace(
    '        waits = sum(1 for r in rows if r["signal"] == "WAIT")\n',
    '',
    1,
)
s = s.replace(
    '               ("WAITING", str(waits), "no clear edge", "amber"),\n',
    '',
    1,
)
s = s.replace(
    '            chg_cls = "good" if r["change"] >= 0 else "bad"\n',
    '            chg_cls = "good" if r["pnl_pct"] >= 0 else "bad"\n',
    1,
)
s = s.replace(
    '        rows = [r for r in auto_rows if r["signal"] in ("LONG", "SHORT")]\n        rows.sort(key=lambda r: r["pnl_pct"], reverse=True)\n',
    '        rows = [r for r in auto_rows if r["signal"] in ("LONG", "SHORT")]\n        rows.sort(key=lambda r: r["pnl_pct"], reverse=True)\n',
    1,
)

p.write_text(s, encoding="utf-8")
