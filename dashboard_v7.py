"""ZIA Research Terminal v7 - hardened dashboard entrypoint.

Loads the existing dashboard source, hardens its ML feature contract,
replaces the misleading Taker-Flow-as-OFI calculation with a real
order-book delta OFI when history is available, and uses a conservative
probability-to-score conversion. The original dashboard UI is preserved.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = (ROOT / "dashboard.py").read_text(encoding="utf-8")

# Persistent order-book history for real OFI.
anchor = 'DATA = ["https://data-api.binance.vision"]\n'
inject = '''DATA = ["https://data-api.binance.vision"]\n\n# Hardened live-signal state.\n_ZIA_OB_HISTORY = {}\n\n'''
if anchor in source:
    source = source.replace(anchor, inject, 1)

# Replace the live feature builder while keeping the original 25 ML features.
start = source.index("def features(df, b, a):")
end = source.index("\n\n@st.cache_resource", start)
new_features = r'''def features(df, b, a, symbol=None):
    """Build the exact 25-feature model contract plus live OFI research data."""
    names = ["top20_bid_sum", "top20_ask_sum", "obi_5", "obi_10", "obi_20", "obi_50", "spread",
             "spread_pct", "bid_ask_ratio_20", "bid_ask_ratio_50", "top20_total_depth",
             "top50_total_depth", "taker_buy_volume", "taker_sell_volume", "taker_flow",
             "taker_flow_ratio", "price_return", "price_change", "sma_distance",
             "realized_volatility", "BOOK_IMB", "QUANT_IMPLY", "ADAPT_CONF", "BAYESIAN", "FOURIER_TREND"]
    f = {k: 0.0 for k in names}
    (o5,b5,a5),(o10,b10,a10),(o20,b20,a20),(o50,b50,a50) = [obi(b,a,k) for k in (5,10,20,50)]
    f.update(top20_bid_sum=b20, top20_ask_sum=a20, obi_5=o5, obi_10=o10, obi_20=o20, obi_50=o50,
             top20_total_depth=b20+a20, top50_total_depth=b50+a50)
    key = str(symbol or "__default__").upper()
    prev = _ZIA_OB_HISTORY.get(key)
    ofi = 0.0
    if prev is not None:
        ofi = (float(b20)-prev[0]) - (float(a20)-prev[1])
    _ZIA_OB_HISTORY[key] = (float(b20), float(a20))
    f["ofi"] = ofi
    f["ofi_norm"] = ofi / max(float(b20+a20), 1e-12)
    if df.empty:
        return f
    c = df.Close
    last = num(c.iloc[-1]); prev_close = num(c.iloc[-2] if len(c)>1 else last)
    sma = num(c.rolling(20).mean().iloc[-1], last)
    total = num(df.Volume.tail(20).sum())
    buy = num(df.TakerBuy.tail(20).sum()); sell = max(total-buy,0.0); flow=buy-sell
    spread = num(a[0,0]-b[0,0]) if len(a) and len(b) else 0.0
    trend = np.tanh((last/sma-1)*100) if sma else 0.0
    rv = num(c.pct_change().tail(30).std())
    four = np.tanh(c.pct_change().tail(16).mean()*1000)
    f.update(spread=spread, spread_pct=spread/last if last else 0.0,
             bid_ask_ratio_20=b20/a20 if a20 else 1.0, bid_ask_ratio_50=b50/a50 if a50 else 1.0,
             taker_buy_volume=buy, taker_sell_volume=sell, taker_flow=flow,
             taker_flow_ratio=flow/total if total else 0.0, price_return=last/prev_close-1 if prev_close else 0.0,
             price_change=last-prev_close, sma_distance=last/sma-1 if sma else 0.0, realized_volatility=rv,
             BOOK_IMB=o20, QUANT_IMPLY=float(np.tanh((o20+o50+trend)/3)),
             ADAPT_CONF=float(np.clip(.5+(abs(o20)+abs(trend))/2,0,1)),
             BAYESIAN=float(np.clip(.5+(o20+trend)/4,0,1)), FOURIER_TREND=float(four))
    return f
'''
source = source[:start] + new_features + source[end:]

# Strict ML feature validation: never silently replace missing model inputs with zero.
start = source.index("def ml_predict(f):")
end = source.index("\n\ndef research", start)
new_ml = r'''def ml_predict(f):
    m = load_model()
    if m is None:
        return None, None, "MODEL NOT FOUND", 0
    try:
        names = list(m.get_booster().feature_names or []) if hasattr(m, "get_booster") else []
        count = int(getattr(m, "n_features_in_", len(names)))
        expected = names or []
        if not expected and count == 25:
            expected = ["top20_bid_sum","top20_ask_sum","obi_5","obi_10","obi_20","obi_50","spread",
                        "spread_pct","bid_ask_ratio_20","bid_ask_ratio_50","top20_total_depth","top50_total_depth",
                        "taker_buy_volume","taker_sell_volume","taker_flow","taker_flow_ratio","price_return",
                        "price_change","sma_distance","realized_volatility","BOOK_IMB","QUANT_IMPLY","ADAPT_CONF","BAYESIAN","FOURIER_TREND"]
        if not expected or len(expected) != count:
            return None, None, f"ML SCHEMA ERROR: expected {count}, got {len(expected)}", 0
        missing = [c for c in expected if c not in f]
        if missing:
            return None, None, "ML SCHEMA ERROR: missing " + ", ".join(missing[:5]), 0
        x = pd.DataFrame([[f[c] for c in expected]], columns=expected)
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            return None, None, "ML INPUT ERROR: non-finite feature", 0
        pred = int(m.predict(x)[0])
        proba = float(m.predict_proba(x)[0][1]) if hasattr(m, "predict_proba") else None
        return pred, proba, "OK", len(expected)
    except Exception as e:
        return None, None, "ML ERROR: " + type(e).__name__, 0
'''
source = source[:start] + new_ml + source[end:]

# Conservative directional ML score; raw probability is displayed separately.
start = source.index("def final_state(f, p, pr, threshold=0.45):")
end = source.index("\n\ndef visible_tri_timeframes", start)
new_final = r'''def final_state(f, p, pr, threshold=0.45):
    scores, weights, rscore = research(f)
    if p is None:
        mlscore = 0.0
        combined = rscore
    else:
        mlscore = float(np.clip((float(pr)-0.5)*2.0, -1.0, 1.0)) if pr is not None else (1.0 if p == 1 else -1.0)
        combined = 0.6*rscore + 0.4*mlscore
    signal = "LONG" if combined >= threshold else "SHORT" if combined <= -threshold else "WAIT"
    confidence = float(np.clip(50.0 + abs(combined)*49.0, 1.0, 99.0))
    return signal, confidence, combined, scores, weights, rscore, mlscore
'''
source = source[:start] + new_final + source[end:]

# Use symbol-aware OFI history in both live and scanner paths.
source = source.replace('f = features(df, bids, asks)\n    pred, prob, _, _ = ml_predict(f)',
                        'f = features(df, bids, asks, symbol=symbol)\n    pred, prob, _, _ = ml_predict(f)')
source = source.replace('f = features(df, bids, asks)\n    pred, prob, mlstat, feature_count = ml_predict(f)',
                        'f = features(df, bids, asks, symbol=symbol)\n    pred, prob, mlstat, feature_count = ml_predict(f)')

# Research score now distinguishes true book OFI from candle taker flow.
source = source.replace('"OFI / Taker": np.clip(f["taker_flow_ratio"] * 2, -1, 1),',
                        '"OFI": np.clip(f.get("ofi_norm", 0.0) * 8, -1, 1),\n        "Taker Flow": np.clip(f["taker_flow_ratio"] * 2, -1, 1),')
source = source.replace('"OBI 20": .22, "OBI 20+50": .14, "OFI / Taker": .20, "Trend / SMA": .14,\n               "Fourier": .10, "Bayesian": .08, "Quant Imply": .07, "Adaptive": .05}',
                        '"OBI 20": .22, "OBI 20+50": .14, "OFI": .12, "Taker Flow": .08, "Trend / SMA": .14,\n               "Fourier": .10, "Bayesian": .08, "Quant Imply": .07, "Adaptive": .05}')

# Execute the hardened copy of the existing dashboard without changing its UI.
exec(compile(source, str(ROOT / "dashboard.py"), "exec"), {"__name__": "__main__", "__file__": str(ROOT / "dashboard.py")})
