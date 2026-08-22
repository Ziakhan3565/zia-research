import os
import time
import math
import datetime as dt
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# Optional original research engine.
try:
    from engine import IntegratedTradingEngine
except Exception:
    IntegratedTradingEngine = None


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="ZIA Research Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

if st_autorefresh:
    st_autorefresh(interval=5000, limit=None, key="zia_refresh")


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = "xgboost_direction_model.pkl"
HISTORY_FILE = "trade_history.csv"

BINANCE_REST = "https://api.binance.com"
BINANCE_DATA = "https://data-api.binance.vision"

COINS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"
]

TIMEFRAMES = {
    "1m (Scalping)": "1m",
    "5m (Scalping)": "5m",
    "15m (Medium TF)": "15m",
    "30m (Medium TF)": "30m",
    "1h (Intraday)": "1h",
    "4h (Swing)": "4h",
}

MIN_CONFIDENCE = 70.0
MIN_OBI = 0.10
MIN_OFI = 0.02
MAX_SPREAD_PCT = 0.0015
ATR_PERIOD = 14
TP1_RR = 2.0
TP2_RR = 3.0


# ============================================================
# SESSION STATE
# ============================================================

if "paper_enabled" not in st.session_state:
    st.session_state.paper_enabled = False

if "trade_history" not in st.session_state:
    st.session_state.trade_history = []

if "last_trade_key" not in st.session_state:
    st.session_state.last_trade_key = None


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    [data-testid="stSidebar"] {
        min-width: 310px;
        max-width: 330px;
    }

    .statusbar {
        background: #101521;
        border: 1px solid #263147;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0 20px 0;
        font-size: 15px;
        font-weight: 600;
    }

    .panel {
        background: #101521;
        border: 1px solid #263147;
        border-radius: 14px;
        padding: 18px;
        min-height: 135px;
    }

    .panel-title {
        color: #8e9bb2;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .big-value {
        font-size: 26px;
        font-weight: 800;
    }

    .green { color: #00e676; }
    .red { color: #ff4d5f; }
    .blue { color: #36a8ff; }
    .yellow { color: #ffc857; }
    .muted { color: #8e9bb2; }

    .signal-long {
        border-left: 5px solid #00e676;
    }

    .signal-short {
        border-left: 5px solid #ff4d5f;
    }

    .signal-wait {
        border-left: 5px solid #ffc857;
    }

    .metric-card {
        background: #101521;
        border: 1px solid #263147;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .metric-label {
        color: #8e9bb2;
        font-size: 12px;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 20px;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return default


def fmt_price(price: float) -> str:
    price = safe_float(price)
    if price >= 1000:
        return f"${price:,.2f}"
    if price >= 1:
        return f"${price:,.4f}"
    return f"${price:,.6f}"


def signal_class(signal: str) -> str:
    if signal == "LONG":
        return "signal-long"
    if signal == "SHORT":
        return "signal-short"
    return "signal-wait"


def signal_color(signal: str) -> str:
    if signal == "LONG":
        return "#00e676"
    if signal == "SHORT":
        return "#ff4d5f"
    return "#ffc857"


def safe_div(a: float, b: float) -> float:
    return safe_float(a) / (safe_float(b) + 1e-12)


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=4, show_spinner=False)
def fetch_klines(symbol: str, interval: str, limit: int = 180) -> pd.DataFrame:
    url = (
        f"{BINANCE_DATA}/api/v3/klines"
        f"?symbol={symbol}&interval={interval}&limit={limit}"
    )

    r = requests.get(url, timeout=8)
    r.raise_for_status()
    raw = r.json()

    cols = [
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore"
    ]

    df = pd.DataFrame(raw, columns=cols)

    for col in ["Open", "High", "Low", "Close", "Volume",
                "quote_volume", "taker_base", "taker_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    return df[[
        "Time", "Open", "High", "Low", "Close", "Volume",
        "quote_volume", "taker_base", "taker_quote"
    ]].reset_index(drop=True)


@st.cache_data(ttl=2, show_spinner=False)
def fetch_order_book(symbol: str, limit: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    url = f"{BINANCE_DATA}/api/v3/depth?symbol={symbol}&limit={limit}"
    r = requests.get(url, timeout=8)
    r.raise_for_status()
    data = r.json()

    bids = np.asarray(data.get("bids", []), dtype=float)
    asks = np.asarray(data.get("asks", []), dtype=float)

    return bids, asks


# ============================================================
# FEATURES
# ============================================================

def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    prev = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev).abs(),
            (df["Low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]

    if not np.isfinite(atr) or atr <= 0:
        atr = df["Close"].iloc[-1] * 0.003

    return safe_float(atr)


def calculate_market_features(df: pd.DataFrame) -> Dict[str, float]:
    close = df["Close"]
    volume = df["Volume"]

    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    ret1 = close.pct_change(1).iloc[-1]
    ret3 = close.pct_change(3).iloc[-1]
    ret5 = close.pct_change(5).iloc[-1]

    volatility = close.pct_change().rolling(20).std().iloc[-1]

    vol_mean = volume.rolling(20).mean().iloc[-1]
    volume_ratio = safe_div(volume.iloc[-1], vol_mean)

    return {
        "close": safe_float(close.iloc[-1]),
        "ema10": safe_float(ema10.iloc[-1]),
        "ema20": safe_float(ema20.iloc[-1]),
        "ema50": safe_float(ema50.iloc[-1]),
        "trend10": safe_div(close.iloc[-1] - ema10.iloc[-1], ema10.iloc[-1]),
        "trend20": safe_div(close.iloc[-1] - ema20.iloc[-1], ema20.iloc[-1]),
        "trend50": safe_div(close.iloc[-1] - ema50.iloc[-1], ema50.iloc[-1]),
        "return1": safe_float(ret1),
        "return3": safe_float(ret3),
        "return5": safe_float(ret5),
        "volatility": safe_float(volatility),
        "volume_ratio": safe_float(volume_ratio),
        "atr": calculate_atr(df),
    }


def calculate_orderbook_features(
    bids: np.ndarray,
    asks: np.ndarray
) -> Dict[str, float]:

    if len(bids) < 20 or len(asks) < 20:
        raise ValueError("Order book has fewer than 20 levels.")

    def side_sum(arr: np.ndarray, n: int) -> float:
        return float(np.sum(arr[:n, 1]))

    bid20 = side_sum(bids, 20)
    ask20 = side_sum(asks, 20)
    bid50 = side_sum(bids, min(50, len(bids)))
    ask50 = side_sum(asks, min(50, len(asks)))

    obi20 = safe_div(bid20 - ask20, bid20 + ask20)
    obi50 = safe_div(bid50 - ask50, bid50 + ask50)

    best_bid = safe_float(bids[0, 0])
    best_ask = safe_float(asks[0, 0])
    mid = (best_bid + best_ask) / 2
    spread = max(0.0, best_ask - best_bid)
    spread_pct = safe_div(spread, mid)

    # Normalized OFI proxy from the current L2 snapshot.
    total20 = bid20 + ask20
    ofi = safe_div(bid20 - ask20, total20)

    return {
        "bid20": bid20,
        "ask20": ask20,
        "bid50": bid50,
        "ask50": ask50,
        "obi20": obi20,
        "obi50": obi50,
        "ofi": ofi,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "spread_pct": spread_pct,
        "bid_ask_ratio": safe_div(bid20, ask20),
    }


def build_feature_dict(
    market: Dict[str, float],
    ob: Dict[str, float]
) -> Dict[str, float]:

    # Aliases cover common names used by training scripts.
    return {
        **market,
        **ob,
        "obi_top20": ob["obi20"],
        "obi_top50": ob["obi50"],
        "ofi_normalized": ob["ofi"],
        "spread_pct": ob["spread_pct"],
        "bid_volume": ob["bid20"],
        "ask_volume": ob["ask20"],
        "bid_volume_top20": ob["bid20"],
        "ask_volume_top20": ob["ask20"],
        "bid_volume_top50": ob["bid50"],
        "ask_volume_top50": ob["ask50"],
    }


# ============================================================
# XGBOOST
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None

    try:
        return joblib.load(MODEL_FILE)
    except Exception:
        return None


def unpack_model(package: Any):
    if package is None:
        return None, []

    if isinstance(package, dict):
        model = (
            package.get("model")
            or package.get("classifier")
            or package.get("xgb_model")
        )
        features = (
            package.get("features")
            or package.get("feature_names")
            or package.get("columns")
            or []
        )
        return model, list(features)

    return package, []


def map_model_class(cls: Any) -> Optional[str]:
    text = str(cls).upper()

    if text in {"0", "NO_TRADE", "NO TRADE", "WAIT", "NONE"}:
        return "NO_TRADE"

    if text in {"1", "LONG", "BUY", "UP"}:
        return "LONG"

    if text in {"2", "SHORT", "SELL", "DOWN"}:
        return "SHORT"

    return None


def xgb_predict(features: Dict[str, float]) -> Optional[Dict[str, float]]:
    package = load_model()
    model, feature_names = unpack_model(package)

    if model is None:
        return None

    # If the saved model is a sklearn Pipeline, feature names may not
    # be stored. Try the model's feature_names_in_.
    if not feature_names:
        feature_names = list(
            getattr(model, "feature_names_in_", [])
        )

    if not feature_names:
        return None

    row = []

    for name in feature_names:
        value = features.get(name, 0.0)
        row.append(safe_float(value))

    X = pd.DataFrame([row], columns=feature_names)

    try:
        probabilities = model.predict_proba(X)[0]
        classes = list(model.classes_)
    except Exception:
        return None

    mapped = {
        "NO_TRADE": 0.0,
        "LONG": 0.0,
        "SHORT": 0.0,
    }

    for cls, prob in zip(classes, probabilities):
        direction = map_model_class(cls)
        if direction:
            mapped[direction] = safe_float(prob)

    direction = max(mapped, key=mapped.get)
    confidence = mapped[direction] * 100.0

    return {
        "direction": direction,
        "confidence": confidence,
        "p_long": mapped["LONG"] * 100.0,
        "p_short": mapped["SHORT"] * 100.0,
        "p_no_trade": mapped["NO_TRADE"] * 100.0,
    }


# ============================================================
# ORIGINAL ENGINE
# ============================================================

@st.cache_resource
def get_engine():
    if IntegratedTradingEngine is None:
        return None

    try:
        return IntegratedTradingEngine()
    except Exception:
        return None


def run_original_engine(
    market_features: Dict[str, float],
    ob_features: Dict[str, float]
) -> Dict[str, Any]:

    engine = get_engine()

    if engine is None:
        return {
            "signal": "WAIT",
            "score": 0.0,
            "confidence": 0.0,
            "risk": "UNKNOWN",
            "features": {},
        }

    # Keep this adapter defensive because different versions of the
    # user's engine may expose different analyze() signatures.
    payload = {
        **market_features,
        **ob_features,
    }

    try:
        result = engine.analyze(payload)
    except Exception:
        try:
            result = engine.analyze(
                market_features,
                ob_features
            )
        except Exception:
            return {
                "signal": "WAIT",
                "score": 0.0,
                "confidence": 0.0,
                "risk": "UNKNOWN",
                "features": {},
            }

    if not isinstance(result, dict):
        return {
            "signal": "WAIT",
            "score": 0.0,
            "confidence": 0.0,
            "risk": "UNKNOWN",
            "features": {},
        }

    signal = str(
        result.get(
            "SIGNAL",
            result.get("signal", "WAIT")
        )
    ).upper()

    if signal not in {"LONG", "SHORT", "WAIT", "NO TRADE"}:
        signal = "WAIT"

    return {
        "signal": signal,
        "score": safe_float(
            result.get(
                "SCORE",
                result.get("score", 0)
            )
        ),
        "confidence": safe_float(
            result.get(
                "CONFIDENCE",
                result.get("confidence", 0)
            )
        ),
        "risk": str(
            result.get(
                "RISK",
                result.get("risk", "UNKNOWN")
            )
        ),
        "features": result.get("FEATURES", {}),
    }


# ============================================================
# SIGNAL FUSION
# ============================================================

def fuse_signal(
    xgb: Optional[Dict[str, float]],
    engine_result: Dict[str, Any],
    ob: Dict[str, float],
    market: Dict[str, float]
) -> Dict[str, Any]:

    xgb_signal = (
        xgb["direction"]
        if xgb
        else "NO_TRADE"
    )

    xgb_conf = (
        xgb["confidence"]
        if xgb
        else 0.0
    )

    engine_signal = engine_result["signal"]

    long_confirm = 0
    short_confirm = 0

    if ob["obi20"] >= MIN_OBI:
        long_confirm += 1
    if ob["obi50"] >= MIN_OBI:
        long_confirm += 1
    if ob["ofi"] >= MIN_OFI:
        long_confirm += 1
    if market["trend20"] > 0:
        long_confirm += 1
    if market["trend50"] > 0:
        long_confirm += 1

    if ob["obi20"] <= -MIN_OBI:
        short_confirm += 1
    if ob["obi50"] <= -MIN_OBI:
        short_confirm += 1
    if ob["ofi"] <= -MIN_OFI:
        short_confirm += 1
    if market["trend20"] < 0:
        short_confirm += 1
    if market["trend50"] < 0:
        short_confirm += 1

    # If XGBoost exists, it is the directional classifier.
    # Existing engine remains a confirmation/risk layer.
    if xgb is not None:
        if xgb_signal == "LONG":
            if (
                xgb_conf >= MIN_CONFIDENCE
                and long_confirm >= 3
                and ob["spread_pct"] <= MAX_SPREAD_PCT
            ):
                final = "LONG"
            else:
                final = "WAIT"

        elif xgb_signal == "SHORT":
            if (
                xgb_conf >= MIN_CONFIDENCE
                and short_confirm >= 3
                and ob["spread_pct"] <= MAX_SPREAD_PCT
            ):
                final = "SHORT"
            else:
                final = "WAIT"

        else:
            final = "WAIT"

        confidence = xgb_conf

    else:
        # Graceful fallback to the existing engine.
        final = (
            engine_signal
            if engine_signal in {"LONG", "SHORT"}
            else "WAIT"
        )
        confidence = engine_result["confidence"]

    if final == "LONG":
        score = (
            0.40 * min(1.0, max(0.0, xgb_conf / 100.0))
            + 0.20 * min(1.0, max(0.0, long_confirm / 5.0))
            + 0.20 * min(1.0, max(0.0, (ob["obi20"] + 1) / 2))
            + 0.20 * min(1.0, max(0.0, (ob["obi50"] + 1) / 2))
        )

    elif final == "SHORT":
        score = (
            0.40 * min(1.0, max(0.0, xgb_conf / 100.0))
            + 0.20 * min(1.0, max(0.0, short_confirm / 5.0))
            + 0.20 * min(1.0, max(0.0, (1 - ob["obi20"]) / 2))
            + 0.20 * min(1.0, max(0.0, (1 - ob["obi50"]) / 2))
        )

    else:
        score = 0.0

    return {
        "signal": final,
        "score": safe_float(score),
        "confidence": safe_float(confidence),
        "long_confirm": long_confirm,
        "short_confirm": short_confirm,
        "xgb_signal": xgb_signal,
        "engine_signal": engine_signal,
    }


# ============================================================
# RISK
# ============================================================

def calculate_risk(
    df: pd.DataFrame,
    ob: Dict[str, float],
    market: Dict[str, float],
    signal: str
) -> Dict[str, float]:

    price = market["close"]
    atr = market["atr"]

    spread_risk = min(
        100.0,
        safe_div(
            ob["spread_pct"],
            MAX_SPREAD_PCT
        ) * 100
    )

    volatility_pct = market["volatility"] * 10000
    volatility_risk = min(
        100.0,
        volatility_pct * 5
    )

    imbalance_risk = abs(ob["obi20"]) * 100

    # Extreme values are intentionally capped so a bad denominator
    # cannot produce millions in the UI.
    squeeze_risk = min(
        100.0,
        volatility_risk * 0.45
        + abs(ob["ofi"]) * 100 * 0.35
        + spread_risk * 0.20
    )

    ltz = min(
        100.0,
        100.0 - abs(ob["obi50"]) * 45.0
        + volatility_risk * 0.25
    )

    if squeeze_risk >= 75:
        risk_status = "HIGH"
    elif squeeze_risk >= 50:
        risk_status = "MEDIUM"
    else:
        risk_status = "LOW-MEDIUM"

    return {
        "atr": atr,
        "spread_risk": spread_risk,
        "volatility_risk": volatility_risk,
        "imbalance_risk": imbalance_risk,
        "squeeze_risk": squeeze_risk,
        "ltz": max(0.0, min(100.0, ltz)),
        "spoof_score": min(
            100.0,
            abs(ob["obi20"] - ob["obi50"]) * 100
        ),
        "market_risk": min(
            100.0,
            squeeze_risk * 0.7
            + spread_risk * 0.3
        ),
        "status": risk_status,
    }


# ============================================================
# TARGETS
# ============================================================

def calculate_targets(
    price: float,
    atr: float,
    signal: str
) -> Dict[str, float]:

    risk_distance = max(
        atr,
        price * 0.0005
    )

    if signal == "LONG":
        sl = price - risk_distance
        tp1 = price + risk_distance * TP1_RR
        tp2 = price + risk_distance * TP2_RR

    elif signal == "SHORT":
        sl = price + risk_distance
        tp1 = price - risk_distance * TP1_RR
        tp2 = price - risk_distance * TP2_RR

    else:
        sl = price
        tp1 = price
        tp2 = price

    return {
        "entry": price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "beam": tp2,
        "base": tp1,
        "rr": TP1_RR,
    }


# ============================================================
# HISTORY
# ============================================================

HISTORY_COLUMNS = [
    "timestamp", "symbol", "timeframe", "direction",
    "entry_price", "stop_loss", "tp1", "tp2",
    "exit_price", "confidence", "outcome",
    "pnl_percent", "status"
]


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        df = pd.read_csv(HISTORY_FILE)
        return df.to_dict("records")
    except Exception:
        return []


def save_history(history: list) -> None:
    if not history:
        return

    try:
        df = pd.DataFrame(history)
        for col in HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        df[HISTORY_COLUMNS].to_csv(
            HISTORY_FILE,
            index=False
        )
    except Exception:
        pass


if not st.session_state.trade_history:
    st.session_state.trade_history = load_history()


def update_open_trades(
    current_high: float,
    current_low: float,
    symbol: str,
    timeframe: str
) -> None:

    changed = False

    for trade in st.session_state.trade_history:
        if trade.get("status") != "Open":
            continue
        if trade.get("symbol") != symbol:
            continue
        if trade.get("timeframe") != timeframe:
            continue

        direction = trade.get("direction")
        entry = safe_float(trade.get("entry_price"))
        sl = safe_float(trade.get("stop_loss"))
        tp1 = safe_float(trade.get("tp1"))

        if direction == "LONG":
            sl_hit = current_low <= sl
            tp_hit = current_high >= tp1

            if sl_hit and tp_hit:
                outcome = "LOSS"
                exit_price = sl
            elif tp_hit:
                outcome = "WIN"
                exit_price = tp1
            elif sl_hit:
                outcome = "LOSS"
                exit_price = sl
            else:
                continue

            pnl = safe_div(
                exit_price - entry,
                entry
            ) * 100

        elif direction == "SHORT":
            sl_hit = current_high >= sl
            tp_hit = current_low <= tp1

            if sl_hit and tp_hit:
                outcome = "LOSS"
                exit_price = sl
            elif tp_hit:
                outcome = "WIN"
                exit_price = tp1
            elif sl_hit:
                outcome = "LOSS"
                exit_price = sl
            else:
                continue

            pnl = safe_div(
                entry - exit_price,
                entry
            ) * 100

        else:
            continue

        trade["exit_price"] = exit_price
        trade["outcome"] = outcome
        trade["pnl_percent"] = pnl
        trade["status"] = "Closed"
        changed = True

    if changed:
        save_history(st.session_state.trade_history)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚡ ZIA Research Controls")

    symbol = st.selectbox(
        "Select Cryptocurrency",
        COINS,
        index=COINS.index("BTCUSDT")
    )

    timeframe_label = st.selectbox(
        "Select Timeframe",
        list(TIMEFRAMES.keys()),
        index=2
    )

    timeframe = TIMEFRAMES[timeframe_label]

    st.markdown("### Forecast Horizon Candles")

    horizon = st.slider(
        "",
        min_value=5,
        max_value=50,
        value=15,
        step=1,
        label_visibility="collapsed"
    )

    st.caption(f"Forecast horizon: **{horizon} candles**")

    st.markdown("---")

    st.markdown("### 🎲 Paper Trading Mode")

    st.session_state.paper_enabled = st.toggle(
        "Enable Live Paper Trading",
        value=st.session_state.paper_enabled
    )

    if st.button(
        "🗑️ Clear Trade History Log",
        use_container_width=True
    ):
        st.session_state.trade_history = []
        try:
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
        except Exception:
            pass
        st.rerun()

    st.markdown("---")

    st.markdown("### Model / Filters")
    st.caption(f"XGBoost: `{MODEL_FILE}`")
    st.caption(f"Minimum confidence: **{MIN_CONFIDENCE:.0f}%**")
    st.caption(f"Minimum OBI: **{MIN_OBI:.2f}**")
    st.caption(f"Minimum OFI: **{MIN_OFI:.2f}**")


# ============================================================
# LOAD MARKET
# ============================================================

try:
    df = fetch_klines(symbol, timeframe, 180)
    bids, asks = fetch_order_book(symbol, 100)
except Exception as exc:
    st.error(f"Market data error: {exc}")
    st.stop()

if df.empty or len(df) < 60:
    st.error("Not enough market candle data.")
    st.stop()

if len(bids) < 20 or len(asks) < 20:
    st.error("Level-2 order book is unavailable.")
    st.stop()


# ============================================================
# COMPUTE
# ============================================================

market = calculate_market_features(df)
ob = calculate_orderbook_features(bids, asks)
features = build_feature_dict(market, ob)

xgb = xgb_predict(features)
engine_result = run_original_engine(market, ob)

fusion = fuse_signal(
    xgb,
    engine_result,
    ob,
    market
)

signal = fusion["signal"]

risk = calculate_risk(
    df,
    ob,
    market,
    signal
)

targets = calculate_targets(
    market["close"],
    market["atr"],
    signal
)

update_open_trades(
    safe_float(df["High"].iloc[-1]),
    safe_float(df["Low"].iloc[-1]),
    symbol,
    timeframe
)


# ============================================================
# STATUS BAR
# ============================================================

if signal == "LONG":
    dot = "🟢"
elif signal == "SHORT":
    dot = "🔴"
else:
    dot = "🟡"

st.markdown(
    f"""
    <div class="statusbar">
        {dot} [{symbol}] |
        Price: {fmt_price(market["close"])} |
        TF: {timeframe_label} |
        SIGNAL: <span style="color:{signal_color(signal)}">{signal}</span> |
        Score: {fusion["score"]:+.3f} |
        Confidence: {fusion["confidence"]:.0f}% |
        ⏳ Auto Refresh: 5s
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"# ⚡ ZIA Quantitative Research Terminal"
)

st.caption(
    "Research Engine + XGBoost + OBI + OFI + Trend Confirmation + Paper Trading"
)


# ============================================================
# EXECUTION PANEL
# ============================================================

st.markdown("## 🎯 Signal Execution Panel")

signal_cls = signal_class(signal)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""
        <div class="panel {signal_cls}">
            <div class="panel-title">Signal Execution Panel</div>
            <div class="big-value" style="color:{signal_color(signal)}">
                {signal}
            </div>
            <div class="muted">
                Entry: {fmt_price(targets["entry"])} |
                SL: {fmt_price(targets["sl"])}
            </div>
            <div class="muted">
                TP1: {fmt_price(targets["tp1"])} |
                TP2: {fmt_price(targets["tp2"])}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Beam Target</div>
            <div class="big-value blue">{fmt_price(targets["beam"])}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Risk / Reward</div>
            <div class="big-value">1 : {targets["rr"]:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">LTZ Score</div>
            <div class="big-value">{risk["ltz"]:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Squeeze Risk</div>
            <div class="big-value red">{risk["squeeze_risk"]:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">BASE TARGET</div>
            <div class="metric-value red">{fmt_price(targets["base"])}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    strength = (
        "HIGH" if fusion["score"] >= 0.70
        else "MEDIUM" if fusion["score"] >= 0.50
        else "LOW"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">SIGNAL STRENGTH</div>
            <div class="metric-value green">{strength}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">MARKET RISK</div>
            <div class="metric-value">{risk["market_risk"]:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHART + MICROSTRUCTURE
# ============================================================

left, right = st.columns([2.2, 1])

with left:
    st.markdown(
        f"## 📈 Price Trajectory & Levels ({symbol})"
    )

    chart_df = df.tail(80)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df["Time"],
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Candles",
        )
    )

    if signal in {"LONG", "SHORT"}:
        future_x = pd.date_range(
            start=chart_df["Time"].iloc[-1],
            periods=max(3, horizon),
            freq=pd.Timedelta(
                minutes={
                    "1m": 1,
                    "5m": 5,
                    "15m": 15,
                    "30m": 30,
                    "1h": 60,
                    "4h": 240,
                }.get(timeframe, 15)
            )
        )

        start = market["close"]

        if signal == "LONG":
            end = targets["beam"]
        else:
            end = targets["beam"]

        trajectory = np.linspace(
            start,
            end,
            len(future_x)
        )

        fig.add_trace(
            go.Scatter(
                x=future_x,
                y=trajectory,
                mode="lines+markers",
                name="Trajectory",
                line=dict(
                    color=signal_color(signal),
                    width=3,
                    dash="dot",
                ),
            )
        )

        fig.add_hline(
            y=targets["sl"],
            line_dash="dash",
            line_color="#ff4d5f",
            annotation_text="SL"
        )

        fig.add_hline(
            y=targets["base"],
            line_dash="dot",
            line_color="#ff4d5f",
            annotation_text="BASE"
        )

        fig.add_hline(
            y=targets["beam"],
            line_dash="dash",
            line_color="#00e676",
            annotation_text="BEAM"
        )

    fig.update_layout(
        height=510,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#101521",
        plot_bgcolor="#101521",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:
    st.markdown("## 📚 Market Microstructure & OB")

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Bid Volume</div>
            <div class="metric-value green">{ob["bid20"]:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Ask Volume</div>
            <div class="metric-value red">{ob["ask20"]:,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Order Book Imbalance (OBI)</div>
            <div class="metric-value">{ob["obi20"]:+.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Spread</div>
            <div class="metric-value">{fmt_price(ob["spread"])}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Risk Status</div>
            <div class="metric-value green">{risk["status"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TOP 20 / TOP 50
# ============================================================

st.markdown("## 📊 Top 20 OBI Analysis")

levels = []

for i in range(min(20, len(bids), len(asks))):
    bid_price = safe_float(bids[i, 0])
    bid_qty = safe_float(bids[i, 1])
    ask_price = safe_float(asks[i, 0])
    ask_qty = safe_float(asks[i, 1])

    level_obi = safe_div(
        bid_qty - ask_qty,
        bid_qty + ask_qty
    )

    levels.append({
        "Level": i + 1,
        "Bid Price": bid_price,
        "Bid Qty": bid_qty,
        "Ask Price": ask_price,
        "Ask Qty": ask_qty,
        "Level OBI": level_obi,
    })

level_df = pd.DataFrame(levels)

st.dataframe(
    level_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SIGNAL DIAGNOSTICS
# ============================================================

st.markdown("## 🧠 Signal Diagnostics")

d1, d2, d3, d4, d5, d6 = st.columns(6)

d1.metric(
    "XGB Signal",
    xgb["direction"] if xgb else "N/A"
)

d2.metric(
    "XGB Confidence",
    f"{xgb['confidence']:.1f}%" if xgb else "N/A"
)

d3.metric(
    "OBI Top20",
    f"{ob['obi20']:+.3f}"
)

d4.metric(
    "OBI Top50",
    f"{ob['obi50']:+.3f}"
)

d5.metric(
    "OFI",
    f"{ob['ofi']:+.3f}"
)

d6.metric(
    "Engine",
    engine_result["signal"]
)


st.caption(
    f"LONG confirmations: {fusion['long_confirm']}/5 | "
    f"SHORT confirmations: {fusion['short_confirm']}/5 | "
    f"Spread: {ob['spread_pct'] * 100:.5f}%"
)


# ============================================================
# PAPER TRADE
# ============================================================

if (
    st.session_state.paper_enabled
    and signal in {"LONG", "SHORT"}
):

    candle_bucket = int(time.time() // 5)

    trade_key = (
        f"{symbol}|{timeframe}|{signal}|{candle_bucket}"
    )

    if trade_key != st.session_state.last_trade_key:

        trade = {
            "timestamp": dt.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": signal,
            "entry_price": targets["entry"],
            "stop_loss": targets["sl"],
            "tp1": targets["tp1"],
            "tp2": targets["tp2"],
            "exit_price": targets["entry"],
            "confidence": fusion["confidence"],
            "outcome": "PENDING",
            "pnl_percent": 0.0,
            "status": "Open",
        }

        st.session_state.trade_history.insert(
            0,
            trade
        )

        save_history(
            st.session_state.trade_history
        )

        st.session_state.last_trade_key = trade_key


# ============================================================
# PERFORMANCE
# ============================================================

st.markdown("---")
st.markdown("## 📊 Performance & Trade History")

history_df = pd.DataFrame(
    st.session_state.trade_history
)

if history_df.empty:
    st.info("No paper trades yet.")
else:
    closed = history_df[
        history_df["status"] == "Closed"
    ].copy()

    pending = history_df[
        history_df["status"] == "Open"
    ].copy()

    wins = closed[
        closed["outcome"] == "WIN"
    ]

    losses = closed[
        closed["outcome"] == "LOSS"
    ]

    closed_count = len(closed)

    win_rate = (
        len(wins) / closed_count * 100
        if closed_count
        else 0
    )

    gross_profit = (
        wins["pnl_percent"].sum()
        if not wins.empty
        else 0
    )

    gross_loss = abs(
        losses["pnl_percent"].sum()
    ) if not losses.empty else 0

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else 0
    )

    net_pnl = (
        closed["pnl_percent"].sum()
        if not closed.empty
        else 0
    )

    p1, p2, p3, p4, p5, p6 = st.columns(6)

    p1.metric(
        "WIN RATE",
        f"{win_rate:.1f}%"
    )

    p2.metric(
        "CLOSED",
        closed_count
    )

    p3.metric(
        "WINS / LOSSES",
        f"{len(wins)}W / {len(losses)}L"
    )

    p4.metric(
        "PENDING",
        len(pending)
    )

    p5.metric(
        "PROFIT FACTOR",
        f"{profit_factor:.2f}"
    )

    p6.metric(
        "NET PNL",
        f"{net_pnl:.2f}%"
    )

    st.markdown("### 📋 Detailed Trade History")

    show_cols = [
        "timestamp",
        "symbol",
        "timeframe",
        "direction",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "exit_price",
        "pnl_percent",
        "outcome",
        "confidence",
        "status",
    ]

    show_cols = [
        c for c in show_cols
        if c in history_df.columns
    ]

    st.dataframe(
        history_df[show_cols],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL STATUS
# ============================================================

st.markdown("---")

if load_model() is not None:
    st.success(
        f"XGBoost model loaded: `{MODEL_FILE}`"
    )
else:
    st.warning(
        f"`{MODEL_FILE}` is not available. "
        "The original Research Engine remains available as fallback; "
        "run train_model.py to enable XGBoost confirmation."
    )

st.caption(
    "ZIA Research Terminal — paper trading only. "
    "Signals are research outputs, not guaranteed trading results."
)
