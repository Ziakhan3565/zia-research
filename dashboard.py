from __future__ import annotations

import os
import time
import math
import joblib
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

from streamlit_autorefresh import st_autorefresh


# ============================================================
# ZIA RESEARCH LAB
# BINANCE USDⓈ-M FUTURES RESEARCH DASHBOARD
# ============================================================

st.set_page_config(
    page_title="ZIA RESEARCH LAB",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

HISTORY_FILE = ROOT / "signal_history.csv"
MODEL_FILE = ROOT / "xgboost_obi_model.pkl"

# IMPORTANT:
# Futures endpoints — NOT Spot endpoints
BINANCE_BASE = "https://fapi.binance.com"

BINANCE_KLINES = f"{BINANCE_BASE}/fapi/v1/klines"
BINANCE_DEPTH = f"{BINANCE_BASE}/fapi/v1/depth"
BINANCE_TICKER = f"{BINANCE_BASE}/fapi/v1/ticker/24hr"
BINANCE_AGG_TRADES = f"{BINANCE_BASE}/fapi/v1/aggTrades"

REQUEST_TIMEOUT = 8


# ============================================================
# COINS
# ============================================================

COINS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "XLMUSDT",
    "NEARUSDT",
    "UNIUSDT",
    "APTUSDT",
    "TAOUSDT",
    "XMRUSDT",
]


# ============================================================
# TRADE MODES
# ============================================================

TRADE_MODES = {
    "SCALPING": {
        "label": "30M SCALPING",
        "analysis_tf": "30m",
        "duration_minutes": 15,
        "max_holding": "15 minutes",
        "reference": ["1h", "4h"],
    },

    "15M": {
        "label": "15M",
        "analysis_tf": "15m",
        "duration_minutes": 90,
        "max_holding": "90 minutes",
        "reference": ["1h", "4h"],
    },

    "1H": {
        "label": "1H",
        "analysis_tf": "1h",
        "duration_minutes": 1440,
        "max_holding": "24 hours",
        "reference": ["1d", "1w"],
    },

    "4H": {
        "label": "4H",
        "analysis_tf": "4h",
        "duration_minutes": 1440,
        "max_holding": "24 hours max",
        "reference": ["1w", "1M"],
    },
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "selected_symbol": "BTCUSDT",
    "selected_mode": "SCALPING",
    "auto_scan": True,
    "show_chart": True,
    "last_signal": {},
    "signal_history": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

[data-testid="stSidebar"] {
    background: #080c12;
}

[data-testid="stSidebar"] * {
    color: #e8edf5;
}

.main-title {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.sub-title {
    color: #8f9bad;
    font-size: 14px;
    margin-top: -8px;
}

.signal-card {
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(
        135deg,
        #101722,
        #0b1018
    );
    border: 1px solid #1f2b3a;
    margin-bottom: 18px;
}

.signal-long {
    border: 1px solid #1f9d68;
    box-shadow: 0 0 25px rgba(31,157,104,.10);
}

.signal-short {
    border: 1px solid #d64a5c;
    box-shadow: 0 0 25px rgba(214,74,92,.10);
}

.signal-wait {
    border: 1px solid #526070;
}

.signal-title {
    font-size: 14px;
    color: #8995a6;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.signal-value {
    font-size: 38px;
    font-weight: 900;
    margin-top: 5px;
}

.price-value {
    font-size: 27px;
    font-weight: 800;
}

.metric-card {
    padding: 18px;
    border-radius: 12px;
    background: #0d131c;
    border: 1px solid #1c2735;
}

.metric-label {
    color: #8d99aa;
    font-size: 12px;
    text-transform: uppercase;
}

.metric-value {
    font-size: 21px;
    font-weight: 800;
    margin-top: 4px;
}

.section-title {
    font-size: 18px;
    font-weight: 800;
    margin-top: 12px;
    margin-bottom: 12px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background: #0c141e;
    border: 1px solid #203044;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass

    return default


def clamp(value, low=-1.0, high=1.0):
    return float(np.clip(safe_float(value), low, high))


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def iso_now():
    return now_utc().isoformat()


def fmt_price(value):
    value = safe_float(value)

    if value <= 0:
        return "—"

    if value >= 1000:
        return f"{value:,.2f}"

    if value >= 1:
        return f"{value:,.4f}"

    return f"{value:.6f}"


def fmt_pct(value):
    return f"{safe_float(value):.2f}%"


def normalize_confidence(value):
    value = safe_float(value)

    if value <= 1.0:
        value *= 100.0

    return float(np.clip(value, 0, 100))


def direction_class(direction):
    direction = str(direction).upper()

    if "LONG" in direction:
        return "signal-long"

    if "SHORT" in direction:
        return "signal-short"

    return "signal-wait"


# ============================================================
# HTTP
# ============================================================

def binance_get(endpoint, params=None):

    try:
        response = requests.get(
            endpoint,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": "ZIA-RESEARCH-LAB/2.0"
            },
        )

        response.raise_for_status()

        data = response.json()

        return data

    except Exception:
        return None


# ============================================================
# FUTURES KLINES
# ============================================================

@st.cache_data(
    ttl=8,
    show_spinner=False,
)
def fetch_klines(
    symbol: str,
    interval: str,
    limit: int = 250,
):

    data = binance_get(
        BINANCE_KLINES,
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
    )

    if not isinstance(data, list):
        return pd.DataFrame()

    rows = []

    for candle in data:

        if len(candle) < 12:
            continue

        try:

            rows.append(
                {
                    "Time": pd.to_datetime(
                        int(candle[0]),
                        unit="ms",
                        utc=True,
                    ),
                    "Open": float(candle[1]),
                    "High": float(candle[2]),
                    "Low": float(candle[3]),
                    "Close": float(candle[4]),
                    "Volume": float(candle[5]),
                    "Trades": int(candle[8]),
                    "Taker_Buy_Base": float(candle[9]),
                    "Taker_Buy_Quote": float(candle[10]),
                }
            )

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    return df.dropna().reset_index(drop=True)


# ============================================================
# FUTURES ORDER BOOK
# ============================================================

@st.cache_data(
    ttl=2,
    show_spinner=False,
)
def fetch_orderbook(
    symbol: str,
    limit: int = 100,
):

    data = binance_get(
        BINANCE_DEPTH,
        {
            "symbol": symbol,
            "limit": limit,
        },
    )

    if not isinstance(data, dict):
        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
        )

    try:

        bids = np.asarray(
            data.get("bids", []),
            dtype=np.float64,
        )

        asks = np.asarray(
            data.get("asks", []),
            dtype=np.float64,
        )

        if bids.ndim != 2:
            bids = np.empty((0, 2))

        if asks.ndim != 2:
            asks = np.empty((0, 2))

        return bids, asks

    except Exception:

        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
        )


# ============================================================
# FUTURES TICKER
# ============================================================

@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def fetch_ticker(symbol):

    data = binance_get(
        BINANCE_TICKER,
        {
            "symbol": symbol,
        },
    )

    return data if isinstance(data, dict) else {}


# ============================================================
# FUTURES AGG TRADES
# ============================================================

@st.cache_data(
    ttl=3,
    show_spinner=False,
)
def fetch_agg_trades(
    symbol,
    limit=1000,
):

    data = binance_get(
        BINANCE_AGG_TRADES,
        {
            "symbol": symbol,
            "limit": limit,
        },
    )

    return data if isinstance(data, list) else []


# ============================================================
# OBI
# ============================================================

def calculate_obi(
    bids,
    asks,
    levels=20,
):

    if len(bids) == 0 or len(asks) == 0:
        return 0.0

    n = min(
        levels,
        len(bids),
        len(asks),
    )

    bid_volume = np.sum(
        np.maximum(
            bids[:n, 1],
            0,
        )
    )

    ask_volume = np.sum(
        np.maximum(
            asks[:n, 1],
            0,
        )
    )

    total = bid_volume + ask_volume

    if total <= 0:
        return 0.0

    return clamp(
        (bid_volume - ask_volume)
        / total
    )


def calculate_weighted_obi(
    bids,
    asks,
    levels=20,
):

    if len(bids) == 0 or len(asks) == 0:
        return 0.0

    n = min(
        levels,
        len(bids),
        len(asks),
    )

    weights = 1.0 / (
        np.arange(n) + 1.0
    )

    bid_volume = np.sum(
        bids[:n, 1] * weights
    )

    ask_volume = np.sum(
        asks[:n, 1] * weights
    )

    total = bid_volume + ask_volume

    if total <= 0:
        return 0.0

    return clamp(
        (bid_volume - ask_volume)
        / total
    )


def calculate_multi_obi(
    bids,
    asks,
):

    values = {
        5: calculate_obi(bids, asks, 5),
        10: calculate_obi(bids, asks, 10),
        20: calculate_obi(bids, asks, 20),
        50: calculate_obi(bids, asks, 50),
    }

    # Stronger weight on top 20 / 50
    score = (
        values[5] * 0.15
        + values[10] * 0.20
        + values[20] * 0.35
        + values[50] * 0.30
    )

    return clamp(score)


# ============================================================
# DEPTH
# ============================================================

def depth_volume(
    bids,
    asks,
    levels,
):

    if len(bids) < levels or len(asks) < levels:
        return 0.0, 0.0

    bid_sum = float(
        np.sum(
            bids[:levels, 1]
        )
    )

    ask_sum = float(
        np.sum(
            asks[:levels, 1]
        )
    )

    return bid_sum, ask_sum


# ============================================================
# TAKER FLOW
# ============================================================

def calculate_taker_flow(trades):

    buy_volume = 0.0
    sell_volume = 0.0

    buy_notional = 0.0
    sell_notional = 0.0

    count = 0

    for trade in trades:

        try:

            price = float(trade["p"])
            qty = float(trade["q"])
            maker = bool(trade["m"])

            notional = price * qty

            if maker:
                sell_volume += qty
                sell_notional += notional
            else:
                buy_volume += qty
                buy_notional += notional

            count += 1

        except Exception:
            continue

    total = buy_volume + sell_volume

    if total <= 0:
        ratio = 0.0
    else:
        ratio = (
            buy_volume - sell_volume
        ) / total

    return {
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "flow": buy_volume - sell_volume,
        "ratio": clamp(ratio),
        "count": count,
    }


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df,
    period=14,
):

    if df is None or len(df) < 2:
        return 0.0

    previous_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (
                df["High"]
                - previous_close
            ).abs(),
            (
                df["Low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1.0 / period,
        adjust=False,
    ).mean()

    return max(
        0.0,
        safe_float(
            atr.iloc[-1]
        ),
    )


# ============================================================
# TECHNICAL FEATURES
# ============================================================

def technical_features(df):

    if df.empty:
        return {}

    close = df["Close"]

    ema20 = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False,
    ).mean()

    ema200 = close.ewm(
        span=200,
        adjust=False,
    ).mean()

    sma20 = close.rolling(
        20,
        min_periods=5,
    ).mean()

    returns = close.pct_change()

    volatility = (
        returns
        .rolling(20)
        .std()
        .iloc[-1]
    )

    momentum_5 = (
        close.iloc[-1]
        / close.iloc[-6]
        - 1
        if len(close) >= 6
        else 0
    )

    momentum_20 = (
        close.iloc[-1]
        / close.iloc[-21]
        - 1
        if len(close) >= 21
        else 0
    )

    price = safe_float(
        close.iloc[-1]
    )

    trend_score = 0.0

    if price > safe_float(ema20.iloc[-1]):
        trend_score += 0.30
    else:
        trend_score -= 0.30

    if price > safe_float(ema50.iloc[-1]):
        trend_score += 0.25
    else:
        trend_score -= 0.25

    if price > safe_float(ema200.iloc[-1]):
        trend_score += 0.20
    else:
        trend_score -= 0.20

    trend_score += clamp(
        momentum_20 * 20,
        -0.25,
        0.25,
    )

    return {
        "price": price,
        "ema20": safe_float(
            ema20.iloc[-1]
        ),
        "ema50": safe_float(
            ema50.iloc[-1]
        ),
        "ema200": safe_float(
            ema200.iloc[-1]
        ),
        "sma20": safe_float(
            sma20.iloc[-1]
        ),
        "volatility": safe_float(
            volatility
        ),
        "momentum_5": safe_float(
            momentum_5
        ),
        "momentum_20": safe_float(
            momentum_20
        ),
        "trend_score": clamp(
            trend_score
        ),
    }


# ============================================================
# HIGHER TIMEFRAME CONFIRMATION
# ============================================================

@st.cache_data(
    ttl=10,
    show_spinner=False,
)
def higher_timeframe_bias(symbol):

    timeframes = [
        "1h",
        "4h",
        "1d",
    ]

    result = {}

    for tf in timeframes:

        df = fetch_klines(
            symbol,
            tf,
            100,
        )

        if df.empty:
            result[tf] = 0.0
            continue

        close = df["Close"]

        ema20 = close.ewm(
            span=20,
            adjust=False,
        ).mean()

        ema50 = close.ewm(
            span=50,
            adjust=False,
        ).mean()

        price = safe_float(
            close.iloc[-1]
        )

        score = 0.0

        if price > safe_float(
            ema20.iloc[-1]
        ):
            score += 0.5
        else:
            score -= 0.5

        if price > safe_float(
            ema50.iloc[-1]
        ):
            score += 0.5
        else:
            score -= 0.5

        result[tf] = clamp(score)

    return result


# ============================================================
# XGBOOST
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def load_model():

    if not MODEL_FILE.exists():
        return None

    try:
        return joblib.load(
            MODEL_FILE
        )
    except Exception:
        return None


def model_signal(
    model,
    features,
):

    if model is None:
        return None

    try:

        expected = getattr(
            model,
            "n_features_in_",
            None,
        )

        if expected is not None:

            if expected != len(features):
                return None

        X = np.asarray(
            [features],
            dtype=float,
        )

        prediction = model.predict(X)[0]

        confidence = 0.50

        if hasattr(
            model,
            "predict_proba",
        ):

            probs = model.predict_proba(X)[0]

            confidence = float(
                np.max(probs)
            )

        return {
            "prediction": int(
                prediction
            ),
            "confidence": confidence,
        }

    except Exception:
        return None


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(
    df,
    bids,
    asks,
    symbol,
    mode,
):

    tech = technical_features(
        df
    )

    if not tech:
        return None

    ticker = fetch_ticker(
        symbol
    )

    trades = fetch_agg_trades(
        symbol,
        1000,
    )

    flow = calculate_taker_flow(
        trades
    )

    obi5 = calculate_obi(
        bids,
        asks,
        5,
    )

    obi10 = calculate_obi(
        bids,
        asks,
        10,
    )

    obi20 = calculate_obi(
        bids,
        asks,
        20,
    )

    obi50 = calculate_obi(
        bids,
        asks,
        50,
    )

    weighted_obi = calculate_weighted_obi(
        bids,
        asks,
        20,
    )

    multi_obi = calculate_multi_obi(
        bids,
        asks,
    )

    bid20, ask20 = depth_volume(
        bids,
        asks,
        20,
    )

    bid50, ask50 = depth_volume(
        bids,
        asks,
        50,
    )

    total20 = bid20 + ask20

    spread = 0.0

    if len(bids) > 0 and len(asks) > 0:

        spread = (
            asks[0, 0]
            - bids[0, 0]
        )

    # --------------------------------------------------------
    # XGBOOST FEATURE SET
    # IMPORTANT:
    # Existing model expects 7 features.
    # --------------------------------------------------------

    xgb_features = [
        bid20,
        ask20,
        obi20,
        spread,
        (
            bid20 / ask20
            if ask20 > 0
            else 0.0
        ),
        total20,
        tech["trend_score"],
    ]

    model = load_model()

    ml = model_signal(
        model,
        xgb_features,
    )

    # --------------------------------------------------------
    # BASE QUANT SCORE
    # --------------------------------------------------------

    score = 0.0

    # Order book
    score += multi_obi * 0.30

    # Aggressive flow
    score += flow["ratio"] * 0.25

    # Trend
    score += tech["trend_score"] * 0.25

    # Short momentum
    score += clamp(
        tech["momentum_5"] * 30,
        -1,
        1,
    ) * 0.10

    # Medium momentum
    score += clamp(
        tech["momentum_20"] * 15,
        -1,
        1,
    ) * 0.10

    # --------------------------------------------------------
    # HIGHER TF CONFIRMATION
    # --------------------------------------------------------

    htf = higher_timeframe_bias(
        symbol
    )

    htf_score = (
        htf.get("1h", 0.0) * 0.45
        + htf.get("4h", 0.0) * 0.35
        + htf.get("1d", 0.0) * 0.20
    )

    score += htf_score * 0.20

    # Keep score bounded
    score = clamp(
        score,
        -1,
        1,
    )

    # --------------------------------------------------------
    # ML VOTE
    # --------------------------------------------------------

    ml_score = 0.0
    ml_confidence = 0.50

    if ml is not None:

        ml_confidence = safe_float(
            ml["confidence"],
            0.50,
        )

        if ml["prediction"] == 1:
            ml_score = 1.0
        else:
            ml_score = -1.0

        # Only allow meaningful ML contribution
        ml_weight = min(
            0.25,
            ml_confidence * 0.25,
        )

        score = clamp(
            score
            + ml_score * ml_weight
        )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    raw_confidence = (
        abs(score) * 55
        + abs(multi_obi) * 20
        + abs(flow["ratio"]) * 15
        + abs(htf_score) * 10
    )

    if ml is not None:
        raw_confidence = (
            raw_confidence * 0.75
            + (
                ml_confidence
                * 100
            ) * 0.25
        )

    confidence = float(
        np.clip(
            raw_confidence,
            0,
            99,
        )
    )

    # --------------------------------------------------------
    # SIGNAL THRESHOLDS
    # --------------------------------------------------------

    if (
        score >= 0.70
        and confidence >= 70
    ):
        direction = "STRONG LONG"

    elif (
        score >= 0.42
        and confidence >= 55
    ):
        direction = "LONG"

    elif (
        score <= -0.70
        and confidence >= 70
    ):
        direction = "STRONG SHORT"

    elif (
        score <= -0.42
        and confidence >= 55
    ):
        direction = "SHORT"

    else:
        direction = "WAIT"

    # --------------------------------------------------------
    # ATR SL / TP
    # --------------------------------------------------------

    atr = calculate_atr(
        df,
        14,
    )

    price = tech["price"]

    if atr <= 0:
        atr = price * 0.005

    # Risk adjusted to volatility
    sl_distance = max(
        atr * 1.15,
        price * 0.0025,
    )

    # Hard cap at 0.6%
    sl_distance = min(
        sl_distance,
        price * 0.006,
    )

    # Minimum 1:2 RR
    tp_distance = sl_distance * 2.0

    if direction in (
        "LONG",
        "STRONG LONG",
    ):

        entry = price
        stop_loss = entry - sl_distance
        target1 = entry + tp_distance
        target2 = entry + (
            sl_distance * 3.0
        )

    elif direction in (
        "SHORT",
        "STRONG SHORT",
    ):

        entry = price
        stop_loss = entry + sl_distance
        target1 = entry - tp_distance
        target2 = entry - (
            sl_distance * 3.0
        )

    else:

        entry = price
        stop_loss = entry
        target1 = entry
        target2 = entry

    # --------------------------------------------------------
    # TICKER
    # --------------------------------------------------------

    change_24h = safe_float(
        ticker.get(
            "priceChangePercent",
            0,
        )
    )

    volume_24h = safe_float(
        ticker.get(
            "quoteVolume",
            0,
        )
    )

    return {
        "symbol": symbol,
        "mode": mode,
        "timestamp": iso_now(),

        "direction": direction,
        "score": score,
        "confidence": confidence,

        "price": price,

        "entry": entry,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,

        "atr": atr,

        "obi5": obi5,
        "obi10": obi10,
        "obi20": obi20,
        "obi50": obi50,
        "weighted_obi": weighted_obi,
        "multi_obi": multi_obi,

        "bid20": bid20,
        "ask20": ask20,
        "bid50": bid50,
        "ask50": ask50,

        "spread": spread,

        "taker_buy": flow["buy_volume"],
        "taker_sell": flow["sell_volume"],
        "taker_flow": flow["flow"],
        "taker_flow_ratio": flow["ratio"],
        "trade_count": flow["count"],

        "trend_score": tech["trend_score"],
        "momentum_5": tech["momentum_5"],
        "momentum_20": tech["momentum_20"],

        "ema20": tech["ema20"],
        "ema50": tech["ema50"],
        "ema200": tech["ema200"],

        "volatility": tech["volatility"],

        "htf_1h": htf.get("1h", 0.0),
        "htf_4h": htf.get("4h", 0.0),
        "htf_1d": htf.get("1d", 0.0),

        "ml_available": ml is not None,
        "ml_confidence": ml_confidence,

        "change_24h": change_24h,
        "volume_24h": volume_24h,
    }


# ============================================================
# HISTORY
# ============================================================

def load_history():

    if not HISTORY_FILE.exists():
        return []

    try:

        df = pd.read_csv(
            HISTORY_FILE
        )

        if df.empty:
            return []

        return df.tail(
            300
        ).to_dict(
            "records"
        )

    except Exception:
        return []


def save_signal(signal):

    if signal is None:
        return

    row = {
        "timestamp": signal["timestamp"],
        "symbol": signal["symbol"],
        "mode": signal["mode"],
        "direction": signal["direction"],
        "score": signal["score"],
        "confidence": signal["confidence"],
        "entry": signal["entry"],
        "stop_loss": signal["stop_loss"],
        "target1": signal["target1"],
        "target2": signal["target2"],
        "obi20": signal["obi20"],
        "obi50": signal["obi50"],
        "taker_flow_ratio": signal[
            "taker_flow_ratio"
        ],
    }

    try:

        exists = HISTORY_FILE.exists()

        pd.DataFrame(
            [row]
        ).to_csv(
            HISTORY_FILE,
            mode="a",
            header=not exists,
            index=False,
        )

    except Exception:
        pass


# ============================================================
# CHART
# ============================================================

def create_chart(
    df,
    signal,
):

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["Time"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    ema20 = df["Close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    ema50 = df["Close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=ema20,
            name="EMA 20",
            line=dict(
                width=1
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Time"],
            y=ema50,
            name="EMA 50",
            line=dict(
                width=1
            ),
        )
    )

    if signal is not None:

        entry = signal["entry"]
        sl = signal["stop_loss"]
        tp1 = signal["target1"]
        tp2 = signal["target2"]

        fig.add_hline(
            y=entry,
            annotation_text="ENTRY",
            line_dash="solid",
        )

        if signal["direction"] != "WAIT":

            fig.add_hline(
                y=sl,
                annotation_text="STOP LOSS",
                line_dash="dot",
            )

            fig.add_hline(
                y=tp1,
                annotation_text="TARGET 1",
                line_dash="dash",
            )

            fig.add_hline(
                y=tp2,
                annotation_text="TARGET 2",
                line_dash="dash",
            )

    fig.update_layout(
        height=520,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
        legend=dict(
            orientation="h"
        ),
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="main-title">
        ⚡ ZIA RESEARCH
        </div>

        <div class="sub-title">
        Quantitative Market Research Terminal
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    selected_symbol = st.selectbox(
        "MARKET",
        COINS,
        index=COINS.index(
            st.session_state.selected_symbol
        )
        if st.session_state.selected_symbol
        in COINS
        else 0,
    )

    st.session_state.selected_symbol = (
        selected_symbol
    )

    mode_key = st.selectbox(
        "SIGNAL MODE",
        list(TRADE_MODES.keys()),
        format_func=lambda x:
            TRADE_MODES[x]["label"],
        index=list(
            TRADE_MODES.keys()
        ).index(
            st.session_state.selected_mode
        ),
    )

    st.session_state.selected_mode = (
        mode_key
    )

    mode = TRADE_MODES[
        mode_key
    ]

    st.divider()

    st.markdown(
        "### TRADE HORIZON"
    )

    st.info(
        f"""
Analysis: {mode["analysis_tf"].upper()}

Trade horizon: {mode["max_holding"]}

Reference: {", ".join(mode["reference"])}
"""
    )

    st.session_state.auto_scan = st.toggle(
        "AUTO SIGNAL SCAN",
        value=st.session_state.auto_scan,
    )

    st.session_state.show_chart = st.toggle(
        "SHOW PRICE CHART",
        value=st.session_state.show_chart,
    )

    st.divider()

    st.caption(
        "Data: Binance USDⓈ-M Futures"
    )

    st.caption(
        "Execution: research only"
    )


# ============================================================
# AUTO REFRESH
# ============================================================

if st.session_state.auto_scan:

    st_autorefresh(
        interval=5000,
        key="zia_research_refresh",
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
    ⚡ ZIA RESEARCH LAB
    </div>

    <div class="sub-title">
    Binance USDⓈ-M Futures • Microstructure • OBI • Taker Flow • ML • Quant Trend
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# FETCH MAIN DATA
# ============================================================

symbol = st.session_state.selected_symbol

mode_key = st.session_state.selected_mode

mode = TRADE_MODES[
    mode_key
]

interval = mode[
    "analysis_tf"
]

df = fetch_klines(
    symbol,
    interval,
    250,
)

bids, asks = fetch_orderbook(
    symbol,
    100,
)


# ============================================================
# DATA VALIDATION
# ============================================================

if df.empty or len(df) < 20:

    st.error(
        "Market data unavailable."
    )

    st.info(
        f"""
        Binance Futures data could not be loaded for {symbol}.

        API:
        {BINANCE_BASE}

        Try refreshing the page.
        """
    )

    st.stop()


if len(bids) < 20 or len(asks) < 20:

    st.warning(
        "Order book depth is temporarily unavailable. "
        "Retrying automatically..."
    )

    st.stop()


# ============================================================
# GENERATE SIGNAL
# ============================================================

signal = generate_signal(
    df=df,
    bids=bids,
    asks=asks,
    symbol=symbol,
    mode=mode_key,
)

if signal is None:

    st.error(
        "Signal engine could not calculate data."
    )

    st.stop()


st.session_state.last_signal = signal


# ============================================================
# SAVE SIGNAL OCCASIONALLY
# ============================================================

current_minute = now_utc().strftime(
    "%Y-%m-%d %H:%M"
)

if (
    st.session_state.get(
        "last_saved_minute"
    )
    != current_minute
):

    save_signal(
        signal
    )

    st.session_state.last_saved_minute = (
        current_minute
    )


# ============================================================
# TOP STATUS
# ============================================================

ticker = fetch_ticker(
    symbol
)

price = signal["price"]

change = signal[
    "change_24h"
]

direction = signal[
    "direction"
]

confidence = signal[
    "confidence"
]

score = signal[
    "score"
]

card_class = direction_class(
    direction
)


# ============================================================
# MAIN SIGNAL CARD
# ============================================================

st.markdown(
    f"""
    <div class="signal-card {card_class}">

        <div class="signal-title">
        FINAL RESEARCH SIGNAL • {symbol} • {mode["label"]}
        </div>

        <div class="signal-value">
        {direction}
        </div>

        <div style="margin-top:10px;color:#9aa6b7;">
        Quant Score: {score:+.3f}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Confidence: {confidence:.1f}%
        </div>

        <div style="margin-top:12px;">
        <span class="price-value">
        ${fmt_price(price)}
        </span>

        &nbsp;&nbsp;

        <span style="color:#8d99aa;">
        24H {change:+.2f}%
        </span>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# KEY METRICS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-label">
        Confidence
        </div>
        <div class="metric-value">
        {confidence:.1f}%
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-label">
        OBI 20
        </div>
        <div class="metric-value">
        {signal["obi20"]:+.4f}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-label">
        OBI 50
        </div>
        <div class="metric-value">
        {signal["obi50"]:+.4f}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-label">
        Taker Flow
        </div>
        <div class="metric-value">
        {signal["taker_flow_ratio"]:+.4f}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-label">
        ATR
        </div>
        <div class="metric-value">
        {fmt_price(signal["atr"])}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ENTRY / SL / TP
# ============================================================

st.markdown(
    '<div class="section-title">TRADE PLAN</div>',
    unsafe_allow_html=True,
)

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.metric(
        "ENTRY",
        fmt_price(
            signal["entry"]
        ),
    )

with p2:
    st.metric(
        "STOP LOSS",
        (
            fmt_price(
                signal["stop_loss"]
            )
            if direction != "WAIT"
            else "WAIT"
        ),
    )

with p3:
    st.metric(
        "TARGET 1 • 1:2",
        (
            fmt_price(
                signal["target1"]
            )
            if direction != "WAIT"
            else "WAIT"
        ),
    )

with p4:
    st.metric(
        "TARGET 2 • 1:3",
        (
            fmt_price(
                signal["target2"]
            )
            if direction != "WAIT"
            else "WAIT"
        ),
    )


# ============================================================
# MICROSTRUCTURE
# ============================================================

st.markdown(
    '<div class="section-title">MICROSTRUCTURE</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "OBI 5",
        f'{signal["obi5"]:+.4f}',
    )

with m2:
    st.metric(
        "OBI 10",
        f'{signal["obi10"]:+.4f}',
    )

with m3:
    st.metric(
        "OBI 20",
        f'{signal["obi20"]:+.4f}',
    )

with m4:
    st.metric(
        "OBI 50",
        f'{signal["obi50"]:+.4f}',
    )


f1, f2, f3, f4 = st.columns(4)

with f1:
    st.metric(
        "Taker Buy",
        f'{signal["taker_buy"]:.3f}',
    )

with f2:
    st.metric(
        "Taker Sell",
        f'{signal["taker_sell"]:.3f}',
    )

with f3:
    st.metric(
        "Flow Ratio",
        f'{signal["taker_flow_ratio"]:+.4f}',
    )

with f4:
    st.metric(
        "Trades",
        f'{signal["trade_count"]:,}',
    )


# ============================================================
# TREND / ML
# ============================================================

st.markdown(
    '<div class="section-title">QUANT + ML</div>',
    unsafe_allow_html=True,
)

q1, q2, q3, q4, q5 = st.columns(5)

with q1:
    st.metric(
        "Trend Score",
        f'{signal["trend_score"]:+.3f}',
    )

with q2:
    st.metric(
        "EMA 20",
        fmt_price(
            signal["ema20"]
        ),
    )

with q3:
    st.metric(
        "EMA 50",
        fmt_price(
            signal["ema50"]
        ),
    )

with q4:
    st.metric(
        "EMA 200",
        fmt_price(
            signal["ema200"]
        ),
    )

with q5:

    ml_text = (
        f'{signal["ml_confidence"] * 100:.1f}%'
        if signal["ml_available"]
        else "OFF"
    )

    st.metric(
        "XGBoost",
        ml_text,
    )


# ============================================================
# HIGHER TIMEFRAME
# ============================================================

st.markdown(
    '<div class="section-title">HIGHER TIMEFRAME CONFIRMATION</div>',
    unsafe_allow_html=True,
)

h1, h2, h3 = st.columns(3)

def bias_text(value):

    value = safe_float(value)

    if value > 0.25:
        return "BULLISH"

    if value < -0.25:
        return "BEARISH"

    return "NEUTRAL"


with h1:
    st.metric(
        "1H",
        bias_text(
            signal["htf_1h"]
        ),
        delta=f'{signal["htf_1h"]:+.2f}',
    )

with h2:
    st.metric(
        "4H",
        bias_text(
            signal["htf_4h"]
        ),
        delta=f'{signal["htf_4h"]:+.2f}',
    )

with h3:
    st.metric(
        "1D",
        bias_text(
            signal["htf_1d"]
        ),
        delta=f'{signal["htf_1d"]:+.2f}',
    )


# ============================================================
# CHART
# ============================================================

if st.session_state.show_chart:

    st.markdown(
        '<div class="section-title">PRICE ACTION</div>',
        unsafe_allow_html=True,
    )

    fig = create_chart(
        df.tail(150),
        signal,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )


# ============================================================
# ORDER BOOK
# ============================================================

with st.expander(
    "ORDER BOOK DEPTH",
    expanded=False,
):

    ob_left, ob_right = st.columns(2)

    with ob_left:

        st.markdown(
            "### BIDS"
        )

        bid_df = pd.DataFrame(
            bids[:20],
            columns=[
                "Price",
                "Quantity",
            ],
        )

        bid_df["Price"] = bid_df[
            "Price"
        ].map(fmt_price)

        st.dataframe(
            bid_df,
            use_container_width=True,
            hide_index=True,
        )

    with ob_right:

        st.markdown(
            "### ASKS"
        )

        ask_df = pd.DataFrame(
            asks[:20],
            columns=[
                "Price",
                "Quantity",
            ],
        )

        ask_df["Price"] = ask_df[
            "Price"
        ].map(fmt_price)

        st.dataframe(
            ask_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# SIGNAL HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">RECENT SIGNALS</div>',
    unsafe_allow_html=True,
)

history = load_history()

if history:

    history_df = pd.DataFrame(
        history
    )

    if not history_df.empty:

        cols = [
            "timestamp",
            "symbol",
            "mode",
            "direction",
            "score",
            "confidence",
            "entry",
            "stop_loss",
            "target1",
            "target2",
        ]

        cols = [
            c for c in cols
            if c in history_df.columns
        ]

        display_df = history_df[
            cols
        ].tail(20).copy()

        if "score" in display_df:
            display_df["score"] = (
                display_df["score"]
                .astype(float)
                .round(3)
            )

        if "confidence" in display_df:
            display_df["confidence"] = (
                display_df["confidence"]
                .astype(float)
                .round(1)
            )

        st.dataframe(
            display_df.iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

else:

    st.info(
        "No signal history yet."
    )


# ============================================================
# FOOTER STATUS
# ============================================================

st.divider()

status1, status2, status3 = st.columns(3)

with status1:
    st.caption(
        f"Market: Binance USDⓈ-M Futures • {symbol}"
    )

with status2:
    st.caption(
        f"Analysis: {interval.upper()}"
    )

with status3:
    st.caption(
        f"Last update: {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
