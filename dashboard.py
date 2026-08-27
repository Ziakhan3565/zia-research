from __future__ import annotations

import os
import time
import math
import pickle
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

from streamlit_autorefresh import st_autorefresh


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ZIA RESEARCH LAB",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

ROOT = Path(__file__).resolve().parent

HISTORY_FILE = ROOT / "signal_history.csv"

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_DEPTH = "https://api.binance.com/api/v3/depth"
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/24hr"


# ============================================================
# TRADE MODES
# ============================================================

TRADE_MODES = {
    "SCALPING": {
        "label": "30M SCALPING",
        "analysis_tf": "30m",
        "engine_mode": "SCALPING",
        "duration_minutes": 15,
        "duration_text": "15 MIN",
        "max_holding": "15 minutes",
        "reference": ["1H", "4H"],
    },

    "15M": {
        "label": "15M",
        "analysis_tf": "15m",
        "engine_mode": "15M",
        "duration_minutes": 90,
        "duration_text": "1.5 HOURS",
        "max_holding": "90 minutes",
        "reference": ["1H", "4H"],
    },

    "1H": {
        "label": "1H",
        "analysis_tf": "1h",
        "engine_mode": "1H",
        "duration_minutes": 1440,
        "duration_text": "24 HOURS",
        "max_holding": "24 hours",
        "reference": ["DAILY", "WEEKLY"],
    },

    "4H": {
        "label": "4H",
        "analysis_tf": "4h",
        "engine_mode": "4H",
        "duration_minutes": 1440,
        "duration_text": "24 HOURS MAX",
        "max_holding": "24 hours maximum",
        "reference": ["WEEKLY", "MONTHLY"],
    },
}


# ============================================================
# COINS
# ============================================================

COINS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "TAOUSDT",
    "XMRUSDT",
]


# ============================================================
# SESSION STATE
# ============================================================

if "trade_history" not in st.session_state:
    st.session_state.trade_history = []

if "active_trades" not in st.session_state:
    st.session_state.active_trades = []

if "last_signals" not in st.session_state:
    st.session_state.last_signals = {}

if "scanner_message" not in st.session_state:
    st.session_state.scanner_message = ""

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "BTCUSDT"

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "SCALPING"


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        df = pd.read_csv(HISTORY_FILE)

        if df.empty:
            return []

        return df.to_dict("records")

    except Exception:
        return []


if not st.session_state.trade_history:
    st.session_state.trade_history = load_history()


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history():

    try:

        if not st.session_state.trade_history:
            return

        pd.DataFrame(
            st.session_state.trade_history
        ).to_csv(
            HISTORY_FILE,
            index=False,
        )

    except Exception:
        pass


# ============================================================
# UTILS
# ============================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def fmt_price(price):

    price = safe_float(price)

    if price <= 0:
        return "—"

    if price >= 1000:
        return f"{price:,.2f}"

    if price >= 1:
        return f"{price:,.4f}"

    return f"{price:.6f}"


def pct(value):

    return f"{safe_float(value) * 100:.1f}%"


def clamp(value, low=-1.0, high=1.0):

    return float(
        np.clip(
            safe_float(value),
            low,
            high,
        )
    )


def now_utc():

    return dt.datetime.now(
        dt.timezone.utc
    )


def iso_now():

    return now_utc().isoformat()


# ============================================================
# SIGNAL DIRECTION
# ============================================================

def direction_label(score, confidence):

    score = safe_float(score)
    confidence = safe_float(confidence)

    if score >= 0.70 and confidence >= 70:
        return "STRONG LONG"

    if score >= 0.45 and confidence >= 55:
        return "LONG"

    if score <= -0.70 and confidence >= 70:
        return "STRONG SHORT"

    if score <= -0.45 and confidence >= 55:
        return "SHORT"

    return "WAIT"


# ============================================================
# TRADE HORIZON
# ============================================================

def get_trade_horizon(mode):

    mode = str(mode).upper()

    config = TRADE_MODES.get(
        mode,
        TRADE_MODES["SCALPING"],
    )

    return config["duration_minutes"]


def get_horizon_text(mode):

    mode = str(mode).upper()

    config = TRADE_MODES.get(
        mode,
        TRADE_MODES["SCALPING"],
    )

    return config["duration_text"]


# ============================================================
# EXPIRY
# ============================================================

def calculate_expiry(created_at, mode):

    minutes = get_trade_horizon(mode)

    try:

        if isinstance(
            created_at,
            str,
        ):

            created = dt.datetime.fromisoformat(
                created_at.replace(
                    "Z",
                    "+00:00",
                )
            )

        else:

            created = created_at

        if created.tzinfo is None:
            created = created.replace(
                tzinfo=dt.timezone.utc
            )

        return created + dt.timedelta(
            minutes=minutes
        )

    except Exception:

        return now_utc() + dt.timedelta(
            minutes=minutes
        )


def seconds_remaining(expiry):

    try:

        if isinstance(
            expiry,
            str,
        ):

            expiry = dt.datetime.fromisoformat(
                expiry.replace(
                    "Z",
                    "+00:00",
                )
            )

        if expiry.tzinfo is None:
            expiry = expiry.replace(
                tzinfo=dt.timezone.utc
            )

        return max(
            0,
            int(
                (
                    expiry
                    - now_utc()
                ).total_seconds()
            ),
        )

    except Exception:

        return 0


def format_remaining(seconds):

    seconds = max(
        0,
        int(seconds),
    )

    days = seconds // 86400
    seconds %= 86400

    hours = seconds // 3600
    seconds %= 3600

    minutes = seconds // 60
    seconds %= 60

    if days > 0:
        return (
            f"{days}d "
            f"{hours}h "
            f"{minutes}m"
        )

    if hours > 0:
        return (
            f"{hours}h "
            f"{minutes}m"
        )

    return (
        f"{minutes}m "
        f"{seconds}s"
    )


# ============================================================
# BINANCE DATA
# ============================================================

@st.cache_data(
    ttl=10,
    show_spinner=False,
)
def fetch_klines(
    symbol,
    interval,
    limit=200,
):

    try:

        response = requests.get(
            BINANCE_KLINES,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            },
            timeout=8,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(
            data,
            list,
        ):
            return pd.DataFrame()

        columns = [
            "Open_Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close_Time",
            "Quote_Volume",
            "Trades",
            "Taker_Buy_Base",
            "Taker_Buy_Quote",
            "Ignore",
        ]

        df = pd.DataFrame(
            data,
            columns=columns,
        )

        numeric_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        for col in numeric_cols:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        df["Time"] = pd.to_datetime(
            df["Open_Time"],
            unit="ms",
            utc=True,
        )

        return df[
            [
                "Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        ].dropna()

    except Exception:

        return pd.DataFrame()


# ============================================================
# ORDER BOOK
# ============================================================

@st.cache_data(
    ttl=3,
    show_spinner=False,
)
def fetch_orderbook(
    symbol,
    limit=50,
):

    try:

        response = requests.get(
            BINANCE_DEPTH,
            params={
                "symbol": symbol,
                "limit": limit,
            },
            timeout=5,
        )

        response.raise_for_status()

        data = response.json()

        bids = np.array(
            data.get(
                "bids",
                [],
            ),
            dtype=float,
        )

        asks = np.array(
            data.get(
                "asks",
                [],
            ),
            dtype=float,
        )

        return bids, asks

    except Exception:

        return (
            np.empty(
                (0, 2)
            ),
            np.empty(
                (0, 2)
            ),
        )


# ============================================================
# 24H TICKER
# ============================================================

@st.cache_data(
    ttl=5,
    show_spinner=False,
)
def fetch_ticker(symbol):

    try:

        response = requests.get(
            BINANCE_TICKER,
            params={
                "symbol": symbol,
            },
            timeout=5,
        )

        response.raise_for_status()

        return response.json()

    except Exception:

        return {}


# ============================================================
# OBI
# ============================================================

def calculate_obi(
    bids,
    asks,
    levels=20,
):

    if (
        len(bids) == 0
        or len(asks) == 0
    ):
        return 0.0

    n = min(
        levels,
        len(bids),
        len(asks),
    )

    bid_sum = 0.0
    ask_sum = 0.0

    for i in range(n):

        weight = 1.0 / (
            i + 1
        )

        bid_sum += (
            max(
                0,
                safe_float(
                    bids[i][1]
                ),
            )
            * weight
        )

        ask_sum += (
            max(
                0,
                safe_float(
                    asks[i][1]
                ),
            )
            * weight
        )

    denominator = (
        bid_sum
        + ask_sum
        + 1e-12
    )

    return clamp(
        (
            bid_sum
            - ask_sum
        )
        / denominator
    )


# ============================================================
# MULTI OBI
# ============================================================

def calculate_multi_obi(
    bids,
    asks,
):

    configs = {
        5: 0.10,
        10: 0.20,
        20: 0.40,
        50: 0.30,
    }

    values = []
    weights = []

    for level, weight in configs.items():

        if (
            len(bids) >= level
            and len(asks) >= level
        ):

            values.append(
                calculate_obi(
                    bids,
                    asks,
                    level,
                )
            )

            weights.append(
                weight
            )

    if not values:
        return 0.0

    return clamp(
        float(
            np.average(
                values,
                weights=weights,
            )
        )
    )


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df,
    period=14,
):

    if (
        df is None
        or df.empty
        or len(df) < 2
    ):
        return 0.0

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (
                high
                - previous_close
            ).abs(),
            (
                low
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = (
        tr.ewm(
            alpha=1.0 / period,
            adjust=False,
        )
        .mean()
    )

    value = safe_float(
        atr.iloc[-1]
    )

    if value <= 0:

        value = safe_float(
            tr.tail(
                period
            ).mean()
        )

    return max(
        0.0,
        value,
    )


# ============================================================
# LOCAL FALLBACK SIGNAL ENGINE
# ============================================================

def local_signal_engine(
    df,
    bids,
    asks,
):

    if (
        df is None
        or df.empty
        or len(df) < 20
    ):

        return {
            "score": 0.0,
            "confidence": 0.0,
            "direction": "WAIT",
            "ml_probability": 0.50,
            "obi": 0.0,
            "trend": 0.0,
            "atr": 0.0,
        }

    close = df["Close"]

    ema20 = (
        close.ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    ema50 = (
        close.ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    atr = calculate_atr(
        df,
        14,
    )

    price = safe_float(
        close.iloc[-1]
    )

    if atr <= 0:
        atr = price * 0.001

    trend = clamp(
        np.tanh(
            (
                ema20.iloc[-1]
                - ema50.iloc[-1]
            )
            / (
                atr * 2
            )
        )
    )

    obi = calculate_multi_obi(
        bids,
        asks,
    )

    returns = (
        close
        .pct_change()
        .dropna()
    )

    momentum = 0.0

    if len(returns) >= 5:

        momentum = clamp(
            np.tanh(
                (
                    (
                        close.iloc[-1]
                        / close.iloc[-5]
                    )
                    - 1
                )
                / 0.005
            )
        )

    score = (
        0.45 * obi
        + 0.35 * trend
        + 0.20 * momentum
    )

    score = clamp(
        score
    )

    confidence = abs(
        score
    ) * 100

    if score >= 0.70:
        direction = "STRONG LONG"

    elif score >= 0.45:
        direction = "LONG"

    elif score <= -0.70:
        direction = "STRONG SHORT"

    elif score <= -0.45:
        direction = "SHORT"

    else:
        direction = "WAIT"

    probability = (
        0.50
        + (
            score
            * 0.45
        )
    )

    probability = float(
        np.clip(
            probability,
            0.05,
            0.95,
        )
    )

    return {
        "score": score,
        "confidence": confidence,
        "direction": direction,
        "ml_probability": probability,
        "obi": obi,
        "trend": trend,
        "atr": atr,
    }


# ============================================================
# TRY REAL RESEARCH ENGINE
# ============================================================

@st.cache_resource(
    show_spinner=False,
)
def load_research_engine():

    try:

        from engine import (
            IntegratedTradingEngine,
        )

        return IntegratedTradingEngine(
            symbol="BTCUSDT"
        )

    except Exception:

        return None


def run_engine(
    symbol,
    df,
    bids,
    asks,
    mode,
):

    engine = load_research_engine()

    if engine is None:

        return local_signal_engine(
            df,
            bids,
            asks,
        )

    try:

        engine.symbol = symbol

        if hasattr(
            engine,
            "tri_engine",
        ):
            engine.tri_engine.set_symbol(
                symbol
            )

        result = engine.analyze(
            df=df,
            bids=bids.tolist()
            if isinstance(
                bids,
                np.ndarray,
            )
            else bids,
            asks=asks.tolist()
            if isinstance(
                asks,
                np.ndarray,
            )
            else asks,
            trade_mode=mode,
        )

        return result

    except Exception:

        return local_signal_engine(
            df,
            bids,
            asks,
        )


# ============================================================
# NORMALIZE ENGINE OUTPUT
# ============================================================

def normalize_signal(
    raw,
    df,
    mode,
):

    price = (
        safe_float(
            raw.get(
                "CURRENT_PRICE",
                raw.get(
                    "current_price",
                    0,
                ),
            )
        )
    )

    if price <= 0 and df is not None and not df.empty:

        price = safe_float(
            df["Close"].iloc[-1]
        )

    score = safe_float(
        raw.get(
            "FINAL_SCORE",
            raw.get(
                "final_score",
                raw.get(
                    "SCORE",
                    raw.get(
                        "score",
                        0,
                    ),
                ),
            ),
        )
    )

    confidence = safe_float(
        raw.get(
            "CONFIDENCE",
            raw.get(
                "confidence",
                abs(score) * 100,
            ),
        )
    )

    ml_probability = safe_float(
        raw.get(
            "ML_PROBABILITY",
            raw.get(
                "ml_probability",
                0.50,
            ),
        ),
        0.50,
    )

    direction = str(
        raw.get(
            "SIGNAL",
            raw.get(
                "signal",
                "",
            ),
        )
    ).upper()

    if direction in [
        "LONG",
        "SHORT",
        "STRONG LONG",
        "STRONG SHORT",
    ]:

        final_direction = direction

    else:

        final_direction = direction_label(
            score,
            confidence,
        )

    atr = calculate_atr(
        df,
        14,
    )

    if atr <= 0:
        atr = price * 0.001

    # ========================================================
    # SL
    # ========================================================

    stop_loss = safe_float(
        raw.get(
            "STOP_LOSS",
            raw.get(
                "stop_loss",
                0,
            ),
        )
    )

    if stop_loss <= 0:

        if "LONG" in final_direction:

            stop_loss = (
                price
                - atr
            )

        elif "SHORT" in final_direction:

            stop_loss = (
                price
                + atr
            )

    # ========================================================
    # TP1 1:2
    # ========================================================

    risk = abs(
        price
        - stop_loss
    )

    if risk <= 0:
        risk = atr

    tp1 = safe_float(
        raw.get(
            "TP1",
            raw.get(
                "tp1",
                0,
            ),
        )
    )

    if tp1 <= 0:

        if "LONG" in final_direction:

            tp1 = (
                price
                + (
                    risk
                    * 2
                )
            )

        elif "SHORT" in final_direction:

            tp1 = (
                price
                - (
                    risk
                    * 2
                )
            )

    # ========================================================
    # TP2 1:3
    # ========================================================

    tp2 = safe_float(
        raw.get(
            "TP2",
            raw.get(
                "tp2",
                0,
            ),
        )
    )

    if tp2 <= 0:

        if "LONG" in final_direction:

            tp2 = (
                price
                + (
                    risk
                    * 3
                )
            )

        elif "SHORT" in final_direction:

            tp2 = (
                price
                - (
                    risk
                    * 3
                )
            )

    return {
        "direction": final_direction,
        "score": score,
        "confidence": confidence,
        "ml_probability": ml_probability,
        "entry": price,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "atr": atr,

        "quant_score": safe_float(
            raw.get(
                "QUANT_SCORE",
                raw.get(
                    "quant_score",
                    score,
                ),
            )
        ),

        "ml_score": safe_float(
            raw.get(
                "ML_SCORE",
                raw.get(
                    "ml_score",
                    0,
                ),
            )
        ),

        "ml_direction": str(
            raw.get(
                "ML_DIRECTION",
                raw.get(
                    "ml_direction",
                    "NEUTRAL",
                ),
            )
        ),

        "tri_signal": str(
            raw.get(
                "TRI_SIGNAL",
                raw.get(
                    "tri_signal",
                    "NEUTRAL",
                ),
            )
        ),

        "tri_touched": bool(
            raw.get(
                "TRI_TOUCHED",
                raw.get(
                    "tri_touched",
                    False,
                ),
            )
        ),

        "tri_timeframe": raw.get(
            "TRI_TIMEFRAME",
            raw.get(
                "tri_timeframe",
                None,
            ),
        ),

        "tri_line": safe_float(
            raw.get(
                "TRI_LINE",
                raw.get(
                    "tri_line",
                    0,
                ),
            )
        ),

        "tri_rr": safe_float(
            raw.get(
                "TRI_RR",
                raw.get(
                    "tri_rr",
                    0,
                ),
            )
        ),

        "tri_reason": str(
            raw.get(
                "TRI_REASON",
                raw.get(
                    "tri_reason",
                    "NO_SETUP",
                ),
            )
        ),

        "features": raw.get(
            "FEATURES",
            raw.get(
                "features",
                {},
            ),
        ),

        "weights": raw.get(
            "WEIGHTS",
            raw.get(
                "weights",
                {},
            ),
        ),
    }


# ============================================================
# TRADE VALIDATION
# ============================================================

def is_trade_signal(direction):

    return direction in [
        "LONG",
        "SHORT",
        "STRONG LONG",
        "STRONG SHORT",
    ]


# ============================================================
# CREATE TRADE
# ============================================================

def create_trade(
    symbol,
    mode,
    signal,
):

    if not is_trade_signal(
        signal["direction"]
    ):
        return None

    created = now_utc()

    expiry = calculate_expiry(
        created,
        mode,
    )

    trade_id = (
        f"{symbol}_"
        f"{mode}_"
        f"{int(time.time())}"
    )

    direction = signal[
        "direction"
    ]

    trade = {

        "trade_id": trade_id,

        "timestamp": created.isoformat(),

        "symbol": symbol,

        "mode": mode,

        "timeframe": TRADE_MODES[
            mode
        ][
            "analysis_tf"
        ],

        "direction": direction,

        "entry_price": signal[
            "entry"
        ],

        "stop_loss": signal[
            "stop_loss"
        ],

        "tp1": signal[
            "tp1"
        ],

        "tp2": signal[
            "tp2"
        ],

        "confidence": signal[
            "confidence"
        ],

        "score": signal[
            "score"
        ],

        "ml_probability": signal[
            "ml_probability"
        ],

        "tri_signal": signal[
            "tri_signal"
        ],

        "tri_line": signal[
            "tri_line"
        ],

        "tri_rr": signal[
            "tri_rr"
        ],

        "duration_minutes":
            get_trade_horizon(
                mode
            ),

        "duration_text":
            get_horizon_text(
                mode
            ),

        "expiry_time":
            expiry.isoformat(),

        "status": "ACTIVE",

        "outcome": "PENDING",

        "exit_price": 0.0,

        "pnl_percent": 0.0,

        "closed_at": "",

        "close_reason": "",
    }

    return trade


# ============================================================
# RESOLVE ACTIVE TRADES
# ============================================================

def resolve_active_trades():

    if not st.session_state.active_trades:
        return

    changed = False

    current_cache = {}

    for trade in st.session_state.active_trades:

        symbol = trade["symbol"]

        if symbol not in current_cache:

            try:

                ticker = fetch_ticker(
                    symbol
                )

                price = safe_float(
                    ticker.get(
                        "lastPrice",
                        0,
                    )
                )

            except Exception:

                price = 0.0

            current_cache[
                symbol
            ] = price

        price = current_cache[
            symbol
        ]

        if price <= 0:
            continue

        direction = str(
            trade["direction"]
        ).upper()

        entry = safe_float(
            trade["entry_price"]
        )

        sl = safe_float(
            trade["stop_loss"]
        )

        tp1 = safe_float(
            trade["tp1"]
        )

        tp2 = safe_float(
            trade["tp2"]
        )

        expiry = trade[
            "expiry_time"
        ]

        remaining = seconds_remaining(
            expiry
        )

        reason = None
        outcome = None

        # ====================================================
        # LONG
        # ====================================================

        if "LONG" in direction:

            if price <= sl:

                reason = "STOP LOSS"
                outcome = "LOSS"

            elif price >= tp2:

                reason = "TP2"
                outcome = "WIN"

            elif price >= tp1:

                # TP1 is marked but trade stays alive
                if trade.get(
                    "status"
                ) == "ACTIVE":

                    trade[
                        "status"
                    ] = "TP1 HIT"

                    changed = True

        # ====================================================
        # SHORT
        # ====================================================

        elif "SHORT" in direction:

            if price >= sl:

                reason = "STOP LOSS"
                outcome = "LOSS"

            elif price <= tp2:

                reason = "TP2"
                outcome = "WIN"

            elif price <= tp1:

                if trade.get(
                    "status"
                ) == "ACTIVE":

                    trade[
                        "status"
                    ] = "TP1 HIT"

                    changed = True

        # ====================================================
        # EXPIRY
        # ====================================================

        if (
            reason is None
            and remaining <= 0
        ):

            reason = "TIME EXPIRY"
            outcome = (
                "WIN"
                if (
                    (
                        "LONG"
                        in direction
                        and price > entry
                    )
                    or
                    (
                        "SHORT"
                        in direction
                        and price < entry
                    )
                )
                else "LOSS"
            )

        # ====================================================
        # CLOSE
        # ====================================================

        if reason is not None:

            if entry > 0:

                if "LONG" in direction:

                    pnl = (
                        (
                            price
                            - entry
                        )
                        / entry
                    ) * 100

                else:

                    pnl = (
                        (
                            entry
                            - price
                        )
                        / entry
                    ) * 100

            else:

                pnl = 0.0

            trade[
                "status"
            ] = "CLOSED"

            trade[
                "outcome"
            ] = outcome

            trade[
                "exit_price"
            ] = price

            trade[
                "pnl_percent"
            ] = round(
                pnl,
                4,
            )

            trade[
                "closed_at"
            ] = iso_now()

            trade[
                "close_reason"
            ] = reason

            st.session_state.trade_history.insert(
                0,
                trade.copy(),
            )

            changed = True

    st.session_state.active_trades = [
        t
        for t in st.session_state.active_trades
        if t.get(
            "status"
        )
        not in [
            "CLOSED",
        ]
    ]

    if changed:
        save_history()


# ============================================================
# AUTO REFRESH
# ============================================================

st_autorefresh(
    interval=5000,
    limit=None,
    key="zia_research_refresh",
)


# ============================================================
# RESOLVE TRADES
# ============================================================

resolve_active_trades()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 20% 0%,
            rgba(0, 255, 180, 0.04),
            transparent 35%
        ),
        #07090d;
    color: #e6edf3;
}

section[data-testid="stSidebar"] {
    background: #0b0f15;
    border-right: 1px solid #1b2430;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}

.brand {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.subtitle {
    color: #7d8a99;
    font-size: 12px;
    margin-top: -5px;
}

.card {
    background: linear-gradient(
        145deg,
        #10151d,
        #0c1118
    );
    border: 1px solid #1d2733;
    border-radius: 14px;
    padding: 17px;
    margin-bottom: 12px;
    box-shadow:
        0 10px 35px
        rgba(0,0,0,0.20);
}

.card-title {
    color: #7f8b99;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.card-value {
    color: #f0f6fc;
    font-size: 25px;
    font-weight: 800;
}

.card-small {
    color: #9ba7b5;
    font-size: 12px;
}

.signal-card {
    background:
        linear-gradient(
            135deg,
            #101821,
            #0b1017
        );
    border: 1px solid #263241;
    border-radius: 18px;
    padding: 25px;
    min-height: 260px;
}

.signal-title {
    color: #7e8c9b;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
}

.signal-value {
    font-size: 42px;
    font-weight: 900;
    margin: 7px 0 5px 0;
}

.long {
    color: #21d19a;
}

.short {
    color: #ff5c72;
}

.wait {
    color: #f5c451;
}

.white {
    color: #f5f7fa;
}

.progress-bg {
    background: #18212b;
    height: 8px;
    border-radius: 20px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 20px;
    background: #22d3a0;
}

.badge {
    display: inline-block;
    border: 1px solid #293544;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 700;
    margin-right: 5px;
}

.timer {
    font-size: 30px;
    font-weight: 800;
}

.section-title {
    font-size: 18px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 10px;
}

hr {
    border-color: #1b2530;
}

[data-testid="stMetric"] {
    background: #0e141c;
    border: 1px solid #1d2733;
    border-radius: 12px;
    padding: 10px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ⚡ ZIA RESEARCH"
)

st.sidebar.caption(
    "Quantitative Market Research Terminal"
)

st.sidebar.markdown("---")


selected_symbol = st.sidebar.selectbox(
    "MARKET",
    COINS,
    index=COINS.index(
        st.session_state.selected_symbol
    )
    if st.session_state.selected_symbol
    in COINS
    else 0,
)

selected_mode = st.sidebar.selectbox(
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

st.session_state.selected_symbol = (
    selected_symbol
)

st.session_state.selected_mode = (
    selected_mode
)


# ============================================================
# MODE INFO
# ============================================================

mode_config = TRADE_MODES[
    selected_mode
]

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### TRADE HORIZON"
)

st.sidebar.info(
    (
        f"Analysis: **{mode_config['analysis_tf']}**\n\n"
        f"Trade: **{mode_config['duration_text']}**\n\n"
        f"Max holding: **{mode_config['max_holding']}**"
    )
)


# ============================================================
# SIDEBAR OPTIONS
# ============================================================

auto_scan = st.sidebar.toggle(
    "AUTO SIGNAL SCAN",
    value=True,
)

show_chart = st.sidebar.toggle(
    "SHOW PRICE CHART",
    value=True,
)

show_features = st.sidebar.toggle(
    "SHOW RESEARCH FEATURES",
    value=True,
)

show_orderbook = st.sidebar.toggle(
    "SHOW ORDER BOOK",
    value=True,
)


# ============================================================
# MANUAL SIGNAL
# ============================================================

st.sidebar.markdown("---")

if st.sidebar.button(
    "🔄 REFRESH NOW",
    use_container_width=True,
):

    st.cache_data.clear()
    st.rerun()


# ============================================================
# DATA
# ============================================================

interval = mode_config[
    "analysis_tf"
]

df = fetch_klines(
    selected_symbol,
    interval,
    200,
)

bids, asks = fetch_orderbook(
    selected_symbol,
    50,
)

ticker = fetch_ticker(
    selected_symbol
)

# ============================================================
# SIGNAL
# ============================================================

if (
    df.empty
    or len(df) < 20
):

    st.error(
        "Market data unavailable."
    )

    st.stop()


raw_signal = run_engine(
    selected_symbol,
    df,
    bids,
    asks,
    selected_mode,
)

signal = normalize_signal(
    raw_signal,
    df,
    selected_mode,
)


# ============================================================
# SAVE LAST SIGNAL
# ============================================================

signal_key = (
    f"{selected_symbol}_"
    f"{selected_mode}"
)

st.session_state.last_signals[
    signal_key
] = signal


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="brand">
⚡ ZIA RESEARCH LAB
</div>
<div class="subtitle">
REAL-TIME QUANTITATIVE MARKET RESEARCH & PAPER SIGNAL TERMINAL
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")


# ============================================================
# TOP MARKET BAR
# ============================================================

price = signal[
    "entry"
]

ticker_change = safe_float(
    ticker.get(
        "priceChangePercent",
        0,
    )
)

volume_24h = safe_float(
    ticker.get(
        "volume",
        0,
    )
)

obi = calculate_multi_obi(
    bids,
    asks,
)

market_change_text = (
    f"{ticker_change:+.2f}%"
)

st.markdown(
    f"""
<div class="card">

<div class="card-small">
MARKET
</div>

<h2 style="margin:3px 0;">
{selected_symbol}
</h2>

<div style="
font-size:30px;
font-weight:800;
">
{fmt_price(price)}
</div>

<div class="card-small">
24H CHANGE:
{market_change_text}
&nbsp;&nbsp; | &nbsp;&nbsp;
OBI:
{obi:+.3f}
&nbsp;&nbsp; | &nbsp;&nbsp;
MODE:
{mode_config['label']}
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN SIGNAL CARD
# ============================================================

direction = signal[
    "direction"
]

if "LONG" in direction:

    direction_class = "long"

elif "SHORT" in direction:

    direction_class = "short"

else:

    direction_class = "wait"


confidence = signal[
    "confidence"
]

score = signal[
    "score"
]

horizon = get_horizon_text(
    selected_mode
)

st.markdown(
    f"""
<div class="signal-card">

<div class="signal-title">
CURRENT RESEARCH SIGNAL
</div>

<div class="signal-value {direction_class}">
{direction}
</div>

<div style="
font-size:13px;
color:#8e9baa;
margin-bottom:14px;
">
{selected_symbol}
&nbsp; • &nbsp;
{mode_config['label']}
&nbsp; • &nbsp;
TRADE HORIZON: {horizon}
</div>

<div class="card-small">
CONFIDENCE
</div>

<div style="
font-size:25px;
font-weight:800;
margin-bottom:8px;
">
{confidence:.1f}%
</div>

<div class="progress-bg">

<div
class="progress-fill"
style="width:{min(100, max(0, confidence))}%">
</div>

</div>

<div style="
margin-top:16px;
">

<span class="badge">
SCORE {score:+.3f}
</span>

<span class="badge">
ML {signal['ml_probability'] * 100:.1f}%
</span>

<span class="badge">
TRI {signal['tri_signal']}
</span>

</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TRADE PLAN
# ============================================================

st.markdown(
    '<div class="section-title">🎯 TRADE PLAN</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.metric(
        "ENTRY",
        fmt_price(
            signal["entry"]
        ),
    )

with c2:

    st.metric(
        "STOP LOSS",
        fmt_price(
            signal["stop_loss"]
        ),
    )

with c3:

    st.metric(
        "TP1 • 1:2",
        fmt_price(
            signal["tp1"]
        ),
    )

with c4:

    st.metric(
        "TP2 • 1:3",
        fmt_price(
            signal["tp2"]
        ),
    )

with c5:

    st.metric(
        "HORIZON",
        horizon,
    )


# ============================================================
# RISK / RESEARCH
# ============================================================

st.markdown(
    '<div class="section-title">🧠 RESEARCH ENGINE</div>',
    unsafe_allow_html=True,
)

r1, r2, r3, r4, r5, r6 = st.columns(6)

with r1:

    st.metric(
        "QUANT SCORE",
        f"{signal['quant_score']:+.3f}",
    )

with r2:

    st.metric(
        "ML SCORE",
        f"{signal['ml_score']:+.3f}",
    )

with r3:

    st.metric(
        "ML UP",
        pct(
            signal[
                "ml_probability"
            ]
        ),
    )

with r4:

    st.metric(
        "OBI",
        f"{obi:+.3f}",
    )

with r5:

    st.metric(
        "TRI RR",
        (
            f"{signal['tri_rr']:.2f}"
            if signal["tri_rr"] > 0
            else "—"
        ),
    )

with r6:

    st.metric(
        "ATR",
        fmt_price(
            signal["atr"]
        ),
    )


# ============================================================
# SIGNAL EXPIRY / PAPER TRADE
# ============================================================

st.markdown(
    '<div class="section-title">⏱️ SIGNAL / TRADE TIMER</div>',
    unsafe_allow_html=True,
)

current_active = None

for trade in st.session_state.active_trades:

    if (
        trade["symbol"]
        == selected_symbol
        and trade["mode"]
        == selected_mode
    ):

        current_active = trade
        break


if current_active is not None:

    remaining = seconds_remaining(
        current_active[
            "expiry_time"
        ]
    )

    st.markdown(
        f"""
<div class="card">

<div class="card-title">
ACTIVE PAPER TRADE
</div>

<div class="timer">
{format_remaining(remaining)}
</div>

<div class="card-small">
EXPIRY:
{current_active['expiry_time']}
</div>

<br>

<div class="card-small">
STATUS:
<b>
{current_active['status']}
</b>
&nbsp;&nbsp; | &nbsp;&nbsp;
ENTRY:
<b>
{fmt_price(current_active['entry_price'])}
</b>
&nbsp;&nbsp; | &nbsp;&nbsp;
DIRECTION:
<b>
{current_active['direction']}
</b>
</div>

</div>
""",
        unsafe_allow_html=True,
    )

else:

    if is_trade_signal(
        direction
    ):

        st.markdown(
            f"""
<div class="card">

<div class="card-title">
SIGNAL READY
</div>

<div style="
font-size:21px;
font-weight:800;
">
{direction}
</div>

<div class="card-small">
If opened now, maximum trade duration:
<b>{horizon}</b>
</div>

</div>
""",
            unsafe_allow_html=True,
        )

        if st.button(
            "🚀 OPEN PAPER TRADE",
            type="primary",
            use_container_width=True,
        ):

            new_trade = create_trade(
                selected_symbol,
                selected_mode,
                signal,
            )

            if new_trade:

                st.session_state.active_trades.append(
                    new_trade
                )

                st.success(
                    (
                        f"{direction} paper trade opened. "
                        f"Expiry: {horizon}."
                    )
                )

                st.rerun()

    else:

        st.info(
            "No valid trade signal. "
            "Waiting for confirmation."
        )


# ============================================================
# CHART
# ============================================================

if show_chart:

    st.markdown(
        '<div class="section-title">📈 PRICE ACTION</div>',
        unsafe_allow_html=True,
    )

    chart = go.Figure()

    chart.add_trace(
        go.Candlestick(
            x=df["Time"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    chart.add_hline(
        y=signal["entry"],
        line_dash="solid",
        annotation_text="ENTRY",
    )

    if signal["stop_loss"] > 0:

        chart.add_hline(
            y=signal["stop_loss"],
            line_dash="dot",
            annotation_text="SL",
        )

    if signal["tp1"] > 0:

        chart.add_hline(
            y=signal["tp1"],
            line_dash="dot",
            annotation_text="TP1 1:2",
        )

    if signal["tp2"] > 0:

        chart.add_hline(
            y=signal["tp2"],
            line_dash="dot",
            annotation_text="TP2 1:3",
        )

    if signal["tri_line"] > 0:

        chart.add_hline(
            y=signal["tri_line"],
            line_dash="dash",
            annotation_text=(
                f"TRI {signal['tri_timeframe']}"
            ),
        )

    chart.update_layout(
        height=570,
        template="plotly_dark",
        paper_bgcolor="#07090d",
        plot_bgcolor="#07090d",
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
        legend=dict(
            orientation="h",
        ),
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )


# ============================================================
# ORDER BOOK
# ============================================================

if show_orderbook:

    st.markdown(
        '<div class="section-title">📚 ORDER BOOK IMBALANCE</div>',
        unsafe_allow_html=True,
    )

    ob1, ob2 = st.columns(2)

    with ob1:

        bid_total = (
            float(
                np.sum(
                    bids[:20, 1]
                )
            )
            if len(bids) >= 20
            else 0
        )

        ask_total = (
            float(
                np.sum(
                    asks[:20, 1]
                )
            )
            if len(asks) >= 20
            else 0
        )

        st.markdown(
            f"""
<div class="card">

<div class="card-title">
TOP 20 DEPTH
</div>

<div style="
font-size:24px;
font-weight:800;
">
BIDS {bid_total:.2f}
</div>

<div style="
font-size:24px;
font-weight:800;
">
ASKS {ask_total:.2f}
</div>

<div class="card-small">
OBI: {calculate_obi(bids, asks, 20):+.4f}
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    with ob2:

        bid_total50 = (
            float(
                np.sum(
                    bids[:50, 1]
                )
            )
            if len(bids) >= 50
            else 0
        )

        ask_total50 = (
            float(
                np.sum(
                    asks[:50, 1]
                )
            )
            if len(asks) >= 50
            else 0
        )

        st.markdown(
            f"""
<div class="card">

<div class="card-title">
TOP 50 DEPTH
</div>

<div style="
font-size:24px;
font-weight:800;
">
BIDS {bid_total50:.2f}
</div>

<div style="
font-size:24px;
font-weight:800;
">
ASKS {ask_total50:.2f}
</div>

<div class="card-small">
OBI: {calculate_obi(bids, asks, 50):+.4f}
</div>

</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# RESEARCH FEATURES
# ============================================================

if show_features:

    features = signal.get(
        "features",
        {},
    )

    if isinstance(
        features,
        dict,
    ) and features:

        st.markdown(
            '<div class="section-title">🔬 RESEARCH FEATURES</div>',
            unsafe_allow_html=True,
        )

        feature_df = pd.DataFrame(
            [
                {
                    "FEATURE": key,
                    "VALUE": safe_float(
                        value
                    ),
                }
                for key, value
                in features.items()
            ]
        )

        if not feature_df.empty:

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# ACTIVE TRADES
# ============================================================

st.markdown(
    '<div class="section-title">🔥 ACTIVE TRADES</div>',
    unsafe_allow_html=True,
)

if st.session_state.active_trades:

    active_rows = []

    for trade in st.session_state.active_trades:

        active_rows.append(
            {
                "Symbol":
                    trade["symbol"],

                "Mode":
                    trade["mode"],

                "Direction":
                    trade["direction"],

                "Entry":
                    fmt_price(
                        trade[
                            "entry_price"
                        ]
                    ),

                "SL":
                    fmt_price(
                        trade[
                            "stop_loss"
                        ]
                    ),

                "TP1":
                    fmt_price(
                        trade[
                            "tp1"
                        ]
                    ),

                "TP2":
                    fmt_price(
                        trade[
                            "tp2"
                        ]
                    ),

                "Horizon":
                    trade[
                        "duration_text"
                    ],

                "Remaining":
                    format_remaining(
                        seconds_remaining(
                            trade[
                                "expiry_time"
                            ]
                        )
                    ),

                "Status":
                    trade["status"],
            }
        )

    st.dataframe(
        pd.DataFrame(
            active_rows
        ),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No active paper trades."
    )


# ============================================================
# HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">📊 TRADE HISTORY</div>',
    unsafe_allow_html=True,
)

history = st.session_state.trade_history

if history:

    hist_df = pd.DataFrame(
        history
    )

    total = len(
        hist_df
    )

    wins = len(
        hist_df[
            hist_df[
                "outcome"
            ].astype(str).str.upper()
            == "WIN"
        ]
    )

    losses = len(
        hist_df[
            hist_df[
                "outcome"
            ].astype(str).str.upper()
            == "LOSS"
        ]
    )

    win_rate = (
        wins
        / max(
            1,
            wins + losses,
        )
    ) * 100

    pnl = (
        pd.to_numeric(
            hist_df.get(
                "pnl_percent",
                pd.Series(
                    [0] * total
                ),
            ),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.metric(
            "TOTAL TRADES",
            total,
        )

    with h2:
        st.metric(
            "WINS",
            wins,
        )

    with h3:
        st.metric(
            "WIN RATE",
            f"{win_rate:.1f}%",
        )

    with h4:
        st.metric(
            "TOTAL PNL",
            f"{pnl:+.2f}%",
        )

    display_cols = [
        "timestamp",
        "symbol",
        "mode",
        "direction",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "confidence",
        "outcome",
        "pnl_percent",
        "close_reason",
    ]

    available = [
        col
        for col in display_cols
        if col in hist_df.columns
    ]

    display_df = hist_df[
        available
    ].copy()

    rename = {
        "timestamp": "TIME",
        "symbol": "SYMBOL",
        "mode": "MODE",
        "direction": "DIRECTION",
        "entry_price": "ENTRY",
        "stop_loss": "SL",
        "tp1": "TP1",
        "tp2": "TP2",
        "confidence": "CONFIDENCE",
        "outcome": "OUTCOME",
        "pnl_percent": "PNL %",
        "close_reason": "REASON",
    }

    display_df = display_df.rename(
        columns=rename
    )

    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No closed trades yet."
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

with st.expander(
    "⚙️ HISTORY MANAGEMENT"
):

    col_a, col_b = st.columns(2)

    with col_a:

        if st.button(
            "DELETE HISTORY",
            use_container_width=True,
        ):

            st.session_state.trade_history = []

            try:

                if HISTORY_FILE.exists():
                    HISTORY_FILE.unlink()

            except Exception:
                pass

            st.success(
                "Trade history deleted."
            )

            st.rerun()

    with col_b:

        if history:

            csv_data = pd.DataFrame(
                history
            ).to_csv(
                index=False
            )

            st.download_button(
                "DOWNLOAD CSV",
                csv_data,
                file_name=(
                    "zia_research_trade_history.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
<div style="
text-align:center;
color:#596675;
font-size:11px;
padding:15px;
">
ZIA RESEARCH LAB • QUANTITATIVE PAPER SIGNAL TERMINAL
<br>
Signal horizons are configured as:
30M SCALPING → 15 MIN |
15M → 90 MIN |
1H → 24 HOURS |
4H → 24 HOURS MAX
</div>
""",
    unsafe_allow_html=True,
)
