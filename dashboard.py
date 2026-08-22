import os
import time
import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from streamlit_autorefresh import st_autorefresh


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ZIA Research Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st_autorefresh(
    interval=5000,
    limit=None,
    key="market_refresh"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = "xgboost_direction_model.pkl"
HISTORY_FILE = "signal_history.csv"

MIN_CONFIDENCE = 70.0

MIN_OBI = 0.10
MIN_OFI = 0.02

MAX_SPREAD_PCT = 0.0010

ATR_MULTIPLIER = 1.0

RR_TP1 = 2.0
RR_TP2 = 3.0


# ============================================================
# SESSION STATE
# ============================================================

if "trade_history" not in st.session_state:
    st.session_state.trade_history = []


# ============================================================
# EXPECTED HISTORY COLUMNS
# ============================================================

HISTORY_COLUMNS = [
    "trade_id",
    "timestamp",
    "symbol",
    "timeframe",
    "direction",
    "entry_price",
    "stop_loss",
    "tp1",
    "tp2",
    "exit_price",
    "confidence",
    "p_long",
    "p_short",
    "p_no_trade",
    "obi_top20",
    "obi_top50",
    "ofi",
    "trend10",
    "trend20",
    "outcome",
    "pnl_percent",
    "status"
]


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    try:

        df = pd.read_csv(
            HISTORY_FILE
        )

        if df.empty:
            return []

        for column in HISTORY_COLUMNS:

            if column not in df.columns:

                if column == "outcome":
                    df[column] = "PENDING"

                elif column == "status":
                    df[column] = "Open"

                else:
                    df[column] = 0

        return df[
            HISTORY_COLUMNS
        ].to_dict("records")

    except Exception as e:

        st.warning(
            f"History file could not be loaded: {e}"
        )

        return []


def save_history(history):

    if not history:
        return

    try:

        df = pd.DataFrame(
            history
        )

        for column in HISTORY_COLUMNS:

            if column not in df.columns:
                df[column] = 0

        df[
            HISTORY_COLUMNS
        ].to_csv(
            HISTORY_FILE,
            index=False
        )

    except Exception as e:

        st.error(
            f"History save error: {e}"
        )


if not st.session_state.trade_history:

    st.session_state.trade_history = (
        load_history()
    )


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_xgboost_model():

    if not os.path.exists(
        MODEL_FILE
    ):
        return None

    try:

        return joblib.load(
            MODEL_FILE
        )

    except Exception as e:

        st.error(
            f"XGBoost model loading error: {e}"
        )

        return None


model_package = load_xgboost_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚡ ZIA Research Controls"
)

coins = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT"
]

timeframes = {
    "1m — Scalping": "1m",
    "15m — Medium": "15m",
    "30m — Medium": "30m",
    "1h — Intraday": "1h",
    "4h — Intraday": "4h"
}

symbol = st.sidebar.selectbox(
    "Cryptocurrency",
    coins
)

tf_label = st.sidebar.selectbox(
    "Timeframe",
    list(timeframes.keys()),
    index=1
)

timeframe = timeframes[
    tf_label
]

paper_trading = st.sidebar.toggle(
    "Paper Trading",
    value=True
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Minimum Confidence: **{MIN_CONFIDENCE:.0f}%**"
)

st.sidebar.write(
    f"Minimum OBI: **{MIN_OBI:.2f}**"
)

st.sidebar.write(
    f"Minimum OFI: **{MIN_OFI:.2f}**"
)

st.sidebar.write(
    "TP1 Risk/Reward: **1:2**"
)

st.sidebar.write(
    "TP2 Risk/Reward: **1:3**"
)

st.sidebar.markdown("---")

if st.sidebar.button(
    "🗑️ Clear Trade History",
    use_container_width=True
):

    st.session_state.trade_history = []

    if os.path.exists(
        HISTORY_FILE
    ):
        try:
            os.remove(
                HISTORY_FILE
            )
        except Exception:
            pass

    st.rerun()


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=5)
def fetch_klines(
    symbol,
    timeframe,
    limit=200
):

    url = (
        "https://data-api.binance.vision/api/v3/klines"
        f"?symbol={symbol}"
        f"&interval={timeframe}"
        f"&limit={limit}"
    )

    try:

        response = requests.get(
            url,
            timeout=8
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(
            data,
            list
        ):
            return pd.DataFrame()

        columns = [
            "open_time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_base",
            "taker_quote",
            "ignore"
        ]

        df = pd.DataFrame(
            data,
            columns=columns
        )

        df["Time"] = pd.to_datetime(
            df["open_time"],
            unit="ms"
        )

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna(
            subset=numeric_columns
        )

        return df[
            [
                "Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        ]

    except Exception as e:

        st.error(
            f"Market data error: {e}"
        )

        return pd.DataFrame()


# ============================================================
# ORDER BOOK
# ============================================================

@st.cache_data(ttl=3)
def fetch_order_book(
    symbol,
    limit=50
):

    url = (
        "https://data-api.binance.vision/api/v3/depth"
        f"?symbol={symbol}"
        f"&limit={limit}"
    )

    try:

        response = requests.get(
            url,
            timeout=8
        )

        response.raise_for_status()

        data = response.json()

        bids = np.asarray(
            data.get("bids", []),
            dtype=float
        )

        asks = np.asarray(
            data.get("asks", []),
            dtype=float
        )

        if len(bids) < 20:
            return None, None

        if len(asks) < 20:
            return None, None

        return bids, asks

    except Exception:

        return None, None


# ============================================================
# ORDER BOOK FEATURES
# ============================================================

def calculate_orderbook_features(
    bids,
    asks
):

    bids20 = bids[:20]
    asks20 = asks[:20]

    bids50 = bids[:50]
    asks50 = asks[:50]

    bid20 = float(
        np.sum(
            bids20[:, 1]
        )
    )

    ask20 = float(
        np.sum(
            asks20[:, 1]
        )
    )

    bid50 = float(
        np.sum(
            bids50[:, 1]
        )
    )

    ask50 = float(
        np.sum(
            asks50[:, 1]
        )
    )

    obi20 = (
        bid20 - ask20
    ) / (
        bid20 + ask20 + 1e-12
    )

    obi50 = (
        bid50 - ask50
    ) / (
        bid50 + ask50 + 1e-12
    )

    best_bid = float(
        bids[0, 0]
    )

    best_ask = float(
        asks[0, 0]
    )

    mid_price = (
        best_bid + best_ask
    ) / 2.0

    spread = (
        best_ask - best_bid
    )

    spread_pct = (
        spread
        /
        (
            mid_price + 1e-12
        )
    )

    total_depth = (
        bid20 + ask20
    )

    bid_ask_ratio = (
        bid20
        /
        (
            ask20 + 1e-12
        )
    )

    # Order-flow proxy
    ofi = (
        obi20
        *
        total_depth
    )

    ofi_normalized = (
        ofi
        /
        (
            total_depth
            +
            1e-12
        )
    )

    return {
        "top20_bid_sum": bid20,
        "top20_ask_sum": ask20,
        "obi_top20": obi20,
        "obi_top50": obi50,
        "spread": spread,
        "bid_ask_ratio": bid_ask_ratio,
        "total_depth": total_depth,
        "ofi_normalized": ofi_normalized,
        "obi_alignment": (
            obi20 * obi50
        ),
        "spread_pct": spread_pct
    }


# ============================================================
# MARKET FEATURES
# ============================================================

def calculate_market_features(
    df
):

    close = df["Close"]

    return_1 = (
        close.pct_change(1).iloc[-1]
    )

    return_3 = (
        close.pct_change(3).iloc[-1]
    )

    return_5 = (
        close.pct_change(5).iloc[-1]
    )

    sma10 = (
        close
        .rolling(10)
        .mean()
        .iloc[-1]
    )

    sma20 = (
        close
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    trend10 = (
        (
            close.iloc[-1]
            -
            sma10
        )
        /
        (
            sma10 + 1e-12
        )
    )

    trend20 = (
        (
            close.iloc[-1]
            -
            sma20
        )
        /
        (
            sma20 + 1e-12
        )
    )

    volatility = (
        close
        .pct_change()
        .rolling(20)
        .std()
        .iloc[-1]
    )

    return {
        "trend_10": float(
            trend10
        ),
        "trend_20": float(
            trend20
        ),
        "return_1": float(
            return_1
        ),
        "return_3": float(
            return_3
        ),
        "return_5": float(
            return_5
        ),
        "volatility": float(
            volatility
        )
    }


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_direction(
    features
):

    if model_package is None:
        return None

    model = model_package.get(
        "model"
    )

    feature_names = (
        model_package.get(
            "features",
            []
        )
    )

    if model is None:
        return None

    if not feature_names:
        return None

    values = []

    for feature in feature_names:

        value = features.get(
            feature,
            0.0
        )

        if pd.isna(value):
            value = 0.0

        values.append(
            float(value)
        )

    X = pd.DataFrame(
        [values],
        columns=feature_names
    )

    try:

        probabilities = (
            model.predict_proba(X)[0]
        )

        classes = model.classes_

    except Exception as e:

        st.error(
            f"Model prediction error: {e}"
        )

        return None

    probability_map = {
        int(cls): float(prob)
        for cls, prob in zip(
            classes,
            probabilities
        )
    }

    p_none = (
        probability_map.get(
            0,
            0.0
        )
    )

    p_long = (
        probability_map.get(
            1,
            0.0
        )
    )

    p_short = (
        probability_map.get(
            2,
            0.0
        )
    )

    probabilities_dict = {
        "NO_TRADE": p_none,
        "LONG": p_long,
        "SHORT": p_short
    }

    direction = max(
        probabilities_dict,
        key=probabilities_dict.get
    )

    confidence = (
        probabilities_dict[
            direction
        ]
        * 100
    )

    return {
        "direction": direction,
        "confidence": confidence,
        "p_long": p_long * 100,
        "p_short": p_short * 100,
        "p_no_trade": p_none * 100
    }


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df,
    period=14
):

    previous_close = (
        df["Close"].shift(1)
    )

    tr1 = (
        df["High"]
        -
        df["Low"]
    )

    tr2 = (
        df["High"]
        -
        previous_close
    ).abs()

    tr3 = (
        df["Low"]
        -
        previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    atr = (
        true_range
        .rolling(period)
        .mean()
        .iloc[-1]
    )

    if pd.isna(atr) or atr <= 0:

        atr = (
            df["Close"].iloc[-1]
            * 0.005
        )

    return float(atr)


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "📊 ZIA Quantitative Research Terminal"
)

st.caption(
    "XGBoost + OBI + OFI + Trend Confirmation + Paper Trading"
)


# ============================================================
# MODEL CHECK
# ============================================================

if model_package is None:

    st.error(
        "❌ xgboost_direction_model.pkl not found."
    )

    st.warning(
        "Pehle train_model.py run karo aur "
        "xgboost_direction_model.pkl generate karo."
    )

    st.stop()


# ============================================================
# FETCH DATA
# ============================================================

df = fetch_klines(
    symbol,
    timeframe
)

bids, asks = fetch_order_book(
    symbol,
    50
)


if df.empty:

    st.error(
        "❌ Market candles unavailable."
    )

    st.stop()


if bids is None or asks is None:

    st.error(
        "❌ Level-2 order book unavailable."
    )

    st.warning(
        "No fake/dummy order book will be used."
    )

    st.stop()


if len(df) < 30:

    st.error(
        "Not enough candle data."
    )

    st.stop()


# ============================================================
# CALCULATE FEATURES
# ============================================================

orderbook_features = (
    calculate_orderbook_features(
        bids,
        asks
    )
)

market_features = (
    calculate_market_features(
        df
    )
)

features = {
    **orderbook_features,
    **market_features
}


# ============================================================
# MODEL PREDICTION
# ============================================================

prediction = predict_direction(
    features
)

if prediction is None:

    st.error(
        "❌ XGBoost prediction failed."
    )

    st.stop()


model_direction = (
    prediction["direction"]
)

confidence = (
    prediction["confidence"]
)

p_long = (
    prediction["p_long"]
)

p_short = (
    prediction["p_short"]
)

p_no_trade = (
    prediction["p_no_trade"]
)


# ============================================================
# FEATURE VALUES
# ============================================================

obi20 = features[
    "obi_top20"
]

obi50 = features[
    "obi_top50"
]

ofi = features[
    "ofi_normalized"
]

trend10 = features[
    "trend_10"
]

trend20 = features[
    "trend_20"
]

volatility = features[
    "volatility"
]

spread_pct = features[
    "spread_pct"
]


# ============================================================
# CONFIRMATION
# ============================================================

long_confirmations = 0

short_confirmations = 0


if obi20 >= MIN_OBI:
    long_confirmations += 1

if obi50 >= MIN_OBI:
    long_confirmations += 1

if ofi >= MIN_OFI:
    long_confirmations += 1

if trend10 > 0:
    long_confirmations += 1

if trend20 > 0:
    long_confirmations += 1


if obi20 <= -MIN_OBI:
    short_confirmations += 1

if obi50 <= -MIN_OBI:
    short_confirmations += 1

if ofi <= -MIN_OFI:
    short_confirmations += 1

if trend10 < 0:
    short_confirmations += 1

if trend20 < 0:
    short_confirmations += 1


# ============================================================
# FINAL SIGNAL
# ============================================================

final_signal = "NO TRADE"

signal_reason = "Waiting for confirmation."


if model_direction == "LONG":

    if confidence < MIN_CONFIDENCE:

        signal_reason = (
            "XGBoost confidence below 70%."
        )

    elif long_confirmations < 3:

        signal_reason = (
            "LONG confirmations below 3/5."
        )

    elif spread_pct > MAX_SPREAD_PCT:

        signal_reason = (
            "Spread is too high."
        )

    else:

        final_signal = "LONG"

        signal_reason = (
            "XGBoost + order book + trend confirmed."
        )


elif model_direction == "SHORT":

    if confidence < MIN_CONFIDENCE:

        signal_reason = (
            "XGBoost confidence below 70%."
        )

    elif short_confirmations < 3:

        signal_reason = (
            "SHORT confirmations below 3/5."
        )

    elif spread_pct > MAX_SPREAD_PCT:

        signal_reason = (
            "Spread is too high."
        )

    else:

        final_signal = "SHORT"

        signal_reason = (
            "XGBoost + order book + trend confirmed."
        )


else:

    signal_reason = (
        "XGBoost predicts NO TRADE."
    )


# ============================================================
# PRICE / RISK
# ============================================================

entry_price = float(
    df["Close"].iloc[-1]
)

atr = calculate_atr(
    df
)

risk_distance = (
    atr * ATR_MULTIPLIER
)


if final_signal == "LONG":

    stop_loss = (
        entry_price
        -
        risk_distance
    )

    tp1 = (
        entry_price
        +
        risk_distance * RR_TP1
    )

    tp2 = (
        entry_price
        +
        risk_distance * RR_TP2
    )


elif final_signal == "SHORT":

    stop_loss = (
        entry_price
        +
        risk_distance
    )

    tp1 = (
        entry_price
        -
        risk_distance * RR_TP1
    )

    tp2 = (
        entry_price
        -
        risk_distance * RR_TP2
    )


else:

    stop_loss = entry_price
    tp1 = entry_price
    tp2 = entry_price


# ============================================================
# TOP METRICS
# ============================================================

st.markdown(
    "### 🧠 Model Decision"
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Model",
    model_direction
)

m2.metric(
    "Confidence",
    f"{confidence:.1f}%"
)

m3.metric(
    "LONG",
    f"{p_long:.1f}%"
)

m4.metric(
    "SHORT",
    f"{p_short:.1f}%"
)

m5.metric(
    "FINAL",
    final_signal
)


# ============================================================
# ORDER BOOK METRICS
# ============================================================

st.markdown(
    "### 📚 Order Book / Market State"
)

a1, a2, a3, a4, a5 = st.columns(5)

a1.metric(
    "OBI Top20",
    f"{obi20:.3f}"
)

a2.metric(
    "OBI Top50",
    f"{obi50:.3f}"
)

a3.metric(
    "OFI",
    f"{ofi:.3f}"
)

a4.metric(
    "Trend 10",
    f"{trend10:.4f}"
)

a5.metric(
    "Trend 20",
    f"{trend20:.4f}"
)


# ============================================================
# TRADE SETUP
# ============================================================

st.markdown(
    "### 🎯 Trade Setup"
)

t1, t2, t3, t4 = st.columns(4)

t1.metric(
    "Entry",
    f"{entry_price:.2f}"
)

t2.metric(
    "Stop Loss",
    f"{stop_loss:.2f}"
)

t3.metric(
    "TP1",
    f"{tp1:.2f}"
)

t4.metric(
    "TP2",
    f"{tp2:.2f}"
)


st.info(
    signal_reason
)


# ============================================================
# FILTER TABLE
# ============================================================

st.markdown(
    "### 🔎 Signal Filters"
)

filter_data = pd.DataFrame(
    [
        [
            "XGBoost Confidence",
            f"{confidence:.2f}%",
            confidence >= MIN_CONFIDENCE
        ],
        [
            "OBI Top20",
            f"{obi20:.4f}",
            abs(obi20) >= MIN_OBI
        ],
        [
            "OBI Top50",
            f"{obi50:.4f}",
            abs(obi50) >= MIN_OBI
        ],
        [
            "OFI",
            f"{ofi:.4f}",
            abs(ofi) >= MIN_OFI
        ],
        [
            "LONG Confirmations",
            f"{long_confirmations}/5",
            long_confirmations >= 3
        ],
        [
            "SHORT Confirmations",
            f"{short_confirmations}/5",
            short_confirmations >= 3
        ],
        [
            "Spread",
            f"{spread_pct * 100:.5f}%",
            spread_pct <= MAX_SPREAD_PCT
        ]
    ],
    columns=[
        "Filter",
        "Value",
        "Passed"
    ]
)

st.dataframe(
    filter_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# TRADE ID
# ============================================================

bucket_seconds = {
    "1m": 60,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400
}

bucket = int(
    time.time()
    /
    bucket_seconds.get(
        timeframe,
        900
    )
)

trade_id = (
    f"{symbol}_"
    f"{timeframe}_"
    f"{bucket}_"
    f"{final_signal}"
)


# ============================================================
# CREATE PAPER TRADE
# ============================================================

if (
    paper_trading
    and
    final_signal in [
        "LONG",
        "SHORT"
    ]
):

    existing_ids = {
        trade.get(
            "trade_id"
        )
        for trade in
        st.session_state.trade_history
    }

    if trade_id not in existing_ids:

        new_trade = {

            "trade_id":
                trade_id,

            "timestamp":
                datetime.datetime.now()
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "symbol":
                symbol,

            "timeframe":
                timeframe,

            "direction":
                final_signal,

            "entry_price":
                round(
                    entry_price,
                    2
                ),

            "stop_loss":
                round(
                    stop_loss,
                    2
                ),

            "tp1":
                round(
                    tp1,
                    2
                ),

            "tp2":
                round(
                    tp2,
                    2
                ),

            "exit_price":
                round(
                    entry_price,
                    2
                ),

            "confidence":
                round(
                    confidence,
                    2
                ),

            "p_long":
                round(
                    p_long,
                    2
                ),

            "p_short":
                round(
                    p_short,
                    2
                ),

            "p_no_trade":
                round(
                    p_no_trade,
                    2
                ),

            "obi_top20":
                round(
                    obi20,
                    5
                ),

            "obi_top50":
                round(
                    obi50,
                    5
                ),

            "ofi":
                round(
                    ofi,
                    5
                ),

            "trend10":
                round(
                    trend10,
                    6
                ),

            "trend20":
                round(
                    trend20,
                    6
                ),

            "outcome":
                "PENDING",

            "pnl_percent":
                0.0,

            "status":
                "Open"
        }

        st.session_state.trade_history.insert(
            0,
            new_trade
        )

        save_history(
            st.session_state.trade_history
        )


# ============================================================
# UPDATE OPEN TRADES
# ============================================================

history_changed = False

current_high = float(
    df["High"].iloc[-1]
)

current_low = float(
    df["Low"].iloc[-1]
)


for trade in (
    st.session_state.trade_history
):

    if trade.get(
        "outcome"
    ) != "PENDING":
        continue

    if trade.get(
        "symbol"
    ) != symbol:
        continue

    if trade.get(
        "timeframe"
    ) != timeframe:
        continue

    direction = trade[
        "direction"
    ]

    entry = float(
        trade["entry_price"]
    )

    sl = float(
        trade["stop_loss"]
    )

    target = float(
        trade["tp1"]
    )


    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if direction == "LONG":

        sl_hit = (
            current_low <= sl
        )

        tp_hit = (
            current_high >= target
        )

        if sl_hit and tp_hit:

            # Conservative assumption
            outcome = "LOSS"
            exit_price = sl

        elif tp_hit:

            outcome = "WIN"
            exit_price = target

        elif sl_hit:

            outcome = "LOSS"
            exit_price = sl

        else:

            continue

        pnl_percent = (
            (
                exit_price - entry
            )
            /
            (
                entry + 1e-12
            )
            *
            100
        )


    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    elif direction == "SHORT":

        sl_hit = (
            current_high >= sl
        )

        tp_hit = (
            current_low <= target
        )

        if sl_hit and tp_hit:

            # Conservative assumption
            outcome = "LOSS"
            exit_price = sl

        elif tp_hit:

            outcome = "WIN"
            exit_price = target

        elif sl_hit:

            outcome = "LOSS"
            exit_price = sl

        else:

            continue

        pnl_percent = (
            (
                entry - exit_price
            )
            /
            (
                entry + 1e-12
            )
            *
            100
        )

    else:

        continue


    trade["outcome"] = outcome

    trade["exit_price"] = round(
        exit_price,
        2
    )

    trade["pnl_percent"] = round(
        pnl_percent,
        3
    )

    trade["status"] = "Closed"

    history_changed = True


if history_changed:

    save_history(
        st.session_state.trade_history
    )


# ============================================================
# PERFORMANCE
# ============================================================

st.markdown("---")

st.header(
    "📈 Paper Trading Performance"
)

history_df = pd.DataFrame(
    st.session_state.trade_history
)


if history_df.empty:

    st.info(
        "No paper trades yet."
    )

else:

    closed = history_df[
        history_df["outcome"]
        !=
        "PENDING"
    ]

    pending = history_df[
        history_df["outcome"]
        ==
        "PENDING"
    ]

    wins = closed[
        closed["outcome"]
        ==
        "WIN"
    ]

    losses = closed[
        closed["outcome"]
        ==
        "LOSS"
    ]

    total_closed = len(
        closed
    )

    win_rate = (
        len(wins)
        /
        total_closed
        *
        100
        if total_closed > 0
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
        gross_profit
        /
        gross_loss
        if gross_loss > 0
        else 0
    )

    net_pnl = (
        closed["pnl_percent"].sum()
        if not closed.empty
        else 0
    )


    p1, p2, p3, p4, p5 = st.columns(5)

    p1.metric(
        "Win Rate",
        f"{win_rate:.1f}%"
    )

    p2.metric(
        "Closed",
        total_closed
    )

    p3.metric(
        "Wins / Losses",
        f"{len(wins)} / {len(losses)}"
    )

    p4.metric(
        "Profit Factor",
        f"{profit_factor:.2f}"
    )

    p5.metric(
        "Net PnL",
        f"{net_pnl:.2f}%"
    )


    # ========================================================
    # TRADE TABLE
    # ========================================================

    st.markdown(
        "### 📋 Trade History"
    )

    display_columns = [
        "timestamp",
        "symbol",
        "timeframe",
        "direction",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "exit_price",
        "confidence",
        "p_long",
        "p_short",
        "obi_top20",
        "obi_top50",
        "ofi",
        "outcome",
        "pnl_percent",
        "status"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in history_df.columns
    ]

    st.dataframe(
        history_df[
            available_columns
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    f"Last update: "
    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
