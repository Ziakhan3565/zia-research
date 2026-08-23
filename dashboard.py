import datetime
import os
import time
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from streamlit_autorefresh import st_autorefresh


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Quant Research Trading Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(
    interval=5000,
    limit=None,
    key="quant_terminal_refresh"
)

CSV_FILE = "signal_history.csv"
MODEL_PATH = "xgboost_obi_model.pkl"
FEEDBACK_FILE = "xgb_trade_feedback.csv"

MIN_FEEDBACK_TO_RETRAIN = 30
RETRAIN_EVERY = 10
MIN_TEST_ACCURACY = 0.55


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top right, rgba(30,41,59,.30), transparent 35%),
        #070b12;
    color: #e5e7eb;
}

section[data-testid="stSidebar"] {
    background: #0b1018 !important;
    border-right: 1px solid #1e293b;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1700px;
}

div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #243044;
    border-radius: 12px;
    padding: 12px;
}

.metric-card {
    background: linear-gradient(145deg,#111827,#0d131d);
    border: 1px solid #243044;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}

.metric-title {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .8px;
}

.metric-value {
    font-size: 23px;
    font-weight: 800;
    margin-top: 5px;
}

.metric-sub {
    color: #64748b;
    font-size: 11px;
    margin-top: 5px;
}

.signal-card {
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    margin-bottom: 15px;
    background: linear-gradient(145deg,#111827,#0b111b);
    border: 1px solid #263449;
}

.signal-title {
    font-size: 11px;
    color: #94a3b8;
    font-weight: 800;
    letter-spacing: 2px;
}

.signal-value {
    font-size: 40px;
    font-weight: 900;
    margin: 7px 0;
}

.signal-info {
    font-size: 13px;
    color: #cbd5e1;
}

.section-title {
    font-size: 18px;
    font-weight: 800;
    margin-top: 8px;
    margin-bottom: 10px;
}

.small-box {
    background: #0e1520;
    border: 1px solid #202c3e;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}

hr {
    border-color: #1e293b !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HISTORY
# ============================================================

EXPECTED_COLS = [
    "trade_id",
    "timestamp",
    "symbol",
    "timeframe",
    "direction",
    "signal_strength",
    "entry_price",
    "stop_loss",
    "tp1",
    "tp2",
    "rr_target",
    "exit_price",
    "confidence",
    "xgb_confidence",
    "xgb_features_json",
    "final_score",
    "outcome",
    "pnl_percent",
    "duration",
    "status",
    "exit_reason",
    "entry_candle_time",
    "exit_time",
]


def load_persistent_history():

    if not os.path.exists(CSV_FILE):
        return []

    try:
        df = pd.read_csv(CSV_FILE)

        for col in EXPECTED_COLS:
            if col not in df.columns:
                if col == "outcome":
                    df[col] = "PENDING"
                elif col == "rr_target":
                    df[col] = "TP1 1:2 | TP2 1:3"
                else:
                    df[col] = ""

        return df.to_dict("records")

    except Exception:
        return []


def save_persistent_history(history):

    try:
        if history:
            pd.DataFrame(history).to_csv(
                CSV_FILE,
                index=False
            )
    except Exception:
        pass


if "trade_history_log" not in st.session_state:
    st.session_state.trade_history_log = load_persistent_history()


# ============================================================
# NORMALIZE HISTORY
# ============================================================

def normalize_trade(trade):

    trade["outcome"] = str(
        trade.get("outcome", "PENDING")
    ).upper()

    trade["status"] = (
        "Closed"
        if trade["outcome"] in ["WIN", "LOSS"]
        else "Open"
    )

    numeric_fields = [
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "exit_price",
        "confidence",
        "xgb_confidence",
        "final_score",
        "pnl_percent",
    ]

    for field in numeric_fields:
        try:
            trade[field] = float(
                trade.get(field, 0)
            )
        except Exception:
            trade[field] = 0.0

    trade.setdefault(
        "signal_strength",
        trade.get("direction", "WAIT")
    )

    trade.setdefault(
        "rr_target",
        "TP1 1:2 | TP2 1:3"
    )

    trade.setdefault("exit_reason", "")

    trade.setdefault(
        "entry_candle_time",
        trade.get("timestamp", "")
    )

    trade.setdefault("exit_time", "")

    trade.setdefault(
        "xgb_features_json",
        ""
    )

    return trade


st.session_state.trade_history_log = [
    normalize_trade(x)
    for x in st.session_state.trade_history_log
]


# ============================================================
# XGBOOST
# ============================================================

XGB_FEATURES = [
    "top20_bid_sum",
    "top20_ask_sum",
    "obi_top20",
    "spread",
    "bid_ask_ratio",
    "total_depth",
    "trend_signal",
]


@st.cache_resource
def load_xgb_model():

    if not os.path.exists(MODEL_PATH):
        return None, "Model not found"

    try:
        model = joblib.load(MODEL_PATH)
        return model, None

    except Exception as e:
        return None, str(e)


xgb_model, xgb_model_error = load_xgb_model()


def load_feedback():

    if not os.path.exists(FEEDBACK_FILE):

        return pd.DataFrame(
            columns=XGB_FEATURES +
            ["target", "trade_id", "closed_at"]
        )

    try:

        df = pd.read_csv(FEEDBACK_FILE)

        required = XGB_FEATURES + ["target"]

        if not all(x in df.columns for x in required):
            return pd.DataFrame(
                columns=XGB_FEATURES +
                ["target", "trade_id", "closed_at"]
            )

        return df.dropna(
            subset=required
        )

    except Exception:
        return pd.DataFrame(
            columns=XGB_FEATURES +
            ["target", "trade_id", "closed_at"]
        )


def append_feedback(trade):

    outcome = str(
        trade.get("outcome", "")
    ).upper()

    if outcome not in ["WIN", "LOSS"]:
        return

    raw = trade.get(
        "xgb_features_json",
        ""
    )

    if not raw:
        return

    try:

        features = json.loads(raw)

        row = {
            k: float(features[k])
            for k in XGB_FEATURES
        }

        direction = str(
            trade.get("direction", "")
        ).upper()

        row["target"] = int(
            (direction == "LONG") ==
            (outcome == "WIN")
        )

        row["trade_id"] = trade.get(
            "trade_id",
            ""
        )

        row["closed_at"] = trade.get(
            "exit_time",
            datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        fb = load_feedback()

        if str(row["trade_id"]) in set(
            fb.get(
                "trade_id",
                pd.Series(dtype=str)
            ).astype(str)
        ):
            return

        fb = pd.concat(
            [
                fb,
                pd.DataFrame([row])
            ],
            ignore_index=True
        )

        fb.to_csv(
            FEEDBACK_FILE,
            index=False
        )

    except Exception:
        pass


# ============================================================
# DATA
# ============================================================

COINS_LIST = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
]


TIMEFRAME_MAP = {
    "1m (Scalping)": ("1m", 1),
    "15m (Medium TF)": ("15m", 15),
    "30m (Medium TF)": ("30m", 30),
    "1h (Intraday)": ("1h", 60),
    "4h (Intraday)": ("4h", 240),
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ⚡ QUANT TERMINAL")

selected_symbol = st.sidebar.selectbox(
    "Cryptocurrency",
    COINS_LIST
)

selected_tf_label = st.sidebar.selectbox(
    "Timeframe",
    list(TIMEFRAME_MAP.keys()),
    index=1
)

forecast_horizon = st.sidebar.slider(
    "Forecast Candles",
    5,
    30,
    15
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🎯 Risk / Reward")

st.sidebar.success(
    "TP1 = 1:2\n\nTP2 = 1:3"
)

st.sidebar.markdown("---")

paper_trading_mode = st.sidebar.toggle(
    "Paper Trading",
    value=True
)

if xgb_model is not None:
    st.sidebar.success(
        "XGBoost: LOADED"
    )
else:
    st.sidebar.error(
        "XGBoost: NOT LOADED"
    )

feedback_count = len(
    load_feedback()
)

st.sidebar.caption(
    f"Learning trades: {feedback_count}"
)

api_interval, tf_minutes = TIMEFRAME_MAP[
    selected_tf_label
]


# ============================================================
# BINANCE DATA
# ============================================================

@st.cache_data(ttl=10)
def fetch_klines_data(
    symbol,
    tf_key,
    limit=150
):

    interval = (
        "1m" if "1m" in tf_key else
        "15m" if "15m" in tf_key else
        "30m" if "30m" in tf_key else
        "1h" if "1h" in tf_key else
        "4h"
    )

    url = (
        "https://data-api.binance.vision"
        f"/api/v3/klines?symbol={symbol}"
        f"&interval={interval}&limit={limit}"
    )

    try:

        response = requests.get(
            url,
            timeout=5
        )

        data = response.json()

        if not isinstance(data, list):
            raise ValueError()

        df = pd.DataFrame(
            data,
            columns=[
                "Open_Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close_Time",
                "QAV",
                "NAT",
                "TBBAV",
                "TBQAV",
                "Ignore",
            ]
        )

        df["Time"] = pd.to_datetime(
            df["Open_Time"],
            unit="ms"
        )

        for c in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]:
            df[c] = df[c].astype(float)

        return df[
            [
                "Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        ]

    except Exception:

        return pd.DataFrame()


@st.cache_data(ttl=5)
def fetch_order_book_depth(
    symbol,
    depth_limit=20
):

    url = (
        "https://data-api.binance.vision"
        f"/api/v3/depth?symbol={symbol}"
        f"&limit={depth_limit}"
    )

    try:

        response = requests.get(
            url,
            timeout=5
        )

        data = response.json()

        bids = np.array(
            data["bids"],
            dtype=float
        )

        asks = np.array(
            data["asks"],
            dtype=float
        )

        return bids, asks

    except Exception:

        return (
            np.empty((0, 2)),
            np.empty((0, 2))
        )


df = fetch_klines_data(
    selected_symbol,
    selected_tf_label
)

bids, asks = fetch_order_book_depth(
    selected_symbol
)


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def build_xgb_features(
    df,
    bids,
    asks
):

    bid_sum = float(
        np.sum(bids[:, 1])
    ) if len(bids) else 0

    ask_sum = float(
        np.sum(asks[:, 1])
    ) if len(asks) else 0

    obi = (
        (bid_sum - ask_sum) /
        (bid_sum + ask_sum + 1e-8)
    )

    spread = (
        abs(
            asks[0, 0] -
            bids[0, 0]
        )
        if len(bids) and len(asks)
        else 0
    )

    ratio = (
        bid_sum /
        (ask_sum + 1e-8)
    )

    total_depth = (
        bid_sum + ask_sum
    )

    sma20 = (
        df["Close"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    trend_signal = (
        float(df["Close"].iloc[-1] - sma20)
    )

    return pd.DataFrame(
        [{
            "top20_bid_sum": bid_sum,
            "top20_ask_sum": ask_sum,
            "obi_top20": obi,
            "spread": spread,
            "bid_ask_ratio": ratio,
            "total_depth": total_depth,
            "trend_signal": trend_signal,
        }],
        columns=XGB_FEATURES
    )


def calculate_ofi(
    current_bids,
    current_asks
):

    current_bid = (
        float(np.sum(current_bids[:, 1]))
        if len(current_bids)
        else 0
    )

    current_ask = (
        float(np.sum(current_asks[:, 1]))
        if len(current_asks)
        else 0
    )

    previous = st.session_state.get(
        "previous_orderbook"
    )

    if previous is None:
        ofi = 0

    else:

        previous_bid, previous_ask = previous

        ofi = (
            current_bid -
            previous_bid
        ) - (
            current_ask -
            previous_ask
        )

    st.session_state.previous_orderbook = (
        current_bid,
        current_ask
    )

    return float(ofi)


# ============================================================
# RESEARCH ENGINE
# ============================================================

class TenPaperResearchLab:

    def __init__(self):

        self.feature_names = [
            "HAWKES",
            "BOOK_IMB",
            "TAKER_FLOW",
            "QUANT_IMPLY",
            "BAYESIAN",
            "QUANTILES",
            "TARGET_INV",
            "ADAPT_CONF",
            "FRAC_KELLY",
            "RMT_DOM",
            "CONF_CROSS",
            "REWARD_RISK",
        ]

        self.dynamic_weights = {
            k: 1 / len(self.feature_names)
            for k in self.feature_names
        }

    def extract_features(
        self,
        df,
        bids,
        asks
    ):

        result = {
            k: 0.0
            for k in self.feature_names
        }

        if (
            df.empty or
            len(df) < 20 or
            not len(bids) or
            not len(asks)
        ):
            return result

        bid = np.sum(bids[:, 1])
        ask = np.sum(asks[:, 1])

        returns = (
            df["Close"]
            .pct_change()
            .dropna()
        )

        realized_vol = (
            returns.std() + 1e-8
        )

        momentum = (
            df["Close"].iloc[-1] /
            df["Close"].iloc[-6] -
            1
        )

        delta = (
            df["Close"].iloc[-1] -
            df["Close"].iloc[-2]
        )

        result["BOOK_IMB"] = np.clip(
            (bid - ask) /
            (bid + ask + 1e-8),
            -1,
            1
        )

        result["TAKER_FLOW"] = (
            1 if delta > 0 else -1
        ) * min(
            abs(delta) /
            (df["Close"].iloc[-1] *
             realized_vol + 1e-8),
            1
        )

        result["QUANT_IMPLY"] = np.clip(
            (
                bids[0, 1] -
                asks[0, 1]
            ) /
            (
                bids[0, 1] +
                asks[0, 1] +
                1e-8
            ),
            -1,
            1
        )

        result["HAWKES"] = np.clip(
            np.sign(momentum) *
            (
                df["Volume"].iloc[-3:].mean() /
                (
                    df["Volume"].iloc[-15:].mean()
                    + 1e-8
                ) - 1
            ),
            -1,
            1
        )

        result["BAYESIAN"] = (
            result["BOOK_IMB"] * 0.8
        )

        q90 = returns.quantile(0.90)
        q10 = returns.quantile(0.10)

        result["QUANTILES"] = np.clip(
            momentum /
            (
                abs(q90) +
                abs(q10) +
                1e-8
            ),
            -1,
            1
        )

        result["TARGET_INV"] = (
            1 if delta > 0 else
            -1 if delta < 0 else
            0
        )

        ema9 = (
            df["Close"]
            .ewm(span=9)
            .mean()
            .iloc[-1]
        )

        ema21 = (
            df["Close"]
            .ewm(span=21)
            .mean()
            .iloc[-1]
        )

        result["ADAPT_CONF"] = np.clip(
            (
                ema9 - ema21
            ) /
            (
                df["Close"].iloc[-1] *
                realized_vol +
                1e-8
            ),
            -1,
            1
        )

        result["FRAC_KELLY"] = (
            np.sign(momentum) *
            min(abs(momentum) * 100, 1)
        )

        result["RMT_DOM"] = np.clip(
            momentum /
            (
                realized_vol *
                np.sqrt(5) +
                1e-8
            ),
            -1,
            1
        )

        result["CONF_CROSS"] = np.sign(
            ema9 - ema21
        )

        rr = (
            abs(q90) /
            (abs(q10) + 1e-8)
        )

        result["REWARD_RISK"] = (
            1 if rr >= 1.2 else
            -1 if rr < .8 else
            0
        )

        return result

    def calculate(
        self,
        df,
        bids,
        asks
    ):

        features = self.extract_features(
            df,
            bids,
            asks
        )

        score = float(
            np.mean(
                list(features.values())
            )
        )

        return (
            features,
            np.clip(score, -1, 1),
            self.dynamic_weights
        )


# ============================================================
# MAIN ENGINE
# ============================================================

if (
    not df.empty and
    len(df) >= 20 and
    len(bids) and
    len(asks)
):

    close_price = float(
        df["Close"].iloc[-1]
    )

    candle_time = pd.Timestamp(
        df["Time"].iloc[-1]
    )

    # --------------------------------------------------------
    # RESEARCH
    # --------------------------------------------------------

    research = TenPaperResearchLab()

    paper_results, research_score, weights = (
        research.calculate(
            df,
            bids,
            asks
        )
    )

    research_direction = (
        "LONG"
        if research_score >= 0.12
        else
        "SHORT"
        if research_score <= -0.12
        else
        "NEUTRAL"
    )

    # --------------------------------------------------------
    # ORDER BOOK
    # --------------------------------------------------------

    bid_sum = float(
        np.sum(bids[:, 1])
    )

    ask_sum = float(
        np.sum(asks[:, 1])
    )

    obi = (
        (bid_sum - ask_sum) /
        (bid_sum + ask_sum + 1e-8)
    )

    ofi = calculate_ofi(
        bids,
        asks
    )

    total_depth = (
        bid_sum + ask_sum
    )

    ofi_norm = np.clip(
        ofi /
        max(total_depth, 1),
        -1,
        1
    )

    micro_score = np.clip(
        .65 * obi +
        .35 * ofi_norm,
        -1,
        1
    )

    micro_direction = (
        "LONG"
        if micro_score >= .08
        else
        "SHORT"
        if micro_score <= -.08
        else
        "NEUTRAL"
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    ema9 = (
        df["Close"]
        .ewm(span=9)
        .mean()
        .iloc[-1]
    )

    ema21 = (
        df["Close"]
        .ewm(span=21)
        .mean()
        .iloc[-1]
    )

    ema50 = (
        df["Close"]
        .ewm(span=50)
        .mean()
        .iloc[-1]
    )

    momentum = (
        df["Close"].iloc[-1] /
        df["Close"].iloc[-6] -
        1
    )

    trend_score = np.clip(
        (
            (ema9 - ema21) /
            (close_price + 1e-8)
        ) * 250
        +
        np.sign(momentum) *
        min(abs(momentum) * 1000, .5),
        -1,
        1
    )

    trend_direction = (
        "LONG"
        if trend_score >= .10
        else
        "SHORT"
        if trend_score <= -.10
        else
        "NEUTRAL"
    )

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    xgb_features = build_xgb_features(
        df,
        bids,
        asks
    )

    xgb_signal = "NEUTRAL"
    xgb_confidence = 0.0

    if xgb_model is not None:

        try:

            prediction = int(
                xgb_model.predict(
                    xgb_features
                )[0]
            )

            probabilities = (
                xgb_model
                .predict_proba(
                    xgb_features
                )[0]
            )

            xgb_confidence = (
                float(
                    np.max(probabilities)
                ) * 100
            )

            xgb_signal = (
                "LONG"
                if prediction == 1
                else
                "SHORT"
            )

        except Exception:
            xgb_signal = "NEUTRAL"

    # --------------------------------------------------------
    # COMBINED SCORE
    # --------------------------------------------------------

    xgb_signed = (
        xgb_confidence / 100
        *
        (
            1
            if xgb_signal == "LONG"
            else -1
        )
        if xgb_signal != "NEUTRAL"
        else 0
    )

    combined_score = np.clip(
        .45 * xgb_signed +
        .25 * research_score +
        .20 * micro_score +
        .10 * trend_score,
        -1,
        1
    )

    votes = [
        xgb_signal,
        research_direction,
        micro_direction,
        trend_direction,
    ]

    long_votes = votes.count("LONG")
    short_votes = votes.count("SHORT")

    # ========================================================
    # SIGNAL CONDITIONS
    # ========================================================

    strong_long = (
        xgb_signal == "LONG"
        and xgb_confidence >= 80
        and long_votes >= 4
        and research_score >= .15
        and micro_score >= .12
        and trend_score >= .10
        and combined_score >= .45
    )

    strong_short = (
        xgb_signal == "SHORT"
        and xgb_confidence >= 80
        and short_votes >= 4
        and research_score <= -.15
        and micro_score <= -.12
        and trend_score <= -.10
        and combined_score <= -.45
    )

    normal_long = (
        xgb_signal == "LONG"
        and xgb_confidence >= 65
        and long_votes >= 3
        and trend_direction != "SHORT"
        and combined_score >= .18
    )

    normal_short = (
        xgb_signal == "SHORT"
        and xgb_confidence >= 65
        and short_votes >= 3
        and trend_direction != "LONG"
        and combined_score <= -.18
    )

    if strong_long:

        direction = "LONG"
        signal_strength = "STRONG LONG"

    elif strong_short:

        direction = "SHORT"
        signal_strength = "STRONG SHORT"

    elif normal_long:

        direction = "LONG"
        signal_strength = "LONG"

    elif normal_short:

        direction = "SHORT"
        signal_strength = "SHORT"

    else:

        direction = "NEUTRAL"
        signal_strength = "WAIT"

    confidence = int(
        np.clip(
            abs(combined_score) * 100,
            0,
            99
        )
    )

    # ========================================================
    # COLORS
    # ========================================================

    if signal_strength == "STRONG LONG":
        signal_color = "#00ff88"

    elif signal_strength == "LONG":
        signal_color = "#22c55e"

    elif signal_strength == "STRONG SHORT":
        signal_color = "#ff1744"

    elif signal_strength == "SHORT":
        signal_color = "#ef4444"

    else:
        signal_color = "#38bdf8"

    # ========================================================
    # ATR + TARGETS
    # ========================================================

    atr = (
        df["High"] -
        df["Low"]
    ).rolling(14).mean().iloc[-1]

    if (
        pd.isna(atr)
        or atr <= 0
    ):
        atr = close_price * .005

    risk_distance = max(
        float(atr),
        close_price * .001
    )

    if direction == "LONG":

        sl = close_price - risk_distance
        tp1 = close_price + risk_distance * 2
        tp2 = close_price + risk_distance * 3

    elif direction == "SHORT":

        sl = close_price + risk_distance
        tp1 = close_price - risk_distance * 2
        tp2 = close_price - risk_distance * 3

    else:

        sl = close_price - risk_distance
        tp1 = close_price + risk_distance * 2
        tp2 = close_price + risk_distance * 3

    # ========================================================
    # PAPER TRADE
    # ========================================================

    lock_seconds = tf_minutes * 60

    current_sec = int(time.time())

    bucket = (
        current_sec -
        current_sec % lock_seconds
    )

    trade_id = (
        f"{selected_symbol}_"
        f"{selected_tf_label}_"
        f"{bucket}_"
        f"{direction}"
    )

    if (
        paper_trading_mode
        and direction != "NEUTRAL"
    ):

        existing = {
            x.get("trade_id")
            for x in
            st.session_state.trade_history_log
        }

        if trade_id not in existing:

            trade = {

                "trade_id": trade_id,

                "timestamp":
                    datetime.datetime.now()
                    .strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "entry_candle_time":
                    candle_time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "symbol":
                    selected_symbol,

                "timeframe":
                    selected_tf_label,

                "direction":
                    direction,

                "signal_strength":
                    signal_strength,

                "entry_price":
                    round(close_price, 2),

                "stop_loss":
                    round(sl, 2),

                "tp1":
                    round(tp1, 2),

                "tp2":
                    round(tp2, 2),

                "rr_target":
                    "TP1 1:2 | TP2 1:3",

                "exit_price":
                    0,

                "confidence":
                    confidence,

                "xgb_confidence":
                    round(
                        xgb_confidence,
                        2
                    ),

                "xgb_features_json":
                    json.dumps({
                        k: float(
                            xgb_features.iloc[0][k]
                        )
                        for k in XGB_FEATURES
                    }),

                "final_score":
                    round(
                        combined_score,
                        4
                    ),

                "outcome":
                    "PENDING",

                "pnl_percent":
                    0,

                "duration":
                    "Active",

                "status":
                    "Open",

                "exit_reason":
                    "",

                "exit_time":
                    "",
            }

            st.session_state.trade_history_log.insert(
                0,
                trade
            )

            save_persistent_history(
                st.session_state.trade_history_log
            )

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        f"""
        <div class="signal-card"
             style="border:1px solid {signal_color};">

            <div class="signal-title">
                CURRENT TRADING SIGNAL
            </div>

            <div class="signal-value"
                 style="color:{signal_color};">
                {signal_strength}
            </div>

            <div class="signal-info">
                <b>{selected_symbol}</b>
                &nbsp; • &nbsp;
                {selected_tf_label}
                &nbsp; • &nbsp;
                Price ${close_price:,.2f}
            </div>

            <div style="
                margin-top:12px;
                color:#94a3b8;
                font-size:12px;
            ">
                Score
                <b>{combined_score:+.3f}</b>
                &nbsp; | &nbsp;

                Confidence
                <b>{confidence}%</b>
                &nbsp; | &nbsp;

                XGB
                <b>{xgb_signal}</b>
                ({xgb_confidence:.1f}%)
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # MAIN METRICS
    # ========================================================

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Direction
                </div>
                <div class="metric-value"
                     style="color:{signal_color};">
                    {direction}
                </div>
                <div class="metric-sub">
                    {long_votes} LONG /
                    {short_votes} SHORT
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Entry
                </div>
                <div class="metric-value">
                    ${close_price:,.2f}
                </div>
                <div class="metric-sub">
                    Current market price
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Stop Loss
                </div>
                <div class="metric-value"
                     style="color:#ef4444;">
                    ${sl:,.2f}
                </div>
                <div class="metric-sub">
                    Risk = ${risk_distance:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    TP1 — 1:2
                </div>
                <div class="metric-value"
                     style="color:#22c55e;">
                    ${tp1:,.2f}
                </div>
                <div class="metric-sub">
                    First target
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    TP2 — 1:3
                </div>
                <div class="metric-value"
                     style="color:#00e676;">
                    ${tp2:,.2f}
                </div>
                <div class="metric-sub">
                    Final target
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c6:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    Confidence
                </div>
                <div class="metric-value"
                     style="color:#38bdf8;">
                    {confidence}%
                </div>
                <div class="metric-sub">
                    Multi-factor score
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # CONFIRMATION PANEL
    # ========================================================

    st.markdown(
        '<div class="section-title">🔎 Signal Confirmation</div>',
        unsafe_allow_html=True
    )

    q1, q2, q3, q4, q5 = st.columns(5)

    def status_color(value, positive="LONG", negative="SHORT"):

        if value == positive:
            return "#22c55e"

        if value == negative:
            return "#ef4444"

        return "#38bdf8"

    with q1:

        st.markdown(
            f"""
            <div class="small-box">
                <div class="metric-title">
                    XGBoost
                </div>
                <b style="color:{status_color(xgb_signal)};">
                    {xgb_signal}
                </b>
                <br>
                <small>
                    {xgb_confidence:.1f}%
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q2:

        st.markdown(
            f"""
            <div class="small-box">
                <div class="metric-title">
                    Research
                </div>
                <b style="color:{status_color(research_direction)};">
                    {research_direction}
                </b>
                <br>
                <small>
                    {research_score:+.3f}
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q3:

        st.markdown(
            f"""
            <div class="small-box">
                <div class="metric-title">
                    OBI / OFI
                </div>
                <b style="color:{status_color(micro_direction)};">
                    {micro_direction}
                </b>
                <br>
                <small>
                    OBI {obi:+.3f}
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q4:

        st.markdown(
            f"""
            <div class="small-box">
                <div class="metric-title">
                    Trend
                </div>
                <b style="color:{status_color(trend_direction)};">
                    {trend_direction}
                </b>
                <br>
                <small>
                    {trend_score:+.3f}
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q5:

        st.markdown(
            f"""
            <div class="small-box">
                <div class="metric-title">
                    Combined
                </div>
                <b style="color:{signal_color};">
                    {combined_score:+.3f}
                </b>
                <br>
                <small>
                    Final score
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # CHART
    # ========================================================

    st.markdown("---")

    chart_col, book_col = st.columns(
        [2.7, 1]
    )

    with chart_col:

        st.markdown(
            '<div class="section-title">📈 Price & Trading Levels</div>',
            unsafe_allow_html=True
        )

        future_times = [
            df["Time"].iloc[-1]
            + pd.Timedelta(
                minutes=tf_minutes * i
            )
            for i in range(
                1,
                forecast_horizon + 1
            )
        ]

        steps = np.linspace(
            0,
            np.pi / 2,
            forecast_horizon
        )

        if direction == "LONG":

            forecast = (
                close_price +
                (tp2 - close_price)
                * np.sin(steps)
            )

        elif direction == "SHORT":

            forecast = (
                close_price -
                (close_price - tp2)
                * np.sin(steps)
            )

        else:

            forecast = np.repeat(
                close_price,
                forecast_horizon
            )

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df["Time"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    df["Time"].iloc[-1]
                ] + future_times,
                y=[
                    close_price
                ] + list(forecast),
                mode="lines+markers",
                name="Signal Path",
                line=dict(
                    color=signal_color,
                    width=3,
                    dash="dot"
                )
            )
        )

        fig.add_hline(
            y=close_price,
            line_dash="dot",
            annotation_text="ENTRY"
        )

        fig.add_hline(
            y=sl,
            line_dash="dash",
            annotation_text="SL"
        )

        fig.add_hline(
            y=tp1,
            line_dash="dash",
            annotation_text="TP1 1:2"
        )

        fig.add_hline(
            y=tp2,
            line_dash="dash",
            annotation_text="TP2 1:3"
        )

        fig.update_layout(
            template="plotly_dark",
            height=520,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#0d131d",
            plot_bgcolor="#0d131d",
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),
            legend=dict(
                orientation="h"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    # ========================================================
    # ORDER BOOK PANEL
    # ========================================================

    with book_col:

        st.markdown(
            '<div class="section-title">📚 Order Book</div>',
            unsafe_allow_html=True
        )

        spread = (
            abs(
                asks[0, 0] -
                bids[0, 0]
            )
        )

        items = [
            (
                "Bid Volume",
                f"{bid_sum:,.2f}",
                "#22c55e"
            ),
            (
                "Ask Volume",
                f"{ask_sum:,.2f}",
                "#ef4444"
            ),
            (
                "OBI",
                f"{obi:+.3f}",
                "#38bdf8"
            ),
            (
                "OFI",
                f"{ofi:+.2f}",
                "#38bdf8"
            ),
            (
                "Spread",
                f"${spread:.2f}",
                "#cbd5e1"
            ),
        ]

        for title, value, color in items:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        {title}
                    </div>
                    <div class="metric-value"
                         style="color:{color};">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # OBI bars

        levels = [5, 10, 20]

        obi_values = []

        for level in levels:

            b = np.sum(
                bids[:level, 1]
            )

            a = np.sum(
                asks[:level, 1]
            )

            obi_level = (
                (b - a) /
                (b + a + 1e-8)
            )

            obi_values.append(
                obi_level
            )

        fig_obi = go.Figure(
            go.Bar(
                x=[
                    "TOP 5",
                    "TOP 10",
                    "TOP 20"
                ],
                y=obi_values
            )
        )

        fig_obi.update_layout(
            height=230,
            template="plotly_dark",
            margin=dict(
                l=5,
                r=5,
                t=5,
                b=5
            ),
            paper_bgcolor="#0d131d",
            plot_bgcolor="#0d131d"
        )

        st.plotly_chart(
            fig_obi,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    # ========================================================
    # RESEARCH SCOREBOARD
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🔬 Quant Research Scoreboard</div>',
        unsafe_allow_html=True
    )

    research_rows = []

    for name, value in paper_results.items():

        status = (
            "BULLISH"
            if value > .10
            else
            "BEARISH"
            if value < -.10
            else
            "NEUTRAL"
        )

        research_rows.append({
            "Factor": name,
            "Score": round(
                float(value),
                3
            ),
            "Weight": f"{weights[name] * 100:.1f}%",
            "Status": status,
        })

    st.dataframe(
        pd.DataFrame(
            research_rows
        ),
        use_container_width=True,
        hide_index=True,
        height=300
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Paper Trading Performance</div>',
        unsafe_allow_html=True
    )

    history_df = pd.DataFrame(
        st.session_state.trade_history_log
    )

    if not history_df.empty:

        f1, f2, f3 = st.columns(3)

        with f1:

            coin_filter = st.selectbox(
                "Coin",
                ["ALL"] + COINS_LIST,
                key="perf_coin"
            )

        with f2:

            tf_filter = st.selectbox(
                "Timeframe",
                ["ALL"] +
                list(TIMEFRAME_MAP.keys()),
                key="perf_tf"
            )

        with f3:

            direction_filter = st.selectbox(
                "Direction",
                [
                    "ALL",
                    "LONG",
                    "SHORT"
                ],
                key="perf_direction"
            )

        filtered = history_df.copy()

        if coin_filter != "ALL":

            filtered = filtered[
                filtered["symbol"]
                == coin_filter
            ]

        if tf_filter != "ALL":

            filtered = filtered[
                filtered["timeframe"]
                == tf_filter
            ]

        if direction_filter != "ALL":

            filtered = filtered[
                filtered["direction"]
                == direction_filter
            ]

        wins = len(
            filtered[
                filtered["outcome"]
                == "WIN"
            ]
        )

        losses = len(
            filtered[
                filtered["outcome"]
                == "LOSS"
            ]
        )

        pending = len(
            filtered[
                filtered["outcome"]
                == "PENDING"
            ]
        )

        closed = wins + losses

        win_rate = (
            wins / closed * 100
            if closed
            else 0
        )

        gross_profit = (
            filtered[
                filtered["outcome"]
                == "WIN"
            ]["pnl_percent"]
            .sum()
        )

        gross_loss = abs(
            filtered[
                filtered["outcome"]
                == "LOSS"
            ]["pnl_percent"]
            .sum()
        )

        net_pnl = (
            gross_profit -
            gross_loss
        )

        profit_factor = (
            gross_profit /
            gross_loss
            if gross_loss > 0
            else 0
        )

        p1, p2, p3, p4, p5, p6 = st.columns(6)

        with p1:
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%"
            )

        with p2:
            st.metric(
                "Closed",
                closed
            )

        with p3:
            st.metric(
                "Wins",
                wins
            )

        with p4:
            st.metric(
                "Losses",
                losses
            )

        with p5:
            st.metric(
                "Profit Factor",
                f"{profit_factor:.2f}"
            )

        with p6:
            st.metric(
                "Net PnL",
                f"{net_pnl:+.2f}%"
            )

        st.markdown(
            "### Trade History"
        )

        display_cols = [
            "timestamp",
            "symbol",
            "timeframe",
            "direction",
            "signal_strength",
            "entry_price",
            "stop_loss",
            "tp1",
            "tp2",
            "rr_target",
            "exit_price",
            "confidence",
            "xgb_confidence",
            "outcome",
            "pnl_percent",
            "exit_reason",
        ]

        for col in display_cols:

            if col not in filtered.columns:

                filtered[col] = ""

        st.dataframe(
            filtered[
                display_cols
            ],
            use_container_width=True,
            hide_index=True,
            height=320
        )

    else:

        st.info(
            "No paper trades recorded yet."
        )


else:

    st.warning(
        "⚠️ Market data unavailable. "
        "Waiting for Binance data..."
    )
