import datetime
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Quant Research Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# AUTO REFRESH
# ============================================================

if st_autorefresh is not None:
    st_autorefresh(
        interval=5000,
        limit=None,
        key="quant_terminal_refresh",
    )


# ============================================================
# FILES
# ============================================================

CSV_FILE = "signal_history.csv"
FEEDBACK_FILE = "xgb_trade_feedback.csv"
MODEL_PATH = "xgboost_obi_model.pkl"


# ============================================================
# CONSTANTS
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

MIN_FEEDBACK_TO_RETRAIN = 30
RETRAIN_EVERY = 10
MIN_TEST_ACCURACY = 0.55


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
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(0, 180, 255, 0.07), transparent 28%),
        radial-gradient(circle at bottom left, rgba(0, 255, 150, 0.04), transparent 25%),
        #070a0f;
    color: #e6edf3;
}

section[data-testid="stSidebar"] {
    background: #0b0f15 !important;
    border-right: 1px solid #1d2633;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    letter-spacing: -0.02em;
}

.terminal-header {
    background: linear-gradient(
        135deg,
        #101722,
        #0d131c
    );
    border: 1px solid #202b3a;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 8px 30px rgba(0,0,0,.25);
}

.terminal-title {
    font-size: 24px;
    font-weight: 800;
    color: #f1f5f9;
}

.terminal-subtitle {
    color: #7f8ea3;
    font-size: 12px;
    margin-top: 3px;
}

.status-bar {
    background: #0e141d;
    border: 1px solid #202b3a;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 10px 0 16px 0;
    font-size: 12px;
}

.card {
    background: linear-gradient(145deg, #111823, #0e141d);
    border: 1px solid #202b3a;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 5px 18px rgba(0,0,0,.20);
}

.card-title {
    color: #8492a6;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 7px;
}

.card-value {
    font-size: 21px;
    font-weight: 800;
}

.card-small {
    font-size: 11px;
    color: #7f8ea3;
    margin-top: 5px;
}

.signal-card {
    background: linear-gradient(145deg, #121a25, #0c1219);
    border: 1px solid #273448;
    border-radius: 14px;
    padding: 20px;
    min-height: 190px;
}

.signal-label {
    color: #8190a5;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.signal-value {
    font-size: 32px;
    font-weight: 900;
    margin-top: 8px;
}

.green {
    color: #00e676;
}

.red {
    color: #ff5252;
}

.blue {
    color: #38bdf8;
}

.yellow {
    color: #fbbf24;
}

.muted {
    color: #7f8ea3;
}

.section-title {
    font-size: 17px;
    font-weight: 800;
    margin-top: 12px;
    margin-bottom: 10px;
}

.badge {
    display: inline-block;
    padding: 4px 9px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 800;
    border: 1px solid #263446;
    background: #111a25;
}

hr {
    border-color: #1c2734 !important;
}

div[data-testid="stMetric"] {
    background: #101721;
    border: 1px solid #202b3a;
    padding: 12px;
    border-radius: 10px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HISTORY
# ============================================================

EXPECTED_HISTORY_COLUMNS = [
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

        for col in EXPECTED_HISTORY_COLUMNS:

            if col not in df.columns:

                if col == "outcome":
                    df[col] = "PENDING"

                elif col in ["signal_strength", "rr_target", "exit_reason"]:
                    df[col] = ""

                else:
                    df[col] = 0.0

        return df.to_dict("records")

    except Exception:
        return []


def save_persistent_history(history):

    try:

        pd.DataFrame(history).to_csv(
            CSV_FILE,
            index=False,
        )

    except Exception as e:

        st.warning(
            f"Could not save trade history: {e}"
        )


if "trade_history_log" not in st.session_state:

    st.session_state.trade_history_log = (
        load_persistent_history()
    )


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_xgb_model():

    if not os.path.exists(MODEL_PATH):

        return None, (
            f"{MODEL_PATH} not found."
        )

    try:

        model = joblib.load(MODEL_PATH)

        if not hasattr(model, "predict"):

            return None, (
                "Model loaded but does not have predict()."
            )

        return model, None

    except Exception as e:

        return None, (
            f"XGBoost loading error: {e}"
        )


xgb_model, xgb_model_error = load_xgb_model()


# ============================================================
# FEEDBACK
# ============================================================

def empty_feedback():

    return pd.DataFrame(
        columns=XGB_FEATURES + [
            "target",
            "trade_id",
            "closed_at",
        ]
    )


def load_feedback():

    if not os.path.exists(FEEDBACK_FILE):
        return empty_feedback()

    try:

        fb = pd.read_csv(FEEDBACK_FILE)

        required = XGB_FEATURES + [
            "target",
            "trade_id",
            "closed_at",
        ]

        for col in required:

            if col not in fb.columns:
                return empty_feedback()

        fb = fb.dropna(
            subset=XGB_FEATURES + ["target"]
        )

        return fb

    except Exception:
        return empty_feedback()


def append_feedback(trade):

    outcome = str(
        trade.get("outcome", "")
    ).upper()

    if outcome not in ["WIN", "LOSS"]:
        return

    raw = trade.get(
        "xgb_features_json",
        "",
    )

    if not raw:
        return

    try:

        features = (
            json.loads(raw)
            if isinstance(raw, str)
            else raw
        )

        row = {}

        for feature in XGB_FEATURES:

            row[feature] = float(
                features[feature]
            )

        direction = str(
            trade.get("direction", "")
        ).upper()

        row["target"] = int(
            (direction == "LONG")
            == (outcome == "WIN")
        )

        row["trade_id"] = str(
            trade.get("trade_id", "")
        )

        row["closed_at"] = trade.get(
            "exit_time",
            datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        )

        fb = load_feedback()

        if (
            "trade_id" in fb.columns
            and str(row["trade_id"])
            in set(
                fb["trade_id"]
                .astype(str)
            )
        ):
            return

        fb = pd.concat(
            [
                fb,
                pd.DataFrame([row]),
            ],
            ignore_index=True,
        )

        fb.to_csv(
            FEEDBACK_FILE,
            index=False,
        )

    except Exception:
        pass


def retrain_xgb_from_feedback(
    current_model,
):

    fb = load_feedback()

    if len(fb) < MIN_FEEDBACK_TO_RETRAIN:
        return current_model, None

    if fb["target"].nunique() < 2:
        return current_model, None

    last_count = int(
        st.session_state.get(
            "xgb_last_retrain_count",
            0,
        )
    )

    if len(fb) < (
        last_count + RETRAIN_EVERY
    ):
        return current_model, None

    fb = fb.sort_values(
        "closed_at",
        kind="stable",
    )

    split = max(
        int(len(fb) * 0.80),
        1,
    )

    if split >= len(fb):
        return current_model, None

    train = fb.iloc[:split]
    test = fb.iloc[split:]

    if train["target"].nunique() < 2:
        return current_model, None

    if test["target"].nunique() < 2:
        return current_model, None

    candidate = XGBClassifier(
        n_estimators=180,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=2,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.5,
        random_state=42,
        eval_metric="logloss",
        n_jobs=2,
    )

    candidate.fit(
        train[XGB_FEATURES],
        train["target"].astype(int),
    )

    predictions = candidate.predict(
        test[XGB_FEATURES]
    )

    accuracy = float(
        accuracy_score(
            test["target"].astype(int),
            predictions,
        )
    )

    st.session_state.xgb_last_retrain_count = len(fb)

    if accuracy < MIN_TEST_ACCURACY:

        return current_model, (
            f"Retrain rejected | "
            f"holdout accuracy "
            f"{accuracy * 100:.1f}%"
        )

    temp_path = MODEL_PATH + ".tmp"

    joblib.dump(
        candidate,
        temp_path,
    )

    os.replace(
        temp_path,
        MODEL_PATH,
    )

    return candidate, (
        f"XGB retrained | "
        f"{len(fb)} trades | "
        f"holdout "
        f"{accuracy * 100:.1f}%"
    )


# ============================================================
# NORMALIZE TRADE
# ============================================================

def normalize_trade(trade):

    trade["outcome"] = str(
        trade.get(
            "outcome",
            "PENDING",
        )
    ).upper()

    trade["status"] = (
        "Closed"
        if trade["outcome"]
        in ["WIN", "LOSS"]
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

    for key in numeric_fields:

        try:

            trade[key] = float(
                trade.get(key, 0)
            )

        except Exception:

            trade[key] = 0.0

    if not trade.get("rr_target"):
        trade["rr_target"] = (
            "TP1 1:2 | TP2 1:3"
        )

    if not trade.get("entry_candle_time"):

        trade["entry_candle_time"] = (
            trade.get(
                "timestamp",
                "",
            )
        )

    if "xgb_features_json" not in trade:
        trade["xgb_features_json"] = ""

    if "signal_strength" not in trade:
        trade["signal_strength"] = ""

    return trade


st.session_state.trade_history_log = [
    normalize_trade(t)
    for t in st.session_state.trade_history_log
]


# ============================================================
# DATA FETCH
# ============================================================

@st.cache_data(ttl=10)
def fetch_klines_data(
    symbol,
    timeframe,
    limit=150,
):

    interval = timeframe

    url = (
        "https://data-api.binance.vision"
        "/api/v3/klines"
    )

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=5,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            raise ValueError(
                "Invalid Binance response"
            )

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
                "Trades",
                "TBBAV",
                "TBQAV",
                "Ignore",
            ],
        )

        df["Time"] = pd.to_datetime(
            df["Open_Time"],
            unit="ms",
        )

        for col in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
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

        return pd.DataFrame(
            columns=[
                "Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        )


@st.cache_data(ttl=5)
def fetch_order_book_depth(
    symbol,
    depth_limit=50,
):

    url = (
        "https://data-api.binance.vision"
        "/api/v3/depth"
    )

    params = {
        "symbol": symbol,
        "limit": depth_limit,
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=5,
        )

        response.raise_for_status()

        data = response.json()

        bids = np.array(
            data["bids"],
            dtype=float,
        )

        asks = np.array(
            data["asks"],
            dtype=float,
        )

        return bids, asks

    except Exception:

        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
        )


# ============================================================
# ORDER BOOK FUNCTIONS
# ============================================================

def calculate_obi(
    bids,
    asks,
    levels=20,
):

    if len(bids) == 0 or len(asks) == 0:
        return 0.0

    bid = float(
        np.sum(
            bids[:levels, 1]
        )
    )

    ask = float(
        np.sum(
            asks[:levels, 1]
        )
    )

    return (
        (bid - ask)
        / (bid + ask + 1e-12)
    )


def calculate_ofi(
    bids,
    asks,
):

    current_bid = (
        float(np.sum(bids[:, 1]))
        if len(bids)
        else 0.0
    )

    current_ask = (
        float(np.sum(asks[:, 1]))
        if len(asks)
        else 0.0
    )

    previous = st.session_state.get(
        "previous_orderbook",
        None,
    )

    if previous is None:

        ofi = 0.0

    else:

        previous_bid, previous_ask = (
            previous
        )

        ofi = (
            current_bid
            - previous_bid
            - (
                current_ask
                - previous_ask
            )
        )

    st.session_state.previous_orderbook = (
        current_bid,
        current_ask,
    )

    return float(ofi)


def orderbook_stats(
    bids,
    asks,
):

    result = {}

    for level in [
        5,
        10,
        20,
        50,
    ]:

        result[
            f"obi_{level}"
        ] = calculate_obi(
            bids,
            asks,
            level,
        )

    result["bid20"] = (
        float(
            np.sum(
                bids[:20, 1]
            )
        )
        if len(bids)
        else 0
    )

    result["ask20"] = (
        float(
            np.sum(
                asks[:20, 1]
            )
        )
        if len(asks)
        else 0
    )

    result["bid50"] = (
        float(
            np.sum(
                bids[:50, 1]
            )
        )
        if len(bids)
        else 0
    )

    result["ask50"] = (
        float(
            np.sum(
                asks[:50, 1]
            )
        )
        if len(asks)
        else 0
    )

    result["spread"] = (
        abs(
            asks[0, 0]
            - bids[0, 0]
        )
        if len(bids)
        and len(asks)
        else 0
    )

    return result


# ============================================================
# XGB FEATURES
# ============================================================

def build_xgb_features(
    df,
    bids,
    asks,
):

    bid20 = (
        float(
            np.sum(
                bids[:20, 1]
            )
        )
        if len(bids)
        else 0
    )

    ask20 = (
        float(
            np.sum(
                asks[:20, 1]
            )
        )
        if len(asks)
        else 0
    )

    obi = (
        (bid20 - ask20)
        / (bid20 + ask20 + 1e-12)
    )

    spread = (
        abs(
            asks[0, 0]
            - bids[0, 0]
        )
        if len(bids)
        and len(asks)
        else 0
    )

    ratio = (
        bid20
        / (ask20 + 1e-8)
    )

    total_depth = (
        bid20 + ask20
    )

    if len(df) >= 20:

        sma20 = (
            df["Close"]
            .rolling(20)
            .mean()
            .iloc[-1]
        )

    else:

        sma20 = df["Close"].mean()

    trend_signal = float(
        df["Close"].iloc[-1]
        - sma20
    )

    return pd.DataFrame(
        [
            {
                "top20_bid_sum": bid20,
                "top20_ask_sum": ask20,
                "obi_top20": obi,
                "spread": spread,
                "bid_ask_ratio": ratio,
                "total_depth": total_depth,
                "trend_signal": trend_signal,
            }
        ],
        columns=XGB_FEATURES,
    )


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
            name: 1.0 / len(
                self.feature_names
            )
            for name
            in self.feature_names
        }

    def extract_features(
        self,
        df,
        bids,
        asks,
    ):

        result = {
            name: 0.0
            for name
            in self.feature_names
        }

        if (
            df.empty
            or len(df) < 15
            or len(bids) == 0
            or len(asks) == 0
        ):
            return result

        bid_volume = float(
            np.sum(
                bids[:20, 1]
            )
        )

        ask_volume = float(
            np.sum(
                asks[:20, 1]
            )
        )

        mid_price = (
            bids[0, 0]
            + asks[0, 0]
        ) / 2

        returns = (
            df["Close"]
            .pct_change()
            .dropna()
        )

        volatility = (
            float(returns.std())
            + 1e-8
        )

        returns_h = (
            df["Close"].iloc[-1]
            - df["Close"].iloc[-5]
        ) / (
            df["Close"].iloc[-5]
            + 1e-8
        )

        delta_price = (
            df["Close"].iloc[-1]
            - df["Close"].iloc[-2]
        )

        # Hawkes-style activity
        volume_changes = (
            df["Volume"]
            .pct_change()
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
            .values
        )

        if len(volume_changes) >= 15:

            recent = np.mean(
                volume_changes[-3:]
            )

            baseline = np.mean(
                volume_changes[-15:]
            )

            hawkes = (
                recent
                / (abs(baseline) + 1e-8)
            )

            result["HAWKES"] = np.clip(
                (
                    hawkes - 1
                )
                * np.sign(
                    returns_h
                ),
                -1,
                1,
            )

        # Book imbalance
        result["BOOK_IMB"] = (
            (
                bid_volume
                - ask_volume
            )
            / (
                bid_volume
                + ask_volume
                + 1e-8
            )
        )

        # Taker flow approximation
        volume_now = float(
            df["Volume"].iloc[-1]
        )

        if delta_price > 0:

            buy = volume_now
            sell = volume_now * 0.3

        else:

            buy = volume_now * 0.3
            sell = volume_now

        result["TAKER_FLOW"] = (
            buy - sell
        ) / (
            buy + sell + 1e-8
        )

        # Best level pressure
        best_bid = float(
            bids[0, 1]
        )

        best_ask = float(
            asks[0, 1]
        )

        result["QUANT_IMPLY"] = np.clip(
            (
                best_bid
                - best_ask
            )
            / (
                best_bid
                + best_ask
                + 1e-8
            )
            * 1.5,
            -1,
            1,
        )

        # Bayesian
        prior = 0.55

        likelihood = (
            0.70
            if result["BOOK_IMB"] > 0
            else 0.30
        )

        posterior = (
            likelihood * prior
        ) / (
            likelihood * prior
            + (
                1 - likelihood
            ) * (
                1 - prior
            )
            + 1e-8
        )

        result["BAYESIAN"] = np.clip(
            (
                posterior - 0.5
            ) * 2,
            -1,
            1,
        )

        # Quantiles
        q90 = (
            returns.quantile(0.90)
            if len(returns) > 5
            else 0.01
        )

        q10 = (
            returns.quantile(0.10)
            if len(returns) > 5
            else -0.01
        )

        result["QUANTILES"] = np.clip(
            (
                (
                    returns_h
                    - q10
                )
                / (
                    q90
                    - q10
                    + 1e-8
                )
                * 2
                - 1
            ),
            -1,
            1,
        )

        # Target/invalidation
        move_pct = (
            delta_price
            / (
                df["Close"].iloc[-1]
                + 1e-8
            )
        )

        if move_pct >= 0.0006:

            result["TARGET_INV"] = 1.0

        elif move_pct <= -0.0006:

            result["TARGET_INV"] = -1.0

        else:

            result["TARGET_INV"] = 0.0

        # Adaptive confirmation
        fast = (
            df["Close"]
            .rolling(3)
            .mean()
            .iloc[-1]
        )

        slow = (
            df["Close"]
            .rolling(10)
            .mean()
            .iloc[-1]
        )

        result["ADAPT_CONF"] = np.clip(
            (
                fast - slow
            )
            / (
                volatility
                * mid_price
                + 1e-8
            ),
            -1,
            1,
        )

        # Fractional Kelly
        win_probability = (
            0.55
            + 0.10
            * np.sign(
                result["BOOK_IMB"]
            )
        )

        kelly = (
            win_probability
            - (
                (
                    1
                    - win_probability
                )
                / 1.5
            )
        )

        result["FRAC_KELLY"] = np.clip(
            kelly
            * 2
            * np.sign(
                returns_h
            ),
            -1,
            1,
        )

        # RMT dominance
        dominance = (
            abs(returns_h)
            / (
                volatility
                * np.sqrt(5)
                + 1e-8
            )
        ) / 3

        result["RMT_DOM"] = np.clip(
            dominance
            * np.sign(
                returns_h
            ),
            -1,
            1,
        )

        # Conformal center
        result["CONF_CROSS"] = np.clip(
            returns_h
            / (
                volatility
                * 2
                + 1e-8
            ),
            -1,
            1,
        )

        # Reward/risk
        rr = (
            abs(q90)
            / (
                abs(q10)
                + 1e-8
            )
        )

        if rr >= 1.2:

            result["REWARD_RISK"] = 1.0

        elif rr < 0.8:

            result["REWARD_RISK"] = -1.0

        else:

            result["REWARD_RISK"] = 0.0

        return result

    def calculate_all_signals(
        self,
        df,
        bids,
        asks,
    ):

        features = self.extract_features(
            df,
            bids,
            asks,
        )

        vector = np.array(
            [
                features[k]
                for k
                in self.feature_names
            ]
        )

        weights = np.array(
            [
                self.dynamic_weights[k]
                for k
                in self.feature_names
            ]
        )

        score = float(
            np.dot(
                vector,
                weights,
            )
        )

        return (
            features,
            score,
            self.dynamic_weights,
        )


# ============================================================
# RISK ENGINE
# ============================================================

class PowerTradingRiskEngine:

    def calculate_risk_metrics(
        self,
        bids,
        asks,
        volatility,
    ):

        bid_depth = (
            np.sum(bids[:, 1])
            if len(bids)
            else 0
        )

        ask_depth = (
            np.sum(asks[:, 1])
            if len(asks)
            else 0
        )

        total_depth = (
            bid_depth
            + ask_depth
        )

        imbalance = (
            abs(
                bid_depth
                - ask_depth
            )
            / (
                total_depth
                + 1e-8
            )
        )

        spread = (
            abs(
                asks[0, 0]
                - bids[0, 0]
            )
            if len(bids)
            and len(asks)
            else 0
        )

        ltz_score = float(
            min(
                imbalance * 100,
                100,
            )
        )

        spoof_score = float(
            min(
                spread
                / (
                    (
                        bids[0, 0]
                        + asks[0, 0]
                    )
                    / 2
                    + 1e-8
                )
                * 100000,
                100,
            )
        )

        squeeze = float(
            min(
                volatility * 1000,
                100,
            )
        )

        market_risk = float(
            min(
                0.45 * ltz_score
                + 0.20 * spoof_score
                + 0.35 * squeeze,
                100,
            )
        )

        if market_risk >= 75:

            risk_status = "EXTREME"

        elif market_risk >= 50:

            risk_status = "HIGH"

        elif market_risk >= 25:

            risk_status = "MEDIUM"

        else:

            risk_status = "LOW"

        return {
            "LTZ_Score": ltz_score,
            "Spoof_Score": spoof_score,
            "Squeeze_Risk": squeeze,
            "Market_Risk": market_risk,
            "Risk_Status": risk_status,
        }


# ============================================================
# PENDING TRADE RESOLUTION
# ============================================================

def resolve_pending_trades(
    history,
    symbol,
    timeframe,
    candle_time,
    candle_high,
    candle_low,
):

    changed = False

    candle_string = (
        pd.Timestamp(
            candle_time
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    for trade in history:

        if str(
            trade.get(
                "outcome",
                "",
            )
        ).upper() != "PENDING":

            continue

        if trade.get("symbol") != symbol:
            continue

        if trade.get("timeframe") != timeframe:
            continue

        entry_candle = str(
            trade.get(
                "entry_candle_time",
                "",
            )
        )

        if (
            entry_candle
            == candle_string
        ):
            continue

        direction = str(
            trade.get(
                "direction",
                "",
            )
        ).upper()

        entry = float(
            trade.get(
                "entry_price",
                0,
            )
        )

        sl = float(
            trade.get(
                "stop_loss",
                0,
            )
        )

        tp = float(
            trade.get(
                "tp1",
                0,
            )
        )

        if (
            entry <= 0
            or sl <= 0
            or tp <= 0
        ):
            continue

        if direction == "LONG":

            tp_hit = (
                candle_high >= tp
            )

            sl_hit = (
                candle_low <= sl
            )

        else:

            tp_hit = (
                candle_low <= tp
            )

            sl_hit = (
                candle_high >= sl
            )

        if not tp_hit and not sl_hit:
            continue

        if tp_hit and sl_hit:

            outcome = "LOSS"
            exit_price = sl
            reason = (
                "SL & TP same candle "
                "(SL-first)"
            )

        elif tp_hit:

            outcome = "WIN"
            exit_price = tp
            reason = "TP1 HIT"

        else:

            outcome = "LOSS"
            exit_price = sl
            reason = "SL HIT"

        if direction == "LONG":

            pnl = (
                (
                    exit_price
                    - entry
                )
                / entry
            ) * 100

        else:

            pnl = (
                (
                    entry
                    - exit_price
                )
                / entry
            ) * 100

        trade["outcome"] = outcome
        trade["exit_price"] = round(
            exit_price,
            2,
        )
        trade["pnl_percent"] = round(
            pnl,
            4,
        )
        trade["status"] = "Closed"
        trade["duration"] = "Closed"
        trade["exit_reason"] = reason
        trade["exit_time"] = (
            datetime.datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        append_feedback(trade)

        changed = True

    return changed


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ⚡ Quant Terminal"
)

st.sidebar.caption(
    "Research • Microstructure • XGBoost • Paper Trading"
)

selected_symbol = st.sidebar.selectbox(
    "Cryptocurrency",
    COINS_LIST,
)

selected_tf_label = st.sidebar.selectbox(
    "Timeframe",
    list(TIMEFRAME_MAP.keys()),
    index=1,
)

api_interval, tf_minutes = TIMEFRAME_MAP[
    selected_tf_label
]

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon",
    5,
    30,
    15,
)

st.sidebar.markdown("---")

paper_trading_mode = st.sidebar.toggle(
    "Enable Paper Trading",
    value=True,
)

show_debug = st.sidebar.toggle(
    "Show Engine Debug",
    value=False,
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 🤖 XGBoost"
)

if xgb_model is not None:

    st.sidebar.success(
        "MODEL LOADED"
    )

else:

    st.sidebar.error(
        "MODEL NOT LOADED"
    )

    if xgb_model_error:

        st.sidebar.caption(
            xgb_model_error
        )

feedback_count = len(
    load_feedback()
)

st.sidebar.caption(
    f"Completed feedback: {feedback_count}"
)

if st.session_state.get(
    "xgb_retrain_message"
):

    st.sidebar.info(
        st.session_state[
            "xgb_retrain_message"
        ]
    )


# ============================================================
# FETCH CURRENT DATA
# ============================================================

df = fetch_klines_data(
    selected_symbol,
    api_interval,
    150,
)

bids, asks = fetch_order_book_depth(
    selected_symbol,
    50,
)


# ============================================================
# HANDLE EMPTY DATA
# ============================================================

if (
    df.empty
    or len(df) < 20
    or len(bids) == 0
    or len(asks) == 0
):

    st.markdown(
        """
<div class="terminal-header">
    <div class="terminal-title">
        ⚡ Quantitative Research Terminal
    </div>
    <div class="terminal-subtitle">
        Waiting for market data...
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.warning(
        "Market data is temporarily unavailable. "
        "The dashboard will retry automatically."
    )

    st.stop()


# ============================================================
# RESOLVE EXISTING PENDING TRADES
# ============================================================

pending_pairs = {
    (
        t.get("symbol"),
        t.get("timeframe"),
    )
    for t
    in st.session_state.trade_history_log
    if str(
        t.get("outcome", "")
    ).upper() == "PENDING"
}

for symbol, timeframe in pending_pairs:

    if not symbol or not timeframe:
        continue

    if (
        symbol == selected_symbol
        and timeframe == selected_tf_label
    ):

        local_df = df

    else:

        local_df = fetch_klines_data(
            symbol,
            timeframe.split(" ")[0],
            3,
        )

    if local_df.empty:
        continue

    last = local_df.iloc[-1]

    resolve_pending_trades(
        st.session_state.trade_history_log,
        symbol,
        timeframe,
        last["Time"],
        float(last["High"]),
        float(last["Low"]),
    )

save_persistent_history(
    st.session_state.trade_history_log
)


# ============================================================
# RETRAIN XGB
# ============================================================

try:

    (
        xgb_model,
        retrain_message,
    ) = retrain_xgb_from_feedback(
        xgb_model
    )

    if retrain_message:

        st.session_state[
            "xgb_retrain_message"
        ] = retrain_message

except Exception as e:

    st.session_state[
        "xgb_retrain_message"
    ] = f"Retrain skipped: {e}"


# ============================================================
# RESEARCH
# ============================================================

lab = TenPaperResearchLab()

(
    paper_results,
    research_score,
    research_weights,
) = lab.calculate_all_signals(
    df,
    bids,
    asks,
)


# ============================================================
# PRICE
# ============================================================

close_price = float(
    df["Close"].iloc[-1]
)

last_high = float(
    df["High"].iloc[-1]
)

last_low = float(
    df["Low"].iloc[-1]
)

last_candle_time = pd.Timestamp(
    df["Time"].iloc[-1]
)


# ============================================================
# ATR
# ============================================================

tr = pd.concat(
    [
        df["High"] - df["Low"],
        (
            df["High"]
            - df["Close"].shift()
        ).abs(),
        (
            df["Low"]
            - df["Close"].shift()
        ).abs(),
    ],
    axis=1,
).max(axis=1)

atr = float(
    tr.rolling(14)
    .mean()
    .iloc[-1]
)

if not np.isfinite(atr) or atr <= 0:

    atr = close_price * 0.005


# ============================================================
# XGB PREDICTION
# ============================================================

xgb_features = build_xgb_features(
    df,
    bids,
    asks,
)

xgb_signal = "NEUTRAL"
xgb_confidence = 0.0
xgb_prediction = None

if xgb_model is not None:

    try:

        xgb_prediction = int(
            xgb_model.predict(
                xgb_features
            )[0]
        )

        probabilities = (
            xgb_model.predict_proba(
                xgb_features
            )[0]
        )

        xgb_confidence = float(
            np.max(
                probabilities
            )
            * 100
        )

        xgb_signal = (
            "LONG"
            if xgb_prediction == 1
            else "SHORT"
        )

    except Exception as e:

        xgb_model_error = (
            f"Prediction error: {e}"
        )


# ============================================================
# OBI / OFI
# ============================================================

stats = orderbook_stats(
    bids,
    asks,
)

obi5 = stats["obi_5"]
obi10 = stats["obi_10"]
obi20 = stats["obi_20"]
obi50 = stats["obi_50"]

ofi = calculate_ofi(
    bids[:20],
    asks[:20],
)

depth20 = (
    stats["bid20"]
    + stats["ask20"]
)

ofi_normalized = float(
    np.clip(
        ofi
        / (
            depth20
            + 1e-8
        ),
        -1,
        1,
    )
)

micro_score = float(
    np.clip(
        0.65 * obi20
        + 0.35 * ofi_normalized,
        -1,
        1,
    )
)

if micro_score >= 0.08:

    micro_direction = "LONG"

elif micro_score <= -0.08:

    micro_direction = "SHORT"

else:

    micro_direction = "NEUTRAL"


# ============================================================
# TREND
# ============================================================

ema9 = float(
    df["Close"]
    .ewm(
        span=9,
        adjust=False,
    )
    .mean()
    .iloc[-1]
)

ema21 = float(
    df["Close"]
    .ewm(
        span=21,
        adjust=False,
    )
    .mean()
    .iloc[-1]
)

momentum = float(
    df["Close"].iloc[-1]
    / (
        df["Close"].iloc[-6]
        + 1e-8
    )
    - 1
)

trend_score = float(
    np.clip(
        (
            (
                ema9
                - ema21
            )
            / (
                close_price
                + 1e-8
            )
        )
        * 250
        + np.sign(momentum)
        * min(
            abs(momentum)
            * 1000,
            0.5,
        ),
        -1,
        1,
    )
)

if trend_score >= 0.10:

    trend_direction = "LONG"

elif trend_score <= -0.10:

    trend_direction = "SHORT"

else:

    trend_direction = "NEUTRAL"


# ============================================================
# RESEARCH DIRECTION
# ============================================================

if research_score >= 0.15:

    research_direction = "LONG"

elif research_score <= -0.15:

    research_direction = "SHORT"

else:

    research_direction = "NEUTRAL"


# ============================================================
# COMBINED ENGINE
# ============================================================

if xgb_signal == "LONG":

    xgb_signed = (
        xgb_confidence / 100
    )

elif xgb_signal == "SHORT":

    xgb_signed = -(
        xgb_confidence / 100
    )

else:

    xgb_signed = 0.0


combined_score = float(
    np.clip(
        0.45 * xgb_signed
        + 0.25 * research_score
        + 0.20 * micro_score
        + 0.10 * trend_score,
        -1,
        1,
    )
)


votes = [
    xgb_signal,
    research_direction,
    micro_direction,
    trend_direction,
]

long_votes = votes.count("LONG")
short_votes = votes.count("SHORT")


xgb_ok = (
    xgb_confidence >= 60
)


long_confirmed = (
    xgb_signal == "LONG"
    and research_direction == "LONG"
    and micro_direction == "LONG"
    and trend_direction != "SHORT"
)

short_confirmed = (
    xgb_signal == "SHORT"
    and research_direction == "SHORT"
    and micro_direction == "SHORT"
    and trend_direction != "LONG"
)


if (
    long_confirmed
    and xgb_ok
    and combined_score >= 0.18
):

    direction = "LONG"

elif (
    short_confirmed
    and xgb_ok
    and combined_score <= -0.18
):

    direction = "SHORT"

elif (
    xgb_signal == "LONG"
    and xgb_confidence >= 65
    and long_votes >= 3
    and trend_direction != "SHORT"
    and combined_score >= 0.18
):

    direction = "LONG"

elif (
    xgb_signal == "SHORT"
    and xgb_confidence >= 65
    and short_votes >= 3
    and trend_direction != "LONG"
    and combined_score <= -0.18
):

    direction = "SHORT"

else:

    direction = "NEUTRAL"


confidence = int(
    np.clip(
        abs(combined_score)
        * 100,
        0,
        99,
    )
)


# ============================================================
# SIGNAL STRENGTH
# ============================================================

if direction == "LONG":

    if (
        xgb_confidence >= 80
        and long_votes >= 4
        and combined_score >= 0.45
    ):

        signal_strength = "STRONG LONG"

    else:

        signal_strength = "LONG"

elif direction == "SHORT":

    if (
        xgb_confidence >= 80
        and short_votes >= 4
        and combined_score <= -0.45
    ):

        signal_strength = "STRONG SHORT"

    else:

        signal_strength = "SHORT"

else:

    signal_strength = "WAIT"


# ============================================================
# TARGETS
# ============================================================

risk_distance = max(
    atr,
    close_price * 0.001,
)

TP1_R = 2.0
TP2_R = 3.0


if direction == "LONG":

    sl_price = (
        close_price
        - risk_distance
    )

    tp1_price = (
        close_price
        + risk_distance * TP1_R
    )

    tp2_price = (
        close_price
        + risk_distance * TP2_R
    )

elif direction == "SHORT":

    sl_price = (
        close_price
        + risk_distance
    )

    tp1_price = (
        close_price
        - risk_distance * TP1_R
    )

    tp2_price = (
        close_price
        - risk_distance * TP2_R
    )

else:

    sl_price = (
        close_price
        - risk_distance
    )

    tp1_price = (
        close_price
        + risk_distance * TP1_R
    )

    tp2_price = (
        close_price
        + risk_distance * TP2_R
    )


# ============================================================
# RISK
# ============================================================

risk_engine = PowerTradingRiskEngine()

volatility = float(
    df["Close"]
    .pct_change()
    .std()
)

risk_metrics = (
    risk_engine.calculate_risk_metrics(
        bids,
        asks,
        volatility,
    )
)


# ============================================================
# PAPER TRADE
# ============================================================

lock_seconds = (
    tf_minutes * 60
)

now_seconds = int(
    time.time()
)

bucket = (
    now_seconds
    - (
        now_seconds
        % lock_seconds
    )
)

time_remaining = (
    lock_seconds
    - (
        now_seconds
        % lock_seconds
    )
)

trade_id = (
    f"{selected_symbol}_"
    f"{api_interval}_"
    f"{bucket}_"
    f"{direction}"
)


if (
    paper_trading_mode
    and direction != "NEUTRAL"
):

    existing_ids = {
        str(
            t.get(
                "trade_id",
                "",
            )
        )
        for t
        in st.session_state.trade_history_log
    }

    if trade_id not in existing_ids:

        new_trade = {

            "trade_id": trade_id,

            "timestamp":
                datetime.datetime.now()
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "entry_candle_time":
                last_candle_time.strftime(
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
                round(
                    close_price,
                    2,
                ),

            "stop_loss":
                round(
                    sl_price,
                    2,
                ),

            "tp1":
                round(
                    tp1_price,
                    2,
                ),

            "tp2":
                round(
                    tp2_price,
                    2,
                ),

            "rr_target":
                "TP1 1:2 | TP2 1:3",

            "exit_price":
                0.0,

            "confidence":
                confidence,

            "xgb_confidence":
                round(
                    xgb_confidence,
                    2,
                ),

            "xgb_features_json":
                json.dumps(
                    {
                        k: float(
                            xgb_features.iloc[
                                0
                            ][k]
                        )
                        for k
                        in XGB_FEATURES
                    }
                ),

            "final_score":
                round(
                    combined_score,
                    4,
                ),

            "outcome":
                "PENDING",

            "pnl_percent":
                0.0,

            "duration":
                "Active",

            "status":
                "Open",

            "exit_reason":
                "",

        }

        st.session_state.trade_history_log.insert(
            0,
            new_trade,
        )

        save_persistent_history(
            st.session_state.trade_history_log
        )


# ============================================================
# HEADER
# ============================================================

if direction == "LONG":

    signal_color = "#00e676"

elif direction == "SHORT":

    signal_color = "#ff5252"

else:

    signal_color = "#38bdf8"


mins, secs = divmod(
    time_remaining,
    60,
)


st.markdown(
    f"""
<div class="terminal-header">

    <div class="terminal-title">
        ⚡ Quantitative Research & Paper Trading Terminal
    </div>

    <div class="terminal-subtitle">
        Multi-factor market research engine
        • Order Book Microstructure
        • XGBoost Direction Model
        • 12 Quantitative Research Factors
    </div>

</div>

<div class="status-bar">

    <b>{selected_symbol}</b>
    &nbsp; | &nbsp;

    Price:
    <b>${close_price:,.2f}</b>

    &nbsp; | &nbsp;

    TF:
    <b>{selected_tf_label}</b>

    &nbsp; | &nbsp;

    SIGNAL:
    <span style="color:{signal_color};font-weight:900;">
        {signal_strength}
    </span>

    &nbsp; | &nbsp;

    SCORE:
    <b>{combined_score:+.3f}</b>

    &nbsp; | &nbsp;

    XGB:
    <b>{xgb_signal}</b>
    ({xgb_confidence:.1f}%)

    &nbsp; | &nbsp;

    OBI:
    <b>{obi20:+.3f}</b>

    &nbsp; | &nbsp;

    OFI:
    <b>{ofi:+.2f}</b>

    &nbsp; | &nbsp;

    CONF:
    <b>{confidence}%</b>

    &nbsp; | &nbsp;

    RESET:
    <b>{mins}m {secs}s</b>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN SIGNAL CARDS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(
    [1.35, 1, 1, 1, 1]
)


with c1:

    st.markdown(
        f"""
<div class="signal-card"
     style="border-left:4px solid {signal_color};">

    <div class="signal-label">
        Signal Execution
    </div>

    <div class="signal-value"
         style="color:{signal_color};">
        {signal_strength}
    </div>

    <div class="card-small">
        Direction: <b>{direction}</b>
    </div>

    <div class="card-small">
        XGB: <b>{xgb_confidence:.1f}%</b>
        &nbsp; • &nbsp;
        Votes:
        <b>{max(long_votes, short_votes)}/4</b>
    </div>

</div>
""",
        unsafe_allow_html=True,
    )


with c2:

    st.markdown(
        f"""
<div class="card">

<div class="card-title">
Entry
</div>

<div class="card-value blue">
${close_price:,.2f}
</div>

<div class="card-small">
Current market price
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with c3:

    st.markdown(
        f"""
<div class="card">

<div class="card-title">
Stop Loss
</div>

<div class="card-value red">
${sl_price:,.2f}
</div>

<div class="card-small">
Risk distance: ${risk_distance:,.2f}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with c4:

    st.markdown(
        f"""
<div class="card">

<div class="card-title">
TP1 • 2R
</div>

<div class="card-value green">
${tp1_price:,.2f}
</div>

<div class="card-small">
Risk / Reward: 1 : 2
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with c5:

    st.markdown(
        f"""
<div class="card">

<div class="card-title">
TP2 • 3R
</div>

<div class="card-value green">
${tp2_price:,.2f}
</div>

<div class="card-small">
Risk / Reward: 1 : 3
</div>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# SECOND METRIC ROW
# ============================================================

m1, m2, m3, m4, m5, m6 = st.columns(6)


with m1:

    st.metric(
        "OBI Top 5",
        f"{obi5:+.3f}",
    )


with m2:

    st.metric(
        "OBI Top 10",
        f"{obi10:+.3f}",
    )


with m3:

    st.metric(
        "OBI Top 20",
        f"{obi20:+.3f}",
    )


with m4:

    st.metric(
        "OBI Top 50",
        f"{obi50:+.3f}",
    )


with m5:

    st.metric(
        "Research Score",
        f"{research_score:+.3f}",
    )


with m6:

    st.metric(
        "Market Risk",
        f"{risk_metrics['Market_Risk']:.1f}",
        risk_metrics["Risk_Status"],
    )


# ============================================================
# PRICE CHART + MICROSTRUCTURE
# ============================================================

st.markdown(
    '<div class="section-title">📈 Price Structure & Forecast</div>',
    unsafe_allow_html=True,
)

chart_col, micro_col = st.columns(
    [2.5, 1]
)


with chart_col:

    future_delta = pd.Timedelta(
        minutes=tf_minutes
    )

    future_times = [
        last_candle_time
        + future_delta * i
        for i
        in range(
            1,
            forecast_horizon + 1,
        )
    ]

    steps = np.linspace(
        0,
        np.pi / 2,
        forecast_horizon,
    )

    if direction == "LONG":

        forecast = (
            close_price
            + (
                tp2_price
                - close_price
            )
            * np.sin(steps)
        )

    elif direction == "SHORT":

        forecast = (
            close_price
            - (
                close_price
                - tp2_price
            )
            * np.sin(steps)
        )

    else:

        forecast = np.full(
            forecast_horizon,
            close_price,
        )

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

    fig.add_trace(
        go.Scatter(
            x=[
                last_candle_time
            ] + future_times,
            y=[
                close_price
            ] + list(forecast),
            mode="lines+markers",
            name="Model Trajectory",
            line=dict(
                color=signal_color,
                width=2,
                dash="dot",
            ),
        )
    )

    fig.add_hline(
        y=sl_price,
        line_dash="dot",
        annotation_text="SL",
    )

    fig.add_hline(
        y=tp1_price,
        line_dash="dash",
        annotation_text="TP1",
    )

    fig.add_hline(
        y=tp2_price,
        line_dash="dash",
        annotation_text="TP2",
    )

    fig.update_layout(
        template="plotly_dark",
        height=470,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e141d",
        plot_bgcolor="#0e141d",
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        legend=dict(
            orientation="h",
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with micro_col:

    st.markdown(
        """
<div class="section-title">
    🧠 Market Microstructure
</div>
""",
        unsafe_allow_html=True,
    )

    bid20 = stats["bid20"]
    ask20 = stats["ask20"]

    st.markdown(
        f"""
<div class="card">

<div class="card-title">
Top 20 Order Book
</div>

<div style="display:flex;justify-content:space-between;">
<span>Bid Volume</span>
<b class="green">
{bid20:,.3f}
</b>
</div>

<br>

<div style="display:flex;justify-content:space-between;">
<span>Ask Volume</span>
<b class="red">
{ask20:,.3f}
</b>
</div>

<br>

<div style="display:flex;justify-content:space-between;">
<span>OBI</span>
<b class="blue">
{obi20:+.4f}
</b>
</div>

<br>

<div style="display:flex;justify-content:space-between;">
<span>OFI</span>
<b class="yellow">
{ofi:+.2f}
</b>
</div>

<br>

<div style="display:flex;justify-content:space-between;">
<span>Spread</span>
<b>
${stats["spread"]:.4f}
</b>
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="card">

<div class="card-title">
Directional Components
</div>

<div style="display:flex;justify-content:space-between;">
<span>XGBoost</span>
<b>{xgb_signal}</b>
</div>

<br>

<div style="display:flex;justify-content:space-between;">
<span>Research</span>
<b>{research_direction}</b>
</div>

<br>

<div style="display:flex;justify-content:space-between;">
<span>Microstructure</span>
<b>{micro_direction}</b>
</div>

<br>

<div style="display:flex;justify-content:space-between;">
<span>Trend</span>
<b>{trend_direction}</b>
</div>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# OBI PROFILE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Order Book Imbalance Profile</div>',
    unsafe_allow_html=True,
)

obi_fig = go.Figure()

obi_fig.add_trace(
    go.Bar(
        x=[
            "Top 5",
            "Top 10",
            "Top 20",
            "Top 50",
        ],
        y=[
            obi5,
            obi10,
            obi20,
            obi50,
        ],
        text=[
            f"{obi5:+.3f}",
            f"{obi10:+.3f}",
            f"{obi20:+.3f}",
            f"{obi50:+.3f}",
        ],
        textposition="auto",
    )
)

obi_fig.add_hline(
    y=0,
    line_dash="dot",
)

obi_fig.update_layout(
    template="plotly_dark",
    height=260,
    paper_bgcolor="#0e141d",
    plot_bgcolor="#0e141d",
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10,
    ),
)

st.plotly_chart(
    obi_fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    },
)


# ============================================================
# RISK + RESEARCH
# ============================================================

st.markdown(
    '<div class="section-title">🛡️ Risk Engine & Research Engine</div>',
    unsafe_allow_html=True,
)

risk_col, research_col = st.columns(
    [1, 2]
)


with risk_col:

    risk_status = (
        risk_metrics[
            "Risk_Status"
        ]
    )

    risk_color = (
        "#00e676"
        if risk_status == "LOW"
        else (
            "#fbbf24"
            if risk_status == "MEDIUM"
            else "#ff5252"
        )
    )

    st.markdown(
        f"""
<div class="card">

<div class="card-title">
Market Risk Status
</div>

<div style="
font-size:28px;
font-weight:900;
color:{risk_color};
">
{risk_status}
</div>

<hr>

<div style="display:flex;justify-content:space-between;">
<span>LTZ Score</span>
<b>{risk_metrics["LTZ_Score"]:.2f}</b>
</div>

<br>

<div style="display:flex;justify-content:space-between;">
<span>Spoof Score</span>
<b>{risk_metrics["Spoof_Score"]:.2f}</b>
</div>

<br>

<div style="display:flex;justify-content:space-between;">
<span>Squeeze Risk</span>
<b>{risk_metrics["Squeeze_Risk"]:.2f}</b>
</div>

<br>

<div style="display:flex;justify-content:space-between;">
<span>Total Risk</span>
<b>{risk_metrics["Market_Risk"]:.2f}</b>
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with research_col:

    paper_rows = []

    for name, value in paper_results.items():

        if value > 0.10:

            status = "PASS 🟢"

        elif value < -0.10:

            status = "FAIL 🔴"

        else:

            status = "NEUTRAL ⚪"

        paper_rows.append(
            {
                "Research Factor": name,
                "Value": round(
                    value,
                    4,
                ),
                "Weight": f"{research_weights[name] * 100:.1f}%",
                "Status": status,
            }
        )

    st.dataframe(
        pd.DataFrame(
            paper_rows
        ),
        use_container_width=True,
        hide_index=True,
        height=330,
    )


# ============================================================
# PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Paper Trading Performance</div>',
    unsafe_allow_html=True,
)

history_df = pd.DataFrame(
    st.session_state.trade_history_log
)


if not history_df.empty:

    if "outcome" not in history_df.columns:
        history_df["outcome"] = "PENDING"

    if "pnl_percent" not in history_df.columns:
        history_df["pnl_percent"] = 0.0

    if "direction" not in history_df.columns:
        history_df["direction"] = ""

    history_df["outcome"] = (
        history_df["outcome"]
        .astype(str)
        .str.upper()
    )

    filter1, filter2, filter3 = st.columns(
        3
    )

    with filter1:

        coin_filter = st.selectbox(
            "Coin",
            ["ALL"] + COINS_LIST,
            key="history_coin",
        )

    with filter2:

        tf_filter = st.selectbox(
            "Timeframe",
            ["ALL"]
            + list(
                TIMEFRAME_MAP.keys()
            ),
            key="history_tf",
        )

    with filter3:

        direction_filter = st.selectbox(
            "Direction",
            [
                "ALL",
                "LONG",
                "SHORT",
            ],
            key="history_direction",
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
        if closed > 0
        else 0
    )

    gross_profit = (
        filtered.loc[
            filtered["outcome"] == "WIN",
            "pnl_percent",
        ].sum()
        if not filtered.empty
        else 0
    )

    gross_loss = abs(
        filtered.loc[
            filtered["outcome"] == "LOSS",
            "pnl_percent",
        ].sum()
        if not filtered.empty
        else 0
    )

    net_pnl = (
        gross_profit
        - gross_loss
    )

    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else 0
    )


    p1, p2, p3, p4, p5, p6 = st.columns(
        6
    )

    with p1:
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
        )

    with p2:
        st.metric(
            "Closed",
            closed,
        )

    with p3:
        st.metric(
            "Wins",
            wins,
        )

    with p4:
        st.metric(
            "Losses",
            losses,
        )

    with p5:
        st.metric(
            "Profit Factor",
            f"{profit_factor:.2f}",
        )

    with p6:
        st.metric(
            "Net PnL",
            f"{net_pnl:+.2f}%",
        )


    # ========================================================
    # PERFORMANCE CHART
    # ========================================================

    chart_data = filtered.copy()

    if not chart_data.empty:

        chart_data["pnl"] = pd.to_numeric(
            chart_data[
                "pnl_percent"
            ],
            errors="coerce",
        ).fillna(0)

        chart_data["equity"] = (
            chart_data["pnl"]
            .cumsum()
        )

        perf_fig = go.Figure()

        perf_fig.add_trace(
            go.Scatter(
                x=np.arange(
                    len(chart_data)
                ),
                y=chart_data[
                    "equity"
                ],
                mode="lines+markers",
                name="Cumulative PnL",
            )
        )

        perf_fig.update_layout(
            template="plotly_dark",
            height=280,
            paper_bgcolor="#0e141d",
            plot_bgcolor="#0e141d",
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),
        )

        st.plotly_chart(
            perf_fig,
            use_container_width=True,
        )


    # ========================================================
    # HISTORY TABLE
    # ========================================================

    st.markdown(
        "##### Detailed Trade History"
    )

    display_columns = [
        "timestamp",
        "symbol",
        "timeframe",
        "direction",
        "signal_strength",
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "exit_price",
        "confidence",
        "xgb_confidence",
        "pnl_percent",
        "outcome",
        "exit_reason",
    ]

    defaults = {
        "timestamp": "",
        "symbol": "",
        "timeframe": "",
        "direction": "",
        "signal_strength": "",
        "entry_price": 0.0,
        "stop_loss": 0.0,
        "tp1": 0.0,
        "tp2": 0.0,
        "exit_price": 0.0,
        "confidence": 0.0,
        "xgb_confidence": 0.0,
        "pnl_percent": 0.0,
        "outcome": "PENDING",
        "exit_reason": "",
    }

    for column in display_columns:

        if column not in filtered.columns:

            filtered[column] = defaults[
                column
            ]

    st.dataframe(
        filtered[
            display_columns
        ],
        use_container_width=True,
        hide_index=True,
        height=350,
    )

else:

    st.info(
        "No paper trades recorded yet."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">🤖 XGBoost Model Status</div>',
    unsafe_allow_html=True,
)

model_col1, model_col2 = st.columns(
    2
)


with model_col1:

    st.markdown(
        f"""
<div class="card">

<div class="card-title">
Model File
</div>

<div class="card-value blue">
{MODEL_PATH}
</div>

<div class="card-small">
Status:
<b>
{
    "LOADED"
    if xgb_model is not None
    else "NOT LOADED"
}
</b>
</div>

</div>
""",
        unsafe_allow_html=True,
    )


with model_col2:

    st.markdown(
        """
<div class="card">

<div class="card-title">
Live XGB Features
</div>

<div style="
font-size:11px;
line-height:1.8;
color:#aab6c5;
">

Top20 Bid Sum •
Top20 Ask Sum •
OBI Top20 •
Spread •
Bid/Ask Ratio •
Total Depth •
Trend Signal

</div>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# DEBUG
# ============================================================

if show_debug:

    st.markdown(
        '<div class="section-title">🔧 Engine Debug</div>',
        unsafe_allow_html=True,
    )

    debug_data = {

        "Symbol":
            selected_symbol,

        "Timeframe":
            selected_tf_label,

        "Price":
            close_price,

        "XGB Signal":
            xgb_signal,

        "XGB Confidence":
            xgb_confidence,

        "Research Score":
            research_score,

        "Micro Score":
            micro_score,

        "Trend Score":
            trend_score,

        "Combined Score":
            combined_score,

        "OBI 5":
            obi5,

        "OBI 10":
            obi10,

        "OBI 20":
            obi20,

        "OBI 50":
            obi50,

        "OFI":
            ofi,

        "Long Votes":
            long_votes,

        "Short Votes":
            short_votes,

        "Direction":
            direction,

        "Confidence":
            confidence,

        "Risk":
            risk_metrics[
                "Market_Risk"
            ],

    }

    st.json(debug_data)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<hr>

<div style="
text-align:center;
color:#536174;
font-size:10px;
padding:10px;
">

QUANTITATIVE RESEARCH TERMINAL
&nbsp; • &nbsp;
PAPER TRADING ONLY
&nbsp; • &nbsp;
XGBoost + OBI + OFI + Research Engine

</div>
""",
    unsafe_allow_html=True,
)
