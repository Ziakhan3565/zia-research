import datetime
import os
import time
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import joblib
import streamlit as st

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from streamlit_autorefresh import st_autorefresh


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Quantitative Research & Paper Trading Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(
    interval=5000,
    limit=None,
    key="research_lab_auto_refresh"
)


# ============================================================
# FILES
# ============================================================

CSV_FILE = "signal_history.csv"
MODEL_PATH = "xgboost_obi_model.pkl"
FEEDBACK_FILE = "xgb_trade_feedback.csv"


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


# ============================================================
# EXPECTED HISTORY COLUMNS
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


# ============================================================
# SESSION STATE
# ============================================================

if "trade_history_log" not in st.session_state:
    st.session_state.trade_history_log = []

if "previous_orderbook" not in st.session_state:
    st.session_state.previous_orderbook = None

if "obi_history" not in st.session_state:
    st.session_state.obi_history = []

if "ofi_history" not in st.session_state:
    st.session_state.ofi_history = []

if "signal_history" not in st.session_state:
    st.session_state.signal_history = []

if "xgb_last_retrain_count" not in st.session_state:
    st.session_state.xgb_last_retrain_count = 0

if "xgb_retrain_message" not in st.session_state:
    st.session_state.xgb_retrain_message = ""


# ============================================================
# PERSISTENT HISTORY
# ============================================================

def normalize_trade(trade):
    trade = dict(trade)

    trade["outcome"] = str(
        trade.get("outcome", "PENDING")
    ).upper()

    trade["status"] = (
        "Closed"
        if trade["outcome"] in ("WIN", "LOSS")
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
            trade[key] = float(trade.get(key, 0.0))
        except Exception:
            trade[key] = 0.0

    defaults = {
        "signal_strength": "",
        "rr_target": "TP1 1:2 | TP2 1:3",
        "duration": "Active",
        "exit_reason": "",
        "entry_candle_time": trade.get("timestamp", ""),
        "exit_time": "",
        "xgb_features_json": "",
    }

    for key, value in defaults.items():
        if not trade.get(key):
            trade[key] = value

    return trade


def load_persistent_history():
    if not os.path.exists(CSV_FILE):
        return []

    try:
        df_hist = pd.read_csv(CSV_FILE)

        for col in EXPECTED_HISTORY_COLUMNS:
            if col not in df_hist.columns:
                if col == "outcome":
                    df_hist[col] = "PENDING"
                elif col == "rr_target":
                    df_hist[col] = "TP1 1:2 | TP2 1:3"
                else:
                    df_hist[col] = 0.0

        records = df_hist.to_dict("records")
        return [normalize_trade(x) for x in records]

    except Exception:
        return []


def save_persistent_history(history_list):
    try:
        if not history_list:
            pd.DataFrame(columns=EXPECTED_HISTORY_COLUMNS).to_csv(
                CSV_FILE,
                index=False
            )
            return

        df_hist = pd.DataFrame(
            [normalize_trade(x) for x in history_list]
        )

        for col in EXPECTED_HISTORY_COLUMNS:
            if col not in df_hist.columns:
                df_hist[col] = ""

        df_hist = df_hist[EXPECTED_HISTORY_COLUMNS]

        df_hist.to_csv(
            CSV_FILE,
            index=False
        )

    except Exception as e:
        st.error(f"Error saving history: {e}")


if not st.session_state.trade_history_log:
    st.session_state.trade_history_log = load_persistent_history()


# ============================================================
# XGBOOST MODEL
# ============================================================

@st.cache_resource
def load_xgb_model():

    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"

    try:
        model = joblib.load(MODEL_PATH)

        if not hasattr(model, "predict"):
            return None, "Invalid XGBoost model."

        return model, None

    except Exception as e:
        return None, f"XGBoost load error: {e}"


xgb_model, xgb_model_error = load_xgb_model()


# ============================================================
# FEEDBACK LEARNING
# ============================================================

def _load_feedback():

    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame(
            columns=XGB_FEATURES + [
                "target",
                "trade_id",
                "closed_at",
            ]
        )

    try:

        fb = pd.read_csv(FEEDBACK_FILE)

        for c in XGB_FEATURES + ["target"]:
            if c not in fb.columns:
                return pd.DataFrame(
                    columns=XGB_FEATURES + [
                        "target",
                        "trade_id",
                        "closed_at",
                    ]
                )

        return fb.dropna(
            subset=XGB_FEATURES + ["target"]
        ).copy()

    except Exception:

        return pd.DataFrame(
            columns=XGB_FEATURES + [
                "target",
                "trade_id",
                "closed_at",
            ]
        )


def _append_feedback(trade):

    raw = trade.get(
        "xgb_features_json",
        ""
    )

    outcome = str(
        trade.get("outcome", "")
    ).upper()

    if not raw or outcome not in ("WIN", "LOSS"):
        return

    try:

        features = (
            json.loads(raw)
            if isinstance(raw, str)
            else raw
        )

        row = {
            k: float(features[k])
            for k in XGB_FEATURES
        }

        direction = str(
            trade.get("direction", "")
        ).upper()

        # Target is "was this direction correct?"
        row["target"] = int(
            (direction == "LONG")
            == (outcome == "WIN")
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

        fb = _load_feedback()

        existing_ids = set(
            fb.get(
                "trade_id",
                pd.Series(dtype=str)
            ).astype(str)
        )

        if str(row["trade_id"]) in existing_ids:
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


def _retrain_xgb_from_feedback(current_model):

    fb = _load_feedback()

    if len(fb) < MIN_FEEDBACK_TO_RETRAIN:
        return current_model, None

    if fb["target"].nunique() < 2:
        return current_model, None

    last_count = int(
        st.session_state.get(
            "xgb_last_retrain_count",
            0
        )
    )

    if len(fb) < last_count + RETRAIN_EVERY:
        return current_model, None

    fb = fb.sort_values(
        "closed_at",
        kind="stable"
    )

    split = max(
        int(len(fb) * 0.80),
        1
    )

    if split >= len(fb):
        return current_model, None

    X_train = fb.iloc[:split][XGB_FEATURES]
    X_test = fb.iloc[split:][XGB_FEATURES]

    y_train = fb.iloc[:split]["target"].astype(int)
    y_test = fb.iloc[split:]["target"].astype(int)

    if y_train.nunique() < 2:
        return current_model, None

    if y_test.nunique() < 2:
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
        X_train,
        y_train
    )

    test_acc = float(
        accuracy_score(
            y_test,
            candidate.predict(X_test)
        )
    )

    st.session_state.xgb_last_retrain_count = len(fb)

    if test_acc < MIN_TEST_ACCURACY:

        return (
            current_model,
            f"XGB retrain rejected: "
            f"holdout accuracy "
            f"{test_acc * 100:.1f}%"
        )

    tmp_path = MODEL_PATH + ".tmp"

    try:

        joblib.dump(
            candidate,
            tmp_path
        )

        os.replace(
            tmp_path,
            MODEL_PATH
        )

    except Exception as e:

        return (
            current_model,
            f"XGB save error: {e}"
        )

    return (
        candidate,
        f"XGB retrained from "
        f"{len(fb)} completed trades | "
        f"holdout accuracy "
        f"{test_acc * 100:.1f}%"
    )


# ============================================================
# XGB FEATURES
# ============================================================

def build_xgb_features(df, bids, asks):

    bid_sum = (
        float(np.sum(bids[:, 1]))
        if len(bids)
        else 0.0
    )

    ask_sum = (
        float(np.sum(asks[:, 1]))
        if len(asks)
        else 0.0
    )

    obi = (
        (bid_sum - ask_sum)
        /
        (bid_sum + ask_sum + 1e-8)
    )

    spread = (
        abs(
            float(asks[0, 0])
            -
            float(bids[0, 0])
        )
        if len(bids) and len(asks)
        else 0.0
    )

    ratio = (
        bid_sum
        /
        (ask_sum + 1e-5)
    )

    total_depth = (
        bid_sum + ask_sum
    )

    sma20 = (
        df["Close"]
        .rolling(
            20,
            min_periods=1
        )
        .mean()
        .iloc[-1]
    )

    trend_signal = float(
        df["Close"].iloc[-1]
        -
        sma20
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


# ============================================================
# ORDER FLOW
# ============================================================

def calculate_ofi(current_bids, current_asks):

    current_bid_sum = (
        float(np.sum(current_bids[:, 1]))
        if len(current_bids)
        else 0.0
    )

    current_ask_sum = (
        float(np.sum(current_asks[:, 1]))
        if len(current_asks)
        else 0.0
    )

    previous = st.session_state.get(
        "previous_orderbook"
    )

    if previous is None:
        ofi = 0.0
    else:

        previous_bid_sum, previous_ask_sum = previous

        ofi = (
            current_bid_sum
            - previous_bid_sum
        ) - (
            current_ask_sum
            - previous_ask_sum
        )

    st.session_state.previous_orderbook = (
        current_bid_sum,
        current_ask_sum
    )

    return float(ofi)


# ============================================================
# Z-SCORE
# ============================================================

def rolling_zscore(history, value, minimum=10):

    if len(history) < minimum:
        return 0.0

    arr = np.asarray(
        history,
        dtype=float
    )

    mean = float(
        np.mean(arr)
    )

    std = float(
        np.std(arr)
    )

    if std < 1e-8:
        return 0.0

    return float(
        (value - mean) / std
    )


# ============================================================
# 12 PAPER RESEARCH ENGINE
# ============================================================

class TenPaperResearchLab:

    def __init__(
        self,
        target_vol=0.15
    ):

        self.target_vol = target_vol

        self.scaler = StandardScaler()

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
            k: 1.0 / len(self.feature_names)
            for k in self.feature_names
        }

    def extract_features(
        self,
        df,
        bids,
        asks
    ):

        results = {}

        if (
            len(bids) == 0
            or len(asks) == 0
            or df.empty
            or len(df) < 15
        ):

            return {
                k: 0.0
                for k in self.feature_names
            }

        bid_vol = float(
            np.sum(bids[:, 1])
        )

        ask_vol = float(
            np.sum(asks[:, 1])
        )

        mid_price = (
            bids[0, 0]
            +
            asks[0, 0]
        ) / 2

        returns = (
            df["Close"]
            .pct_change()
            .dropna()
        )

        realized_vol = (
            float(returns.std())
            + 1e-8
        )

        returns_h = (
            df["Close"].iloc[-1]
            -
            df["Close"].iloc[-5]
        ) / (
            df["Close"].iloc[-5]
            + 1e-8
        )

        delta_p = (
            df["Close"].iloc[-1]
            -
            df["Close"].iloc[-2]
        )

        # 1 HAWKES
        vol_changes = (
            df["Volume"]
            .pct_change()
            .dropna()
            .replace(
                [np.inf, -np.inf],
                0
            )
            .fillna(0)
            .values
        )

        if len(vol_changes) >= 15:

            recent = np.mean(
                vol_changes[-3:]
            )

            baseline = np.mean(
                vol_changes[-15:]
            )

            hawkes_intensity = (
                recent
                /
                (abs(baseline) + 1e-8)
            )

        else:

            hawkes_intensity = 1.0

        results["HAWKES"] = np.clip(
            (hawkes_intensity - 1.0)
            *
            np.sign(returns_h),
            -1,
            1
        )

        # 2 BOOK IMBALANCE
        results["BOOK_IMB"] = (
            bid_vol - ask_vol
        ) / (
            bid_vol + ask_vol + 1e-8
        )

        # 3 TAKER FLOW
        taker_buy = (
            df["Volume"].iloc[-1]
            *
            (
                1.0
                if delta_p > 0
                else 0.3
            )
        )

        taker_sell = (
            df["Volume"].iloc[-1]
            *
            (
                1.0
                if delta_p <= 0
                else 0.3
            )
        )

        results["TAKER_FLOW"] = (
            taker_buy - taker_sell
        ) / (
            taker_buy + taker_sell + 1e-8
        )

        # 4 DEPTH SKEW
        depth_skew = (
            bids[0, 1]
            -
            asks[0, 1]
        ) / (
            bids[0, 1]
            +
            asks[0, 1]
            +
            1e-8
        )

        results["QUANT_IMPLY"] = np.clip(
            depth_skew * 1.5,
            -1,
            1
        )

        # 5 BAYESIAN
        prior = 0.55

        if results["BOOK_IMB"] > 0:
            likelihood = 0.65
        elif results["BOOK_IMB"] < 0:
            likelihood = 0.35
        else:
            likelihood = 0.50

        posterior = (
            likelihood * prior
        ) / (
            likelihood * prior
            +
            (1 - likelihood)
            *
            (1 - prior)
            +
            1e-8
        )

        results["BAYESIAN"] = np.clip(
            (posterior - 0.5) * 2,
            -1,
            1
        )

        # 6 QUANTILES
        if len(returns) > 5:

            q90 = returns.quantile(
                0.90
            )

            q10 = returns.quantile(
                0.10
            )

        else:

            q90 = 0.01
            q10 = -0.01

        results["QUANTILES"] = np.clip(
            (
                (
                    returns_h - q10
                )
                /
                (
                    q90 - q10
                    + 1e-8
                )
            )
            * 2
            - 1,
            -1,
            1
        )

        # 7 TARGET / INVALIDATION
        target_diff = (
            delta_p
            /
            (
                df["Close"].iloc[-1]
                +
                1e-8
            )
        )

        results["TARGET_INV"] = (
            1.0
            if target_diff >= 0.0006
            else (
                -1.0
                if target_diff <= -0.0006
                else 0.0
            )
        )

        # 8 ADAPTIVE CONF
        ma_fast = (
            df["Close"]
            .rolling(3)
            .mean()
            .iloc[-1]
        )

        ma_slow = (
            df["Close"]
            .rolling(10)
            .mean()
            .iloc[-1]
        )

        results["ADAPT_CONF"] = np.clip(
            (
                ma_fast - ma_slow
            )
            /
            (
                realized_vol
                *
                mid_price
                +
                1e-8
            ),
            -1,
            1
        )

        # 9 FRACTIONAL KELLY
        win_prob = (
            0.55
            +
            0.10
            *
            np.sign(
                results["BOOK_IMB"]
            )
        )

        kelly_fraction = (
            win_prob
            -
            (
                (1 - win_prob)
                /
                1.5
            )
        )

        results["FRAC_KELLY"] = np.clip(
            kelly_fraction
            *
            2
            *
            np.sign(returns_h),
            -1,
            1
        )

        # 10 RMT
        rmt_dom = (
            abs(returns_h)
            /
            (
                realized_vol
                *
                np.sqrt(5)
                +
                1e-8
            )
        ) / 3

        results["RMT_DOM"] = np.clip(
            rmt_dom
            *
            np.sign(returns_h),
            -1,
            1
        )

        # 11 CONF CROSS
        # FIXED: old formula was always exactly zero.
        rolling_mid = (
            df["Close"]
            .rolling(20)
            .mean()
            .iloc[-1]
        )

        if realized_vol > 0:

            deviation = (
                mid_price
                -
                rolling_mid
            ) / (
                realized_vol
                *
                mid_price
                +
                1e-8
            )

            results["CONF_CROSS"] = np.clip(
                deviation / 2.0,
                -1,
                1
            )

        else:

            results["CONF_CROSS"] = 0.0

        # 12 REWARD / RISK
        rr_ratio = (
            abs(q90)
            /
            (
                abs(q10)
                +
                1e-8
            )
        )

        results["REWARD_RISK"] = (
            1.0
            if rr_ratio >= 1.2
            else (
                -1.0
                if rr_ratio < 0.8
                else 0.0
            )
        )

        return results

    def calculate_all_signals(
        self,
        df,
        bids,
        asks,
        current_inventory=0,
        performance_history=None
    ):

        results = self.extract_features(
            df,
            bids,
            asks
        )

        feature_vector = np.array(
            [
                results[k]
                for k in self.feature_names
            ]
        )

        weight_vector = np.array(
            list(
                self.dynamic_weights.values()
            )
        )

        final_score = float(
            np.dot(
                feature_vector,
                weight_vector
            )
        )

        return (
            results,
            final_score,
            self.dynamic_weights
        )


# ============================================================
# RISK ENGINE
# ============================================================

class PowerTradingRiskEngine:

    def calculate_risk_metrics(
        self,
        liquidation_volumes,
        displayed_vol,
        cancelled_vol,
        time_exists,
        obs_window,
        open_interest,
        leverage,
        volatility
    ):

        total_ltz = (
            np.sum(liquidation_volumes)
            if len(liquidation_volumes)
            else 0.0
        )

        max_ltz = (
            np.max(liquidation_volumes)
            if len(liquidation_volumes)
            else 0.0
        )

        ltz_score = (
            max_ltz
            /
            (
                total_ltz
                +
                1e-8
            )
        ) * 100

        spoof_ratio = (
            cancelled_vol
            /
            (
                displayed_vol
                +
                1e-8
            )
        )

        persistence = min(
            max(
                time_exists
                /
                (
                    obs_window
                    +
                    1e-8
                ),
                0
            ),
            1
        )

        spoof_score = (
            spoof_ratio
            *
            (
                1 - persistence
            )
        )

        squeeze_risk = (
            total_ltz
            *
            open_interest
            *
            leverage
            *
            volatility
        )

        market_risk = (
            ltz_score
            +
            spoof_score
            +
            squeeze_risk
        )

        return {
            "LTZ_Score": ltz_score,
            "Spoof_Score": spoof_score,
            "Squeeze_Risk": squeeze_risk,
            "Market_Risk": market_risk,
        }


# ============================================================
# DATA API
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


@st.cache_data(ttl=10)
def fetch_klines_data(
    symbol,
    tf_key,
    limit=150,
    allow_fallback=False
):

    binance_tf = tf_key

    url = (
        "https://data-api.binance.vision/api/v3/klines"
        f"?symbol={symbol}"
        f"&interval={binance_tf}"
        f"&limit={limit}"
    )

    try:

        response = requests.get(
            url,
            timeout=5
        )

        response.raise_for_status()

        res = response.json()

        if (
            isinstance(res, dict)
            or not isinstance(res, list)
        ):
            raise ValueError(
                "Invalid Binance response"
            )

        df = pd.DataFrame(
            res,
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
            ],
        )

        df["Time"] = pd.to_datetime(
            df["Open_Time"],
            unit="ms"
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
                errors="coerce"
            )

        df = df.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
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
        ].reset_index(drop=True)

    except Exception:

        if not allow_fallback:

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

        dates = pd.date_range(
            end=datetime.datetime.now(),
            periods=limit,
            freq=binance_tf
        )

        base_p = 60000.0

        closes = (
            base_p
            +
            np.cumsum(
                np.random.normal(
                    0,
                    10,
                    limit
                )
            )
        )

        return pd.DataFrame(
            {
                "Time": dates,
                "Open": closes - 5,
                "High": closes + 15,
                "Low": closes - 15,
                "Close": closes,
                "Volume": np.random.uniform(
                    50,
                    500,
                    limit
                ),
            }
        )


@st.cache_data(ttl=5)
def fetch_order_book_depth(
    symbol,
    depth_limit=20
):

    url = (
        "https://data-api.binance.vision/api/v3/depth"
        f"?symbol={symbol}"
        f"&limit={depth_limit}"
    )

    try:

        response = requests.get(
            url,
            timeout=5
        )

        response.raise_for_status()

        res = response.json()

        if (
            "bids" not in res
            or
            "asks" not in res
        ):
            raise ValueError(
                "Invalid order book"
            )

        bids = np.array(
            res["bids"],
            dtype=float
        )

        asks = np.array(
            res["asks"],
            dtype=float
        )

        return bids, asks

    except Exception:

        return (
            np.empty((0, 2)),
            np.empty((0, 2))
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "### ⚡ Terminal Controls"
)

selected_symbol = st.sidebar.selectbox(
    "Select Cryptocurrency",
    COINS_LIST,
    index=0
)

selected_tf_label = st.sidebar.selectbox(
    "Select Timeframe",
    list(TIMEFRAME_MAP.keys()),
    index=1
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon Candles",
    5,
    30,
    15
)

st.sidebar.markdown(
    "**Risk / Reward Targets**"
)

st.sidebar.info(
    "TP1 = 1:2  •  TP2 = 1:3"
)

tp1_rr_multiple = 2.0
tp2_rr_multiple = 3.0

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 🎛️ Paper Trading Mode"
)

paper_trading_mode = st.sidebar.toggle(
    "Enable Live Paper Trading",
    value=True
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 🎯 Signal Engine"
)

st.sidebar.caption(
    "Confluence + Regime + Persistence + XGB"
)

st.sidebar.caption(
    "Strong signal requires higher-quality confirmation."
)

if xgb_model is not None:

    st.sidebar.success(
        "XGBoost model: LOADED"
    )

else:

    st.sidebar.error(
        "XGBoost model: NOT LOADED"
    )

    if xgb_model_error:
        st.sidebar.caption(
            xgb_model_error
        )

feedback_count = len(
    _load_feedback()
)

st.sidebar.caption(
    f"Auto-learning feedback: "
    f"{feedback_count} completed trades"
)

if st.session_state.get(
    "xgb_retrain_message"
):

    st.sidebar.info(
        st.session_state[
            "xgb_retrain_message"
        ]
    )

api_interval, tf_minutes = TIMEFRAME_MAP[
    selected_tf_label
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background-color: #080a0f;
    color: #e2e8f0;
}

section[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #161b22;
}

.metric-card {
    background: #111622;
    border: 1px solid #1e2638;
    border-radius: 12px;
    padding: 14px;
    box-shadow:
        0 4px 20px rgba(0,0,0,0.25);
    margin-bottom: 10px;
}

.metric-label {
    font-size: 11px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.metric-val-green {
    font-size: 18px;
    font-weight: 700;
    color: #00e676;
}

.metric-val-red {
    font-size: 18px;
    font-weight: 700;
    color: #ff5252;
}

.metric-val-blue {
    font-size: 18px;
    font-weight: 700;
    color: #38bdf8;
}

.metric-val-yellow {
    font-size: 18px;
    font-weight: 700;
    color: #facc15;
}

.top-status-bar {
    background: #111622;
    border: 1px solid #1e2638;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 18px;
    font-weight: 600;
    font-size: 13px;
}

.signal-reason {
    background: #111622;
    border: 1px solid #1e2638;
    border-radius: 10px;
    padding: 14px;
    line-height: 1.7;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# FETCH DATA
# ============================================================

df = fetch_klines_data(
    selected_symbol,
    api_interval,
    limit=150,
    allow_fallback=False
)

bids, asks = fetch_order_book_depth(
    selected_symbol,
    depth_limit=20
)


# ============================================================
# PENDING TRADE RESOLUTION
# ============================================================

def resolve_pending_trades(
    history,
    symbol,
    timeframe,
    current_candle_time,
    candle_high,
    candle_low
):

    changed = False

    current_candle_str = (
        pd.Timestamp(
            current_candle_time
        )
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    for trade in history:

        if (
            str(
                trade.get(
                    "outcome",
                    ""
                )
            ).upper()
            != "PENDING"
        ):
            continue

        if (
            trade.get("symbol")
            != symbol
            or
            trade.get("timeframe")
            != timeframe
        ):
            continue

        entry_candle = str(
            trade.get(
                "entry_candle_time",
                ""
            )
        )

        if (
            entry_candle
            ==
            current_candle_str
        ):
            continue

        direction = str(
            trade.get(
                "direction",
                ""
            )
        ).upper()

        try:

            entry = float(
                trade.get(
                    "entry_price",
                    0
                )
            )

            sl = float(
                trade.get(
                    "stop_loss",
                    0
                )
            )

            tp = float(
                trade.get(
                    "tp1",
                    0
                )
            )

        except Exception:

            continue

        if (
            entry <= 0
            or
            sl <= 0
            or
            tp <= 0
        ):
            continue

        if direction == "LONG":

            tp_hit = (
                candle_high >= tp
            )

            sl_hit = (
                candle_low <= sl
            )

        elif direction == "SHORT":

            tp_hit = (
                candle_low <= tp
            )

            sl_hit = (
                candle_high >= sl
            )

        else:

            continue

        if not tp_hit and not sl_hit:
            continue

        if tp_hit and sl_hit:

            result = "LOSS"
            exit_price = sl

            reason = (
                "SL & TP touched "
                "in same candle "
                "(SL-first)"
            )

        elif tp_hit:

            result = "WIN"
            exit_price = tp
            reason = "TP1 HIT"

        else:

            result = "LOSS"
            exit_price = sl
            reason = "SL HIT"

        if direction == "LONG":

            pnl = (
                (
                    exit_price
                    -
                    entry
                )
                /
                entry
            ) * 100

        else:

            pnl = (
                (
                    entry
                    -
                    exit_price
                )
                /
                entry
            ) * 100

        trade["outcome"] = result

        trade["exit_price"] = round(
            float(exit_price),
            2
        )

        trade["pnl_percent"] = round(
            float(pnl),
            4
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

        _append_feedback(
            trade
        )

        changed = True

    return changed


def resolve_all_pending_trades(
    history,
    selected_symbol,
    selected_tf_label,
    selected_df
):

    pairs = {
        (
            t.get("symbol"),
            t.get("timeframe")
        )
        for t in history
        if str(
            t.get(
                "outcome",
                ""
            )
        ).upper()
        == "PENDING"
    }

    for symbol, timeframe in pairs:

        if not symbol or not timeframe:
            continue

        if (
            symbol == selected_symbol
            and
            timeframe == selected_tf_label
        ):

            local_df = selected_df

        else:

            try:

                local_df = fetch_klines_data(
                    symbol,
                    timeframe,
                    limit=3,
                    allow_fallback=False
                )

            except Exception:

                continue

        if (
            local_df is None
            or
            local_df.empty
        ):
            continue

        last = local_df.iloc[-1]

        resolve_pending_trades(
            history,
            symbol,
            timeframe,
            pd.Timestamp(
                last["Time"]
            ),
            float(
                last["High"]
            ),
            float(
                last["Low"]
            )
        )


resolve_all_pending_trades(
    st.session_state.trade_history_log,
    selected_symbol,
    selected_tf_label,
    df
)

save_persistent_history(
    st.session_state.trade_history_log
)


# ============================================================
# RETRAIN
# ============================================================

try:

    (
        xgb_model,
        retrain_message
    ) = _retrain_xgb_from_feedback(
        xgb_model
    )

    if retrain_message:

        st.session_state.xgb_retrain_message = (
            retrain_message
        )

except Exception as retrain_error:

    st.session_state.xgb_retrain_message = (
        f"XGB retrain skipped: "
        f"{retrain_error}"
    )


# ============================================================
# MAIN ENGINE
# ============================================================

if (
    not df.empty
    and
    len(df) >= 30
    and
    len(bids) > 0
    and
    len(asks) > 0
):

    # --------------------------------------------------------
    # RESEARCH
    # --------------------------------------------------------

    lab = TenPaperResearchLab()

    (
        paper_results,
        research_score,
        evolved_weights
    ) = lab.calculate_all_signals(
        df,
        bids,
        asks,
        current_inventory=0,
        performance_history=(
            st.session_state.trade_history_log
        )
    )

    close_p = float(
        df["Close"].iloc[-1]
    )

    high_p = float(
        df["High"].iloc[-1]
    )

    low_p = float(
        df["Low"].iloc[-1]
    )

    candle_time = pd.Timestamp(
        df["Time"].iloc[-1]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    tr1 = (
        df["High"]
        -
        df["Low"]
    )

    tr2 = (
        df["High"]
        -
        df["Close"].shift(1)
    ).abs()

    tr3 = (
        df["Low"]
        -
        df["Close"].shift(1)
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1
    ).max(axis=1)

    atr_val = (
        true_range
        .rolling(14)
        .mean()
        .iloc[-1]
    )

    if (
        np.isnan(atr_val)
        or
        atr_val <= 0
    ):

        atr_val = (
            close_p * 0.005
        )

    # --------------------------------------------------------
    # XGB
    # --------------------------------------------------------

    xgb_features = build_xgb_features(
        df,
        bids,
        asks
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

            if hasattr(
                xgb_model,
                "predict_proba"
            ):

                probs = (
                    xgb_model
                    .predict_proba(
                        xgb_features
                    )[0]
                )

                xgb_confidence = float(
                    np.max(probs)
                    *
                    100
                )

            else:

                xgb_confidence = 50.0

            xgb_signal = (
                "LONG"
                if xgb_prediction == 1
                else "SHORT"
            )

        except Exception as e:

            xgb_model_error = (
                f"XGB prediction error: {e}"
            )

            xgb_signal = "NEUTRAL"
            xgb_confidence = 0.0

    # --------------------------------------------------------
    # OBI
    # --------------------------------------------------------

    bid_vol_sum = float(
        np.sum(bids[:, 1])
    )

    ask_vol_sum = float(
        np.sum(asks[:, 1])
    )

    total_depth = (
        bid_vol_sum
        +
        ask_vol_sum
    )

    obi_val = (
        bid_vol_sum
        -
        ask_vol_sum
    ) / (
        total_depth
        +
        1e-8
    )

    # --------------------------------------------------------
    # OFI
    # --------------------------------------------------------

    ofi_val = calculate_ofi(
        bids,
        asks
    )

    ofi_scale = max(
        total_depth,
        1.0
    )

    ofi_norm = float(
        np.clip(
            ofi_val
            /
            ofi_scale,
            -1,
            1
        )
    )

    # --------------------------------------------------------
    # OBI / OFI HISTORY
    # --------------------------------------------------------

    st.session_state.obi_history.append(
        obi_val
    )

    st.session_state.ofi_history.append(
        ofi_norm
    )

    if len(
        st.session_state.obi_history
    ) > 100:

        st.session_state.obi_history = (
            st.session_state
            .obi_history[-100:]
        )

    if len(
        st.session_state.ofi_history
    ) > 100:

        st.session_state.ofi_history = (
            st.session_state
            .ofi_history[-100:]
        )

    obi_z = rolling_zscore(
        st.session_state.obi_history,
        obi_val,
        minimum=10
    )

    ofi_z = rolling_zscore(
        st.session_state.ofi_history,
        ofi_norm,
        minimum=10
    )

    # --------------------------------------------------------
    # MICRO SCORE
    # --------------------------------------------------------

    micro_score = float(
        np.clip(
            0.55 * np.tanh(obi_z / 2.0)
            +
            0.45 * np.tanh(ofi_z / 2.0),
            -1,
            1
        )
    )

    micro_direction = (
        "LONG"
        if micro_score >= 0.18
        else (
            "SHORT"
            if micro_score <= -0.18
            else "NEUTRAL"
        )
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    ema9 = (
        df["Close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema21 = (
        df["Close"]
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema50 = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    momentum5 = (
        df["Close"].iloc[-1]
        /
        (
            df["Close"].iloc[-6]
            +
            1e-8
        )
        -
        1
    )

    ema_slope = (
        ema9
        -
        df["Close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
        .iloc[-4]
    )

    trend_component = (
        (
            ema9 - ema21
        )
        /
        (
            close_p
            +
            1e-8
        )
    ) * 250

    momentum_component = (
        np.sign(momentum5)
        *
        min(
            abs(momentum5)
            *
            1000,
            0.5
        )
    )

    slope_component = (
        np.sign(ema_slope)
        *
        min(
            abs(
                ema_slope
                /
                (
                    close_p
                    +
                    1e-8
                )
            )
            *
            5000,
            0.3
        )
    )

    trend_score = float(
        np.clip(
            trend_component
            +
            momentum_component
            +
            slope_component,
            -1,
            1
        )
    )

    # Stronger trend classification
    if (
        ema9 > ema21
        and
        ema21 > ema50
        and
        momentum5 > 0
        and
        ema_slope > 0
    ):

        trend_direction = "LONG"

    elif (
        ema9 < ema21
        and
        ema21 < ema50
        and
        momentum5 < 0
        and
        ema_slope < 0
    ):

        trend_direction = "SHORT"

    else:

        trend_direction = (
            "LONG"
            if trend_score >= 0.12
            else (
                "SHORT"
                if trend_score <= -0.12
                else "NEUTRAL"
            )
        )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    atr_ratio = (
        atr_val
        /
        (
            close_p
            +
            1e-8
        )
    )

    atr_history = (
        true_range
        /
        df["Close"]
    )

    atr_baseline = (
        atr_history
        .rolling(30)
        .mean()
        .iloc[-1]
    )

    if (
        np.isnan(atr_baseline)
        or
        atr_baseline <= 0
    ):

        atr_baseline = atr_ratio

    volatility_ratio = (
        atr_ratio
        /
        (
            atr_baseline
            +
            1e-8
        )
    )

    volatility_ok = (
        0.70
        <=
        volatility_ratio
        <=
        2.50
    )

    # --------------------------------------------------------
    # SPREAD
    # --------------------------------------------------------

    spread_val = abs(
        asks[0, 0]
        -
        bids[0, 0]
    )

    spread_ratio = (
        spread_val
        /
        (
            close_p
            +
            1e-8
        )
    )

    # For normal liquid crypto markets,
    # extremely wide spread is rejected.
    spread_ok = (
        spread_ratio
        <
        0.0015
    )

    # --------------------------------------------------------
    # RESEARCH DIRECTION
    # --------------------------------------------------------

    research_direction = (
        "LONG"
        if research_score >= 0.10
        else (
            "SHORT"
            if research_score <= -0.10
            else "NEUTRAL"
        )
    )

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    recent_obi = (
        st.session_state
        .obi_history[-3:]
    )

    recent_ofi = (
        st.session_state
        .ofi_history[-3:]
    )

    bullish_obi_persistence = (
        len(recent_obi) >= 3
        and
        all(
            x > 0
            for x in recent_obi
        )
    )

    bearish_obi_persistence = (
        len(recent_obi) >= 3
        and
        all(
            x < 0
            for x in recent_obi
        )
    )

    bullish_ofi_persistence = (
        len(recent_ofi) >= 3
        and
        all(
            x > 0
            for x in recent_ofi
        )
    )

    bearish_ofi_persistence = (
        len(recent_ofi) >= 3
        and
        all(
            x < 0
            for x in recent_ofi
        )
    )

    long_persistence = (
        bullish_obi_persistence
        and
        bullish_ofi_persistence
    )

    short_persistence = (
        bearish_obi_persistence
        and
        bearish_ofi_persistence
    )

    # --------------------------------------------------------
    # COMPONENT SCORES
    # --------------------------------------------------------

    trend_component_score = (
        np.clip(
            trend_score,
            -1,
            1
        )
    )

    micro_component_score = (
        np.clip(
            micro_score,
            -1,
            1
        )
    )

    research_component_score = (
        np.clip(
            research_score,
            -1,
            1
        )
    )

    xgb_component_score = (
        (
            xgb_confidence
            /
            100
        )
        *
        (
            1
            if xgb_signal == "LONG"
            else (
                -1
                if xgb_signal == "SHORT"
                else 0
            )
        )
    )

    momentum_component_score = float(
        np.clip(
            momentum5 * 100,
            -1,
            1
        )
    )

    # --------------------------------------------------------
    # FINAL QUALITY SCORE
    # --------------------------------------------------------

    combined_score = float(
        np.clip(
            0.30
            *
            trend_component_score
            +
            0.25
            *
            micro_component_score
            +
            0.20
            *
            research_component_score
            +
            0.15
            *
            xgb_component_score
            +
            0.10
            *
            momentum_component_score,
            -1,
            1
        )
    )

    raw_quality = (
        abs(combined_score)
        *
        100
    )

    # --------------------------------------------------------
    # HARD DIRECTION FILTERS
    # --------------------------------------------------------

    long_alignment = 0

    short_alignment = 0

    if trend_direction == "LONG":
        long_alignment += 1

    if trend_direction == "SHORT":
        short_alignment += 1

    if research_direction == "LONG":
        long_alignment += 1

    if research_direction == "SHORT":
        short_alignment += 1

    if micro_direction == "LONG":
        long_alignment += 1

    if micro_direction == "SHORT":
        short_alignment += 1

    if xgb_signal == "LONG":
        long_alignment += 1

    if xgb_signal == "SHORT":
        short_alignment += 1

    # --------------------------------------------------------
    # XGB CONFIDENCE GATE
    # --------------------------------------------------------

    xgb_normal_ok = (
        xgb_confidence >= 60
    )

    xgb_strong_ok = (
        xgb_confidence >= 75
    )

    # --------------------------------------------------------
    # NORMAL SIGNAL
    # --------------------------------------------------------

    long_normal = (
        long_alignment >= 3
        and
        xgb_signal == "LONG"
        and
        xgb_normal_ok
        and
        trend_direction != "SHORT"
        and
        micro_direction != "SHORT"
        and
        combined_score >= 0.45
        and
        volatility_ok
        and
        spread_ok
    )

    short_normal = (
        short_alignment >= 3
        and
        xgb_signal == "SHORT"
        and
        xgb_normal_ok
        and
        trend_direction != "LONG"
        and
        micro_direction != "LONG"
        and
        combined_score <= -0.45
        and
        volatility_ok
        and
        spread_ok
    )

    # --------------------------------------------------------
    # STRONG SIGNAL
    # --------------------------------------------------------

    strong_long = (
        long_alignment >= 4
        and
        xgb_signal == "LONG"
        and
        xgb_strong_ok
        and
        trend_direction == "LONG"
        and
        micro_direction == "LONG"
        and
        research_direction == "LONG"
        and
        long_persistence
        and
        combined_score >= 0.70
        and
        volatility_ok
        and
        spread_ok
    )

    strong_short = (
        short_alignment >= 4
        and
        xgb_signal == "SHORT"
        and
        xgb_strong_ok
        and
        trend_direction == "SHORT"
        and
        micro_direction == "SHORT"
        and
        research_direction == "SHORT"
        and
        short_persistence
        and
        combined_score <= -0.70
        and
        volatility_ok
        and
        spread_ok
    )

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    if strong_long:

        direction = "LONG"
        signal_strength = "STRONG LONG"

    elif strong_short:

        direction = "SHORT"
        signal_strength = "STRONG SHORT"

    elif long_normal:

        direction = "LONG"
        signal_strength = "LONG"

    elif short_normal:

        direction = "SHORT"
        signal_strength = "SHORT"

    elif (
        xgb_signal == "LONG"
        and
        long_alignment >= 2
        and
        combined_score >= 0.25
    ):

        direction = "NEUTRAL"
        signal_strength = "CONFIRM LONG"

    elif (
        xgb_signal == "SHORT"
        and
        short_alignment >= 2
        and
        combined_score <= -0.25
    ):

        direction = "NEUTRAL"
        signal_strength = "CONFIRM SHORT"

    else:

        direction = "NEUTRAL"
        signal_strength = "WAIT"

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    confidence = int(
        np.clip(
            raw_quality,
            0,
            99
        )
    )

    # Strong signal quality floor
    if strong_long or strong_short:
        confidence = max(
            confidence,
            85
        )

    # --------------------------------------------------------
    # WHY SIGNAL / WAIT
    # --------------------------------------------------------

    reasons = []

    reasons.append(
        f"Trend: {trend_direction}"
    )

    reasons.append(
        f"Research: {research_direction}"
    )

    reasons.append(
        f"OBI: {obi_val:+.3f}"
    )

    reasons.append(
        f"OBI Z: {obi_z:+.2f}"
    )

    reasons.append(
        f"OFI Z: {ofi_z:+.2f}"
    )

    reasons.append(
        f"Micro: {micro_direction}"
    )

    reasons.append(
        f"XGB: {xgb_signal} "
        f"({xgb_confidence:.1f}%)"
    )

    reasons.append(
        f"Alignment: "
        f"L={long_alignment} "
        f"S={short_alignment}"
    )

    reasons.append(
        f"Persistence: "
        f"{'LONG' if long_persistence else ('SHORT' if short_persistence else 'NO')}"
    )

    reasons.append(
        f"Volatility: "
        f"{'OK' if volatility_ok else 'BAD'}"
    )

    reasons.append(
        f"Spread: "
        f"{'OK' if spread_ok else 'WIDE'}"
    )

    if signal_strength == "WAIT":

        if not volatility_ok:
            reasons.append(
                "WAIT: volatility filter"
            )

        if not spread_ok:
            reasons.append(
                "WAIT: spread too wide"
            )

        if xgb_confidence < 60:
            reasons.append(
                "WAIT: XGB confidence < 60%"
            )

        if (
            long_alignment < 3
            and
            short_alignment < 3
        ):

            reasons.append(
                "WAIT: insufficient confluence"
            )

    why_signal = " | ".join(
        reasons
    )

    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    risk_distance = max(
        float(atr_val),
        close_p * 0.001
    )

    if direction == "LONG":

        sl_val = (
            close_p
            -
            risk_distance
        )

        tp1_val = (
            close_p
            +
            risk_distance
            *
            tp1_rr_multiple
        )

        tp2_val = (
            close_p
            +
            risk_distance
            *
            tp2_rr_multiple
        )

    elif direction == "SHORT":

        sl_val = (
            close_p
            +
            risk_distance
        )

        tp1_val = (
            close_p
            -
            risk_distance
            *
            tp1_rr_multiple
        )

        tp2_val = (
            close_p
            -
            risk_distance
            *
            tp2_rr_multiple
        )

    else:

        sl_val = (
            close_p
            -
            risk_distance
        )

        tp1_val = (
            close_p
            +
            risk_distance
            *
            tp1_rr_multiple
        )

        tp2_val = (
            close_p
            +
            risk_distance
            *
            tp2_rr_multiple
        )

    actual_risk = abs(
        close_p
        -
        sl_val
    )

    tp1_reward = abs(
        tp1_val
        -
        close_p
    )

    tp2_reward = abs(
        tp2_val
        -
        close_p
    )

    actual_rr = (
        tp1_reward
        /
        actual_risk
        if actual_risk > 0
        else 0
    )

    tp2_rr = (
        tp2_reward
        /
        actual_risk
        if actual_risk > 0
        else 0
    )

    beam_level = tp2_val
    base_level = sl_val

    # --------------------------------------------------------
    # SIGNAL LOCK
    # --------------------------------------------------------

    lock_seconds = (
        tf_minutes
        *
        60
    )

    current_time_sec = int(
        time.time()
    )

    time_bucket = (
        current_time_sec
        -
        (
            current_time_sec
            %
            lock_seconds
        )
    )

    time_remaining = (
        lock_seconds
        -
        (
            current_time_sec
            %
            lock_seconds
        )
    )

    trade_id = (
        f"{selected_symbol}_"
        f"{selected_tf_label}_"
        f"{time_bucket}_"
        f"{direction}"
    )

    # --------------------------------------------------------
    # PAPER TRADE
    # --------------------------------------------------------

    if (
        paper_trading_mode
        and
        direction != "NEUTRAL"
    ):

        existing_trade_ids = {
            item.get(
                "trade_id"
            )
            for item
            in st.session_state.trade_history_log
        }

        if trade_id not in existing_trade_ids:

            new_trade = {

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
                    round(
                        close_p,
                        2
                    ),

                "stop_loss":
                    round(
                        sl_val,
                        2
                    ),

                "tp1":
                    round(
                        tp1_val,
                        2
                    ),

                "tp2":
                    round(
                        tp2_val,
                        2
                    ),

                "rr_target":
                    "TP1 1:2 | TP2 1:3",

                "exit_price":
                    None,

                "confidence":
                    confidence,

                "xgb_confidence":
                    round(
                        xgb_confidence,
                        2
                    ),

                "xgb_features_json":
                    json.dumps(
                        {
                            k: float(
                                xgb_features
                                .iloc[0][k]
                            )
                            for k
                            in XGB_FEATURES
                        }
                    ),

                "final_score":
                    round(
                        combined_score,
                        4
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
                new_trade
            )

    save_persistent_history(
        st.session_state.trade_history_log
    )

    # ========================================================
    # RISK
    # ========================================================

    risk_engine = (
        PowerTradingRiskEngine()
    )

    disp_vol = (
        np.sum(asks[:, 1])
        if len(asks)
        else 1.0
    )

    risk_metrics = (
        risk_engine.calculate_risk_metrics(
            liquidation_volumes=np.array(
                [1000, 2500]
            ),
            displayed_vol=disp_vol,
            cancelled_vol=disp_vol * 0.1,
            time_exists=15.0,
            obs_window=60.0,
            open_interest=150000.0,
            leverage=20.0,
            volatility=(
                df["Close"]
                .pct_change()
                .std()
                +
                1e-8
            )
        )
    )

    # ========================================================
    # HEADER
    # ========================================================

    if direction == "LONG":
        dir_color = "#00e676"
    elif direction == "SHORT":
        dir_color = "#ff5252"
    else:
        dir_color = "#38bdf8"

    if signal_strength == "STRONG LONG":
        signal_color = "#00e676"
    elif signal_strength == "STRONG SHORT":
        signal_color = "#ff1744"
    elif signal_strength == "LONG":
        signal_color = "#00e676"
    elif signal_strength == "SHORT":
        signal_color = "#ff5252"
    elif "CONFIRM" in signal_strength:
        signal_color = "#facc15"
    else:
        signal_color = "#38bdf8"

    mins_rem, secs_rem = divmod(
        int(time_remaining),
        60
    )

    st.markdown(
        f"""
<div class="top-status-bar">

🟢 <b>[{selected_symbol}]</b>
&nbsp;|&nbsp;

Price:
<b>${close_p:,.2f}</b>
&nbsp;|&nbsp;

TF:
<b>{selected_tf_label}</b>
&nbsp;|&nbsp;

SIGNAL:
<span style="color:{signal_color}; font-size:16px;">
<b>{signal_strength}</b>
</span>
&nbsp;|&nbsp;

Quality:
<b>{confidence}/100</b>
&nbsp;|&nbsp;

Score:
<b>{combined_score:+.3f}</b>
&nbsp;|&nbsp;

OBI:
<b>{obi_val:+.3f}</b>
&nbsp;|&nbsp;

OFI:
<b>{ofi_val:+.2f}</b>
&nbsp;|&nbsp;

XGB:
<b>{xgb_signal}</b>
({xgb_confidence:.1f}%)
&nbsp;|&nbsp;

Trend:
<b>{trend_direction}</b>
&nbsp;|&nbsp;

⏳ Reset:
<b>{mins_rem}m {secs_rem}s</b>

</div>
""",
        unsafe_allow_html=True
    )

    # ========================================================
    # SIGNAL PANEL
    # ========================================================

    col_sig, col_m1, col_m2, col_m3, col_m4 = (
        st.columns(
            [1.35, 1, 1, 1, 1]
        )
    )

    with col_sig:

        st.markdown(
            f"""
<div class="metric-card"
     style="border-left:4px solid {dir_color};">

<div class="metric-label">
Signal Execution Panel
</div>

<div style="
font-size:25px;
font-weight:800;
color:{signal_color};
">
{signal_strength}
</div>

<div style="
font-size:11px;
color:#8b949e;
margin-top:5px;
">
Quality: {confidence}/100
|
XGB: {xgb_confidence:.1f}%
|
L: {long_alignment}/4
|
S: {short_alignment}/4
</div>

<div style="
font-size:11px;
color:#cbd5e1;
margin-top:6px;
">
Entry:
${close_p:,.2f}
|
SL:
${sl_val:,.2f}
</div>

<div style="
font-size:11px;
color:#38bdf8;
margin-top:4px;
">
TP1:
${tp1_val:,.2f}
|
TP2:
${tp2_val:,.2f}
</div>

</div>
""",
            unsafe_allow_html=True
        )

    with col_m1:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
TP2 / BEAM
</div>

<div class="metric-val-blue">
${beam_level:,.2f}
</div>

</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
BASE / SL
</div>

<div class="metric-val-red">
${base_level:,.2f}
</div>

</div>
""",
            unsafe_allow_html=True
        )

    with col_m2:

        st.markdown(
            """
<div class="metric-card">

<div class="metric-label">
Risk / Reward
</div>

<div class="metric-val-blue">
TP1 1 : 2
</div>

<div class="metric-val-blue">
TP2 1 : 3
</div>

</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
Signal Quality
</div>

<div class="metric-val-green">
{confidence}/100
</div>

</div>
""",
            unsafe_allow_html=True
        )

    with col_m3:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
OBI Z-Score
</div>

<div class="metric-val-blue">
{obi_z:+.2f}
</div>

</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
OFI Z-Score
</div>

<div class="metric-val-blue">
{ofi_z:+.2f}
</div>

</div>
""",
            unsafe_allow_html=True
        )

    with col_m4:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
Volatility
</div>

<div class="
metric-val-{'green' if volatility_ok else 'red'}
">
{volatility_ratio:.2f}x
</div>

</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
Spread
</div>

<div class="metric-val-blue">
${spread_val:.2f}
</div>

</div>
""",
            unsafe_allow_html=True
        )

    # ========================================================
    # WHY SIGNAL
    # ========================================================

    st.markdown(
        "### 🧠 Signal Reasoning"
    )

    st.markdown(
        f"""
<div class="signal-reason">

<b>Current Decision:</b>
<span style="color:{signal_color};">
{signal_strength}
</span>
<br><br>

{why_signal}

</div>
""",
        unsafe_allow_html=True
    )

    # ========================================================
    # CHART
    # ========================================================

    col_chart, col_risk_panel = (
        st.columns(
            [2.5, 1]
        )
    )

    with col_chart:

        st.subheader(
            f"Price Trajectory & Levels "
            f"({selected_symbol})"
        )

        time_delta = pd.Timedelta(
            minutes=tf_minutes
        )

        future_times = [
            df["Time"].iloc[-1]
            +
            i * time_delta
            for i
            in range(
                1,
                forecast_horizon + 1
            )
        ]

        t_steps = np.linspace(
            0,
            np.pi / 2,
            forecast_horizon
        )

        if direction == "LONG":

            forecast_prices = (
                close_p
                +
                (
                    tp2_val
                    -
                    close_p
                )
                *
                np.sin(
                    t_steps
                )
            )

        elif direction == "SHORT":

            forecast_prices = (
                close_p
                -
                (
                    close_p
                    -
                    tp2_val
                )
                *
                np.sin(
                    t_steps
                )
            )

        else:

            forecast_prices = [
                close_p
            ] * forecast_horizon

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df["Time"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candles",
                increasing_line_color="#00e676",
                decreasing_line_color="#ff5252",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    df["Time"].iloc[-1]
                ]
                +
                future_times,

                y=[
                    close_p
                ]
                +
                list(
                    forecast_prices
                ),

                mode="lines+markers",

                name="Trajectory",

                line=dict(
                    color=dir_color,
                    width=2,
                    dash="dot"
                )
            )
        )

        fig.add_hline(
            y=tp2_val,
            line_dash="dash",
            line_color="#00e676",
            annotation_text=(
                f"TP2: ${tp2_val:,.2f}"
            )
        )

        fig.add_hline(
            y=tp1_val,
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text=(
                f"TP1: ${tp1_val:,.2f}"
            )
        )

        fig.add_hline(
            y=sl_val,
            line_dash="dash",
            line_color="#ff5252",
            annotation_text=(
                f"SL: ${sl_val:,.2f}"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=440,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#111622",
            plot_bgcolor="#111622",
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ========================================================
    # MICROSTRUCTURE
    # ========================================================

    with col_risk_panel:

        st.subheader(
            "Market Microstructure"
        )

        st.markdown(
            f"""
<div class="metric-card">

<div style="
display:flex;
justify-content:space-between;
margin-bottom:7px;
">
<span>Bid Volume</span>
<b style="color:#00e676;">
{bid_vol_sum:,.2f}
</b>
</div>

<div style="
display:flex;
justify-content:space-between;
margin-bottom:7px;
">
<span>Ask Volume</span>
<b style="color:#ff5252;">
{ask_vol_sum:,.2f}
</b>
</div>

<div style="
display:flex;
justify-content:space-between;
margin-bottom:7px;
">
<span>OBI</span>
<b style="color:#38bdf8;">
{obi_val:+.3f}
</b>
</div>

<div style="
display:flex;
justify-content:space-between;
margin-bottom:7px;
">
<span>OBI Z</span>
<b style="color:#38bdf8;">
{obi_z:+.2f}
</b>
</div>

<div style="
display:flex;
justify-content:space-between;
margin-bottom:7px;
">
<span>OFI Z</span>
<b style="color:#38bdf8;">
{ofi_z:+.2f}
</b>
</div>

<div style="
display:flex;
justify-content:space-between;
">
<span>Persistence</span>
<b style="color:{'#00e676' if long_persistence or short_persistence else '#facc15'};">
{
    "LONG"
    if long_persistence
    else (
        "SHORT"
        if short_persistence
        else "WAIT"
    )
}
</b>
</div>

</div>
""",
            unsafe_allow_html=True
        )

        st.subheader(
            "Top 20 OBI Analysis"
        )

        fig_obi = go.Figure(
            go.Bar(
                x=[
                    "Top 5",
                    "Top 10",
                    "Top 20"
                ],
                y=[
                    obi_val * 0.75,
                    obi_val * 0.90,
                    obi_val
                ],
                marker_color="#38bdf8"
            )
        )

        fig_obi.update_layout(
            height=180,
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            ),
            paper_bgcolor="#111622",
            plot_bgcolor="#111622"
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

    st.subheader(
        "🔬 12-Paper Quantitative Research Scoreboard"
    )

    col_sc1, col_sc2 = (
        st.columns(
            [1.5, 1]
        )
    )

    with col_sc1:

        paper_table_data = []

        for k, v in paper_results.items():

            status = (
                "PASS 🟢"
                if v > 0.10
                else (
                    "FAIL 🔴"
                    if v < -0.10
                    else
                    "NEUTRAL ⚪"
                )
            )

            paper_table_data.append(
                {
                    "Paper": k,
                    "Value": f"{v:+.3f}",
                    "Weight":
                        f"{evolved_weights.get(k, 0.083) * 100:.1f}%",
                    "Status": status,
                }
            )

        st.dataframe(
            pd.DataFrame(
                paper_table_data
            ),
            use_container_width=True,
            hide_index=True,
            height=280
        )

    with col_sc2:

        st.markdown(
            """
<div class="metric-card">

<div style="
font-weight:700;
color:#38bdf8;
margin-bottom:8px;
">
Advanced Signal Engine
</div>

<div style="
font-size:12px;
color:#cbd5e1;
line-height:1.8;
">

• Trend regime<br>
• OBI Z-score<br>
• OFI Z-score<br>
• Order-flow persistence<br>
• XGBoost confirmation<br>
• Momentum confirmation<br>
• Volatility filter<br>
• Spread filter<br>
• 1:2 TP1<br>
• 1:3 TP2

</div>

</div>
""",
            unsafe_allow_html=True
        )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Performance Summary & Win Rate"
    )

    if st.session_state.trade_history_log:

        df_log = pd.DataFrame(
            st.session_state.trade_history_log
        )

        f_col1, f_col2, f_col3 = (
            st.columns(3)
        )

        with f_col1:

            coin_filter = st.selectbox(
                "Filter Coin",
                ["ALL"] + COINS_LIST
            )

        with f_col2:

            tf_filter = st.selectbox(
                "Filter Timeframe",
                ["ALL"]
                +
                list(
                    TIMEFRAME_MAP.keys()
                )
            )

        with f_col3:

            dir_filter = st.selectbox(
                "Filter Direction",
                [
                    "ALL",
                    "LONG",
                    "SHORT"
                ]
            )

        filtered_df = (
            df_log.copy()
        )

        if coin_filter != "ALL":

            filtered_df = (
                filtered_df[
                    filtered_df[
                        "symbol"
                    ]
                    ==
                    coin_filter
                ]
            )

        if tf_filter != "ALL":

            filtered_df = (
                filtered_df[
                    filtered_df[
                        "timeframe"
                    ]
                    ==
                    tf_filter
                ]
            )

        if dir_filter != "ALL":

            filtered_df = (
                filtered_df[
                    filtered_df[
                        "direction"
                    ]
                    ==
                    dir_filter
                ]
            )

        total_signals = len(
            filtered_df
        )

        wins = len(
            filtered_df[
                filtered_df[
                    "outcome"
                ]
                ==
                "WIN"
            ]
        )

        losses = len(
            filtered_df[
                filtered_df[
                    "outcome"
                ]
                ==
                "LOSS"
            ]
        )

        pending = len(
            filtered_df[
                filtered_df[
                    "outcome"
                ]
                ==
                "PENDING"
            ]
        )

        closed_trades = (
            wins
            +
            losses
        )

        win_rate = (
            wins
            /
            closed_trades
            *
            100
            if closed_trades > 0
            else 0
        )

        winning_trades_df = (
            filtered_df[
                filtered_df[
                    "outcome"
                ]
                ==
                "WIN"
            ]
        )

        losing_trades_df = (
            filtered_df[
                filtered_df[
                    "outcome"
                ]
                ==
                "LOSS"
            ]
        )

        gross_profit = (
            winning_trades_df[
                "pnl_percent"
            ].sum()
            if not winning_trades_df.empty
            else 0
        )

        gross_loss = abs(
            losing_trades_df[
                "pnl_percent"
            ].sum()
        ) if not losing_trades_df.empty else 0

        net_pnl = (
            gross_profit
            -
            gross_loss
        )

        profit_factor = (
            gross_profit
            /
            gross_loss
            if gross_loss > 0
            else (
                gross_profit
                if gross_profit > 0
                else 0
            )
        )

        p1, p2, p3, p4, p5, p6 = (
            st.columns(6)
        )

        with p1:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-label">
Win Rate
</div>

<div class="metric-val-green">
{win_rate:.1f}%
</div>

</div>
""",
                unsafe_allow_html=True
            )

        with p2:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-label">
Closed Trades
</div>

<div class="metric-val-blue">
{closed_trades}
</div>

</div>
""",
                unsafe_allow_html=True
            )

        with p3:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-label">
Wins / Losses
</div>

<div class="metric-val-green">
{wins}W / {losses}L
</div>

</div>
""",
                unsafe_allow_html=True
            )

        with p4:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-label">
Pending
</div>

<div class="metric-val-blue">
{pending}
</div>

</div>
""",
                unsafe_allow_html=True
            )

        with p5:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-label">
Profit Factor
</div>

<div class="metric-val-blue">
{profit_factor:.2f}
</div>

</div>
""",
                unsafe_allow_html=True
            )

        with p6:

            pnl_color = (
                "#00e676"
                if net_pnl >= 0
                else "#ff5252"
            )

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-label">
Net PnL %
</div>

<div style="
font-size:18px;
font-weight:700;
color:{pnl_color};
">
{net_pnl:+.2f}%
</div>

</div>
""",
                unsafe_allow_html=True
            )

        st.markdown(
            "##### Detailed Trade History"
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
            "pnl_percent",
            "outcome",
            "confidence",
            "xgb_confidence",
            "exit_reason",
        ]

        default_values = {
            "timestamp": "",
            "symbol": "",
            "timeframe": "",
            "direction": "",
            "signal_strength": "",
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "tp1": 0.0,
            "tp2": 0.0,
            "rr_target":
                "TP1 1:2 | TP2 1:3",
            "exit_price": 0.0,
            "pnl_percent": 0.0,
            "outcome": "PENDING",
            "confidence": 0.0,
            "xgb_confidence": 0.0,
            "exit_reason": "",
        }

        for col in display_cols:

            if col not in filtered_df.columns:

                filtered_df[col] = (
                    default_values[col]
                )

        st.dataframe(
            filtered_df[
                display_cols
            ],
            use_container_width=True,
            hide_index=True,
            height=300
        )

        if st.sidebar.button(
            "Clear Trade History Log"
        ):

            st.session_state.trade_history_log = []

            if os.path.exists(
                CSV_FILE
            ):
                os.remove(
                    CSV_FILE
                )

            if os.path.exists(
                FEEDBACK_FILE
            ):
                os.remove(
                    FEEDBACK_FILE
                )

            st.session_state.xgb_last_retrain_count = 0

            st.rerun()

    else:

        st.info(
            "No paper trade history recorded yet."
        )


# ============================================================
# DATA PIPELINE ERROR
# ============================================================

else:

    st.warning(
        "⚠️ Data pipeline is not ready."
    )

    if df.empty:

        st.error(
            f"Binance candle data unavailable "
            f"for {selected_symbol} / "
            f"{api_interval}."
        )

    if len(bids) == 0 or len(asks) == 0:

        st.error(
            "Binance order-book data unavailable."
        )

    st.info(
        "Signal generation waits for real market "
        "data instead of generating fake signals."
    )
