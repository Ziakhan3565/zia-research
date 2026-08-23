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

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
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
# XGBOOST FEATURES
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
# HISTORY
# ============================================================

def load_persistent_history():

    if not os.path.exists(CSV_FILE):
        return []

    try:

        df_hist = pd.read_csv(CSV_FILE)

        expected_cols = [
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

        for col in expected_cols:

            if col not in df_hist.columns:

                if col in [
                    "outcome"
                ]:
                    df_hist[col] = "PENDING"

                elif col in [
                    "signal_strength",
                    "exit_reason",
                    "entry_candle_time",
                    "exit_time"
                ]:
                    df_hist[col] = ""

                else:
                    df_hist[col] = 0.0

        return df_hist.to_dict("records")

    except Exception:
        return []


def save_persistent_history(history_list):

    try:

        pd.DataFrame(history_list).to_csv(
            CSV_FILE,
            index=False
        )

    except Exception as e:

        st.error(
            f"Error saving history to CSV: {e}"
        )


if "trade_history_log" not in st.session_state:

    st.session_state.trade_history_log = (
        load_persistent_history()
    )


# ============================================================
# XGBOOST MODEL
# ============================================================

@st.cache_resource
def load_xgb_model():

    if not os.path.exists(MODEL_PATH):

        return None, (
            f"Model file not found: {MODEL_PATH}"
        )

    try:

        model = joblib.load(MODEL_PATH)

        return model, None

    except Exception as e:

        return None, (
            f"XGBoost load error: {e}"
        )


xgb_model, xgb_model_error = load_xgb_model()


# ============================================================
# FEEDBACK
# ============================================================

def _load_feedback():

    empty = pd.DataFrame(
        columns=XGB_FEATURES + [
            "target",
            "trade_id",
            "closed_at"
        ]
    )

    if not os.path.exists(FEEDBACK_FILE):
        return empty

    try:

        fb = pd.read_csv(FEEDBACK_FILE)

        required = (
            XGB_FEATURES +
            ["target"]
        )

        for c in required:

            if c not in fb.columns:
                return empty

        return fb.dropna(
            subset=required
        ).copy()

    except Exception:

        return empty


def _append_feedback(trade):

    raw = trade.get(
        "xgb_features_json",
        ""
    )

    outcome = str(
        trade.get(
            "outcome",
            ""
        )
    ).upper()

    if not raw:
        return

    if outcome not in [
        "WIN",
        "LOSS"
    ]:
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
            trade.get(
                "direction",
                ""
            )
        ).upper()

        # Correct directional target
        row["target"] = int(
            (direction == "LONG")
            ==
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

        fb = _load_feedback()

        existing = set(
            fb.get(
                "trade_id",
                pd.Series(dtype=str)
            ).astype(str)
        )

        if str(row["trade_id"]) in existing:
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


def _retrain_xgb_from_feedback(
    current_model
):

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

    if len(fb) < (
        last_count +
        RETRAIN_EVERY
    ):
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

    X_train = fb.iloc[
        :split
    ][XGB_FEATURES]

    X_test = fb.iloc[
        split:
    ][XGB_FEATURES]

    y_train = fb.iloc[
        :split
    ]["target"].astype(int)

    y_test = fb.iloc[
        split:
    ]["target"].astype(int)

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

    if test_acc < MIN_TEST_ACCURACY:

        st.session_state.xgb_last_retrain_count = len(
            fb
        )

        return (
            current_model,
            (
                "XGB retrain rejected: "
                f"holdout accuracy "
                f"{test_acc * 100:.1f}%"
            )
        )

    tmp_path = (
        MODEL_PATH +
        ".tmp"
    )

    joblib.dump(
        candidate,
        tmp_path
    )

    os.replace(
        tmp_path,
        MODEL_PATH
    )

    st.session_state.xgb_last_retrain_count = len(
        fb
    )

    return (
        candidate,
        (
            f"XGB retrained from "
            f"{len(fb)} trades | "
            f"holdout accuracy "
            f"{test_acc * 100:.1f}%"
        )
    )


# ============================================================
# XGB FEATURES
# ============================================================

def build_xgb_features(
    df,
    bids,
    asks
):

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
        bid_sum /
        (ask_sum + 1e-5)
    )

    total_depth = (
        bid_sum +
        ask_sum
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
# OFI
# ============================================================

def calculate_ofi(
    current_bids,
    current_asks
):

    prev = st.session_state.get(
        "previous_orderbook"
    )

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

    if prev is None:

        ofi = 0.0

    else:

        prev_bid_sum, prev_ask_sum = prev

        ofi = (
            current_bid_sum -
            prev_bid_sum
        ) - (
            current_ask_sum -
            prev_ask_sum
        )

    st.session_state.previous_orderbook = (
        current_bid_sum,
        current_ask_sum
    )

    return float(ofi)


# ============================================================
# NORMALIZE TRADE
# ============================================================

def normalize_trade(trade):

    trade["outcome"] = str(
        trade.get(
            "outcome",
            "PENDING"
        )
    ).upper()

    trade["status"] = (
        "Closed"
        if trade["outcome"]
        in ["WIN", "LOSS"]
        else "Open"
    )

    numeric = [
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

    for key in numeric:

        try:

            value = trade.get(
                key,
                0.0
            )

            if value is None:
                value = 0.0

            trade[key] = float(value)

        except Exception:

            trade[key] = 0.0

    if not trade.get("rr_target"):

        trade["rr_target"] = (
            "TP1 1:2 | TP2 1:3"
        )

    if not trade.get(
        "exit_reason"
    ):

        trade["exit_reason"] = ""

    if not trade.get(
        "entry_candle_time"
    ):

        trade["entry_candle_time"] = (
            trade.get(
                "timestamp",
                ""
            )
        )

    if "xgb_features_json" not in trade:

        trade[
            "xgb_features_json"
        ] = ""

    if "signal_strength" not in trade:

        trade[
            "signal_strength"
        ] = ""

    return trade


st.session_state.trade_history_log = [
    normalize_trade(t)
    for t in st.session_state.trade_history_log
]


# ============================================================
# RESOLVE TRADES
# ============================================================

def resolve_pending_trades(
    history,
    symbol,
    timeframe,
    current_candle_time,
    candle_high,
    candle_low,
    current_price
):

    changed = False

    current_candle_str = (
        pd.Timestamp(
            current_candle_time
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    for trade in history:

        if str(
            trade.get(
                "outcome",
                ""
            )
        ).upper() != "PENDING":

            continue

        if trade.get(
            "symbol"
        ) != symbol:

            continue

        if trade.get(
            "timeframe"
        ) != timeframe:

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

        if (
            entry <= 0
            or sl <= 0
            or tp <= 0
        ):

            continue

        if direction == "LONG":

            tp_hit = (
                float(candle_high)
                >= tp
            )

            sl_hit = (
                float(candle_low)
                <= sl
            )

        else:

            tp_hit = (
                float(candle_low)
                <= tp
            )

            sl_hit = (
                float(candle_high)
                >= sl
            )

        if not tp_hit and not sl_hit:
            continue

        if tp_hit and sl_hit:

            result = "LOSS"
            exit_price = sl
            reason = (
                "SL & TP same candle "
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
                (exit_price - entry)
                /
                entry
            ) * 100

        else:

            pnl = (
                (entry - exit_price)
                /
                entry
            ) * 100

        trade["outcome"] = result
        trade["exit_price"] = round(
            exit_price,
            2
        )
        trade["pnl_percent"] = round(
            pnl,
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


# ============================================================
# 12-PAPER RESEARCH ENGINE
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
            k: 1.0 /
            len(self.feature_names)
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

        bid_vol = np.sum(
            bids[:, 1]
        )

        ask_vol = np.sum(
            asks[:, 1]
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
            returns.std()
            +
            1e-8
        )

        returns_h = (
            (
                df["Close"].iloc[-1]
                -
                df["Close"].iloc[-5]
            )
            /
            (
                df["Close"].iloc[-5]
                +
                1e-8
            )
        )

        delta_p = (
            df["Close"].iloc[-1]
            -
            df["Close"].iloc[-2]
        )

        vol_changes = (
            df["Volume"]
            .pct_change()
            .dropna()
            .values
        )

        if len(vol_changes) >= 15:

            hawkes_intensity = (
                np.mean(
                    vol_changes[-3:]
                )
                /
                (
                    np.mean(
                        vol_changes[-15:]
                    )
                    +
                    1e-8
                )
            )

        else:

            hawkes_intensity = 1.0

        results["HAWKES"] = np.clip(
            (
                hawkes_intensity
                - 1.0
            )
            *
            np.sign(returns_h),
            -1,
            1
        )

        results["BOOK_IMB"] = (
            bid_vol -
            ask_vol
        ) / (
            bid_vol +
            ask_vol +
            1e-8
        )

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
            taker_buy -
            taker_sell
        ) / (
            taker_buy +
            taker_sell +
            1e-8
        )

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

        prior = 0.745

        likelihood = (
            1.0
            if results["BOOK_IMB"] > 0
            else 0.25
        )

        posterior = (
            likelihood * prior
        ) / (
            (
                likelihood * prior
            )
            +
            (
                (1 - likelihood)
                *
                (1 - prior)
            )
            +
            1e-8
        )

        results["BAYESIAN"] = np.clip(
            (posterior - 0.5)
            * 2.0,
            -1,
            1
        )

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

        results["QUANTILES"] = np.clip(
            (
                (
                    returns_h -
                    q10
                )
                /
                (
                    q90 -
                    q10 +
                    1e-8
                )
                * 2.0
                - 1.0
            ),
            -1,
            1
        )

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
                ma_fast -
                ma_slow
            )
            /
            (
                realized_vol *
                mid_price +
                1e-8
            ),
            -1,
            1
        )

        win_prob = (
            0.55
            +
            0.15 *
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
            2.0
            *
            np.sign(
                returns_h
            ),
            -1,
            1
        )

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
        ) / 3.0

        results["RMT_DOM"] = np.clip(
            rmt_dom *
            np.sign(returns_h),
            -1,
            1
        )

        conformal_spread = (
            realized_vol * 1.96
        )

        upper_b = (
            mid_price *
            (
                1 +
                conformal_spread
            )
        )

        lower_b = (
            mid_price *
            (
                1 -
                conformal_spread
            )
        )

        results["CONF_CROSS"] = (
            1.0
            if mid_price >
            (
                upper_b +
                lower_b
            ) / 2
            else (
                -1.0
                if mid_price <
                (
                    upper_b +
                    lower_b
                ) / 2
                else 0.0
            )
        )

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
        ).reshape(1, -1)

        weight_vector = np.array(
            list(
                self.dynamic_weights.values()
            )
        )

        final_score = float(
            np.dot(
                feature_vector[0],
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
                total_ltz +
                1e-8
            )
        ) * 100

        spoof_ratio = (
            cancelled_vol
            /
            (
                displayed_vol +
                1e-8
            )
        )

        persistence = min(
            max(
                time_exists
                /
                (
                    obs_window +
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
                1 -
                persistence
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
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
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
        0 4px 20px
        rgba(0,0,0,0.25);
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

.top-status-bar {
    background: #111622;
    border: 1px solid #1e2638;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 18px;
    font-weight: 600;
    font-size: 13px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
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

    "1m (Scalping)": (
        "1m",
        1
    ),

    "15m (Medium TF)": (
        "15m",
        15
    ),

    "30m (Medium TF)": (
        "30m",
        30
    ),

    "1h (Intraday)": (
        "1h",
        60
    ),

    "4h (Intraday)": (
        "4h",
        240
    ),
}


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


api_interval, tf_minutes = (
    TIMEFRAME_MAP[
        selected_tf_label
    ]
)


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=15)
def fetch_klines_data(
    symbol,
    tf_key,
    limit=100,
    allow_fallback=True
):

    binance_tf = (
        "1m"
        if "1m" in tf_key
        else (
            "15m"
            if "15m" in tf_key
            else (
                "30m"
                if "30m" in tf_key
                else (
                    "1h"
                    if "1h" in tf_key
                    else "4h"
                )
            )
        )
    )

    url = (
        "https://data-api.binance.vision"
        f"/api/v3/klines?"
        f"symbol={symbol}"
        f"&interval={binance_tf}"
        f"&limit={limit}"
    )

    try:

        res = requests.get(
            url,
            timeout=4
        ).json()

        if (
            isinstance(res, dict)
            or not isinstance(res, list)
        ):

            raise ValueError(
                "API invalid response"
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
            ]
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
            "Volume"
        ]:

            df[col] = df[col].astype(
                float
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
        ]

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

    try:

        url = (
            "https://data-api.binance.vision"
            f"/api/v3/depth?"
            f"symbol={symbol}"
            f"&limit={depth_limit}"
        )

        res = requests.get(
            url,
            timeout=4
        ).json()

        if (
            "bids" in res
            and
            "asks" in res
        ):

            return (
                np.array(
                    res["bids"],
                    dtype=float
                ),
                np.array(
                    res["asks"],
                    dtype=float
                )
            )

    except Exception:
        pass

    dummy_bids = np.array(
        [
            [
                60000 - i * 2,
                1.5
            ]
            for i in range(20)
        ],
        dtype=float
    )

    dummy_asks = np.array(
        [
            [
                60000 + i * 2,
                1.5
            ]
            for i in range(20)
        ],
        dtype=float
    )

    return (
        dummy_bids,
        dummy_asks
    )


df = fetch_klines_data(
    selected_symbol,
    selected_tf_label
)

bids, asks = fetch_order_book_depth(
    selected_symbol
)


# ============================================================
# RESOLVE ALL TRADES
# ============================================================

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
                    limit=2,
                    allow_fallback=False
                )

            except Exception:

                continue

        if (
            local_df is None
            or local_df.empty
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
            float(last["High"]),
            float(last["Low"]),
            float(last["Close"])
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

    xgb_model, retrain_message = (
        _retrain_xgb_from_feedback(
            xgb_model
        )
    )

    if retrain_message:

        st.session_state[
            "xgb_retrain_message"
        ] = retrain_message

except Exception as err:

    st.session_state[
        "xgb_retrain_message"
    ] = (
        f"XGB retrain skipped: {err}"
    )


# ============================================================
# SIGNAL ENGINE
# ============================================================

if (
    not df.empty
    and len(df) >= 20
    and len(bids) > 0
    and len(asks) > 0
):

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
        performance_history=
            st.session_state.trade_history_log
    )

    close_p = float(
        df["Close"].iloc[-1]
    )

    candle_time = pd.Timestamp(
        df["Time"].iloc[-1]
    )

    # ========================================================
    # ATR
    # ========================================================

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
        [tr1, tr2, tr3],
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

        atr_val = close_p * 0.005


    # ========================================================
    # XGBOOST
    # ========================================================

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

            probs = (
                xgb_model
                .predict_proba(
                    xgb_features
                )[0]
            )

            xgb_confidence = float(
                np.max(probs) * 100
            )

            xgb_signal = (
                "LONG"
                if xgb_prediction == 1
                else "SHORT"
            )

        except Exception as e:

            xgb_model_error = (
                f"XGBoost prediction error: {e}"
            )


    # ========================================================
    # ORDER BOOK
    # ========================================================

    bid_vol_sum = float(
        np.sum(bids[:, 1])
    )

    ask_vol_sum = float(
        np.sum(asks[:, 1])
    )

    obi_val = (
        bid_vol_sum -
        ask_vol_sum
    ) / (
        bid_vol_sum +
        ask_vol_sum +
        1e-8
    )

    ofi_val = calculate_ofi(
        bids,
        asks
    )

    depth_scale = max(
        bid_vol_sum +
        ask_vol_sum,
        1.0
    )

    ofi_norm = float(
        np.clip(
            ofi_val /
            depth_scale,
            -1,
            1
        )
    )

    micro_score = float(
        np.clip(
            0.60 * obi_val
            +
            0.40 * ofi_norm,
            -1,
            1
        )
    )


    # ========================================================
    # EMA TREND
    # ========================================================

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

    momentum5 = float(
        df["Close"].iloc[-1]
        /
        (
            df["Close"].iloc[-6]
            +
            1e-8
        )
        - 1
    )

    momentum10 = float(
        df["Close"].iloc[-1]
        /
        (
            df["Close"].iloc[-11]
            +
            1e-8
        )
        - 1
    )


    # ========================================================
    # TREND SCORE
    # ========================================================

    ema_fast_score = (
        (ema9 - ema21)
        /
        close_p
    ) * 300

    ema_slow_score = (
        (ema21 - ema50)
        /
        close_p
    ) * 150

    momentum_score = (
        np.sign(momentum5)
        *
        min(
            abs(momentum5) * 800,
            0.50
        )
    )

    trend_score = float(
        np.clip(
            ema_fast_score
            +
            ema_slow_score
            +
            momentum_score,
            -1,
            1
        )
    )


    # ========================================================
    # TREND DIRECTION
    # ========================================================

    if (
        ema9 > ema21
        and
        ema21 > ema50
        and
        momentum5 > 0
    ):

        trend_direction = "LONG"

    elif (
        ema9 < ema21
        and
        ema21 < ema50
        and
        momentum5 < 0
    ):

        trend_direction = "SHORT"

    else:

        trend_direction = "NEUTRAL"


    # ========================================================
    # RESEARCH DIRECTION
    # ========================================================

    if research_score >= 0.12:

        research_direction = "LONG"

    elif research_score <= -0.12:

        research_direction = "SHORT"

    else:

        research_direction = "NEUTRAL"


    # ========================================================
    # OBI / OFI DIRECTION
    # ========================================================

    if (
        obi_val >= 0.10
        and
        ofi_norm >= 0.05
    ):

        micro_direction = "LONG"

    elif (
        obi_val <= -0.10
        and
        ofi_norm <= -0.05
    ):

        micro_direction = "SHORT"

    else:

        micro_direction = "NEUTRAL"


    # ========================================================
    # XGB SIGNED SCORE
    # ========================================================

    if xgb_signal == "LONG":

        xgb_signed = (
            xgb_confidence /
            100
        )

    elif xgb_signal == "SHORT":

        xgb_signed = -(
            xgb_confidence /
            100
        )

    else:

        xgb_signed = 0.0


    # ========================================================
    # COMBINED SCORE
    # ========================================================

    combined_score = float(
        np.clip(
            0.40 * xgb_signed
            +
            0.25 * research_score
            +
            0.20 * micro_score
            +
            0.15 * trend_score,
            -1,
            1
        )
    )


    # ========================================================
    # CONFIRMATION VOTES
    # ========================================================

    long_votes = sum(
        [
            xgb_signal == "LONG",
            research_direction == "LONG",
            micro_direction == "LONG",
            trend_direction == "LONG",
        ]
    )

    short_votes = sum(
        [
            xgb_signal == "SHORT",
            research_direction == "SHORT",
            micro_direction == "SHORT",
            trend_direction == "SHORT",
        ]
    )


    # ========================================================
    # STRONG LONG CONDITIONS
    # ========================================================

    strong_long = (

        xgb_signal == "LONG"

        and

        xgb_confidence >= 80

        and

        research_score >= 0.20

        and

        obi_val >= 0.18

        and

        ofi_norm >= 0.08

        and

        trend_direction == "LONG"

        and

        trend_score >= 0.15

        and

        long_votes >= 4

        and

        combined_score >= 0.42
    )


    # ========================================================
    # NORMAL LONG CONDITIONS
    # ========================================================

    normal_long = (

        xgb_signal == "LONG"

        and

        xgb_confidence >= 65

        and

        long_votes >= 3

        and

        combined_score >= 0.18

        and

        trend_direction != "SHORT"

        and

        not (
            obi_val <= -0.15
            and
            ofi_norm <= -0.08
        )
    )


    # ========================================================
    # STRONG SHORT CONDITIONS
    # ========================================================

    strong_short = (

        xgb_signal == "SHORT"

        and

        xgb_confidence >= 80

        and

        research_score <= -0.20

        and

        obi_val <= -0.18

        and

        ofi_norm <= -0.08

        and

        trend_direction == "SHORT"

        and

        trend_score <= -0.15

        and

        short_votes >= 4

        and

        combined_score <= -0.42
    )


    # ========================================================
    # NORMAL SHORT CONDITIONS
    # ========================================================

    normal_short = (

        xgb_signal == "SHORT"

        and

        xgb_confidence >= 65

        and

        short_votes >= 3

        and

        combined_score <= -0.18

        and

        trend_direction != "LONG"

        and

        not (
            obi_val >= 0.15
            and
            ofi_norm >= 0.08
        )
    )


    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    if strong_long:

        direction = "LONG"

        signal_strength = (
            "STRONG LONG"
        )

    elif strong_short:

        direction = "SHORT"

        signal_strength = (
            "STRONG SHORT"
        )

    elif normal_long:

        direction = "LONG"

        signal_strength = "LONG"

    elif normal_short:

        direction = "SHORT"

        signal_strength = "SHORT"

    else:

        direction = "NEUTRAL"

        signal_strength = "WAIT"


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = int(
        np.clip(
            abs(combined_score)
            * 100,
            0,
            99
        )
    )


    # ========================================================
    # ATR RISK / TARGETS
    # ========================================================

    risk_distance = max(
        float(atr_val),
        close_p * 0.001
    )


    if direction == "LONG":

        sl_val = (
            close_p -
            risk_distance
        )

        tp1_val = (
            close_p +
            risk_distance *
            tp1_rr_multiple
        )

        tp2_val = (
            close_p +
            risk_distance *
            tp2_rr_multiple
        )

    elif direction == "SHORT":

        sl_val = (
            close_p +
            risk_distance
        )

        tp1_val = (
            close_p -
            risk_distance *
            tp1_rr_multiple
        )

        tp2_val = (
            close_p -
            risk_distance *
            tp2_rr_multiple
        )

    else:

        sl_val = (
            close_p -
            risk_distance
        )

        tp1_val = (
            close_p +
            risk_distance *
            tp1_rr_multiple
        )

        tp2_val = (
            close_p +
            risk_distance *
            tp2_rr_multiple
        )


    actual_risk = abs(
        close_p -
        sl_val
    )

    actual_rr = (
        abs(tp1_val - close_p)
        /
        actual_risk
        if actual_risk > 0
        else 0
    )

    tp2_rr = (
        abs(tp2_val - close_p)
        /
        actual_risk
        if actual_risk > 0
        else 0
    )


    beam_level = tp2_val
    base_level = sl_val


    # ========================================================
    # CANDLE TIMER
    # ========================================================

    lock_seconds = (
        tf_minutes *
        60
    )

    current_time_sec = int(
        time.time()
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

    time_bucket = (
        current_time_sec
        -
        (
            current_time_sec
            %
            lock_seconds
        )
    )


    # ========================================================
    # PAPER TRADE
    # ========================================================

    trade_id = (
        f"{selected_symbol}_"
        f"{selected_tf_label}_"
        f"{time_bucket}_"
        f"{direction}"
    )


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

                "trade_id":
                    trade_id,

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
                    0.0,

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
                            k:
                            float(
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
                        3
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
        else 1
    )

    risk_metrics = (
        risk_engine.calculate_risk_metrics(
            liquidation_volumes=np.array(
                [
                    1000,
                    2500
                ]
            ),
            displayed_vol=disp_vol,
            cancelled_vol=
                disp_vol * 0.1,
            time_exists=15.0,
            obs_window=60.0,
            open_interest=150000.0,
            leverage=20.0,
            volatility=
                df["Close"]
                .pct_change()
                .std()
                +
                1e-8
        )
    )


    # ========================================================
    # COLORS
    # ========================================================

    if signal_strength == "STRONG LONG":

        dir_color = "#00e676"
        signal_color = "#00ff88"

    elif signal_strength == "LONG":

        dir_color = "#00e676"
        signal_color = "#00e676"

    elif signal_strength == "STRONG SHORT":

        dir_color = "#ff1744"
        signal_color = "#ff1744"

    elif signal_strength == "SHORT":

        dir_color = "#ff5252"
        signal_color = "#ff5252"

    else:

        dir_color = "#38bdf8"
        signal_color = "#38bdf8"


    mins_rem, secs_rem = divmod(
        time_remaining,
        60
    )


    # ========================================================
    # HEADER
    # ========================================================

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

<span style="
color:{signal_color};
font-size:16px;
font-weight:800;
">
{signal_strength}
</span>

&nbsp;|&nbsp;

Score:
<b>{combined_score:+.3f}</b>

&nbsp;|&nbsp;

Research:
<b>{research_score:+.3f}</b>

&nbsp;|&nbsp;

OBI:
<b>{obi_val:+.3f}</b>

&nbsp;|&nbsp;

OFI:
<b>{ofi_norm:+.3f}</b>

&nbsp;|&nbsp;

XGB:
<b>{xgb_signal}</b>
({xgb_confidence:.1f}%)

&nbsp;|&nbsp;

Trend:
<b>{trend_direction}</b>

&nbsp;|&nbsp;

Confidence:
<b>{confidence}%</b>

&nbsp;|&nbsp;

⏳
<b>{mins_rem}m {secs_rem}s</b>

</div>
""",
        unsafe_allow_html=True
    )


    # ========================================================
    # SIGNAL PANEL
    # ========================================================

    col_sig, col1, col2, col3, col4 = st.columns(
        [1.3, 1, 1, 1, 1]
    )


    with col_sig:

        st.markdown(
            f"""
<div class="metric-card"
style="
border-left:
4px solid {signal_color};
">

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

XGB:
{xgb_confidence:.1f}%

|

Long Votes:
{long_votes}/4

|

Short Votes:
{short_votes}/4

</div>

<div style="
font-size:11px;
color:#8b949e;
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


    with col1:

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

<div class="metric-card">

<div class="metric-label">
Stop / BASE
</div>

<div class="metric-val-red">
${base_level:,.2f}
</div>

</div>
""",
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
Risk / Reward
</div>

<div class="metric-val-blue">
1 : 2
</div>

<div style="
font-size:10px;
color:#8b949e;
">
TP1
</div>

</div>

<div class="metric-card">

<div class="metric-label">
TP2 Reward
</div>

<div class="metric-val-blue">
1 : 3
</div>

</div>
""",
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
Direction Confidence
</div>

<div class="metric-val-green">
{confidence}%
</div>

</div>

<div class="metric-card">

<div class="metric-label">
Research Score
</div>

<div class="metric-val-blue">
{research_score:+.3f}
</div>

</div>
""",
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
Market Risk
</div>

<div class="metric-val-red">
{risk_metrics["Market_Risk"]:.2f}
</div>

</div>

<div class="metric-card">

<div class="metric-label">
Squeeze Risk
</div>

<div class="metric-val-red">
{risk_metrics["Squeeze_Risk"]:.2f}
</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # SIGNAL CONDITION TABLE
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🎯 Signal Conditions"
    )

    condition_df = pd.DataFrame(
        [
            {
                "Condition": "XGBoost",
                "Value":
                    f"{xgb_signal} "
                    f"({xgb_confidence:.1f}%)",
            },
            {
                "Condition": "Research",
                "Value":
                    f"{research_direction} "
                    f"({research_score:+.3f})",
            },
            {
                "Condition": "OBI",
                "Value":
                    f"{obi_val:+.3f}",
            },
            {
                "Condition": "OFI",
                "Value":
                    f"{ofi_norm:+.3f}",
            },
            {
                "Condition": "Trend",
                "Value":
                    trend_direction,
            },
            {
                "Condition": "Long Votes",
                "Value":
                    f"{long_votes}/4",
            },
            {
                "Condition": "Short Votes",
                "Value":
                    f"{short_votes}/4",
            },
            {
                "Condition": "Combined Score",
                "Value":
                    f"{combined_score:+.3f}",
            },
        ]
    )

    st.dataframe(
        condition_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # CHART
    # ========================================================

    col_chart, col_micro = st.columns(
        [2.5, 1]
    )


    with col_chart:

        st.subheader(
            f"Price Trajectory ({selected_symbol})"
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
                    tp2_val -
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
                    close_p -
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
                name="Candles"
            )
        )


        fig.add_trace(
            go.Scatter(
                x=[
                    df["Time"].iloc[-1]
                ] + future_times,
                y=[
                    close_p
                ] + list(
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
            annotation_text=
                f"TP2: ${tp2_val:,.2f}"
        )

        fig.add_hline(
            y=tp1_val,
            line_dash="dot",
            annotation_text=
                f"TP1: ${tp1_val:,.2f}"
        )

        fig.add_hline(
            y=sl_val,
            line_dash="dash",
            annotation_text=
                f"SL: ${sl_val:,.2f}"
        )


        fig.update_layout(
            template="plotly_dark",
            height=430,
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

    with col_micro:

        st.subheader(
            "Market Microstructure"
        )

        spread_val = (
            abs(
                asks[0, 0]
                -
                bids[0, 0]
            )
            if len(bids)
            and len(asks)
            else 0
        )

        st.markdown(
            f"""
<div class="metric-card">

<div style="
display:flex;
justify-content:space-between;
margin-bottom:8px;
">

<span>Bid Volume</span>

<b style="color:#00e676;">
{bid_vol_sum:,.2f}
</b>

</div>


<div style="
display:flex;
justify-content:space-between;
margin-bottom:8px;
">

<span>Ask Volume</span>

<b style="color:#ff5252;">
{ask_vol_sum:,.2f}
</b>

</div>


<div style="
display:flex;
justify-content:space-between;
margin-bottom:8px;
">

<span>OBI</span>

<b style="color:#38bdf8;">
{obi_val:+.3f}
</b>

</div>


<div style="
display:flex;
justify-content:space-between;
margin-bottom:8px;
">

<span>OFI Normalized</span>

<b style="color:#38bdf8;">
{ofi_norm:+.3f}
</b>

</div>


<div style="
display:flex;
justify-content:space-between;
">

<span>Spread</span>

<b>
${spread_val:.2f}
</b>

</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # 12 PAPERS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔬 12-Paper Quantitative Research Scoreboard"
    )

    paper_table = []

    for k, v in paper_results.items():

        status = (
            "PASS 🟢"
            if v > 0.1
            else (
                "FAIL 🔴"
                if v < -0.1
                else
                "NEUTRAL ⚪"
            )
        )

        paper_table.append(
            {
                "Paper":
                    k,

                "Value":
                    f"{v:+.3f}",

                "Weight":
                    f"{evolved_weights.get(k, 0.083) * 100:.1f}%",

                "Status":
                    status,
            }
        )


    st.dataframe(
        pd.DataFrame(
            paper_table
        ),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Performance & Win Rate"
    )


    if st.session_state.trade_history_log:

        df_log = pd.DataFrame(
            st.session_state.trade_history_log
        )


        f1, f2, f3 = st.columns(3)


        with f1:

            coin_filter = st.selectbox(
                "Filter Coin",
                ["ALL"] + COINS_LIST
            )


        with f2:

            tf_filter = st.selectbox(
                "Filter Timeframe",
                ["ALL"] +
                list(
                    TIMEFRAME_MAP.keys()
                )
            )


        with f3:

            dir_filter = st.selectbox(
                "Filter Direction",
                [
                    "ALL",
                    "LONG",
                    "SHORT"
                ]
            )


        filtered_df = df_log.copy()


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
            wins + losses
        )

        win_rate = (
            wins /
            closed_trades *
            100
            if closed_trades > 0
            else 0
        )


        winning_df = filtered_df[
            filtered_df[
                "outcome"
            ]
            ==
            "WIN"
        ]

        losing_df = filtered_df[
            filtered_df[
                "outcome"
            ]
            ==
            "LOSS"
        ]


        gross_profit = (
            winning_df[
                "pnl_percent"
            ].sum()
            if not winning_df.empty
            else 0
        )

        gross_loss = abs(
            losing_df[
                "pnl_percent"
            ].sum()
            if not losing_df.empty
            else 0
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


        p1, p2, p3, p4, p5, p6 = (
            st.columns(6)
        )


        with p1:

            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%"
            )


        with p2:

            st.metric(
                "Closed Trades",
                closed_trades
            )


        with p3:

            st.metric(
                "Wins / Losses",
                f"{wins}W / {losses}L"
            )


        with p4:

            st.metric(
                "Pending",
                pending
            )


        with p5:

            st.metric(
                "Profit Factor",
                f"{profit_factor:.2f}"
            )


        with p6:

            st.metric(
                "Net PnL %",
                f"{net_pnl:+.2f}%"
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

            "rr_target":
                "TP1 1:2 | TP2 1:3",

            "exit_price": 0.0,

            "pnl_percent": 0.0,

            "outcome":
                "PENDING",

            "confidence": 0.0,

            "xgb_confidence": 0.0,

            "exit_reason": "",
        }


        for col in display_cols:

            if col not in filtered_df.columns:

                filtered_df[col] = (
                    defaults[col]
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

            st.session_state[
                "xgb_last_retrain_count"
            ] = 0

            st.rerun()


    else:

        st.info(
            "No paper trade history yet."
        )


else:

    st.warning(
        "⚠️ Data pipeline initializing "
        "or connection restricted."
    )
