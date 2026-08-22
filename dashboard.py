import datetime
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import joblib

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
XGB_MODEL_FILE = "xgboost_obi_model.pkl"


# ============================================================
# PERSISTENT HISTORY
# ============================================================

def load_persistent_history():

    if os.path.exists(CSV_FILE):

        try:

            df_hist = pd.read_csv(CSV_FILE)

            expected_cols = [
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
                "final_score",
                "xgb_probability",
                "xgb_signal",
                "outcome",
                "pnl_percent",
                "duration",
                "status"
            ]

            for col in expected_cols:

                if col not in df_hist.columns:

                    if col == "outcome":
                        df_hist[col] = "PENDING"

                    elif col == "xgb_signal":
                        df_hist[col] = "DISABLED"

                    elif col == "duration":
                        df_hist[col] = "Active"

                    elif col == "status":
                        df_hist[col] = "Open"

                    else:
                        df_hist[col] = 0.0

            return df_hist.to_dict("records")

        except Exception:
            return []

    return []


def save_persistent_history(history_list):

    try:

        if history_list:

            df_hist = pd.DataFrame(history_list)

            df_hist.to_csv(
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
# XGBOOST MODEL LOADER
# ============================================================

@st.cache_resource
def load_xgb_model():

    if not os.path.exists(XGB_MODEL_FILE):

        return None, "MODEL_NOT_FOUND"

    try:

        file_size = os.path.getsize(
            XGB_MODEL_FILE
        )

        if file_size <= 0:

            return None, "MODEL_EMPTY"

        model = joblib.load(
            XGB_MODEL_FILE
        )

        return model, "OK"

    except Exception as e:

        return None, f"LOAD_ERROR: {e}"


xgb_model, xgb_status = load_xgb_model()


# ============================================================
# XGBOOST PREDICTION
# ============================================================

def xgb_predict(model, feature_dict):

    if model is None:

        return {
            "available": False,
            "probability": 0.5,
            "signal": "DISABLED",
            "reason": xgb_status
        }

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        # Model must be trained using these 12 features.
        # ----------------------------------------------------

        feature_names = [
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
            "REWARD_RISK"
        ]

        X = np.array(
            [
                feature_dict.get(
                    name,
                    0.0
                )
                for name in feature_names
            ],
            dtype=float
        ).reshape(1, -1)

        # ----------------------------------------------------
        # Check expected number of features
        # ----------------------------------------------------

        expected_features = getattr(
            model,
            "n_features_in_",
            None
        )

        if expected_features is not None:

            if expected_features != X.shape[1]:

                return {
                    "available": False,
                    "probability": 0.5,
                    "signal": "INCOMPATIBLE",
                    "reason": (
                        f"Model expects "
                        f"{expected_features} features "
                        f"but dashboard provides "
                        f"{X.shape[1]}"
                    )
                }

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                X
            )[0]

            if len(probabilities) >= 2:

                probability = float(
                    probabilities[1]
                )

            else:

                probability = float(
                    probabilities[0]
                )

        else:

            prediction = model.predict(
                X
            )[0]

            probability = (
                1.0
                if float(prediction) > 0
                else 0.0
            )

        # ----------------------------------------------------
        # XGBoost signal
        # ----------------------------------------------------

        if probability >= 0.65:

            signal = "LONG"

        elif probability <= 0.35:

            signal = "SHORT"

        else:

            signal = "NEUTRAL"

        return {
            "available": True,
            "probability": probability,
            "signal": signal,
            "reason": "OK"
        }

    except Exception as e:

        return {
            "available": False,
            "probability": 0.5,
            "signal": "ERROR",
            "reason": str(e)
        }


# ============================================================
# RESEARCH LAB
# ============================================================

class TenPaperResearchLab:

    def __init__(self, target_vol=0.15):

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
            "REWARD_RISK"

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


        # ----------------------------------------------------
        # MARKET DATA
        # ----------------------------------------------------

        bid_vol = np.sum(
            bids[:, 1]
        )

        ask_vol = np.sum(
            asks[:, 1]
        )

        mid_price = (
            bids[0, 0]
            + asks[0, 0]
        ) / 2.0


        returns = (
            df["Close"]
            .pct_change()
            .dropna()
        )

        realized_vol = (
            returns.std()
            + 1e-8
        )


        returns_h = (
            df["Close"].iloc[-1]
            - df["Close"].iloc[-5]
        ) / (
            df["Close"].iloc[-5]
            + 1e-8
        )


        delta_p = (
            df["Close"].iloc[-1]
            - df["Close"].iloc[-2]
        )


        # ====================================================
        # 1 HAWKES
        # ====================================================

        vol_changes = (
            df["Volume"]
            .pct_change()
            .dropna()
            .replace(
                [np.inf, -np.inf],
                0
            )
            .values
        )

        if len(vol_changes) >= 15:

            hawkes_intensity = (
                np.mean(vol_changes[-3:])
                /
                (
                    np.mean(vol_changes[-15:])
                    + 1e-8
                )
            )

        else:

            hawkes_intensity = 1.0


        results["HAWKES"] = np.clip(
            (
                hawkes_intensity
                - 1.0
            )
            * np.sign(returns_h),
            -1,
            1
        )


        # ====================================================
        # 2 BOOK IMBALANCE
        # ====================================================

        results["BOOK_IMB"] = (
            bid_vol
            - ask_vol
        ) / (
            bid_vol
            + ask_vol
            + 1e-8
        )


        # ====================================================
        # 3 TAKER FLOW
        # ====================================================

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
            taker_buy
            - taker_sell
        ) / (
            taker_buy
            + taker_sell
            + 1e-8
        )


        # ====================================================
        # 4 QUANT IMPLY
        # ====================================================

        depth_skew = (
            bids[0, 1]
            - asks[0, 1]
        ) / (
            bids[0, 1]
            + asks[0, 1]
            + 1e-8
        )

        results["QUANT_IMPLY"] = np.clip(
            depth_skew * 1.5,
            -1,
            1
        )


        # ====================================================
        # 5 BAYESIAN
        # ====================================================

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
                * (1 - prior)
            )
            + 1e-8
        )

        results["BAYESIAN"] = np.clip(
            (
                posterior
                - 0.5
            ) * 2.0,
            -1,
            1
        )


        # ====================================================
        # 6 QUANTILES
        # ====================================================

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
                    returns_h
                    - q10
                )
                /
                (
                    q90
                    - q10
                    + 1e-8
                )
                * 2.0
            )
            - 1.0,
            -1,
            1
        )


        # ====================================================
        # 7 TARGET INVALIDATION
        # ====================================================

        target_diff = (
            delta_p
            /
            (
                df["Close"].iloc[-1]
                + 1e-8
            )
        )

        results["TARGET_INV"] = (

            1.0
            if target_diff >= 0.0006

            else

            -1.0
            if target_diff <= -0.0006

            else 0.0

        )


        # ====================================================
        # 8 ADAPTIVE CONF
        # ====================================================

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
                ma_fast
                - ma_slow
            )
            /
            (
                realized_vol
                * mid_price
                + 1e-8
            ),
            -1,
            1
        )


        # ====================================================
        # 9 FRACTIONAL KELLY
        # ====================================================

        win_prob = (
            0.55
            +
            (
                0.15
                * np.sign(
                    results["BOOK_IMB"]
                )
            )
        )

        kelly_fraction = (
            win_prob
            -
            (
                (1 - win_prob)
                / 1.5
            )
        )

        results["FRAC_KELLY"] = np.clip(
            kelly_fraction
            * 2.0
            * np.sign(returns_h),
            -1,
            1
        )


        # ====================================================
        # 10 RMT
        # ====================================================

        rmt_dom = (
            abs(returns_h)
            /
            (
                realized_vol
                * np.sqrt(5)
                + 1e-8
            )
        ) / 3.0

        results["RMT_DOM"] = np.clip(
            rmt_dom
            * np.sign(returns_h),
            -1,
            1
        )


        # ====================================================
        # 11 CONF CROSS
        # ====================================================

        conformal_spread = (
            realized_vol
            * 1.96
        )

        upper_b = (
            mid_price
            * (1 + conformal_spread)
        )

        lower_b = (
            mid_price
            * (1 - conformal_spread)
        )

        center = (
            upper_b
            + lower_b
        ) / 2.0

        results["CONF_CROSS"] = (

            1.0
            if mid_price > center

            else

            -1.0
            if mid_price < center

            else 0.0

        )


        # ====================================================
        # 12 REWARD RISK
        # ====================================================

        rr_ratio = (
            abs(q90)
            /
            (
                abs(q10)
                + 1e-8
            )
        )

        results["REWARD_RISK"] = (

            1.0
            if rr_ratio >= 1.2

            else

            -1.0
            if rr_ratio < 0.8

            else 0.0

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
            ],
            dtype=float
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
            if len(liquidation_volumes) > 0
            else 0.0
        )

        max_ltz = (
            np.max(liquidation_volumes)
            if len(liquidation_volumes) > 0
            else 0.0
        )

        ltz_score = (
            max_ltz
            /
            (
                total_ltz
                + 1e-8
            )
        ) * 100


        spoof_ratio = (
            cancelled_vol
            /
            (
                displayed_vol
                + 1e-8
            )
        )

        persistence = min(
            max(
                time_exists
                /
                (
                    obs_window
                    + 1e-8
                ),
                0
            ),
            1
        )

        spoof_score = (
            spoof_ratio
            *
            (1 - persistence)
        )


        squeeze_risk = (
            total_ltz
            * open_interest
            * leverage
            * volatility
        )

        market_risk = (
            ltz_score
            + spoof_score
            + squeeze_risk
        )

        return {
            "LTZ_Score": ltz_score,
            "Spoof_Score": spoof_score,
            "Squeeze_Risk": squeeze_risk,
            "Market_Risk": market_risk
        }


# ============================================================
# STYLE
# ============================================================

st.markdown(
"""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
);

html,
body,
[class*="css"] {
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
        rgba(0, 0, 0, 0.25);
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
    "LINKUSDT"
]


TIMEFRAME_MAP = {

    "1m (Scalping)": ("1m", 1),

    "15m (Medium TF)": ("15m", 15),

    "30m (Medium TF)": ("30m", 30),

    "1h (Intraday)": ("1h", 60),

    "4h (Intraday)": ("4h", 240)

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
    "### 📏 TRI Line Controls"
)

show_tri = st.sidebar.toggle(
    "Show TRI Lines",
    value=True
)


tri_enabled = {}

for _tf in [
    "YEARLY",
    "MONTHLY",
    "WEEKLY",
    "DAILY",
    "4H",
    "1H",
    "30M",
    "15M"
]:

    tri_enabled[_tf] = st.sidebar.checkbox(
        _tf,
        value=True,
        key=f"tri_{_tf}"
    )


api_interval, tf_minutes = TIMEFRAME_MAP[
    selected_tf_label
]


# ============================================================
# TRI LINE
# ============================================================

TRI_COLORS = {

    "YEARLY": "#87CEEB",
    "MONTHLY": "#FF0000",
    "WEEKLY": "#00C853",
    "DAILY": "#FFFFFF",
    "4H": "#FFA500",
    "1H": "#A855F7",
    "30M": "#006400",
    "15M": "#2196F3"

}


TRI_INTERVALS = {

    "MONTHLY": "1M",
    "WEEKLY": "1w",
    "DAILY": "1d",
    "4H": "4h",
    "1H": "1h",
    "30M": "30m",
    "15M": "15m"

}


@st.cache_data(ttl=30)
def fetch_tri_candle(
    symbol,
    interval
):

    try:

        url = (
            "https://data-api.binance.vision/"
            "api/v3/klines"
        )

        res = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": 3
            },
            timeout=5
        )

        data = res.json()

        if (
            not isinstance(data, list)
            or len(data) < 2
        ):

            return None

        row = data[-2]

        return {

            "Open": float(row[1]),
            "High": float(row[2]),
            "Low": float(row[3]),
            "Close": float(row[4])

        }

    except Exception:

        return None


@st.cache_data(ttl=60)
def fetch_tri_yearly_candle(
    symbol
):

    try:

        url = (
            "https://data-api.binance.vision/"
            "api/v3/klines"
        )

        res = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": "1M",
                "limit": 36
            },
            timeout=5
        )

        data = res.json()

        if (
            not isinstance(data, list)
            or len(data) < 14
        ):

            return None

        rows = []

        for row in data:

            rows.append({

                "Time": pd.to_datetime(
                    row[0],
                    unit="ms",
                    utc=True
                ),

                "Open": float(row[1]),
                "High": float(row[2]),
                "Low": float(row[3]),
                "Close": float(row[4])

            })

        monthly = pd.DataFrame(
            rows
        ).set_index("Time")

        yearly = monthly.resample(
            "YS"
        ).agg({

            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"

        }).dropna()

        if len(yearly) < 2:

            return None

        r = yearly.iloc[-2]

        return {

            "Open": float(r["Open"]),
            "High": float(r["High"]),
            "Low": float(r["Low"]),
            "Close": float(r["Close"])

        }

    except Exception:

        return None


def calculate_tri_levels(
    candle
):

    if candle is None:
        return None

    try:

        o = float(candle["Open"])
        h = float(candle["High"])
        l = float(candle["Low"])
        c = float(candle["Close"])

        body_high = max(o, c)
        body_low = min(o, c)

        return {

            "body_50":
                (body_high + body_low) / 2.0,

            "upper_50":
                (h + body_high) / 2.0,

            "lower_50":
                (l + body_low) / 2.0

        }

    except Exception:

        return None


@st.cache_data(ttl=30)
def get_all_tri_levels(
    symbol
):

    levels = {}

    yearly = calculate_tri_levels(
        fetch_tri_yearly_candle(symbol)
    )

    if yearly:
        levels["YEARLY"] = yearly

    for name, interval in TRI_INTERVALS.items():

        tri = calculate_tri_levels(
            fetch_tri_candle(
                symbol,
                interval
            )
        )

        if tri:
            levels[name] = tri

    return levels


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=15)
def fetch_klines_data(
    symbol,
    tf_key,
    limit=100
):

    binance_tf = (
        "1m"
        if "1m" in tf_key
        else
        "15m"
        if "15m" in tf_key
        else
        "30m"
        if "30m" in tf_key
        else
        "1h"
        if "1h" in tf_key
        else
        "4h"
    )

    url = (
        "https://data-api.binance.vision/"
        "api/v3/klines"
    )

    try:

        res = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": binance_tf,
                "limit": limit
            },
            timeout=5
        )

        res.raise_for_status()

        data = res.json()

        if (
            not isinstance(data, list)
            or len(data) == 0
        ):

            return pd.DataFrame()

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
                "Ignore"
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

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna()

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

    except Exception:

        return pd.DataFrame()


@st.cache_data(ttl=5)
def fetch_order_book_depth(
    symbol,
    depth_limit=20
):

    try:

        url = (
            "https://data-api.binance.vision/"
            "api/v3/depth"
        )

        res = requests.get(
            url,
            params={
                "symbol": symbol,
                "limit": depth_limit
            },
            timeout=5
        )

        res.raise_for_status()

        data = res.json()

        if (
            "bids" not in data
            or "asks" not in data
        ):

            return (
                np.empty((0, 2)),
                np.empty((0, 2))
            )

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


# ============================================================
# FETCH DATA
# ============================================================

df = fetch_klines_data(
    selected_symbol,
    selected_tf_label
)

bids, asks = fetch_order_book_depth(
    selected_symbol,
    depth_limit=20
)


# ============================================================
# DATA VALIDATION
# ============================================================

if (
    df.empty
    or len(df) < 15
    or len(bids) == 0
    or len(asks) == 0
):

    st.error(
        "⚠️ Real market data is currently unavailable."
    )

    st.info(
        "Dashboard ne fake market data use nahi karega."
        " Binance API connection restore hone par"
        " signals automatically resume honge."
    )

    st.stop()


# ============================================================
# ENGINE
# ============================================================

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


# ============================================================
# XGBOOST
# ============================================================

xgb_result = xgb_predict(
    xgb_model,
    paper_results
)

xgb_probability = float(
    xgb_result["probability"]
)

xgb_signal = xgb_result["signal"]


# ============================================================
# COMBINE RESEARCH + XGBOOST
# ============================================================

# XGBoost probability ko -1/+1 score mein convert
xgb_score = (
    xgb_probability * 2.0
) - 1.0


if xgb_result["available"]:

    # 60% quantitative research
    # 40% XGBoost
    final_score = (
        0.60 * research_score
        +
        0.40 * xgb_score
    )

else:

    # Agar model missing/incompatible hai
    # dashboard research model se chalega
    final_score = research_score


final_score = float(
    np.clip(
        final_score,
        -1,
        1
    )
)


# ============================================================
# PRICE / ATR
# ============================================================

close_p = float(
    df["Close"].iloc[-1]
)

atr_val = (
    df["High"]
    - df["Low"]
).rolling(
    14
).mean().iloc[-1]


if np.isnan(atr_val):

    atr_val = (
        close_p
        * 0.005
    )


# ============================================================
# TARGETS
# ============================================================

beam_level = (
    close_p
    +
    (1.8 * atr_val)
)

base_level = (
    close_p
    -
    (1.8 * atr_val)
)


if final_score >= 0:

    tp1_val = (
        close_p
        +
        atr_val
    )

    tp2_val = beam_level

    sl_val = (
        close_p
        -
        atr_val
    )

else:

    tp1_val = (
        close_p
        -
        atr_val
    )

    tp2_val = base_level

    sl_val = (
        close_p
        +
        atr_val
    )


# ============================================================
# FINAL SIGNAL
# ============================================================

direction = (

    "LONG"
    if final_score >= 0.15

    else

    "SHORT"
    if final_score <= -0.15

    else

    "NEUTRAL"

)


research_confidence = int(
    min(
        max(
            abs(research_score) * 100,
            15
        ),
        98
    )
)


xgb_confidence = int(
    abs(
        xgb_probability - 0.5
    )
    * 200
)


if xgb_result["available"]:

    confidence = int(
        (
            research_confidence
            * 0.60
        )
        +
        (
            xgb_confidence
            * 0.40
        )
    )

else:

    confidence = research_confidence


confidence = int(
    np.clip(
        confidence,
        15,
        98
    )
)


# ============================================================
# TIME LOCK
# ============================================================

lock_seconds = (
    tf_minutes * 60
)

current_time_sec = int(
    time.time()
)

time_bucket = (
    current_time_sec
    -
    (
        current_time_sec
        % lock_seconds
    )
)

time_remaining = (
    lock_seconds
    -
    (
        current_time_sec
        % lock_seconds
    )
)


trade_id = (
    f"{selected_symbol}_"
    f"{selected_tf_label}_"
    f"{time_bucket}_"
    f"{direction}"
)


# ============================================================
# PAPER TRADE
# ============================================================

if (
    paper_trading_mode
    and direction != "NEUTRAL"
):

    existing_trade_ids = [
        item.get("trade_id")
        for item
        in st.session_state.trade_history_log
    ]

    if trade_id not in existing_trade_ids:

        new_trade = {

            "trade_id":
                trade_id,

            "timestamp":
                datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "symbol":
                selected_symbol,

            "timeframe":
                selected_tf_label,

            "direction":
                direction,

            "entry_price":
                round(close_p, 2),

            "stop_loss":
                round(sl_val, 2),

            "tp1":
                round(tp1_val, 2),

            "tp2":
                round(tp2_val, 2),

            "exit_price":
                round(close_p, 2),

            "confidence":
                confidence,

            "final_score":
                round(final_score, 3),

            "xgb_probability":
                round(
                    xgb_probability,
                    4
                ),

            "xgb_signal":
                xgb_signal,

            "outcome":
                "PENDING",

            "pnl_percent":
                0.0,

            "duration":
                "Active",

            "status":
                "Open"

        }

        st.session_state.trade_history_log.insert(
            0,
            new_trade
        )

        save_persistent_history(
            st.session_state.trade_history_log
        )


# ============================================================
# CLOSE PAPER TRADES
# ============================================================

for trade in st.session_state.trade_history_log:

    if (
        trade.get("outcome") == "PENDING"
        and
        trade.get("symbol") == selected_symbol
    ):

        curr_price = close_p

        entry = float(
            trade["entry_price"]
        )

        sl = float(
            trade["stop_loss"]
        )

        tp = float(
            trade["tp1"]
        )


        if trade["direction"] == "LONG":

            if curr_price >= tp:

                trade["outcome"] = "WIN"

                trade["exit_price"] = curr_price

                trade["pnl_percent"] = round(
                    (
                        (
                            curr_price
                            - entry
                        )
                        /
                        entry
                    )
                    * 100,
                    2
                )

                trade["status"] = "Closed"


            elif curr_price <= sl:

                trade["outcome"] = "LOSS"

                trade["exit_price"] = curr_price

                trade["pnl_percent"] = round(
                    (
                        (
                            curr_price
                            - entry
                        )
                        /
                        entry
                    )
                    * 100,
                    2
                )

                trade["status"] = "Closed"


        elif trade["direction"] == "SHORT":

            if curr_price <= tp:

                trade["outcome"] = "WIN"

                trade["exit_price"] = curr_price

                trade["pnl_percent"] = round(
                    (
                        (
                            entry
                            - curr_price
                        )
                        /
                        entry
                    )
                    * 100,
                    2
                )

                trade["status"] = "Closed"


            elif curr_price >= sl:

                trade["outcome"] = "LOSS"

                trade["exit_price"] = curr_price

                trade["pnl_percent"] = round(
                    (
                        (
                            entry
                            - curr_price
                        )
                        /
                        entry
                    )
                    * 100,
                    2
                )

                trade["status"] = "Closed"


save_persistent_history(
    st.session_state.trade_history_log
)


# ============================================================
# RISK ENGINE
# ============================================================

risk_engine = PowerTradingRiskEngine()

disp_vol = float(
    np.sum(asks[:, 1])
)

risk_metrics = (
    risk_engine.calculate_risk_metrics(

        liquidation_volumes=np.array(
            [1000, 2500]
        ),

        displayed_vol=disp_vol,

        cancelled_vol=(
            disp_vol * 0.1
        ),

        time_exists=15.0,

        obs_window=60.0,

        open_interest=150000.0,

        leverage=20.0,

        volatility=(
            df["Close"]
            .pct_change()
            .std()
            + 1e-8
        )

    )
)


# ============================================================
# HEADER
# ============================================================

dir_color = (

    "#00e676"
    if direction == "LONG"

    else

    "#ff5252"
    if direction == "SHORT"

    else

    "#38bdf8"

)


mins_rem, secs_rem = divmod(
    time_remaining,
    60
)


model_status_text = (

    "ACTIVE"
    if xgb_result["available"]

    else

    xgb_result["signal"]

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
{selected_tf_label}

&nbsp;|&nbsp;

SIGNAL:
<span style="color:{dir_color};">
{direction}
</span>

&nbsp;|&nbsp;

Research:
<b>{research_score:+.3f}</b>

&nbsp;|&nbsp;

XGB:
<b>{xgb_probability:.1%}</b>

&nbsp;|&nbsp;

Final:
<b>{final_score:+.3f}</b>

&nbsp;|&nbsp;

Confidence:
<b>{confidence}%</b>

&nbsp;|&nbsp;

XGB Model:
<b>{model_status_text}</b>

&nbsp;|&nbsp;

⏳ Reset:
<b>{mins_rem}m {secs_rem}s</b>

</div>
""",
unsafe_allow_html=True
)


# ============================================================
# SIGNAL PANEL
# ============================================================

col_sig, col_m1, col_m2, col_m3, col_m4 = st.columns(
    [1.2, 1, 1, 1, 1]
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
            font-size:22px;
            font-weight:700;
            color:{dir_color};
        ">
            {direction}
        </div>

        <div style="
            font-size:11px;
            color:#8b949e;
            margin-top:4px;
        ">
            Entry: ${close_p:,.2f}
            |
            SL: ${sl_val:,.2f}
        </div>

        <div style="
            font-size:11px;
            color:#38bdf8;
        ">
            TP1: ${tp1_val:,.2f}
            |
            TP2: ${tp2_val:,.2f}
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
            Research Score
        </div>

        <div class="metric-val-blue">
            {research_score:+.3f}
        </div>

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown(
    f"""
    <div class="metric-card">

        <div class="metric-label">
            Final Score
        </div>

        <div class="metric-val-green">
            {final_score:+.3f}
        </div>

    </div>
    """,
    unsafe_allow_html=True
    )


with col_m2:

    st.markdown(
    f"""
    <div class="metric-card">

        <div class="metric-label">
            XGBoost Probability
        </div>

        <div class="metric-val-blue">
            {xgb_probability:.1%}
        </div>

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown(
    f"""
    <div class="metric-card">

        <div class="metric-label">
            XGBoost Signal
        </div>

        <div class="metric-val-green">
            {xgb_signal}
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
            LTZ Score
        </div>

        <div class="metric-val-blue">
            {risk_metrics["LTZ_Score"]:.2f}
        </div>

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown(
    f"""
    <div class="metric-card">

        <div class="metric-label">
            Spoof Score
        </div>

        <div class="metric-val-red">
            {risk_metrics["Spoof_Score"]:.3f}
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
            Squeeze Risk
        </div>

        <div class="metric-val-red">
            {risk_metrics["Squeeze_Risk"]:.2f}
        </div>

    </div>
    """,
    unsafe_allow_html=True
    )


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
    """,
    unsafe_allow_html=True
    )


# ============================================================
# CHART + MICROSTRUCTURE
# ============================================================

col_chart, col_risk_panel = st.columns(
    [2.5, 1],
    gap="medium"
)


with col_chart:

    st.subheader(
        f"Price Trajectory & Levels ({selected_symbol})"
    )


    time_delta = pd.Timedelta(
        minutes=tf_minutes
    )


    future_times = [

        df["Time"].iloc[-1]
        +
        (
            i * time_delta
        )

        for i in range(
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
                beam_level
                - close_p
            )
            *
            np.sin(t_steps)
        )

    elif direction == "SHORT":

        forecast_prices = (
            close_p
            -
            (
                close_p
                - base_level
            )
            *
            np.sin(t_steps)
        )

    else:

        forecast_prices = np.repeat(
            close_p,
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

            name="Candles",

            increasing_line_color="#00e676",

            decreasing_line_color="#ff5252"

        )
    )


    fig.add_trace(
        go.Scatter(

            x=[
                df["Time"].iloc[-1]
            ]
            + future_times,

            y=[
                close_p
            ]
            + list(
                forecast_prices
            ),

            mode="lines+markers",

            name="Trajectory",

            line=dict(
                color=dir_color,
                width=2,
                dash="dot"
            ),

            marker=dict(
                size=4
            )

        )
    )


    recent_df = df.tail(
        min(
            120,
            len(df)
        )
    )


    chart_low = float(
        recent_df["Low"].min()
    )

    chart_high = float(
        recent_df["High"].max()
    )


    if len(forecast_prices) > 0:

        chart_low = min(
            chart_low,
            float(
                np.min(
                    forecast_prices
                )
            )
        )

        chart_high = max(
            chart_high,
            float(
                np.max(
                    forecast_prices
                )
            )
        )


    active_levels = [
        beam_level,
        base_level,
        sl_val
    ]


    for level in active_levels:

        if chart_low <= level <= chart_high:

            continue

        if (
            abs(
                level
                - close_p
            )
            <=
            max(
                close_p * 0.025,
                (
                    chart_high
                    - chart_low
                )
                * 0.50
            )
        ):

            chart_low = min(
                chart_low,
                float(level)
            )

            chart_high = max(
                chart_high,
                float(level)
            )


    chart_span = max(
        chart_high
        - chart_low,
        close_p * 0.005
    )


    chart_padding = (
        chart_span * 0.08
    )


    visible_low = (
        chart_low
        - chart_padding
    )

    visible_high = (
        chart_high
        + chart_padding
    )


    fig.add_hline(
        y=beam_level,
        line_dash="dash",
        line_color="#00e676",
        annotation_text=(
            f"BEAM: ${beam_level:,.2f}"
        ),
        annotation_position="right"
    )


    fig.add_hline(
        y=base_level,
        line_dash="dash",
        line_color="#ff5252",
        annotation_text=(
            f"BASE: ${base_level:,.2f}"
        ),
        annotation_position="right"
    )


    fig.add_hline(
        y=sl_val,
        line_dash="dot",
        line_color="#ff5252",
        annotation_text=(
            f"SL: ${sl_val:,.2f}"
        ),
        annotation_position="right"
    )


    # ========================================================
    # TRI LINES
    # ========================================================

    if show_tri:

        tri_levels = get_all_tri_levels(
            selected_symbol
        )

        for tri_tf, tri in tri_levels.items():

            if not tri_enabled.get(
                tri_tf,
                True
            ):

                continue


            tri_color = TRI_COLORS.get(
                tri_tf,
                "#38bdf8"
            )


            for (
                level_name,
                width,
                opacity,
                dash
            ) in [

                (
                    "body_50",
                    3,
                    0.95,
                    "solid"
                ),

                (
                    "upper_50",
                    1,
                    0.55,
                    "dot"
                ),

                (
                    "lower_50",
                    1,
                    0.55,
                    "dot"
                )

            ]:

                level = float(
                    tri[level_name]
                )


                if (
                    level < visible_low
                    or
                    level > visible_high
                ):

                    continue


                fig.add_hline(

                    y=level,

                    line_color=tri_color,

                    line_width=width,

                    line_dash=dash,

                    opacity=opacity,

                    layer="above"

                )


    fig.update_layout(

        template="plotly_dark",

        height=470,

        xaxis_rangeslider_visible=False,

        dragmode="pan",

        hovermode="x unified",

        paper_bgcolor="#111622",

        plot_bgcolor="#111622",

        margin=dict(
            l=10,
            r=90,
            t=10,
            b=10
        ),

        xaxis=dict(
            showgrid=True,
            gridcolor="#202938",
            rangeslider_visible=False
        ),

        yaxis=dict(

            showgrid=True,

            gridcolor="#202938",

            zeroline=False,

            fixedrange=False,

            range=[
                visible_low,
                visible_high
            ],

            autorange=False

        ),

        legend=dict(

            orientation="h",

            yanchor="top",

            y=1.0,

            xanchor="right",

            x=1.0,

            bgcolor=(
                "rgba(17,22,34,0.75)"
            ),

            bordercolor="#202938",

            borderwidth=1,

            font=dict(size=10)

        )

    )


    st.plotly_chart(

        fig,

        use_container_width=True,

        config={

            "displaylogo": False,

            "scrollZoom": True,

            "doubleClick":
                "reset+autosize",

            "modeBarButtonsToAdd": [

                "zoom2d",
                "pan2d",
                "autoScale2d",
                "resetScale2d"

            ],

            "modeBarButtonsToRemove": [

                "lasso2d",
                "select2d",
                "toImage"

            ]

        }

    )


# ============================================================
# MICROSTRUCTURE
# ============================================================

with col_risk_panel:

    st.subheader(
        "Market Microstructure & OB"
    )


    bid_vol_sum = float(
        np.sum(
            bids[:, 1]
        )
    )

    ask_vol_sum = float(
        np.sum(
            asks[:, 1]
        )
    )


    obi_val = (
        bid_vol_sum
        - ask_vol_sum
    ) / (
        bid_vol_sum
        + ask_vol_sum
        + 1e-12
    )


    spread_val = abs(
        float(
            asks[0, 0]
        )
        -
        float(
            bids[0, 0]
        )
    )


    spread_mid = (
        float(
            asks[0, 0]
        )
        +
        float(
            bids[0, 0]
        )
    ) / 2.0


    spread_pct = (
        spread_val
        /
        spread_mid
        * 100.0
    )


    if obi_val >= 0.50:

        risk_status = "LOW"
        risk_color = "#00e676"

    elif obi_val >= 0.15:

        risk_status = "LOW-MEDIUM"
        risk_color = "#00e676"

    elif obi_val <= -0.50:

        risk_status = "HIGH"
        risk_color = "#ff5252"

    elif obi_val <= -0.15:

        risk_status = "MEDIUM"
        risk_color = "#ffa500"

    else:

        risk_status = "NEUTRAL"
        risk_color = "#38bdf8"


    st.markdown(
    f"""
    <div class="metric-card"
         style="padding:16px;">

        <div style="
            display:flex;
            justify-content:space-between;
            margin-bottom:10px;
        ">
            <span>Bid Volume</span>
            <b style="color:#00e676;">
                {bid_vol_sum:,.2f}
            </b>
        </div>

        <div style="
            display:flex;
            justify-content:space-between;
            margin-bottom:10px;
        ">
            <span>Ask Volume</span>
            <b style="color:#ff5252;">
                {ask_vol_sum:,.2f}
            </b>
        </div>

        <div style="
            display:flex;
            justify-content:space-between;
            margin-bottom:10px;
        ">
            <span>OBI</span>
            <b style="color:#38bdf8;">
                {obi_val:+.3f}
            </b>
        </div>

        <div style="
            display:flex;
            justify-content:space-between;
            margin-bottom:10px;
        ">
            <span>Spread</span>
            <b>
                ${spread_val:.2f}
            </b>
        </div>

        <div style="
            display:flex;
            justify-content:space-between;
        ">
            <span>Risk Status</span>
            <b style="color:{risk_color};">
                {risk_status}
            </b>
        </div>

    </div>
    """,
    unsafe_allow_html=True
    )


    st.subheader(
        "Top 20 OBI Analysis"
    )


    def calc_obi(depth):

        b = float(
            np.sum(
                bids[
                    :depth,
                    1
                ]
            )
        )

        a = float(
            np.sum(
                asks[
                    :depth,
                    1
                ]
            )
        )

        return (
            b - a
        ) / (
            b + a + 1e-12
        )


    top5_obi = calc_obi(5)
    top10_obi = calc_obi(10)
    top20_obi = calc_obi(20)


    fig_obi = go.Figure(

        go.Bar(

            x=[
                "Top 5",
                "Top 10",
                "Top 20"
            ],

            y=[
                top5_obi,
                top10_obi,
                top20_obi
            ],

            marker_color="#38bdf8",

            text=[
                f"{v:+.3f}"
                for v in [
                    top5_obi,
                    top10_obi,
                    top20_obi
                ]
            ],

            textposition="outside",

            cliponaxis=False

        )

    )


    fig_obi.add_hline(
        y=0,
        line_color="#64748b",
        line_width=1
    )


    fig_obi.update_layout(

        height=190,

        margin=dict(
            l=0,
            r=5,
            t=18,
            b=0
        ),

        paper_bgcolor="#111622",

        plot_bgcolor="#111622",

        font=dict(size=10),

        yaxis=dict(
            range=[-1, 1],
            gridcolor="#202938",
            zeroline=False
        ),

        xaxis=dict(
            showgrid=False
        ),

        showlegend=False

    )


    st.plotly_chart(

        fig_obi,

        use_container_width=True,

        config={
            "displayModeBar": False,
            "displaylogo": False
        }

    )


# ============================================================
# XGBOOST / RESEARCH SCOREBOARD
# ============================================================

st.markdown("---")

st.subheader(
    "🔬 Quantitative Research + XGBoost"
)


score_col1, score_col2 = st.columns(
    [1.6, 1]
)


with score_col1:

    paper_table_data = []

    for k, v in paper_results.items():

        status = (

            "PASS 🟢"
            if v > 0.1

            else

            "FAIL 🔴"
            if v < -0.1

            else

            "NEUTRAL ⚪"

        )

        paper_table_data.append({

            "Paper":
                k,

            "Value":
                f"{v:+.3f}",

            "Weight":
                f"{evolved_weights.get(k, 0.083)*100:.1f}%",

            "Status":
                status

        })


    st.dataframe(

        pd.DataFrame(
            paper_table_data
        ),

        use_container_width=True,

        hide_index=True,

        height=300

    )


with score_col2:

    model_message = (

        "XGBoost model loaded successfully."
        if xgb_result["available"]

        else

        f"XGBoost unavailable: "
        f"{xgb_result['reason']}"

    )


    st.markdown(
    f"""
    <div class="metric-card">

        <div style="
            font-weight:700;
            color:#38bdf8;
            margin-bottom:10px;
        ">
            XGBoost Model
        </div>

        <div style="
            font-size:13px;
            line-height:1.7;
            color:#cbd5e1;
        ">

            <b>Status:</b>
            {model_status_text}

            <br>

            <b>Probability:</b>
            {xgb_probability:.2%}

            <br>

            <b>Signal:</b>
            {xgb_signal}

            <br>

            <b>Research Score:</b>
            {research_score:+.3f}

            <br>

            <b>Final Score:</b>
            {final_score:+.3f}

            <br><br>

            {model_message}

        </div>

    </div>
    """,
    unsafe_allow_html=True
    )


# ============================================================
# PERFORMANCE
# ============================================================

st.markdown("---")

st.subheader(
    "📊 Performance Summary & Win Rate"
)


if st.session_state.trade_history_log:

    df_log = pd.DataFrame(
        st.session_state.trade_history_log
    )


    f_col1, f_col2, f_col3 = st.columns(3)


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


    filtered_df = df_log.copy()


    if coin_filter != "ALL":

        filtered_df = filtered_df[
            filtered_df["symbol"]
            ==
            coin_filter
        ]


    if tf_filter != "ALL":

        filtered_df = filtered_df[
            filtered_df["timeframe"]
            ==
            tf_filter
        ]


    if dir_filter != "ALL":

        filtered_df = filtered_df[
            filtered_df["direction"]
            ==
            dir_filter
        ]


    total_signals = len(
        filtered_df
    )


    wins = len(
        filtered_df[
            filtered_df["outcome"]
            ==
            "WIN"
        ]
    )


    losses = len(
        filtered_df[
            filtered_df["outcome"]
            ==
            "LOSS"
        ]
    )


    pending = len(
        filtered_df[
            filtered_df["outcome"]
            ==
            "PENDING"
        ]
    )


    closed_trades = (
        wins + losses
    )


    win_rate = (

        (
            wins
            /
            closed_trades
            * 100
        )

        if closed_trades > 0

        else 0.0

    )


    winning_trades_df = filtered_df[
        filtered_df["outcome"]
        ==
        "WIN"
    ]


    losing_trades_df = filtered_df[
        filtered_df["outcome"]
        ==
        "LOSS"
    ]


    gross_profit = (

        winning_trades_df[
            "pnl_percent"
        ].sum()

        if not winning_trades_df.empty

        else 0.0

    )


    gross_loss = (

        abs(
            losing_trades_df[
                "pnl_percent"
            ].sum()
        )

        if not losing_trades_df.empty

        else 0.0

    )


    net_pnl = (
        gross_profit
        - gross_loss
    )


    profit_factor = (

        gross_profit
        /
        gross_loss

        if gross_loss > 0

        else

        (
            gross_profit
            if gross_profit > 0
            else 0.0
        )

    )


    p1, p2, p3, p4, p5, p6 = st.columns(6)


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

            <div style="
                font-size:16px;
                font-weight:750;
                color:#00e676;
            ">
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
        "##### Detailed Trade History Table"
    )


    display_cols = [

        "timestamp",
        "symbol",
        "timeframe",
        "direction",
        "entry_price",
        "stop_loss",
        "tp1",
        "exit_price",
        "pnl_percent",
        "outcome",
        "confidence"

    ]


    available_cols = [

        c
        for c in display_cols
        if c in filtered_df.columns

    ]


    st.dataframe(

        filtered_df[
            available_cols
        ],

        use_container_width=True,

        hide_index=True,

        height=280

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

        st.rerun()


else:

    st.info(
        "No paper trade history recorded yet."
    )
