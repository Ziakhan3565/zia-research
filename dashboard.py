import datetime
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from sklearn.preprocessing import StandardScaler
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Quant Research Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(
    interval=5000,
    limit=None,
    key="research_lab_auto_refresh",
)


# ============================================================
# FILES
# ============================================================

CSV_FILE = "signal_history.csv"
MODEL_FILE = "xgboost_obi_model.pkl"


# ============================================================
# PROFESSIONAL DARK UI
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
        border-right: 1px solid #1e2638;
    }

    .metric-card {
        background: #111622;
        border: 1px solid #1e2638;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 19px;
        font-weight: 700;
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

    .section-title {
        font-size: 18px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .model-box {
        background: #111622;
        border: 1px solid #1e2638;
        border-radius: 10px;
        padding: 12px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PERSISTENT TRADE HISTORY
# ============================================================

EXPECTED_COLUMNS = [
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
    "outcome",
    "pnl_percent",
    "duration",
    "status",
]


def load_persistent_history():

    if not os.path.exists(CSV_FILE):
        return []

    try:

        df = pd.read_csv(CSV_FILE)

        for col in EXPECTED_COLUMNS:

            if col not in df.columns:

                if col == "outcome":
                    df[col] = "PENDING"

                elif col == "status":
                    df[col] = "Open"

                elif col == "duration":
                    df[col] = "Active"

                elif col in [
                    "entry_price",
                    "stop_loss",
                    "tp1",
                    "tp2",
                    "exit_price",
                    "confidence",
                    "final_score",
                    "pnl_percent",
                ]:
                    df[col] = 0.0

                else:
                    df[col] = ""

        return df[EXPECTED_COLUMNS].to_dict("records")

    except Exception:
        return []


def save_persistent_history(history):

    try:

        pd.DataFrame(history, columns=EXPECTED_COLUMNS).to_csv(
            CSV_FILE,
            index=False
        )

    except Exception as e:

        st.warning(f"Could not save trade history: {e}")


if "trade_history_log" not in st.session_state:

    st.session_state.trade_history_log = load_persistent_history()


# ============================================================
# XGBOOST MODEL LOADER
# ============================================================

@st.cache_resource
def load_xgboost_model():

    if not os.path.exists(MODEL_FILE):
        return None

    try:

        model = joblib.load(MODEL_FILE)

        return model

    except Exception:

        return None


xgb_model = load_xgboost_model()


# ============================================================
# XGBOOST PREDICTION
# ============================================================

def get_xgb_prediction(model, features):

    if model is None:
        return None, None

    try:

        feature_values = np.array(
            [features[key] for key in features],
            dtype=float
        ).reshape(1, -1)

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(feature_values)[0]

            prediction = int(np.argmax(probability))

            confidence = float(np.max(probability)) * 100

            return prediction, confidence

        prediction = int(model.predict(feature_values)[0])

        return prediction, None

    except Exception:

        return None, None


# ============================================================
# RESEARCH ENGINE
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
            "REWARD_RISK",

        ]

        self.dynamic_weights = {
            key: 1.0 / len(self.feature_names)
            for key in self.feature_names
        }


    def extract_features(self, df, bids, asks):

        results = {}

        if (
            len(bids) == 0
            or len(asks) == 0
            or df.empty
            or len(df) < 15
        ):

            return {
                key: 0.0
                for key in self.feature_names
            }


        bid_vol = np.sum(bids[:, 1])

        ask_vol = np.sum(asks[:, 1])

        mid_price = (
            bids[0, 0] + asks[0, 0]
        ) / 2


        returns = (
            df["Close"]
            .pct_change()
            .dropna()
        )

        realized_vol = returns.std() + 1e-8


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


        # ----------------------------------------------------
        # 1. HAWKES
        # ----------------------------------------------------

        vol_changes = (
            df["Volume"]
            .pct_change()
            .dropna()
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
            (hawkes_intensity - 1.0)
            * np.sign(returns_h),
            -1,
            1
        )


        # ----------------------------------------------------
        # 2. BOOK IMBALANCE
        # ----------------------------------------------------

        results["BOOK_IMB"] = (
            bid_vol - ask_vol
        ) / (
            bid_vol + ask_vol + 1e-8
        )


        # ----------------------------------------------------
        # 3. TAKER FLOW
        # ----------------------------------------------------

        taker_buy = (
            df["Volume"].iloc[-1]
            *
            (1.0 if delta_p > 0 else 0.3)
        )

        taker_sell = (
            df["Volume"].iloc[-1]
            *
            (1.0 if delta_p <= 0 else 0.3)
        )

        results["TAKER_FLOW"] = (
            taker_buy - taker_sell
        ) / (
            taker_buy + taker_sell + 1e-8
        )


        # ----------------------------------------------------
        # 4. QUANTITATIVE IMBALANCE
        # ----------------------------------------------------

        depth_skew = (
            bids[0, 1] - asks[0, 1]
        ) / (
            bids[0, 1] + asks[0, 1] + 1e-8
        )

        results["QUANT_IMPLY"] = np.clip(
            depth_skew * 1.5,
            -1,
            1
        )


        # ----------------------------------------------------
        # 5. BAYESIAN
        # ----------------------------------------------------

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
            (posterior - 0.5) * 2.0,
            -1,
            1
        )


        # ----------------------------------------------------
        # 6. QUANTILES
        # ----------------------------------------------------

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
                (returns_h - q10)
                /
                (q90 - q10 + 1e-8)
                * 2.0
            )
            - 1.0,
            -1,
            1
        )


        # ----------------------------------------------------
        # 7. TARGET / INVALIDATION
        # ----------------------------------------------------

        target_diff = (
            delta_p
            /
            (
                df["Close"].iloc[-1]
                + 1e-8
            )
        )

        if target_diff >= 0.0006:

            results["TARGET_INV"] = 1.0

        elif target_diff <= -0.0006:

            results["TARGET_INV"] = -1.0

        else:

            results["TARGET_INV"] = 0.0


        # ----------------------------------------------------
        # 8. ADAPTIVE CONFIRMATION
        # ----------------------------------------------------

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
                * mid_price
                + 1e-8
            ),
            -1,
            1
        )


        # ----------------------------------------------------
        # 9. FRACTIONAL KELLY
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # 10. RMT DOMINANCE
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # 11. CONFORMAL CROSS
        # ----------------------------------------------------

        conformal_spread = (
            realized_vol * 1.96
        )

        upper_b = (
            mid_price
            * (1 + conformal_spread)
        )

        lower_b = (
            mid_price
            * (1 - conformal_spread)
        )

        middle = (
            upper_b + lower_b
        ) / 2

        if mid_price > middle:

            results["CONF_CROSS"] = 1.0

        elif mid_price < middle:

            results["CONF_CROSS"] = -1.0

        else:

            results["CONF_CROSS"] = 0.0


        # ----------------------------------------------------
        # 12. REWARD / RISK
        # ----------------------------------------------------

        rr_ratio = (
            abs(q90)
            /
            (
                abs(q10)
                + 1e-8
            )
        )

        if rr_ratio >= 1.2:

            results["REWARD_RISK"] = 1.0

        elif rr_ratio < 0.8:

            results["REWARD_RISK"] = -1.0

        else:

            results["REWARD_RISK"] = 0.0


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
                results[key]
                for key in self.feature_names
            ]
        )

        final_score = float(
            np.dot(
                feature_vector,
                np.array(
                    list(
                        self.dynamic_weights.values()
                    )
                )
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
            * (1 - persistence)
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

            "Market_Risk": market_risk,

        }


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

    "1m": ("1m", 1),

    "15m": ("15m", 15),

    "30m": ("30m", 30),

    "1h": ("1h", 60),

    "4h": ("4h", 240),

}


st.sidebar.markdown("## Terminal Controls")


selected_symbol = st.sidebar.selectbox(
    "Market",
    COINS_LIST,
    index=0
)


selected_tf = st.sidebar.selectbox(
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


st.sidebar.markdown("## Paper Trading")


paper_trading_mode = st.sidebar.toggle(
    "Enable Paper Trading",
    value=True
)


st.sidebar.markdown("---")


st.sidebar.markdown("## TRI Line Controls")


show_tri = st.sidebar.toggle(
    "Show TRI Lines",
    value=True
)


tri_enabled = {}

for tf in [
    "YEARLY",
    "MONTHLY",
    "WEEKLY",
    "DAILY",
    "4H",
    "1H",
    "30M",
    "15M",
]:

    tri_enabled[tf] = st.sidebar.checkbox(
        tf,
        value=True,
        key=f"tri_{tf}"
    )


api_interval, tf_minutes = TIMEFRAME_MAP[
    selected_tf
]


# ============================================================
# TRI CONFIG
# ============================================================

TRI_COLORS = {

    "YEARLY": "#87CEEB",

    "MONTHLY": "#FF0000",

    "WEEKLY": "#00C853",

    "DAILY": "#FFFFFF",

    "4H": "#FFA500",

    "1H": "#A855F7",

    "30M": "#006400",

    "15M": "#2196F3",

}


TRI_INTERVALS = {

    "MONTHLY": "1M",

    "WEEKLY": "1w",

    "DAILY": "1d",

    "4H": "4h",

    "1H": "1h",

    "30M": "30m",

    "15M": "15m",

}


# ============================================================
# TRI DATA
# ============================================================

@st.cache_data(ttl=30)
def fetch_tri_candle(symbol, interval):

    try:

        url = (
            "https://data-api.binance.vision/"
            "api/v3/klines"
        )

        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": 3,
            },
            timeout=5
        )

        data = response.json()

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

            "Close": float(row[4]),

        }

    except Exception:

        return None


@st.cache_data(ttl=60)
def fetch_tri_yearly_candle(symbol):

    try:

        url = (
            "https://data-api.binance.vision/"
            "api/v3/klines"
        )

        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": "1M",
                "limit": 36,
            },
            timeout=5
        )

        data = response.json()

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

                "Close": float(row[4]),

            })

        monthly = (
            pd.DataFrame(rows)
            .set_index("Time")
        )

        yearly = (
            monthly
            .resample("YS")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
            })
            .dropna()
        )

        if len(yearly) < 2:

            return None

        row = yearly.iloc[-2]

        return {

            "Open": float(row["Open"]),

            "High": float(row["High"]),

            "Low": float(row["Low"]),

            "Close": float(row["Close"]),

        }

    except Exception:

        return None


def calculate_tri_levels(candle):

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
                (body_high + body_low) / 2,

            "upper_50":
                (h + body_high) / 2,

            "lower_50":
                (l + body_low) / 2,

        }

    except Exception:

        return None


@st.cache_data(ttl=30)
def get_all_tri_levels(symbol):

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
    timeframe,
    limit=150
):

    try:

        url = (
            "https://data-api.binance.vision/"
            "api/v3/klines"
        )

        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": timeframe,
                "limit": limit,
            },
            timeout=5
        )

        data = response.json()

        if (
            not isinstance(data, list)
            or len(data) == 0
        ):
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
            "Volume",
        ]:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
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

        dates = pd.date_range(
            end=datetime.datetime.now(),
            periods=limit,
            freq=timeframe
        )

        base_price = 60000

        close = (
            base_price
            +
            np.cumsum(
                np.random.normal(
                    0,
                    10,
                    limit
                )
            )
        )

        return pd.DataFrame({

            "Time": dates,

            "Open": close - 5,

            "High": close + 15,

            "Low": close - 15,

            "Close": close,

            "Volume": np.random.uniform(
                50,
                500,
                limit
            ),

        })


@st.cache_data(ttl=10)
def fetch_order_book_depth(
    symbol,
    depth_limit=20
):

    try:

        url = (
            "https://data-api.binance.vision/"
            "api/v3/depth"
        )

        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "limit": depth_limit,
            },
            timeout=5
        )

        data = response.json()

        if (
            "bids" in data
            and "asks" in data
        ):

            return (
                np.array(
                    data["bids"],
                    dtype=float
                ),

                np.array(
                    data["asks"],
                    dtype=float
                ),
            )

    except Exception:

        pass

    dummy_bids = np.array(
        [
            [60000 - i * 2, 1.5]
            for i in range(depth_limit)
        ],
        dtype=float
    )

    dummy_asks = np.array(
        [
            [60000 + i * 2, 1.5]
            for i in range(depth_limit)
        ],
        dtype=float
    )

    return dummy_bids, dummy_asks


# ============================================================
# FETCH DATA
# ============================================================

df = fetch_klines_data(
    selected_symbol,
    api_interval
)

bids, asks = fetch_order_book_depth(
    selected_symbol,
    20
)


# ============================================================
# MAIN ENGINE
# ============================================================

if (
    not df.empty
    and len(df) >= 15
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
        performance_history=(
            st.session_state.trade_history_log
        )
    )


    # ========================================================
    # OBI
    # ========================================================

    bid_volume = float(
        np.sum(bids[:, 1])
    )

    ask_volume = float(
        np.sum(asks[:, 1])
    )

    obi = (
        bid_volume - ask_volume
    ) / (
        bid_volume
        + ask_volume
        + 1e-12
    )


    # ========================================================
    # PRICE
    # ========================================================

    close_price = float(
        df["Close"].iloc[-1]
    )


    # ========================================================
    # ATR
    # ========================================================

    atr = (
        df["High"]
        - df["Low"]
    ).rolling(14).mean().iloc[-1]

    if np.isnan(atr):

        atr = close_price * 0.005


    # ========================================================
    # TARGETS
    # ========================================================

    beam = (
        close_price
        + 1.8 * atr
    )

    base = (
        close_price
        - 1.8 * atr
    )


    # ========================================================
    # SIGNAL
    # ========================================================

    direction = "NEUTRAL"

    if research_score >= 0.15:

        direction = "LONG"

    elif research_score <= -0.15:

        direction = "SHORT"


    # ========================================================
    # XGBOOST
    # ========================================================

    xgb_prediction, xgb_confidence = (
        get_xgb_prediction(
            xgb_model,
            paper_results
        )
    )


    # --------------------------------------------------------
    # Optional XGBoost confirmation
    # --------------------------------------------------------

    if xgb_prediction is not None:

        # Expected:
        # 0 = SHORT
        # 1 = NEUTRAL
        # 2 = LONG

        if xgb_prediction == 2:

            if direction == "SHORT":

                direction = "NEUTRAL"

            elif direction == "LONG":

                direction = "LONG"

        elif xgb_prediction == 0:

            if direction == "LONG":

                direction = "NEUTRAL"

            elif direction == "SHORT":

                direction = "SHORT"


    # ========================================================
    # TP / SL
    # ========================================================

    if direction == "LONG":

        tp1 = close_price + atr

        tp2 = beam

        sl = close_price - atr

    elif direction == "SHORT":

        tp1 = close_price - atr

        tp2 = base

        sl = close_price + atr

    else:

        tp1 = close_price

        tp2 = close_price

        sl = close_price


    # ========================================================
    # CONFIDENCE
    # ========================================================

    research_confidence = min(
        max(
            abs(research_score) * 100,
            15
        ),
        98
    )

    if xgb_confidence is not None:

        confidence = int(
            (
                research_confidence
                +
                xgb_confidence
            )
            / 2
        )

    else:

        confidence = int(
            research_confidence
        )


    # ========================================================
    # SIGNAL COLOR
    # ========================================================

    if direction == "LONG":

        signal_color = "#00e676"

    elif direction == "SHORT":

        signal_color = "#ff5252"

    else:

        signal_color = "#38bdf8"


    # ========================================================
    # TIME LOCK
    # ========================================================

    lock_seconds = (
        tf_minutes * 60
    )

    current_time = int(time.time())

    time_bucket = (
        current_time
        -
        (
            current_time
            % lock_seconds
        )
    )

    remaining = (
        lock_seconds
        -
        (
            current_time
            % lock_seconds
        )
    )

    mins, secs = divmod(
        remaining,
        60
    )


    # ========================================================
    # TRADE ID
    # ========================================================

    trade_id = (
        f"{selected_symbol}_"
        f"{selected_tf}_"
        f"{time_bucket}_"
        f"{direction}"
    )


    # ========================================================
    # PAPER TRADE ENTRY
    # ========================================================

    if (
        paper_trading_mode
        and direction != "NEUTRAL"
    ):

        existing_ids = [
            item.get("trade_id")
            for item
            in st.session_state.trade_history_log
        ]

        if trade_id not in existing_ids:

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
                    selected_tf,

                "direction":
                    direction,

                "entry_price":
                    round(
                        close_price,
                        2
                    ),

                "stop_loss":
                    round(
                        sl,
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
                        close_price,
                        2
                    ),

                "confidence":
                    confidence,

                "final_score":
                    round(
                        research_score,
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

            }

            st.session_state.trade_history_log.insert(
                0,
                new_trade
            )

            save_persistent_history(
                st.session_state.trade_history_log
            )


    # ========================================================
    # UPDATE OPEN TRADES
    # ========================================================

    history_changed = False

    for trade in st.session_state.trade_history_log:

        if (
            trade.get("outcome") != "PENDING"
            or trade.get("symbol") != selected_symbol
        ):
            continue

        entry = float(
            trade.get(
                "entry_price",
                0
            )
        )

        trade_sl = float(
            trade.get(
                "stop_loss",
                0
            )
        )

        trade_tp = float(
            trade.get(
                "tp1",
                0
            )
        )

        if entry <= 0:
            continue


        # LONG
        if trade["direction"] == "LONG":

            if close_price >= trade_tp:

                trade["outcome"] = "WIN"

                trade["exit_price"] = (
                    close_price
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            close_price
                            - entry
                        )
                        / entry
                    )
                    * 100,
                    2
                )

                trade["status"] = "Closed"

                history_changed = True


            elif close_price <= trade_sl:

                trade["outcome"] = "LOSS"

                trade["exit_price"] = (
                    close_price
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            close_price
                            - entry
                        )
                        / entry
                    )
                    * 100,
                    2
                )

                trade["status"] = "Closed"

                history_changed = True


        # SHORT
        elif trade["direction"] == "SHORT":

            if close_price <= trade_tp:

                trade["outcome"] = "WIN"

                trade["exit_price"] = (
                    close_price
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            entry
                            - close_price
                        )
                        / entry
                    )
                    * 100,
                    2
                )

                trade["status"] = "Closed"

                history_changed = True


            elif close_price >= trade_sl:

                trade["outcome"] = "LOSS"

                trade["exit_price"] = (
                    close_price
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            entry
                            - close_price
                        )
                        / entry
                    )
                    * 100,
                    2
                )

                trade["status"] = "Closed"

                history_changed = True


    if history_changed:

        save_persistent_history(
            st.session_state.trade_history_log
        )


    # ========================================================
    # RISK ENGINE
    # ========================================================

    risk_engine = PowerTradingRiskEngine()

    volatility = (
        df["Close"]
        .pct_change()
        .std()
        + 1e-8
    )

    risk_metrics = (
        risk_engine.calculate_risk_metrics(

            liquidation_volumes=np.array(
                [1000, 2500]
            ),

            displayed_vol=ask_volume,

            cancelled_vol=(
                ask_volume * 0.10
            ),

            time_exists=15.0,

            obs_window=60.0,

            open_interest=150000.0,

            leverage=20.0,

            volatility=volatility,

        )
    )


    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        f"""
        <div class="top-status-bar">

        🟢 <b>{selected_symbol}</b>
        &nbsp; | &nbsp;

        Price:
        <b>${close_price:,.2f}</b>

        &nbsp; | &nbsp;

        Timeframe:
        <b>{selected_tf}</b>

        &nbsp; | &nbsp;

        Signal:
        <span style="color:{signal_color};">
        <b>{direction}</b>
        </span>

        &nbsp; | &nbsp;

        Research Score:
        <b>{research_score:+.3f}</b>

        &nbsp; | &nbsp;

        Confidence:
        <b>{confidence}%</b>

        &nbsp; | &nbsp;

        Next Reset:
        <b>{mins}m {secs}s</b>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # MODEL STATUS
    # ========================================================

    if xgb_model is not None:

        model_status = "XGBoost Model: ACTIVE"

        model_color = "#00e676"

    else:

        model_status = (
            "XGBoost Model: NOT LOADED"
        )

        model_color = "#ffa500"


    st.markdown(
        f"""
        <div class="model-box">

        <b>Machine Learning Engine</b>

        &nbsp;&nbsp;

        <span style="color:{model_color};">
        ● {model_status}
        </span>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SIGNAL PANEL
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)


    with c1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-label">
            Signal
            </div>

            <div class="metric-value"
                 style="color:{signal_color};">

            {direction}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-label">
            Entry
            </div>

            <div class="metric-value"
                 style="color:#38bdf8;">

            ${close_price:,.2f}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-label">
            Stop Loss
            </div>

            <div class="metric-value"
                 style="color:#ff5252;">

            ${sl:,.2f}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-label">
            Take Profit 1
            </div>

            <div class="metric-value"
                 style="color:#00e676;">

            ${tp1:,.2f}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c5:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-label">
            Take Profit 2
            </div>

            <div class="metric-value"
                 style="color:#00e676;">

            ${tp2:,.2f}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # SECOND METRICS
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)


    with c1:

        st.metric(
            "OBI",
            f"{obi:+.3f}"
        )


    with c2:

        st.metric(
            "Bid Volume",
            f"{bid_volume:,.2f}"
        )


    with c3:

        st.metric(
            "Ask Volume",
            f"{ask_volume:,.2f}"
        )


    with c4:

        spread = abs(
            float(asks[0, 0])
            -
            float(bids[0, 0])
        )

        st.metric(
            "Spread",
            f"${spread:.2f}"
        )


    with c5:

        st.metric(
            "ATR",
            f"${atr:,.2f}"
        )


    # ========================================================
    # CHART
    # ========================================================

    st.markdown(
        '<div class="section-title">Price Analysis</div>',
        unsafe_allow_html=True
    )


    future_times = [

        df["Time"].iloc[-1]
        +
        pd.Timedelta(
            minutes=tf_minutes * i
        )

        for i
        in range(
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

        forecast_prices = (
            close_price
            +
            (
                beam
                - close_price
            )
            *
            np.sin(steps)
        )

    elif direction == "SHORT":

        forecast_prices = (
            close_price
            -
            (
                close_price
                - base
            )
            *
            np.sin(steps)
        )

    else:

        forecast_prices = np.repeat(
            close_price,
            forecast_horizon
        )


    fig = go.Figure()


    # Candles
    fig.add_trace(
        go.Candlestick(

            x=df["Time"],

            open=df["Open"],

            high=df["High"],

            low=df["Low"],

            close=df["Close"],

            name="Market",

        )
    )


    # Forecast
    fig.add_trace(
        go.Scatter(

            x=[
                df["Time"].iloc[-1]
            ]
            +
            future_times,

            y=[
                close_price
            ]
            +
            list(forecast_prices),

            mode="lines+markers",

            name="Forecast",

            line=dict(
                color=signal_color,
                width=2,
                dash="dot"
            ),

            marker=dict(
                size=4
            ),

        )
    )


    # Active levels
    fig.add_hline(
        y=beam,
        line_dash="dash",
        line_color="#00e676",
        annotation_text="BEAM"
    )

    fig.add_hline(
        y=base,
        line_dash="dash",
        line_color="#ff5252",
        annotation_text="BASE"
    )

    fig.add_hline(
        y=sl,
        line_dash="dot",
        line_color="#ff5252",
        annotation_text="SL"
    )


    # ========================================================
    # TRI LINES
    # ========================================================

    if show_tri:

        tri_levels = get_all_tri_levels(
            selected_symbol
        )

        recent = df.tail(
            min(120, len(df))
        )

        chart_low = float(
            recent["Low"].min()
        )

        chart_high = float(
            recent["High"].max()
        )

        for tri_tf, tri in tri_levels.items():

            if not tri_enabled.get(
                tri_tf,
                True
            ):
                continue

            color = TRI_COLORS.get(
                tri_tf,
                "#38bdf8"
            )

            for name, width, opacity, dash in [

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
                ),

            ]:

                level = float(
                    tri[name]
                )

                span = max(
                    chart_high
                    - chart_low,
                    close_price * 0.005
                )

                if (
                    level
                    <
                    chart_low
                    - span * 0.5
                    or
                    level
                    >
                    chart_high
                    + span * 0.5
                ):
                    continue

                fig.add_hline(

                    y=level,

                    line_color=color,

                    line_width=width,

                    line_dash=dash,

                    opacity=opacity,

                )


    fig.update_layout(

        template="plotly_dark",

        height=500,

        paper_bgcolor="#111622",

        plot_bgcolor="#111622",

        xaxis_rangeslider_visible=False,

        dragmode="pan",

        hovermode="x unified",

        margin=dict(
            l=10,
            r=70,
            t=10,
            b=10
        ),

        xaxis=dict(
            showgrid=True,
            gridcolor="#202938"
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="#202938"
        ),

    )


    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "doubleClick": "reset+autosize",
        }
    )


    # ========================================================
    # MICROSTRUCTURE
    # ========================================================

    st.markdown(
        '<div class="section-title">Order Book Analysis</div>',
        unsafe_allow_html=True
    )


    m1, m2, m3, m4 = st.columns(4)


    def calculate_obi(depth):

        bid = float(
            np.sum(
                bids[
                    :depth,
                    1
                ]
            )
        )

        ask = float(
            np.sum(
                asks[
                    :depth,
                    1
                ]
            )
        )

        return (
            bid - ask
        ) / (
            bid
            + ask
            + 1e-12
        )


    top5 = calculate_obi(5)

    top10 = calculate_obi(10)

    top20 = calculate_obi(20)


    with m1:
        st.metric(
            "Top 5 OBI",
            f"{top5:+.3f}"
        )

    with m2:
        st.metric(
            "Top 10 OBI",
            f"{top10:+.3f}"
        )

    with m3:
        st.metric(
            "Top 20 OBI",
            f"{top20:+.3f}"
        )

    with m4:
        st.metric(
            "Book Pressure",
            (
                "BUY"
                if obi > 0.15
                else
                "SELL"
                if obi < -0.15
                else
                "NEUTRAL"
            )
        )


    obi_fig = go.Figure(

        go.Bar(

            x=[
                "Top 5",
                "Top 10",
                "Top 20"
            ],

            y=[
                top5,
                top10,
                top20
            ],

            text=[
                f"{top5:+.3f}",
                f"{top10:+.3f}",
                f"{top20:+.3f}",
            ],

            textposition="outside",

        )

    )


    obi_fig.add_hline(
        y=0,
        line_width=1
    )


    obi_fig.update_layout(

        template="plotly_dark",

        height=250,

        paper_bgcolor="#111622",

        plot_bgcolor="#111622",

        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),

        yaxis=dict(
            range=[-1, 1]
        ),

    )


    st.plotly_chart(
        obi_fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    # ========================================================
    # RISK
    # ========================================================

    st.markdown(
        '<div class="section-title">Risk Monitor</div>',
        unsafe_allow_html=True
    )


    r1, r2, r3, r4 = st.columns(4)


    with r1:

        st.metric(
            "LTZ Score",
            f"{risk_metrics['LTZ_Score']:.2f}"
        )


    with r2:

        st.metric(
            "Spoof Score",
            f"{risk_metrics['Spoof_Score']:.3f}"
        )


    with r3:

        st.metric(
            "Squeeze Risk",
            f"{risk_metrics['Squeeze_Risk']:.2f}"
        )


    with r4:

        st.metric(
            "Market Risk",
            f"{risk_metrics['Market_Risk']:.2f}"
        )


    # ========================================================
    # RESEARCH SCOREBOARD
    # ========================================================

    st.markdown("---")

    st.subheader(
        "12-Factor Quantitative Research"
    )


    scoreboard = []

    for key, value in paper_results.items():

        if value > 0.10:

            status = "PASS"

        elif value < -0.10:

            status = "FAIL"

        else:

            status = "NEUTRAL"


        scoreboard.append({

            "Factor":
                key,

            "Value":
                f"{value:+.3f}",

            "Weight":
                f"{evolved_weights[key] * 100:.1f}%",

            "Status":
                status,

        })


    st.dataframe(
        pd.DataFrame(scoreboard),
        use_container_width=True,
        hide_index=True,
        height=300
    )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Paper Trading Performance"
    )


    if st.session_state.trade_history_log:

        df_log = pd.DataFrame(
            st.session_state.trade_history_log
        )


        f1, f2, f3 = st.columns(3)


        with f1:

            coin_filter = st.selectbox(
                "Market Filter",
                ["ALL"] + COINS_LIST
            )


        with f2:

            tf_filter = st.selectbox(
                "Timeframe Filter",
                ["ALL"] + list(
                    TIMEFRAME_MAP.keys()
                )
            )


        with f3:

            direction_filter = st.selectbox(
                "Direction Filter",
                [
                    "ALL",
                    "LONG",
                    "SHORT"
                ]
            )


        filtered = df_log.copy()


        if coin_filter != "ALL":

            filtered = filtered[
                filtered["symbol"]
                ==
                coin_filter
            ]


        if tf_filter != "ALL":

            filtered = filtered[
                filtered["timeframe"]
                ==
                tf_filter
            ]


        if direction_filter != "ALL":

            filtered = filtered[
                filtered["direction"]
                ==
                direction_filter
            ]


        total = len(filtered)

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

            wins
            /
            closed
            * 100

            if closed > 0

            else 0

        )


        gross_profit = filtered.loc[
            filtered["outcome"] == "WIN",
            "pnl_percent"
        ].sum()


        gross_loss = abs(
            filtered.loc[
                filtered["outcome"] == "LOSS",
                "pnl_percent"
            ].sum()
        )


        net_pnl = (
            gross_profit
            - gross_loss
        )


        if gross_loss > 0:

            profit_factor = (
                gross_profit
                /
                gross_loss
            )

        else:

            profit_factor = (
                gross_profit
                if gross_profit > 0
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
            "#### Trade History"
        )


        display_columns = [

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
            "confidence",

        ]


        available_columns = [

            col
            for col in display_columns
            if col in filtered.columns

        ]


        st.dataframe(

            filtered[
                available_columns
            ],

            use_container_width=True,

            hide_index=True,

            height=300

        )


        if st.sidebar.button(
            "Clear Trade History"
        ):

            st.session_state.trade_history_log = []

            if os.path.exists(CSV_FILE):

                os.remove(CSV_FILE)

            st.rerun()


    else:

        st.info(
            "No paper trades recorded yet."
        )


else:

    st.warning(
        "Market data is initializing. "
        "Please wait for the next refresh."
    )
