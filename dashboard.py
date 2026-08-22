import datetime
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.preprocessing import StandardScaler
from streamlit_autorefresh import st_autorefresh


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="TRI Quant Research Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(
    interval=5000,
    limit=None,
    key="tri_quant_auto_refresh"
)

CSV_FILE = "signal_history.csv"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(56,189,248,0.06), transparent 25%),
        radial-gradient(circle at 85% 15%, rgba(139,92,246,0.05), transparent 25%),
        #080b12;
    color: #e5e7eb;
}

section[data-testid="stSidebar"] {
    background: #0b0f17 !important;
    border-right: 1px solid #1b2433;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

h1, h2, h3, h4 {
    color: #f8fafc !important;
}

.topbar {
    background: linear-gradient(
        90deg,
        rgba(17,24,39,0.98),
        rgba(15,23,42,0.98)
    );
    border: 1px solid #263247;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

.top-title {
    font-size: 20px;
    font-weight: 800;
    color: #f8fafc;
}

.top-subtitle {
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
}

.card {
    background: linear-gradient(
        145deg,
        #111722,
        #0d131d
    );
    border: 1px solid #202b3d;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.20);
}

.card-title {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 8px;
}

.card-value {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 800;
}

.card-small {
    color: #64748b;
    font-size: 11px;
    margin-top: 5px;
}

.green {
    color: #22c55e !important;
}

.red {
    color: #ef4444 !important;
}

.blue {
    color: #38bdf8 !important;
}

.orange {
    color: #f59e0b !important;
}

.purple {
    color: #a78bfa !important;
}

.signal-long {
    color: #22c55e;
    font-size: 27px;
    font-weight: 900;
}

.signal-short {
    color: #ef4444;
    font-size: 27px;
    font-weight: 900;
}

.signal-neutral {
    color: #38bdf8;
    font-size: 27px;
    font-weight: 900;
}

.section-title {
    font-size: 18px;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 10px;
    margin-bottom: 12px;
}

.rr-box {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
}

.rr-value {
    font-size: 24px;
    font-weight: 900;
    color: #38bdf8;
}

.status-open {
    color: #38bdf8;
    font-weight: 800;
}

.status-win {
    color: #22c55e;
    font-weight: 800;
}

.status-loss {
    color: #ef4444;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PERSISTENT HISTORY
# =========================================================

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

        for col in expected_cols:

            if col not in df_hist.columns:

                if col == "outcome":
                    df_hist[col] = "PENDING"

                elif col == "status":
                    df_hist[col] = "Open"

                elif col == "duration":
                    df_hist[col] = "Active"

                else:
                    df_hist[col] = 0.0

        return df_hist.to_dict("records")

    except Exception:
        return []


def save_persistent_history(history):

    try:

        pd.DataFrame(history).to_csv(
            CSV_FILE,
            index=False
        )

    except Exception as e:

        st.error(f"History save error: {e}")


if "trade_history_log" not in st.session_state:

    st.session_state.trade_history_log = (
        load_persistent_history()
    )


# =========================================================
# QUANT RESEARCH ENGINE
# =========================================================

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
            k: 1.0 / len(self.feature_names)
            for k in self.feature_names
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
                k: 0.0
                for k in self.feature_names
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

        realized_vol = (
            returns.std() + 1e-8
        )

        returns_h = (
            df["Close"].iloc[-1]
            - df["Close"].iloc[-5]
        ) / (
            df["Close"].iloc[-5] + 1e-8
        )

        delta_p = (
            df["Close"].iloc[-1]
            - df["Close"].iloc[-2]
        )

        # 1
        vol_changes = (
            df["Volume"]
            .pct_change()
            .dropna()
            .values
        )

        if len(vol_changes) >= 15:

            hawkes = (
                np.mean(vol_changes[-3:])
                /
                (
                    np.mean(vol_changes[-15:])
                    + 1e-8
                )
            )

        else:

            hawkes = 1.0

        results["HAWKES"] = np.clip(
            (hawkes - 1.0)
            * np.sign(returns_h),
            -1,
            1
        )

        # 2
        results["BOOK_IMB"] = (
            bid_vol - ask_vol
        ) / (
            bid_vol + ask_vol + 1e-8
        )

        # 3
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

        # 4
        depth_skew = (
            bids[0, 1] - asks[0, 1]
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

        # 5
        prior = 0.745

        likelihood = (
            1.0
            if results["BOOK_IMB"] > 0
            else 0.25
        )

        posterior = (
            likelihood * prior
        ) / (
            likelihood * prior
            +
            (1 - likelihood)
            * (1 - prior)
            + 1e-8
        )

        results["BAYESIAN"] = np.clip(
            (posterior - 0.5) * 2,
            -1,
            1
        )

        # 6
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
                * 2
                - 1
            ),
            -1,
            1
        )

        # 7
        target_diff = (
            delta_p
            /
            (df["Close"].iloc[-1] + 1e-8)
        )

        results["TARGET_INV"] = (

            1.0
            if target_diff >= 0.0006

            else -1.0
            if target_diff <= -0.0006

            else 0.0
        )

        # 8
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

        # 9
        win_prob = (
            0.55
            +
            0.15
            * np.sign(
                results["BOOK_IMB"]
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
            * 2
            * np.sign(returns_h),
            -1,
            1
        )

        # 10
        rmt_dom = (
            abs(returns_h)
            /
            (
                realized_vol
                * np.sqrt(5)
                + 1e-8
            )
        ) / 3

        results["RMT_DOM"] = np.clip(
            rmt_dom * np.sign(returns_h),
            -1,
            1
        )

        # 11
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

        results["CONF_CROSS"] = (

            1.0
            if mid_price > (
                upper_b + lower_b
            ) / 2

            else -1.0
            if mid_price < (
                upper_b + lower_b
            ) / 2

            else 0.0
        )

        # 12
        rr_ratio = (
            abs(q90)
            /
            (abs(q10) + 1e-8)
        )

        results["REWARD_RISK"] = (

            1.0
            if rr_ratio >= 1.2

            else -1.0
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

        feature_vector = np.array([
            results[k]
            for k in self.feature_names
        ])

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


# =========================================================
# RISK ENGINE
# =========================================================

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
            else 0
        )

        max_ltz = (
            np.max(liquidation_volumes)
            if len(liquidation_volumes)
            else 0
        )

        ltz_score = (
            max_ltz
            /
            (total_ltz + 1e-8)
        ) * 100

        spoof_ratio = (
            cancelled_vol
            /
            (displayed_vol + 1e-8)
        )

        persistence = min(
            max(
                time_exists
                /
                (obs_window + 1e-8),
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


# =========================================================
# SIDEBAR
# =========================================================

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


st.sidebar.markdown(
    "## ⚙️ Terminal Controls"
)

selected_symbol = st.sidebar.selectbox(
    "Cryptocurrency",
    COINS_LIST
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

paper_trading_mode = st.sidebar.toggle(
    "Paper Trading",
    value=True
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### TRI Levels"
)

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
    "15M"
]:

    tri_enabled[tf] = st.sidebar.checkbox(
        tf,
        value=True
    )


api_interval, tf_minutes = TIMEFRAME_MAP[
    selected_tf
]


# =========================================================
# TRI LINE
# =========================================================

TRI_COLORS = {

    "YEARLY": "#38bdf8",
    "MONTHLY": "#ef4444",
    "WEEKLY": "#22c55e",
    "DAILY": "#f8fafc",
    "4H": "#f59e0b",
    "1H": "#a78bfa",
    "30M": "#16a34a",
    "15M": "#60a5fa",

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


@st.cache_data(ttl=30)
def fetch_tri_candle(symbol, interval):

    try:

        url = (
            "https://data-api.binance.vision/api/v3/klines"
        )

        res = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": 3,
            },
            timeout=5,
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
            "Close": float(row[4]),
        }

    except Exception:

        return None


@st.cache_data(ttl=60)
def fetch_tri_yearly_candle(symbol):

    try:

        url = (
            "https://data-api.binance.vision/api/v3/klines"
        )

        res = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": "1M",
                "limit": 36,
            },
            timeout=5,
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

        r = yearly.iloc[-2]

        return {
            "Open": float(r["Open"]),
            "High": float(r["High"]),
            "Low": float(r["Low"]),
            "Close": float(r["Close"]),
        }

    except Exception:

        return None


def calculate_tri_levels(candle):

    if candle is None:
        return None

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


# =========================================================
# MARKET DATA
# =========================================================

@st.cache_data(ttl=15)
def fetch_klines_data(
    symbol,
    timeframe,
    limit=150
):

    try:

        url = (
            "https://data-api.binance.vision/api/v3/klines"
        )

        res = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": timeframe,
                "limit": limit,
            },
            timeout=5,
        )

        data = res.json()

        if (
            not isinstance(data, list)
            or len(data) < 20
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

            df[col] = df[col].astype(float)

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

        dates = pd.date_range(
            end=datetime.datetime.now(),
            periods=limit,
            freq=timeframe
        )

        base = 60000

        closes = (
            base
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
            "Open": closes - 5,
            "High": closes + 15,
            "Low": closes - 15,
            "Close": closes,
            "Volume": np.random.uniform(
                50,
                500,
                limit
            ),

        })


@st.cache_data(ttl=5)
def fetch_order_book_depth(
    symbol,
    depth_limit=20
):

    try:

        url = (
            "https://data-api.binance.vision/api/v3/depth"
        )

        res = requests.get(
            url,
            params={
                "symbol": symbol,
                "limit": depth_limit,
            },
            timeout=4,
        )

        data = res.json()

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

    dummy_bids = np.array([
        [60000 - i * 2, 1.5]
        for i in range(20)
    ])

    dummy_asks = np.array([
        [60000 + i * 2, 1.5]
        for i in range(20)
    ])

    return dummy_bids, dummy_asks


df = fetch_klines_data(
    selected_symbol,
    api_interval
)

bids, asks = fetch_order_book_depth(
    selected_symbol,
    20
)


# =========================================================
# ENGINE
# =========================================================

if (
    not df.empty
    and len(df) >= 20
    and len(bids) > 0
    and len(asks) > 0
):

    lab = TenPaperResearchLab()

    (
        paper_results,
        final_score,
        evolved_weights
    ) = lab.calculate_all_signals(
        df,
        bids,
        asks,
        performance_history=(
            st.session_state.trade_history_log
        )
    )


    # =====================================================
    # PRICE / ATR
    # =====================================================

    close_p = float(
        df["Close"].iloc[-1]
    )

    atr_val = (
        df["High"]
        - df["Low"]
    ).rolling(14).mean().iloc[-1]

    if np.isnan(atr_val):
        atr_val = close_p * 0.005


    # =====================================================
    # REAL 2:1 RISK REWARD
    # =====================================================

    risk_distance = 1.0 * atr_val
    reward_distance = 2.0 * atr_val


    direction = (

        "LONG"
        if final_score >= 0.15

        else "SHORT"
        if final_score <= -0.15

        else "NEUTRAL"

    )


    if direction == "LONG":

        sl_val = close_p - risk_distance
        tp1_val = close_p + reward_distance
        tp2_val = close_p + (3.0 * atr_val)

    elif direction == "SHORT":

        sl_val = close_p + risk_distance
        tp1_val = close_p - reward_distance
        tp2_val = close_p - (3.0 * atr_val)

    else:

        sl_val = close_p - risk_distance
        tp1_val = close_p + reward_distance
        tp2_val = close_p + (3.0 * atr_val)


    # =====================================================
    # ACTUAL RR CALCULATION
    # =====================================================

    if direction == "LONG":

        actual_risk = abs(
            close_p - sl_val
        )

        actual_reward = abs(
            tp1_val - close_p
        )

    elif direction == "SHORT":

        actual_risk = abs(
            sl_val - close_p
        )

        actual_reward = abs(
            close_p - tp1_val
        )

    else:

        actual_risk = abs(
            close_p - sl_val
        )

        actual_reward = abs(
            tp1_val - close_p
        )


    rr_ratio = (
        actual_reward
        /
        (actual_risk + 1e-12)
    )


    confidence = int(
        min(
            max(
                abs(final_score) * 100,
                15
            ),
            98
        )
    )


    # =====================================================
    # BEAM / BASE
    # =====================================================

    beam_level = (
        close_p
        +
        3.0 * atr_val
    )

    base_level = (
        close_p
        -
        3.0 * atr_val
    )


    # =====================================================
    # TRADE ID
    # =====================================================

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
        f"{selected_tf}_"
        f"{time_bucket}_"
        f"{direction}"
    )


    # =====================================================
    # PAPER TRADE SAVE
    # =====================================================

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

                "trade_id": trade_id,

                "timestamp":
                    datetime.datetime.now()
                    .strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "symbol":
                    selected_symbol,

                "timeframe":
                    selected_tf,

                "direction":
                    direction,

                "entry_price":
                    round(close_p, 2),

                "stop_loss":
                    round(sl_val, 2),

                # ACTUAL 2R TARGET
                "tp1":
                    round(tp1_val, 2),

                "tp2":
                    round(tp2_val, 2),

                "exit_price":
                    round(close_p, 2),

                "confidence":
                    confidence,

                "final_score":
                    round(
                        final_score,
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

            }

            st.session_state.trade_history_log.insert(
                0,
                new_trade
            )

            save_persistent_history(
                st.session_state.trade_history_log
            )


    # =====================================================
    # CLOSE PAPER TRADES
    # =====================================================

    for trade in st.session_state.trade_history_log:

        if (
            trade.get("outcome") == "PENDING"
            and trade.get("symbol") == selected_symbol
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
                        ) * 100,
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
                        ) * 100,
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
                        ) * 100,
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
                        ) * 100,
                        2
                    )

                    trade["status"] = "Closed"


    save_persistent_history(
        st.session_state.trade_history_log
    )


    # =====================================================
    # RISK
    # =====================================================

    risk_engine = PowerTradingRiskEngine()

    bid_vol_sum = float(
        np.sum(bids[:, 1])
    )

    ask_vol_sum = float(
        np.sum(asks[:, 1])
    )

    displayed_volume = (
        bid_vol_sum
        +
        ask_vol_sum
    )

    risk_metrics = (
        risk_engine.calculate_risk_metrics(

            liquidation_volumes=
                np.array([
                    1000,
                    2500
                ]),

            displayed_vol=
                displayed_volume,

            cancelled_vol=
                displayed_volume * 0.1,

            time_exists=15,

            obs_window=60,

            open_interest=150000,

            leverage=20,

            volatility=
                df["Close"]
                .pct_change()
                .std()
                +
                1e-8,
        )
    )


    # =====================================================
    # COLORS
    # =====================================================

    if direction == "LONG":

        dir_color = "#22c55e"
        signal_class = "signal-long"

    elif direction == "SHORT":

        dir_color = "#ef4444"
        signal_class = "signal-short"

    else:

        dir_color = "#38bdf8"
        signal_class = "signal-neutral"


    mins_rem, secs_rem = divmod(
        time_remaining,
        60
    )


    # =====================================================
    # HEADER
    # =====================================================

    st.markdown(
        f"""
        <div class="topbar">

            <div class="top-title">
                📊 TRI QUANT RESEARCH TERMINAL
            </div>

            <div class="top-subtitle">
                Live Market Analysis •
                Order Book •
                Quantitative Research •
                Paper Trading
            </div>

            <hr style="
                border:0;
                border-top:1px solid #202b3d;
                margin:12px 0;
            ">

            <div style="
                display:flex;
                gap:25px;
                flex-wrap:wrap;
                font-size:13px;
            ">

                <span>
                    <b>{selected_symbol}</b>
                </span>

                <span>
                    Price:
                    <b>
                        ${close_p:,.2f}
                    </b>
                </span>

                <span>
                    TF:
                    <b>
                        {selected_tf}
                    </b>
                </span>

                <span>
                    Signal:
                    <b style="
                        color:{dir_color};
                    ">
                        {direction}
                    </b>
                </span>

                <span>
                    Score:
                    <b>
                        {final_score:+.3f}
                    </b>
                </span>

                <span>
                    Confidence:
                    <b>
                        {confidence}%
                    </b>
                </span>

                <span>
                    Reset:
                    <b>
                        {mins_rem}m {secs_rem}s
                    </b>
                </span>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # MAIN SIGNAL ROW
    # =====================================================

    c1, c2, c3, c4, c5 = st.columns(5)


    with c1:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Signal
                </div>

                <div class="{signal_class}">
                    {direction}
                </div>

                <div class="card-small">
                    Quantitative composite signal
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Entry
                </div>

                <div class="card-value">
                    ${close_p:,.2f}
                </div>

                <div class="card-small">
                    Current market price
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Stop Loss
                </div>

                <div class="card-value red">
                    ${sl_val:,.2f}
                </div>

                <div class="card-small">
                    Risk = {actual_risk:,.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Take Profit
                </div>

                <div class="card-value green">
                    ${tp1_val:,.2f}
                </div>

                <div class="card-small">
                    Reward = {actual_reward:,.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c5:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Risk / Reward
                </div>

                <div class="rr-value">
                    1 : {rr_ratio:.2f}
                </div>

                <div class="card-small">
                    ACTUAL calculated RR
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # TARGET ROW
    # =====================================================

    st.markdown(
        '<div class="section-title">🎯 Trade Levels</div>',
        unsafe_allow_html=True
    )

    t1, t2, t3, t4 = st.columns(4)


    with t1:

        st.metric(
            "Entry",
            f"${close_p:,.2f}"
        )


    with t2:

        st.metric(
            "Stop Loss",
            f"${sl_val:,.2f}"
        )


    with t3:

        st.metric(
            "TP1 — 2R",
            f"${tp1_val:,.2f}"
        )


    with t4:

        st.metric(
            "TP2 — 3R",
            f"${tp2_val:,.2f}"
        )


    # =====================================================
    # CHART
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Price Trajectory</div>',
        unsafe_allow_html=True
    )

    time_delta = pd.Timedelta(
        minutes=tf_minutes
    )

    future_times = [

        df["Time"].iloc[-1]
        +
        (i * time_delta)

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
            (tp2_val - close_p)
            *
            np.sin(t_steps)
        )

    elif direction == "SHORT":

        forecast_prices = (
            close_p
            -
            (close_p - tp2_val)
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

            name="Price",

            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",

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

            name="Forecast",

            line=dict(
                color=dir_color,
                width=2,
                dash="dot"
            ),

            marker=dict(
                size=4
            ),

        )
    )


    fig.add_hline(
        y=sl_val,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="SL",
        annotation_position="right"
    )


    fig.add_hline(
        y=tp1_val,
        line_dash="dash",
        line_color="#22c55e",
        annotation_text="TP1 2R",
        annotation_position="right"
    )


    fig.add_hline(
        y=tp2_val,
        line_dash="dot",
        line_color="#38bdf8",
        annotation_text="TP2 3R",
        annotation_position="right"
    )


    # TRI LINES

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

            color = TRI_COLORS.get(
                tri_tf,
                "#38bdf8"
            )

            for level_name, width, opacity, dash in [

                ("body_50", 3, 0.90, "solid"),

                ("upper_50", 1, 0.50, "dot"),

                ("lower_50", 1, 0.50, "dot"),

            ]:

                fig.add_hline(

                    y=float(
                        tri[level_name]
                    ),

                    line_color=color,

                    line_width=width,

                    opacity=opacity,

                    line_dash=dash,

                    layer="above",

                )


    fig.update_layout(

        template="plotly_dark",

        height=520,

        paper_bgcolor="#0d131d",

        plot_bgcolor="#0d131d",

        xaxis_rangeslider_visible=False,

        dragmode="pan",

        hovermode="x unified",

        margin=dict(
            l=10,
            r=80,
            t=10,
            b=10
        ),

        xaxis=dict(
            showgrid=True,
            gridcolor="#1e293b",
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="#1e293b",
            zeroline=False,
        ),

        legend=dict(
            orientation="h",
            y=1.02,
            x=1,
            xanchor="right",
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


    # =====================================================
    # MICROSTRUCTURE
    # =====================================================

    st.markdown(
        '<div class="section-title">📚 Market Microstructure</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4, m5 = st.columns(5)


    bid_vol_sum = float(
        np.sum(bids[:, 1])
    )

    ask_vol_sum = float(
        np.sum(asks[:, 1])
    )

    obi_val = (
        bid_vol_sum
        -
        ask_vol_sum
    ) / (
        bid_vol_sum
        +
        ask_vol_sum
        +
        1e-12
    )

    spread_val = abs(
        asks[0, 0]
        -
        bids[0, 0]
    )

    spread_mid = (
        asks[0, 0]
        +
        bids[0, 0]
    ) / 2

    spread_pct = (
        spread_val
        /
        spread_mid
        *
        100
        if spread_mid
        else 0
    )


    with m1:

        st.metric(
            "Bid Volume",
            f"{bid_vol_sum:,.2f}"
        )


    with m2:

        st.metric(
            "Ask Volume",
            f"{ask_vol_sum:,.2f}"
        )


    with m3:

        st.metric(
            "OBI",
            f"{obi_val:+.3f}"
        )


    with m4:

        st.metric(
            "Spread",
            f"${spread_val:.2f}"
        )


    with m5:

        st.metric(
            "Spread %",
            f"{spread_pct:.4f}%"
        )


    # =====================================================
    # TOP OBI
    # =====================================================

    def calc_obi(depth):

        b = float(
            np.sum(
                bids[:depth, 1]
            )
        )

        a = float(
            np.sum(
                asks[:depth, 1]
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


    fig_obi = go.Figure()

    fig_obi.add_trace(
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

            text=[
                f"{x:+.3f}"
                for x in [
                    top5_obi,
                    top10_obi,
                    top20_obi
                ]
            ],

            textposition="outside",

        )
    )

    fig_obi.add_hline(
        y=0,
        line_width=1
    )

    fig_obi.update_layout(

        height=240,

        template="plotly_dark",

        paper_bgcolor="#0d131d",

        plot_bgcolor="#0d131d",

        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),

        yaxis=dict(
            range=[-1, 1]
        ),

        showlegend=False,

    )


    st.plotly_chart(
        fig_obi,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    # =====================================================
    # 12 PAPER SCOREBOARD
    # =====================================================

    st.markdown(
        '<div class="section-title">🔬 Quantitative Research Scoreboard</div>',
        unsafe_allow_html=True
    )

    paper_table = []

    for k, v in paper_results.items():

        status = (

            "PASS"
            if v > 0.1

            else "FAIL"
            if v < -0.1

            else "NEUTRAL"

        )

        paper_table.append({

            "Research Factor":
                k,

            "Value":
                f"{v:+.3f}",

            "Weight":
                f"{evolved_weights[k] * 100:.1f}%",

            "Status":
                status,

        })


    st.dataframe(
        pd.DataFrame(
            paper_table
        ),
        use_container_width=True,
        hide_index=True,
        height=300
    )


    # =====================================================
    # PERFORMANCE
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Paper Trading Performance</div>',
        unsafe_allow_html=True
    )


    if st.session_state.trade_history_log:

        df_log = pd.DataFrame(
            st.session_state.trade_history_log
        )


        f1, f2, f3 = st.columns(3)


        with f1:

            coin_filter = st.selectbox(
                "Coin",
                ["ALL"] + COINS_LIST,
                key="coin_filter"
            )


        with f2:

            tf_filter = st.selectbox(
                "Timeframe",
                ["ALL"]
                +
                list(
                    TIMEFRAME_MAP.keys()
                ),
                key="tf_filter"
            )


        with f3:

            dir_filter = st.selectbox(
                "Direction",
                [
                    "ALL",
                    "LONG",
                    "SHORT"
                ],
                key="dir_filter"
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
                == "WIN"
            ]
        )

        losses = len(
            filtered_df[
                filtered_df["outcome"]
                == "LOSS"
            ]
        )

        pending = len(
            filtered_df[
                filtered_df["outcome"]
                == "PENDING"
            ]
        )

        closed = wins + losses


        win_rate = (

            wins
            /
            closed
            *
            100

            if closed > 0
            else 0
        )


        gross_profit = (
            filtered_df[
                filtered_df["outcome"]
                == "WIN"
            ]["pnl_percent"]
            .sum()
        )

        gross_loss = abs(
            filtered_df[
                filtered_df["outcome"]
                == "LOSS"
            ]["pnl_percent"]
            .sum()
        )


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


        display_cols = [

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
            "pnl_percent",
            "outcome",
            "status",

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

            height=330,

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
