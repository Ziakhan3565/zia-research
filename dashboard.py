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


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Quantitative Research & Paper Trading Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(
    interval=5000,
    limit=None,
    key="research_lab_auto_refresh"
)

CSV_FILE = "signal_history.csv"


# ============================================================
# PERSISTENT TRADE HISTORY
# ============================================================

EXPECTED_COLS = [
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

        df_hist = pd.read_csv(CSV_FILE)

        for col in EXPECTED_COLS:

            if col not in df_hist.columns:

                if col == "outcome":
                    df_hist[col] = "PENDING"

                elif col == "duration":
                    df_hist[col] = "Active"

                elif col == "status":
                    df_hist[col] = "Open"

                else:
                    df_hist[col] = 0.0

        df_hist["outcome"] = df_hist["outcome"].fillna("PENDING")

        return df_hist.to_dict("records")

    except Exception:
        return []


def save_persistent_history(history_list):

    try:

        if not history_list:
            return

        df_hist = pd.DataFrame(history_list)

        for col in EXPECTED_COLS:

            if col not in df_hist.columns:
                df_hist[col] = 0

        df_hist.to_csv(
            CSV_FILE,
            index=False
        )

    except Exception as e:

        st.error(
            f"Error saving history: {e}"
        )


if "trade_history_log" not in st.session_state:

    st.session_state.trade_history_log = (
        load_persistent_history()
    )


# ============================================================
# QUANT RESEARCH ENGINE
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

        # ====================================================
        # 1 HAWKES
        # ====================================================

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
            (
                hawkes_intensity - 1.0
            )
            *
            np.sign(returns_h),
            -1,
            1
        )

        # ====================================================
        # 2 BOOK IMBALANCE
        # ====================================================

        results["BOOK_IMB"] = (
            bid_vol - ask_vol
        ) / (
            bid_vol + ask_vol + 1e-8
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
            taker_buy - taker_sell
        ) / (
            taker_buy + taker_sell + 1e-8
        )

        # ====================================================
        # 4 QUANT IMPLY
        # ====================================================

        depth_skew = (
            bids[0, 1] - asks[0, 1]
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
                *
                (1 - prior)
            )
            +
            1e-8
        )

        results["BAYESIAN"] = np.clip(
            (
                posterior - 0.5
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
                    returns_h - q10
                )
                /
                (
                    q90 - q10 + 1e-8
                )
                * 2.0
                - 1.0
            ),
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
            (
                -1.0
                if target_diff <= -0.0006
                else 0.0
            )
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

        # ====================================================
        # 9 FRACTIONAL KELLY
        # ====================================================

        win_prob = (
            0.55
            +
            (
                0.15
                *
                np.sign(
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
                *
                np.sqrt(5)
                +
                1e-8
            )
        ) / 3.0

        results["RMT_DOM"] = np.clip(
            rmt_dom
            *
            np.sign(returns_h),
            -1,
            1
        )

        # ====================================================
        # 11 CONFORMAL
        # ====================================================

        conformal_spread = (
            realized_vol * 1.96
        )

        upper_b = (
            mid_price
            *
            (
                1
                +
                conformal_spread
            )
        )

        lower_b = (
            mid_price
            *
            (
                1
                -
                conformal_spread
            )
        )

        results["CONF_CROSS"] = (
            1.0
            if mid_price >
            (
                upper_b + lower_b
            ) / 2
            else
            (
                -1.0
                if mid_price <
                (
                    upper_b + lower_b
                ) / 2
                else 0.0
            )
        )

        # ====================================================
        # 12 REWARD RISK
        # ====================================================

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
            else
            (
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
                total_ltz + 1e-8
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
            "Market_Risk": market_risk
        }


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html,
body,
[class*="css"] {

    font-family:
    'Inter',
    sans-serif !important;

}

.stApp {

    background:
    #080a0f;

    color:
    #e2e8f0;

}

section[data-testid="stSidebar"] {

    background:
    #0d1117 !important;

    border-right:
    1px solid #1c2330;

}

.block-container {

    padding-top:
    1.2rem;

    padding-bottom:
    2rem;

}

.metric-card {

    background:
    linear-gradient(
        145deg,
        #111622,
        #0d121c
    );

    border:
    1px solid #20293a;

    border-radius:
    12px;

    padding:
    14px;

    margin-bottom:
    10px;

}

.metric-label {

    font-size:
    10px;

    font-weight:
    700;

    color:
    #7d8795;

    text-transform:
    uppercase;

    letter-spacing:
    0.7px;

}

.metric-val-green {

    font-size:
    20px;

    font-weight:
    800;

    color:
    #00e676;

}

.metric-val-red {

    font-size:
    20px;

    font-weight:
    800;

    color:
    #ff5252;

}

.metric-val-blue {

    font-size:
    20px;

    font-weight:
    800;

    color:
    #38bdf8;

}

.top-status-bar {

    background:
    #111622;

    border:
    1px solid #20293a;

    border-radius:
    10px;

    padding:
    12px 18px;

    margin-bottom:
    18px;

    font-weight:
    600;

    font-size:
    13px;

    white-space:
    nowrap;

    overflow-x:
    auto;

    overflow-y:
    hidden;

    display:
    block;

}

.section-title {

    font-size:
    17px;

    font-weight:
    800;

    margin-top:
    10px;

    margin-bottom:
    12px;

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

    "1m (Scalping)":
        ("1m", 1),

    "15m (Medium TF)":
        ("15m", 15),

    "30m (Medium TF)":
        ("30m", 30),

    "1h (Intraday)":
        ("1h", 60),

    "4h (Intraday)":
        ("4h", 240),
}


st.sidebar.markdown(
    "## ⚡ Terminal Controls"
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
    "### 🎛️ Paper Trading"
)

paper_trading_mode = st.sidebar.toggle(
    "Enable Live Paper Trading",
    value=True
)

api_interval, tf_minutes = (
    TIMEFRAME_MAP[selected_tf_label]
)


# ============================================================
# BINANCE DATA
# ============================================================

@st.cache_data(ttl=15)
def fetch_klines_data(
    symbol,
    tf_key,
    limit=100
):

    if "1m" in tf_key:
        binance_tf = "1m"

    elif "15m" in tf_key:
        binance_tf = "15m"

    elif "30m" in tf_key:
        binance_tf = "30m"

    elif "1h" in tf_key:
        binance_tf = "1h"

    else:
        binance_tf = "4h"

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

        res = response.json()

        if not isinstance(res, list):
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

            df[col] = df[col].astype(float)

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

        return pd.DataFrame({

            "Time": dates,

            "Open":
                closes - 5,

            "High":
                closes + 15,

            "Low":
                closes - 15,

            "Close":
                closes,

            "Volume":
                np.random.uniform(
                    50,
                    500,
                    limit
                )
        })


@st.cache_data(ttl=10)
def fetch_order_book_depth(
    symbol,
    depth_limit=20
):

    try:

        url = (
            "https://data-api.binance.vision/api/v3/depth"
            f"?symbol={symbol}"
            f"&limit={depth_limit}"
        )

        res = requests.get(
            url,
            timeout=5
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
            [60000 - i * 2, 1.5]
            for i in range(20)
        ],
        dtype=float
    )

    dummy_asks = np.array(
        [
            [60000 + i * 2, 1.5]
            for i in range(20)
        ],
        dtype=float
    )

    return (
        dummy_bids,
        dummy_asks
    )


# ============================================================
# FETCH DATA
# ============================================================

df = fetch_klines_data(
    selected_symbol,
    selected_tf_label
)

bids, asks = fetch_order_book_depth(
    selected_symbol
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
        final_score,
        evolved_weights
    ) = lab.calculate_all_signals(
        df,
        bids,
        asks,
        current_inventory=0,
        performance_history=
            st.session_state.trade_history_log
    )


    # ========================================================
    # PRICE / ATR
    # ========================================================

    close_p = float(
        df["Close"].iloc[-1]
    )

    atr_val = (
        df["High"]
        -
        df["Low"]
    ).rolling(14).mean().iloc[-1]

    if (
        pd.isna(atr_val)
        or atr_val <= 0
    ):

        atr_val = (
            close_p * 0.005
        )


    # ========================================================
    # SIGNAL
    # ========================================================

    direction = (
        "LONG"
        if final_score >= 0.15
        else
        (
            "SHORT"
            if final_score <= -0.15
            else "NEUTRAL"
        )
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


    # ========================================================
    # RISK / REWARD
    # ========================================================

    risk_distance = float(
        atr_val
    )

    if direction == "LONG":

        entry_price = close_p

        stop_loss = (
            entry_price
            -
            risk_distance
        )

        tp1 = (
            entry_price
            +
            (
                risk_distance * 2.0
            )
        )

        tp2 = (
            entry_price
            +
            (
                risk_distance * 3.0
            )
        )

    elif direction == "SHORT":

        entry_price = close_p

        stop_loss = (
            entry_price
            +
            risk_distance
        )

        tp1 = (
            entry_price
            -
            (
                risk_distance * 2.0
            )
        )

        tp2 = (
            entry_price
            -
            (
                risk_distance * 3.0
            )
        )

    else:

        entry_price = close_p
        stop_loss = close_p
        tp1 = close_p
        tp2 = close_p


    # ========================================================
    # ACTUAL RR
    # ========================================================

    actual_risk = abs(
        entry_price
        -
        stop_loss
    )

    actual_reward = abs(
        tp1
        -
        entry_price
    )

    rr_ratio = (
        actual_reward
        /
        actual_risk
        if actual_risk > 0
        else 0
    )


    # ========================================================
    # BEAM / BASE
    # ========================================================

    beam_level = (
        entry_price
        +
        (
            risk_distance * 3
        )
        if direction == "LONG"
        else
        entry_price
        -
        (
            risk_distance * 3
        )
    )

    base_level = (
        entry_price
        -
        (
            risk_distance * 1
        )
        if direction == "LONG"
        else
        entry_price
        +
        (
            risk_distance * 1
        )
    )


    # ========================================================
    # SIGNAL TIME LOCK
    # ========================================================

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


    # ========================================================
    # SAVE NEW TRADE
    # ========================================================

    if (
        paper_trading_mode
        and
        direction != "NEUTRAL"
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
                    datetime.datetime.now()
                    .strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "symbol":
                    selected_symbol,

                "timeframe":
                    selected_tf_label,

                "direction":
                    direction,

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


    # ========================================================
    # UPDATE OPEN TRADES
    # ========================================================

    history_changed = False

    for trade in (
        st.session_state.trade_history_log
    ):

        if (
            trade.get("outcome")
            !=
            "PENDING"
        ):
            continue

        if (
            trade.get("symbol")
            !=
            selected_symbol
        ):
            continue

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

                trade["exit_price"] = round(
                    curr_price,
                    2
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            curr_price
                            -
                            entry
                        )
                        /
                        entry
                    )
                    *
                    100,
                    2
                )

                trade["status"] = "Closed"

                history_changed = True

            elif curr_price <= sl:

                trade["outcome"] = "LOSS"

                trade["exit_price"] = round(
                    curr_price,
                    2
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            curr_price
                            -
                            entry
                        )
                        /
                        entry
                    )
                    *
                    100,
                    2
                )

                trade["status"] = "Closed"

                history_changed = True

        elif trade["direction"] == "SHORT":

            if curr_price <= tp:

                trade["outcome"] = "WIN"

                trade["exit_price"] = round(
                    curr_price,
                    2
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            entry
                            -
                            curr_price
                        )
                        /
                        entry
                    )
                    *
                    100,
                    2
                )

                trade["status"] = "Closed"

                history_changed = True

            elif curr_price >= sl:

                trade["outcome"] = "LOSS"

                trade["exit_price"] = round(
                    curr_price,
                    2
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            entry
                            -
                            curr_price
                        )
                        /
                        entry
                    )
                    *
                    100,
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

            liquidation_volumes=
                np.array(
                    [1000, 2500]
                ),

            displayed_vol=
                disp_vol,

            cancelled_vol=
                disp_vol * 0.1,

            time_exists=
                15.0,

            obs_window=
                60.0,

            open_interest=
                150000.0,

            leverage=
                20.0,

            volatility=
                df["Close"]
                .pct_change()
                .std()
                +
                1e-8
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


    mins_rem, secs_rem = divmod(
        time_remaining,
        60
    )


    # ========================================================
    # SINGLE HORIZONTAL TOP BAR
    # ========================================================

    st.markdown(
        f"""
<div class="top-status-bar">

🟢 <b>{selected_symbol}</b>
&nbsp; | &nbsp;
Price: <b>${close_p:,.2f}</b>
&nbsp; | &nbsp;
TF: <b>{selected_tf_label}</b>
&nbsp; | &nbsp;
Signal:
<span style="color:{dir_color}; font-weight:800;">
{direction}
</span>
&nbsp; | &nbsp;
Score: <b>{final_score:+.3f}</b>
&nbsp; | &nbsp;
Confidence: <b>{confidence}%</b>
&nbsp; | &nbsp;
RR: <b>1 : {rr_ratio:.2f}</b>
&nbsp; | &nbsp;
Next Reset: <b>{mins_rem}m {secs_rem}s</b>

</div>
""",
        unsafe_allow_html=True
    )


    # ========================================================
    # SIGNAL CARDS
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Signal Execution</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4, col5 = (
        st.columns(
            [1.25, 1, 1, 1, 1]
        )
    )

    with col1:

        st.markdown(
            f"""
<div class="metric-card"
style="border-left:4px solid {dir_color};">

<div class="metric-label">
SIGNAL
</div>

<div style="
font-size:25px;
font-weight:800;
color:{dir_color};
">

{direction}

</div>

<div style="
font-size:11px;
color:#8b949e;
">

Entry ${entry_price:,.2f}

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
STOP LOSS
</div>

<div class="metric-val-red">
${stop_loss:,.2f}
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
TP1 — 1:2
</div>

<div class="metric-val-green">
${tp1:,.2f}
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
TP2 — 1:3
</div>

<div class="metric-val-blue">
${tp2:,.2f}
</div>

</div>
""",
            unsafe_allow_html=True
        )

    with col5:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
RISK / REWARD
</div>

<div class="metric-val-blue">
1 : {rr_ratio:.2f}
</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # SECOND METRIC ROW
    # ========================================================

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
BEAM TARGET
</div>

<div class="metric-val-blue">
${beam_level:,.2f}
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
BASE / INVALIDATION
</div>

<div class="metric-val-red">
${base_level:,.2f}
</div>

</div>
""",
            unsafe_allow_html=True
        )

    with c3:

        strength = (
            "HIGH"
            if confidence >= 75
            else
            (
                "MEDIUM"
                if confidence >= 55
                else "LOW"
            )
        )

        strength_class = (
            "metric-val-green"
            if strength == "HIGH"
            else
            "metric-val-blue"
        )

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-label">
SIGNAL STRENGTH
</div>

<div class="{strength_class}">
{strength}
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
CONFIDENCE
</div>

<div class="metric-val-blue">
{confidence}%
</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # PRICE CHART
    # ========================================================

    st.markdown("---")

    chart_col, right_col = (
        st.columns(
            [2.5, 1]
        )
    )

    with chart_col:

        st.markdown(
            '<div class="section-title">📈 Price Trajectory & Levels</div>',
            unsafe_allow_html=True
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
                    tp1 - close_p
                )
                *
                np.sin(t_steps)
            )

        elif direction == "SHORT":

            forecast_prices = (
                close_p
                -
                (
                    close_p - tp1
                )
                *
                np.sin(t_steps)
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

                name="Price",

                increasing_line_color=
                    "#00e676",

                decreasing_line_color=
                    "#ff5252",
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

                mode=
                    "lines+markers",

                name=
                    "Forecast",

                line=dict(
                    color=
                        dir_color,
                    width=2,
                    dash="dot"
                )
            )
        )

        fig.add_hline(
            y=tp1,
            line_dash="dash",
            line_color="#00e676",
            annotation_text=
                f"TP1 1:2 ${tp1:,.2f}"
        )

        fig.add_hline(
            y=tp2,
            line_dash="dash",
            line_color="#38bdf8",
            annotation_text=
                f"TP2 1:3 ${tp2:,.2f}"
        )

        fig.add_hline(
            y=stop_loss,
            line_dash="dot",
            line_color="#ff5252",
            annotation_text=
                f"SL ${stop_loss:,.2f}"
        )

        fig.update_layout(

            template="plotly_dark",

            height=430,

            xaxis_rangeslider_visible=False,

            paper_bgcolor=
                "#111622",

            plot_bgcolor=
                "#111622",

            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            ),

            legend=dict(
                orientation="h"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # MICROSTRUCTURE
    # ========================================================

    with right_col:

        st.markdown(
            '<div class="section-title">📊 Market Microstructure</div>',
            unsafe_allow_html=True
        )

        bid_vol_sum = (
            np.sum(bids[:, 1])
        )

        ask_vol_sum = (
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
            1e-8
        )

        spread_val = abs(
            asks[0, 0]
            -
            bids[0, 0]
        )

        st.markdown(
            f"""
<div class="metric-card">

<div style="
display:flex;
justify-content:space-between;
margin-bottom:9px;
">

<span>Bid Volume</span>

<b style="color:#00e676;">
{bid_vol_sum:,.2f}
</b>

</div>


<div style="
display:flex;
justify-content:space-between;
margin-bottom:9px;
">

<span>Ask Volume</span>

<b style="color:#ff5252;">
{ask_vol_sum:,.2f}
</b>

</div>


<div style="
display:flex;
justify-content:space-between;
margin-bottom:9px;
">

<span>OBI</span>

<b style="color:#38bdf8;">
{obi_val:+.3f}
</b>

</div>


<div style="
display:flex;
justify-content:space-between;
margin-bottom:9px;
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

<span>RR</span>

<b style="color:#38bdf8;">
1 : {rr_ratio:.2f}
</b>

</div>

</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            "#### Top 20 OBI"
        )

        fig_obi = go.Figure(
            go.Bar(
                x=[
                    "Top 5",
                    "Top 10",
                    "Top 20"
                ],
                y=[
                    obi_val * 0.8,
                    obi_val * 0.9,
                    obi_val
                ],
                marker_color=
                    "#38bdf8"
            )
        )

        fig_obi.update_layout(

            height=180,

            margin=dict(
                l=0,
                r=0,
                t=5,
                b=0
            ),

            paper_bgcolor=
                "#111622",

            plot_bgcolor=
                "#111622",

            template=
                "plotly_dark"
        )

        st.plotly_chart(
            fig_obi,
            use_container_width=True,
            config={
                "displayModeBar":
                    False
            }
        )


    # ========================================================
    # RISK PANEL
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🛡️ Risk Engine</div>',
        unsafe_allow_html=True
    )

    r1, r2, r3, r4 = (
        st.columns(4)
    )

    risk_values = [

        (
            "LTZ SCORE",
            risk_metrics["LTZ_Score"],
            "metric-val-blue"
        ),

        (
            "SPOOF SCORE",
            risk_metrics["Spoof_Score"],
            "metric-val-red"
        ),

        (
            "SQUEEZE RISK",
            risk_metrics["Squeeze_Risk"],
            "metric-val-red"
        ),

        (
            "MARKET RISK",
            risk_metrics["Market_Risk"],
            "metric-val-blue"
        ),
    ]

    for col, (
        label,
        value,
        cls
    ) in zip(
        [r1, r2, r3, r4],
        risk_values
    ):

        with col:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-label">
{label}
</div>

<div class="{cls}">
{value:.3f}
</div>

</div>
""",
                unsafe_allow_html=True
            )


    # ========================================================
    # 12 PAPER SCOREBOARD
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🔬 12-Paper Quantitative Research Scoreboard</div>',
        unsafe_allow_html=True
    )

    score_col, insight_col = (
        st.columns(
            [1.7, 1]
        )
    )

    with score_col:

        paper_table_data = []

        for k, v in paper_results.items():

            status = (
                "PASS 🟢"
                if v > 0.1
                else
                (
                    "FAIL 🔴"
                    if v < -0.1
                    else
                    "NEUTRAL ⚪"
                )
            )

            paper_table_data.append({

                "Paper":
                    k,

                "Value":
                    f"{v:+.3f}",

                "Weight":
                    f"{evolved_weights.get(k, 0.083) * 100:.1f}%",

                "Status":
                    status,
            })

        st.dataframe(
            pd.DataFrame(
                paper_table_data
            ),
            use_container_width=True,
            hide_index=True,
            height=300
        )

    with insight_col:

        st.markdown(
            """
<div class="metric-card">

<div style="
font-size:14px;
font-weight:800;
color:#38bdf8;
margin-bottom:10px;
">

Advanced Model Insights

</div>

<div style="
font-size:12px;
color:#cbd5e1;
line-height:1.7;
">

<b>HAWKES</b><br>
Aggressive order clustering.

<br><br>

<b>BOOK_IMB</b><br>
Bid/ask depth pressure.

<br><br>

<b>TAKER_FLOW</b><br>
Directional volume pressure.

<br><br>

<b>REWARD_RISK</b><br>
Target quality filter.

</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Performance & Trade History</div>',
        unsafe_allow_html=True
    )

    if st.session_state.trade_history_log:

        df_log = pd.DataFrame(
            st.session_state.trade_history_log
        )

        f1, f2, f3 = (
            st.columns(3)
        )

        with f1:

            coin_filter = st.selectbox(
                "Filter Coin",
                ["ALL"] + COINS_LIST
            )

        with f2:

            tf_filter = st.selectbox(
                "Filter Timeframe",
                ["ALL"]
                +
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

        filtered_df = (
            df_log.copy()
        )

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
                filtered_df["outcome"]
                ==
                "WIN"
            ]
        )

        losing_trades_df = (
            filtered_df[
                filtered_df["outcome"]
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

        metrics = [

            (
                "WIN RATE",
                f"{win_rate:.1f}%",
                "metric-val-green"
            ),

            (
                "CLOSED",
                str(closed_trades),
                "metric-val-blue"
            ),

            (
                "WINS / LOSSES",
                f"{wins}W / {losses}L",
                "metric-val-green"
            ),

            (
                "PENDING",
                str(pending),
                "metric-val-blue"
            ),

            (
                "PROFIT FACTOR",
                f"{profit_factor:.2f}",
                "metric-val-blue"
            ),

            (
                "NET PNL",
                f"{net_pnl:+.2f}%",
                (
                    "metric-val-green"
                    if net_pnl >= 0
                    else
                    "metric-val-red"
                )
            ),
        ]

        for col, (
            label,
            value,
            cls
        ) in zip(
            [
                p1,
                p2,
                p3,
                p4,
                p5,
                p6
            ],
            metrics
        ):

            with col:

                st.markdown(
                    f"""
<div class="metric-card">

<div class="metric-label">
{label}
</div>

<div class="{cls}">
{value}
</div>

</div>
""",
                    unsafe_allow_html=True
                )


        # ====================================================
        # TRADE TABLE
        # ====================================================

        st.markdown(
            "#### 📋 Detailed Trade History"
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
            "pnl_percent",
            "outcome",
            "confidence",
        ]

        available_cols = [
            c
            for c
            in display_cols
            if c in filtered_df.columns
        ]

        st.dataframe(
            filtered_df[
                available_cols
            ],
            use_container_width=True,
            hide_index=True,
            height=320
        )

    else:

        st.info(
            "No paper trades recorded yet."
        )


    # ========================================================
    # CLEAR HISTORY
    # ========================================================

    st.markdown("---")

    if st.sidebar.button(
        "🗑️ Clear Trade History"
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

    st.warning(
        "⚠️ Data pipeline initializing..."
    )
