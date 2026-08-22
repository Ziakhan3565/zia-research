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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TRI Quant Research Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_autorefresh(
    interval=5000,
    limit=None,
    key="tri_quant_auto_refresh",
)


# ============================================================
# CONSTANTS
# ============================================================

CSV_FILE = "signal_history.csv"

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

TRI_COLORS = {
    "YEARLY": "#38bdf8",
    "MONTHLY": "#ef4444",
    "WEEKLY": "#22c55e",
    "DAILY": "#f8fafc",
    "4H": "#f59e0b",
    "1H": "#a855f7",
    "30M": "#16a34a",
    "15M": "#3b82f6",
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
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(56,189,248,.07), transparent 30%),
        radial-gradient(circle at 90% 10%, rgba(168,85,247,.06), transparent 30%),
        #070b12;
    color: #e5e7eb;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: #0a0f18 !important;
    border-right: 1px solid #182233;
}

section[data-testid="stSidebar"] * {
    color: #dbe4f0;
}

h1, h2, h3, h4 {
    letter-spacing: -0.3px;
}

.terminal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 22px;
    margin-bottom: 14px;
    border: 1px solid #1b2738;
    border-radius: 16px;
    background: rgba(14,20,31,.92);
    box-shadow: 0 10px 35px rgba(0,0,0,.18);
}

.brand-title {
    font-size: 24px;
    font-weight: 800;
    color: #f8fafc;
}

.brand-sub {
    font-size: 11px;
    color: #718096;
    margin-top: 4px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.live-badge {
    padding: 7px 12px;
    border-radius: 20px;
    border: 1px solid #14532d;
    background: rgba(34,197,94,.08);
    color: #4ade80;
    font-size: 11px;
    font-weight: 700;
}

.section-title {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 20px 0 10px 2px;
}

.card {
    background: #0e141f;
    border: 1px solid #1b2738;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 8px 28px rgba(0,0,0,.15);
}

.signal-card {
    background: linear-gradient(
        135deg,
        rgba(14,20,31,.98),
        rgba(16,24,38,.98)
    );
    border: 1px solid #25344a;
    border-radius: 16px;
    padding: 20px;
    min-height: 185px;
}

.signal-title {
    color: #64748b;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
}

.signal-main {
    font-size: 34px;
    font-weight: 850;
    margin: 8px 0 8px 0;
}

.signal-price {
    font-size: 14px;
    color: #cbd5e1;
}

.metric {
    background: #0e141f;
    border: 1px solid #1b2738;
    border-radius: 13px;
    padding: 14px 15px;
    min-height: 78px;
}

.metric-label {
    color: #64748b;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .9px;
    font-weight: 700;
}

.metric-value {
    color: #f1f5f9;
    font-size: 20px;
    font-weight: 800;
    margin-top: 6px;
}

.metric-small {
    color: #94a3b8;
    font-size: 11px;
    margin-top: 2px;
}

.price-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}

.price-box {
    background: #0b111b;
    border: 1px solid #1b2738;
    border-radius: 10px;
    padding: 11px;
}

.price-label {
    color: #64748b;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .8px;
}

.price-value {
    font-size: 14px;
    font-weight: 750;
    margin-top: 5px;
}

.long {
    color: #4ade80;
}

.short {
    color: #fb7185;
}

.neutral {
    color: #38bdf8;
}

.blue {
    color: #38bdf8;
}

.warning {
    color: #fbbf24;
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.panel-title {
    color: #e2e8f0;
    font-size: 14px;
    font-weight: 750;
}

.panel-subtitle {
    color: #64748b;
    font-size: 10px;
}

.rr-box {
    border: 1px solid #25405c;
    background: rgba(56,189,248,.045);
    border-radius: 12px;
    padding: 13px;
    text-align: center;
}

.rr-value {
    font-size: 25px;
    font-weight: 850;
    color: #38bdf8;
}

.rr-label {
    color: #64748b;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.micro-row {
    display: flex;
    justify-content: space-between;
    padding: 9px 0;
    border-bottom: 1px solid #172131;
    font-size: 12px;
}

.micro-row:last-child {
    border-bottom: none;
}

.micro-name {
    color: #94a3b8;
}

.micro-value {
    color: #e2e8f0;
    font-weight: 700;
}

.status-pill {
    display: inline-block;
    padding: 4px 9px;
    border-radius: 12px;
    font-size: 9px;
    font-weight: 800;
}

.pill-green {
    color: #4ade80;
    background: rgba(34,197,94,.10);
    border: 1px solid rgba(34,197,94,.22);
}

.pill-red {
    color: #fb7185;
    background: rgba(244,63,94,.10);
    border: 1px solid rgba(244,63,94,.22);
}

.pill-blue {
    color: #38bdf8;
    background: rgba(56,189,248,.10);
    border: 1px solid rgba(56,189,248,.22);
}

div[data-testid="stMetric"] {
    background: #0e141f;
    border: 1px solid #1b2738;
    border-radius: 13px;
    padding: 10px;
}

.stButton > button {
    border-radius: 9px;
    border: 1px solid #25344a;
    background: #111a28;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HISTORY
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


def load_history():
    if not os.path.exists(CSV_FILE):
        return []

    try:
        df = pd.read_csv(CSV_FILE)

        for col in EXPECTED_COLS:
            if col not in df.columns:
                if col == "outcome":
                    df[col] = "PENDING"
                elif col == "duration":
                    df[col] = "Active"
                elif col == "status":
                    df[col] = "Open"
                else:
                    df[col] = 0.0

        return df[EXPECTED_COLS].to_dict("records")

    except Exception:
        return []


def save_history(history):
    try:
        pd.DataFrame(history, columns=EXPECTED_COLS).to_csv(
            CSV_FILE,
            index=False,
        )
    except Exception:
        pass


if "trade_history" not in st.session_state:
    st.session_state.trade_history = load_history()


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

    def extract_features(self, df, bids, asks):

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

        results = {}

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
            df["Close"].iloc[-5] + 1e-8
        )

        delta_p = (
            df["Close"].iloc[-1]
            - df["Close"].iloc[-2]
        )

        # HAWKES
        vol_changes = (
            df["Volume"]
            .pct_change()
            .dropna()
            .values
        )

        if len(vol_changes) >= 15:
            hawkes = (
                np.mean(vol_changes[-3:])
                / (
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
            1,
        )

        # BOOK IMBALANCE
        results["BOOK_IMB"] = (
            (bid_vol - ask_vol)
            / (bid_vol + ask_vol + 1e-8)
        )

        # TAKER FLOW
        taker_buy = (
            df["Volume"].iloc[-1]
            * (1.0 if delta_p > 0 else 0.3)
        )

        taker_sell = (
            df["Volume"].iloc[-1]
            * (1.0 if delta_p <= 0 else 0.3)
        )

        results["TAKER_FLOW"] = (
            (taker_buy - taker_sell)
            / (taker_buy + taker_sell + 1e-8)
        )

        # DEPTH SKEW
        depth_skew = (
            bids[0, 1] - asks[0, 1]
        ) / (
            bids[0, 1] + asks[0, 1] + 1e-8
        )

        results["QUANT_IMPLY"] = np.clip(
            depth_skew * 1.5,
            -1,
            1,
        )

        # BAYESIAN
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
            + (1 - likelihood)
            * (1 - prior)
            + 1e-8
        )

        results["BAYESIAN"] = np.clip(
            (posterior - 0.5) * 2,
            -1,
            1,
        )

        # QUANTILES
        q90 = (
            returns.quantile(.90)
            if len(returns) > 5
            else .01
        )

        q10 = (
            returns.quantile(.10)
            if len(returns) > 5
            else -.01
        )

        results["QUANTILES"] = np.clip(
            (
                (returns_h - q10)
                / (q90 - q10 + 1e-8)
            ) * 2 - 1,
            -1,
            1,
        )

        # TARGET / INVALIDATION
        target_diff = (
            delta_p
            / (df["Close"].iloc[-1] + 1e-8)
        )

        if target_diff >= .0006:
            results["TARGET_INV"] = 1.0
        elif target_diff <= -.0006:
            results["TARGET_INV"] = -1.0
        else:
            results["TARGET_INV"] = 0.0

        # ADAPTIVE CONF
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
            (ma_fast - ma_slow)
            / (
                realized_vol
                * mid_price
                + 1e-8
            ),
            -1,
            1,
        )

        # FRACTIONAL KELLY
        win_prob = (
            .55
            + .15
            * np.sign(results["BOOK_IMB"])
        )

        kelly = (
            win_prob
            - ((1 - win_prob) / 1.5)
        )

        results["FRAC_KELLY"] = np.clip(
            kelly
            * 2
            * np.sign(returns_h),
            -1,
            1,
        )

        # RMT
        rmt = (
            abs(returns_h)
            / (
                realized_vol
                * np.sqrt(5)
                + 1e-8
            )
        ) / 3

        results["RMT_DOM"] = np.clip(
            rmt * np.sign(returns_h),
            -1,
            1,
        )

        # CONF CROSS
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

        midpoint = (
            upper_b + lower_b
        ) / 2

        if mid_price > midpoint:
            results["CONF_CROSS"] = 1.0
        elif mid_price < midpoint:
            results["CONF_CROSS"] = -1.0
        else:
            results["CONF_CROSS"] = 0.0

        # REWARD RISK
        rr_ratio = (
            abs(q90)
            / (abs(q10) + 1e-8)
        )

        if rr_ratio >= 1.2:
            results["REWARD_RISK"] = 1.0
        elif rr_ratio < .8:
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
        performance_history=None,
    ):

        features = self.extract_features(
            df,
            bids,
            asks,
        )

        vector = np.array(
            [
                features[k]
                for k in self.feature_names
            ]
        )

        weights = np.array(
            list(self.dynamic_weights.values())
        )

        score = float(
            np.dot(vector, weights)
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
        liquidation_volumes,
        displayed_vol,
        cancelled_vol,
        time_exists,
        obs_window,
        open_interest,
        leverage,
        volatility,
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
            / (total_ltz + 1e-8)
        ) * 100

        spoof_ratio = (
            cancelled_vol
            / (displayed_vol + 1e-8)
        )

        persistence = min(
            max(
                time_exists
                / (obs_window + 1e-8),
                0,
            ),
            1,
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
# BINANCE DATA
# ============================================================

@st.cache_data(ttl=10)
def fetch_klines(symbol, interval, limit=150):

    try:

        url = (
            "https://data-api.binance.vision"
            "/api/v3/klines"
        )

        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            },
            timeout=5,
        )

        data = response.json()

        if not isinstance(data, list):
            return pd.DataFrame()

        columns = [
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

        df = pd.DataFrame(
            data,
            columns=columns,
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

        dates = pd.date_range(
            end=datetime.datetime.now(),
            periods=limit,
            freq=interval,
        )

        price = 60000 + np.cumsum(
            np.random.normal(
                0,
                10,
                limit,
            )
        )

        return pd.DataFrame(
            {
                "Time": dates,
                "Open": price - 5,
                "High": price + 15,
                "Low": price - 15,
                "Close": price,
                "Volume": np.random.uniform(
                    50,
                    500,
                    limit,
                ),
            }
        )


@st.cache_data(ttl=5)
def fetch_orderbook(symbol):

    try:

        url = (
            "https://data-api.binance.vision"
            "/api/v3/depth"
        )

        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "limit": 50,
            },
            timeout=5,
        )

        data = response.json()

        if (
            isinstance(data, dict)
            and "bids" in data
            and "asks" in data
        ):

            return (
                np.array(
                    data["bids"],
                    dtype=float,
                ),
                np.array(
                    data["asks"],
                    dtype=float,
                ),
            )

    except Exception:
        pass

    bids = np.array(
        [
            [60000 - i * 2, 1.5]
            for i in range(50)
        ],
        dtype=float,
    )

    asks = np.array(
        [
            [60000 + i * 2, 1.5]
            for i in range(50)
        ],
        dtype=float,
    )

    return bids, asks


# ============================================================
# TRI DATA
# ============================================================

@st.cache_data(ttl=30)
def fetch_tri_candle(symbol, interval):

    try:

        url = (
            "https://data-api.binance.vision"
            "/api/v3/klines"
        )

        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": 3,
            },
            timeout=5,
        )

        data = response.json()

        if not isinstance(data, list):
            return None

        if len(data) < 2:
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
def fetch_yearly(symbol):

    try:

        url = (
            "https://data-api.binance.vision"
            "/api/v3/klines"
        )

        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": "1M",
                "limit": 36,
            },
            timeout=5,
        )

        data = response.json()

        if not isinstance(data, list):
            return None

        rows = []

        for row in data:

            rows.append(
                {
                    "Time": pd.to_datetime(
                        row[0],
                        unit="ms",
                        utc=True,
                    ),
                    "Open": float(row[1]),
                    "High": float(row[2]),
                    "Low": float(row[3]),
                    "Close": float(row[4]),
                }
            )

        monthly = (
            pd.DataFrame(rows)
            .set_index("Time")
        )

        yearly = (
            monthly
            .resample("YS")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                }
            )
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


def tri_levels(candle):

    if candle is None:
        return None

    o = float(candle["Open"])
    h = float(candle["High"])
    l = float(candle["Low"])
    c = float(candle["Close"])

    body_high = max(o, c)
    body_low = min(o, c)

    return {
        "body_50": (
            body_high + body_low
        ) / 2,

        "upper_50": (
            h + body_high
        ) / 2,

        "lower_50": (
            l + body_low
        ) / 2,
    }


@st.cache_data(ttl=30)
def get_tri_levels(symbol):

    result = {}

    yearly = tri_levels(
        fetch_yearly(symbol)
    )

    if yearly:
        result["YEARLY"] = yearly

    for name, interval in TRI_INTERVALS.items():

        level = tri_levels(
            fetch_tri_candle(
                symbol,
                interval,
            )
        )

        if level:
            result[name] = level

    return result


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ⚡ Terminal Controls"
)

selected_symbol = st.sidebar.selectbox(
    "Asset",
    COINS_LIST,
)

selected_tf = st.sidebar.selectbox(
    "Timeframe",
    list(TIMEFRAME_MAP.keys()),
    index=1,
)

forecast_horizon = st.sidebar.slider(
    "Forecast Candles",
    5,
    30,
    15,
)

paper_mode = st.sidebar.toggle(
    "Paper Trading",
    value=True,
)

st.sidebar.markdown("---")

show_tri = st.sidebar.toggle(
    "TRI Lines",
    value=True,
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
        key=f"tri_{tf}",
    )

interval, tf_minutes = TIMEFRAME_MAP[
    selected_tf
]


# ============================================================
# FETCH MARKET DATA
# ============================================================

df = fetch_klines(
    selected_symbol,
    interval,
    150,
)

bids, asks = fetch_orderbook(
    selected_symbol
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
<div class="terminal-header">

    <div>
        <div class="brand-title">
            TRI Quant Research Terminal
        </div>

        <div class="brand-sub">
            Market Research • Order Flow • Paper Trading
        </div>
    </div>

    <div class="live-badge">
        ● LIVE DATA
    </div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN ENGINE
# ============================================================

if (
    not df.empty
    and len(df) >= 20
    and len(bids) > 0
    and len(asks) > 0
):

    lab = TenPaperResearchLab()

    paper_results, final_score, weights = (
        lab.calculate_all_signals(
            df,
            bids,
            asks,
            performance_history=st.session_state.trade_history,
        )
    )

    close_price = float(
        df["Close"].iloc[-1]
    )

    atr = (
        df["High"] - df["Low"]
    ).rolling(14).mean().iloc[-1]

    if np.isnan(atr) or atr <= 0:
        atr = close_price * 0.005

    # ========================================================
    # SIGNAL
    # ========================================================

    if final_score >= 0.15:
        direction = "LONG"
    elif final_score <= -0.15:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    confidence = int(
        min(
            max(abs(final_score) * 100, 15),
            98,
        )
    )

    # ========================================================
    # REAL TRADE LEVELS
    # ========================================================

    risk_distance = atr

    reward_distance = atr * 2.0

    if direction == "LONG":

        entry = close_price

        stop_loss = (
            entry - risk_distance
        )

        tp1 = (
            entry + reward_distance
        )

        tp2 = (
            entry + reward_distance * 1.5
        )

    elif direction == "SHORT":

        entry = close_price

        stop_loss = (
            entry + risk_distance
        )

        tp1 = (
            entry - reward_distance
        )

        tp2 = (
            entry - reward_distance * 1.5
        )

    else:

        entry = close_price
        stop_loss = close_price
        tp1 = close_price
        tp2 = close_price

    # ========================================================
    # ACTUAL RISK / REWARD
    # ========================================================

    if direction == "LONG":

        risk = abs(
            entry - stop_loss
        )

        reward = abs(
            tp1 - entry
        )

    elif direction == "SHORT":

        risk = abs(
            stop_loss - entry
        )

        reward = abs(
            entry - tp1
        )

    else:

        risk = 0
        reward = 0

    rr_ratio = (
        reward / risk
        if risk > 0
        else 0
    )

    # ========================================================
    # BEAM / BASE
    # ========================================================

    beam_level = (
        entry + atr * 1.8
        if direction != "SHORT"
        else entry + atr * 1.8
    )

    base_level = (
        entry - atr * 1.8
    )

    # ========================================================
    # TRADE ID
    # ========================================================

    lock_seconds = (
        tf_minutes * 60
    )

    current_seconds = int(
        time.time()
    )

    bucket = (
        current_seconds
        - (
            current_seconds
            % lock_seconds
        )
    )

    trade_id = (
        f"{selected_symbol}_"
        f"{selected_tf}_"
        f"{bucket}_"
        f"{direction}"
    )

    # ========================================================
    # SAVE PAPER TRADE
    # ========================================================

    if (
        paper_mode
        and direction != "NEUTRAL"
    ):

        existing_ids = [
            x.get("trade_id")
            for x in st.session_state.trade_history
        ]

        if trade_id not in existing_ids:

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
                    round(entry, 4),

                "stop_loss":
                    round(stop_loss, 4),

                "tp1":
                    round(tp1, 4),

                "tp2":
                    round(tp2, 4),

                "exit_price":
                    round(entry, 4),

                "confidence":
                    confidence,

                "final_score":
                    round(
                        final_score,
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
            }

            st.session_state.trade_history.insert(
                0,
                new_trade,
            )

            save_history(
                st.session_state.trade_history
            )

    # ========================================================
    # UPDATE OPEN TRADES
    # ========================================================

    for trade in st.session_state.trade_history:

        if (
            trade["outcome"] != "PENDING"
            or trade["symbol"]
            != selected_symbol
        ):
            continue

        current_price = close_price
        trade_entry = float(
            trade["entry_price"]
        )

        trade_sl = float(
            trade["stop_loss"]
        )

        trade_tp = float(
            trade["tp1"]
        )

        if trade["direction"] == "LONG":

            if current_price >= trade_tp:

                trade["outcome"] = "WIN"

                trade["exit_price"] = (
                    current_price
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            current_price
                            - trade_entry
                        )
                        / trade_entry
                    )
                    * 100,
                    2,
                )

                trade["status"] = "Closed"

            elif current_price <= trade_sl:

                trade["outcome"] = "LOSS"

                trade["exit_price"] = (
                    current_price
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            current_price
                            - trade_entry
                        )
                        / trade_entry
                    )
                    * 100,
                    2,
                )

                trade["status"] = "Closed"

        elif trade["direction"] == "SHORT":

            if current_price <= trade_tp:

                trade["outcome"] = "WIN"

                trade["exit_price"] = (
                    current_price
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            trade_entry
                            - current_price
                        )
                        / trade_entry
                    )
                    * 100,
                    2,
                )

                trade["status"] = "Closed"

            elif current_price >= trade_sl:

                trade["outcome"] = "LOSS"

                trade["exit_price"] = (
                    current_price
                )

                trade["pnl_percent"] = round(
                    (
                        (
                            trade_entry
                            - current_price
                        )
                        / trade_entry
                    )
                    * 100,
                    2,
                )

                trade["status"] = "Closed"

    save_history(
        st.session_state.trade_history
    )

    # ========================================================
    # RISK ENGINE
    # ========================================================

    risk_engine = PowerTradingRiskEngine()

    displayed_volume = float(
        np.sum(asks[:, 1])
    )

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
            displayed_vol=displayed_volume,
            cancelled_vol=
                displayed_volume * .10,
            time_exists=15,
            obs_window=60,
            open_interest=150000,
            leverage=20,
            volatility=volatility,
        )
    )

    # ========================================================
    # HEADER STATUS
    # ========================================================

    if direction == "LONG":
        signal_color = "#4ade80"
        signal_class = "long"
    elif direction == "SHORT":
        signal_color = "#fb7185"
        signal_class = "short"
    else:
        signal_color = "#38bdf8"
        signal_class = "neutral"

    st.markdown(
        f"""
<div class="card" style="
    margin-bottom:14px;
    display:flex;
    justify-content:space-between;
    align-items:center;
">

    <div>
        <span style="color:#94a3b8;font-size:12px;">
            {selected_symbol}
        </span>

        <span style="color:#334155;margin:0 10px;">
            /
        </span>

        <span style="color:#94a3b8;font-size:12px;">
            {selected_tf}
        </span>
    </div>

    <div style="
        color:{signal_color};
        font-weight:800;
        font-size:13px;
    ">
        {direction}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        SCORE {final_score:+.3f}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        CONFIDENCE {confidence}%
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    # ========================================================
    # SIGNAL + RR
    # ========================================================

    col1, col2, col3 = st.columns(
        [1.45, 1.45, 1.0],
        gap="medium",
    )

    with col1:

        st.markdown(
            f"""
<div class="signal-card">

    <div class="signal-title">
        Current Signal
    </div>

    <div class="signal-main {signal_class}">
        {direction}
    </div>

    <div class="signal-price">
        Entry
        <b>${entry:,.2f}</b>
    </div>

    <div style="
        margin-top:12px;
        color:#64748b;
        font-size:10px;
    ">
        Confidence
    </div>

    <div style="
        margin-top:5px;
        height:5px;
        background:#172131;
        border-radius:5px;
    ">

        <div style="
            width:{confidence}%;
            height:5px;
            border-radius:5px;
            background:{signal_color};
        "></div>

    </div>

    <div style="
        margin-top:7px;
        color:#94a3b8;
        font-size:10px;
    ">
        {confidence}% model confidence
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
<div class="signal-card">

    <div class="signal-title">
        Trade Levels
    </div>

    <div class="price-grid"
         style="margin-top:12px;">

        <div class="price-box">
            <div class="price-label">
                Entry
            </div>
            <div class="price-value blue">
                ${entry:,.2f}
            </div>
        </div>

        <div class="price-box">
            <div class="price-label">
                Stop Loss
            </div>
            <div class="price-value short">
                ${stop_loss:,.2f}
            </div>
        </div>

        <div class="price-box">
            <div class="price-label">
                TP1
            </div>
            <div class="price-value long">
                ${tp1:,.2f}
            </div>
        </div>

        <div class="price-box">
            <div class="price-label">
                TP2
            </div>
            <div class="price-value long">
                ${tp2:,.2f}
            </div>
        </div>

    </div>

</div>
""",
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
<div class="signal-card">

    <div class="signal-title">
        Risk Management
    </div>

    <div class="rr-box"
         style="margin-top:12px;">

        <div class="rr-label">
            Risk / Reward
        </div>

        <div class="rr-value">
            1 : {rr_ratio:.2f}
        </div>

    </div>

    <div style="
        display:flex;
        justify-content:space-between;
        margin-top:12px;
        font-size:11px;
    ">
        <span style="color:#64748b;">
            Risk
        </span>

        <b>
            ${risk:,.2f}
        </b>
    </div>

    <div style="
        display:flex;
        justify-content:space-between;
        margin-top:8px;
        font-size:11px;
    ">
        <span style="color:#64748b;">
            Reward
        </span>

        <b style="color:#4ade80;">
            ${reward:,.2f}
        </b>
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

    # ========================================================
    # METRICS
    # ========================================================

    st.markdown(
        '<div class="section-title">Market Snapshot</div>',
        unsafe_allow_html=True,
    )

    bid_volume = float(
        np.sum(bids[:20, 1])
    )

    ask_volume = float(
        np.sum(asks[:20, 1])
    )

    obi = (
        (bid_volume - ask_volume)
        / (
            bid_volume
            + ask_volume
            + 1e-12
        )
    )

    spread = abs(
        float(asks[0, 0])
        - float(bids[0, 0])
    )

    spread_pct = (
        spread / close_price * 100
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">
Price
</div>
<div class="metric-value">
${close_price:,.2f}
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">
Top 20 OBI
</div>
<div class="metric-value">
{obi:+.3f}
</div>
<div class="metric-small">
{"Bid pressure" if obi > 0 else "Ask pressure"}
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">
Spread
</div>
<div class="metric-value">
${spread:.2f}
</div>
<div class="metric-small">
{spread_pct:.4f}%
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">
LTZ Score
</div>
<div class="metric-value">
{risk_metrics["LTZ_Score"]:.2f}
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with m5:
        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">
Market Risk
</div>
<div class="metric-value">
{risk_metrics["Market_Risk"]:.2f}
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ========================================================
    # CHART
    # ========================================================

    st.markdown(
        '<div class="section-title">Price Structure</div>',
        unsafe_allow_html=True,
    )

    chart_col, side_col = st.columns(
        [2.65, 1],
        gap="medium",
    )

    with chart_col:

        future_times = [
            df["Time"].iloc[-1]
            + pd.Timedelta(
                minutes=tf_minutes * i
            )
            for i in range(
                1,
                forecast_horizon + 1,
            )
        ]

        t = np.linspace(
            0,
            np.pi / 2,
            forecast_horizon,
        )

        if direction == "LONG":

            forecast = (
                close_price
                + (
                    tp2
                    - close_price
                )
                * np.sin(t)
            )

        elif direction == "SHORT":

            forecast = (
                close_price
                - (
                    close_price
                    - tp2
                )
                * np.sin(t)
            )

        else:

            forecast = np.repeat(
                close_price,
                forecast_horizon,
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
                + future_times,
                y=[
                    close_price
                ]
                + list(forecast),
                mode="lines+markers",
                name="Projection",
                line=dict(
                    color=signal_color,
                    width=2,
                    dash="dot",
                ),
                marker=dict(
                    size=4
                ),
            )
        )

        # Entry
        fig.add_hline(
            y=entry,
            line_color="#38bdf8",
            line_width=1.5,
            annotation_text="ENTRY",
            annotation_position="right",
        )

        # SL
        fig.add_hline(
            y=stop_loss,
            line_color="#fb7185",
            line_dash="dot",
            annotation_text="SL",
            annotation_position="right",
        )

        # TP1
        fig.add_hline(
            y=tp1,
            line_color="#4ade80",
            line_dash="dash",
            annotation_text="TP1",
            annotation_position="right",
        )

        # TP2
        fig.add_hline(
            y=tp2,
            line_color="#22c55e",
            line_dash="dash",
            annotation_text="TP2",
            annotation_position="right",
        )

        # TRI
        if show_tri:

            levels = get_tri_levels(
                selected_symbol
            )

            for tri_tf, tri in levels.items():

                if not tri_enabled.get(
                    tri_tf,
                    True,
                ):
                    continue

                color = TRI_COLORS.get(
                    tri_tf,
                    "#38bdf8",
                )

                for level_name, width, dash in [
                    ("body_50", 2.5, "solid"),
                    ("upper_50", 1, "dot"),
                    ("lower_50", 1, "dot"),
                ]:

                    fig.add_hline(
                        y=float(
                            tri[level_name]
                        ),
                        line_color=color,
                        line_width=width,
                        line_dash=dash,
                        opacity=.65,
                    )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            paper_bgcolor="#0e141f",
            plot_bgcolor="#0e141f",
            margin=dict(
                l=10,
                r=70,
                t=15,
                b=10,
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="#172131",
                rangeslider_visible=False,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#172131",
                zeroline=False,
            ),
            hovermode="x unified",
            dragmode="pan",
            legend=dict(
                orientation="h",
                y=1,
                x=1,
                xanchor="right",
                bgcolor="rgba(0,0,0,0)",
            ),
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
                    "resetScale2d",
                ],
                "modeBarButtonsToRemove": [
                    "lasso2d",
                    "select2d",
                    "toImage",
                ],
            },
        )

    # ========================================================
    # MICROSTRUCTURE
    # ========================================================

    with side_col:

        st.markdown(
            """
<div class="panel-header">
<div class="panel-title">
Order Flow
</div>
<div class="panel-subtitle">
TOP 20
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        top5_b = np.sum(
            bids[:5, 1]
        )

        top5_a = np.sum(
            asks[:5, 1]
        )

        top10_b = np.sum(
            bids[:10, 1]
        )

        top10_a = np.sum(
            asks[:10, 1]
        )

        top20_b = np.sum(
            bids[:20, 1]
        )

        top20_a = np.sum(
            asks[:20, 1]
        )

        def get_obi(b, a):

            return (
                b - a
            ) / (
                b + a + 1e-12
            )

        obi5 = get_obi(
            top5_b,
            top5_a,
        )

        obi10 = get_obi(
            top10_b,
            top10_a,
        )

        obi20 = get_obi(
            top20_b,
            top20_a,
        )

        st.markdown(
            f"""
<div class="card">

<div class="micro-row">
<span class="micro-name">
Top 5 OBI
</span>
<span class="micro-value">
{obi5:+.3f}
</span>
</div>

<div class="micro-row">
<span class="micro-name">
Top 10 OBI
</span>
<span class="micro-value">
{obi10:+.3f}
</span>
</div>

<div class="micro-row">
<span class="micro-name">
Top 20 OBI
</span>
<span class="micro-value">
{obi20:+.3f}
</span>
</div>

<div class="micro-row">
<span class="micro-name">
Bid Volume
</span>
<span class="micro-value">
{top20_b:,.2f}
</span>
</div>

<div class="micro-row">
<span class="micro-name">
Ask Volume
</span>
<span class="micro-value">
{top20_a:,.2f}
</span>
</div>

<div class="micro-row">
<span class="micro-name">
Spread
</span>
<span class="micro-value">
${spread:.2f}
</span>
</div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Risk Monitor</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="card">

<div class="micro-row">
<span class="micro-name">
LTZ
</span>
<span class="micro-value">
{risk_metrics["LTZ_Score"]:.2f}
</span>
</div>

<div class="micro-row">
<span class="micro-name">
Spoof
</span>
<span class="micro-value">
{risk_metrics["Spoof_Score"]:.3f}
</span>
</div>

<div class="micro-row">
<span class="micro-name">
Squeeze
</span>
<span class="micro-value">
{risk_metrics["Squeeze_Risk"]:.2f}
</span>
</div>

<div class="micro-row">
<span class="micro-name">
Market Risk
</span>
<span class="micro-value">
{risk_metrics["Market_Risk"]:.2f}
</span>
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    # ========================================================
    # RESEARCH SCOREBOARD
    # ========================================================

    st.markdown(
        '<div class="section-title">Quantitative Research</div>',
        unsafe_allow_html=True,
    )

    research_col, research_info = st.columns(
        [2, 1],
        gap="medium",
    )

    with research_col:

        research_data = []

        for name, value in paper_results.items():

            if value > .10:
                status = "PASS"
            elif value < -.10:
                status = "FAIL"
            else:
                status = "NEUTRAL"

            research_data.append(
                {
                    "Model": name,
                    "Signal": f"{value:+.3f}",
                    "Weight":
                        f"{weights[name] * 100:.1f}%",
                    "Status": status,
                }
            )

        st.dataframe(
            pd.DataFrame(
                research_data
            ),
            use_container_width=True,
            hide_index=True,
            height=310,
        )

    with research_info:

        st.markdown(
            """
<div class="card">

<div style="
font-size:14px;
font-weight:800;
margin-bottom:12px;
">
Model Summary
</div>

<div style="
font-size:11px;
color:#94a3b8;
line-height:1.8;
">

<b style="color:#e2e8f0;">
HAWKES
</b>
<br>
Order activity clustering.

<br><br>

<b style="color:#e2e8f0;">
BOOK IMB
</b>
<br>
Bid/ask depth pressure.

<br><br>

<b style="color:#e2e8f0;">
TAKER FLOW
</b>
<br>
Directional volume pressure.

<br><br>

<b style="color:#e2e8f0;">
REWARD RISK
</b>
<br>
Distribution-based reward/risk filter.

</div>

</div>
""",
            unsafe_allow_html=True,
        )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section-title">Paper Trading Performance</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.trade_history:

        history_df = pd.DataFrame(
            st.session_state.trade_history
        )

        filter1, filter2, filter3 = st.columns(
            3
        )

        with filter1:

            coin_filter = st.selectbox(
                "Asset Filter",
                ["ALL"] + COINS_LIST,
            )

        with filter2:

            tf_filter = st.selectbox(
                "Timeframe Filter",
                ["ALL"]
                + list(TIMEFRAME_MAP.keys()),
            )

        with filter3:

            direction_filter = st.selectbox(
                "Direction Filter",
                [
                    "ALL",
                    "LONG",
                    "SHORT",
                ],
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

        gross_profit = filtered[
            filtered["outcome"]
            == "WIN"
        ]["pnl_percent"].sum()

        gross_loss = abs(
            filtered[
                filtered["outcome"]
                == "LOSS"
            ]["pnl_percent"].sum()
        )

        net_pnl = (
            gross_profit
            - gross_loss
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0
        )

        p1, p2, p3, p4, p5 = st.columns(5)

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
                "Wins / Losses",
                f"{wins} / {losses}",
            )

        with p4:
            st.metric(
                "Profit Factor",
                f"{profit_factor:.2f}",
            )

        with p5:
            st.metric(
                "Net PnL",
                f"{net_pnl:+.2f}%",
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
            "tp2",
            "exit_price",
            "confidence",
            "pnl_percent",
            "outcome",
            "status",
        ]

        st.dataframe(
            filtered[
                display_columns
            ],
            use_container_width=True,
            hide_index=True,
            height=300,
        )

        if st.sidebar.button(
            "Clear Trade History"
        ):

            st.session_state.trade_history = []

            if os.path.exists(
                CSV_FILE
            ):
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
