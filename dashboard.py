# ============================================================
# TRI QUANT TRADING DASHBOARD
# Complete Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

from datetime import datetime, timezone


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TRI Quant Trading Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #080b11;
        color: #ffffff;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1600px;
    }

    h1, h2, h3, h4 {
        color: #f8fafc;
    }

    .metric-card {
        background: #111622;
        border: 1px solid #202938;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .metric-title {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 5px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 20px;
        font-weight: 700;
    }

    .ob-card {
        background: #111622;
        border: 1px solid #202938;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .ob-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 9px;
        font-size: 14px;
    }

    .ob-label {
        color: #cbd5e1;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .tri-box {
        background: #111622;
        border: 1px solid #202938;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BINANCE API
# ============================================================

BINANCE_KLINES_URL = (
    "https://data-api.binance.vision/api/v3/klines"
)

BINANCE_BOOK_URL = (
    "https://data-api.binance.vision/api/v3/depth"
)


# ============================================================
# SYMBOL MAP
# ============================================================

SYMBOLS = {
    "BTCUSDT": "BTCUSDT",
    "ETHUSDT": "ETHUSDT",
    "BNBUSDT": "BNBUSDT",
    "SOLUSDT": "SOLUSDT",
    "XRPUSDT": "XRPUSDT",
    "DOGEUSDT": "DOGEUSDT",
    "ADAUSDT": "ADAUSDT",
    "AVAXUSDT": "AVAXUSDT",
}


# ============================================================
# TRI TIMEFRAMES
# ============================================================

TRI_TIMEFRAMES = {
    "YEARLY": "YEARLY",
    "MONTHLY": "1M",
    "WEEKLY": "1w",
    "DAILY": "1d",
    "4H": "4h",
    "1H": "1h",
    "30M": "30m",
    "15M": "15m",
}


# ============================================================
# TRI COLORS
# ============================================================

TRI_COLORS = {

    "YEARLY": "#87CEEB",     # Sky blue

    "MONTHLY": "#FF0000",    # Red

    "WEEKLY": "#00C853",     # Green

    "DAILY": "#FFFFFF",      # White

    "4H": "#FFA500",         # Orange

    "1H": "#A855F7",         # Purple

    "30M": "#006400",        # Dark green

    "15M": "#2196F3",        # Blue
}


# ============================================================
# FETCH KLINES
# ============================================================

@st.cache_data(ttl=15)
def fetch_klines(symbol, interval, limit=200):

    try:

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        response = requests.get(
            BINANCE_KLINES_URL,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            return None

        if len(data) < 10:
            return None

        columns = [
            "Open_Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close_Time",
            "Quote_Volume",
            "Trades",
            "Taker_Buy_Base",
            "Taker_Buy_Quote",
            "Ignore",
        ]

        df = pd.DataFrame(
            data,
            columns=columns
        )

        df["Time"] = pd.to_datetime(
            df["Open_Time"],
            unit="ms",
            utc=True
        )

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        for col in numeric_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        return df

    except Exception as e:

        st.warning(
            f"Market data error: {e}"
        )

        return None


# ============================================================
# FETCH PREVIOUS COMPLETED CANDLE
# ============================================================

@st.cache_data(ttl=30)
def fetch_tri_candle(symbol, interval):

    try:

        df = fetch_klines(
            symbol,
            interval,
            5
        )

        if df is None:
            return None

        if len(df) < 2:
            return None

        # Previous completed candle
        return df.iloc[-2]

    except Exception:

        return None


# ============================================================
# YEARLY CANDLE
# ============================================================

@st.cache_data(ttl=60)
def fetch_tri_yearly_candle(symbol):

    try:

        monthly = fetch_klines(
            symbol,
            "1M",
            60
        )

        if monthly is None:
            return None

        if len(monthly) < 13:
            return None

        monthly = monthly.copy()

        monthly["Year"] = (
            monthly["Time"]
            .dt.year
        )

        yearly = (
            monthly
            .groupby("Year")
            .agg(
                Open=("Open", "first"),
                High=("High", "max"),
                Low=("Low", "min"),
                Close=("Close", "last"),
            )
            .dropna()
        )

        if len(yearly) < 2:
            return None

        # Previous completed year
        previous_year = yearly.iloc[-2]

        return previous_year

    except Exception:

        return None


# ============================================================
# CALCULATE TRI LEVELS
# ============================================================

def calculate_tri_levels(candle):

    if candle is None:
        return None

    try:

        o = float(candle["Open"])
        h = float(candle["High"])
        l = float(candle["Low"])
        c = float(candle["Close"])

        # ----------------------------------------------------
        # BODY
        # ----------------------------------------------------

        body_high = max(
            o,
            c
        )

        body_low = min(
            o,
            c
        )

        # ----------------------------------------------------
        # BODY 50%
        # ----------------------------------------------------

        body_50 = (
            body_high +
            body_low
        ) / 2.0

        # ----------------------------------------------------
        # UPPER WICK 50%
        # ----------------------------------------------------

        upper_50 = (
            h +
            body_high
        ) / 2.0

        # ----------------------------------------------------
        # LOWER WICK 50%
        # ----------------------------------------------------

        lower_50 = (
            l +
            body_low
        ) / 2.0

        return {

            "body_50": body_50,

            "upper_50": upper_50,

            "lower_50": lower_50,

        }

    except Exception:

        return None


# ============================================================
# GET ALL TRI LEVELS
# ============================================================

def get_all_tri_levels(symbol):

    levels = {}

    # --------------------------------------------------------
    # YEARLY
    # --------------------------------------------------------

    yearly_candle = fetch_tri_yearly_candle(
        symbol
    )

    yearly_levels = calculate_tri_levels(
        yearly_candle
    )

    if yearly_levels is not None:

        levels["YEARLY"] = yearly_levels

    # --------------------------------------------------------
    # OTHER TIMEFRAMES
    # --------------------------------------------------------

    for name, interval in TRI_TIMEFRAMES.items():

        if name == "YEARLY":
            continue

        candle = fetch_tri_candle(
            symbol,
            interval
        )

        tri = calculate_tri_levels(
            candle
        )

        if tri is not None:

            levels[name] = tri

    return levels


# ============================================================
# ORDER BOOK
# ============================================================

@st.cache_data(ttl=5)
def fetch_orderbook(symbol, limit=50):

    try:

        params = {
            "symbol": symbol,
            "limit": limit,
        }

        response = requests.get(
            BINANCE_BOOK_URL,
            params=params,
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        bids = data.get(
            "bids",
            []
        )

        asks = data.get(
            "asks",
            []
        )

        if not bids or not asks:
            return None

        bids = [
            (
                float(price),
                float(volume)
            )
            for price, volume in bids
        ]

        asks = [
            (
                float(price),
                float(volume)
            )
            for price, volume in asks
        ]

        bid_volume = sum(
            volume
            for _, volume in bids
        )

        ask_volume = sum(
            volume
            for _, volume in asks
        )

        total_volume = (
            bid_volume +
            ask_volume
        )

        if total_volume > 0:

            obi = (
                bid_volume -
                ask_volume
            ) / total_volume

        else:

            obi = 0.0

        best_bid = bids[0][0]

        best_ask = asks[0][0]

        mid_price = (
            best_bid +
            best_ask
        ) / 2.0

        spread = (
            best_ask -
            best_bid
        )

        spread_pct = (
            spread /
            mid_price *
            100
        )

        return {

            "bid_volume": bid_volume,

            "ask_volume": ask_volume,

            "obi": obi,

            "best_bid": best_bid,

            "best_ask": best_ask,

            "mid_price": mid_price,

            "spread": spread,

            "spread_pct": spread_pct,

        }

    except Exception:

        return None


# ============================================================
# TRAJECTORY
# ============================================================

def create_trajectory(df, horizon):

    if df is None:
        return [], []

    if len(df) < 20:
        return [], []

    close = float(
        df["Close"].iloc[-1]
    )

    recent = df[
        "Close"
    ].tail(20)

    returns = recent.pct_change().dropna()

    if len(returns) == 0:
        volatility = 0.001
    else:
        volatility = float(
            returns.std()
        )

    if not np.isfinite(volatility):
        volatility = 0.001

    volatility = max(
        volatility,
        0.0005
    )

    # Simple directional trajectory
    start = float(
        recent.iloc[0]
    )

    end = float(
        recent.iloc[-1]
    )

    if start == 0:
        direction = 0
    else:
        direction = (
            end -
            start
        ) / start

    if direction >= 0:
        drift = 1
    else:
        drift = -1

    future_prices = []

    price = close

    for i in range(
        1,
        horizon + 1
    ):

        step = (
            volatility *
            0.35 *
            drift
        )

        price = (
            price *
            (1 + step)
        )

        future_prices.append(
            price
        )

    last_time = df[
        "Time"
    ].iloc[-1]

    if len(df) >= 2:

        candle_delta = (
            df["Time"].iloc[-1] -
            df["Time"].iloc[-2]
        )

    else:

        candle_delta = pd.Timedelta(
            minutes=15
        )

    future_times = []

    for i in range(
        1,
        horizon + 1
    ):

        future_times.append(
            last_time +
            candle_delta * i
        )

    return (
        future_times,
        future_prices
    )


# ============================================================
# MAIN SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ⚡ Terminal Controls"
)

selected_symbol = st.sidebar.selectbox(
    "Select Cryptocurrency",
    list(SYMBOLS.keys()),
    index=0
)

st.sidebar.markdown(
    "### Select Timeframe"
)

timeframe_options = {
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h (Intraday)": "4h",
    "1d": "1d",
}

selected_tf_label = st.sidebar.selectbox(
    "",
    list(timeframe_options.keys()),
    index=3
)

selected_interval = (
    timeframe_options[
        selected_tf_label
    ]
)

st.sidebar.markdown(
    "### Forecast Horizon Candles"
)

forecast_horizon = st.sidebar.slider(
    "",
    min_value=5,
    max_value=30,
    value=15
)


# ============================================================
# LOAD MARKET DATA
# ============================================================

df = fetch_klines(
    selected_symbol,
    selected_interval,
    200
)


# ============================================================
# ERROR CHECK
# ============================================================

if df is None:

    st.error(
        "Unable to load market data."
    )

    st.stop()


# ============================================================
# CURRENT PRICE
# ============================================================

close_price = float(
    df["Close"].iloc[-1]
)

previous_close = float(
    df["Close"].iloc[-2]
)

price_change = (
    close_price -
    previous_close
)

price_change_pct = (
    price_change /
    previous_close *
    100
)


# ============================================================
# TOP HEADER
# ============================================================

st.markdown(
    f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom:15px;
    ">

        <div>

            <h1 style="
                margin:0;
                font-size:30px;
            ">
                TRI Quant Trading Terminal
            </h1>

            <div style="
                color:#94a3b8;
                margin-top:4px;
            ">
                {selected_symbol} • {selected_interval}
            </div>

        </div>

        <div style="
            text-align:right;
        ">

            <div style="
                font-size:28px;
                font-weight:700;
            ">
                ${close_price:,.2f}
            </div>

            <div style="
                color:{'#00e676' if price_change >= 0 else '#ff5252'};
                font-weight:600;
            ">
                {price_change_pct:+.2f}%
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN COLUMNS
# ============================================================

left_col, right_col = st.columns(
    [3.3, 1.2]
)


# ============================================================
# LEFT: CHART
# ============================================================

with left_col:

    st.markdown(
        f"""
        <div class="section-title">
            Price Trajectory & Levels ({selected_symbol})
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TRAJECTORY
    # --------------------------------------------------------

    future_times, forecast_prices = (
        create_trajectory(
            df,
            forecast_horizon
        )
    )

    if forecast_prices:

        if forecast_prices[-1] >= close_price:
            dir_color = "#00e676"
        else:
            dir_color = "#ff5252"

    else:

        dir_color = "#38bdf8"


    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    fig = go.Figure()


    # --------------------------------------------------------
    # CANDLES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # TRAJECTORY
    # --------------------------------------------------------

    if future_prices:

        fig.add_trace(
            go.Scatter(
                x=[
                    df["Time"].iloc[-1]
                ] + future_times,

                y=[
                    close_price
                ] + list(
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


    # ========================================================
    # TRI LINE ANALYSIS
    # ========================================================

    tri_levels = get_all_tri_levels(
        selected_symbol
    )


    # --------------------------------------------------------
    # ADD TRI HORIZONTAL LEVELS
    # --------------------------------------------------------

    for tri_tf, tri in tri_levels.items():

        tri_color = TRI_COLORS.get(
            tri_tf,
            "#38bdf8"
        )


        # ----------------------------------------------------
        # BODY 50%
        # ----------------------------------------------------

        fig.add_hline(
            y=tri["body_50"],

            line_color=tri_color,

            line_width=3,

            line_dash="solid",

            opacity=0.95,

            layer="above"
        )


        # ----------------------------------------------------
        # UPPER WICK 50%
        # ----------------------------------------------------

        fig.add_hline(
            y=tri["upper_50"],

            line_color=tri_color,

            line_width=1,

            line_dash="dot",

            opacity=0.70,

            layer="above"
        )


        # ----------------------------------------------------
        # LOWER WICK 50%
        # ----------------------------------------------------

        fig.add_hline(
            y=tri["lower_50"],

            line_color=tri_color,

            line_width=1,

            line_dash="dot",

            opacity=0.70,

            layer="above"
        )


    # ========================================================
    # BEAM / BASE / SL
    # ========================================================

    # Simple levels based on recent price range
    recent_high = float(
        df["High"].tail(50).max()
    )

    recent_low = float(
        df["Low"].tail(50).min()
    )

    beam_level = recent_high

    base_level = recent_low

    risk_range = (
        recent_high -
        recent_low
    )

    sl_val = close_price - (
        risk_range * 0.10
    )


    # --------------------------------------------------------
    # BEAM
    # --------------------------------------------------------

    fig.add_hline(
        y=beam_level,

        line_dash="dash",

        line_color="#00e676",

        opacity=0.8,

        annotation_text=(
            f"BEAM: ${beam_level:,.2f}"
        ),

        annotation_position="right",

        annotation_font=dict(
            size=9,
            color="#00e676"
        )
    )


    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    fig.add_hline(
        y=base_level,

        line_dash="dash",

        line_color="#ff5252",

        opacity=0.8,

        annotation_text=(
            f"BASE: ${base_level:,.2f}"
        ),

        annotation_position="right",

        annotation_font=dict(
            size=9,
            color="#ff5252"
        )
    )


    # --------------------------------------------------------
    # STOP LOSS
    # --------------------------------------------------------

    fig.add_hline(
        y=sl_val,

        line_dash="dot",

        line_color="#ff5252",

        opacity=0.65,

        annotation_text=(
            f"SL: ${sl_val:,.2f}"
        ),

        annotation_position="right",

        annotation_font=dict(
            size=9,
            color="#ff5252"
        )
    )


    # ========================================================
    # TRI LEGEND
    # ========================================================

    tri_legend_parts = []

    for tri_tf in tri_levels.keys():

        tri_color = TRI_COLORS.get(
            tri_tf,
            "#38bdf8"
        )

        tri_legend_parts.append(
            f"""
            <span style="
                color:{tri_color};
                font-weight:700;
                margin-right:10px;
            ">
                {tri_tf}
            </span>
            """
        )


    tri_legend_html = "".join(
        tri_legend_parts
    )


    # --------------------------------------------------------
    # LEGEND ABOVE CHART
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div style="
            background:#111622;
            border:1px solid #202938;
            border-radius:8px;
            padding:8px 12px;
            margin-bottom:8px;
            font-size:11px;
            white-space:nowrap;
            overflow-x:auto;
        ">

            <span style="
                color:#94a3b8;
                margin-right:8px;
            ">
                TRI:
            </span>

            {tri_legend_html}

            <span style="
                color:#94a3b8;
                margin-left:8px;
            ">
                ━ Body 50%
            </span>

            <span style="
                color:#64748b;
                margin-left:8px;
            ">
                ··· Wick 50%
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # CHART DESIGN
    # ========================================================

    fig.update_layout(

        template="plotly_dark",

        height=560,

        xaxis_rangeslider_visible=False,

        paper_bgcolor="#111622",

        plot_bgcolor="#111622",

        margin=dict(
            l=10,
            r=95,
            t=20,
            b=10
        ),

        hovermode="x unified",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1
        ),

        xaxis=dict(
            showgrid=False
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="#202938",
            fixedrange=False
        )

    )


    # ========================================================
    # DISPLAY CHART
    # ========================================================

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "displayModeBar": True
        }
    )


# ============================================================
# RIGHT SIDE
# ============================================================

with right_col:

    st.markdown(
        """
        <div class="section-title">
            Market Microstructure & OB
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # ORDER BOOK
    # ========================================================

    orderbook = fetch_orderbook(
        selected_symbol,
        50
    )


    if orderbook is not None:

        bid_volume = orderbook[
            "bid_volume"
        ]

        ask_volume = orderbook[
            "ask_volume"
        ]

        obi = orderbook[
            "obi"
        ]

        best_bid = orderbook[
            "best_bid"
        ]

        best_ask = orderbook[
            "best_ask"
        ]

        spread = orderbook[
            "spread"
        ]

        spread_pct = orderbook[
            "spread_pct"
        ]


        # ----------------------------------------------------
        # OBI STATUS
        # ----------------------------------------------------

        if obi >= 0.15:

            obi_status = "BUY PRESSURE"

            obi_color = "#00e676"

        elif obi <= -0.15:

            obi_status = "SELL PRESSURE"

            obi_color = "#ff5252"

        else:

            obi_status = "BALANCED"

            obi_color = "#facc15"


        # ====================================================
        # ORDER BOOK CARD
        # ====================================================

        st.markdown(
            f"""
            <div class="ob-card">

                <div class="ob-row">

                    <span class="ob-label">
                        Bid Volume
                    </span>

                    <b style="
                        color:#00e676;
                    ">
                        {bid_volume:,.2f}
                    </b>

                </div>


                <div class="ob-row">

                    <span class="ob-label">
                        Ask Volume
                    </span>

                    <b style="
                        color:#ff5252;
                    ">
                        {ask_volume:,.2f}
                    </b>

                </div>


                <div class="ob-row">

                    <span class="ob-label">
                        OBI
                    </span>

                    <b style="
                        color:{obi_color};
                    ">
                        {obi:+.4f}
                    </b>

                </div>


                <div class="ob-row">

                    <span class="ob-label">
                        Status
                    </span>

                    <b style="
                        color:{obi_color};
                    ">
                        {obi_status}
                    </b>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # PRICE / SPREAD CARD
        # ====================================================

        st.markdown(
            f"""
            <div class="ob-card">

                <div class="ob-row">

                    <span class="ob-label">
                        Best Bid
                    </span>

                    <b style="
                        color:#00e676;
                    ">
                        ${best_bid:,.2f}
                    </b>

                </div>


                <div class="ob-row">

                    <span class="ob-label">
                        Best Ask
                    </span>

                    <b style="
                        color:#ff5252;
                    ">
                        ${best_ask:,.2f}
                    </b>

                </div>


                <div class="ob-row">

                    <span class="ob-label">
                        Spread
                    </span>

                    <b>
                        ${spread:,.4f}
                    </b>

                </div>


                <div class="ob-row">

                    <span class="ob-label">
                        Spread %
                    </span>

                    <b>
                        {spread_pct:.4f}%
                    </b>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.warning(
            "Order book data unavailable."
        )


    # ========================================================
    # TRI LEVEL TABLE
    # ========================================================

    st.markdown(
        """
        <div class="section-title"
             style="font-size:20px;margin-top:15px;">
            TRI Levels
        </div>
        """,
        unsafe_allow_html=True
    )


    if tri_levels:

        for tri_tf, tri in tri_levels.items():

            tri_color = TRI_COLORS.get(
                tri_tf,
                "#38bdf8"
            )

            st.markdown(
                f"""
                <div class="tri-box">

                    <div style="
                        color:{tri_color};
                        font-weight:700;
                        margin-bottom:8px;
                    ">
                        {tri_tf}
                    </div>

                    <div class="ob-row">

                        <span class="ob-label">
                            Body 50%
                        </span>

                        <b>
                            ${tri["body_50"]:,.2f}
                        </b>

                    </div>

                    <div class="ob-row">

                        <span class="ob-label">
                            Upper 50%
                        </span>

                        <b>
                            ${tri["upper_50"]:,.2f}
                        </b>

                    </div>

                    <div class="ob-row">

                        <span class="ob-label">
                            Lower 50%
                        </span>

                        <b>
                            ${tri["lower_50"]:,.2f}
                        </b>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        font-size:11px;
        padding:20px 0 5px 0;
    ">
        TRI Quant Trading Terminal • Market data via Binance
    </div>
    """,
    unsafe_allow_html=True
)
