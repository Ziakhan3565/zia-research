import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="TRI LINE ANALYSIS",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# BINANCE
# ============================================================

BINANCE_URL = "https://api.binance.com/api/v3/klines"


# ============================================================
# TIMEFRAME CONFIG
# ============================================================

TIMEFRAMES = {
    "Yearly": "1y",
    "Monthly": "1M",
    "Weekly": "1w",
    "Daily": "1d",
    "4H": "4h",
    "1H": "1h",
    "30M": "30m",
    "15M": "15m",
}


# ============================================================
# DEFAULT COLORS
# ============================================================

DEFAULT_COLORS = {
    "Yearly": "#87CEEB",      # Sky
    "Monthly": "#FF0000",     # Red
    "Weekly": "#00A000",      # Green
    "Daily": "#000000",       # Black
    "4H": "#FFA500",          # Orange
    "1H": "#800080",          # Purple
    "30M": "#006400",         # Dark Green
    "15M": "#0000FF",         # Blue
}


# ============================================================
# GET CANDLES
# ============================================================

@st.cache_data(ttl=10)
def get_candles(symbol, interval, limit=500):

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    response = requests.get(
        BINANCE_URL,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "buy_base",
        "buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(data, columns=columns)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms"
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms"
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# ============================================================
# GET PREVIOUS COMPLETED CANDLE
# ============================================================

@st.cache_data(ttl=10)
def get_previous_candle(symbol, interval):

    df = get_candles(
        symbol,
        interval,
        10
    )

    if len(df) < 2:
        return None

    # Last candle can still be forming.
    # [-2] = previous completed candle.
    return df.iloc[-2]


# ============================================================
# TRI LINE CALCULATION
# ============================================================

def calculate_tri_levels(candle):

    if candle is None:
        return None

    o = float(candle["open"])
    h = float(candle["high"])
    l = float(candle["low"])
    c = float(candle["close"])

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    body_high = max(o, c)
    body_low = min(o, c)

    # --------------------------------------------------------
    # BODY 50%
    # --------------------------------------------------------

    body_50 = (
        body_high + body_low
    ) / 2.0

    # --------------------------------------------------------
    # UPPER WICK 50%
    # --------------------------------------------------------

    upper_50 = (
        h + body_high
    ) / 2.0

    # --------------------------------------------------------
    # LOWER WICK 50%
    # --------------------------------------------------------

    lower_50 = (
        l + body_low
    ) / 2.0

    return {
        "body_50": body_50,
        "upper_50": upper_50,
        "lower_50": lower_50,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ TRI LINE SETTINGS")

symbol = st.sidebar.text_input(
    "Symbol",
    value="BTCUSDT"
).upper().strip()

st.sidebar.subheader("Timeframes")

enabled = {}

for timeframe in TIMEFRAMES:

    enabled[timeframe] = st.sidebar.checkbox(
        timeframe,
        value=True,
        key=f"enable_{timeframe}"
    )


# ============================================================
# COLOR SETTINGS
# ============================================================

st.sidebar.subheader("Line Colors")

colors = {}

for timeframe in TIMEFRAMES:

    colors[timeframe] = st.sidebar.color_picker(
        f"{timeframe} Color",
        DEFAULT_COLORS[timeframe],
        key=f"color_{timeframe}"
    )


# ============================================================
# LINE WIDTH
# ============================================================

st.sidebar.subheader("Line Width")

body_width = st.sidebar.slider(
    "Body 50% Width",
    min_value=1,
    max_value=6,
    value=3
)

wick_width = st.sidebar.slider(
    "Wick 50% Width",
    min_value=1,
    max_value=6,
    value=1
)


# ============================================================
# CHART SETTINGS
# ============================================================

st.sidebar.subheader("Chart")

chart_timeframe = st.sidebar.selectbox(
    "Candle Timeframe",
    [
        "15m",
        "30m",
        "1h",
        "4h",
        "1d",
    ],
    index=0
)

candle_limit = st.sidebar.slider(
    "Candles",
    min_value=100,
    max_value=1000,
    value=300,
    step=100
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 TRI LINE ANALYSIS")

st.caption(
    f"{symbol} — Previous Completed Candle 50% Analysis"
)


# ============================================================
# LOAD CHART DATA
# ============================================================

try:

    chart_df = get_candles(
        symbol,
        chart_timeframe,
        candle_limit
    )

except Exception as error:

    st.error(
        f"Market data error: {error}"
    )

    st.stop()


# ============================================================
# CANDLE CHART
# ============================================================

fig = go.Figure()


fig.add_trace(
    go.Candlestick(
        x=chart_df["open_time"],

        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"],

        name=symbol,

        increasing_line_color="#26A69A",
        decreasing_line_color="#EF5350",

        increasing_fillcolor="#26A69A",
        decreasing_fillcolor="#EF5350",
    )
)


# ============================================================
# TRI LINES
# ============================================================

tri_results = {}


for timeframe, interval in TIMEFRAMES.items():

    if not enabled[timeframe]:
        continue

    try:

        candle = get_previous_candle(
            symbol,
            interval
        )

        levels = calculate_tri_levels(
            candle
        )

        if levels is None:
            continue

        tri_results[timeframe] = levels

        # ----------------------------------------------------
        # BODY 50%
        # ----------------------------------------------------

        fig.add_hline(
            y=levels["body_50"],

            line_color=colors[timeframe],
            line_width=body_width,

            line_dash="solid",

            annotation_text=(
                f"{timeframe} Body 50%"
            ),

            annotation_position="right",

            annotation_font_size=10,

            layer="above",
        )

        # ----------------------------------------------------
        # UPPER WICK 50%
        # ----------------------------------------------------

        fig.add_hline(
            y=levels["upper_50"],

            line_color=colors[timeframe],
            line_width=wick_width,

            line_dash="dot",

            annotation_text=(
                f"{timeframe} Upper 50%"
            ),

            annotation_position="right",

            annotation_font_size=8,

            layer="above",
        )

        # ----------------------------------------------------
        # LOWER WICK 50%
        # ----------------------------------------------------

        fig.add_hline(
            y=levels["lower_50"],

            line_color=colors[timeframe],
            line_width=wick_width,

            line_dash="dot",

            annotation_text=(
                f"{timeframe} Lower 50%"
            ),

            annotation_position="right",

            annotation_font_size=8,

            layer="above",
        )

    except Exception as error:

        st.warning(
            f"{timeframe}: {error}"
        )


# ============================================================
# CURRENT PRICE
# ============================================================

current_price = float(
    chart_df["close"].iloc[-1]
)

fig.add_hline(
    y=current_price,

    line_color="#FFD700",
    line_width=2,

    line_dash="dash",

    annotation_text=(
        f"PRICE {current_price:,.2f}"
    ),

    annotation_position="right",

    annotation_font_size=11,
)


# ============================================================
# CHART DESIGN
# ============================================================

fig.update_layout(

    height=750,

    template="plotly_dark",

    xaxis_rangeslider_visible=False,

    hovermode="x unified",

    margin=dict(
        l=20,
        r=180,
        t=50,
        b=20,
    ),

    title={
        "text": (
            f"{symbol} | "
            f"{chart_timeframe.upper()} | "
            "TRI LINE"
        ),
        "x": 0.5,
    },

    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
    ),

    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        fixedrange=False,
    ),

)


# ============================================================
# DISPLAY CHART
# ============================================================

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
    }
)


# ============================================================
# LEVEL TABLE
# ============================================================

st.subheader("📐 TRI LINE Levels")


table_data = []


for timeframe, levels in tri_results.items():

    table_data.append({

        "Timeframe": timeframe,

        "Body 50%": round(
            levels["body_50"],
            4
        ),

        "Upper 50%": round(
            levels["upper_50"],
            4
        ),

        "Lower 50%": round(
            levels["lower_50"],
            4
        ),

        "Color": colors[timeframe],
    })


if table_data:

    table_df = pd.DataFrame(
        table_data
    )

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CURRENT PRICE INFO
# ============================================================

st.metric(
    "Current Price",
    f"{current_price:,.4f}"
)
