# ==========================================
# TRI LINE ANALYSIS ENGINE
# ==========================================

TRI_TIMEFRAMES = {
    "YEARLY": "1M",      # yearly monthly candles se calculate hoga
    "MONTHLY": "1M",
    "WEEKLY": "1w",
    "DAILY": "1d",
    "4H": "4h",
    "1H": "1h",
    "30M": "30m",
    "15M": "15m",
}

TRI_COLORS = {
    "YEARLY": "#87CEEB",     # Sky
    "MONTHLY": "#FF0000",    # Red
    "WEEKLY": "#00C853",     # Green
    "DAILY": "#FFFFFF",      # White on dark dashboard
    "4H": "#FFA500",         # Orange
    "1H": "#A855F7",         # Purple
    "30M": "#006400",        # Dark Green
    "15M": "#2196F3",        # Blue
}


@st.cache_data(ttl=15)
def fetch_tri_candle(symbol, interval):

    try:

        url = "https://data-api.binance.vision/api/v3/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 5,
        }

        response = requests.get(
            url,
            params=params,
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list) or len(data) < 2:
            return None

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

        temp = pd.DataFrame(
            data,
            columns=columns
        )

        for col in [
            "Open",
            "High",
            "Low",
            "Close"
        ]:
            temp[col] = pd.to_numeric(
                temp[col],
                errors="coerce"
            )

        # Previous completed candle
        return temp.iloc[-2]

    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_tri_yearly_candle(symbol):

    try:

        # Binance does not provide a direct "1y"
        # kline interval, therefore fetch monthly candles
        # and construct yearly candles.

        url = "https://data-api.binance.vision/api/v3/klines"

        params = {
            "symbol": symbol,
            "interval": "1M",
            "limit": 36,
        }

        response = requests.get(
            url,
            params=params,
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list) or len(data) < 13:
            return None

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

        temp = pd.DataFrame(
            data,
            columns=columns
        )

        temp["Time"] = pd.to_datetime(
            temp["Open_Time"],
            unit="ms",
            utc=True
        )

        for col in [
            "Open",
            "High",
            "Low",
            "Close"
        ]:
            temp[col] = pd.to_numeric(
                temp[col],
                errors="coerce"
            )

        # Build yearly candles
        yearly = (
            temp
            .set_index("Time")
            .resample("YE")
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

        # Previous completed yearly candle
        previous_year = yearly.iloc[-2]

        return previous_year

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

        # ==========================================
        # BODY
        # ==========================================

        body_high = max(o, c)
        body_low = min(o, c)

        # ==========================================
        # BODY 50%
        # ==========================================

        body_50 = (
            body_high + body_low
        ) / 2.0

        # ==========================================
        # UPPER WICK 50%
        # ==========================================

        upper_50 = (
            h + body_high
        ) / 2.0

        # ==========================================
        # LOWER WICK 50%
        # ==========================================

        lower_50 = (
            l + body_low
        ) / 2.0

        return {
            "body_50": body_50,
            "upper_50": upper_50,
            "lower_50": lower_50,
        }

    except Exception:
        return None


def get_all_tri_levels(symbol):

    levels = {}

    # ==========================================
    # YEARLY
    # ==========================================

    yearly_candle = fetch_tri_yearly_candle(symbol)

    yearly_levels = calculate_tri_levels(
        yearly_candle
    )

    if yearly_levels is not None:

        levels["YEARLY"] = yearly_levels

    # ==========================================
    # MONTHLY / WEEKLY / DAILY / INTRADAY
    # ==========================================

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
        x=[df["Time"].iloc[-1]] + future_times,
        y=[close_p] + list(forecast_prices),
        mode="lines+markers",
        name="Trajectory",
        line=dict(
            color=dir_color,
            width=2,
            dash="dot"
        )
    )
)# ==========================================
# TRI LINE OVERLAY
# ==========================================

tri_levels = get_all_tri_levels(
    selected_symbol
)


# ==========================================
# ADD TRI HORIZONTAL LEVELS
# ==========================================

for tri_tf, tri in tri_levels.items():

    tri_color = TRI_COLORS.get(
        tri_tf,
        "#38bdf8"
    )

    # ------------------------------------------
    # BODY 50%
    # ------------------------------------------

    fig.add_hline(
        y=tri["body_50"],
        line_color=tri_color,
        line_width=3,
        line_dash="solid",
        opacity=0.95,
        annotation_text=(
            f"{tri_tf} • BODY 50%"
        ),
        annotation_position="right",
        annotation_font=dict(
            size=9,
            color=tri_color
        ),
        layer="above"
    )

    # ------------------------------------------
    # UPPER WICK 50%
    # ------------------------------------------

    fig.add_hline(
        y=tri["upper_50"],
        line_color=tri_color,
        line_width=1,
        line_dash="dot",
        opacity=0.75,
        annotation_text=(
            f"{tri_tf} • UPPER 50%"
        ),
        annotation_position="right",
        annotation_font=dict(
            size=8,
            color=tri_color
        ),
        layer="above"
    )

    # ------------------------------------------
    # LOWER WICK 50%
    # ------------------------------------------

    fig.add_hline(
        y=tri["lower_50"],
        line_color=tri_color,
        line_width=1,
        line_dash="dot",
        opacity=0.75,
        annotation_text=(
            f"{tri_tf} • LOWER 50%"
        ),
        annotation_position="right",
        annotation_font=dict(
            size=8,
            color=tri_color
        ),
        layer="above"
    )fig = go.Figure()

# ==========================================
# EXISTING CANDLES
# ==========================================

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


# ==========================================
# EXISTING TRAJECTORY
# ==========================================

fig.add_trace(
    go.Scatter(
        x=[df["Time"].iloc[-1]] + future_times,
        y=[close_p] + list(forecast_prices),
        mode="lines+markers",
        name="Trajectory",
        line=dict(
            color=dir_color,
            width=2,
            dash="dot"
        )
    )
)


# ==========================================
# TRI LINE ANALYSIS
# ==========================================

tri_levels = get_all_tri_levels(
    selected_symbol
)

for tri_tf, tri in tri_levels.items():

    tri_color = TRI_COLORS[tri_tf]

    # BODY 50%
    fig.add_hline(
        y=tri["body_50"],
        line_color=tri_color,
        line_width=3,
        line_dash="solid",
        opacity=0.95,
        annotation_text=f"{tri_tf} • BODY 50%",
        annotation_position="right",
        annotation_font=dict(
            size=9,
            color=tri_color
        ),
        layer="above"
    )

    # UPPER WICK 50%
    fig.add_hline(
        y=tri["upper_50"],
        line_color=tri_color,
        line_width=1,
        line_dash="dot",
        opacity=0.75,
        annotation_text=f"{tri_tf} • UPPER 50%",
        annotation_position="right",
        annotation_font=dict(
            size=8,
            color=tri_color
        ),
        layer="above"
    )

    # LOWER WICK 50%
    fig.add_hline(
        y=tri["lower_50"],
        line_color=tri_color,
        line_width=1,
        line_dash="dot",
        opacity=0.75,
        annotation_text=f"{tri_tf} • LOWER 50%",
        annotation_position="right",
        annotation_font=dict(
            size=8,
            color=tri_color
        ),
        layer="above"
    )


# ==========================================
# EXISTING BEAM / BASE / SL
# ==========================================

fig.add_hline(
    y=beam_level,
    line_dash="dash",
    line_color="#00e676",
    annotation_text=f"BEAM: ${beam_level:,.2f}"
)

fig.add_hline(
    y=base_level,
    line_dash="dash",
    line_color="#ff5252",
    annotation_text=f"BASE: ${base_level:,.2f}"
)

fig.add_hline(
    y=sl_val,
    line_dash="dot",
    line_color="#ff5252",
    annotation_text=f"SL: ${sl_val:,.2f}"
)


# ==========================================
# EXISTING CHART DESIGN
# ==========================================

fig.update_layout(
    template="plotly_dark",
    height=420,
    xaxis_rangeslider_visible=False,
    paper_bgcolor="#111622",
    plot_bgcolor="#111622",
    margin=dict(
        l=10,
        r=150,
        t=10,
        b=10
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)
