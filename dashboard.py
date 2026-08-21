import streamlit as st
import plotly.graph_objects as go
import requests
from tri_line_engine import TRILineEngine

st.set_page_config(page_title="TRI Line Research Dashboard", layout="wide")

st.title("📊 Price Trajectory & Tri-Line Levels Dashboard")

# Sidebar Controls
selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], index=0)
limit = st.sidebar.slider("Candle Limit", 50, 500, 100)

# Wrap engine calculation in try-except to prevent app crash/black screen
try:
    with st.spinner("Calculating Tri-Line Levels..."):
        engine = TRILineEngine(symbol=selected_crypto, timeout=5)
        tri_data = engine.calculate_all_dict()
except Exception as e:
    st.error(f"Engine Error: {e}")
    tri_data = {}

# Fetch historical candles safely
@st.cache_data(ttl=60)
def fetch_klines(symbol, limit):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "1d", "limit": limit}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

raw_candles = fetch_klines(selected_crypto, limit)

if not raw_candles:
    st.warning("⚠️ Failed to fetch chart data from Binance API. Please check your internet connection or try again.")
else:
    # Parse candlestick data safely
    import pandas as pd
    df = pd.DataFrame(raw_candles, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)

    # Create Plotly Candlestick Figure
    fig = go.Figure(data=[go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candles"
    )])

    # ============================================================
    # ADD TRI-LINE LEVELS TO THE PLOTLY CHART
    # ============================================================
    for timeframe, data in tri_data.items():
        if not isinstance(data, dict) or "error" in data:
            continue
        
        color = data.get("color", "gray")
        
        # Body 50% Line
        if "body_50" in data:
            fig.add_hline(
                y=data["body_50"],
                line_dash="dash",
                line_color=color,
                annotation_text=f"{timeframe} Body 50%",
                annotation_position="bottom right"
            )
        
        # Upper Wick 50% Line
        if "upper_50" in data:
            fig.add_hline(
                y=data["upper_50"],
                line_dash="dot",
                line_color=color,
                annotation_text=f"{timeframe} Upper 50%"
            )
        
        # Lower Wick 50% Line
        if "lower_50" in data:
            fig.add_hline(
                y=data["lower_50"],
                line_dash="dot",
                line_color=color,
                annotation_text=f"{timeframe} Lower 50%"
            )

    fig.update_layout(
        title=f"Price Trajectory & Multi-Timeframe Tri-Lines ({selected_crypto})",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Debug / Data view in expander
with st.expander("🔍 View Calculated Tri-Line Values"):
    st.json(tri_data)
