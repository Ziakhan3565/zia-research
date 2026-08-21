from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import requests
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="TRI Line Engine Analysis",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================
BINANCE_API = "https://api.binance.com/api/v3/klines"
DEFAULT_SYMBOL = "BTCUSDT"

TIMEFRAME_MAPPING = {
    "YEARLY": "1y",
    "MONTHLY": "1M",
    "WEEKLY": "1w",
    "DAILY": "1d",
    "4H": "4h",
    "1H": "1h",
    "30M": "30m",
    "15M": "15m",
    "5M": "5m",
    "1M": "1m",
}

# ============================================================
# DATA CLASS
# ============================================================
@dataclass
class TRILineLevels:
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    body_high: float
    body_low: float
    body_50: float
    upper_50: float
    lower_50: float
    candle_time: int

# ============================================================
# TRI LINE ENGINE
# ============================================================
class TRILineEngine:
    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        enabled_timeframes: Optional[Dict[str, bool]] = None,
        timeout: int = 10,
    ):
        self.symbol = symbol.upper()
        self.timeout = timeout

        self.enabled = {tf: True for tf in TIMEFRAME_MAPPING}
        if enabled_timeframes:
            self.enabled.update(enabled_timeframes)

        self.colors = {
            "YEARLY": "sky",
            "MONTHLY": "red",
            "WEEKLY": "green",
            "DAILY": "black",
            "4H": "orange",
            "1H": "purple",
            "30M": "darkgreen",
            "15M": "blue",
            "5M": "magenta",
            "1M": "cyan",
        }

    def set_symbol(self, symbol: str):
        self.symbol = symbol.upper()

    def get_klines(self, interval: str, limit: int = 5):
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit,
        }
        response = requests.get(
            BINANCE_API,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_previous_candle(self, interval: str):
        candles = self.get_klines(interval=interval, limit=5)
        if len(candles) < 2:
            raise RuntimeError(f"Not enough candle data for {interval}")
        
        candle = candles[-2]
        return {
            "time": int(candle[0]),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        }

    def calculate_levels(self, timeframe: str) -> TRILineLevels:
        timeframe = timeframe.upper()
        interval = TIMEFRAME_MAPPING[timeframe]
        candle = self.get_previous_candle(interval)

        o = candle["open"]
        h = candle["high"]
        l = candle["low"]
        c = candle["close"]

        body_high = max(o, c)
        body_low = min(o, c)
        body_50 = (body_high + body_low) / 2.0
        upper_50 = (h + body_high) / 2.0
        lower_50 = (l + body_low) / 2.0

        return TRILineLevels(
            timeframe=timeframe,
            open=o,
            high=h,
            low=l,
            close=c,
            body_high=body_high,
            body_low=body_low,
            body_50=body_50,
            upper_50=upper_50,
            lower_50=lower_50,
            candle_time=candle["time"],
        )

    def calculate_all(self):
        results = {}
        for timeframe in TIMEFRAME_MAPPING:
            if not self.enabled.get(timeframe, False):
                continue
            try:
                levels = self.calculate_levels(timeframe)
                results[timeframe] = levels
            except Exception as error:
                results[timeframe] = {"error": str(error)}
        return results

# ============================================================
# STREAMLIT UI APP
# ============================================================
st.title("📊 TRI Line Market Analysis Hub")
st.markdown("Analyze multi-timeframe key levels (**1m to Yearly**) seamlessly.")

# Sidebar controls
st.sidebar.header("Configuration")
selected_symbol = st.sidebar.text_input("Trading Symbol", value="BTCUSDT").upper()

engine = TRILineEngine(symbol=selected_symbol)

st.sidebar.subheader("Select Timeframes")
enabled_tfs = {}
for tf in TIMEFRAME_MAPPING.keys():
    enabled_tfs[tf] = st.sidebar.checkbox(tf, value=True)

engine.enabled = enabled_tfs

if st.sidebar.button("Fetch & Calculate Levels", type="primary"):
    with st.spinner("Fetching data from Binance..."):
        data = engine.calculate_all()
        
    st.success(f"Successfully calculated levels for {selected_symbol}!")
    
    for tf, val in data.items():
        with st.expander(f"Timeframe: [{tf}]", expanded=True):
            if isinstance(val, TRILineLevels):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Open", f"{val.open:.4f}")
                col2.metric("High", f"{val.high:.4f}")
                col3.metric("Low", f"{val.low:.4f}")
                col4.metric("Close", f"{val.close:.4f}")
                
                st.markdown("---")
                
                s1, s2, s3 = st.columns(3)
                s1.metric("Body 50%", f"{val.body_50:.4f}")
                s2.metric("Upper Wick 50%", f"{val.upper_50:.4f}")
                s3.metric("Lower Wick 50%", f"{val.lower_50:.4f}")
            else:
                st.error(f"Error fetching data: {val.get('error')}")
else:
    st.info("👈 Click 'Fetch & Calculate Levels' in the sidebar to load the market data.")
