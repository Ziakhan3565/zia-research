from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
import requests


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

        # ----------------------------------------------------
        # DEFAULT TIMEFRAME SETTINGS
        # ----------------------------------------------------

        self.enabled = {tf: True for tf in TIMEFRAME_MAPPING}

        # User settings override
        if enabled_timeframes:
            self.enabled.update(enabled_timeframes)

        # ----------------------------------------------------
        # COLORS
        # ----------------------------------------------------

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

    # ========================================================
    # CHANGE SYMBOL
    # ========================================================

    def set_symbol(self, symbol: str):
        self.symbol = symbol.upper()

    # ========================================================
    # ENABLE / DISABLE TIMEFRAME
    # ========================================================

    def set_timeframe(self, timeframe: str, enabled: bool):
        timeframe = timeframe.upper()
        if timeframe not in TIMEFRAME_MAPPING:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self.enabled[timeframe] = enabled

    # ========================================================
    # CHANGE COLOR
    # ========================================================

    def set_color(self, timeframe: str, color: str):
        timeframe = timeframe.upper()
        if timeframe not in TIMEFRAME_MAPPING:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self.colors[timeframe] = color

    # ========================================================
    # GET BINANCE CANDLES (Supports historical depth up to limit)
    # ========================================================

    def get_klines(
        self,
        interval: str,
        limit: int = 100,  # 3 months tak ka data ya zyada candles ke liye limit barha sakte hain
    ):

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

    # ========================================================
    # GET HISTORICAL CANDLES LIST (e.g., last 3 months candles)
    # ========================================================

    def get_historical_candles(
        self,
        interval: str,
        limit: int = 100
    ) -> List[dict]:
        
        candles = self.get_klines(interval=interval, limit=limit)
        
        formatted_candles = []
        for candle in candles:
            formatted_candles.append({
                "time": int(candle[0]),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            })
            
        return formatted_candles

    # ========================================================
    # CALCULATE TRI LEVELS FOR A SPECIFIC CANDLE INDEX
    # ========================================================

    def calculate_levels_from_candle(self, timeframe: str, candle: dict) -> TRILineLevels:
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

    # ========================================================
    # CALCULATE LATEST LEVELS
    # ========================================================

    def calculate_levels(
        self,
        timeframe: str,
    ) -> TRILineLevels:

        timeframe = timeframe.upper()
        interval = TIMEFRAME_MAPPING[timeframe]

        candles = self.get_historical_candles(interval, limit=5)
        if len(candles) < 2:
            raise RuntimeError(f"Not enough candle data for {interval}")

        # [-2] = previous completed candle
        return self.calculate_levels_from_candle(timeframe, candles[-2])

    # ========================================================
    # CALCULATE ALL ENABLED TIMEFRAMES
    # ========================================================

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

    # ========================================================
    # PRINT LEVELS
    # ========================================================

    def print_levels(self):

        results = self.calculate_all()

        print()
        print("=" * 75)
        print(f"TRI LINE ANALYSIS | {self.symbol}")
        print("=" * 75)

        for timeframe, value in results.items():

            print()
            print(f"[{timeframe}]")

            if isinstance(value, TRILineLevels):
                print(f"Color       : {self.colors[timeframe]}")
                print(f"Open        : {value.open:.8f}")
                print(f"High        : {value.high:.8f}")
                print(f"Low         : {value.low:.8f}")
                print(f"Close       : {value.close:.8f}")
                print(f"Body 50%    : {value.body_50:.8f}")
                print(f"Upper 50%   : {value.upper_50:.8f}")
                print(f"Lower 50%   : {value.lower_50:.8f}")
            else:
                print(f"ERROR       : {value['error']}")

        print()
        print("=" * 75)


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    engine = TRILineEngine(symbol="BTCUSDT")

    # Enable all timeframes including 1m, 5m etc.
    for tf in TIMEFRAME_MAPPING.keys():
        engine.set_timeframe(tf, True)

    engine.print_levels()
    
    # Example: Agar aapko pichlay 3 mahino ka data ya multiple candles nikalni hon:
    # historical_1h = engine.get_historical_candles("1h", limit=200)
    # print(f"Fetched {len(historical_1h)} candles for 1H timeframe.")
