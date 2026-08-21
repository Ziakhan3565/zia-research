from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import requests


# ============================================================
# CONFIGURATION
# ============================================================

BINANCE_API = "https://api.binance.com/api/v3/klines"

DEFAULT_SYMBOL = "BTCUSDT"

SUPPORTED_TIMEFRAMES = {
    "YEARLY": "1y",
    "MONTHLY": "1M",
    "WEEKLY": "1w",
    "DAILY": "1d",
    "4H": "4h",
    "1H": "1h",
    "30M": "30m",
    "15M": "15m",
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
# TRI LINE ENGINE WITH ENTRY SIGNALS
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

        self.enabled = {
            "YEARLY": True,
            "MONTHLY": True,
            "WEEKLY": True,
            "DAILY": True,
            "4H": True,
            "1H": True,
            "30M": True,
            "15M": True,
        }

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
        }

    def set_symbol(self, symbol: str):
        self.symbol = symbol.upper()

    def set_timeframe(self, timeframe: str, enabled: bool):
        timeframe = timeframe.upper()
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self.enabled[timeframe] = enabled

    def set_color(self, timeframe: str, color: str):
        timeframe = timeframe.upper()
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self.colors[timeframe] = color

    def get_klines(self, interval: str, limit: int = 5):
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit,
        }
        response = requests.get(BINANCE_API, params=params, timeout=self.timeout)
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
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        interval = SUPPORTED_TIMEFRAMES[timeframe]
        candle = self.get_previous_candle(interval)

        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]

        body_high = max(o, c)
        body_low = min(o, c)

        body_50 = (body_high + body_low) / 2.0
        upper_50 = (h + body_high) / 2.0
        lower_50 = (l + body_low) / 2.0

        return TRILineLevels(
            timeframe=timeframe,
            open=o, high=h, low=l, close=c,
            body_high=body_high, body_low=body_low,
            body_50=body_50, upper_50=upper_50, lower_50=lower_50,
            candle_time=candle["time"],
        )

    # ========================================================
    # NEW: ENTRY SIGNAL GENERATOR METHOD
    # ========================================================
    def get_entry_signal(self, timeframe: str) -> dict:
        timeframe = timeframe.upper()
        interval = SUPPORTED_TIMEFRAMES[timeframe]
        
        # Calculate TRI levels from previous completed candle
        levels = self.calculate_levels(timeframe)
        
        # Get latest live/forming candle close price to compare
        candles = self.get_klines(interval, limit=1)
        current_price = float(candles[0][4])

        signal = "NEUTRAL"
        stop_loss = 0.0
        take_profit = 0.0

        # Entry Rules based on Body 50% midpoint
        if current_price > levels.body_50:
            signal = "LONG"
            stop_loss = levels.body_low
            take_profit = levels.upper_50
        elif current_price < levels.body_50:
            signal = "SHORT"
            stop_loss = levels.body_high
            take_profit = levels.lower_50

        return {
            "timeframe": timeframe,
            "signal": signal,
            "current_price": current_price,
            "body_50": levels.body_50,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def calculate_all(self):
        results = {}
        for timeframe in SUPPORTED_TIMEFRAMES:
            if not self.enabled.get(timeframe, False):
                continue
            try:
                results[timeframe] = self.calculate_levels(timeframe)
            except Exception as error:
                results[timeframe] = {"error": str(error)}
        return results

    def print_levels(self):
        results = self.calculate_all()

        print()
        print("=" * 75)
        print(f"TRI LINE ANALYSIS & ENTRIES | {self.symbol}")
        print("=" * 75)

        for timeframe, value in results.items():
            print()
            print(f"[{timeframe}]")
            if isinstance(value, TRILineLevels):
                # Fetch entry signal for this timeframe
                sig_data = self.get_entry_signal(timeframe)
                
                print(f"Color         : {self.colors[timeframe]}")
                print(f"Body 50%      : {value.body_50:.2f}")
                print(f"Upper 50%     : {value.upper_50:.2f}")
                print(f"Lower 50%     : {value.lower_50:.2f}")
                print("-" * 40)
                print(f"SIGNAL        : {sig_data['signal']}")
                print(f"Entry Price   : {sig_data['current_price']:.2f}")
                print(f"Stop Loss     : {sig_data['stop_loss']:.2f}")
                print(f"Take Profit   : {sig_data['take_profit']:.2f}")
            else:
                print(f"ERROR         : {value['error']}")

        print()
        print("=" * 75)


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":
    engine = TRILineEngine(symbol="BTCUSDT")
    engine.print_levels()
