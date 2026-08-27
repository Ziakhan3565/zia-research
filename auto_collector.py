# auto_collector.py

from __future__ import annotations

import os
import time
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ============================================================
# ZIA RESEARCH
# BINANCE USDⓈ-M FUTURES AUTO COLLECTOR
# ============================================================
#
# DATA FLOW
#
# Binance USDⓈ-M Futures
#          ↓
# auto_collector.py
#          ↓
# ┌─────────────────────────────┐
# │ Futures Order Book           │
# │ Futures Recent Trades        │
# │ Futures OHLCV                │
# └─────────────────────────────┘
#          ↓
# research_lab.py
#          ↓
# OBI 5 / 10 / 20 / 50
# Taker Flow
# Fourier
# Bayesian
# Adaptive Trend
# ML
# TRI
#          ↓
# bot_engine.py
#          ↓
# FINAL SIGNAL
#          ↓
# dashboard.py
#
# ============================================================


# ============================================================
# 1. COINS
# ============================================================

COINS_LIST = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "XLMUSDT",
    "NEARUSDT",
    "UNIUSDT",
    "APTUSDT",
    "TAOUSDT",
    "XMRUSDT",
]


# ============================================================
# 2. BINANCE USDⓈ-M FUTURES CONFIG
# ============================================================

BASE_URL = "https://fapi.binance.com"

ORDER_BOOK_LIMIT = 100

REQUEST_TIMEOUT = 7

SYMBOL_DELAY = 0.20

CYCLE_DELAY = 5

TRADES_LIMIT = 1000

# OHLCV timeframes required by research engine
KLINE_INTERVALS = [
    "5m",
    "15m",
    "1h",
    "4h",
]

# How many candles to request
KLINE_LIMIT = 250


# ============================================================
# 3. CSV FILES
# ============================================================

MARKET_DATA_FILE = "market_data_log.csv"

TRADES_DATA_FILE = "futures_trades_log.csv"

OHLCV_DATA_FILE = "futures_ohlcv_log.csv"


# ============================================================
# 4. HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent": "Zia-Research-Futures-Collector/1.0"
    }
)


# ============================================================
# 5. TIME HELPERS
# ============================================================

def utc_now_string() -> str:
    """
    UTC timestamp for CSV logs.
    """

    return dt.datetime.now(
        dt.timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def ms_to_datetime_string(
    milliseconds: int,
) -> str:
    """
    Binance milliseconds -> UTC string.
    """

    try:
        value = dt.datetime.fromtimestamp(
            milliseconds / 1000.0,
            tz=dt.timezone.utc,
        )

        return value.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    except Exception:
        return utc_now_string()


# ============================================================
# 6. SAFE BINANCE GET
# ============================================================

def binance_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
):
    """
    Safe GET request to Binance USDⓈ-M Futures API.
    """

    url = f"{BASE_URL}{endpoint}"

    try:

        response = session.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        # Explicit Binance error handling
        if response.status_code != 200:

            print(
                f"⚠️ Binance HTTP {response.status_code} "
                f"| {endpoint}"
            )

            try:
                print(
                    f"   {response.json()}"
                )
            except Exception:
                pass

            return None

        return response.json()

    except requests.exceptions.Timeout:

        print(
            f"⚠️ Binance timeout | {endpoint}"
        )

        return None

    except requests.exceptions.ConnectionError:

        print(
            f"⚠️ Binance connection error | {endpoint}"
        )

        return None

    except requests.exceptions.RequestException as e:

        print(
            f"⚠️ Binance request error: {e}"
        )

        return None

    except ValueError:

        print(
            f"⚠️ Binance returned invalid JSON | {endpoint}"
        )

        return None

    except Exception as e:

        print(
            f"⚠️ Unexpected Binance error: {e}"
        )

        return None


# ============================================================
# 7. FUTURES PRICE
# ============================================================

def fetch_futures_price(
    symbol: str,
) -> Optional[float]:
    """
    Binance USDⓈ-M Futures mark-independent
    latest contract price.

    Endpoint:
        /fapi/v1/ticker/price
    """

    data = binance_get(
        "/fapi/v1/ticker/price",
        {
            "symbol": symbol
        },
    )

    if not data:
        return None

    try:

        return float(
            data["price"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# 8. FUTURES ORDER BOOK
# ============================================================

def fetch_futures_order_book(
    symbol: str,
    limit: int = ORDER_BOOK_LIMIT,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Binance USDⓈ-M Futures order book.

    Endpoint:
        /fapi/v1/depth

    Returns:

        bids:
            [[price, quantity], ...]

        asks:
            [[price, quantity], ...]
    """

    data = binance_get(
        "/fapi/v1/depth",
        {
            "symbol": symbol,
            "limit": limit,
        },
    )

    if not data:
        return None, None

    try:

        bids_raw = data.get(
            "bids",
            [],
        )

        asks_raw = data.get(
            "asks",
            [],
        )

        if not bids_raw or not asks_raw:
            return None, None

        bids = np.asarray(
            bids_raw,
            dtype=np.float64,
        )

        asks = np.asarray(
            asks_raw,
            dtype=np.float64,
        )

        if bids.ndim != 2:
            return None, None

        if asks.ndim != 2:
            return None, None

        if bids.shape[1] < 2:
            return None, None

        if asks.shape[1] < 2:
            return None, None

        return bids, asks

    except Exception as e:

        print(
            f"⚠️ Order book parsing error "
            f"[{symbol}]: {e}"
        )

        return None, None


# ============================================================
# 9. OBI
# ============================================================

def calculate_obi(
    bids: np.ndarray,
    asks: np.ndarray,
    levels: int,
) -> float:
    """
    OBI formula:

        OBI =
        (Bid Volume - Ask Volume)
        /
        (Bid Volume + Ask Volume)

    Range approximately:
        -1 → strong sell-side depth
         0 → balanced
        +1 → strong buy-side depth
    """

    if bids is None or asks is None:
        return 0.0

    if len(bids) < levels:
        return 0.0

    if len(asks) < levels:
        return 0.0

    bid_volume = float(
        np.sum(
            bids[:levels, 1]
        )
    )

    ask_volume = float(
        np.sum(
            asks[:levels, 1]
        )
    )

    total = (
        bid_volume
        + ask_volume
    )

    if total <= 0:
        return 0.0

    return (
        bid_volume - ask_volume
    ) / total


# ============================================================
# 10. DEPTH VOLUME
# ============================================================

def calculate_depth(
    bids: np.ndarray,
    asks: np.ndarray,
    levels: int,
) -> Tuple[float, float, float]:
    """
    Returns:

        bid_volume
        ask_volume
        total_volume
    """

    if len(bids) < levels:
        return 0.0, 0.0, 0.0

    if len(asks) < levels:
        return 0.0, 0.0, 0.0

    bid_volume = float(
        np.sum(
            bids[:levels, 1]
        )
    )

    ask_volume = float(
        np.sum(
            asks[:levels, 1]
        )
    )

    total_volume = (
        bid_volume
        + ask_volume
    )

    return (
        bid_volume,
        ask_volume,
        total_volume,
    )


# ============================================================
# 11. BID / ASK RATIO
# ============================================================

def calculate_bid_ask_ratio(
    bid_volume: float,
    ask_volume: float,
) -> float:

    if ask_volume <= 0:
        return 0.0

    return (
        bid_volume
        / ask_volume
    )


# ============================================================
# 12. SPREAD
# ============================================================

def calculate_spread(
    bids: np.ndarray,
    asks: np.ndarray,
) -> Tuple[float, float]:

    if (
        bids is None
        or asks is None
        or len(bids) == 0
        or len(asks) == 0
    ):
        return 0.0, 0.0

    best_bid = float(
        bids[0, 0]
    )

    best_ask = float(
        asks[0, 0]
    )

    spread = (
        best_ask
        - best_bid
    )

    midpoint = (
        best_ask
        + best_bid
    ) / 2.0

    if midpoint > 0:

        spread_pct = (
            spread
            / midpoint
        ) * 100.0

    else:

        spread_pct = 0.0

    return (
        spread,
        spread_pct,
    )


# ============================================================
# 13. FUTURES TRADES
# ============================================================

def fetch_futures_trades(
    symbol: str,
    limit: int = TRADES_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Binance USDⓈ-M Futures recent trades.

    Endpoint:
        /fapi/v1/aggTrades

    Used by Research Lab for Taker Flow.

    Binance field 'm':

        True
            buyer is market maker
            → aggressive seller

        False
            buyer is taker
            → aggressive buyer
    """

    data = binance_get(
        "/fapi/v1/aggTrades",
        {
            "symbol": symbol,
            "limit": limit,
        },
    )

    if not isinstance(data, list):
        return []

    return data


# ============================================================
# 14. TAKER FLOW
# ============================================================

def calculate_taker_flow(
    trades: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate aggressive buy/sell flow.

    BUY:
        buyer is taker

    SELL:
        seller is taker

    Flow:

        buy_volume - sell_volume
    """

    buy_volume = 0.0

    sell_volume = 0.0

    buy_notional = 0.0

    sell_notional = 0.0

    trade_count = 0

    if not trades:
        return {
            "taker_buy_volume": 0.0,
            "taker_sell_volume": 0.0,
            "taker_buy_notional": 0.0,
            "taker_sell_notional": 0.0,
            "taker_flow": 0.0,
            "taker_flow_ratio": 0.0,
            "trade_count": 0,
        }

    for trade in trades:

        try:

            price = float(
                trade["p"]
            )

            quantity = float(
                trade["q"]
            )

            is_buyer_maker = bool(
                trade["m"]
            )

            notional = (
                price * quantity
            )

            # buyer is maker
            # therefore seller is aggressive
            if is_buyer_maker:

                sell_volume += quantity

                sell_notional += notional

            else:

                buy_volume += quantity

                buy_notional += notional

            trade_count += 1

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            continue

    total_volume = (
        buy_volume
        + sell_volume
    )

    flow = (
        buy_volume
        - sell_volume
    )

    if total_volume > 0:

        flow_ratio = (
            flow
            / total_volume
        )

    else:

        flow_ratio = 0.0

    return {
        "taker_buy_volume": buy_volume,
        "taker_sell_volume": sell_volume,
        "taker_buy_notional": buy_notional,
        "taker_sell_notional": sell_notional,
        "taker_flow": flow,
        "taker_flow_ratio": flow_ratio,
        "trade_count": trade_count,
    }


# ============================================================
# 15. SAVE RAW TRADE SUMMARY
# ============================================================

def save_trade_summary(
    symbol: str,
    flow: Dict[str, float],
    file_path: str = TRADES_DATA_FILE,
) -> bool:

    try:

        row = {
            "timestamp": utc_now_string(),
            "symbol": symbol,
            **flow,
        }

        df = pd.DataFrame(
            [row]
        )

        file_exists = os.path.isfile(
            file_path
        )

        df.to_csv(
            file_path,
            mode="a",
            header=not file_exists,
            index=False,
        )

        return True

    except Exception as e:

        print(
            f"⚠️ Trade CSV error: {e}"
        )

        return False


# ============================================================
# 16. FUTURES OHLCV
# ============================================================

def fetch_futures_ohlcv(
    symbol: str,
    interval: str,
    limit: int = KLINE_LIMIT,
) -> Optional[pd.DataFrame]:
    """
    Binance USDⓈ-M Futures Klines.

    Endpoint:
        /fapi/v1/klines
    """

    data = binance_get(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
    )

    if not isinstance(data, list):
        return None

    rows = []

    for candle in data:

        if len(candle) < 12:
            continue

        try:

            rows.append(
                {
                    "open_time": int(
                        candle[0]
                    ),
                    "open": float(
                        candle[1]
                    ),
                    "high": float(
                        candle[2]
                    ),
                    "low": float(
                        candle[3]
                    ),
                    "close": float(
                        candle[4]
                    ),
                    "volume": float(
                        candle[5]
                    ),
                    "close_time": int(
                        candle[6]
                    ),
                    "quote_volume": float(
                        candle[7]
                    ),
                    "trade_count": int(
                        candle[8]
                    ),
                    "taker_buy_base_volume": float(
                        candle[9]
                    ),
                    "taker_buy_quote_volume": float(
                        candle[10]
                    ),
                }
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

    if not rows:
        return None

    df = pd.DataFrame(
        rows
    )

    df["open_time_utc"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close_time_utc"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True,
    )

    df["symbol"] = symbol

    df["timeframe"] = interval

    return df


# ============================================================
# 17. SAVE OHLCV
# ============================================================

def save_ohlcv(
    df: Optional[pd.DataFrame],
    file_path: str = OHLCV_DATA_FILE,
) -> bool:

    if df is None or df.empty:
        return False

    try:

        file_exists = os.path.isfile(
            file_path
        )

        # Save only latest candle from each request
        latest = df.tail(1)

        latest.to_csv(
            file_path,
            mode="a",
            header=not file_exists,
            index=False,
        )

        return True

    except Exception as e:

        print(
            f"⚠️ OHLCV CSV error: {e}"
        )

        return False


# ============================================================
# 18. COMPLETE MARKET SNAPSHOT
# ============================================================

def collect_market_snapshot(
    symbol: str,
) -> Optional[Dict[str, Any]]:
    """
    Collect:

        Price
        Order Book
        OBI 5/10/20/50
        Top20 depth
        Top50 depth
        Spread
        Ratios
        Taker Flow
    """

    price = fetch_futures_price(
        symbol
    )

    bids, asks = fetch_futures_order_book(
        symbol,
        ORDER_BOOK_LIMIT,
    )

    if price is None:
        return None

    if bids is None or asks is None:
        return None

    # --------------------------------------------------------
    # OBI
    # --------------------------------------------------------

    obi_5 = calculate_obi(
        bids,
        asks,
        5,
    )

    obi_10 = calculate_obi(
        bids,
        asks,
        10,
    )

    obi_20 = calculate_obi(
        bids,
        asks,
        20,
    )

    obi_50 = calculate_obi(
        bids,
        asks,
        50,
    )

    # --------------------------------------------------------
    # TOP 20
    # --------------------------------------------------------

    (
        bid20,
        ask20,
        total20,
    ) = calculate_depth(
        bids,
        asks,
        20,
    )

    # --------------------------------------------------------
    # TOP 50
    # --------------------------------------------------------

    (
        bid50,
        ask50,
        total50,
    ) = calculate_depth(
        bids,
        asks,
        50,
    )

    # --------------------------------------------------------
    # SPREAD
    # --------------------------------------------------------

    (
        spread,
        spread_pct,
    ) = calculate_spread(
        bids,
        asks,
    )

    # --------------------------------------------------------
    # RATIOS
    # --------------------------------------------------------

    ratio20 = calculate_bid_ask_ratio(
        bid20,
        ask20,
    )

    ratio50 = calculate_bid_ask_ratio(
        bid50,
        ask50,
    )

    # --------------------------------------------------------
    # TRADES
    # --------------------------------------------------------

    trades = fetch_futures_trades(
        symbol,
        TRADES_LIMIT,
    )

    taker = calculate_taker_flow(
        trades
    )

    # --------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------

    data = {
        "timestamp": utc_now_string(),

        "symbol": symbol,

        "market": "BINANCE_USDM_FUTURES",

        "current_price": price,

        # OBI
        "obi_5": round(
            obi_5,
            8,
        ),

        "obi_10": round(
            obi_10,
            8,
        ),

        "obi_20": round(
            obi_20,
            8,
        ),

        "obi_50": round(
            obi_50,
            8,
        ),

        # Compatibility with existing Research Lab
        "obi_top20": round(
            obi_20,
            8,
        ),

        # Top 20
        "top20_bid_sum": round(
            bid20,
            8,
        ),

        "top20_ask_sum": round(
            ask20,
            8,
        ),

        "top20_total_depth": round(
            total20,
            8,
        ),

        # Top 50
        "top50_bid_sum": round(
            bid50,
            8,
        ),

        "top50_ask_sum": round(
            ask50,
            8,
        ),

        "top50_total_depth": round(
            total50,
            8,
        ),

        # Ratios
        "bid_ask_ratio_20": round(
            ratio20,
            8,
        ),

        "bid_ask_ratio_50": round(
            ratio50,
            8,
        ),

        # Spread
        "spread": round(
            spread,
            8,
        ),

        "spread_pct": round(
            spread_pct,
            8,
        ),

        # Taker Flow
        "taker_buy_volume": round(
            taker["taker_buy_volume"],
            8,
        ),

        "taker_sell_volume": round(
            taker["taker_sell_volume"],
            8,
        ),

        "taker_buy_notional": round(
            taker["taker_buy_notional"],
            8,
        ),

        "taker_sell_notional": round(
            taker["taker_sell_notional"],
            8,
        ),

        "taker_flow": round(
            taker["taker_flow"],
            8,
        ),

        "taker_flow_ratio": round(
            taker["taker_flow_ratio"],
            8,
        ),

        "trade_count": int(
            taker["trade_count"]
        ),

        # Depth levels
        "bid_levels": int(
            len(bids)
        ),

        "ask_levels": int(
            len(asks)
        ),
    }

    return data


# ============================================================
# 19. SAVE MARKET SNAPSHOT
# ============================================================

def save_market_data(
    data: Optional[Dict[str, Any]],
    file_path: str = MARKET_DATA_FILE,
) -> bool:

    if data is None:
        return False

    try:

        df = pd.DataFrame(
            [data]
        )

        file_exists = os.path.isfile(
            file_path
        )

        df.to_csv(
            file_path,
            mode="a",
            header=not file_exists,
            index=False,
        )

        return True

    except Exception as e:

        print(
            f"⚠️ Market CSV error: {e}"
        )

        return False


# ============================================================
# 20. COLLECT OHLCV FOR ALL TIMEFRAMES
# ============================================================

def collect_ohlcv_for_symbol(
    symbol: str,
):
    """
    Collect latest 5m / 15m / 1h / 4h candles.
    """

    for interval in KLINE_INTERVALS:

        try:

            df = fetch_futures_ohlcv(
                symbol,
                interval,
                KLINE_LIMIT,
            )

            if df is not None:

                save_ohlcv(
                    df
                )

        except Exception as e:

            print(
                f"⚠️ OHLCV error "
                f"[{symbol} {interval}]: {e}"
            )


# ============================================================
# 21. ONE SYMBOL COLLECTION
# ============================================================

def collect_symbol(
    symbol: str,
    count: int,
) -> int:
    """
    Collect all required data for one symbol.

    Returns updated count.
    """

    print(
        f"\n📡 Collecting {symbol}..."
    )

    # --------------------------------------------------------
    # Main snapshot
    # --------------------------------------------------------

    data = collect_market_snapshot(
        symbol
    )

    if data is None:

        print(
            f"❌ No market data: {symbol}"
        )

        return count

    # --------------------------------------------------------
    # Save market data
    # --------------------------------------------------------

    if save_market_data(
        data
    ):

        count += 1

    # --------------------------------------------------------
    # Save taker flow
    # --------------------------------------------------------

    taker_flow = {
        "taker_buy_volume":
            data["taker_buy_volume"],

        "taker_sell_volume":
            data["taker_sell_volume"],

        "taker_buy_notional":
            data["taker_buy_notional"],

        "taker_sell_notional":
            data["taker_sell_notional"],

        "taker_flow":
            data["taker_flow"],

        "taker_flow_ratio":
            data["taker_flow_ratio"],

        "trade_count":
            data["trade_count"],
    }

    save_trade_summary(
        symbol,
        taker_flow,
    )

    # --------------------------------------------------------
    # OHLCV
    # --------------------------------------------------------

    collect_ohlcv_for_symbol(
        symbol
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print(
        f"✅ [{count}] "
        f"{symbol} | "
        f"Price: {data['current_price']:.8f} | "
        f"OBI5: {data['obi_5']:+.4f} | "
        f"OBI10: {data['obi_10']:+.4f} | "
        f"OBI20: {data['obi_20']:+.4f} | "
        f"OBI50: {data['obi_50']:+.4f} | "
        f"Taker: {data['taker_flow']:+.4f}"
    )

    return count


# ============================================================
# 22. AUTO COLLECTOR
# ============================================================

def log_auto_data(
    file_path: str = MARKET_DATA_FILE,
):
    """
    Continuous collector.
    """

    print(
        "\n"
        "============================================================"
    )

    print(
        "🚀 ZIA RESEARCH"
    )

    print(
        "📡 BINANCE USDⓈ-M FUTURES AUTO COLLECTOR"
    )

    print(
        "============================================================"
    )

    print(
        f"🪙 Coins       : {len(COINS_LIST)}"
    )

    print(
        f"📚 Order Book  : {ORDER_BOOK_LIMIT} levels"
    )

    print(
        "📊 OBI         : 5 / 10 / 20 / 50"
    )

    print(
        "🔥 Taker Flow  : Futures AggTrades"
    )

    print(
        "📈 OHLCV       : 5m / 15m / 1h / 4h"
    )

    print(
        "🌐 Market      : Binance USDⓈ-M Futures"
    )

    print(
        f"💾 Market CSV  : {file_path}"
    )

    print(
        "============================================================\n"
    )

    count = 0

    cycle_number = 0

    while True:

        cycle_number += 1

        cycle_start = time.time()

        successful = 0

        failed = 0

        print(
            f"\n🔄 STARTING CYCLE #{cycle_number}"
        )

        print(
            "------------------------------------------------------------"
        )

        for symbol in COINS_LIST:

            try:

                previous_count = count

                count = collect_symbol(
                    symbol,
                    count,
                )

                if count > previous_count:

                    successful += 1

                else:

                    failed += 1

            except KeyboardInterrupt:

                raise

            except Exception as e:

                failed += 1

                print(
                    f"❌ Symbol error "
                    f"[{symbol}]: {e}"
                )

            time.sleep(
                SYMBOL_DELAY
            )

        elapsed = (
            time.time()
            - cycle_start
        )

        print(
            "------------------------------------------------------------"
        )

        print(
            f"✅ CYCLE #{cycle_number} COMPLETE"
        )

        print(
            f"   Success : {successful}"
        )

        print(
            f"   Failed  : {failed}"
        )

        print(
            f"   Time    : {elapsed:.2f}s"
        )

        print(
            f"   Total snapshots: {count}"
        )

        print(
            f"⏳ Waiting {CYCLE_DELAY}s..."
        )

        time.sleep(
            CYCLE_DELAY
        )


# ============================================================
# 23. TEST MODE
# ============================================================

def test_collector(
    symbol: str = "BTCUSDT",
):
    """
    One-shot collector test.

    Does NOT start infinite loop.
    """

    print(
        "\n============================================================"
    )

    print(
        f"🔎 TESTING {symbol}"
    )

    print(
        "============================================================"
    )

    data = collect_market_snapshot(
        symbol
    )

    if data is None:

        print(
            "❌ Test failed."
        )

        return

    print(
        "\n========== FUTURES MARKET DATA =========="
    )

    important_fields = [
        "market",
        "symbol",
        "current_price",

        "obi_5",
        "obi_10",
        "obi_20",
        "obi_50",

        "top20_bid_sum",
        "top20_ask_sum",

        "top50_bid_sum",
        "top50_ask_sum",

        "bid_ask_ratio_20",
        "bid_ask_ratio_50",

        "spread",
        "spread_pct",

        "taker_buy_volume",
        "taker_sell_volume",
        "taker_flow",
        "taker_flow_ratio",

        "trade_count",

        "bid_levels",
        "ask_levels",
    ]

    for key in important_fields:

        print(
            f"{key:28} : "
            f"{data.get(key)}"
        )

    print(
        "=========================================\n"
    )

    # --------------------------------------------------------
    # Test OHLCV
    # --------------------------------------------------------

    print(
        "📈 OHLCV TEST"
    )

    for interval in KLINE_INTERVALS:

        df = fetch_futures_ohlcv(
            symbol,
            interval,
            KLINE_LIMIT,
        )

        if df is None:

            print(
                f"❌ {interval}: failed"
            )

        else:

            print(
                f"✅ {interval}: "
                f"{len(df)} candles | "
                f"Latest close: "
                f"{df.iloc[-1]['close']}"
            )


# ============================================================
# 24. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # True:
    # Test only BTCUSDT once.
    #
    # False:
    # Start continuous collector.
    # --------------------------------------------------------

    TEST_MODE = False

    try:

        if TEST_MODE:

            test_collector(
                "BTCUSDT"
            )

        else:

            log_auto_data()

    except KeyboardInterrupt:

        print(
            "\n\n🛑 Collector stopped by user."
        )

    except Exception as e:

        print(
            f"\n❌ FATAL COLLECTOR ERROR: {e}"
        )
