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
# MARKET DATA ENGINE
#
# Binance Futures
#      ↓
# Order Book
# Recent Trades
# OHLCV
#      ↓
# Feature Engineering
#      ↓
# market_data_log.csv
# futures_trades_log.csv
# futures_ohlcv_log.csv
#      ↓
# research_lab.py
#
# TIMEFRAME DESIGN
#
# SCALPING
#   Signal horizon : 30 minutes
#   Max trade time : 15 minutes
#
# 1H
#   Signal horizon : 1 hour
#   Max trade time : 1.5 hours
#
# 4H
#   Signal horizon : 4 hours
#   Max trade time : 24 hours
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
# 2. BINANCE CONFIG
# ============================================================

BASE_URL = "https://fapi.binance.com"

ORDER_BOOK_LIMIT = 100

REQUEST_TIMEOUT = 10

SYMBOL_DELAY = 0.15

CYCLE_DELAY = 5

TRADES_LIMIT = 1000

# Keep historical candles available for research
KLINE_LIMIT = 250

KLINE_INTERVALS = [
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
]


# ============================================================
# 3. TRADE HORIZONS
# ============================================================

TIMEFRAME_CONFIG = {
    "SCALP": {
        "signal_timeframe": "30m",
        "max_trade_minutes": 15,
        "evaluation_minutes": 15,
    },
    "1H": {
        "signal_timeframe": "1h",
        "max_trade_minutes": 90,
        "evaluation_minutes": 90,
    },
    "4H": {
        "signal_timeframe": "4h",
        "max_trade_minutes": 1440,
        "evaluation_minutes": 1440,
    },
}


# ============================================================
# 4. CSV FILES
# ============================================================

MARKET_DATA_FILE = "market_data_log.csv"

TRADES_DATA_FILE = "futures_trades_log.csv"

OHLCV_DATA_FILE = "futures_ohlcv_log.csv"


# ============================================================
# 5. HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent": "Zia-Research-Futures-Collector/2.0"
    }
)


# ============================================================
# 6. REQUEST RETRY SETTINGS
# ============================================================

MAX_RETRIES = 3

RETRY_DELAY = 1.0


# ============================================================
# 7. TIME HELPERS
# ============================================================

def utc_now_string() -> str:
    return dt.datetime.now(
        dt.timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def utc_now_iso() -> str:
    return dt.datetime.now(
        dt.timezone.utc
    ).isoformat()


def ms_to_datetime_string(
    milliseconds: int,
) -> str:

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
# 8. SAFE NUMBER
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        number = float(value)

        if not np.isfinite(number):
            return default

        return number

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================
# 9. BINANCE GET
# ============================================================

def binance_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
):
    """
    Safe Binance Futures GET with retry logic.
    """

    url = f"{BASE_URL}{endpoint}"

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            if response.status_code == 200:

                try:
                    return response.json()

                except ValueError:

                    print(
                        f"⚠️ Invalid JSON | {endpoint}"
                    )

                    return None

            # ------------------------------------------------
            # Rate limit
            # ------------------------------------------------

            if response.status_code in (
                418,
                429,
            ):

                retry_after = response.headers.get(
                    "Retry-After"
                )

                try:
                    wait_time = float(
                        retry_after
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    wait_time = RETRY_DELAY * attempt

                print(
                    f"⚠️ Binance rate limit | "
                    f"waiting {wait_time:.1f}s"
                )

                time.sleep(
                    wait_time
                )

                continue

            # ------------------------------------------------
            # Server errors
            # ------------------------------------------------

            if response.status_code >= 500:

                print(
                    f"⚠️ Binance server error "
                    f"{response.status_code} | "
                    f"attempt {attempt}/{MAX_RETRIES}"
                )

                time.sleep(
                    RETRY_DELAY * attempt
                )

                continue

            # ------------------------------------------------
            # Other HTTP errors
            # ------------------------------------------------

            print(
                f"⚠️ Binance HTTP "
                f"{response.status_code} | "
                f"{endpoint}"
            )

            try:
                print(response.json())
            except Exception:
                pass

            return None

        except requests.exceptions.Timeout:

            print(
                f"⚠️ Timeout | "
                f"{endpoint} | "
                f"attempt {attempt}/{MAX_RETRIES}"
            )

        except requests.exceptions.ConnectionError:

            print(
                f"⚠️ Connection error | "
                f"attempt {attempt}/{MAX_RETRIES}"
            )

        except requests.exceptions.RequestException as error:

            print(
                f"⚠️ Request error: {error}"
            )

        except Exception as error:

            print(
                f"⚠️ Unexpected Binance error: "
                f"{error}"
            )

        if attempt < MAX_RETRIES:

            time.sleep(
                RETRY_DELAY * attempt
            )

    return None


# ============================================================
# 10. FUTURES PRICE
# ============================================================

def fetch_futures_price(
    symbol: str,
) -> Optional[float]:

    data = binance_get(
        "/fapi/v1/ticker/price",
        {
            "symbol": symbol
        },
    )

    if not isinstance(data, dict):
        return None

    return safe_float(
        data.get("price"),
        0.0,
    ) or None


# ============================================================
# 11. ORDER BOOK
# ============================================================

def fetch_futures_order_book(
    symbol: str,
    limit: int = ORDER_BOOK_LIMIT,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
]:

    data = binance_get(
        "/fapi/v1/depth",
        {
            "symbol": symbol,
            "limit": limit,
        },
    )

    if not isinstance(data, dict):
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

        if (
            bids.ndim != 2
            or asks.ndim != 2
        ):
            return None, None

        if (
            bids.shape[1] < 2
            or asks.shape[1] < 2
        ):
            return None, None

        return bids, asks

    except Exception as error:

        print(
            f"⚠️ Order book parsing error "
            f"[{symbol}]: {error}"
        )

        return None, None


# ============================================================
# 12. OBI
# ============================================================

def calculate_obi(
    bids: np.ndarray,
    asks: np.ndarray,
    levels: int,
) -> float:

    if bids is None or asks is None:
        return 0.0

    if (
        len(bids) < levels
        or len(asks) < levels
    ):
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
        bid_volume
        - ask_volume
    ) / total


# ============================================================
# 13. DEPTH
# ============================================================

def calculate_depth(
    bids: np.ndarray,
    asks: np.ndarray,
    levels: int,
) -> Tuple[float, float, float]:

    if (
        bids is None
        or asks is None
    ):
        return 0.0, 0.0, 0.0

    if (
        len(bids) < levels
        or len(asks) < levels
    ):
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

    total = (
        bid_volume
        + ask_volume
    )

    return (
        bid_volume,
        ask_volume,
        total,
    )


# ============================================================
# 14. BID ASK RATIO
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
# 15. SPREAD
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

    best_bid = safe_float(
        bids[0, 0]
    )

    best_ask = safe_float(
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
# 16. RECENT FUTURES TRADES
# ============================================================

def fetch_futures_trades(
    symbol: str,
    limit: int = TRADES_LIMIT,
) -> List[Dict[str, Any]]:

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
# 17. TAKER FLOW
# ============================================================

def calculate_taker_flow(
    trades: List[Dict[str, Any]],
) -> Dict[str, float]:

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

            buyer_maker = bool(
                trade["m"]
            )

            notional = (
                price
                * quantity
            )

            if buyer_maker:

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
# 18. TECHNICAL INDICATORS
# ============================================================

def calculate_ema(
    series: pd.Series,
    period: int,
) -> pd.Series:

    return (
        series
        .ewm(
            span=period,
            adjust=False,
        )
        .mean()
    )


def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / (
        avg_loss + 1e-12
    )

    rsi = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    return rsi


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    high = df["high"]

    low = df["low"]

    close = df["close"]

    previous_close = close.shift(1)

    tr1 = (
        high
        - low
    )

    tr2 = (
        high
        - previous_close
    ).abs()

    tr3 = (
        low
        - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


def calculate_vwap(
    df: pd.DataFrame,
) -> pd.Series:

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3.0

    cumulative_volume = (
        df["volume"]
        .cumsum()
    )

    cumulative_pv = (
        typical_price
        * df["volume"]
    ).cumsum()

    return (
        cumulative_pv
        / (
            cumulative_volume
            + 1e-12
        )
    )


def enrich_ohlcv(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df is None or df.empty:
        return df

    result = df.copy()

    result["ema_20"] = calculate_ema(
        result["close"],
        20,
    )

    result["ema_50"] = calculate_ema(
        result["close"],
        50,
    )

    result["ema_200"] = calculate_ema(
        result["close"],
        200,
    )

    result["rsi_14"] = calculate_rsi(
        result["close"],
        14,
    )

    result["atr_14"] = calculate_atr(
        result,
        14,
    )

    result["atr_pct"] = (
        result["atr_14"]
        / (
            result["close"]
            + 1e-12
        )
    ) * 100.0

    result["vwap"] = calculate_vwap(
        result
    )

    result["ema_trend"] = np.where(
        (
            result["ema_20"]
            > result["ema_50"]
        )
        & (
            result["ema_50"]
            > result["ema_200"]
        ),
        1,
        np.where(
            (
                result["ema_20"]
                < result["ema_50"]
            )
            & (
                result["ema_50"]
                < result["ema_200"]
            ),
            -1,
            0,
        ),
    )

    result["price_vs_vwap"] = (
        result["close"]
        - result["vwap"]
    )

    result["return_1"] = (
        result["close"]
        .pct_change()
    )

    result["volatility_20"] = (
        result["return_1"]
        .rolling(
            20,
            min_periods=5,
        )
        .std()
        * 100.0
    )

    result["volume_sma_20"] = (
        result["volume"]
        .rolling(
            20,
            min_periods=1,
        )
        .mean()
    )

    result["volume_ratio"] = (
        result["volume"]
        / (
            result["volume_sma_20"]
            + 1e-12
        )
    )

    return result


# ============================================================
# 19. FUTURES OHLCV
# ============================================================

def fetch_futures_ohlcv(
    symbol: str,
    interval: str,
    limit: int = KLINE_LIMIT,
) -> Optional[pd.DataFrame]:

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

    df = enrich_ohlcv(
        df
    )

    return df


# ============================================================
# 20. SAVE OHLCV WITHOUT DUPLICATES
# ============================================================

def save_ohlcv(
    df: Optional[pd.DataFrame],
    file_path: str = OHLCV_DATA_FILE,
) -> bool:

    if df is None or df.empty:
        return False

    try:

        latest = df.tail(1).copy()

        file_exists = os.path.isfile(
            file_path
        )

        # ----------------------------------------------------
        # Existing data
        # ----------------------------------------------------

        if file_exists:

            try:

                existing = pd.read_csv(
                    file_path,
                    usecols=[
                        "symbol",
                        "timeframe",
                        "open_time",
                    ],
                )

                keys = set(
                    zip(
                        existing["symbol"].astype(str),
                        existing["timeframe"].astype(str),
                        existing["open_time"].astype(str),
                    )
                )

                new_key = (
                    str(latest.iloc[0]["symbol"]),
                    str(latest.iloc[0]["timeframe"]),
                    str(latest.iloc[0]["open_time"]),
                )

                if new_key in keys:

                    return True

            except Exception:
                pass

        latest.to_csv(
            file_path,
            mode="a",
            header=not file_exists,
            index=False,
        )

        return True

    except Exception as error:

        print(
            f"⚠️ OHLCV CSV error: {error}"
        )

        return False


# ============================================================
# 21. SAVE TRADE SUMMARY
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

    except Exception as error:

        print(
            f"⚠️ Trade CSV error: {error}"
        )

        return False


# ============================================================
# 22. MARKET SNAPSHOT
# ============================================================

def collect_market_snapshot(
    symbol: str,
) -> Optional[Dict[str, Any]]:

    price = fetch_futures_price(
        symbol
    )

    bids, asks = fetch_futures_order_book(
        symbol,
        ORDER_BOOK_LIMIT,
    )

    if price is None:
        return None

    if (
        bids is None
        or asks is None
    ):
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
    # Depth
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
    # Spread
    # --------------------------------------------------------

    (
        spread,
        spread_pct,
    ) = calculate_spread(
        bids,
        asks,
    )

    # --------------------------------------------------------
    # Ratios
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
    # Trades
    # --------------------------------------------------------

    trades = fetch_futures_trades(
        symbol,
        TRADES_LIMIT,
    )

    taker = calculate_taker_flow(
        trades
    )

    # --------------------------------------------------------
    # Snapshot
    # --------------------------------------------------------

    data = {

        "timestamp": utc_now_string(),

        "timestamp_iso": utc_now_iso(),

        "symbol": symbol,

        "market": "BINANCE_USDM_FUTURES",

        "current_price": price,

        # ----------------------------------------------------
        # OBI
        # ----------------------------------------------------

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

        # Existing compatibility
        "obi_top20": round(
            obi_20,
            8,
        ),

        # ----------------------------------------------------
        # TOP 20
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # TOP 50
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # RATIOS
        # ----------------------------------------------------

        "bid_ask_ratio_20": round(
            ratio20,
            8,
        ),

        "bid_ask_ratio_50": round(
            ratio50,
            8,
        ),

        # ----------------------------------------------------
        # SPREAD
        # ----------------------------------------------------

        "spread": round(
            spread,
            8,
        ),

        "spread_pct": round(
            spread_pct,
            8,
        ),

        # ----------------------------------------------------
        # TAKER FLOW
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # BOOK QUALITY
        # ----------------------------------------------------

        "bid_levels": int(
            len(bids)
        ),

        "ask_levels": int(
            len(asks)
        ),

        "orderbook_levels_requested": int(
            ORDER_BOOK_LIMIT
        ),

        "data_quality": (
            "OK"
            if (
                len(bids) >= 50
                and len(asks) >= 50
            )
            else "LIMITED"
        ),

        # ----------------------------------------------------
        # SIGNAL HORIZON METADATA
        # ----------------------------------------------------

        "scalp_signal_tf": "30m",

        "scalp_max_trade_minutes": 15,

        "one_hour_signal_tf": "1h",

        "one_hour_max_trade_minutes": 90,

        "four_hour_signal_tf": "4h",

        "four_hour_max_trade_minutes": 1440,
    }

    return data


# ============================================================
# 23. SAVE MARKET DATA
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

    except Exception as error:

        print(
            f"⚠️ Market CSV error: {error}"
        )

        return False


# ============================================================
# 24. COLLECT OHLCV
# ============================================================

def collect_ohlcv_for_symbol(
    symbol: str,
):
    """
    Collect latest candles for:

        5m
        15m
        30m
        1h
        4h
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

        except Exception as error:

            print(
                f"⚠️ OHLCV error "
                f"[{symbol} {interval}]: "
                f"{error}"
            )


# ============================================================
# 25. COLLECT ONE SYMBOL
# ============================================================

def collect_symbol(
    symbol: str,
    count: int,
) -> int:

    print(
        f"\n📡 Collecting {symbol}..."
    )

    # --------------------------------------------------------
    # Main market snapshot
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
    # Taker flow
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
# 26. DATA DIRECTORY CHECK
# ============================================================

def ensure_data_files():

    files = [
        MARKET_DATA_FILE,
        TRADES_DATA_FILE,
        OHLCV_DATA_FILE,
    ]

    for file_path in files:

        directory = os.path.dirname(
            os.path.abspath(
                file_path
            )
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )


# ============================================================
# 27. COLLECTOR STATUS
# ============================================================

def print_system_config():

    print(
        "\n============================================================"
    )

    print(
        "🚀 ZIA RESEARCH"
    )

    print(
        "📡 BINANCE USDⓈ-M FUTURES AUTO COLLECTOR v2"
    )

    print(
        "============================================================"
    )

    print(
        f"🪙 Coins              : {len(COINS_LIST)}"
    )

    print(
        f"📚 Order Book         : {ORDER_BOOK_LIMIT} levels"
    )

    print(
        "📊 OBI                : 5 / 10 / 20 / 50"
    )

    print(
        "🔥 Taker Flow         : Futures AggTrades"
    )

    print(
        "📈 OHLCV              : "
        "5m / 15m / 30m / 1h / 4h"
    )

    print(
        "📐 EMA                : 20 / 50 / 200"
    )

    print(
        "📊 RSI                : 14"
    )

    print(
        "📏 ATR                : 14"
    )

    print(
        "📍 VWAP               : Enabled"
    )

    print(
        "🌊 Volatility         : Enabled"
    )

    print(
        "📈 Volume Ratio       : Enabled"
    )

    print(
        "🌐 Market             : Binance USDⓈ-M Futures"
    )

    print(
        "\nTIMEFRAME TRADE WINDOWS"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        "SCALPING  | Signal: 30m | Max trade: 15 minutes"
    )

    print(
        "1H        | Signal: 1h  | Max trade: 90 minutes"
    )

    print(
        "4H        | Signal: 4h  | Max trade: 24 hours"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"💾 Market CSV         : {MARKET_DATA_FILE}"
    )

    print(
        f"💾 Trades CSV         : {TRADES_DATA_FILE}"
    )

    print(
        f"💾 OHLCV CSV          : {OHLCV_DATA_FILE}"
    )

    print(
        "============================================================\n"
    )


# ============================================================
# 28. AUTO COLLECTOR
# ============================================================

def log_auto_data(
    file_path: str = MARKET_DATA_FILE,
):

    ensure_data_files()

    print_system_config()

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

            except Exception as error:

                failed += 1

                print(
                    f"❌ Symbol error "
                    f"[{symbol}]: {error}"
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
            f"   Success          : {successful}"
        )

        print(
            f"   Failed           : {failed}"
        )

        print(
            f"   Cycle Time       : {elapsed:.2f}s"
        )

        print(
            f"   Total Snapshots  : {count}"
        )

        print(
            f"⏳ Waiting {CYCLE_DELAY}s..."
        )

        time.sleep(
            CYCLE_DELAY
        )


# ============================================================
# 29. TEST COLLECTOR
# ============================================================

def test_collector(
    symbol: str = "BTCUSDT",
):

    ensure_data_files()

    print(
        "\n============================================================"
    )

    print(
        f"🔎 TESTING {symbol}"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Market snapshot
    # --------------------------------------------------------

    data = collect_market_snapshot(
        symbol
    )

    if data is None:

        print(
            "❌ Market test failed."
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

        "data_quality",

        "scalp_signal_tf",
        "scalp_max_trade_minutes",

        "one_hour_signal_tf",
        "one_hour_max_trade_minutes",

        "four_hour_signal_tf",
        "four_hour_max_trade_minutes",
    ]

    for key in important_fields:

        print(
            f"{key:30} : "
            f"{data.get(key)}"
        )

    print(
        "=========================================\n"
    )

    # --------------------------------------------------------
    # OHLCV test
    # --------------------------------------------------------

    print(
        "📈 OHLCV / INDICATOR TEST"
    )

    print(
        "------------------------------------------------------------"
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

            continue

        latest = df.iloc[-1]

        print(
            f"\n✅ {interval}"
        )

        print(
            f"   Candles      : {len(df)}"
        )

        print(
            f"   Latest Close : "
            f"{latest['close']}"
        )

        print(
            f"   EMA20        : "
            f"{latest.get('ema_20', 0)}"
        )

        print(
            f"   EMA50        : "
            f"{latest.get('ema_50', 0)}"
        )

        print(
            f"   EMA200       : "
            f"{latest.get('ema_200', 0)}"
        )

        print(
            f"   RSI14        : "
            f"{latest.get('rsi_14', 0)}"
        )

        print(
            f"   ATR14        : "
            f"{latest.get('atr_14', 0)}"
        )

        print(
            f"   ATR%         : "
            f"{latest.get('atr_pct', 0)}"
        )

        print(
            f"   VWAP         : "
            f"{latest.get('vwap', 0)}"
        )

        print(
            f"   EMA Trend    : "
            f"{latest.get('ema_trend', 0)}"
        )

        print(
            f"   Volatility   : "
            f"{latest.get('volatility_20', 0)}"
        )

        print(
            f"   Volume Ratio : "
            f"{latest.get('volume_ratio', 0)}"
        )

    print(
        "\n============================================================"
    )

    print(
        "✅ TEST COMPLETE"
    )

    print(
        "============================================================"
    )


# ============================================================
# 30. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # TEST_MODE
    #
    # True:
    #   BTCUSDT one-shot test
    #
    # False:
    #   Continuous collector
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

    except Exception as error:

        print(
            f"\n❌ FATAL COLLECTOR ERROR: "
            f"{error}"
        )
