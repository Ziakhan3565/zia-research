import time
import datetime
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

COINS_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "LTCUSDT", "BCHUSDT", "APTUSDT", "TRXUSDT",
    "SHIBUSDT", "UNIUSDT", "ATOMUSDT", "SUIUSDT", "INJUSDT",
    "ICPUSDT"
]

BINANCE_BASE_URL = "https://api.binance.com"

# Order-book depth
DEPTH_LIMIT = 50

# How many levels are used for OBI
TOP20 = 20
TOP50 = 50

# Collector interval
CYCLE_SLEEP = 10

# Network timeout
REQUEST_TIMEOUT = 5

# Future movement research threshold
MOVE_THRESHOLD = 0.004       # 0.40%

# Research horizons
HORIZON_1_MIN = 60
HORIZON_5_MIN = 300
HORIZON_15_MIN = 900

# Files
RAW_FILE = "market_data_log.csv"
TRADE_LABEL_FILE = "trade_labels.csv"


# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "ResearchLab-AutoCollector/1.0"
})


# ============================================================
# MEMORY FOR PREVIOUS ORDER BOOK
# ============================================================

previous_book = {}

# Previous prices for each coin
price_history = defaultdict(
    lambda: deque(maxlen=120)
)


# ============================================================
# SAFE REQUEST
# ============================================================

def safe_get(url, params=None):
    """
    Safe Binance REST request.
    """

    try:
        response = session.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict) and "code" in data:
            print("⚠️ Binance API:", data)
            return None

        return data

    except Exception as e:
        print(f"⚠️ Request error: {e}")
        return None


# ============================================================
# ORDER BOOK
# ============================================================

def fetch_order_book(symbol, depth_limit=DEPTH_LIMIT):
    """
    Fetch Binance order book.

    Returns:
        bids: [[price, qty], ...]
        asks: [[price, qty], ...]
    """

    url = f"{BINANCE_BASE_URL}/api/v3/depth"

    data = safe_get(
        url,
        params={
            "symbol": symbol,
            "limit": depth_limit
        }
    )

    if data is None:
        return None, None

    try:

        bids = np.array(
            data["bids"],
            dtype=float
        )

        asks = np.array(
            data["asks"],
            dtype=float
        )

        if len(bids) == 0 or len(asks) == 0:
            return None, None

        return bids, asks

    except Exception:
        return None, None


# ============================================================
# CANDLE DATA
# ============================================================

def fetch_latest_candles(symbol, interval="1m", limit=30):
    """
    Fetch recent Binance candles.

    Used for:
        volatility
        returns
        volume
        price movement
    """

    url = f"{BINANCE_BASE_URL}/api/v3/klines"

    data = safe_get(
        url,
        params={
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
    )

    if data is None:
        return None

    try:

        rows = []

        for candle in data:

            rows.append({
                "open_time": candle[0],
                "Open": float(candle[1]),
                "High": float(candle[2]),
                "Low": float(candle[3]),
                "Close": float(candle[4]),
                "Volume": float(candle[5]),
                "close_time": candle[6]
            })

        return pd.DataFrame(rows)

    except Exception:
        return None


# ============================================================
# BASIC MATH
# ============================================================

def calculate_obi(bid_volume, ask_volume):
    """
    Standard normalized Order Book Imbalance:

        OBI = (Bid - Ask) / (Bid + Ask)
    """

    total = bid_volume + ask_volume

    if total <= 0:
        return 0.0

    return float(
        (bid_volume - ask_volume) / total
    )


def calculate_spread(bids, asks):
    """
    Absolute and percentage spread.
    """

    best_bid = float(bids[0, 0])
    best_ask = float(asks[0, 0])

    mid = (best_bid + best_ask) / 2.0

    spread = best_ask - best_bid

    spread_pct = (
        spread / mid
        if mid > 0
        else 0.0
    )

    return spread, spread_pct, mid


# ============================================================
# ORDER BOOK DEPTH FEATURES
# ============================================================

def calculate_depth_features(bids, asks):
    """
    Calculate Top 20 and Top 50 order-book features.
    """

    bid20 = float(
        np.sum(bids[:TOP20, 1])
    )

    ask20 = float(
        np.sum(asks[:TOP20, 1])
    )

    bid50 = float(
        np.sum(bids[:TOP50, 1])
    )

    ask50 = float(
        np.sum(asks[:TOP50, 1])
    )

    obi20 = calculate_obi(
        bid20,
        ask20
    )

    obi50 = calculate_obi(
        bid50,
        ask50
    )

    return {
        "top20_bid_volume": bid20,
        "top20_ask_volume": ask20,
        "top50_bid_volume": bid50,
        "top50_ask_volume": ask50,
        "obi_top20": obi20,
        "obi_top50": obi50
    }


# ============================================================
# ORDER FLOW IMBALANCE
# ============================================================

def calculate_ofi(symbol, bids, asks):
    """
    Snapshot-based OFI approximation.

    Important:
    True exchange-level OFI is ideally calculated from
    order-book update events.

    This function compares consecutive snapshots.
    """

    if symbol not in previous_book:

        previous_book[symbol] = {
            "bids": bids.copy(),
            "asks": asks.copy()
        }

        return 0.0, 0.0, 0.0

    old_bids = previous_book[symbol]["bids"]
    old_asks = previous_book[symbol]["asks"]

    old_bid_map = {
        round(float(price), 8): float(qty)
        for price, qty in old_bids
    }

    old_ask_map = {
        round(float(price), 8): float(qty)
        for price, qty in old_asks
    }

    bid_flow = 0.0
    ask_flow = 0.0

    # Current bid changes
    for price, qty in bids:

        price = round(float(price), 8)
        qty = float(qty)

        old_qty = old_bid_map.get(price, 0.0)

        bid_flow += qty - old_qty

    # Current ask changes
    for price, qty in asks:

        price = round(float(price), 8)
        qty = float(qty)

        old_qty = old_ask_map.get(price, 0.0)

        ask_flow += qty - old_qty

    ofi = bid_flow - ask_flow

    total_flow = abs(bid_flow) + abs(ask_flow) + 1e-8

    normalized_ofi = ofi / total_flow

    previous_book[symbol] = {
        "bids": bids.copy(),
        "asks": asks.copy()
    }

    return (
        float(ofi),
        float(bid_flow),
        float(ask_flow)
    )


# ============================================================
# ORDER BOOK PRESSURE
# ============================================================

def calculate_pressure(bid_volume, ask_volume):
    """
    Pressure normalized to [-1, +1].
    """

    total = bid_volume + ask_volume

    if total <= 0:
        return 0.0

    return float(
        (bid_volume - ask_volume) / total
    )


# ============================================================
# VOLATILITY
# ============================================================

def calculate_volatility(df):
    """
    Realized volatility from 1-minute returns.
    """

    if df is None or len(df) < 10:
        return 0.0

    returns = (
        df["Close"]
        .pct_change()
        .dropna()
    )

    if len(returns) < 5:
        return 0.0

    return float(
        returns.std()
    )


# ============================================================
# SHORT TERM MOMENTUM
# ============================================================

def calculate_momentum(df):
    """
    Short-term momentum.
    """

    if df is None or len(df) < 6:
        return 0.0

    old_price = float(
        df["Close"].iloc[-6]
    )

    current_price = float(
        df["Close"].iloc[-1]
    )

    if old_price <= 0:
        return 0.0

    return float(
        (current_price - old_price) / old_price
    )


# ============================================================
# VOLUME PRESSURE
# ============================================================

def calculate_volume_change(df):
    """
    Current volume vs average recent volume.
    """

    if df is None or len(df) < 10:
        return 1.0

    recent = float(
        df["Volume"].iloc[-1]
    )

    average = float(
        df["Volume"].iloc[-10:-1].mean()
    )

    if average <= 0:
        return 1.0

    return float(
        recent / average
    )


# ============================================================
# PRICE VELOCITY
# ============================================================

def calculate_velocity(symbol, current_price):
    """
    Price velocity based on previous snapshot.
    """

    history = price_history[symbol]

    if len(history) == 0:
        history.append(
            (time.time(), current_price)
        )

        return 0.0

    old_time, old_price = history[-1]

    now = time.time()

    delta_time = now - old_time

    if delta_time <= 0 or old_price <= 0:
        velocity = 0.0

    else:

        velocity = (
            (current_price - old_price)
            / old_price
        ) / delta_time

    history.append(
        (now, current_price)
    )

    return float(velocity)


# ============================================================
# FUTURE MOVE CLASSIFICATION
# ============================================================

def classify_move(current_price, future_price):
    """
    0.40% movement classifier.

    +1 = UP >= 0.40%
    -1 = DOWN <= -0.40%
     0 = small/no move
    """

    if current_price <= 0 or future_price <= 0:
        return 0

    future_return = (
        future_price - current_price
    ) / current_price

    if future_return >= MOVE_THRESHOLD:
        return 1

    if future_return <= -MOVE_THRESHOLD:
        return -1

    return 0


# ============================================================
# RISK / REWARD
# ============================================================

def calculate_rr_levels(
    entry_price,
    direction,
    stop_distance_pct=0.002
):
    """
    Research target calculation.

    Stop = 0.20%
    Target 1 = 1:2
    Target 2 = 1:3
    """

    stop_distance = (
        entry_price * stop_distance_pct
    )

    if direction == 1:

        stop = entry_price - stop_distance

        target_1 = (
            entry_price +
            stop_distance * 2
        )

        target_2 = (
            entry_price +
            stop_distance * 3
        )

    elif direction == -1:

        stop = entry_price + stop_distance

        target_1 = (
            entry_price -
            stop_distance * 2
        )

        target_2 = (
            entry_price -
            stop_distance * 3
        )

    else:

        stop = np.nan
        target_1 = np.nan
        target_2 = np.nan

    return stop, target_1, target_2


# ============================================================
# BUILD DATA POINT
# ============================================================

def build_data_point(symbol, bids, asks):

    if bids is None or asks is None:
        return None

    if len(bids) < TOP20 or len(asks) < TOP20:
        return None

    # --------------------------------------------------------
    # Price / spread
    # --------------------------------------------------------

    spread, spread_pct, mid_price = calculate_spread(
        bids,
        asks
    )

    # --------------------------------------------------------
    # Depth
    # --------------------------------------------------------

    depth = calculate_depth_features(
        bids,
        asks
    )

    # --------------------------------------------------------
    # OFI
    # --------------------------------------------------------

    ofi, bid_flow, ask_flow = calculate_ofi(
        symbol,
        bids,
        asks
    )

    # --------------------------------------------------------
    # Candles
    # --------------------------------------------------------

    candles = fetch_latest_candles(
        symbol,
        interval="1m",
        limit=30
    )

    volatility = calculate_volatility(
        candles
    )

    momentum = calculate_momentum(
        candles
    )

    volume_ratio = calculate_volume_change(
        candles
    )

    velocity = calculate_velocity(
        symbol,
        mid_price
    )

    # --------------------------------------------------------
    # Pressure
    # --------------------------------------------------------

    pressure20 = calculate_pressure(
        depth["top20_bid_volume"],
        depth["top20_ask_volume"]
    )

    pressure50 = calculate_pressure(
        depth["top50_bid_volume"],
        depth["top50_ask_volume"]
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = datetime.datetime.now(
        datetime.timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------------
    # Data point
    # --------------------------------------------------------

    data_point = {

        "timestamp": timestamp,

        "symbol": symbol,

        # PRICE
        "current_price": mid_price,

        "best_bid": float(bids[0, 0]),
        "best_ask": float(asks[0, 0]),

        # SPREAD
        "spread": spread,
        "spread_pct": spread_pct,

        # TOP 20
        "top20_bid_volume":
            depth["top20_bid_volume"],

        "top20_ask_volume":
            depth["top20_ask_volume"],

        "obi_top20":
            depth["obi_top20"],

        "pressure_top20":
            pressure20,

        # TOP 50
        "top50_bid_volume":
            depth["top50_bid_volume"],

        "top50_ask_volume":
            depth["top50_ask_volume"],

        "obi_top50":
            depth["obi_top50"],

        "pressure_top50":
            pressure50,

        # ORDER FLOW
        "ofi":
            ofi,

        "bid_flow":
            bid_flow,

        "ask_flow":
            ask_flow,

        # MARKET STATE
        "volatility":
            volatility,

        "momentum":
            momentum,

        "volume_ratio":
            volume_ratio,

        "velocity":
            velocity,

        # ----------------------------------------------------
        # RAW LABEL PLACEHOLDERS
        # These are filled later using future price.
        # ----------------------------------------------------

        "future_return_1m":
            np.nan,

        "future_return_5m":
            np.nan,

        "future_return_15m":
            np.nan,

        "move_1m_040":
            np.nan,

        "move_5m_040":
            np.nan,

        "move_15m_040":
            np.nan,

        # ----------------------------------------------------
        # RR research fields
        # ----------------------------------------------------

        "rr_1_to_2_long":
            np.nan,

        "rr_1_to_3_long":
            np.nan,

        "rr_1_to_2_short":
            np.nan,

        "rr_1_to_3_short":
            np.nan,

        "research_signal":
            "DATA_ONLY"
    }

    return data_point


# ============================================================
# SAVE DATA
# ============================================================

def save_data_point(
    data_point,
    file_path=RAW_FILE
):

    if data_point is None:
        return

    df_new = pd.DataFrame(
        [data_point]
    )

    if not os.path.isfile(file_path):

        df_new.to_csv(
            file_path,
            index=False
        )

    else:

        df_new.to_csv(
            file_path,
            mode="a",
            header=False,
            index=False
        )


# ============================================================
# GENERATE FUTURE LABELS
# ============================================================

def generate_future_labels(
    file_path=RAW_FILE
):
    """
    Uses historical collected prices to generate
    future movement labels.

    This does NOT create fake labels.
    """

    if not os.path.isfile(file_path):
        return

    try:

        df = pd.read_csv(
            file_path
        )

        if len(df) < 2:
            return

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

        df = df.sort_values(
            ["symbol", "timestamp"]
        )

        # ----------------------------------------------------
        # Future returns by time interpolation
        # ----------------------------------------------------

        for symbol in df["symbol"].unique():

            idx = df[
                df["symbol"] == symbol
            ].index

            symbol_df = df.loc[idx].copy()

            symbol_df = symbol_df.sort_values(
                "timestamp"
            )

            timestamps = (
                symbol_df["timestamp"]
                .astype("int64")
                .values
            )

            prices = (
                symbol_df["current_price"]
                .astype(float)
                .values
            )

            for horizon, col_return, col_label in [

                (
                    HORIZON_1_MIN,
                    "future_return_1m",
                    "move_1m_040"
                ),

                (
                    HORIZON_5_MIN,
                    "future_return_5m",
                    "move_5m_040"
                ),

                (
                    HORIZON_15_MIN,
                    "future_return_15m",
                    "move_15m_040"
                )

            ]:

                horizon_ns = (
                    horizon * 1_000_000_000
                )

                future_returns = np.full(
                    len(symbol_df),
                    np.nan
                )

                labels = np.full(
                    len(symbol_df),
                    np.nan
                )

                for i in range(
                    len(symbol_df)
                ):

                    target_time = (
                        timestamps[i]
                        + horizon_ns
                    )

                    future_idx = np.searchsorted(
                        timestamps,
                        target_time,
                        side="left"
                    )

                    if future_idx >= len(prices):
                        continue

                    current_price = prices[i]
                    future_price = prices[future_idx]

                    if current_price <= 0:
                        continue

                    future_return = (
                        future_price -
                        current_price
                    ) / current_price

                    future_returns[i] = future_return

                    labels[i] = classify_move(
                        current_price,
                        future_price
                    )

                df.loc[
                    symbol_df.index,
                    col_return
                ] = future_returns

                df.loc[
                    symbol_df.index,
                    col_label
                ] = labels

        # ----------------------------------------------------
        # Save updated data
        # ----------------------------------------------------

        df.to_csv(
            file_path,
            index=False
        )

    except Exception as e:

        print(
            f"⚠️ Label generation error: {e}"
        )


# ============================================================
# MAIN COLLECTOR
# ============================================================

def log_auto_data(
    file_path=RAW_FILE
):

    print(
        "\n"
        "====================================================\n"
        "🚀 RESEARCH LAB AUTO COLLECTOR\n"
        "====================================================\n"
        f"Coins          : {len(COINS_LIST)}\n"
        f"Depth          : Top {DEPTH_LIMIT}\n"
        f"OBI            : Top {TOP20} + Top {TOP50}\n"
        f"Move threshold : {MOVE_THRESHOLD * 100:.2f}%\n"
        f"RR             : 1:2 + 1:3\n"
        f"Raw file       : {file_path}\n"
        "====================================================\n"
    )

    count = 0

    while True:

        cycle_start = time.time()

        for symbol in COINS_LIST:

            try:

                # ------------------------------------------------
                # ORDER BOOK
                # ------------------------------------------------

                bids, asks = fetch_order_book(
                    symbol,
                    DEPTH_LIMIT
                )

                if bids is None or asks is None:

                    print(
                        f"❌ No order book: {symbol}"
                    )

                    continue

                # ------------------------------------------------
                # BUILD DATA
                # ------------------------------------------------

                data_point = build_data_point(
                    symbol,
                    bids,
                    asks
                )

                if data_point is None:
                    continue

                # ------------------------------------------------
                # SAVE
                # ------------------------------------------------

                save_data_point(
                    data_point,
                    file_path
                )

                count += 1

                print(
                    f"✅ [{count}] "
                    f"{symbol} | "
                    f"Price: {data_point['current_price']:.6f} | "
                    f"OBI20: {data_point['obi_top20']:+.3f} | "
                    f"OBI50: {data_point['obi_top50']:+.3f} | "
                    f"OFI: {data_point['ofi']:+.4f} | "
                    f"Vol: {data_point['volatility']:.6f}"
                )

            except Exception as e:

                print(
                    f"⚠️ {symbol} error: {e}"
                )

        # --------------------------------------------------------
        # UPDATE FUTURE LABELS
        # --------------------------------------------------------

        try:

            generate_future_labels(
                file_path
            )

        except Exception as e:

            print(
                f"⚠️ Future-label update failed: {e}"
            )

        # --------------------------------------------------------
        # CYCLE TIME
        # --------------------------------------------------------

        elapsed = (
            time.time() -
            cycle_start
        )

        sleep_time = max(
            1,
            CYCLE_SLEEP - elapsed
        )

        print(
            "\n"
            f"🔄 Cycle complete | "
            f"{elapsed:.2f}s | "
            f"Sleeping {sleep_time:.2f}s\n"
        )

        time.sleep(
            sleep_time
        )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    try:

        log_auto_data()

    except KeyboardInterrupt:

        print(
            "\n🛑 Collector stopped by user."
        )
