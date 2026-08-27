from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

MARKET_DATA_FILE = "market_data_log.csv"
MODEL_FILE = "xgboost_obi_model.pkl"

TRADE_HISTORY_FILE = "backtest_trade_history.csv"
SUMMARY_FILE = "backtest_summary.csv"

INITIAL_BALANCE = 100.0
TRADE_AMOUNT = 10.0

FEE_RATE = 0.0004
SLIPPAGE_RATE = 0.0001

MIN_CONFIDENCE = 0.55
MIN_OBI = 0.20

MIN_RR = 1.50
MAX_RR = 20.0

MAX_SL_PCT = 0.006

FOURIER_WINDOW = 30
FOURIER_KEEP_RATIO = 0.15

# ============================================================
# TIMEFRAME CONFIG
# ============================================================

TIMEFRAMES = {

    "SCALP": {
        "signal_minutes": 30,
        "trade_minutes": 15,
    },

    "1.5H": {
        "signal_minutes": 90,
        "trade_minutes": 90,
    },

    "1H": {
        "signal_minutes": 60,
        "trade_minutes": 1440,
    },

    "4H": {
        "signal_minutes": 240,
        "trade_minutes": 1440,
    },

}


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:

        return default


def normalize_signal(value):

    try:
        value = int(value)
    except Exception:
        return -1

    if value == 1:
        return 1

    if value == 0:
        return 0

    return -1


# ============================================================
# FOURIER
# ============================================================

def calculate_fourier_trend(
    prices: np.ndarray,
) -> float:

    prices = np.asarray(prices, dtype=float)

    if len(prices) < 15:
        return 0.0

    prices = prices[np.isfinite(prices)]

    if len(prices) < 15:
        return 0.0

    mean_price = np.mean(prices)

    centered = prices - mean_price

    fft_values = np.fft.fft(centered)

    n = len(fft_values)

    keep = max(
        1,
        int(n * FOURIER_KEEP_RATIO)
    )

    filtered = np.zeros_like(
        fft_values
    )

    filtered[:keep] = fft_values[:keep]

    if keep > 1:
        filtered[-keep:] = fft_values[-keep:]

    reconstructed = np.real(
        np.fft.ifft(filtered)
    )

    if len(reconstructed) < 2:
        return 0.0

    gradient = np.gradient(
        reconstructed
    )

    return safe_float(
        gradient[-1]
    )


# ============================================================
# FOURIER SL / TP
# ============================================================

def get_dynamic_fourier_sl_tp(
    prices: np.ndarray,
    entry: float,
    signal: int,
) -> Tuple[float, float]:

    entry = safe_float(entry)

    if entry <= 0:

        return 0.0, 0.0

    prices = np.asarray(
        prices,
        dtype=float
    )

    prices = prices[
        np.isfinite(prices)
    ]

    if len(prices) < 15:

        if signal == 1:

            sl = entry * (
                1.0 - MAX_SL_PCT
            )

            tp = entry * (
                1.0 + MAX_SL_PCT * MIN_RR
            )

        else:

            sl = entry * (
                1.0 + MAX_SL_PCT
            )

            tp = entry * (
                1.0 - MAX_SL_PCT * MIN_RR
            )

        return sl, tp

    mean_price = np.mean(prices)

    centered = prices - mean_price

    fft_values = np.fft.fft(
        centered
    )

    n = len(fft_values)

    keep = max(
        1,
        int(n * FOURIER_KEEP_RATIO)
    )

    filtered = np.zeros_like(
        fft_values
    )

    filtered[:keep] = fft_values[:keep]

    if keep > 1:
        filtered[-keep:] = fft_values[-keep:]

    trend_curve = np.real(
        np.fft.ifft(filtered)
    )

    cycle_low = float(
        np.min(trend_curve)
    )

    cycle_peak = float(
        np.max(trend_curve)
    )

    noise_buffer = (
        entry * 0.001
    )

    # ========================================================
    # LONG
    # ========================================================

    if signal == 1:

        fourier_distance = abs(
            entry - (
                mean_price + cycle_low
            )
        )

        risk_distance = min(
            fourier_distance + noise_buffer,
            entry * MAX_SL_PCT
        )

        risk_distance = max(
            risk_distance,
            entry * 0.001
        )

        sl = entry - risk_distance

        reward_distance = max(
            abs(
                (
                    mean_price
                    + cycle_peak
                ) - entry
            ),
            risk_distance * MIN_RR
        )

        tp = entry + reward_distance

    # ========================================================
    # SHORT
    # ========================================================

    else:

        fourier_distance = abs(
            (
                mean_price
                + cycle_peak
            ) - entry
        )

        risk_distance = min(
            fourier_distance + noise_buffer,
            entry * MAX_SL_PCT
        )

        risk_distance = max(
            risk_distance,
            entry * 0.001
        )

        sl = entry + risk_distance

        reward_distance = max(
            abs(
                entry
                - (
                    mean_price
                    + cycle_low
                )
            ),
            risk_distance * MIN_RR
        )

        tp = entry - reward_distance

    # ========================================================
    # FINAL RR PROTECTION
    # ========================================================

    risk = abs(
        entry - sl
    )

    if risk <= 0:

        return 0.0, 0.0

    reward = abs(
        tp - entry
    )

    required_reward = (
        risk * MIN_RR
    )

    if reward < required_reward:

        if signal == 1:

            tp = (
                entry
                + required_reward
            )

        else:

            tp = (
                entry
                - required_reward
            )

    rr = abs(
        tp - entry
    ) / risk

    if rr > MAX_RR:

        if signal == 1:

            tp = (
                entry
                + risk * MAX_RR
            )

        else:

            tp = (
                entry
                - risk * MAX_RR
            )

    return (
        round(float(sl), 8),
        round(float(tp), 8),
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Basic microstructure
    # --------------------------------------------------------

    df["top20_bid_sum"] = pd.to_numeric(
        df["top20_bid_sum"],
        errors="coerce"
    )

    df["top20_ask_sum"] = pd.to_numeric(
        df["top20_ask_sum"],
        errors="coerce"
    )

    df["current_price"] = pd.to_numeric(
        df["current_price"],
        errors="coerce"
    )

    df["obi_top20"] = pd.to_numeric(
        df["obi_top20"],
        errors="coerce"
    )

    df["spread"] = pd.to_numeric(
        df.get(
            "spread",
            0
        ),
        errors="coerce"
    )

    # --------------------------------------------------------
    # Ratio
    # --------------------------------------------------------

    df["bid_ask_ratio"] = (
        df["top20_bid_sum"]
        /
        (
            df["top20_ask_sum"]
            + 1e-9
        )
    )

    # --------------------------------------------------------
    # Depth
    # --------------------------------------------------------

    df["total_depth"] = (
        df["top20_bid_sum"]
        +
        df["top20_ask_sum"]
    )

    # --------------------------------------------------------
    # Sort first
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        df["_time"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True
        )

        df = df.sort_values(
            ["symbol", "_time"]
        )

    else:

        df["_time"] = pd.NaT

        df = df.sort_values(
            ["symbol"]
        )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    df["sma_20"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .transform(
            lambda x:
            x.rolling(
                20,
                min_periods=5
            ).mean()
        )
    )

    df["trend_signal"] = (
        df["current_price"]
        -
        df["sma_20"]
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["volatility_proxy"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .transform(
            lambda x:
            x.pct_change()
            .rolling(
                10,
                min_periods=5
            )
            .std()
        )
    )

    # --------------------------------------------------------
    # Hawkes-style intensity
    # --------------------------------------------------------

    df["hawkes_intensity"] = (
        df.groupby("symbol")[
            "obi_top20"
        ]
        .transform(
            lambda x:
            x.rolling(
                5,
                min_periods=2
            )
            .mean()
            .abs()
        )
    )

    # --------------------------------------------------------
    # Book pressure
    # --------------------------------------------------------

    df["book_pressure"] = (
        df["obi_top20"]
        *
        df["total_depth"]
    )

    # --------------------------------------------------------
    # Fourier trend
    #
    # IMPORTANT:
    # Only PAST data is used.
    # --------------------------------------------------------

    fourier_values = []

    for symbol, group in df.groupby(
        "symbol",
        sort=False
    ):

        prices = (
            group["current_price"]
            .values
        )

        values = []

        for i in range(
            len(group)
        ):

            start = max(
                0,
                i - FOURIER_WINDOW + 1
            )

            past_prices = prices[
                start:i + 1
            ]

            values.append(
                calculate_fourier_trend(
                    past_prices
                )
            )

        fourier_values.extend(
            values
        )

    df["FOURIER_TREND"] = (
        fourier_values
    )

    return df


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [

    "top20_bid_sum",
    "top20_ask_sum",
    "obi_top20",
    "spread",
    "bid_ask_ratio",
    "total_depth",
    "trend_signal",
    "volatility_proxy",
    "hawkes_intensity",
    "book_pressure",
    "FOURIER_TREND",

]


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_model(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:

    df = df.copy()

    available = [
        x
        for x in FEATURES
        if x in df.columns
    ]

    expected = getattr(
        model,
        "n_features_in_",
        len(available)
    )

    if expected < len(available):

        available = available[
            :expected
        ]

    elif expected > len(available):

        raise ValueError(
            "XGBoost model expects "
            f"{expected} features, "
            f"but only "
            f"{len(available)} "
            "are available."
        )

    df["ml_signal"] = (
        model.predict(
            df[available]
        )
    )

    try:

        probabilities = (
            model.predict_proba(
                df[available]
            )
        )

        df["confidence"] = (
            np.max(
                probabilities,
                axis=1
            )
        )

    except Exception:

        df["confidence"] = 1.0

    df["ml_signal"] = (
        df["ml_signal"]
        .apply(normalize_signal)
    )

    return df


# ============================================================
# TIME WINDOW
# ============================================================

def get_future_window(
    symbol_df: pd.DataFrame,
    entry_time: pd.Timestamp,
    duration_minutes: int,
) -> pd.DataFrame:

    if pd.isna(entry_time):

        return pd.DataFrame()

    end_time = (
        entry_time
        +
        pd.Timedelta(
            minutes=duration_minutes
        )
    )

    future = symbol_df[
        (
            symbol_df["_time"]
            > entry_time
        )
        &
        (
            symbol_df["_time"]
            <= end_time
        )
    ].copy()

    return future


# ============================================================
# BACKTEST ONE TIMEFRAME
# ============================================================

def backtest_timeframe(
    df: pd.DataFrame,
    timeframe_name: str,
    config: Dict,
) -> Tuple[List[Dict], Dict]:

    signal_minutes = (
        config["signal_minutes"]
    )

    trade_minutes = (
        config["trade_minutes"]
    )

    trades = []

    balance = INITIAL_BALANCE

    wins = 0
    losses = 0
    expired = 0
    skipped = 0

    symbols = (
        df["symbol"]
        .dropna()
        .unique()
    )

    last_trade_end = {}

    for symbol in symbols:

        symbol_df = (
            df[
                df["symbol"]
                == symbol
            ]
            .sort_values("_time")
            .reset_index(drop=True)
        )

        for i in range(
            len(symbol_df)
        ):

            row = symbol_df.iloc[i]

            entry_time = row["_time"]

            if pd.isna(entry_time):
                continue

            # ------------------------------------------------
            # Prevent overlapping trades
            # ------------------------------------------------

            if symbol in last_trade_end:

                if (
                    entry_time
                    <= last_trade_end[symbol]
                ):
                    continue

            signal = normalize_signal(
                row["ml_signal"]
            )

            confidence = safe_float(
                row["confidence"]
            )

            obi = safe_float(
                row["obi_top20"]
            )

            trend = safe_float(
                row["trend_signal"]
            )

            entry = safe_float(
                row["current_price"]
            )

            # ------------------------------------------------
            # Signal validation
            # ------------------------------------------------

            if signal not in (
                0,
                1
            ):

                skipped += 1
                continue

            if (
                confidence
                < MIN_CONFIDENCE
            ):

                skipped += 1
                continue

            if (
                abs(obi)
                < MIN_OBI
            ):

                skipped += 1
                continue

            # ------------------------------------------------
            # Direction + trend filter
            # ------------------------------------------------

            if signal == 1:

                if trend < 0:

                    skipped += 1
                    continue

                if obi < 0:

                    skipped += 1
                    continue

            else:

                if trend > 0:

                    skipped += 1
                    continue

                if obi > 0:

                    skipped += 1
                    continue

            if entry <= 0:

                skipped += 1
                continue

            # ------------------------------------------------
            # Signal window
            #
            # We require enough historical context.
            # ------------------------------------------------

            signal_start_time = (
                entry_time
                -
                pd.Timedelta(
                    minutes=signal_minutes
                )
            )

            historical = symbol_df[
                (
                    symbol_df["_time"]
                    < entry_time
                )
                &
                (
                    symbol_df["_time"]
                    >= signal_start_time
                )
            ]

            if len(historical) < 5:

                skipped += 1
                continue

            # ------------------------------------------------
            # Fourier SL/TP
            #
            # ONLY historical prices.
            # ------------------------------------------------

            past_prices = (
                symbol_df[
                    symbol_df["_time"]
                    <= entry_time
                ]
                ["current_price"]
                .tail(
                    FOURIER_WINDOW
                )
                .values
            )

            sl, tp = (
                get_dynamic_fourier_sl_tp(
                    past_prices,
                    entry,
                    signal
                )
            )

            if sl <= 0 or tp <= 0:

                skipped += 1
                continue

            risk = abs(
                entry - sl
            )

            reward = abs(
                tp - entry
            )

            if risk <= 0:

                skipped += 1
                continue

            rr = reward / risk

            if (
                rr < MIN_RR
                or rr > MAX_RR
            ):

                skipped += 1
                continue

            # ------------------------------------------------
            # Future trade window
            # ------------------------------------------------

            future = get_future_window(
                symbol_df,
                entry_time,
                trade_minutes
            )

            if future.empty:

                continue

            outcome = "EXPIRED"

            exit_price = (
                safe_float(
                    future.iloc[-1][
                        "current_price"
                    ],
                    entry
                )
            )

            exit_time = future.iloc[-1][
                "_time"
            ]

            pnl_pct = 0.0

            # ------------------------------------------------
            # Candle-by-candle
            # ------------------------------------------------

            for _, future_row in (
                future.iterrows()
            ):

                price = safe_float(
                    future_row[
                        "current_price"
                    ],
                    entry
                )

                # ------------------------------------------------
                # LONG
                # ------------------------------------------------

                if signal == 1:

                    if price >= tp:

                        outcome = "WIN"

                        exit_price = tp

                        exit_time = (
                            future_row[
                                "_time"
                            ]
                        )

                        gross_return = (
                            tp - entry
                        ) / entry

                        pnl_pct = (
                            gross_return
                            -
                            2 * FEE_RATE
                            -
                            SLIPPAGE_RATE
                        )

                        break

                    if price <= sl:

                        outcome = "LOSS"

                        exit_price = sl

                        exit_time = (
                            future_row[
                                "_time"
                            ]
                        )

                        gross_return = (
                            sl - entry
                        ) / entry

                        pnl_pct = (
                            gross_return
                            -
                            2 * FEE_RATE
                            -
                            SLIPPAGE_RATE
                        )

                        break

                # ------------------------------------------------
                # SHORT
                # ------------------------------------------------

                else:

                    if price <= tp:

                        outcome = "WIN"

                        exit_price = tp

                        exit_time = (
                            future_row[
                                "_time"
                            ]
                        )

                        gross_return = (
                            entry - tp
                        ) / entry

                        pnl_pct = (
                            gross_return
                            -
                            2 * FEE_RATE
                            -
                            SLIPPAGE_RATE
                        )

                        break

                    if price >= sl:

                        outcome = "LOSS"

                        exit_price = sl

                        exit_time = (
                            future_row[
                                "_time"
                            ]
                        )

                        gross_return = (
                            entry - sl
                        ) / entry

                        pnl_pct = (
                            gross_return
                            -
                            2 * FEE_RATE
                            -
                            SLIPPAGE_RATE
                        )

                        break

            # ------------------------------------------------
            # Expired
            # ------------------------------------------------

            if outcome == "EXPIRED":

                expired += 1

                exit_price = safe_float(
                    exit_price,
                    entry
                )

                if signal == 1:

                    gross_return = (
                        exit_price
                        - entry
                    ) / entry

                else:

                    gross_return = (
                        entry
                        - exit_price
                    ) / entry

                pnl_pct = (
                    gross_return
                    -
                    2 * FEE_RATE
                    -
                    SLIPPAGE_RATE
                )

            elif outcome == "WIN":

                wins += 1

            elif outcome == "LOSS":

                losses += 1

            # ------------------------------------------------
            # PNL
            # ------------------------------------------------

            trade_pnl = (
                TRADE_AMOUNT
                * pnl_pct
            )

            balance += trade_pnl

            # ------------------------------------------------
            # Prevent overlapping trades
            # ------------------------------------------------

            last_trade_end[
                symbol
            ] = exit_time

            # ------------------------------------------------
            # Save trade
            # ------------------------------------------------

            trades.append({

                "timeframe":
                    timeframe_name,

                "symbol":
                    symbol,

                "signal":
                    "LONG"
                    if signal == 1
                    else "SHORT",

                "entry_time":
                    entry_time,

                "exit_time":
                    exit_time,

                "entry":
                    entry,

                "exit":
                    exit_price,

                "stop_loss":
                    sl,

                "take_profit":
                    tp,

                "rr":
                    rr,

                "confidence":
                    confidence,

                "obi":
                    obi,

                "trend":
                    trend,

                "signal_window_minutes":
                    signal_minutes,

                "trade_window_minutes":
                    trade_minutes,

                "outcome":
                    outcome,

                "pnl_pct":
                    pnl_pct * 100,

                "pnl_usdt":
                    trade_pnl,

                "balance":
                    balance,

            })

    executed = (
        wins
        +
        losses
        +
        expired
    )

    closed_wins = wins

    win_rate = (
        wins
        /
        (wins + losses)
        * 100
        if (wins + losses) > 0
        else 0.0
    )

    total_return = (
        (
            balance
            -
            INITIAL_BALANCE
        )
        /
        INITIAL_BALANCE
        * 100
    )

    summary = {

        "timeframe":
            timeframe_name,

        "signal_window_minutes":
            signal_minutes,

        "trade_window_minutes":
            trade_minutes,

        "executed_trades":
            executed,

        "wins":
            wins,

        "losses":
            losses,

        "expired":
            expired,

        "skipped":
            skipped,

        "win_rate":
            win_rate,

        "net_profit":
            balance
            -
            INITIAL_BALANCE,

        "return_pct":
            total_return,

        "final_balance":
            balance,

    }

    return trades, summary


# ============================================================
# MAIN BACKTEST
# ============================================================

def run_filtered_trend_backtest():

    print()
    print("=" * 72)
    print(
        "🧪 ZIA RESEARCH MULTI-TIMEFRAME BACKTEST"
    )
    print("=" * 72)

    print(
        "SCALP : 30m signal / 15m trade"
    )

    print(
        "1.5H  : 90m signal / 90m trade"
    )

    print(
        "1H    : 60m signal / 24h max trade"
    )

    print(
        "4H    : 240m signal / 24h max trade"
    )

    print("=" * 72)
    print()

    # --------------------------------------------------------
    # Load market data
    # --------------------------------------------------------

    if not os.path.exists(
        MARKET_DATA_FILE
    ):

        print(
            f"❌ Missing {MARKET_DATA_FILE}"
        )

        return

    if not os.path.exists(
        MODEL_FILE
    ):

        print(
            f"❌ Missing {MODEL_FILE}"
        )

        return

    try:

        df = pd.read_csv(
            MARKET_DATA_FILE
        )

        model = joblib.load(
            MODEL_FILE
        )

    except Exception as error:

        print(
            f"❌ File loading error: {error}"
        )

        return

    if df.empty:

        print(
            "❌ Market data is empty."
        )

        return

    required_columns = [

        "symbol",
        "current_price",
        "top20_bid_sum",
        "top20_ask_sum",
        "obi_top20",

    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        print(
            "❌ Missing columns:"
        )

        for col in missing:
            print(
                f"   - {col}"
            )

        return

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print(
        "⚙️ Building features..."
    )

    df = build_features(
        df
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "symbol",
            "current_price",
            "top20_bid_sum",
            "top20_ask_sum",
            "obi_top20",
            "_time",
        ]
    ).copy()

    if df.empty:

        print(
            "❌ No valid rows after cleaning."
        )

        return

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print(
        "🤖 Running XGBoost predictions..."
    )

    try:

        df = predict_model(
            df,
            model
        )

    except Exception as error:

        print(
            "❌ Model prediction error:"
        )

        print(error)

        return

    # --------------------------------------------------------
    # Run all timeframes
    # --------------------------------------------------------

    all_trades = []
    summaries = []

    for timeframe_name, config in (
        TIMEFRAMES.items()
    ):

        print()
        print(
            "-" * 72
        )

        print(
            f"⏱️ BACKTESTING {timeframe_name}"
        )

        print(
            f"Signal Window : "
            f"{config['signal_minutes']} minutes"
        )

        print(
            f"Trade Window  : "
            f"{config['trade_minutes']} minutes"
        )

        print(
            "-" * 72
        )

        trades, summary = (
            backtest_timeframe(
                df,
                timeframe_name,
                config
            )
        )

        all_trades.extend(
            trades
        )

        summaries.append(
            summary
        )

        print(
            f"Trades     : "
            f"{summary['executed_trades']}"
        )

        print(
            f"Wins       : "
            f"{summary['wins']}"
        )

        print(
            f"Losses     : "
            f"{summary['losses']}"
        )

        print(
            f"Expired    : "
            f"{summary['expired']}"
        )

        print(
            f"Win Rate   : "
            f"{summary['win_rate']:.2f}%"
        )

        print(
            f"Net Profit : "
            f"${summary['net_profit']:.4f}"
        )

        print(
            f"Return     : "
            f"{summary['return_pct']:.2f}%"
        )

        print(
            f"Final Bal. : "
            f"${summary['final_balance']:.4f}"
        )

    # --------------------------------------------------------
    # Save trade history
    # --------------------------------------------------------

    if all_trades:

        trades_df = pd.DataFrame(
            all_trades
        )

        trades_df.to_csv(
            TRADE_HISTORY_FILE,
            index=False
        )

        print()
        print(
            f"💾 Trade history saved: "
            f"{TRADE_HISTORY_FILE}"
        )

    else:

        print()
        print(
            "⚠️ No trades generated."
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        f"💾 Summary saved: "
        f"{SUMMARY_FILE}"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "📊 FINAL MULTI-TIMEFRAME REPORT"
    )
    print("=" * 72)

    for summary in summaries:

        print()

        print(
            f"⏱️ {summary['timeframe']}"
        )

        print(
            f"   Signal Window : "
            f"{summary['signal_window_minutes']}m"
        )

        print(
            f"   Trade Window  : "
            f"{summary['trade_window_minutes']}m"
        )

        print(
            f"   Trades        : "
            f"{summary['executed_trades']}"
        )

        print(
            f"   Wins          : "
            f"{summary['wins']}"
        )

        print(
            f"   Losses        : "
            f"{summary['losses']}"
        )

        print(
            f"   Expired       : "
            f"{summary['expired']}"
        )

        print(
            f"   Win Rate      : "
            f"{summary['win_rate']:.2f}%"
        )

        print(
            f"   Net Profit    : "
            f"${summary['net_profit']:.4f}"
        )

        print(
            f"   Return        : "
            f"{summary['return_pct']:.2f}%"
        )

        print(
            f"   Final Balance : "
            f"${summary['final_balance']:.4f}"
        )

    print()
    print("=" * 72)
    print(
        "✅ BACKTEST COMPLETE"
    )
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_filtered_trend_backtest()
