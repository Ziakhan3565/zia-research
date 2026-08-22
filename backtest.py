import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "market_data_log.csv"

INITIAL_BALANCE = 100.0
TRADE_AMOUNT = 10.0

# Trading costs
FEE_RATE = 0.0004       # 0.04% per side
SLIPPAGE_RATE = 0.0001  # 0.01% estimated slippage per side

# Signal filters
MIN_CONFIDENCE = 0.55
MIN_OBI = 0.20
MIN_OFI_NORMALIZED = 0.05

# Minimum meaningful move
MIN_MOVE = 0.004        # 0.40%

# Risk
STOP_LOSS_PCT = 0.002   # 0.20%

# RR targets
TP_1_2 = STOP_LOSS_PCT * 2.0   # 0.40%
TP_1_3 = STOP_LOSS_PCT * 3.0   # 0.60%

# Maximum holding periods
HORIZON_1M = 1
HORIZON_5M = 5
HORIZON_15M = 15


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(DATA_FILE):
        print(f"❌ File not found: {DATA_FILE}")
        return None

    try:

        df = pd.read_csv(DATA_FILE)

    except Exception as e:

        print(f"❌ CSV loading error: {e}")
        return None

    required = [
        "timestamp",
        "symbol",
        "current_price",
        "obi_top20",
        "obi_top50"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        print("\n❌ Missing columns:")
        for col in missing:
            print(f"   - {col}")

        print("\nYour collector must be updated first.")
        return None

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    df["current_price"] = pd.to_numeric(
        df["current_price"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "symbol",
            "current_price"
        ]
    )

    df = df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Bid/Ask ratio
    # --------------------------------------------------------

    if (
        "top20_bid_volume" in df.columns and
        "top20_ask_volume" in df.columns
    ):

        df["bid_ask_ratio"] = (
            df["top20_bid_volume"] /
            (
                df["top20_ask_volume"] +
                1e-12
            )
        )

        df["total_depth"] = (
            df["top20_bid_volume"] +
            df["top20_ask_volume"]
        )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    df["sma_20"] = (
        df.groupby("symbol")["current_price"]
        .transform(
            lambda x:
            x.rolling(
                20,
                min_periods=10
            ).mean()
        )
    )

    df["trend_signal"] = (
        df["current_price"] -
        df["sma_20"]
    )

    df["trend_pct"] = (
        df["trend_signal"] /
        (df["sma_20"] + 1e-12)
    )

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    df["return_1"] = (
        df.groupby("symbol")["current_price"]
        .pct_change(1)
    )

    df["return_5"] = (
        df.groupby("symbol")["current_price"]
        .pct_change(5)
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["rolling_volatility"] = (
        df.groupby("symbol")["return_1"]
        .transform(
            lambda x:
            x.rolling(
                20,
                min_periods=10
            ).std()
        )
    )

    return df


# ============================================================
# OFI NORMALIZATION
# ============================================================

def normalize_ofi(df):

    df = df.copy()

    if "ofi" not in df.columns:
        df["ofi_normalized"] = 0.0
        return df

    if (
        "top20_bid_volume" in df.columns and
        "top20_ask_volume" in df.columns
    ):

        depth = (
            df["top20_bid_volume"] +
            df["top20_ask_volume"] +
            1e-12
        )

        df["ofi_normalized"] = (
            df["ofi"] / depth
        )

        df["ofi_normalized"] = (
            df["ofi_normalized"]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0.0)
            .clip(-1, 1)
        )

    else:

        df["ofi_normalized"] = 0.0

    return df


# ============================================================
# SIGNAL CREATION
# ============================================================

def create_signals(df):

    df = df.copy()

    df["signal"] = 0

    # --------------------------------------------------------
    # Base OBI direction
    # --------------------------------------------------------

    long_condition = (
        (df["obi_top20"] >= MIN_OBI) &
        (df["obi_top50"] >= 0)
    )

    short_condition = (
        (df["obi_top20"] <= -MIN_OBI) &
        (df["obi_top50"] <= 0)
    )

    df.loc[
        long_condition,
        "signal"
    ] = 1

    df.loc[
        short_condition,
        "signal"
    ] = -1

    # --------------------------------------------------------
    # OFI confirmation
    # --------------------------------------------------------

    long_ofi = (
        df["ofi_normalized"] >=
        MIN_OFI_NORMALIZED
    )

    short_ofi = (
        df["ofi_normalized"] <=
        -MIN_OFI_NORMALIZED
    )

    # If OFI exists, require confirmation
    if "ofi" in df.columns:

        df.loc[
            (df["signal"] == 1) &
            (~long_ofi),
            "signal"
        ] = 0

        df.loc[
            (df["signal"] == -1) &
            (~short_ofi),
            "signal"
        ] = 0

    # --------------------------------------------------------
    # Trend filter
    # --------------------------------------------------------

    df.loc[
        (df["signal"] == 1) &
        (df["trend_pct"] <= 0),
        "signal"
    ] = 0

    df.loc[
        (df["signal"] == -1) &
        (df["trend_pct"] >= 0),
        "signal"
    ] = 0

    return df


# ============================================================
# FUTURE PRICE
# ============================================================

def get_future_prices(
    symbol_df,
    current_index,
    horizon
):

    start = current_index + 1

    end = min(
        current_index + 1 + horizon,
        len(symbol_df)
    )

    if start >= end:
        return None

    return symbol_df.iloc[
        start:end
    ]["current_price"].values


# ============================================================
# CHECK 0.40% MOVE
# ============================================================

def has_minimum_move(
    entry,
    future_prices,
    direction
):

    if future_prices is None:
        return False

    if len(future_prices) == 0:
        return False

    if direction == 1:

        max_price = np.max(
            future_prices
        )

        move = (
            max_price - entry
        ) / entry

    else:

        min_price = np.min(
            future_prices
        )

        move = (
            entry - min_price
        ) / entry

    return move >= MIN_MOVE


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    symbol_df,
    local_index,
    direction,
    rr
):

    entry = float(
        symbol_df.iloc[
            local_index
        ]["current_price"]
    )

    if entry <= 0:
        return None

    stop_distance = (
        entry *
        STOP_LOSS_PCT
    )

    if direction == 1:

        # Long
        effective_entry = (
            entry *
            (
                1 +
                SLIPPAGE_RATE
            )
        )

        stop_price = (
            effective_entry -
            stop_distance
        )

        if rr == 2:

            target_price = (
                effective_entry +
                stop_distance * 2
            )

        else:

            target_price = (
                effective_entry +
                stop_distance * 3
            )

    else:

        # Short
        effective_entry = (
            entry *
            (
                1 -
                SLIPPAGE_RATE
            )
        )

        stop_price = (
            effective_entry +
            stop_distance
        )

        if rr == 2:

            target_price = (
                effective_entry -
                stop_distance * 2
            )

        else:

            target_price = (
                effective_entry -
                stop_distance * 3
            )

    # --------------------------------------------------------
    # Look forward 15 bars
    # --------------------------------------------------------

    future = symbol_df.iloc[
        local_index + 1:
        local_index + 1 + HORIZON_15M
    ]

    if len(future) == 0:
        return None

    outcome = "TIMEOUT"

    exit_price = float(
        future["current_price"].iloc[-1]
    )

    exit_index = (
        local_index +
        len(future)
    )

    # --------------------------------------------------------
    # Bar-by-bar execution
    # --------------------------------------------------------

    for i, row in future.iterrows():

        price = float(
            row["current_price"]
        )

        if direction == 1:

            # Stop first
            if price <= stop_price:

                outcome = "LOSS"
                exit_price = stop_price

                exit_index = i

                break

            if price >= target_price:

                outcome = "WIN"
                exit_price = target_price

                exit_index = i

                break

        else:

            # Stop first
            if price >= stop_price:

                outcome = "LOSS"
                exit_price = stop_price

                exit_index = i

                break

            if price <= target_price:

                outcome = "WIN"
                exit_price = target_price

                exit_index = i

                break

    # --------------------------------------------------------
    # Timeout is NOT automatically a win/loss
    # --------------------------------------------------------

    if outcome == "TIMEOUT":

        if direction == 1:

            pnl_pct = (
                exit_price -
                effective_entry
            ) / effective_entry

        else:

            pnl_pct = (
                effective_entry -
                exit_price
            ) / effective_entry

    elif outcome == "WIN":

        pnl_pct = (
            TP_1_2
            if rr == 2
            else TP_1_3
        )

    else:

        pnl_pct = -STOP_LOSS_PCT

    # --------------------------------------------------------
    # Trading costs
    # --------------------------------------------------------

    total_cost = (
        2 *
        FEE_RATE
    )

    pnl_after_cost = (
        pnl_pct -
        total_cost
    )

    return {
        "outcome": outcome,
        "entry": effective_entry,
        "exit": exit_price,
        "pnl_pct": pnl_after_cost,
        "exit_index": exit_index
    }


# ============================================================
# RUN BACKTEST
# ============================================================

def run_backtest(
    df,
    rr
):

    balance = INITIAL_BALANCE

    wins = 0
    losses = 0
    timeouts = 0

    gross_profit = 0.0
    gross_loss = 0.0

    trades = []

    equity_curve = [
        balance
    ]

    # --------------------------------------------------------
    # Process each symbol independently
    # --------------------------------------------------------

    for symbol in df["symbol"].unique():

        symbol_df = (
            df[
                df["symbol"] == symbol
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(symbol_df) < 30:
            continue

        local_index = 20

        while local_index < len(symbol_df):

            row = symbol_df.iloc[
                local_index
            ]

            signal = int(
                row["signal"]
            )

            if signal == 0:

                local_index += 1
                continue

            entry = float(
                row["current_price"]
            )

            # ------------------------------------------------
            # 0.40% minimum move filter
            # ------------------------------------------------

            future_5 = get_future_prices(
                symbol_df,
                local_index,
                HORIZON_5M
            )

            if not has_minimum_move(
                entry,
                future_5,
                signal
            ):

                local_index += 1
                continue

            # ------------------------------------------------
            # Execute trade
            # ------------------------------------------------

            result = simulate_trade(
                symbol_df,
                local_index,
                signal,
                rr
            )

            if result is None:

                local_index += 1
                continue

            pnl_pct = result[
                "pnl_pct"
            ]

            trade_pnl = (
                TRADE_AMOUNT *
                pnl_pct
            )

            balance += trade_pnl

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            if result["outcome"] == "WIN":

                wins += 1

                gross_profit += (
                    trade_pnl
                )

            elif result["outcome"] == "LOSS":

                losses += 1

                gross_loss += abs(
                    trade_pnl
                )

            else:

                timeouts += 1

                if trade_pnl > 0:
                    gross_profit += (
                        trade_pnl
                    )

                elif trade_pnl < 0:
                    gross_loss += abs(
                        trade_pnl
                    )

            trades.append({

                "symbol": symbol,

                "timestamp":
                    row["timestamp"],

                "direction":
                    "LONG"
                    if signal == 1
                    else "SHORT",

                "entry":
                    result["entry"],

                "exit":
                    result["exit"],

                "outcome":
                    result["outcome"],

                "pnl_pct":
                    pnl_pct,

                "pnl":
                    trade_pnl,

                "rr":
                    f"1:{rr}"

            })

            equity_curve.append(
                balance
            )

            # ------------------------------------------------
            # Prevent overlapping trades
            # ------------------------------------------------

            next_index = (
                result["exit_index"]
            )

            if next_index <= local_index:

                local_index += 1

            else:

                local_index = (
                    next_index + 1
                )

    # ========================================================
    # METRICS
    # ========================================================

    total_trades = (
        wins +
        losses +
        timeouts
    )

    decisive_trades = (
        wins +
        losses
    )

    win_rate = (
        wins /
        decisive_trades *
        100
        if decisive_trades > 0
        else 0
    )

    net_profit = (
        balance -
        INITIAL_BALANCE
    )

    return_pct = (
        net_profit /
        INITIAL_BALANCE *
        100
    )

    profit_factor = (
        gross_profit /
        gross_loss
        if gross_loss > 0
        else np.inf
    )

    # --------------------------------------------------------
    # Max Drawdown
    # --------------------------------------------------------

    equity = np.array(
        equity_curve
    )

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity -
        running_max
    ) / running_max

    max_drawdown = (
        abs(
            np.min(drawdown)
        ) * 100
    )

    # --------------------------------------------------------
    # Expectancy
    # --------------------------------------------------------

    if total_trades > 0:

        expectancy = (
            net_profit /
            total_trades
        )

    else:

        expectancy = 0.0

    # --------------------------------------------------------
    # Sharpe approximation
    # --------------------------------------------------------

    if len(trades) > 1:

        returns = np.array([
            t["pnl_pct"]
            for t in trades
        ])

        if np.std(returns) > 0:

            sharpe = (
                np.mean(returns) /
                np.std(returns)
            ) * np.sqrt(
                len(returns)
            )

        else:

            sharpe = 0.0

    else:

        sharpe = 0.0

    return {
        "rr": rr,
        "balance": balance,
        "net_profit": net_profit,
        "return_pct": return_pct,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "trades": trades
    }


# ============================================================
# REPORT
# ============================================================

def print_report(result):

    print("\n")
    print("=" * 60)
    print(
        f"📊 RESEARCH BACKTEST — RR 1:{result['rr']}"
    )
    print("=" * 60)

    print(
        f"Starting Balance : "
        f"${INITIAL_BALANCE:.2f}"
    )

    print(
        f"Final Balance    : "
        f"${result['balance']:.2f}"
    )

    print(
        f"Net Profit       : "
        f"${result['net_profit']:.2f}"
    )

    print(
        f"Return           : "
        f"{result['return_pct']:.2f}%"
    )

    print("-" * 60)

    print(
        f"Total Trades     : "
        f"{result['total_trades']}"
    )

    print(
        f"Wins             : "
        f"{result['wins']}"
    )

    print(
        f"Losses           : "
        f"{result['losses']}"
    )

    print(
        f"Timeouts         : "
        f"{result['timeouts']}"
    )

    print(
        f"Win Rate         : "
        f"{result['win_rate']:.2f}%"
    )

    print("-" * 60)

    print(
        f"Profit Factor    : "
        f"{result['profit_factor']:.3f}"
    )

    print(
        f"Expectancy/Trade : "
        f"${result['expectancy']:.4f}"
    )

    print(
        f"Max Drawdown     : "
        f"{result['max_drawdown']:.2f}%"
    )

    print(
        f"Sharpe           : "
        f"{result['sharpe']:.3f}"
    )

    print("=" * 60)


# ============================================================
# SAVE TRADES
# ============================================================

def save_trades(
    result,
    filename
):

    if len(result["trades"]) == 0:

        print(
            f"⚠️ No trades to save: {filename}"
        )

        return

    trade_df = pd.DataFrame(
        result["trades"]
    )

    trade_df.to_csv(
        filename,
        index=False
    )

    print(
        f"💾 Saved: {filename}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n🧪 Starting Research Backtest..."
    )

    df = load_data()

    if df is None:
        return

    print(
        f"📥 Loaded rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    df = add_features(df)

    df = normalize_ofi(df)

    # --------------------------------------------------------
    # Required values
    # --------------------------------------------------------

    required_features = [
        "obi_top20",
        "obi_top50",
        "trend_pct",
        "ofi_normalized"
    ]

    for col in required_features:

        if col not in df.columns:

            print(
                f"❌ Missing feature: {col}"
            )

            return

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    df = create_signals(df)

    # --------------------------------------------------------
    # Run both RR versions
    # --------------------------------------------------------

    result_1_2 = run_backtest(
        df,
        rr=2
    )

    result_1_3 = run_backtest(
        df,
        rr=3
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    print_report(
        result_1_2
    )

    print_report(
        result_1_3
    )

    # --------------------------------------------------------
    # Save trade logs
    # --------------------------------------------------------

    save_trades(
        result_1_2,
        "backtest_trades_1_2.csv"
    )

    save_trades(
        result_1_3,
        "backtest_trades_1_3.csv"
    )

    print(
        "\n✅ Backtest completed."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
