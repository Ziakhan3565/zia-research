from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import ccxt
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_LEVERAGE = 5
DEFAULT_TRADE_USDT = 10.0

DEFAULT_STOP_LOSS_PCT = 0.006       # 0.60%
DEFAULT_MIN_RR = 1.50
MAX_ALLOWED_RR = 20.0

POLL_INTERVAL = 5

SIGNAL_FILE = "signal_history.csv"
CONFIG_FILE = "config.json"


# ============================================================
# MEXC SETUP
# ============================================================

mexc = ccxt.mexc({
    "apiKey": os.getenv("MEXC_API_KEY", "YOUR_MEXC_API_KEY"),
    "secret": os.getenv("MEXC_SECRET_KEY", "YOUR_MEXC_SECRET_KEY"),
    "enableRateLimit": True,
})


# ============================================================
# SAFE NUMBER HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)

        if number != number:  # NaN
            return default

        if number in (float("inf"), float("-inf")):
            return default

        return number

    except (TypeError, ValueError):
        return default


def normalize_intent(intent: Any) -> str:
    """
    Convert different signal names into LONG / SHORT / NEUTRAL.
    """

    value = str(intent or "NEUTRAL").upper().strip()

    if value in {
        "LONG",
        "BUY",
        "STRONG LONG",
        "STRONG_LONG",
    }:
        return "LONG"

    if value in {
        "SHORT",
        "SELL",
        "STRONG SHORT",
        "STRONG_SHORT",
    }:
        return "SHORT"

    return "NEUTRAL"


def is_strong_signal(intent: Any) -> bool:
    value = str(intent or "").upper().strip()

    return value in {
        "STRONG LONG",
        "STRONG_LONG",
        "STRONG SHORT",
        "STRONG_SHORT",
    }


# ============================================================
# CONFIG LOADER
# ============================================================

def load_config() -> Dict[str, Any]:

    if not os.path.exists(CONFIG_FILE):
        return {
            "is_running": False,
            "leverage": DEFAULT_LEVERAGE,
            "trade_amount_usdt": DEFAULT_TRADE_USDT,
            "selected_coins": [],
            "min_rr": DEFAULT_MIN_RR,
        }

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            config = json.load(file)

        if not isinstance(config, dict):
            raise ValueError("config.json must contain an object")

        return config

    except Exception as error:
        print(f"⚠️ Config error: {error}")

        return {
            "is_running": False,
            "leverage": DEFAULT_LEVERAGE,
            "trade_amount_usdt": DEFAULT_TRADE_USDT,
            "selected_coins": [],
            "min_rr": DEFAULT_MIN_RR,
        }


# ============================================================
# LEVERAGE / MARGIN
# ============================================================

def set_leverage_and_margin(
    symbol: str,
    leverage: int,
) -> bool:

    try:

        leverage = max(1, int(leverage))

        try:
            mexc.set_margin_mode(
                "isolated",
                symbol,
            )
        except Exception as margin_error:
            print(
                f"⚠️ Margin mode warning for {symbol}: "
                f"{margin_error}"
            )

        try:
            mexc.set_leverage(
                leverage,
                symbol,
            )

        except Exception as leverage_error:
            print(
                f"⚠️ Leverage warning for {symbol}: "
                f"{leverage_error}"
            )

        print(
            f"⚙️ Leverage: {leverage}x | "
            f"Margin: Isolated | {symbol}"
        )

        return True

    except Exception as error:

        print(
            f"❌ Leverage/Margin setup failed: {error}"
        )

        return False


# ============================================================
# SIGNAL VALIDATION
# ============================================================

def validate_signal(
    row: pd.Series,
) -> bool:
    """
    Final safety validation before execution.

    Signal generation is NOT performed here.
    This method validates an already-generated signal.
    """

    intent = normalize_intent(
        row.get("intent", "NEUTRAL")
    )

    if intent == "NEUTRAL":
        return False

    entry = safe_float(
        row.get("current_price", 0)
    )

    stop_loss = safe_float(
        row.get("stop_loss", 0)
    )

    take_profit = safe_float(
        row.get("take_profit", 0)
    )

    if entry <= 0:
        print("⚠️ Invalid entry price.")
        return False

    if stop_loss <= 0:
        print("⚠️ Invalid stop-loss.")
        return False

    if take_profit <= 0:
        print("⚠️ Invalid take-profit.")
        return False

    # LONG validation
    if intent == "LONG":

        if not stop_loss < entry:
            print(
                "⚠️ LONG rejected: "
                "SL must be below entry."
            )
            return False

        if not take_profit > entry:
            print(
                "⚠️ LONG rejected: "
                "TP must be above entry."
            )
            return False

    # SHORT validation
    if intent == "SHORT":

        if not stop_loss > entry:
            print(
                "⚠️ SHORT rejected: "
                "SL must be above entry."
            )
            return False

        if not take_profit < entry:
            print(
                "⚠️ SHORT rejected: "
                "TP must be below entry."
            )
            return False

    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)

    if risk <= 0:
        print("⚠️ Invalid risk distance.")
        return False

    rr = reward / risk

    if rr <= 0 or rr > MAX_ALLOWED_RR:
        print(
            f"⚠️ Invalid RR: {rr:.2f}"
        )
        return False

    return True


# ============================================================
# RISK / RR CALCULATOR
# ============================================================

def calculate_rr(
    entry: float,
    stop_loss: float,
    take_profit: float,
) -> float:

    risk = abs(entry - stop_loss)

    if risk <= 0:
        return 0.0

    reward = abs(take_profit - entry)

    return reward / risk


def apply_minimum_rr(
    intent: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    minimum_rr: float = DEFAULT_MIN_RR,
):
    """
    Keeps the existing TRI/Fourier target when it already
    satisfies the required RR.

    If target is too close, extends TP to minimum RR.
    """

    intent = normalize_intent(intent)

    entry = safe_float(entry)
    stop_loss = safe_float(stop_loss)
    take_profit = safe_float(take_profit)

    minimum_rr = max(
        1.0,
        safe_float(
            minimum_rr,
            DEFAULT_MIN_RR,
        ),
    )

    if entry <= 0 or stop_loss <= 0:
        return stop_loss, take_profit

    risk = abs(entry - stop_loss)

    if risk <= 0:
        return stop_loss, take_profit

    required_reward = risk * minimum_rr

    current_reward = abs(
        take_profit - entry
    )

    if current_reward >= required_reward:
        return stop_loss, take_profit

    if intent == "LONG":

        take_profit = (
            entry + required_reward
        )

    elif intent == "SHORT":

        take_profit = (
            entry - required_reward
        )

    return (
        round(stop_loss, 8),
        round(take_profit, 8),
    )


# ============================================================
# SIGNAL METADATA
# ============================================================

def print_signal_information(
    row: pd.Series,
    stop_loss: float,
    take_profit: float,
):

    intent = normalize_intent(
        row.get("intent", "NEUTRAL")
    )

    entry = safe_float(
        row.get("current_price", 0)
    )

    rr = calculate_rr(
        entry,
        stop_loss,
        take_profit,
    )

    score = safe_float(
        row.get("score", 0)
    )

    confidence = safe_float(
        row.get("confidence", 0)
    )

    ml_probability = safe_float(
        row.get(
            "ml_probability",
            row.get("ml_confidence", 0),
        )
    )

    obi = safe_float(
        row.get(
            "obi_top20",
            row.get("BOOK_IMB", 0),
        )
    )

    print()
    print("=" * 65)
    print("SIGNAL VALIDATION")
    print("=" * 65)

    print(f"Direction       : {intent}")
    print(f"Entry           : {entry:.8f}")
    print(f"Stop Loss       : {stop_loss:.8f}")
    print(f"Take Profit     : {take_profit:.8f}")
    print(f"Risk / Reward   : 1:{rr:.2f}")

    if score != 0:
        print(f"Research Score  : {score:.4f}")

    if confidence != 0:
        print(f"Confidence      : {confidence:.2f}")

    if ml_probability != 0:
        print(
            f"ML Probability  : {ml_probability:.2f}"
        )

    if obi != 0:
        print(f"OBI             : {obi:.4f}")

    # Optional TRI information
    tri_signal = row.get(
        "tri_signal",
        row.get("TRI_SIGNAL", ""),
    )

    if str(tri_signal).strip():
        print(
            f"TRI Confirmation : {tri_signal}"
        )

    tri_target = row.get(
        "tri_target",
        row.get("TRI_TARGET", ""),
    )

    if str(tri_target).strip():
        print(
            f"TRI Target       : {tri_target}"
        )

    print("=" * 65)
    print()


# ============================================================
# MARKET ORDER EXECUTION
# ============================================================

def execute_trade(
    row: pd.Series,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    symbol = str(
        row.get("symbol", "")
    ).upper().strip()

    intent = normalize_intent(
        row.get("intent", "NEUTRAL")
    )

    if not symbol:
        print("❌ Symbol missing.")
        return None

    if intent == "NEUTRAL":
        print(
            "⏸️ NEUTRAL signal - no trade."
        )
        return None

    entry_price = safe_float(
        row.get("current_price", 0)
    )

    stop_loss = safe_float(
        row.get("stop_loss", 0)
    )

    take_profit = safe_float(
        row.get("take_profit", 0)
    )

    # --------------------------------------------------------
    # Validate signal
    # --------------------------------------------------------

    if not validate_signal(row):
        print(
            f"❌ Signal rejected for {symbol}"
        )
        return None

    # --------------------------------------------------------
    # Minimum RR protection
    # --------------------------------------------------------

    minimum_rr = safe_float(
        config.get(
            "min_rr",
            DEFAULT_MIN_RR,
        ),
        DEFAULT_MIN_RR,
    )

    stop_loss, take_profit = apply_minimum_rr(
        intent=intent,
        entry=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        minimum_rr=minimum_rr,
    )

    rr = calculate_rr(
        entry_price,
        stop_loss,
        take_profit,
    )

    print_signal_information(
        row,
        stop_loss,
        take_profit,
    )

    # --------------------------------------------------------
    # Exchange setup
    # --------------------------------------------------------

    leverage = int(
        safe_float(
            config.get(
                "leverage",
                DEFAULT_LEVERAGE,
            ),
            DEFAULT_LEVERAGE,
        )
    )

    set_leverage_and_margin(
        symbol,
        leverage,
    )

    # --------------------------------------------------------
    # Position size
    # --------------------------------------------------------

    trade_usdt = safe_float(
        config.get(
            "trade_amount_usdt",
            DEFAULT_TRADE_USDT,
        ),
        DEFAULT_TRADE_USDT,
    )

    if trade_usdt <= 0:
        print(
            "❌ Invalid trade amount."
        )
        return None

    if entry_price <= 0:
        print(
            "❌ Invalid entry price."
        )
        return None

    amount = (
        trade_usdt / entry_price
    )

    try:
        amount = float(
            mexc.amount_to_precision(
                symbol,
                amount,
            )
        )

    except Exception:
        amount = float(amount)

    if amount <= 0:
        print(
            "❌ Calculated order amount is zero."
        )
        return None

    side = (
        "buy"
        if intent == "LONG"
        else "sell"
    )

    # --------------------------------------------------------
    # Final execution message
    # --------------------------------------------------------

    print(
        f"🚀 EXECUTING {intent} | "
        f"{symbol} | "
        f"Amount: {amount}"
    )

    print(
        f"Entry ~ {entry_price}"
    )

    print(
        f"SL = {stop_loss} | "
        f"TP = {take_profit} | "
        f"RR = 1:{rr:.2f}"
    )

    # --------------------------------------------------------
    # MEXC parameters
    # --------------------------------------------------------

    order_params = {
        "stopLossPrice": stop_loss,
        "takeProfitPrice": take_profit,
    }

    try:

        order = mexc.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params=order_params,
        )

        order_id = order.get(
            "id",
            "UNKNOWN",
        )

        print()
        print(
            "✅ ORDER SUCCESSFULLY PLACED"
        )
        print(
            f"Order ID: {order_id}"
        )
        print()

        return {
            "success": True,
            "order_id": order_id,
            "symbol": symbol,
            "intent": intent,
            "side": side,
            "amount": amount,
            "entry": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "rr": rr,
        }

    except Exception as order_error:

        print()
        print(
            "❌ ORDER EXECUTION ERROR"
        )
        print(
            f"{order_error}"
        )
        print()

        return None


# ============================================================
# SIGNAL FILE READER
# ============================================================

def get_latest_signal() -> Optional[pd.Series]:

    if not os.path.exists(
        SIGNAL_FILE
    ):
        return None

    try:

        df = pd.read_csv(
            SIGNAL_FILE
        )

        if df.empty:
            return None

        # Newest signal first
        if "timestamp" in df.columns:

            try:

                df["_parsed_time"] = pd.to_datetime(
                    df["timestamp"],
                    errors="coerce",
                )

                df = df.sort_values(
                    "_parsed_time",
                    ascending=False,
                )

            except Exception:
                pass

        return df.iloc[0]

    except Exception as error:

        print(
            f"⚠️ Signal file error: {error}"
        )

        return None


# ============================================================
# SIGNAL ID
# ============================================================

def make_signal_id(
    row: pd.Series,
) -> str:

    timestamp = str(
        row.get(
            "timestamp",
            "",
        )
    )

    symbol = str(
        row.get(
            "symbol",
            "",
        )
    ).upper()

    intent = str(
        row.get(
            "intent",
            "",
        )
    ).upper()

    entry = str(
        row.get(
            "current_price",
            "",
        )
    )

    return (
        f"{timestamp}|"
        f"{symbol}|"
        f"{intent}|"
        f"{entry}"
    )


# ============================================================
# MAIN BOT
# ============================================================

def run_bot():

    processed_signals = set()

    print()
    print("=" * 65)
    print("🤖 ZIA RESEARCH BOT ENGINE")
    print("=" * 65)
    print(
        "Execution engine loaded."
    )
    print(
        "Research Lab / ML / TRI "
        "signals are consumed from signal_history.csv."
    )
    print("=" * 65)
    print()

    while True:

        try:

            # ------------------------------------------------
            # Load configuration
            # ------------------------------------------------

            config = load_config()

            # ------------------------------------------------
            # Bot OFF
            # ------------------------------------------------

            if not config.get(
                "is_running",
                False,
            ):

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            # ------------------------------------------------
            # Latest signal
            # ------------------------------------------------

            latest = get_latest_signal()

            if latest is None:

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            symbol = str(
                latest.get(
                    "symbol",
                    "",
                )
            ).upper().strip()

            selected_coins = [
                str(x).upper().strip()
                for x in config.get(
                    "selected_coins",
                    [],
                )
            ]

            # ------------------------------------------------
            # Coin filter
            # ------------------------------------------------

            if (
                selected_coins
                and symbol not in selected_coins
            ):

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            # ------------------------------------------------
            # Signal ID
            # ------------------------------------------------

            signal_id = make_signal_id(
                latest
            )

            if signal_id in processed_signals:

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            # ------------------------------------------------
            # Signal status
            # ------------------------------------------------

            intent = normalize_intent(
                latest.get(
                    "intent",
                    "NEUTRAL",
                )
            )

            if intent == "NEUTRAL":

                processed_signals.add(
                    signal_id
                )

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            # ------------------------------------------------
            # Optional confidence protection
            # ------------------------------------------------

            confidence = safe_float(
                latest.get(
                    "confidence",
                    0,
                )
            )

            min_confidence = safe_float(
                config.get(
                    "min_confidence",
                    0,
                )
            )

            if (
                min_confidence > 0
                and confidence > 0
                and confidence < min_confidence
            ):

                print(
                    f"⏸️ Signal confidence "
                    f"{confidence:.2f} below "
                    f"minimum {min_confidence:.2f}"
                )

                processed_signals.add(
                    signal_id
                )

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            # ------------------------------------------------
            # Execute
            # ------------------------------------------------

            result = execute_trade(
                latest,
                config,
            )

            # Mark processed whether successful or rejected
            processed_signals.add(
                signal_id
            )

            # Prevent unlimited memory growth
            if len(processed_signals) > 5000:

                processed_signals = set(
                    list(processed_signals)[-2500:]
                )

            if result is not None:

                print(
                    f"✅ Trade completed for {symbol}"
                )

            else:

                print(
                    f"⚠️ No trade executed for {symbol}"
                )

        except KeyboardInterrupt:

            print()
            print(
                "🛑 Bot stopped by user."
            )

            break

        except Exception as error:

            print()
            print(
                f"🔥 Bot loop error: {error}"
            )
            print(
                "Retrying..."
            )
            print()

        time.sleep(
            POLL_INTERVAL
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_bot()
