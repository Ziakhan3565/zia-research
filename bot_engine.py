from __future__ import annotations

import json
import os
import time
import uuid
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List

import ccxt
import pandas as pd


# ============================================================
# ZIA RESEARCH - EXECUTION ENGINE
# ============================================================
#
# SIGNAL FLOW
#
# 30M / SCALPING -> maximum 15 minutes
# 1H             -> maximum 90 minutes
# 4H             -> maximum 24 hours
#
# Trade can close before expiry when:
#   1. TAKE PROFIT is reached
#   2. STOP LOSS is reached
#   3. MAX HOLD TIME is reached
#
# Default:
#   PAPER TRADING = ON
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_LEVERAGE = 5
DEFAULT_TRADE_USDT = 10.0

DEFAULT_STOP_LOSS_PCT = 0.006
DEFAULT_MIN_RR = 1.50
MAX_ALLOWED_RR = 20.0

POLL_INTERVAL = 5

SIGNAL_FILE = "signal_history.csv"
CONFIG_FILE = "config.json"

ACTIVE_TRADES_FILE = "active_trades.csv"
TRADE_HISTORY_FILE = "trade_history.csv"

# ------------------------------------------------------------
# SAFETY
# ------------------------------------------------------------

# IMPORTANT:
# Keep False while testing.
#
# False = no real MEXC order
# True  = real MEXC order
#
LIVE_TRADING = False


# ============================================================
# TRADE DURATION
# ============================================================

TRADE_DURATIONS_MINUTES = {
    "30M": 15,
    "SCALPING": 15,
    "SCALP": 15,

    "1H": 90,

    "4H": 24 * 60,
}


def normalize_trade_mode(value: Any) -> str:
    """
    Normalize dashboard/research trade mode.
    """

    text = str(value or "30M").upper().strip()

    aliases = {
        "SCALP": "SCALPING",
        "SCALPING": "SCALPING",
        "30M": "SCALPING",

        "1H": "1H",
        "60M": "1H",

        "4H": "4H",
        "240M": "4H",
    }

    return aliases.get(text, "SCALPING")


def get_max_hold_minutes(trade_mode: Any) -> int:
    """
    Return maximum allowed trade duration.
    """

    mode = normalize_trade_mode(trade_mode)

    if mode == "SCALPING":
        return 15

    if mode == "1H":
        return 90

    if mode == "4H":
        return 24 * 60

    return 15


# ============================================================
# MEXC SETUP
# ============================================================

mexc = ccxt.mexc({
    "apiKey": os.getenv(
        "MEXC_API_KEY",
        "YOUR_MEXC_API_KEY",
    ),

    "secret": os.getenv(
        "MEXC_SECRET_KEY",
        "YOUR_MEXC_SECRET_KEY",
    ),

    "enableRateLimit": True,
})


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        number = float(value)

        if number != number:
            return default

        if number in (
            float("inf"),
            float("-inf"),
        ):
            return default

        return number

    except (
        TypeError,
        ValueError,
    ):

        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:
        return int(float(value))

    except Exception:
        return default


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat()


def parse_datetime(value: Any) -> Optional[datetime]:

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:

        dt = pd.to_datetime(
            text,
            errors="coerce",
            utc=True,
        )

        if pd.isna(dt):
            return None

        return dt.to_pydatetime()

    except Exception:

        return None


def normalize_intent(
    intent: Any,
) -> str:

    value = str(
        intent or "NEUTRAL"
    ).upper().strip()

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


def is_strong_signal(
    intent: Any,
) -> bool:

    value = str(
        intent or ""
    ).upper().strip()

    return value in {
        "STRONG LONG",
        "STRONG_LONG",
        "STRONG SHORT",
        "STRONG_SHORT",
    }


# ============================================================
# FILE HELPERS
# ============================================================

def load_csv(
    filename: str,
) -> pd.DataFrame:

    if not os.path.exists(filename):
        return pd.DataFrame()

    try:

        df = pd.read_csv(filename)

        if df is None:
            return pd.DataFrame()

        return df

    except Exception as error:

        print(
            f"⚠️ CSV read error "
            f"{filename}: {error}"
        )

        return pd.DataFrame()


def save_csv(
    df: pd.DataFrame,
    filename: str,
) -> bool:

    try:

        df.to_csv(
            filename,
            index=False,
        )

        return True

    except Exception as error:

        print(
            f"❌ CSV save error "
            f"{filename}: {error}"
        )

        return False


def append_csv_row(
    filename: str,
    row: Dict[str, Any],
) -> bool:

    try:

        existing = load_csv(filename)

        new_row = pd.DataFrame([row])

        if existing.empty:

            final = new_row

        else:

            final = pd.concat(
                [
                    existing,
                    new_row,
                ],
                ignore_index=True,
            )

        return save_csv(
            final,
            filename,
        )

    except Exception as error:

        print(
            f"❌ Append CSV error: {error}"
        )

        return False


# ============================================================
# CONFIG LOADER
# ============================================================

def default_config() -> Dict[str, Any]:

    return {
        "is_running": False,

        "live_trading": False,

        "leverage": DEFAULT_LEVERAGE,

        "trade_amount_usdt":
            DEFAULT_TRADE_USDT,

        "selected_coins": [],

        "min_rr":
            DEFAULT_MIN_RR,

        "min_confidence": 0,

        "allow_multiple_positions":
            False,

        "close_on_expiry": True,

        "poll_interval":
            POLL_INTERVAL,
    }


def load_config() -> Dict[str, Any]:

    defaults = default_config()

    if not os.path.exists(
        CONFIG_FILE
    ):

        return defaults

    try:

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            config = json.load(file)

        if not isinstance(
            config,
            dict,
        ):

            return defaults

        defaults.update(config)

        return defaults

    except Exception as error:

        print(
            f"⚠️ Config error: {error}"
        )

        return defaults


# ============================================================
# LEVERAGE / MARGIN
# ============================================================

def set_leverage_and_margin(
    symbol: str,
    leverage: int,
) -> bool:

    try:

        leverage = max(
            1,
            int(leverage),
        )

        try:

            mexc.set_margin_mode(
                "isolated",
                symbol,
            )

        except Exception as error:

            print(
                f"⚠️ Margin warning "
                f"{symbol}: {error}"
            )

        try:

            mexc.set_leverage(
                leverage,
                symbol,
            )

        except Exception as error:

            print(
                f"⚠️ Leverage warning "
                f"{symbol}: {error}"
            )

        print(
            f"⚙️ {symbol} | "
            f"Leverage {leverage}x | "
            f"Isolated"
        )

        return True

    except Exception as error:

        print(
            f"❌ Leverage setup failed: "
            f"{error}"
        )

        return False


# ============================================================
# RR
# ============================================================

def calculate_rr(
    entry: float,
    stop_loss: float,
    take_profit: float,
) -> float:

    risk = abs(
        entry - stop_loss
    )

    if risk <= 0:
        return 0.0

    reward = abs(
        take_profit - entry
    )

    return reward / risk


def apply_minimum_rr(
    intent: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    minimum_rr: float =
        DEFAULT_MIN_RR,
):

    intent = normalize_intent(
        intent
    )

    entry = safe_float(entry)

    stop_loss = safe_float(
        stop_loss
    )

    take_profit = safe_float(
        take_profit
    )

    minimum_rr = max(
        1.0,
        safe_float(
            minimum_rr,
            DEFAULT_MIN_RR,
        ),
    )

    if entry <= 0:
        return (
            stop_loss,
            take_profit,
        )

    if stop_loss <= 0:
        return (
            stop_loss,
            take_profit,
        )

    risk = abs(
        entry - stop_loss
    )

    if risk <= 0:
        return (
            stop_loss,
            take_profit,
        )

    required_reward = (
        risk * minimum_rr
    )

    current_reward = abs(
        take_profit - entry
    )

    if current_reward >= required_reward:

        return (
            stop_loss,
            take_profit,
        )

    if intent == "LONG":

        take_profit = (
            entry
            + required_reward
        )

    elif intent == "SHORT":

        take_profit = (
            entry
            - required_reward
        )

    return (
        round(
            stop_loss,
            8,
        ),

        round(
            take_profit,
            8,
        ),
    )


# ============================================================
# SIGNAL VALIDATION
# ============================================================

def validate_signal(
    row: pd.Series,
) -> bool:

    intent = normalize_intent(
        row.get(
            "intent",
            row.get(
                "direction",
                "NEUTRAL",
            ),
        )
    )

    if intent == "NEUTRAL":

        return False

    entry = safe_float(
        row.get(
            "current_price",
            row.get(
                "entry",
                0,
            ),
        )
    )

    stop_loss = safe_float(
        row.get(
            "stop_loss",
            row.get(
                "STOP_LOSS",
                0,
            ),
        )
    )

    take_profit = safe_float(
        row.get(
            "take_profit",
            row.get(
                "TAKE_PROFIT",
                0,
            ),
        )
    )

    if entry <= 0:

        print(
            "⚠️ Invalid entry."
        )

        return False

    if stop_loss <= 0:

        print(
            "⚠️ Invalid stop-loss."
        )

        return False

    if take_profit <= 0:

        print(
            "⚠️ Invalid take-profit."
        )

        return False

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

    elif intent == "SHORT":

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

    rr = calculate_rr(
        entry,
        stop_loss,
        take_profit,
    )

    if rr <= 0:

        return False

    if rr > MAX_ALLOWED_RR:

        print(
            f"⚠️ RR too high: "
            f"{rr:.2f}"
        )

        return False

    return True


# ============================================================
# SIGNAL INFORMATION
# ============================================================

def print_signal_information(
    row: pd.Series,
    stop_loss: float,
    take_profit: float,
):

    intent = normalize_intent(
        row.get(
            "intent",
            "NEUTRAL",
        )
    )

    entry = safe_float(
        row.get(
            "current_price",
            0,
        )
    )

    rr = calculate_rr(
        entry,
        stop_loss,
        take_profit,
    )

    score = safe_float(
        row.get(
            "score",
            row.get(
                "final_score",
                0,
            ),
        )
    )

    confidence = safe_float(
        row.get(
            "confidence",
            0,
        )
    )

    ml_probability = safe_float(
        row.get(
            "ml_probability",
            row.get(
                "ML_PROBABILITY",
                0,
            ),
        )
    )

    trade_mode = normalize_trade_mode(
        row.get(
            "trade_mode",
            row.get(
                "TRADE_MODE",
                "SCALPING",
            ),
        )
    )

    max_hold = get_max_hold_minutes(
        trade_mode
    )

    tri_signal = row.get(
        "tri_signal",
        row.get(
            "TRI_SIGNAL",
            "",
        ),
    )

    print()
    print("=" * 70)
    print("SIGNAL VALIDATION")
    print("=" * 70)

    print(
        f"Direction        : {intent}"
    )

    print(
        f"Trade Mode       : {trade_mode}"
    )

    print(
        f"Max Hold         : "
        f"{max_hold} minutes"
    )

    print(
        f"Entry            : "
        f"{entry:.8f}"
    )

    print(
        f"Stop Loss        : "
        f"{stop_loss:.8f}"
    )

    print(
        f"Take Profit      : "
        f"{take_profit:.8f}"
    )

    print(
        f"Risk / Reward    : "
        f"1:{rr:.2f}"
    )

    if score != 0:

        print(
            f"Research Score   : "
            f"{score:.4f}"
        )

    if confidence != 0:

        print(
            f"Confidence       : "
            f"{confidence:.2f}"
        )

    if ml_probability != 0:

        print(
            f"ML Probability   : "
            f"{ml_probability:.2%}"
        )

    if str(tri_signal).strip():

        print(
            f"TRI Confirmation : "
            f"{tri_signal}"
        )

    print("=" * 70)
    print()


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
    ).upper().strip()

    intent = normalize_intent(
        row.get(
            "intent",
            "NEUTRAL",
        )
    )

    entry = str(
        row.get(
            "current_price",
            "",
        )
    )

    trade_mode = normalize_trade_mode(
        row.get(
            "trade_mode",
            "SCALPING",
        )
    )

    raw = (
        f"{timestamp}|"
        f"{symbol}|"
        f"{intent}|"
        f"{entry}|"
        f"{trade_mode}"
    )

    # Keep ID stable
    return raw


# ============================================================
# ACTIVE TRADE STORAGE
# ============================================================

def get_active_trades() -> pd.DataFrame:

    df = load_csv(
        ACTIVE_TRADES_FILE
    )

    if df.empty:
        return pd.DataFrame()

    if "status" not in df.columns:

        df["status"] = "OPEN"

    return df


def save_active_trades(
    df: pd.DataFrame,
) -> bool:

    return save_csv(
        df,
        ACTIVE_TRADES_FILE,
    )


def get_trade_history() -> pd.DataFrame:

    return load_csv(
        TRADE_HISTORY_FILE
    )


# ============================================================
# ACTIVE TRADE CHECK
# ============================================================

def has_active_trade(
    symbol: str,
) -> bool:

    symbol = str(
        symbol
    ).upper().strip()

    df = get_active_trades()

    if df.empty:
        return False

    if "symbol" not in df.columns:
        return False

    if "status" not in df.columns:
        return False

    rows = df[
        (
            df["symbol"]
            .astype(str)
            .str.upper()
            .str.strip()
            == symbol
        )
        &
        (
            df["status"]
            .astype(str)
            .str.upper()
            == "OPEN"
        )
    ]

    return not rows.empty


# ============================================================
# CREATE ACTIVE TRADE
# ============================================================

def create_active_trade(
    row: pd.Series,
    order_result: Optional[Dict[str, Any]],
    stop_loss: float,
    take_profit: float,
    amount: float,
) -> Dict[str, Any]:

    symbol = str(
        row.get(
            "symbol",
            "",
        )
    ).upper().strip()

    intent = normalize_intent(
        row.get(
            "intent",
            "NEUTRAL",
        )
    )

    trade_mode = normalize_trade_mode(
        row.get(
            "trade_mode",
            row.get(
                "TRADE_MODE",
                "SCALPING",
            ),
        )
    )

    entry = safe_float(
        row.get(
            "current_price",
            0,
        )
    )

    opened_at = now_utc()

    max_hold_minutes = (
        get_max_hold_minutes(
            trade_mode
        )
    )

    expires_at = (
        opened_at
        + timedelta(
            minutes=max_hold_minutes
        )
    )

    trade_id = str(
        uuid.uuid4()
    )

    rr = calculate_rr(
        entry,
        stop_loss,
        take_profit,
    )

    order_id = ""

    if order_result:

        order_id = str(
            order_result.get(
                "order_id",
                "",
            )
        )

    trade = {

        "trade_id":
            trade_id,

        "signal_id":
            make_signal_id(row),

        "symbol":
            symbol,

        "direction":
            intent,

        "trade_mode":
            trade_mode,

        "status":
            "OPEN",

        "opened_at":
            opened_at.isoformat(),

        "expires_at":
            expires_at.isoformat(),

        "max_hold_minutes":
            max_hold_minutes,

        "entry":
            entry,

        "stop_loss":
            stop_loss,

        "take_profit":
            take_profit,

        "rr":
            rr,

        "amount":
            amount,

        "leverage":
            safe_int(
                row.get(
                    "leverage",
                    DEFAULT_LEVERAGE,
                ),
                DEFAULT_LEVERAGE,
            ),

        "order_id":
            order_id,

        "close_time":
            "",

        "close_price":
            "",

        "close_reason":
            "",

        "pnl_usdt":
            "",

        "duration_minutes":
            "",

        "confidence":
            safe_float(
                row.get(
                    "confidence",
                    0,
                )
            ),

        "score":
            safe_float(
                row.get(
                    "score",
                    0,
                )
            ),

        "ml_probability":
            safe_float(
                row.get(
                    "ml_probability",
                    0,
                )
            ),

        "tri_signal":
            str(
                row.get(
                    "tri_signal",
                    "",
                )
            ),

        "tri_timeframe":
            str(
                row.get(
                    "tri_timeframe",
                    "",
                )
            ),

        "tri_rr":
            safe_float(
                row.get(
                    "tri_rr",
                    0,
                )
            ),

        "execution_mode":
            "LIVE"
            if LIVE_TRADING
            else "PAPER",
    }

    return trade


def add_active_trade(
    trade: Dict[str, Any],
) -> bool:

    df = get_active_trades()

    new_row = pd.DataFrame(
        [trade]
    )

    if df.empty:

        final = new_row

    else:

        final = pd.concat(
            [
                df,
                new_row,
            ],
            ignore_index=True,
        )

    return save_active_trades(
        final
    )


# ============================================================
# MOVE TRADE TO HISTORY
# ============================================================

def archive_trade(
    trade: Dict[str, Any],
) -> bool:

    history = get_trade_history()

    row = pd.DataFrame(
        [trade]
    )

    if history.empty:

        final = row

    else:

        final = pd.concat(
            [
                history,
                row,
            ],
            ignore_index=True,
        )

    return save_csv(
        final,
        TRADE_HISTORY_FILE,
    )


# ============================================================
# REMOVE / UPDATE ACTIVE TRADE
# ============================================================

def update_active_trade(
    trade_id: str,
    updates: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    df = get_active_trades()

    if df.empty:
        return None

    if "trade_id" not in df.columns:
        return None

    matches = df[
        df["trade_id"].astype(str)
        == str(trade_id)
    ]

    if matches.empty:
        return None

    index = matches.index[0]

    for key, value in updates.items():

        df.loc[
            index,
            key,
        ] = value

    save_active_trades(df)

    return df.loc[
        index
    ].to_dict()


def close_active_trade_record(
    trade: Dict[str, Any],
    close_price: float,
    reason: str,
) -> bool:

    opened_at = parse_datetime(
        trade.get(
            "opened_at"
        )
    )

    closed_at = now_utc()

    duration_minutes = 0.0

    if opened_at:

        duration_minutes = (
            closed_at - opened_at
        ).total_seconds() / 60.0

    entry = safe_float(
        trade.get(
            "entry",
            0,
        )
    )

    amount = safe_float(
        trade.get(
            "amount",
            0,
        )
    )

    direction = normalize_intent(
        trade.get(
            "direction",
            "NEUTRAL",
        )
    )

    pnl = 0.0

    if direction == "LONG":

        pnl = (
            close_price
            - entry
        ) * amount

    elif direction == "SHORT":

        pnl = (
            entry
            - close_price
        ) * amount

    trade["status"] = reason

    trade["close_time"] = (
        closed_at.isoformat()
    )

    trade["close_price"] = (
        close_price
    )

    trade["close_reason"] = (
        reason
    )

    trade["pnl_usdt"] = (
        pnl
    )

    trade["duration_minutes"] = (
        duration_minutes
    )

    archive_trade(trade)

    df = get_active_trades()

    if not df.empty:

        if "trade_id" in df.columns:

            df = df[
                df["trade_id"].astype(str)
                != str(
                    trade.get(
                        "trade_id",
                        "",
                    )
                )
            ]

            save_active_trades(df)

    print()
    print(
        f"📕 TRADE CLOSED | "
        f"{trade.get('symbol')} | "
        f"{reason}"
    )

    print(
        f"Close Price : "
        f"{close_price}"
    )

    print(
        f"PnL         : "
        f"{pnl:.6f} USDT"
    )

    print(
        f"Duration    : "
        f"{duration_minutes:.2f} min"
    )

    print()

    return True


# ============================================================
# CURRENT PRICE
# ============================================================

def fetch_current_price(
    symbol: str,
) -> float:

    try:

        ticker = mexc.fetch_ticker(
            symbol
        )

        price = ticker.get(
            "last",
            ticker.get(
                "close",
                0,
            ),
        )

        return safe_float(
            price
        )

    except Exception as error:

        print(
            f"⚠️ Price error "
            f"{symbol}: {error}"
        )

        return 0.0


# ============================================================
# CHECK TP / SL / EXPIRY
# ============================================================

def evaluate_trade_status(
    trade: Dict[str, Any],
    current_price: float,
) -> Optional[str]:

    if current_price <= 0:
        return None

    direction = normalize_intent(
        trade.get(
            "direction",
            "NEUTRAL",
        )
    )

    stop_loss = safe_float(
        trade.get(
            "stop_loss",
            0,
        )
    )

    take_profit = safe_float(
        trade.get(
            "take_profit",
            0,
        )
    )

    expires_at = parse_datetime(
        trade.get(
            "expires_at"
        )
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if direction == "LONG":

        if take_profit > 0:

            if current_price >= take_profit:

                return "TP_HIT"

        if stop_loss > 0:

            if current_price <= stop_loss:

                return "SL_HIT"

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    elif direction == "SHORT":

        if take_profit > 0:

            if current_price <= take_profit:

                return "TP_HIT"

        if stop_loss > 0:

            if current_price >= stop_loss:

                return "SL_HIT"

    # --------------------------------------------------------
    # EXPIRY
    # --------------------------------------------------------

    if expires_at:

        if now_utc() >= expires_at:

            return "EXPIRED"

    return None


# ============================================================
# LIVE EXCHANGE CLOSE
# ============================================================

def close_live_position(
    trade: Dict[str, Any],
) -> bool:

    if not LIVE_TRADING:

        return True

    symbol = str(
        trade.get(
            "symbol",
            "",
        )
    ).upper().strip()

    amount = safe_float(
        trade.get(
            "amount",
            0,
        )
    )

    direction = normalize_intent(
        trade.get(
            "direction",
            "NEUTRAL",
        )
    )

    if not symbol:
        return False

    if amount <= 0:
        return False

    close_side = (
        "sell"
        if direction == "LONG"
        else "buy"
    )

    try:

        print(
            f"🔴 Closing LIVE position "
            f"{symbol} | "
            f"{close_side}"
        )

        mexc.create_order(
            symbol=symbol,
            type="market",
            side=close_side,
            amount=amount,
        )

        return True

    except Exception as error:

        print(
            f"❌ Live close error: "
            f"{error}"
        )

        return False


# ============================================================
# MONITOR OPEN TRADES
# ============================================================

def monitor_open_trades():

    df = get_active_trades()

    if df.empty:
        return

    if "status" not in df.columns:
        return

    open_rows = df[
        df["status"]
        .astype(str)
        .str.upper()
        == "OPEN"
    ]

    if open_rows.empty:
        return

    for _, row in open_rows.iterrows():

        trade = row.to_dict()

        symbol = str(
            trade.get(
                "symbol",
                "",
            )
        ).upper().strip()

        if not symbol:
            continue

        current_price = (
            fetch_current_price(
                symbol
            )
        )

        if current_price <= 0:
            continue

        status = evaluate_trade_status(
            trade,
            current_price,
        )

        if status is None:

            expires_at = parse_datetime(
                trade.get(
                    "expires_at"
                )
            )

            remaining = ""

            if expires_at:

                seconds = (
                    expires_at
                    - now_utc()
                ).total_seconds()

                remaining = max(
                    0,
                    int(
                        seconds
                        / 60
                    ),
                )

            print(
                f"📊 OPEN | "
                f"{symbol} | "
                f"{trade.get('direction')} | "
                f"Price {current_price} | "
                f"{remaining}m left"
            )

            continue

        # ----------------------------------------------------
        # Close live exchange position
        # ----------------------------------------------------

        if LIVE_TRADING:

            if not close_live_position(
                trade
            ):

                print(
                    f"❌ Could not close "
                    f"live position "
                    f"{symbol}"
                )

                continue

        # ----------------------------------------------------
        # Close local record
        # ----------------------------------------------------

        close_active_trade_record(
            trade,
            current_price,
            status,
        )


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_amount(
    symbol: str,
    entry_price: float,
    trade_usdt: float,
) -> float:

    if entry_price <= 0:
        return 0.0

    if trade_usdt <= 0:
        return 0.0

    amount = (
        trade_usdt
        / entry_price
    )

    try:

        amount = float(
            mexc.amount_to_precision(
                symbol,
                amount,
            )
        )

    except Exception:

        amount = float(
            amount
        )

    return amount


# ============================================================
# EXECUTE PAPER TRADE
# ============================================================

def execute_paper_trade(
    row: pd.Series,
    stop_loss: float,
    take_profit: float,
    amount: float,
) -> Optional[Dict[str, Any]]:

    symbol = str(
        row.get(
            "symbol",
            "",
        )
    ).upper().strip()

    intent = normalize_intent(
        row.get(
            "intent",
            "NEUTRAL",
        )
    )

    entry = safe_float(
        row.get(
            "current_price",
            0,
        )
    )

    trade = create_active_trade(
        row=row,
        order_result={
            "id":
                "PAPER-"
                + str(
                    uuid.uuid4()
                )
        },
        stop_loss=stop_loss,
        take_profit=take_profit,
        amount=amount,
    )

    if add_active_trade(
        trade
    ):

        print()
        print(
            "📝 PAPER TRADE OPENED"
        )

        print(
            f"Symbol       : "
            f"{symbol}"
        )

        print(
            f"Direction    : "
            f"{intent}"
        )

        print(
            f"Entry        : "
            f"{entry}"
        )

        print(
            f"SL           : "
            f"{stop_loss}"
        )

        print(
            f"TP           : "
            f"{take_profit}"
        )

        print(
            f"Max Hold     : "
            f"{trade['max_hold_minutes']} min"
        )

        print(
            f"Expires      : "
            f"{trade['expires_at']}"
        )

        print(
            f"Trade ID     : "
            f"{trade['trade_id']}"
        )

        print()

        return trade

    return None


# ============================================================
# EXECUTE LIVE TRADE
# ============================================================

def execute_live_trade(
    row: pd.Series,
    config: Dict[str, Any],
    stop_loss: float,
    take_profit: float,
    amount: float,
) -> Optional[Dict[str, Any]]:

    symbol = str(
        row.get(
            "symbol",
            "",
        )
    ).upper().strip()

    intent = normalize_intent(
        row.get(
            "intent",
            "NEUTRAL",
        )
    )

    side = (
        "buy"
        if intent == "LONG"
        else "sell"
    )

    try:

        order = mexc.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
        )

        order_id = order.get(
            "id",
            "UNKNOWN",
        )

        print()
        print(
            "🚀 LIVE ORDER SUCCESS"
        )

        print(
            f"Order ID: "
            f"{order_id}"
        )

        print()

        trade = create_active_trade(
            row=row,
            order_result={
                "id":
                    order_id
            },
            stop_loss=stop_loss,
            take_profit=take_profit,
            amount=amount,
        )

        # NOTE:
        # Local monitoring is still used
        # for TP/SL/expiry.
        #
        # Exchange-native TP/SL behavior
        # depends on MEXC/CCXT market support.

        if add_active_trade(
            trade
        ):

            return trade

        return None

    except Exception as error:

        print()
        print(
            "❌ LIVE ORDER ERROR"
        )

        print(
            error
        )

        print()

        return None


# ============================================================
# EXECUTE TRADE
# ============================================================

def execute_trade(
    row: pd.Series,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    symbol = str(
        row.get(
            "symbol",
            "",
        )
    ).upper().strip()

    intent = normalize_intent(
        row.get(
            "intent",
            "NEUTRAL",
        )
    )

    if not symbol:

        print(
            "❌ Symbol missing."
        )

        return None

    if intent == "NEUTRAL":

        print(
            "⏸️ NEUTRAL - no trade."
        )

        return None

    # --------------------------------------------------------
    # Duplicate protection
    # --------------------------------------------------------

    allow_multiple = bool(
        config.get(
            "allow_multiple_positions",
            False,
        )
    )

    if (
        not allow_multiple
        and has_active_trade(
            symbol
        )
    ):

        print(
            f"⏸️ Existing OPEN trade "
            f"for {symbol}. "
            f"New trade rejected."
        )

        return None

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not validate_signal(
        row
    ):

        print(
            f"❌ Signal rejected "
            f"for {symbol}"
        )

        return None

    entry_price = safe_float(
        row.get(
            "current_price",
            0,
        )
    )

    stop_loss = safe_float(
        row.get(
            "stop_loss",
            0,
        )
    )

    take_profit = safe_float(
        row.get(
            "take_profit",
            0,
        )
    )

    # --------------------------------------------------------
    # RR
    # --------------------------------------------------------

    minimum_rr = safe_float(
        config.get(
            "min_rr",
            DEFAULT_MIN_RR,
        ),
        DEFAULT_MIN_RR,
    )

    stop_loss, take_profit = (
        apply_minimum_rr(
            intent=intent,
            entry=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            minimum_rr=minimum_rr,
        )
    )

    rr = calculate_rr(
        entry_price,
        stop_loss,
        take_profit,
    )

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    trade_mode = normalize_trade_mode(
        row.get(
            "trade_mode",
            row.get(
                "TRADE_MODE",
                "SCALPING",
            ),
        )
    )

    max_hold = (
        get_max_hold_minutes(
            trade_mode
        )
    )

    print_signal_information(
        row,
        stop_loss,
        take_profit,
    )

    print(
        f"⏱️ Trade Duration: "
        f"{trade_mode} -> "
        f"{max_hold} minutes"
    )

    # --------------------------------------------------------
    # Leverage
    # --------------------------------------------------------

    leverage = safe_int(
        config.get(
            "leverage",
            DEFAULT_LEVERAGE,
        ),
        DEFAULT_LEVERAGE,
    )

    leverage = max(
        1,
        leverage,
    )

    if LIVE_TRADING:

        set_leverage_and_margin(
            symbol,
            leverage,
        )

    # --------------------------------------------------------
    # Amount
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

    amount = calculate_amount(
        symbol,
        entry_price,
        trade_usdt,
    )

    if amount <= 0:

        print(
            "❌ Amount calculated "
            "as zero."
        )

        return None

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print(
        "=================================================="
    )

    print(
        f"🚀 {intent} "
        f"{'LIVE' if LIVE_TRADING else 'PAPER'}"
    )

    print(
        f"Symbol       : "
        f"{symbol}"
    )

    print(
        f"Trade Mode   : "
        f"{trade_mode}"
    )

    print(
        f"Entry        : "
        f"{entry_price}"
    )

    print(
        f"SL           : "
        f"{stop_loss}"
    )

    print(
        f"TP           : "
        f"{take_profit}"
    )

    print(
        f"RR           : "
        f"1:{rr:.2f}"
    )

    print(
        f"Amount       : "
        f"{amount}"
    )

    print(
        f"Max Hold     : "
        f"{max_hold} minutes"
    )

    print(
        "=================================================="
    )

    print()

    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    if not LIVE_TRADING:

        return execute_paper_trade(
            row,
            stop_loss,
            take_profit,
            amount,
        )

    # --------------------------------------------------------
    # LIVE
    # --------------------------------------------------------

    return execute_live_trade(
        row,
        config,
        stop_loss,
        take_profit,
        amount,
    )


# ============================================================
# LATEST SIGNAL
# ============================================================

def get_latest_signal(
) -> Optional[pd.Series]:

    df = load_csv(
        SIGNAL_FILE
    )

    if df.empty:
        return None

    if "timestamp" in df.columns:

        try:

            df["_parsed_time"] = (
                pd.to_datetime(
                    df["timestamp"],
                    errors="coerce",
                    utc=True,
                )
            )

            df = df.sort_values(
                "_parsed_time",
                ascending=False,
            )

        except Exception:
            pass

    return df.iloc[0]


# ============================================================
# SIGNAL AGE
# ============================================================

def signal_is_fresh(
    row: pd.Series,
) -> bool:

    timestamp = row.get(
        "timestamp",
        None,
    )

    if timestamp is None:
        return True

    parsed = parse_datetime(
        timestamp
    )

    if parsed is None:
        return True

    age_seconds = (
        now_utc() - parsed
    ).total_seconds()

    trade_mode = normalize_trade_mode(
        row.get(
            "trade_mode",
            row.get(
                "TRADE_MODE",
                "SCALPING",
            ),
        )
    )

    # Don't allow an old signal
    # to suddenly open a new trade.

    max_age = (
        get_max_hold_minutes(
            trade_mode
        )
        * 60
    )

    # Slight tolerance
    max_age += 60

    return (
        age_seconds
        <= max_age
    )


# ============================================================
# MIN CONFIDENCE
# ============================================================

def passes_confidence_filter(
    row: pd.Series,
    config: Dict[str, Any],
) -> bool:

    min_confidence = safe_float(
        config.get(
            "min_confidence",
            0,
        )
    )

    if min_confidence <= 0:
        return True

    confidence = safe_float(
        row.get(
            "confidence",
            0,
        )
    )

    if confidence <= 0:
        return False

    return (
        confidence
        >= min_confidence
    )


# ============================================================
# BOT STATUS
# ============================================================

def print_engine_status():

    active = get_active_trades()

    if active.empty:

        print(
            "📭 No active trades."
        )

        return

    if "status" not in active.columns:

        print(
            "⚠️ Active trade file "
            "missing status."
        )

        return

    open_trades = active[
        active["status"]
        .astype(str)
        .str.upper()
        == "OPEN"
    ]

    print()
    print(
        f"📌 Active trades: "
        f"{len(open_trades)}"
    )

    for _, row in open_trades.iterrows():

        symbol = row.get(
            "symbol",
            "",
        )

        direction = row.get(
            "direction",
            "",
        )

        mode = row.get(
            "trade_mode",
            "",
        )

        expires = row.get(
            "expires_at",
            "",
        )

        print(
            f"   {symbol} | "
            f"{direction} | "
            f"{mode} | "
            f"Expires {expires}"
        )


# ============================================================
# MAIN BOT
# ============================================================

def run_bot():

    processed_signals = set()

    print()
    print(
        "=" * 70
    )

    print(
        "🤖 ZIA RESEARCH "
        "EXECUTION ENGINE"
    )

    print(
        "=" * 70
    )

    print(
        "30M / SCALPING : "
        "15 MINUTES"
    )

    print(
        "1H             : "
        "90 MINUTES"
    )

    print(
        "4H             : "
        "24 HOURS"
    )

    print(
        f"Execution Mode : "
        f"{'LIVE' if LIVE_TRADING else 'PAPER'}"
    )

    print(
        "=" * 70
    )

    while True:

        try:

            # =================================================
            # CONFIG
            # =================================================

            config = load_config()

            # =================================================
            # ALWAYS MONITOR EXISTING TRADES
            # =================================================

            monitor_open_trades()

            # =================================================
            # BOT OFF
            # =================================================

            if not config.get(
                "is_running",
                False,
            ):

                time.sleep(
                    safe_int(
                        config.get(
                            "poll_interval",
                            POLL_INTERVAL,
                        ),
                        POLL_INTERVAL,
                    )
                )

                continue

            # =================================================
            # SIGNAL
            # =================================================

            latest = (
                get_latest_signal()
            )

            if latest is None:

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            # =================================================
            # SYMBOL
            # =================================================

            symbol = str(
                latest.get(
                    "symbol",
                    "",
                )
            ).upper().strip()

            if not symbol:

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            # =================================================
            # COIN FILTER
            # =================================================

            selected_coins = [

                str(x)
                .upper()
                .strip()

                for x in config.get(
                    "selected_coins",
                    [],
                )

            ]

            if (
                selected_coins
                and symbol
                not in selected_coins
            ):

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            # =================================================
            # SIGNAL ID
            # =================================================

            signal_id = (
                make_signal_id(
                    latest
                )
            )

            if signal_id in (
                processed_signals
            ):

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            # =================================================
            # FRESHNESS
            # =================================================

            if not signal_is_fresh(
                latest
            ):

                print(
                    f"⏳ Old signal ignored "
                    f"for {symbol}"
                )

                processed_signals.add(
                    signal_id
                )

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            # =================================================
            # INTENT
            # =================================================

            intent = normalize_intent(
                latest.get(
                    "intent",
                    latest.get(
                        "direction",
                        "NEUTRAL",
                    ),
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

            # =================================================
            # CONFIDENCE
            # =================================================

            if not passes_confidence_filter(
                latest,
                config,
            ):

                confidence = (
                    safe_float(
                        latest.get(
                            "confidence",
                            0,
                        )
                    )
                )

                minimum = (
                    safe_float(
                        config.get(
                            "min_confidence",
                            0,
                        )
                    )
                )

                print(
                    f"⏸️ Confidence "
                    f"{confidence:.2f} "
                    f"< {minimum:.2f}"
                )

                processed_signals.add(
                    signal_id
                )

                time.sleep(
                    POLL_INTERVAL
                )

                continue

            # =================================================
            # EXECUTE
            # =================================================

            result = execute_trade(
                latest,
                config,
            )

            processed_signals.add(
                signal_id
            )

            # =================================================
            # MEMORY PROTECTION
            # =================================================

            if len(
                processed_signals
            ) > 5000:

                processed_signals = set(
                    list(
                        processed_signals
                    )[-2500:]
                )

            if result:

                print(
                    f"✅ Trade opened "
                    f"for {symbol}"
                )

            else:

                print(
                    f"⚠️ No trade opened "
                    f"for {symbol}"
                )

            # =================================================
            # STATUS
            # =================================================

            print_engine_status()

        except KeyboardInterrupt:

            print()
            print(
                "🛑 Bot stopped."
            )

            break

        except Exception as error:

            print()
            print(
                "🔥 BOT ERROR"
            )

            print(
                error
            )

            traceback.print_exc()

            print(
                "Retrying..."
            )

            print()

        config = load_config()

        sleep_time = safe_int(
            config.get(
                "poll_interval",
                POLL_INTERVAL,
            ),
            POLL_INTERVAL,
        )

        sleep_time = max(
            1,
            sleep_time,
        )

        time.sleep(
            sleep_time
        )


# ============================================================
# TEST / STATUS
# ============================================================

def print_trade_duration_table():

    print()
    print(
        "=" * 60
    )

    print(
        "TRADE DURATION RULES"
    )

    print(
        "=" * 60
    )

    print(
        "SCALPING / 30M -> "
        "15 minutes"
    )

    print(
        "1H              -> "
        "90 minutes"
    )

    print(
        "4H              -> "
        "24 hours"
    )

    print(
        "=" * 60
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print_trade_duration_table()

    print(
        "ZIA RESEARCH "
        "EXECUTION ENGINE READY"
    )

    print(
        f"Mode: "
        f"{'LIVE' if LIVE_TRADING else 'PAPER'}"
    )

    print()

    run_bot()
