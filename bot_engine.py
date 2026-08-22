import os
import time
import json
import math
import pandas as pd
import ccxt


# ============================================================
# CONFIG
# ============================================================

CONFIG_FILE = "config.json"
SIGNAL_FILE = "signal_history.csv"
TRADE_LOG_FILE = "bot_trade_log.csv"

# IMPORTANT:
# False = Paper trading
# True  = Real MEXC futures orders
LIVE_TRADING = False

DEFAULT_CONFIG = {
    "is_running": False,

    "leverage": 5,

    # Risk / Reward
    "rr": 2,
    "sl_pct": 0.0020,       # 0.20%
    "tp_pct": 0.0040,       # 0.40% for 1:2

    # Minimum expected movement
    "min_move_pct": 0.0040, # 0.40%

    # Signal filters
    "min_confidence": 0.70,
    "min_obi": 0.20,

    # Position sizing
    "trade_amount_usdt": 10.0,

    # Safety
    "max_open_positions": 1,

    "selected_coins": [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT"
    ]
}


# ============================================================
# MEXC
# ============================================================

mexc = ccxt.mexc({
    "apiKey": os.getenv(
        "MEXC_API_KEY",
        ""
    ),
    "secret": os.getenv(
        "MEXC_SECRET_KEY",
        ""
    ),
    "enableRateLimit": True,

    "options": {
        "defaultType": "swap"
    }
})


# ============================================================
# CONFIG LOADER
# ============================================================

def load_config():

    if not os.path.exists(CONFIG_FILE):

        with open(
            CONFIG_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                DEFAULT_CONFIG,
                f,
                indent=4
            )

        return DEFAULT_CONFIG.copy()

    try:

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            cfg = json.load(f)

    except Exception as e:

        print(
            f"❌ Config error: {e}"
        )

        return DEFAULT_CONFIG.copy()

    # Fill missing values
    for key, value in DEFAULT_CONFIG.items():

        if key not in cfg:
            cfg[key] = value

    return cfg


# ============================================================
# SYMBOL NORMALIZATION
# ============================================================

def normalize_symbol(symbol):

    symbol = str(symbol).upper().strip()

    conversions = {
        "BTCUSDT": "BTC/USDT:USDT",
        "ETHUSDT": "ETH/USDT:USDT",
        "SOLUSDT": "SOL/USDT:USDT",
        "BNBUSDT": "BNB/USDT:USDT",
        "XRPUSDT": "XRP/USDT:USDT",
        "DOGEUSDT": "DOGE/USDT:USDT",
        "ADAUSDT": "ADA/USDT:USDT",
        "AVAXUSDT": "AVAX/USDT:USDT",
        "DOTUSDT": "DOT/USDT:USDT",
        "LINKUSDT": "LINK/USDT:USDT"
    }

    return conversions.get(
        symbol,
        symbol
    )


# ============================================================
# LOAD MARKETS
# ============================================================

def load_markets():

    try:

        mexc.load_markets()

        print(
            "✅ MEXC markets loaded."
        )

        return True

    except Exception as e:

        print(
            f"❌ Market loading error: {e}"
        )

        return False


# ============================================================
# LEVERAGE
# ============================================================

def set_leverage(
    symbol,
    leverage
):

    symbol = normalize_symbol(
        symbol
    )

    try:

        mexc.set_leverage(
            leverage,
            symbol
        )

        print(
            f"⚙️ Leverage: {leverage}x | {symbol}"
        )

        return True

    except Exception as e:

        print(
            f"⚠️ Leverage error: {e}"
        )

        return False


# ============================================================
# POSITION CHECK
# ============================================================

def get_open_positions():

    if not LIVE_TRADING:
        return []

    try:

        positions = mexc.fetch_positions()

        active = []

        for p in positions:

            contracts = p.get(
                "contracts",
                0
            )

            try:
                contracts = float(
                    contracts or 0
                )
            except:
                contracts = 0

            if abs(contracts) > 0:
                active.append(p)

        return active

    except Exception as e:

        print(
            f"⚠️ Position check error: {e}"
        )

        return []


# ============================================================
# CHECK IF SYMBOL ALREADY OPEN
# ============================================================

def symbol_has_position(
    symbol
):

    symbol = normalize_symbol(
        symbol
    )

    positions = get_open_positions()

    for position in positions:

        position_symbol = position.get(
            "symbol"
        )

        if position_symbol == symbol:

            return True

    return False


# ============================================================
# SIGNAL VALIDATION
# ============================================================

def validate_signal(
    signal,
    cfg
):

    reasons = []

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction = str(
        signal.get(
            "signal",
            ""
        )
    ).upper()

    if direction in [
        "BUY",
        "LONG",
        "1"
    ]:

        direction = "LONG"

    elif direction in [
        "SELL",
        "SHORT",
        "-1"
    ]:

        direction = "SHORT"

    else:

        reasons.append(
            "Invalid signal direction"
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(
        signal.get(
            "confidence",
            0
        ) or 0
    )

    if confidence > 1:
        confidence /= 100.0

    min_confidence = float(
        cfg.get(
            "min_confidence",
            0.70
        )
    )

    if confidence < min_confidence:

        reasons.append(
            f"Confidence too low: "
            f"{confidence:.2%}"
        )

    # --------------------------------------------------------
    # OBI
    # --------------------------------------------------------

    obi = float(
        signal.get(
            "obi_top20",
            0
        ) or 0
    )

    min_obi = float(
        cfg.get(
            "min_obi",
            0.20
        )
    )

    if abs(obi) < min_obi:

        reasons.append(
            f"OBI too weak: {obi:.3f}"
        )

    # --------------------------------------------------------
    # Direction / OBI agreement
    # --------------------------------------------------------

    if direction == "LONG" and obi < min_obi:

        reasons.append(
            "LONG signal not confirmed by OBI"
        )

    if direction == "SHORT" and obi > -min_obi:

        reasons.append(
            "SHORT signal not confirmed by OBI"
        )

    # --------------------------------------------------------
    # Minimum move
    # --------------------------------------------------------

    expected_move = float(
        signal.get(
            "expected_move",
            signal.get(
                "move_pct",
                0
            )
        ) or 0
    )

    if expected_move > 1:

        expected_move /= 100.0

    min_move = float(
        cfg.get(
            "min_move_pct",
            0.004
        )
    )

    if expected_move < min_move:

        reasons.append(
            f"Expected move "
            f"{expected_move:.2%} < "
            f"minimum {min_move:.2%}"
        )

    # --------------------------------------------------------
    # Risk / Reward
    # --------------------------------------------------------

    rr = float(
        cfg.get(
            "rr",
            2
        )
    )

    if rr < 2:

        reasons.append(
            "RR below minimum 1:2"
        )

    return (
        len(reasons) == 0,
        direction,
        reasons
    )


# ============================================================
# CALCULATE TP / SL
# ============================================================

def calculate_tp_sl(
    entry,
    direction,
    cfg
):

    sl_pct = float(
        cfg.get(
            "sl_pct",
            0.002
        )
    )

    rr = float(
        cfg.get(
            "rr",
            2
        )
    )

    tp_pct = sl_pct * rr

    if direction == "LONG":

        sl = (
            entry *
            (1 - sl_pct)
        )

        tp = (
            entry *
            (1 + tp_pct)
        )

    else:

        sl = (
            entry *
            (1 + sl_pct)
        )

        tp = (
            entry *
            (1 - tp_pct)
        )

    return tp, sl, tp_pct, sl_pct


# ============================================================
# TRADE LOG
# ============================================================

def log_trade(
    data
):

    row = pd.DataFrame([
        data
    ])

    if not os.path.exists(
        TRADE_LOG_FILE
    ):

        row.to_csv(
            TRADE_LOG_FILE,
            index=False
        )

    else:

        row.to_csv(
            TRADE_LOG_FILE,
            mode="a",
            header=False,
            index=False
        )


# ============================================================
# PAPER TRADE
# ============================================================

def paper_trade(
    symbol,
    direction,
    cfg
):

    try:

        ticker = mexc.fetch_ticker(
            symbol
        )

        entry = float(
            ticker["last"]
        )

    except Exception as e:

        print(
            f"❌ Price error: {e}"
        )

        return False

    tp, sl, tp_pct, sl_pct = calculate_tp_sl(
        entry,
        direction,
        cfg
    )

    print("\n" + "=" * 60)
    print("🧪 PAPER TRADE")
    print("=" * 60)

    print(
        f"Symbol     : {symbol}"
    )

    print(
        f"Direction  : {direction}"
    )

    print(
        f"Entry      : {entry}"
    )

    print(
        f"Stop Loss  : {sl}"
    )

    print(
        f"Take Profit: {tp}"
    )

    print(
        f"Risk       : {sl_pct:.2%}"
    )

    print(
        f"Reward     : {tp_pct:.2%}"
    )

    print(
        f"RR         : 1:{cfg.get('rr', 2)}"
    )

    print("=" * 60)

    log_trade({

        "timestamp":
            pd.Timestamp.utcnow(),

        "symbol":
            symbol,

        "direction":
            direction,

        "entry":
            entry,

        "stop_loss":
            sl,

        "take_profit":
            tp,

        "rr":
            f"1:{cfg.get('rr', 2)}",

        "mode":
            "PAPER"

    })

    return True


# ============================================================
# REAL ORDER
# ============================================================

def execute_live_trade(
    symbol,
    direction,
    cfg
):

    if not LIVE_TRADING:

        return paper_trade(
            symbol,
            direction,
            cfg
        )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not os.getenv(
        "MEXC_API_KEY"
    ) or not os.getenv(
        "MEXC_SECRET_KEY"
    ):

        print(
            "❌ MEXC API keys missing."
        )

        return False

    # --------------------------------------------------------
    # Existing position
    # --------------------------------------------------------

    if symbol_has_position(
        symbol
    ):

        print(
            f"⛔ Existing position on {symbol}"
        )

        return False

    leverage = int(
        cfg.get(
            "leverage",
            5
        )
    )

    if not set_leverage(
        symbol,
        leverage
    ):

        return False

    # --------------------------------------------------------
    # Current price
    # --------------------------------------------------------

    try:

        ticker = mexc.fetch_ticker(
            symbol
        )

        entry = float(
            ticker["last"]
        )

    except Exception as e:

        print(
            f"❌ Ticker error: {e}"
        )

        return False

    # --------------------------------------------------------
    # TP / SL
    # --------------------------------------------------------

    tp, sl, tp_pct, sl_pct = calculate_tp_sl(
        entry,
        direction,
        cfg
    )

    # --------------------------------------------------------
    # Position amount
    # --------------------------------------------------------

    usdt_amount = float(
        cfg.get(
            "trade_amount_usdt",
            10.0
        )
    )

    notional = (
        usdt_amount *
        leverage
    )

    amount = (
        notional /
        entry
    )

    try:

        amount = float(
            mexc.amount_to_precision(
                symbol,
                amount
            )
        )

    except Exception:

        amount = float(
            amount
        )

    if amount <= 0:

        print(
            "❌ Invalid order amount."
        )

        return False

    # --------------------------------------------------------
    # Side
    # --------------------------------------------------------

    side = (
        "buy"
        if direction == "LONG"
        else "sell"
    )

    print("\n" + "=" * 60)
    print("🚨 LIVE TRADE")
    print("=" * 60)

    print(
        f"Symbol    : {symbol}"
    )

    print(
        f"Direction : {direction}"
    )

    print(
        f"Amount    : {amount}"
    )

    print(
        f"Leverage  : {leverage}x"
    )

    print(
        f"Entry     : {entry}"
    )

    print(
        f"TP        : {tp}"
    )

    print(
        f"SL        : {sl}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # MARKET ENTRY
    # --------------------------------------------------------

    try:

        order = mexc.create_order(
            symbol,
            "market",
            side,
            amount
        )

        print(
            "✅ Entry order submitted."
        )

        print(
            f"Order ID: "
            f"{order.get('id')}"
        )

    except Exception as e:

        print(
            f"❌ Entry order failed: {e}"
        )

        return False

    # --------------------------------------------------------
    # IMPORTANT
    #
    # TP/SL order parameters differ by MEXC contract mode.
    # We do NOT blindly submit exchange-specific conditional
    # parameters here.
    #
    # The position must be verified before attaching exits.
    # --------------------------------------------------------

    log_trade({

        "timestamp":
            pd.Timestamp.utcnow(),

        "symbol":
            symbol,

        "direction":
            direction,

        "entry":
            entry,

        "amount":
            amount,

        "leverage":
            leverage,

        "stop_loss":
            sl,

        "take_profit":
            tp,

        "rr":
            f"1:{cfg.get('rr', 2)}",

        "order_id":
            order.get("id"),

        "mode":
            "LIVE"

    })

    return True


# ============================================================
# READ LATEST SIGNAL
# ============================================================

def get_latest_signal():

    if not os.path.exists(
        SIGNAL_FILE
    ):

        return None

    try:

        df = pd.read_csv(
            SIGNAL_FILE
        )

    except Exception as e:

        print(
            f"❌ Signal CSV error: {e}"
        )

        return None

    if df.empty:

        return None

    # Newest timestamp first
    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.sort_values(
            "timestamp",
            ascending=False
        )

    return df.iloc[0].to_dict()


# ============================================================
# BOT ENGINE
# ============================================================

def run_bot():

    processed_signals = set()

    print(
        "🤖 Research Trading Bot started."
    )

    print(
        f"MODE: "
        f"{'LIVE' if LIVE_TRADING else 'PAPER'}"
    )

    if not load_markets():

        return

    while True:

        try:

            cfg = load_config()

            # ------------------------------------------------
            # BOT OFF
            # ------------------------------------------------

            if not cfg.get(
                "is_running",
                False
            ):

                print(
                    "⏸️ Bot OFF"
                )

                time.sleep(5)

                continue

            # ------------------------------------------------
            # Latest signal
            # ------------------------------------------------

            latest = get_latest_signal()

            if latest is None:

                time.sleep(5)

                continue

            symbol = normalize_symbol(
                latest.get(
                    "symbol",
                    ""
                )
            )

            timestamp = str(
                latest.get(
                    "timestamp",
                    ""
                )
            )

            sig_id = (
                f"{timestamp}_"
                f"{symbol}"
            )

            # ------------------------------------------------
            # Duplicate
            # ------------------------------------------------

            if sig_id in processed_signals:

                time.sleep(5)

                continue

            # ------------------------------------------------
            # Selected coins
            # ------------------------------------------------

            selected = [
                normalize_symbol(x)
                for x in cfg.get(
                    "selected_coins",
                    []
                )
            ]

            if symbol not in selected:

                processed_signals.add(
                    sig_id
                )

                print(
                    f"⏭️ {symbol} not selected."
                )

                time.sleep(5)

                continue

            # ------------------------------------------------
            # Validate signal
            # ------------------------------------------------

            valid, direction, reasons = validate_signal(
                latest,
                cfg
            )

            if not valid:

                print(
                    f"\n⛔ SIGNAL REJECTED: "
                    f"{symbol}"
                )

                for reason in reasons:

                    print(
                        f"   • {reason}"
                    )

                processed_signals.add(
                    sig_id
                )

                time.sleep(5)

                continue

            # ------------------------------------------------
            # Maximum positions
            # ------------------------------------------------

            if LIVE_TRADING:

                positions = get_open_positions()

                max_positions = int(
                    cfg.get(
                        "max_open_positions",
                        1
                    )
                )

                if len(positions) >= max_positions:

                    print(
                        "⛔ Maximum open "
                        "positions reached."
                    )

                    processed_signals.add(
                        sig_id
                    )

                    time.sleep(5)

                    continue

            # ------------------------------------------------
            # Existing position
            # ------------------------------------------------

            if LIVE_TRADING and symbol_has_position(
                symbol
            ):

                print(
                    f"⛔ Already have "
                    f"position: {symbol}"
                )

                processed_signals.add(
                    sig_id
                )

                time.sleep(5)

                continue

            # ------------------------------------------------
            # EXECUTE
            # ------------------------------------------------

            print(
                f"\n🚀 VALID SIGNAL: "
                f"{symbol} | {direction}"
            )

            success = execute_live_trade(
                symbol,
                direction,
                cfg
            )

            if success:

                processed_signals.add(
                    sig_id
                )

        except KeyboardInterrupt:

            print(
                "\n🛑 Bot stopped."
            )

            break

        except Exception as e:

            print(
                f"❌ Bot error: {e}"
            )

        time.sleep(5)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    run_bot()
