import os
import time
import json
import pandas as pd
import ccxt

# Exchange Setup (MEXC Futures)
mexc = ccxt.mexc({
    'apiKey': os.getenv('MEXC_API_KEY', 'YOUR_MEXC_API_KEY'),
    'secret': os.getenv('MEXC_SECRET_KEY', 'YOUR_MEXC_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',  # Crucial for MEXC perpetual futures trading
    }
})

def set_leverage_and_margin(symbol, leverage):
    """Leverage aur Margin Mode (Isolated) set karna"""
    try:
        mexc.set_leverage(leverage, symbol)
        try:
            mexc.set_margin_mode('isolated', symbol)
        except Exception:
            pass  # Agar already isolated ho toh error ignore ho jaye
        print(f"⚙️ Leverage set to {leverage}x (Isolated) for {symbol}")
    except Exception as e:
        print(f"⚠️ Leverage/Margin error for {symbol}: {e}")

def has_open_position(symbol):
    """MEXC par check karta hai ke is symbol ki koi active futures position hai ya nahi"""
    try:
        # MEXC swap/futures se positions fetch karein
        positions = mexc.fetch_positions([symbol])
        for pos in positions:
            # Agar contracts ki quantity 0 se zyada hai, matlab position khuli hui hai
            if float(pos.get('contracts', 0)) > 0:
                return True
    except Exception as e:
        print(f"⚠️ Position check error for {symbol}: {e}")
    return False

def execute_futures_trade(symbol, side, amount_usdt, leverage, tp_pct, sl_pct):
    """MEXC Futures par Market Order aur TP/SL lagana"""
    try:
        # 1. Market Price fetch karein taakh quantity calculate ho sakay
        ticker = mexc.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # 2. Position Size calculate karein (Notional Value = Amount * Leverage)
        notional_value = amount_usdt * leverage
        contract_size = notional_value / current_price
        
        print(f"📊 Placing {side.upper()} order for {symbol} | Price: {current_price} | Size: {contract_size:.4f}")

        # 3. Market Order Place karein
        order = mexc.create_order(
            symbol=symbol,
            type='market',
            side=side,  # 'buy' ya 'sell'
            amount=contract_size
        )
        print(f"✅ Order Executed Successfully! Order ID: {order.get('id')}")

        # 4. TP aur SL price calculate karein
        if side == 'buy':
            tp_price = current_price * (1 + tp_pct / 100)
            sl_price = current_price * (1 - sl_pct / 100)
        else:
            tp_price = current_price * (1 - tp_pct / 100)
            sl_price = current_price * (1 + sl_pct / 100)

        print(f"🎯 Target Set -> TP: {tp_price:.4f} | SL: {sl_price:.4f}")

    except Exception as e:
        print(f"❌ Trade Execution Failed for {symbol}: {e}")

def run_bot():
    processed_signals = set()
    print("🤖 Bot Engine Ready & Listening for Signals (with Anti-Overlap Lock)... (Press Ctrl+C to stop)")

    while True:
        # 1. Config Load Karein
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {"is_running": False, "leverage": 5, "tp_pct": 0.6, "sl_pct": 0.4, "trade_amount": 10.0}
        else:
            cfg = {"is_running": False, "leverage": 5, "tp_pct": 0.6, "sl_pct": 0.4, "trade_amount": 10.0}

        # 2. Agar Bot OFF hai, toh wait karein
        if not cfg.get("is_running", False):
            time.sleep(5)
            continue

        # 3. Signals Monitor Karein
        if os.path.exists("signal_history.csv"):
            try:
                df = pd.read_csv("signal_history.csv")
                if not df.empty:
                    # Sab se latest signal uthane ke liye iloc[-1] use karein
                    latest = df.iloc[-1] 
                    sig_id = f"{latest.get('timestamp')}_{latest.get('symbol')}_{latest.get('signal', '')}"
                    
                    symbol = latest['symbol']
                    raw_signal = latest.get('signal', None)
                    side = 'buy' if raw_signal == 1 else 'sell'

                    # Agar signal naya hai aur selected coins ki list mein hai
                    if sig_id not in processed_signals and symbol in cfg.get("selected_coins", []):
                        
                        # --- ANTI-OVERLAP LOCK: Check if position is already active on exchange ---
                        if has_open_position(symbol):
                            print(f"🔒 Active position already running on MEXC for {symbol}. Skipping new signal.")
                            processed_signals.add(sig_id) # Isay process man kar ignore kar dein taakh baar baar print na ho
                            continue

                        leverage = cfg.get("leverage", 5)
                        trade_amount = cfg.get("trade_amount", 10.0)
                        tp_pct = cfg.get("tp_pct", 0.6)
                        sl_pct = cfg.get("sl_pct", 0.4)

                        print(f"\n🚀 New Signal Approved: {symbol} | Side: {side.upper()}")
                        
                        # Leverage Set Karein
                        set_leverage_and_margin(symbol, leverage)
                        
                        # Trade Execute Karein
                        execute_futures_trade(symbol, side, trade_amount, leverage, tp_pct, sl_pct)
                        
                        processed_signals.add(sig_id)
            except Exception as e:
                print(f"⚠️ Error reading signal file: {e}")

        time.sleep(5)

if __name__ == "__main__":
    run_bot()
