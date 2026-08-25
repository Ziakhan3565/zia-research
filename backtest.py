import time
import datetime
import os
import numpy as np
import pandas as pd
import joblib

def run_filtered_trend_backtest():
    print("🧪 Running Optimized Trend-Filtered Microstructure Backtest with Fourier Features...\n")
    
    try:
        df = pd.read_csv("market_data_log.csv")
        model = joblib.load("xgboost_obi_model.pkl")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # --- 1. Feature Engineering & Fourier Trend Integration ---
    df['bid_ask_ratio'] = df['top20_bid_sum'] / (df['top20_ask_sum'] + 1e-5)
    df['total_depth'] = df['top20_bid_sum'] + df['top20_ask_sum']
    df['sma_20'] = df.groupby('symbol')['current_price'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['trend_signal'] = df['current_price'] - df['sma_20']
    df['volatility_proxy'] = df.groupby('symbol')['current_price'].transform(lambda x: x.rolling(10, min_periods=1).std().fillna(0))
    df['hawkes_intensity'] = df.groupby('symbol')['obi_top20'].transform(lambda x: x.rolling(5, min_periods=1).mean().abs())
    df['book_pressure'] = df['obi_top20'] * df['total_depth']

    # Fourier Trend Feature Calculation for Backtest DataFrame per symbol
    def compute_fourier_trend(sub_df):
        prices = sub_df['current_price'].values
        if len(prices) < 15:
            return pd.Series(0.0, index=sub_df.index)
        xc = prices - np.mean(prices)
        fft_vals = np.fft.fft(xc)
        num_keep = max(1, int(len(fft_vals) * 0.15))
        fft_masked = np.zeros_like(fft_vals)
        fft_masked[:num_keep] = fft_vals[:num_keep]
        fft_masked[-num_keep:] = fft_vals[-num_keep:]
        trend_curve = np.real(np.fft.ifft(fft_masked))
        diffs = np.gradient(trend_curve)
        return pd.Series(diffs, index=sub_df.index)

    df['FOURIER_TREND'] = df.groupby('symbol').apply(compute_fourier_trend).reset_index(level=0, drop=True)

    # Base features pool including Fourier Trend
    features = [
        'top20_bid_sum', 'top20_ask_sum', 'obi_top20', 'spread', 
        'bid_ask_ratio', 'total_depth', 'trend_signal', 
        'volatility_proxy', 'hawkes_intensity', 'book_pressure', 'FOURIER_TREND'
    ]
    
    # Automatically adjust feature list based on what the loaded model expects
    if hasattr(model, "n_features_in_"):
        expected_n = model.n_features_in_
        if expected_n <= len(features):
            features = features[:expected_n]

    df = df.dropna(subset=features + ['current_price', 'symbol']).copy()
    
    if 'timestamp' in df.columns:
        df = df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)

    df['ml_signal'] = model.predict(df[features])
    
    try:
        probs = model.predict_proba(df[features])
        df['confidence'] = np.max(probs, axis=1)
    except Exception:
        df['confidence'] = 1.0

    df_filtered = df[
        (df['confidence'] >= 0.55) & 
        (df['obi_top20'].abs() >= 0.20)
    ].copy()

    initial_balance = 100.0
    current_balance = initial_balance
    trade_amount = 10.0      
    fee_rate = 0.0004        
    
    tp_pct = 0.0060           # 0.60% Take Profit
    sl_pct = 0.0040           # 0.40% Stop Loss

    wins = 0
    losses = 0
    total_profit_loss = 0.0

    # --- 2. Active Trade Lock / Cooldown to Prevent Duplicate Listing ---
    last_trade_index = {}

    for idx, row in df_filtered.iterrows():
        symbol = row['symbol']
        
        # Check if symbol is in cooldown (already active in a recent trade)
        if symbol in last_trade_index and idx < last_trade_index[symbol]:
            continue

        signal = row['ml_signal']
        entry = row['current_price']
        trend = row['trend_signal']

        # Trend Filter
        if signal == 1 and trend < 0: 
            continue
        if signal == 0 and trend > 0: 
            continue

        # Slice future prices step-by-step for accurate TP/SL evaluation
        future_window = df[
            (df['symbol'] == symbol) & 
            (df.index > idx) & 
            (df.index <= idx + 6)
        ]

        if len(future_window) == 0:
            continue

        # Set Lock / Cooldown up to the max index checked
        last_trade_index[symbol] = idx + len(future_window)

        tp_price_long = entry * (1.0 + tp_pct)
        sl_price_long = entry * (1.0 - sl_pct)
        tp_price_short = entry * (1.0 - tp_pct)
        sl_price_short = entry * (1.0 + sl_pct)

        outcome = "PENDING"
        pnl_pct = 0.0

        # Strict Candle-by-Candle Evaluation for precise SL/TP execution
        for _, fut_row in future_window.iterrows():
            high_p = fut_row.get('current_price', entry) 
            low_p = fut_row.get('current_price', entry)
            
            if signal == 1: # LONG
                if high_p >= tp_price_long:
                    outcome = "WIN"
                    pnl_pct = tp_pct - (2 * fee_rate)
                    wins += 1
                    break
                elif low_p <= sl_price_long:
                    outcome = "LOSS"
                    pnl_pct = -sl_pct - (2 * fee_rate)
                    losses += 1
                    break
            elif signal == 0: # SHORT
                if low_p <= tp_price_short:
                    outcome = "WIN"
                    pnl_pct = tp_pct - (2 * fee_rate)
                    wins += 1
                    break
                elif high_p >= sl_price_short:
                    outcome = "LOSS"
                    pnl_pct = -sl_pct - (2 * fee_rate)
                    losses += 1
                    break

        if outcome == "PENDING":
            continue

        trade_pnl = trade_amount * pnl_pct
        current_balance += trade_pnl
        total_profit_loss += trade_pnl

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_return_pct = ((current_balance - initial_balance) / initial_balance) * 100

    print("=" * 40)
    print("📊 OPTIMIZED P&L REPORT (FOURIER + OBI INTEGRATED)")
    print("=" * 40)
    print(f" Starting Capital     : ${initial_balance:.2f}")
    print(f" Total Executed Trades: {total_trades}")
    print(f" ✅ TP Hit (Wins)      : {wins}")
    print(f" ❌ SL Hit (Losses)    : {losses}")
    print(f" 🎯 Win Rate           : {win_rate:.2f}%")
    print("-" * 40)
    print(f" 💰 Total Net Profit   : ${total_profit_loss:.2f}")
    print(f" 📈 Account Growth %   : {total_return_pct:.2f}%")
    print(f" 🏦 Final Balance      : ${current_balance:.2f}")
    print("=" * 40)

if __name__ == "__main__":
    run_filtered_trend_backtest()
