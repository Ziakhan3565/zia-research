import pandas as pd
import numpy as np
import joblib

def run_filtered_trend_backtest():
    print("🧪 Running Optimized & Clean Microstructure Backtest...\n")
    
    try:
        df = pd.read_csv("market_data_log.csv")
        model = joblib.load("xgboost_obi_model.pkl")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # Ensure chronological sorting per symbol
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)

    # Feature Engineering
    df['bid_ask_ratio'] = df['top20_bid_sum'] / (df['top20_ask_sum'] + 1e-5)
    df['total_depth'] = df['top20_bid_sum'] + df['top20_ask_sum']
    df['sma_20'] = df.groupby('symbol')['current_price'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['trend_signal'] = df['current_price'] - df['sma_20']

    features = ['top20_bid_sum', 'top20_ask_sum', 'obi_top20', 'spread', 'bid_ask_ratio', 'total_depth', 'trend_signal']
    
    df = df.dropna(subset=features + ['current_price', 'symbol']).copy()

    # Generate Model Predictions
    df['ml_signal'] = model.predict(df[features])
    
    try:
        probs = model.predict_proba(df[features])
        df['confidence'] = np.max(probs, axis=1)
    except:
        df['confidence'] = 1.0

    # Signal Filter Flags
    df['valid_signal'] = (
        (df['confidence'] >= 0.55) & 
        (df['obi_top20'].abs() >= 0.20)
    )

    initial_balance = 100.0
    current_balance = initial_balance
    trade_amount = 10.0      
    fee_rate = 0.0004        
    
    # 1:1.5 Risk-Reward Ratio
    tp_pct = 0.0060           # 0.60% Take Profit
    sl_pct = 0.0040           # 0.40% Stop Loss

    wins = 0
    losses = 0
    total_profit_loss = 0.0

    # Loop strictly per symbol using clean numpy arrays to prevent any cross-contamination
    for symbol, group in df.groupby('symbol'):
        group = group.reset_index(drop=True)
        prices = group['current_price'].values
        signals = group['ml_signal'].values
        trends = group['trend_signal'].values
        valid_flags = group['valid_signal'].values

        for i in range(len(group) - 6):
            if not valid_flags[i]:
                continue

            signal = signals[i]
            entry = prices[i]
            trend = trends[i]

            # Rule: Filter Out Against-Trend Signals
            if signal == 1 and trend < 0:  # Long only in Uptrend
                continue
            if signal == 0 and trend > 0:  # Short only in Downtrend
                continue

            # Look ahead strictly within the next 6 periods for THIS symbol only
            future_prices = prices[i + 1 : i + 7]
            if len(future_prices) == 0:
                continue

            max_p = np.max(future_prices)
            min_p = np.min(future_prices)

            pnl_pct = 0.0
            trade_resolved = False

            if signal == 1:  # BUY
                if (max_p - entry) / entry >= tp_pct:
                    pnl_pct = tp_pct - (2 * fee_rate)
                    wins += 1
                    trade_resolved = True
                elif (entry - min_p) / entry >= sl_pct:
                    pnl_pct = -sl_pct - (2 * fee_rate)
                    losses += 1
                    trade_resolved = True
            elif signal == 0:  # SELL
                if (entry - min_p) / entry >= tp_pct:
                    pnl_pct = tp_pct - (2 * fee_rate)
                    wins += 1
                    trade_resolved = True
                elif (max_p - entry) / entry >= sl_pct:
                    pnl_pct = -sl_pct - (2 * fee_rate)
                    losses += 1
                    trade_resolved = True

            if trade_resolved:
                trade_pnl = trade_amount * pnl_pct
                current_balance += trade_pnl
                total_profit_loss += trade_pnl

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_return_pct = ((current_balance - initial_balance) / initial_balance) * 100

    print("=" * 40)
    print("📊 TREND-FILTERED BACKTEST REPORT")
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
