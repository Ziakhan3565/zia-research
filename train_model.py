import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_trend_aligned_model():
    print("🚀 Training Trend-Aligned Microstructure Model with Advanced Formulas...")
    
    try:
        df = pd.read_csv("market_data_log.csv")
    except FileNotFoundError:
        print("❌ 'market_data_log.csv' nahi mili!")
        return

    # 1. Feature Engineering & Custom Formulas
    # Purane features
    df['bid_ask_ratio'] = df['top20_bid_sum'] / (df['top20_ask_sum'] + 1e-5)
    df['total_depth'] = df['top20_bid_sum'] + df['top20_ask_sum']
    
    df['sma_20'] = df.groupby('symbol')['current_price'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['trend_signal'] = df['current_price'] - df['sma_20']

    # Naye 5 Advanced Formulas (Research Lab wale)
    # Note: Agar CSV mein 'Volume' column nahi hai toh default 0 ya suitable column use karna par sakta hai
    if 'Volume' not in df.columns:
        df['Volume'] = 1.0  # Fallback agar volume column na ho

    # Delta price for taker flow calculation
    df['delta_p'] = df.groupby('symbol')['current_price'].diff().fillna(0)
    
    # Returns for realized volatility
    df['returns'] = df.groupby('symbol')['current_price'].pct_change().fillna(0)
    df['realized_vol'] = df.groupby('symbol')['returns'].transform(lambda x: x.rolling(15, min_periods=1).std()) + 1e-8
    
    # 1. BOOK_IMB
    df['BOOK_IMB'] = (df['top20_bid_sum'] - df['top20_ask_sum']) / (df['top20_bid_sum'] + df['top20_ask_sum'] + 1e-8)

    # 2. TAKER_FLOW
    taker_buy = df['Volume'] * np.where(df['delta_p'] > 0, 1.0, 0.3)
    taker_sell = df['Volume'] * np.where(df['delta_p'] <= 0, 1.0, 0.3)
    df['TAKER_FLOW'] = (taker_buy - taker_sell) / (taker_buy + taker_sell + 1e-8)

    # 3. QUANT_IMPLY (Assuming top level bid/ask quantities if available, or approximating from top20 sums)
    # Agar alag se bid/ask qty column nahi hain toh top20 sums ko scale kar ke use kar rahe hain
    depth_skew = (df['top20_bid_sum'] - df['top20_ask_sum']) / (df['top20_bid_sum'] + df['top20_ask_sum'] + 1e-8)
    df['QUANT_IMPLY'] = np.clip(depth_skew * 1.5, -1, 1)

    # 4. ADAPT_CONF
    ma_fast = df.groupby('symbol')['current_price'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    ma_slow = df.groupby('symbol')['current_price'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df['ADAPT_CONF'] = np.clip((ma_fast - ma_slow) / (df['realized_vol'] * df['current_price'] + 1e-8), -1, 1)

    # 5. BAYESIAN
    prior = 0.745
    likelihood = np.where(df['BOOK_IMB'] > 0, 1.0, 0.25)
    posterior = (likelihood * prior) / ((likelihood * prior) + ((1 - likelihood) * (1 - prior)) + 1e-8)
    df['BAYESIAN'] = np.clip((posterior - 0.5) * 2.0, -1, 1)

    # Updated Features List (Purane + Naye Advanced Formulas)
    features = [
        'top20_bid_sum', 'top20_ask_sum', 'obi_top20', 'spread', 
        'bid_ask_ratio', 'total_depth', 'trend_signal',
        'BOOK_IMB', 'TAKER_FLOW', 'QUANT_IMPLY', 'ADAPT_CONF', 'BAYESIAN'
    ]
    
    df = df.dropna(subset=features + ['current_price', 'symbol']).copy()

    # Target: Aglay 5 steps mein price move
    df['future_price'] = df.groupby('symbol')['current_price'].shift(-5)
    df = df.dropna(subset=['future_price']).copy()
    
    df['target'] = (df['future_price'] > df['current_price']).astype(int)

    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=4,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Model Accuracy with Advanced Formulas: {acc * 100:.2f}%")

    joblib.dump(model, "xgboost_obi_model.pkl")
    print("💾 Model saved successfully as 'xgboost_obi_model.pkl'!")

if __name__ == "__main__":
    train_trend_aligned_model()
