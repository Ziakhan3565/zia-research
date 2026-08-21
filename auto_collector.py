import time
import datetime
import os
import numpy as np
import pandas as pd
import requests

COINS_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "LTCUSDT", "BCHUSDT", "APTUSDT", "TRXUSDT",
    "SHIBUSDT", "UNIUSDT", "ATOMUSDT", "SUIUSDT", "INJUSDT", "ICPUSDT"
]

def fetch_order_book_and_price(symbol, depth_limit=20):
    try:
        session = requests.Session()
        # Fetch Price
        price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        price_res = session.get(price_url, timeout=5)
        if price_res.status_code != 200:
            return None, None, None
        current_price = float(price_res.json()['price'])

        # Fetch Depth
        depth_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={depth_limit}"
        depth_res = session.get(depth_url, timeout=5)
        if depth_res.status_code != 200:
            return None, None, None
            
        depth_data = depth_res.json()
        if 'bids' not in depth_data or 'asks' not in depth_data:
            return None, None, None
            
        bids = np.array(depth_data['bids'], dtype=float)
        asks = np.array(depth_data['asks'], dtype=float)
        
        return current_price, bids, asks
    except Exception as e:
        return None, None, None

def log_auto_data(file_path="market_data_log.csv"):
    print("🚀 Auto Data Collector Started... (Press Ctrl+C to stop)")
    count = 0
    
    while True:
        cycle_data = []
        timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for symbol in COINS_LIST:
            price, bids, asks = fetch_order_book_and_price(symbol)
            
            if price is not None and bids is not None and asks is not None and len(bids) >= 20 and len(asks) >= 20:
                top20_bid_sum = float(np.sum(bids[:20, 1]))
                top20_ask_sum = float(np.sum(asks[:20, 1]))
                
                total_vol = top20_bid_sum + top20_ask_sum
                obi_top20 = (top20_bid_sum - top20_ask_sum) / total_vol if total_vol > 0 else 0.0
                
                spread = asks[0, 0] - bids[0, 0]
                
                data_point = {
                    'timestamp': timestamp_str,
                    'symbol': symbol,
                    'timeframe': '15m',
                    'top20_bid_sum': round(top20_bid_sum, 4),
                    'top20_ask_sum': round(top20_ask_sum, 4),
                    'obi_top20': round(obi_top20, 6),
                    'spread': round(spread, 6),
                    'current_price': price
                }
                
                cycle_data.append(data_point)
                count += 1
                print(f"✅ [{count}] Collected data for {symbol} | Price: ${price} | OBI: {obi_top20:.4f}")
            
            # Rate limit handling
            time.sleep(0.3)
        
        # Batch save to CSV after 1 full cycle to optimize disk writing
        if cycle_data:
            df_batch = pd.DataFrame(cycle_data)
            if not os.path.isfile(file_path):
                df_batch.to_csv(file_path, index=False)
            else:
                df_batch.to_csv(file_path, mode='a', header=False, index=False)
            print(f"💾 Batch saved {len(cycle_data)} records to {file_path}")
            
        print("🔄 Finished 1 full cycle. Pausing 10 seconds before next cycle...\n")
        time.sleep(10)

if __name__ == "__main__":
    log_auto_data()
