from config import BINANCE_API_KEY, TWITTER_KEYS, SYMBOLS_TO_WATCH, RISK_SETTINGS
from data_fetcher import fetch_market_data, analyze_sentiment
from model import train_model, predict_growth
from trade_executor import execute_trade
from risk_manager import apply_risk_management

def main():
    print("Starting Money Maker Bot 💸")
    
    while True:
        for symbol in SYMBOLS_TO_WATCH:
            print(f"\nAnalyzing {symbol}...")
            
            # Fetch market data and analyze sentiment.
            df = fetch_market_data(symbol)
            sentiment_score = analyze_sentiment(symbol.replace("USDT", ""))

            # Train model and make predictions.
            train_model(df)
            latest_data = df.iloc[-1][["close", "volume"]].values.tolist()
            prediction = predict_growth(latest_data)

            # Apply risk management logic.
            action = apply_risk_management(prediction, sentiment_score, df)

            if action:
                execute_trade(symbol, action)

if __name__ == "__main__":
    main()
