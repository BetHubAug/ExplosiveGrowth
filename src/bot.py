import time
from datetime import datetime
from config import BINANCE_API_KEY, SYMBOLS_TO_WATCH
from data_fetcher import fetch_market_data_from_binance, fetch_sentiment_data
from model import train_model, predict_growth
from trade_executor import execute_trade
from risk_manager import apply_risk_management

def log_event(message):
    """Log events with timestamps."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def main():
    """
    Main loop for the Money Maker Bot: integrates multiple data sources,
    predicts growth potential, and executes trades based on AI models
    combined with sentiment analysis and risk management rules.
    """
    log_event("Starting Money Maker Bot 🚀")

    while True:
        try:
            for symbol in SYMBOLS_TO_WATCH:
                log_event(f"Analyzing {symbol}...")

                # Step 1: Fetch market data from Binance
                market_data = fetch_market_data_from_binance(symbol)
                if not market_data or market_data.empty:
                    log_event(f"No market data available for {symbol}. Skipping...")
                    continue

                # Step 2: Fetch sentiment data from multiple sources
                sentiment_score = fetch_sentiment_data(symbol.replace("USDT", ""))
                log_event(f"Sentiment score for {symbol}: {sentiment_score:.2f}")

                # Step 3: Train AI model on historical market data
                train_model(market_data)
                latest_data = market_data.iloc[-1][["close", "volume"]].values.tolist()
                prediction = predict_growth(latest_data)
                log_event(f"Prediction for {symbol}: {'BUY' if prediction == 1 else 'SELL'}")

                # Step 4: Apply risk management logic
                action = apply_risk_management(prediction, sentiment_score, market_data)
                if action:
                    execute_trade(symbol, action)
                    log_event(f"Executed {action} trade for {symbol}.")
                else:
                    log_event(f"No trade executed for {symbol}. Conditions not met.")

            # Wait before the next iteration (e.g., every minute or longer)
            time.sleep(300)

        except Exception as e:
            log_event(f"Error occurred: {str(e)}")
            time.sleep(60)  # Retry after a delay

if __name__ == "__main__":
    main()
