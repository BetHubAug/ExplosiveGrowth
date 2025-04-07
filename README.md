# Explosive Growth Prediction Bot 🚀

## Overview
The Explosive Growth Prediction Bot is an AI-powered trading bot designed to identify cryptocurrencies that are likely to experience massive short-term gains (500%+). It combines real-time market data, sentiment analysis from social media, and machine learning models to predict price surges and automate trades.

## Features
- **Real-Time Market Analysis**: Fetches live price data from Binance API.
- **Sentiment Analysis**: Analyzes Twitter and Reddit data to detect hype around specific assets.
- **AI Predictions**: Uses machine learning to predict explosive price movements.
- **Automated Trading**: Places buy/sell orders based on clear signals.
- **Backtesting**: Simulates strategies on historical data for optimization.

## Installation

### Requirements
- Python 3.8+
- Termux or any Linux-based environment

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/explosive-growth-bot.git
   cd explosive-growth-bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your API keys in `src/config.py`.

4. Run the bot:
   ```
   python src/bot.py
   ```

## Usage

The bot automatically fetches market data, analyzes sentiment, and predicts price movements. It will execute trades based on predefined thresholds.

## License

This project is licensed under the MIT License.
