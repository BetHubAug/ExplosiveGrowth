import requests

def fetch_market_data(symbol="BTCUSDT", interval="1h", limit=100):
    """Fetch historical market data from Binance."""
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Error fetching market data")

def analyze_sentiment(keyword):
    """Analyze sentiment from social media."""
    tweets = [
        f"{keyword} is going to moon!",
        f"I hate {keyword}, it's trash!",
        f"{keyword} is the best investment!"
    ]
    
    sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
    return sum(sentiments) / len(sentiments)
