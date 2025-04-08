import gradio as gr
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import requests
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import yfinance as yf

# --- Constants ---
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD"]
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
INTERVAL_OPTIONS = ["1h", "1d", "1wk"]
DEFAULT_FORECAST_STEPS = 24
DEFAULT_DAILY_SEASONALITY = True
DEFAULT_WEEKLY_SEASONALITY = True
DEFAULT_YEARLY_SEASONALITY = False
DEFAULT_SEASONALITY_MODE = "additive"
DEFAULT_CHANGEPOINT_PRIOR_SCALE = 0.05
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

# --- Data Fetching Functions ---
def fetch_crypto_data(symbol, interval="1h", limit=100):
    try:
        ticker = yf.Ticker(symbol)
        if interval == "1h":
            period = "1d"
            df = ticker.history(period=period, interval="1h")
        elif interval == "1d":
            df = ticker.history(period="1y", interval=interval)
        elif interval == "1wk":
            df = ticker.history(period="5y", interval=interval)
        else:
            raise ValueError("Invalid interval for yfinance.")
        if df.empty:
            raise Exception("No data returned from yfinance.")
        df.reset_index(inplace=True)
        df.rename(columns={"Datetime": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        return df.dropna()
    except Exception as e:
        raise Exception(f"Error fetching crypto data from yfinance: {e}")

def fetch_stock_data(symbol, interval="1d"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y", interval=interval)
        if df.empty:
            raise Exception("No data returned from yfinance.")
        df.reset_index(inplace=True)
        df.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        return df.dropna()
    except Exception as e:
        raise Exception(f"Error fetching stock data from yfinance: {e}")

def fetch_sentiment_data(keyword):
    try:
        tweets = [
            f"{keyword} is going to moon!",
            f"I hate {keyword}, it's trash!",
            f"{keyword} is amazing!"
        ]
        sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
        return sum(sentiments) / len(sentiments) if sentiments else 0
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return 0

# --- Technical Analysis Functions ---
def calculate_technical_indicators(df):
    if df.empty:
        return df

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['MA20'] = df['close'].rolling(window=20).mean()
    df['BB_upper'] = df['MA20'] + 2 * df['close'].rolling(window=20).std()
    df['BB_lower'] = df['MA20'] - 2 * df['close'].rolling(window=20).std()

    return df

def create_technical_charts(df):
    if df.empty:
        return None, None, None

    fig1 = go.Figure()
    fig1.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_upper'], name='Upper BB', line=dict(color='gray', dash='dash')))
    fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_lower'], name='Lower BB', line=dict(color='gray', dash='dash')))
    fig1.update_layout(title='Price and Bollinger Bands', xaxis_title='Date', yaxis_title='Price')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], name='RSI'))
    fig2.add_hline(y=70, line_dash="dash", line_color="red")
    fig2.add_hline(y=30, line_dash="dash", line_color="green")
    fig2.update_layout(title='RSI Indicator', xaxis_title='Date', yaxis_title='RSI')

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df['timestamp'], y=df['MACD'], name='MACD'))
    fig3.add_trace(go.Scatter(x=df['timestamp'], y=df['Signal_Line'], name='Signal Line'))
    fig3.update_layout(title='MACD', xaxis_title='Date', yaxis_title='Value')

    return fig1, fig2, fig3

# --- Prophet Forecasting Functions ---
def prepare_data_for_prophet(df):
    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    df_prophet = df.rename(columns={"timestamp": "ds", "close": "y"})
    return df_prophet[["ds", "y"]]

def prophet_forecast(df_prophet, periods=10, freq="h", daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False, seasonality_mode="additive", changepoint_prior_scale=0.05):
    if df_prophet.empty:
        return pd.DataFrame(), "No data for Prophet."

    try:
        model = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        return forecast, ""
    except Exception as e:
        return pd.DataFrame(), f"Forecast error: {e}"

def prophet_wrapper(df_prophet, forecast_steps, freq, daily_seasonality, weekly_seasonality, yearly_seasonality, seasonality_mode, changepoint_prior_scale):
    if len(df_prophet) < 10:
        return pd.DataFrame(), "Not enough data for forecasting (need >=10 rows)."

    full_forecast, err = prophet_forecast(
        df_prophet,
        forecast_steps,
        freq,
        daily_seasonality,
        weekly_seasonality,
        yearly_seasonality,
        seasonality_mode,
        changepoint_prior_scale,
    )
    if err:
        return pd.DataFrame(), err

    future_only = full_forecast.loc[len(df_prophet):, ["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return future_only, ""

def create_forecast_plot(forecast_df):
    if forecast_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color="blue", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat_lower"],
        fill=None,
        mode="lines",
        line=dict(width=0),
        showlegend=True,
        name="Lower Bound"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat_upper"],
        fill="tonexty",
        mode="lines",
        line=dict(width=0),
        name="Upper Bound"
    ))

    fig.update_layout(
        title="Price Forecast",
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig

# --- Model Training and Prediction ---
model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)

def train_model(df):
    if df.empty:
        return
    df["target"] = (df["close"].pct_change() > 0.05).astype(int)
    features = df[["close", "volume"]].dropna()
    target = df["target"].dropna()
    if not features.empty and not target.empty:
        model.fit(features, target)
    else:
        print("Not enough data for model training.")

def predict_growth(latest_data):
    if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
        return [0]

    try:
        prediction = model.predict(latest_data.reshape(1, -1))
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return [0]

# --- Main Prediction and Display Function ---
def analyze_market(market_type, symbol, interval, forecast_steps, daily_seasonality, weekly_seasonality, yearly_seasonality, seasonality_mode, changepoint_prior_scale, sentiment_keyword=""):
    df = pd.DataFrame()
    error_message = ""
    sentiment_score = 0

    try:
        if market_type == "Crypto":
            df = fetch_crypto_data(symbol, interval=interval)
        elif market_type == "Stock":
            df = fetch_stock_data(symbol, interval=interval)
        else:
            error_message = "Invalid market type selected."
            return None, None, None, None, None, "", error_message, 0

        if sentiment_keyword:
            sentiment_score = fetch_sentiment_data(sentiment_keyword)
    except Exception as e:
        error_message = f"Data Fetching Error: {e}"
        return None, None, None, None, None, "", error_message, 0

    if df.empty:
        error_message = "No data fetched."
        return None, None, None, None, None, "", error_message, 0

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = calculate_technical_indicators(df)

    df_prophet = prepare_data_for_prophet(df)
    freq = "h" if interval == "1h" or interval == "60min" else "d"
    forecast_df, prophet_error = prophet_wrapper(
        df_prophet,
        forecast_steps,
        freq,
        daily_seasonality,
        weekly_seasonality,
        yearly_seasonality,
        seasonality_mode,
        changepoint_prior_scale,
    )

    if prophet_error:
        error_message = f"Prophet Error: {prophet_error}"
        return None, None, None, None, None, "", error_message, sentiment_score

    forecast_plot = create_forecast_plot(forecast_df)
    tech_plot, rsi_plot, macd_plot = create_technical_charts(df)

    try:
        train_model(df.copy())
        if not df.empty:
            latest_data = df[["close", "volume"]].iloc[-1].values
            growth_prediction = predict_growth(latest_data)
            growth_label = "Yes" if growth_prediction[0] == 1 else "No"
        else:
            growth_label = "N/A: Insufficient Data"
    except Exception as e:
        error_message = f"Model Error: {e}"
        growth_label = "N/A"

    forecast_df_display = forecast_df.loc[:, ["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_df_display.rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"}, inplace=True)
    return forecast_plot, tech_plot, rsi_plot, macd_plot, forecast_df_display, growth_label, error_message, sentiment_score

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# Market Analysis and Prediction")

    with gr.Row():
        with gr.Column():
            market_type_dd = gr.Radio(label="Market Type", choices=["Crypto", "Stock"], value="Crypto")
            symbol_dd = gr.Dropdown(label="Symbol", choices=CRYPTO_SYMBOLS, value="BTC-USD")
            interval_dd = gr.Dropdown(label="Interval", choices=INTERVAL_OPTIONS, value="1h")
            forecast_steps_slider = gr.Slider(label="Forecast Steps", minimum=1, maximum=100, value=DEFAULT_FORECAST_STEPS, step=1)
            daily_box = gr.Checkbox(label="Daily Seasonality", value=DEFAULT_DAILY_SEASONALITY)
            weekly_box = gr.Checkbox(label="Weekly Seasonality", value=DEFAULT_WEEKLY_SEASONALITY)
            yearly_box = gr.Checkbox(label="Yearly Seasonality", value=DEFAULT_YEARLY_SEASONALITY)
            seasonality_mode_dd = gr.Dropdown(label="Seasonality Mode", choices=["additive", "multiplicative"], value=DEFAULT_SEASONALITY_MODE)
            changepoint_scale_slider = gr.Slider(label="Changepoint Prior Scale", minimum=0.01, maximum=1.0, step=0.01, value=DEFAULT_CHANGEPOINT_PRIOR_SCALE)
            sentiment_keyword_txt = gr.Textbox(label="Sentiment Keyword (optional)")

        with gr.Column():
            forecast_plot = gr.Plot(label="Price Forecast")
            with gr.Row():
                tech_plot = gr.Plot(label="Technical Analysis")
                rsi_plot = gr.Plot(label="RSI Indicator")
            with gr.Row():
                macd_plot = gr.Plot(label="MACD")
            forecast_df = gr.Dataframe(label="Forecast Data", headers=["Date", "Forecast", "Lower Bound", "Upper Bound"])
            growth_label_output = gr.Label(label="Explosive Growth Prediction")
            sentiment_label_output = gr.Number(label="Sentiment Score")

    def update_symbol_choices(market_type):
        if market_type == "Crypto":
            return gr.Dropdown(choices=CRYPTO_SYMBOLS, value="BTC-USD")
        elif market_type == "Stock":
            return gr.Dropdown(choices=STOCK_SYMBOLS, value="AAPL")
        return gr.Dropdown(choices=[], value=None)
    market_type_dd.change(fn=update_symbol_choices, inputs=[market_type_dd], outputs=[symbol_dd])

    analyze_button = gr.Button("Analyze Market", variant="primary")
    analyze_button.click(
        fn=analyze_market,
        inputs=[
            market_type_dd,
            symbol_dd,
            interval_dd,
            forecast_steps_slider,
            daily_box,
            weekly_box,
            yearly_box,
            seasonality_mode_dd,
            changepoint_scale_slider,
            sentiment_keyword_txt,
        ],
        outputs=[forecast_plot, tech_plot, rsi_plot, macd_plot, forecast_df, growth_label_output, gr.Label(label="Error Message"), sentiment_label_output]
    )

if __name__ == "__main__":
    demo.launch()