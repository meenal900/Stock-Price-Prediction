import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Meen's Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Meenal Saini](https://www.linkedin.com/in/meenal-saini-50b320227/)")

@st.cache_data
def download_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return df

def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])

    ticker = st.sidebar.text_input('Enter a Stock Symbol', value='SPY').upper()
    today = datetime.date.today()
    duration = st.sidebar.number_input('Enter the duration (days)', value=3000, min_value=1)
    before = today - datetime.timedelta(days=duration)
    start_date = st.sidebar.date_input('Start Date', value=before)
    end_date = st.sidebar.date_input('End date', value=today)

    if start_date > end_date:
        st.sidebar.error('Error: End date must fall after start date')
        return

    data = download_data(ticker, start_date, end_date)

    if data.empty:
        st.warning(f"No data found for '{ticker}' between {start_date} and {end_date}. Please try a different ticker or date range.")
        return

    if option == 'Visualize':
        tech_indicators(data)
    elif option == 'Recent Data':
        dataframe(data)
    else:
        predict(data)

def tech_indicators(data):
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    if len(data) < 20:
        st.warning("Not enough data to compute technical indicators. Need at least 20 rows.")
        return

    data = data.fillna(method='ffill')

    try:
        bb_indicator = BollingerBands(close=data['Close'], window=20, window_dev=2)
        bb = pd.DataFrame({
            'Close': data['Close'],
            'bb_h': bb_indicator.bollinger_hband(),
            'bb_l': bb_indicator.bollinger_lband(),
        })
    except Exception as e:
        st.error(f"Error calculating Bollinger Bands: {e}")
        bb = None

    try:
        macd = MACD(data['Close']).macd()
    except Exception:
        macd = None

    try:
        rsi = RSIIndicator(data['Close']).rsi()
    except Exception:
        rsi = None

    try:
        sma = SMAIndicator(data['Close'], window=14).sma_indicator()
    except Exception:
        sma = None

    try:
        ema = EMAIndicator(data['Close']).ema_indicator()
    except Exception:
        ema = None

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data['Close'])
    elif option == 'BB':
        if bb is not None:
            st.write('Bollinger Bands')
            st.line_chart(bb)
        else:
            st.warning("Could not compute Bollinger Bands.")
    elif option == 'MACD':
        if macd is not None:
            st.write('Moving Average Convergence Divergence')
            st.line_chart(macd)
        else:
            st.warning("Could not compute MACD.")
    elif option == 'RSI':
        if rsi is not None:
            st.write('Relative Strength Indicator')
            st.line_chart(rsi)
        else:
            st.warning("Could not compute RSI.")
    elif option == 'SMA':
        if sma is not None:
            st.write('Simple Moving Average')
            st.line_chart(sma)
        else:
            st.warning("Could not compute SMA.")
    else:
        if ema is not None:
            st.write('Exponential Moving Average')
            st.line_chart(ema)
        else:
            st.warning("Could not compute EMA.")

def dataframe(data):
    st.header('Recent Data')
    st.dataframe(data.tail(10))

def predict(data):
    model_name = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num_days = st.number_input('How many days forecast?', value=5, min_value=1)
    if st.button('Predict'):
        if len(data) < num_days + 20:
            st.warning(f"Not enough data to predict {num_days} days ahead. Please increase the duration or choose fewer days.")
            return

        if model_name == 'LinearRegression':
            model = LinearRegression()
        elif model_name == 'RandomForestRegressor':
            model = RandomForestRegressor()
        elif model_name == 'ExtraTreesRegressor':
            model = ExtraTreesRegressor()
        elif model_name == 'KNeighborsRegressor':
            model = KNeighborsRegressor()
        else:
            model = XGBRegressor()

        model_engine(model, num_days, data)

def model_engine(model, num, data):
    scaler = StandardScaler()
    df = data[['Close']].copy()
    df['preds'] = df['Close'].shift(-num)
    df.dropna(inplace=True)

    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)

    # Data for forecasting (last num days)
    x_forecast = x[-num:]
    x = x[:-num]

    y = df['preds'].values
    y = y[:-num]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    st.text(f'R2 score: {r2_score(y_test, preds):.4f}')
    st.text(f'Mean Absolute Error: {mean_absolute_error(y_test, preds):.4f}')

    forecast_pred = model.predict(x_forecast)
    st.subheader(f'Forecast for next {num} days:')
    for i, price in enumerate(forecast_pred, 1):
        st.write(f'Day {i}: {price:.2f}')

if __name__ == '__main__':
    main()
