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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error



st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info("Created and designed by [Meenal Saini](https://www.linkedin.com/in/meenal-saini-50b320227/)")

def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()



@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df



option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')




data = download_data(option, start_date, end_date)
scaler = StandardScaler()

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    if data is None or data.empty or len(data) < 20:
        st.error("Not enough data to compute technical indicators. Please choose a longer duration or a different stock.")
        return

    close = data['Close'].squeeze()  # Ensure 1D Series

    try:
        bb_indicator = BollingerBands(close=close)
        bb = data.copy()
        bb['bb_h'] = bb_indicator.bollinger_hband()
        bb['bb_l'] = bb_indicator.bollinger_lband()
        bb = bb[['Close', 'bb_h', 'bb_l']]
    except Exception as e:
        st.warning(f"Error calculating Bollinger Bands: {e}")
        bb = None

    try:
        macd = MACD(close=close).macd()
    except Exception as e:
        st.warning(f"Error calculating MACD: {e}")
        macd = None

    try:
        rsi = RSIIndicator(close=close).rsi()
    except Exception as e:
        st.warning(f"Error calculating RSI: {e}")
        rsi = None

    try:
        sma = SMAIndicator(close=close, window=14).sma_indicator()
    except Exception as e:
        st.warning(f"Error calculating SMA: {e}")
        sma = None

    try:
        ema = EMAIndicator(close=close).ema_indicator()
    except Exception as e:
        st.warning(f"Error calculating EMA: {e}")
        ema = None

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(close)
    elif option == 'BB':
        st.write('Bollinger Bands')
        st.line_chart(bb) if bb is not None else st.error("Could not calculate BB.")
    elif option == 'MACD':
        st.write('MACD')
        st.line_chart(macd) if macd is not None else st.error("Could not calculate MACD.")
    elif option == 'RSI':
        st.write('RSI')
        st.line_chart(rsi) if rsi is not None else st.error("Could not calculate RSI.")
    elif option == 'SMA':
        st.write('SMA')
        st.line_chart(sma) if sma is not None else st.error("Could not calculate SMA.")
    elif option == 'EMA':
        st.write('EMA')
        st.line_chart(ema) if ema is not None else st.error("Could not calculate EMA.")

def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))



def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        else:
            engine = XGBRegressor()
            model_engine(engine, num)


def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


if __name__ == '__main__':
    main()
