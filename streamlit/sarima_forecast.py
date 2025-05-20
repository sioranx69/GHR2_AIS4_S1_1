import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

st.set_page_config(page_title='SARIMA Forecast', page_icon="🔮")

with st.sidebar:
    st.markdown("# Hello again 👋")
    st.markdown("# This page shows the sales forecasts for the Favorita Store using SARIMA model to predict sales from 2017-01-01 to 2017-08-15.")
    st.markdown("# Train data for the forecasts is from 2013-01-08 to 2016-12-31.")

# Define a function to load data
@st.cache
def load_data():
    df = pd.read_csv('store_sales_raw.csv')
    df['date'] = pd.to_datetime(df['date'])  # 👈 Convert string to datetime
    return df

# Define a function to preprocess data and aggregate by date
def aggregate(df):
    # Aggregate sales by date using sum
    date_total_sales = df.groupby('date')['sales'].sum().reset_index()
    return date_total_sales

# Define a function to perform train-test split
def train_test_split(date_total_sales, split_date):
    train_data = date_total_sales.loc[date_total_sales['date'] <= split_date]
    test_data = date_total_sales.loc[date_total_sales['date'] > split_date]
    return train_data, test_data

# Define a function to train SARIMA model and forecast
def train_and_forecast_sarima(train_data, test_data):
    # Search over model orders by running auto_arima to find the best ARIMA parameters
    model = auto_arima(train_data['sales'],
                       start_p=1, start_q=1,
                       max_p=3, max_q=3, m=12,
                       start_P=0, seasonal=True,
                       d=1, D=1, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    # Fit SARIMA based on the best auto_arima parameters result
    sarima_model = SARIMAX(train_data['sales'],
                           order=model.order,
                           seasonal_order=model.seasonal_order,
                           enforce_stationarity=False,
                           enforce_invertibility=False).fit(disp=0)

    # Forecast
    forecast = sarima_model.get_forecast(steps=len(test_data))
    forecast_df = test_data.copy()
    forecast_df['sales_forecast'] = forecast.predicted_mean
    return forecast_df, sarima_model

# Load and preprocess the data
store_sales_df = load_data()
date_total_sales = aggregate(store_sales_df)

# Splitting the data into train and test sets
split_date = st.sidebar.date_input('Train-test split date', value=pd.to_datetime('2016-12-31'))
split_date = pd.to_datetime(split_date)  # <--- Convert from datetime.date to pd.Timestamp

train_data, test_data = train_test_split(date_total_sales, split_date)

# Train and forecast with the SARIMA model
if st.sidebar.button('Forecast with SARIMA'):
    with st.spinner('Training SARIMA model and forecasting...'):
        forecast_df, sarima_model = train_and_forecast_sarima(train_data, test_data)
        st.success('SARIMA forecasting completed!')

        # Display forecast
        st.subheader("Forecast Visualizations")

        st.markdown("### 📅 Daily Forecast")
        st.image("../images/sarima_actual_forecasts_daily.jpeg", caption="Daily Sales - Actual vs Forecast", use_column_width=True)

        st.markdown("### 📈 Weekly Forecast")
        st.image("../images/sarima_actual_forecasts_weekly.jpeg", caption="Weekly Sales - Actual vs Forecast", use_column_width=True)

        st.markdown("### 📆 Monthly Forecast")
        st.image("../images/sarima_actual_forecasts_monthly.jpeg", caption="Monthly Sales - Actual vs Forecast", use_column_width=True)
