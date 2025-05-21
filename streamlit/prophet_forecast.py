import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

st.set_page_config(page_title='Prophet Forecast', page_icon="🔮")

with st.sidebar:
    st.markdown("# Hello again 👋")
    st.markdown("This page shows the sales forecasts for Favorita Grocery using Prophet to predict 2017-01-01 → 2017-08-15.")
    st.markdown("Train data: 2013-01-08 → 2016-12-31.")

# ——————————————
# 1. Load & cache data
# ——————————————
@st.cache_data
def load_data():
    # parse_dates will immediately give us a datetime64 column
    df = pd.read_csv(
        'store_sales_raw.csv',
        parse_dates=['date'],
        infer_datetime_format=True,
    )
    return df.copy()

# ——————————————
# 2. Aggregate by day
# ——————————————
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # 'date' is already datetime64, group & sum
    return df.groupby('date', as_index=False)['sales'].sum()

# ——————————————
# 3. Train/test split
# ——————————————
def train_test_split(date_total_sales: pd.DataFrame, split_date: pd.Timestamp):
    mask = date_total_sales['date'] <= split_date
    return date_total_sales[mask], date_total_sales[~mask]

# ——————————————
# 4. Prep for Prophet
# ——————————————
def preprocess_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'date': 'ds', 'sales': 'y'})
    return df.reset_index(drop=True)

# ——————————————
# 5. Prophet training & forecasting
# ——————————————
def train_prophet(df: pd.DataFrame) -> Prophet:
    m = Prophet()
    m.fit(df)
    return m

def make_forecast(model: Prophet, future: pd.DataFrame) -> pd.DataFrame:
    return model.predict(future)

# ——————————————
# 6. Plot helpers
# ——————————————
def plot_results(model: Prophet, forecast: pd.DataFrame):
    fig1 = model.plot(forecast)
    plt.ticklabel_format(style='plain', axis='y')
    st.pyplot(fig1)

    fig2 = model.plot_components(forecast)
    plt.ticklabel_format(style='plain', axis='y')
    st.pyplot(fig2)

# ——————————————
# Main
# ——————————————
df_raw = load_data()
date_total_sales = aggregate(df_raw)

# Sidebar date picker → convert to pd.Timestamp
split_date = st.sidebar.date_input(
    'Train/test split date',
    value=pd.to_datetime('2016-12-31')
)
split_date = pd.to_datetime(split_date)

train_df, test_df = train_test_split(date_total_sales, split_date)

# Prep for Prophet
train_prophet_df = preprocess_for_prophet(train_df)
test_prophet_df  = preprocess_for_prophet(test_df)

if st.button('Get the Prophet forecast'):
    model = train_prophet(train_prophet_df)

    future = test_prophet_df[['ds']]
    forecast = make_forecast(model, future)

   
    pred = forecast[['ds', 'yhat']].rename(columns={'ds':'date', 'yhat':'sales_pred'})
    pred['sales_actual'] = test_prophet_df['y'].values
    pred = pred.set_index('date')

    st.write(pred)
    st.line_chart(pred[['sales_actual','sales_pred']])
    st.subheader("Actual vs Forecast (Daily, Weekly, Monthly)")

    st.markdown("#### Daily Forecast")
    st.image("../images/prophet_actual_forecasts_daily.jpeg", caption="Prophet Daily Forecast", use_column_width=True)

    st.markdown("#### Weekly Forecast")
    st.image("../images/prophet_actual_forecasts_weekly.jpeg", caption="Prophet Weekly Forecast", use_column_width=True)

    st.markdown("#### Monthly Forecast")
    st.image("../images/prophet_actual_forecasts_monthly.jpeg", caption="Prophet Monthly Forecast", use_column_width=True)

    