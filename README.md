# 🛒 Store Sales Forecasting during Natural Disasters

This project aims to forecast daily sales for the Ecuadorian retail chain **Favorita** using historical data enriched with external signals such as holidays, oil prices, and promotional events. The unique challenge addressed in this project is understanding how **natural disasters and national events** impact retail demand and how forecasting models can remain accurate under such volatility.

---

## 📌 Problem Statement

Forecasting is a core component of any successful retail operation, but it's particularly crucial for grocery chains where perishable items and supply volatility present a constant challenge.

In this project, we focus on:
- Building accurate sales forecasts across various stores and product families.
- Investigating how **natural disasters** like the **2016 Ecuador earthquake** influence demand.
- Supporting better **operational and strategic decisions** using time-stamped sales data and time series models.

---

## 🎯 Project Goal

- Develop a machine learning model to forecast sales based on historical records.
- Explore and quantify the impact of **holidays, oil prices, promotions, and major national events** on sales patterns.
- Improve business foresight to support supply chain, inventory, and marketing planning—especially in **disrupted conditions** like economic downturns or natural disasters.

---

## 🧾 Dataset Overview

Data is adapted from the [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition, representing real sales data from Favorita, a major Ecuadorian grocery retailer.

### 🗂 Main Files:
- `store_sales_raw.csv`: Historical daily sales of items in stores.
- `stores.csv`: Metadata for stores (location, type, cluster).
- `holidays_events.csv`: National, local, and regional holidays/events.

---

## 🧰 Tools & Technologies

- **Python**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **Scikit-learn**
- **XGBoost / LightGBM**
- **Facebook Prophet**
- **Streamlit**
- **Jupyter Notebook**
- **GitHub**

---

## 📊 Methodology

1. **Data Cleaning & Integration**  
   - Convert date columns and merge all relevant datasets.
   - Handle transferred holidays, bridge days, and oil price gaps.

2. **Feature Engineering**  
   - Lagged sales features, rolling averages.
   - Promotion indicators, holiday/event types, weekday/weekend flags.
   - Integration of oil price trends as a macroeconomic signal.

3. **Exploratory Data Analysis**  
   - Seasonal and holiday-driven trends.
   - Anomaly detection: spikes during promotions, dips post-earthquake.
   - Impact analysis of events like the **2016 Terremoto de Manabí**.

4. **Modeling & Evaluation**  
   - Time-aware train-test split
   - Baseline and advanced models like XGBoost, Prophet
   - Evaluation metrics: **MAE, RMSE**

5. **Deployment**  
   - Forecasts visualized and structured for use in business environments

---

## 📈 Key Findings

- Sales are seasonal, peaking around holidays and year-end.
- The 2016 earthquake caused major disruption in buying behavior—sales dipped then surged due to panic buying and relief efforts.
- Oil prices show a moderate correlation with general sales trends.
- Tree-based models outperformed Prophet due to richer feature support.

---

## 📦 Business Value

- Enables proactive inventory and supply chain decisions.
- Helps in planning promotion timing and stock levels around holidays.
- Empowers retailers to prepare for unexpected disruptions like natural disasters or economic shocks.

---

## 🧪 Evaluation

- **Root Mean Squared Error (RMSE)**: Evaluates the model’s error sensitivity.
- **Mean Absolute Error (MAE)**: Measures average deviation from true values.
- Evaluation was conducted using chronological holdout sets to simulate real-world forecasting.

---
## 🚀 Deployment & Interactive Demo

To make our forecasting models accessible and actionable, we deployed both the **SARIMA** and **Prophet** models using **Streamlit**.

### 🖥️ Features of the Deployment

- Interactive visualizations of historical and forecasted sales.
- Dual-model comparison between **SARIMA** and **Prophet** outputs.
- Seasonal trends, promotional impacts, and anomalies clearly displayed.
- Real-time insights to support inventory, marketing, and operations planning.

### 📺 Live Demo (Prophet & SARIMA Visualizations)

[![Watch Demo](https://img.shields.io/badge/▶️%20Click%20Here%20to%20Watch-Demo-blue?logo=google-drive)](https://drive.google.com/file/d/1ylC97JnCt20frFisl_GSsGh1VWELjNyq/view?usp=sharing)

> _"Forecasting is only as good as how it's used—this demo bridges the gap between data science and real-world business value."_

## 🧠 Future Work

- Integrate real-time weather and seismic data for disaster impact forecasting.
- Fine-tune neural models like LSTM or hybrid deep learning approaches.
- Deploy automated retraining pipelines with MLOps tools.
- Extend forecasts to regional or city-level granularity.

---

## 📚 References

- [Kaggle Store Sales Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Streamlit](https://streamlit.io/)
- [Python Docs](https://docs.python.org/3/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

---

## 🤝 Team

GHR2_AIS4_S1_1  
[Project Repository](https://github.com/sioranx69/GHR2_AIS4_S1_1)
