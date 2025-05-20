import pandas as pd
store_sales_df = pd.read_csv('store_sales_raw.csv')

#  Load the 'holidays_events.csv' file into a DataFrame
holidays_df = pd.read_csv('holidays_events.csv')

# Load the 'store_df' file into a DataFrame
store_df = pd.read_csv('stores.csv')

print(store_sales_df.columns)
