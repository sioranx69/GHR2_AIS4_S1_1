import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO


st.set_page_config(page_title='Data Visualization', page_icon="📈")

with st.sidebar:
    st.markdown("# Hello 👋")
    st.markdown("## Welcome to this app on visualizing and forecasting the sales of the Favorita Grocery Stores in Ecuador from 2013 to 2017!")

# Attach the 'store_sales_raw.csv'
st.title('Favorita Store Sales Source Data')
st.subheader('Sales data from 2013-01-01 to 2017-08-15')
st.markdown('Raw data can be downloaded from the download button below')

# Function to convert DataFrame into stringIO for download
def convert_df_to_csv_string(df):
    # Convert DataFrame to CSV and then encode it to string using StringIO
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

# Load the 'store_sales_raw.csv' file into a DataFrame
store_sales_df = pd.read_csv('store_sales_raw.csv')

#  Load the 'holidays_events.csv' file into a DataFrame
holidays_df = pd.read_csv('holidays_events.csv')

# Load the 'store_df' file into a DataFrame
store_df = pd.read_csv('stores.csv')

# Convert DataFrames to CSV for download
store_sales_csv_string = convert_df_to_csv_string(store_sales_df)
holidays_csv_string = convert_df_to_csv_string(holidays_df)
store_df_csv_string = convert_df_to_csv_string(store_df)

'''Download csv buttons'''
col1, col2, col3 = st.columns(3)

# Create download button for 'store_sales_raw.csv'
with col1:
    st.download_button(
        label="Download the raw sales data",
        data=store_sales_csv_string,
        file_name='store_sales_raw.csv',
        mime='text/csv',
    )

# Create download button for 'holidays_events.csv'
with col2:
        st.download_button(
        label="Download the holidays & events data",
        data=holidays_csv_string,
        file_name='data/holidays_events.csv',
        mime='text/csv',
    )

# Create download button for 'holidays_events.csv'
with col3:
    st.download_button(
        label="Download the stores data",
        data=store_df_csv_string,
        file_name='data/holidays_events.csv',
        mime='text/csv',
    )

print(store_sales_df)
print(holidays_df)

# Data Visualization and EDA
st.title('Store Sales Data Visualization and EDA')
st.subheader('Visualization to explore and analyze some patterns & trends in the Favorita store sales data')

'''Data Prepping with Datetime'''
# Transform the 'date' columns in both 'store_sales_df' and 'holidays_df' dataframes into Datetime format
store_sales_df['date'] = pd.to_datetime(store_sales_df['date'])
holidays_df['date'] = pd.to_datetime(holidays_df['date'])

'''Plot the holidays and events calendar'''

# This line hides a warning about the global use of pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

# Plotting the dataframe
fig, ax = plt.subplots(figsize=(15, 15))
sns.scatterplot(data=holidays_df, x='date', y='description', hue='locale', palette='viridis', marker='o')

plt.title('Holidays & Events Calendar 2013-2017')
plt.xlabel('Date')
plt.ylabel('Holiday Description')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.legend(title='Locale', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the holidays and events calendar plot
st.pyplot(fig)

'''Plot the unique number of stores for each state and city in descending order'''

# Count the unique number of stores in each state
store_count_state = store_df.groupby('state').store_nbr.nunique().sort_values(ascending=False).reset_index()

# Count the unique number of stores in each city with state as hue
store_count_city = store_df.groupby(['city', 'state']).store_nbr.nunique().sort_values(ascending=False).reset_index()

# Custom palette
custom_palette = {
    'Pichincha': '#1f77b4',
    'Guayas': '#aec7e8',
    'Azuay': '#ff7f0e',
    'Manabi': '#ffbb78',
    'Santo Domingo de los Tsachilas': '#2ca02c',
    'Cotopaxi': '#98df8a',
    'El Oro': '#d62728',
    'Los Rios': '#ff9896',
    'Tungurahua': '#9467bd',
    'Bolivar': '#c5b0d5',
    'Chimborazo': '#8c564b',
    'Esmeraldas': '#c49c94',
    'Imbabura': '#e377c2',
    'Loja': '#f7b6d2',
    'Pastaza': '#7f7f7f',
    'Santa Elena': '#c7c7c7'
}

# Plotting the dataframes
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sns.barplot(data=store_count_state, x='state', y='store_nbr', palette='tab20')
    plt.yticks(list(range(0, store_count_state['store_nbr'].max() + 1)))
    plt.title('Number of Stores per State')
    plt.ylabel('Number of Stores')
    plt.xlabel('State')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.barplot(data=store_count_city, x='city', y='store_nbr', hue='state', palette=custom_palette, dodge=False, width=0.8)
    plt.yticks(list(range(0, store_count_city['store_nbr'].max() + 1)))
    plt.title('Number of Stores per City')
    plt.ylabel('Number of Stores')
    plt.xlabel('City')
    plt.xticks(rotation=90)
    plt.legend(title='State', loc='upper right', bbox_to_anchor=(1, 1.05))
    plt.tight_layout()
    st.pyplot(fig2)

'''Data Prepping - Adding more datetime features'''

# Add more datetime columns to 'store_sales_df'
store_sales_df['year'] = store_sales_df['date'].dt.year
store_sales_df['month'] = store_sales_df['date'].dt.month
store_sales_df['week'] = store_sales_df['date'].dt.isocalendar().week.astype(int)
store_sales_df['day_name'] = store_sales_df['date'].dt.day_name()

'''Plot the total sales by year to observe how much sales made in total over the years'''

# Aggregate sales by year using the total sales
yearly_sales = store_sales_df.groupby('year')['sales'].sum().reset_index()

# Plotting
fig, ax = plt.subplots(figsize=(15, 8))  # Create a matplotlib figure
sns.barplot(data=yearly_sales, x='year', y='sales', palette='tab20')
plt.ticklabel_format(style='plain', axis='y')  # Set y-axis labels to plain format
plt.title('Total Sales by Year 2013-2017')
plt.ylabel('Total Sales')
plt.xlabel('Year')
plt.tight_layout()
st.pyplot(fig)

'''Plot the average sales by month to observe the fluctuation of sales over the months'''

# Average sales by month
monthly_sales = store_sales_df.groupby('month')['sales'].mean().reset_index()

# Plotting
plt.figure(figsize=(15, 8))
sns.barplot(data=monthly_sales, x='month', y='sales', palette='tab20')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Avg Sales per Month')
plt.ylabel('Avg Sales')
plt.xlabel('Month')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
st.pyplot(fig)

'''Plot the average sales by the day name to observe the fluctuations of sales over each day of the week'''
# Create a list of days in order
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Average sales by day of the week
daily_sales = store_sales_df.groupby('day_name')['sales'].mean().reindex(days_order).reset_index()

# Plotting
plt.figure(figsize=(15, 8))
sns.barplot(data=daily_sales, x='day_name', y='sales', order=days_order, palette='tab20')
plt.ticklabel_format(style='plain', axis='y')
plt.title('Avg Sales by Day of the Week')
plt.ylabel('Avg Sales')
plt.xlabel('Day of the Week')
plt.tight_layout()
st.pyplot(fig)

'''Plot the average sales by 'family' to observe the structure of sales per product family'''
# Group by product 'family' and calculate the average sales
avg_sales = store_sales_df.groupby('family')['sales'].mean().sort_values(ascending=False)

# Take the top 20 product families
top_20_families = avg_sales.head(20)

# Plotting
plt.figure(figsize=(15, 7))
top_20_families.plot(kind='bar', color='steelblue')
plt.title('Average Sales per Top 20 Product Family')
plt.ylabel('Average Sales')
plt.xlabel('Product Family')
plt.xticks(rotation=90)
plt.tight_layout()
st.pyplot(fig)

'''Plot and compare the yearly average promotion vs sales to observe any connected pattern'''

# Prepare the data for plotting
yearly_promotions = store_sales_df.groupby('year')['onpromotion'].mean().reset_index()
yearly_sales = store_sales_df.groupby('year')['sales'].mean().reset_index()

# Create a subplot with 1 row and 2 columns
fig, ax = plt.subplots(1, 2, figsize=(30, 8))  # Total fig size is double a single plot

# Plot average promotions by year
ax[0].bar(yearly_promotions['year'], yearly_promotions['onpromotion'], color='steelblue')
ax[0].set_title('Avg Promotion by Year')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Avg Promotion')
ax[0].ticklabel_format(style='plain', axis='y')
ax[0].set_xticks(yearly_promotions['year'])  # Set x-ticks to be the years
ax[0].set_xticklabels(yearly_promotions['year'], rotation=0)

# Plot average sales by year
ax[1].bar(yearly_sales['year'], yearly_sales['sales'], color='steelblue')
ax[1].set_title('Avg Sales by Year')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('Avg Sales')
ax[1].ticklabel_format(style='plain', axis='y')
ax[1].set_xticks(yearly_sales['year'])  # Set x-ticks to be the years
ax[1].set_xticklabels(yearly_sales['year'], rotation=0)

# Tight layout to ensure no overlap of subplots
plt.tight_layout()

# Display the plot in the Streamlit app
st.pyplot(fig)

