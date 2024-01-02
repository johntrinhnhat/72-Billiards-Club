import subprocess
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import base64

github_csv_url = "https://raw.githubusercontent.com/johntrinhnhat/72-Billiards-Club/main/kioviet.csv"
# Load data
def load_data():
    return pd.read_csv(github_csv_url)

# Streamlit app
def run_app():
    # Streamlit Web App
    st.set_page_config(page_title="72 Billiards Club",
                    page_icon="🎱",
                    layout="wide")
    
    df = load_data()

    ## Side Bar
    st.sidebar.header("Feature Variables:")

    year = st.sidebar.multiselect(
        "Select the Year:",
        options=df['Year'].unique(),
        default=df['Year'].unique()
    )
    month = st.sidebar.multiselect(
        "Select the Month:",
        options=df['Month'].unique(),
        default=df['Month'].unique()
    )
    day = st.sidebar.multiselect(
        "Select the Day:",
        options=df['Day'].unique(),
        default=df['Day'].unique()
    )
    hour = st.sidebar.multiselect(
        "Select the Hour:",
        options=df['Hour'].unique(),
        default=df['Hour'].unique()
    )
    dayofweek = st.sidebar.multiselect(
        "Select the DayOfWeek:",
        options=df['DayOfWeek'].unique(),
        default=df['DayOfWeek'].unique()
    )

    df_selection = df.query(
        "Year == @year & Month == @month & Day == @day & Hour == @hour & DayOfWeek == @dayofweek"
    )


    ## ---- MAIN PAGE ----
    st.image('./logo.png')
    st.title("🎱 SALES DASHBOARD")
    st.markdown("##")

 
    # TOP KPI's
    total_sales = int(df_selection['Sales'].sum())  
    average_sale_per_transaction = round(df_selection['Sales'].mean(), 2)

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Total Sales")
        st.subheader(f"{total_sales:,} đ")
        

    with right_column:
        st.subheader("Average Sales")
        st.subheader(f"{average_sale_per_transaction:,} đ")

    st.markdown("---")
    st.dataframe(df_selection)

    # Download data
    def filedownload(df_selection):
        csv = df_selection.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode() # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64, {b64}" download="data.csv">Download Excel file</a>'
        return href 
    # Display the download link
    st.markdown(filedownload(df_selection), unsafe_allow_html=True)
    
    # Button Show Plots
    if st.button('Show Plots'):
        # Line Chart ( Peak Houly Sales Trend)
        st.title('Peak Hourly Sales Trend')
        hour_sale = df_selection[['Hour', 'Sales']]
        hourly_sales = hour_sale.groupby('Hour')['Sales'].max().reset_index()
        st.line_chart(hourly_sales.set_index('Hour'), color='#FFA500')

        # Bar Chart (Number Of Transaction By Hour)
        st.title('Number of Transactions by Hour')
        
        transactions_per_hour = df_selection['Hour'].value_counts().sort_index()
        transactions_per_hour = transactions_per_hour.sort_index()
        st.bar_chart(transactions_per_hour,  color='#FFA500')

        # Line Chart ( Peak Weekday Sales Trend)
        st.title('Peek Weekday Sales Trend')
        dayofweek_sale = df_selection[['DayOfWeek', 'Sales']]
        dayofweek_sales = dayofweek_sale.groupby('DayOfWeek')['Sales'].max().reset_index()

        st.line_chart(dayofweek_sales.set_index('DayOfWeek'), color='#3B8132')

        # Bar chart (Sum Of Sales by Weekday)
        st.title('Sum Of Sales by Weekday')
        # Define the correct order of the days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Ensure that 'DayOfWeek' is a categorical type with the specified order
        df_selection['DayOfWeek'] = pd.Categorical(df_selection['DayOfWeek'], categories=days_order, ordered=True)

        # Group by 'DayOfWeek' and sum the 'Sales'
        weekly_sales = df_selection.groupby('DayOfWeek', observed=True)['Sales'].sum()

        # Reset the index so 'DayOfWeek' becomes a column again
        weekly_sales = weekly_sales.reset_index()

        # Plot the results using st.bar_chart
        st.bar_chart(weekly_sales.rename(columns={'DayOfWeek': 'index'}).set_index('index'), color='#3B8132')

# Run the app
if __name__ == "__main__":
    run_app()