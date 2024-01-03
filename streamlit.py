import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import base64


github_csv_url = "https://raw.githubusercontent.com/johntrinhnhat/72-Billiards-Club/main/kioviet.csv"
github_csv_customer_url = "https://raw.githubusercontent.com/johntrinhnhat/72-Billiards-Club/main/kioviet_customer.csv"

# Load data
def load_data():
    return pd.read_csv(github_csv_url)

def load_customer_data():
    return pd.read_csv(github_csv_customer_url)

df = load_data()
df_customer = load_customer_data()

# Identify repeat and one-time customers
customer_freq = df['Customer_Name'].value_counts().reset_index()
customer_freq.columns = ['Customer_Name', 'Frequency']
repeat_customers = customer_freq[customer_freq['Frequency'] > 1]['Customer_Name']
one_time_customers = customer_freq[customer_freq['Frequency'] == 1]['Customer_Name']

# Filter the main dataframe for repeat customers
df_repeat_customers = df[df['Customer_Name'].isin(repeat_customers)]

# Streamlit Web App
st.set_page_config(page_title="72 Billiards Club",
                page_icon="🎱",
                layout="wide")


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


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
st.markdown("##")

tab1, tab2= st.tabs(["SALE", "CUSTOMER"])

with tab1:

    # TOP KPI's
    total_sales = int(df_selection['Sales'].sum())  
    average_sale_per_transaction = round(df_selection['Sales'].mean(), 2)

    left_column, right_column = st.columns(2)
    with left_column:
        # st.subheader("Total Sales")
        # st.subheader(f"{total_sales:,} đ")
        st.metric(label="Total Sales", value=f"{total_sales:,} đ")
        

    with right_column:
        # st.subheader("Average Sales")
        # st.subheader(f"{average_sale_per_transaction:,} đ")
        st.metric(label="Average Sales", value=f"{average_sale_per_transaction:,} đ")

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

    #### Button Show Plots
    if st.button('Show Plots'):

        ### Line Chart ( Peak Houly Sales Trend)
        st.title('Peak Hourly Sales Trend')
        hour_sale = df_selection[['Hour', 'Sales']]
        hourly_sales = hour_sale.groupby('Hour')['Sales'].max().reset_index()
        fig = px.line(hourly_sales, x='Hour', y='Sales', line_shape='linear', markers=False)
        fig.update_traces(line=dict(color='#FFA500'))
        st.plotly_chart(fig)

        ### Bar Chart (Number Of Transaction By Hour)
        st.title('Number of Transactions by Hour')
        transactions_per_hour = df_selection['Hour'].value_counts().sort_index()
        fig = px.bar(transactions_per_hour, x=transactions_per_hour.index, y=transactions_per_hour.values, labels={'y':'Transactions', 'x':'Hour'})
        fig.update_traces(marker_color='#FFA500')
        st.plotly_chart(fig)

        ### Line Chart ( Peak Weekday Sales Trend)
        st.title('Peek Weekday Sales Trend')
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dayofweek_sale = df_selection[['DayOfWeek', 'Sales']].copy()
        dayofweek_sale['DayOfWeek'] = pd.Categorical(dayofweek_sale['DayOfWeek'], categories=days_order, ordered=True)
        dayofweek_sales = dayofweek_sale.groupby('DayOfWeek', observed=False)['Sales'].max().reset_index()
        fig = px.line(dayofweek_sales, x='DayOfWeek', y='Sales', line_shape='linear', markers=False)
        fig.update_traces(line=dict(color='#ED64A6'))  # Updated to a valid HEX color
        st.plotly_chart(fig)

        ### Pie chart (Sum Of Sales by Weekday)
        st.title('Sum Of Sales by Weekday')
        df_selection['DayOfWeek'] = pd.Categorical(df_selection['DayOfWeek'], categories=days_order, ordered=True)
        weekly_sales = df_selection.groupby('DayOfWeek', observed=False)['Sales'].sum().reset_index()
        fig = px.pie(weekly_sales, names='DayOfWeek', values='Sales')
        st.plotly_chart(fig)

        ### Line Chart (Purchasing Pattern)
        st.title("Purchasing Behavior of 'khách lẻ'")
        khach_le_transactions = df[df['Customer_Name'] == 'khách lẻ'].copy()
        khach_le_transactions['PurchaseDate'] =  pd.to_datetime(khach_le_transactions['PurchaseDate'], errors='coerce')
        purchasing_pattern = khach_le_transactions.groupby('PurchaseDate')['Sales'].sum().reset_index()
        # Plotting with Plotly
        fig = px.line(purchasing_pattern,
                    x='PurchaseDate',
                    y='Sales',
                    line_shape='linear',
                    markers=False
        )
        st.plotly_chart(fig)

        ### Line Chart (Purchasing Pattern)
        khach_le_transactions = df[df['Customer_Name'] == 'khách lẻ'].copy()
        khach_le_transactions['DayOfWeek'] = pd.Categorical(khach_le_transactions['DayOfWeek'], categories=days_order, ordered=True)
        purchasing_pattern = khach_le_transactions.groupby('DayOfWeek', observed=False)['Sales'].sum().reset_index()
        # Plotting with Plotly
        fig = px.line(purchasing_pattern,
                    x='DayOfWeek',
                    y='Sales',
                    line_shape='linear',
                    markers=False
        )
        fig.update_traces(line=dict(color='#ED64A6'))  # Updated to a valid HEX color
        st.plotly_chart(fig)

        ### Line Chart (Purchasing Pattern)
        khach_le_transactions = df[df['Customer_Name'] == 'khách lẻ']
        # khach_le_transactions['DayOfWeek'] = pd.Categorical(khach_le_transactions['DayOfWeek'], categories=days_order, ordered=True)
        purchasing_pattern = khach_le_transactions.groupby('Hour')['Sales'].sum().reset_index()
        # Plotting with Plotly
        fig = px.line(purchasing_pattern,
                    x='Hour',
                    y='Sales',
                    line_shape='linear',
                    markers=False
        )
        fig.update_traces(line=dict(color='#FFA500'))
        st.plotly_chart(fig)
        


with tab2:
    total_customer = len(df_customer)
    st.metric(label="Total Customers", value=f"{total_customer}")
    st.dataframe(df_customer)

    # Button Show Plots
    if st.button('Show Plot'):
        # Bar Chart (Number of Membership)
        st.title('Number of Membership')
        membership_counts = df_customer['Membership'].value_counts(dropna=False).reset_index()
        membership_counts.columns = ['Membership', 'Count']
        membership_counts['Membership'] = membership_counts['Membership'].fillna('None')
        fig = px.bar(membership_counts, x='Membership', y='Count', 
                    hover_data=['Membership', 'Count'], color='Count')
        st.plotly_chart(fig)


        



        

