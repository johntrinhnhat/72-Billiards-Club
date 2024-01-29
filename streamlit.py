import datetime
import pandas as pd
import numpy as np
import streamlit as st
import base64
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

github_csv_url = "https://raw.githubusercontent.com/johntrinhnhat/72-Billiards-Club/main/kioviet.csv"
github_csv_customer_url = "https://raw.githubusercontent.com/johntrinhnhat/72-Billiards-Club/main/kioviet_customer.csv"

# Streamlit Web App
st.set_page_config(page_title="72 Billiards Club",
                page_icon="🎱",
                layout="wide")

# @st.cache_data
# Load data
def load_data():
    return pd.read_csv(github_csv_url)

def load_customer_data():
    return pd.read_csv(github_csv_customer_url)
df = load_data()
df_customer = load_customer_data()

# CSS styling
st.markdown("""
<style>

[data-testid="stMetric"] {
    padding: 15px 0;
}

</style>
""", unsafe_allow_html=True)


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


## Side Bar
with st.sidebar:
    # st.title("🎱 DASHBOARD")

    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate']).dt.date

    # Ensure there are no NaT values and find the minimum and maximum dates
    valid_dates = df['PurchaseDate'].dropna()
    min_date = valid_dates.min()
    max_date = valid_dates.max()

    # Create the datetime slider
    date = st.sidebar.slider(
        "Date:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )

    hour = st.sidebar.slider(
        "Hour:",
        min_value=int(min(df['Hour'].unique())),
        max_value=int(max(df['Hour'].unique())),
        value=(int(min(df['Hour'].unique())), int(max(df['Hour'].unique())))
    )

    dayofweek = st.sidebar.multiselect(
        "DayOfWeek:",
        options=df['DayOfWeek'].unique(),
        default=df['DayOfWeek'].unique()
    )


    df_selection = df.query(
        # "Year >= @year[0] & Year <= @year[1] & "
        # "Month >= @month[0] & Month <= @month[1] & "
        # "Day >= @day[0] & Day <= @day[1] & "
        "Hour >= @hour[0] & Hour <= @hour[1] & "
        "DayOfWeek == @dayofweek &"
        "PurchaseDate >= @date[0] & PurchaseDate <= @date[1]"
    )
    
    # PROCESS SALE DATAFRAME 
    df_selection = df_selection[['Customer_Name', 'PurchaseDate', 'Hour', 'DayOfWeek', 'Sales', 'Status']]
    df_selection['Sales'] = df_selection['Sales'].astype(int)

    def highlight_sales(val):
        color = 'green' if val > 360000 else ''
        return f'background-color: {color}'
    
    styled_df_selection = df_selection.style.format({"Sales": "{:,.0f} đ"}).map(highlight_sales, subset=['Sales'])


## ---- MAIN PAGE ----
st.image('./logo.png')
st.markdown("##")

tab1, tab2, tab3 = st.tabs(["SALE", "MEMBERSHIP", "ML"])

with tab1:
    # TOP KPI's
    st.divider()

    # Calculate current period KPIs
    total_sales = int(df_selection['Sales'].sum())  
    average_sale_per_transaction = round(df_selection['Sales'].mean(), 2)
    total_invoices = len(df_selection)

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.metric(label="Total Sales", value=f"{total_sales:,} đ")
        # delta=f"{delta_total_sales_percentage:+,.2f} %"

    with middle_column:
        st.metric(label="Average Sales", value=f"{average_sale_per_transaction:,} đ")
        # delta=f"{delta_average_sale_per_transaction_percentage:+,.2f} %"
    with right_column:
        st.metric(label="Total Invoices", value=total_invoices)
        # delta=f"{delta_total_invoices_percentage:+,.2f} %"
    st.divider()

    st.dataframe(styled_df_selection)
    
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
        st.title('Sales Trend')
        hour_sale = df_selection[['Hour','Sales']]
        # Aggregate sales by hour
        hourly_sales = df_selection.groupby('Hour')['Sales'].max().reset_index()

        # Aggregate sales by day of the week
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_selection['DayOfWeek'] = pd.Categorical(df_selection['DayOfWeek'], categories=days_order, ordered=True)
        dayofweek_sales = df_selection.groupby('DayOfWeek', observed=True)['Sales'].sum().reset_index()

        # Aggregate sales by purchase date
        df_selection['PurchaseDate'] = pd.to_datetime(df_selection['PurchaseDate'])
        purchasedate_sales = df_selection.groupby(df_selection['PurchaseDate'].dt.date)['Sales'].sum().reset_index()
        purchasedate_sales = purchasedate_sales.sort_values('PurchaseDate')

        # Create a subplot figure
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Peak Hourly Sales Trend", "Sales Trend by Day of the Week", "Sales Trend by Purchase Date"))

        # Add Hourly Sales trace
        fig.add_trace(
            go.Scatter(x=hourly_sales['Hour'], y=hourly_sales['Sales'], name="Hourly Sales", marker_color='#FFA500'),
            row=1, col=1,
        )

        # Add Day of Week Sales trace
        fig.add_trace(
            go.Scatter(x=dayofweek_sales['DayOfWeek'], y=dayofweek_sales['Sales'], name="Day of Week Sales", marker_color='#ED64A6'),
            row=2, col=1,
        )

        # Add Purchase Date Sales trace
        fig.add_trace(
            go.Scatter(x=purchasedate_sales['PurchaseDate'], y=purchasedate_sales['Sales'], name="Purchase Date Sales", marker_color='#00BFFF'),
            row=3, col=1,
        )

        # Update x-axis titles
        fig.update_xaxes(title_text="Hour", row=1, col=1)
        fig.update_xaxes(title_text="Day of Week", row=2, col=1)
        fig.update_xaxes(title_text="Purchase Date", row=3, col=1)

        # Update y-axis titles
        fig.update_yaxes(title_text="Sales", row=1, col=1)
        fig.update_yaxes(title_text="Sales", row=2, col=1)
        fig.update_yaxes(title_text="Sales", row=3, col=1)

        # Update layout
        fig.update_layout(height=900, showlegend=False)

        # Display the figure in Streamlit
        st.plotly_chart(fig)

        ### Line Chart (Purchasing Pattern)
        st.title("Purchasing Behavior of 'khách lẻ'")
        # Filter transactions for 'khách lẻ'
        df_guest = df_selection.copy()
        df_guest = df_guest[df_guest['Customer_Name'] == 'khách lẻ']
        print(df_guest)
        # Aggregate sales by purchase date
        purchasedate_sales = df_guest.groupby(df_guest['PurchaseDate'].dt.date)['Sales'].sum().reset_index()

        # Aggregate sales by day of the week
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_guest['DayOfWeek'] = pd.Categorical(df_guest['DayOfWeek'], categories=days_order, ordered=True)
        dayofweek_sales = df_guest.groupby('DayOfWeek', observed=True)['Sales'].sum().reset_index()

        # Aggregate sales by hour
        hourly_sales = df_guest.groupby('Hour')['Sales'].max().reset_index()

        # Create a subplot figure
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Purchasing Behavior by Purchase Date", "Purchasing Behavior by Day of the Week", "Purchasing Behavior by Hour"))

        # Add Purchase Date Sales trace
        fig.add_trace(
            go.Scatter(x=purchasedate_sales['PurchaseDate'], y=purchasedate_sales['Sales'], name="Purchase Date Sales", marker_color='#00BFFF'),
            row=1, col=1,
        )

        # Add Day of Week Sales trace
        fig.add_trace(
            go.Scatter(x=dayofweek_sales['DayOfWeek'], y=dayofweek_sales['Sales'], name="Day of Week Sales", marker_color='#ED64A6'),
            row=2, col=1,
        )

        # Add Hourly Sales trace
        fig.add_trace(
            go.Scatter(x=hourly_sales['Hour'], y=hourly_sales['Sales'], name="Hourly Sales", marker_color='#FFA500'),
            row=3, col=1,
        )

        # Update x-axis titles
        fig.update_xaxes(title_text="Purchase Date", row=1, col=1)
        fig.update_xaxes(title_text="Day of the Week", row=2, col=1)
        fig.update_xaxes(title_text="Hour", row=3, col=1)

        # Update y-axis titles
        fig.update_yaxes(title_text="Sales", row=1, col=1)
        fig.update_yaxes(title_text="Sales", row=2, col=1)
        fig.update_yaxes(title_text="Sales", row=3, col=1)

        # Update layout
        fig.update_layout(height=900, showlegend=False)

        # Display the figure in Streamlit
        st.plotly_chart(fig)
        


with tab2:
    total_customer = len(df_customer)
    st.markdown("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.metric(label="Total Membership", value=f"{total_customer}")
    with right_column:
        st.metric(label="Top Membership", value=None)
        df_customer_sorted = df_customer.sort_values(by='Total_Revenue',ascending=False)
        df_customer_sorted = df_customer_sorted[['Name', 'Total_Revenue']]
        df_customer_sorted['Total_Revenue'] = df_customer_sorted['Total_Revenue'].apply(lambda x: f"{x:,}")
        # print(df_customer_sorted)
        st.dataframe(df_customer_sorted,
            column_order=("Name", "Total_Revenue"),
            hide_index=True,
            width=None,
            column_config={
                "Name": st.column_config.TextColumn(
                    "Name",
                ),
                "Total_Revenue": st.column_config.ProgressColumn(
                    "Total_Revenue",
                    format="%d đ",
                    min_value=0,
                    max_value=int(max(df_customer.Total_Revenue)),
                ),
                }
            )
    st.markdown("---")
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

with tab3:
    st.divider()

    
    


        



        

