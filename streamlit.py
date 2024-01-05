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

    year = st.sidebar.slider(
        "Year:",
        min_value=int(min(df['Year'].unique())),
        max_value=int(max(df['Year'].unique())),
        value=(int(min(df['Year'].unique())), int(max(df['Year'].unique())))
    )

    month = st.sidebar.slider(
        "Month:",
        min_value=int(min(df['Month'].unique())),
        max_value=int(max(df['Month'].unique())),
        value=(int(min(df['Month'].unique())), int(max(df['Month'].unique())))
    )

    day = st.sidebar.slider(
        "Day:",
        min_value=int(min(df['Day'].unique())),
        max_value=int(max(df['Day'].unique())),
        value=(int(min(df['Day'].unique())), int(max(df['Day'].unique())))
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
        "Year >= @year[0] & Year <= @year[1] & "
        "Month >= @month[0] & Month <= @month[1] & "
        "Day >= @day[0] & Day <= @day[1] & "
        "Hour >= @hour[0] & Hour <= @hour[1] & "
        "DayOfWeek == @dayofweek"
    )



## ---- MAIN PAGE ----
st.image('./logo.png')
st.markdown("##")

tab1, tab2= st.tabs(["SALE", "MEMBERSHIP"])

with tab1:
    # TOP KPI's
    st.markdown("---")

    # Assuming 'year' is a list with the current year range selected in the sidebar
    current_year_range = year
    previous_year_range = [year[0] - 1, year[1] - 1]

    df_previous_period = df.query(
        "Year >= @previous_year_range[0] & Year <= @previous_year_range[1] & "
        "Month >= @month[0] & Month <= @month[1] & "
        "Day >= @day[0] & Day <= @day[1] & "
        "Hour >= @hour[0] & Hour <= @hour[1] & "
        "DayOfWeek == @dayofweek"
    )
    # Calculate previous period KPIs
    total_sales_previous = int(df_previous_period['Sales'].sum())
    average_sale_per_transaction_previous = round(df_previous_period['Sales'].mean(), 2)
    total_invoices_previous = len(df_previous_period)

    # Calculate current period KPIs
    total_sales = int(df_selection['Sales'].sum())  
    average_sale_per_transaction = round(df_selection['Sales'].mean(), 2)
    total_invoices = len(df_selection)

    # Calculate deltas
    delta_total_sales = total_sales - total_sales_previous
    delta_average_sale_per_transaction = average_sale_per_transaction - average_sale_per_transaction_previous
    delta_total_invoices = total_invoices - total_invoices_previous

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.metric(label="Total Sales", value=f"{total_sales:,} đ", delta=f"{delta_total_sales:+,} đ")
        

    with middle_column:
        st.metric(label="Average Sales", value=f"{average_sale_per_transaction:,} đ", delta=f"{delta_average_sale_per_transaction:+,.2f} đ")

    with right_column:
        st.metric(label="Total Invoices", value=total_invoices, delta=f"{delta_total_invoices:+,}")

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
    st.markdown("---")
    left_column, right_column = st.columns(2)
    with left_column:
        # non_khach_le_transaction = df[df['Customer_Name'] != 'khách lẻ']
        # non_khach_le_transaction_pattern = non_khach_le_transaction.groupby('Customer_Name')['PurchaseDate'].sum().reset_index()
        # print(non_khach_le_transaction_pattern)
        st.metric(label="Total Membership", value=f"{total_customer}")
        # st.dataframe(non_khach_le_transaction_pattern,
        #     column_order=("Customer_Name", "PurchaseDate"),
        #     hide_index=True,
        #     width=None,
        #     column_config={
        #         "Customer_Name": st.column_config.TextColumn(
        #             "Customer_Name",
        #         ),
        #         "Behaviors": st.column_config.LineChartColumn(

        #         )
        #     }
        #     )
    with right_column:
        st.metric(label="Top Membership", value=None)
        df_customer_sorted = df_customer.sort_values(by='Total_Revenue',ascending=False)
        df_customer_sorted = df_customer_sorted[['Name', 'Total_Revenue']]
        print(df_customer_sorted)
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
                    max_value=max(df_customer.Total_Revenue),
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

    
    


        



        

