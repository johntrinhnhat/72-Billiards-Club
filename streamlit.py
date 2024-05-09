import pandas as pd
from plots.sale_plot import sale_plot
import streamlit as st
import base64
import plotly.express as px


github_csv_url = "https://raw.githubusercontent.com/johntrinhnhat/72-Billiards-Club/main/kioviet.csv"
github_csv_customer_url = "https://raw.githubusercontent.com/johntrinhnhat/72-Billiards-Club/main/kioviet_customer.csv"
github_csv_pool_url = "https://raw.githubusercontent.com/johntrinhnhat/72-Billiards-Club/main/kioviet_pool.csv"
# ----------------- STREAMLIT APP -----------------

st.set_page_config(page_title="72 Billiards Club",
                page_icon="🎱",
                layout="wide")

# ----------------- LOAD DATA -----------------
# @st.cache_data
def load_data():
    return pd.read_csv(github_csv_url)
def load_customer_data():
    return pd.read_csv(github_csv_customer_url)

df = load_data()
df_customer = load_customer_data()

# ----------------- PROCESS DATA -----------------
pd.set_option('display.float_format', '{:.2f}'.format)
# SALE DATA
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate']).dt.date
df['Id'] = df['Id'].fillna(0)

df['Time'] = df['Time'].apply(lambda x: int(x.split(':')[0]) if isinstance(x, str) and ':' in x else 0)

df['Discount'] = df['Discount'].astype(int)
df['Sales'] = df['Sales'].astype(int)

df = df.sort_values(by='PurchaseDate', ascending=True) 

# SUMMARY DATAFRAME
df_summary = df.groupby('PurchaseDate').agg({'Sales': 'sum', 
                                          'Discount': 'sum', 
                                          'DayOfWeek': 'first'}).reset_index()
df_summary.rename(columns={"PurchaseDate": "Date", "Sales": "Total_Sale"}, inplace=True)
df_summary = df_summary[["Date", "DayOfWeek", "Discount", "Total_Sale"]]
df_summary = df_summary.sort_values(by="Date", ascending=False)
df_summary['Total_Sale'] = df_summary['Total_Sale'].astype(int)
# print(df_summary)
# ----------------- CSS STYLE -----------------
st.markdown("""
<style>

[data-testid="stMetric"] {
    padding: 15px 0;
}

</style>
""", unsafe_allow_html=True)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------

with st.sidebar:
    # Ensure there are no NaT values and find the minimum and maximum dates
    valid_dates = df['PurchaseDate'].dropna()
    min_date = valid_dates.min()
    max_date = valid_dates.max()

    # Replace the datetime slider with a date input for selecting a range of dates
    start_date = st.sidebar.date_input(
        "From:",
        value = min_date,
        min_value = min_date,
        max_value = max_date,
    )

    end_date = st.sidebar.date_input(
        "To:",
        value= max_date,
        min_value=min_date,
        max_value=max_date,
    )


    hour = st.sidebar.slider(
        "Hour:",
        min_value=int(min(df['Time'].unique())),
        max_value=int(max(df['Time'].unique())),
        value=(int(min(df['Time'].unique())), int(max(df['Time'].unique())))
    )

    dayofweek = st.sidebar.multiselect(
        "DayOfWeek:",
        options=df['DayOfWeek'].unique(),
        default=df['DayOfWeek'].unique()
    )

    df_selection = df.query(
        "Time >= @hour[0] & Time <= @hour[1] & "
        "DayOfWeek == @dayofweek &"
        "PurchaseDate >= @start_date & PurchaseDate <= @end_date"
    )

    df_selection = df_selection[['Cashier', 'Customer_Name', 'PurchaseDate', 'DayOfWeek', 'Time', 'Duration(hour)', 'Sales','Discount', 'Status']]
    # df_selection['Time'] = df_selection['Time'].apply(lambda x: f"{int(x):02d}:{int((x-int(x))*60):02d}")

    print(df_selection)

    def highlight_sales(val):
        color = 'green' if val > 360000 else ''
        return f'background-color: {color}'
    
    styled_df_selection = df_selection.style.format({"Sales": "{:,.0f} đ", "Duration(hour)": "{:.2f}"}).map(highlight_sales, subset=['Sales', 'Duration(hour)'])


# ----------------- MAIN PAGE -----------------
    
# LOGO
st.image('./logo.png')
st.markdown("##")

# TAB
tab1, tab2, tab3 = st.tabs(["SALE", "MEMBERSHIP", "TABLE"])

# TAB_1
with tab1:
    # TOP KPI's

    # Calculate current period KPIs
    total_sales = int(df_selection['Sales'].sum())  
    average_sale_per_transaction = round(df_selection['Sales'].mean(), 2)
    total_invoices = len(df_selection)
    average_monthly_sale = round(df_summary['Total_Sale'].mean(), 2)

    st.divider()
    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.metric(label="Total Sales", value=f"{total_sales:,} đ")
    with middle_column:
        st.metric(label="Average Sales", value=f"{average_sale_per_transaction:,} đ")
    with right_column:
        st.metric(label="Total Invoices", value=total_invoices)
    st.divider()


    # Display Sale Dataframe
    st.dataframe(styled_df_selection, width=850)
    # Download data
    def filedownload(df_selection):
        csv = df_selection.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode() # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64, {b64}" download="data.csv">Download Excel file</a>'
        return href 
    # Display the download link
    st.markdown(filedownload(df_selection), unsafe_allow_html=True)

    ## Button Show Plots
    if st.button('Show Plots'):
        sale_plot(df_selection)

    st.divider()

    sale_frame_column, metric_column = st.columns([4,2])
    with sale_frame_column:
        st.dataframe(df_summary, width=650)
    with metric_column:
        summary_stats = df_summary.describe().astype(int)
        # print(summary_stats)
        st.write(summary_stats)
    st.divider()

# TAB_2
with tab2:
    total_customer = len(df_customer)
    st.markdown("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.metric(label="Total Membership", value=f"{total_customer}")
    with right_column:
        st.metric(label="Top Debt", value=None)
        df_customer_sorted = df_customer.sort_values(by='Debt',ascending=False)
        df_customer_sorted['Debt'] = df_customer_sorted['Debt'].apply(lambda x: f"{x:,}")
        df_customer_sorted = df_customer_sorted[['Name', 'Debt']]
        # print(df_customer_sorted)
        st.dataframe(df_customer_sorted,
            column_order=("Name", "Debt"),
            hide_index=True,
            width=None,
            column_config={
                "Name": st.column_config.TextColumn(
                    "Name",
                ),
                "Debt": st.column_config.ProgressColumn(
                    "Debt",
                    format="%d đ",
                    min_value=0,
                    max_value=int(max(df_customer.Debt)),
                ),
                }
            )
    st.markdown("---")
    st.dataframe(df_customer[['Name', 'Gender', 'Contact_Number', 'Created_Date', 'Debt']])

    # # Button Show Plots
    # if st.button('Show Plot'):
    #     # Bar Chart (Number of Membership)
    #     st.title('Number of Membership')
    #     membership_counts = df_customer['Membership'].value_counts(dropna=False).reset_index()
    #     membership_counts.columns = ['Membership', 'Count']
    #     membership_counts['Membership'] = membership_counts['Membership'].fillna('None')
    #     fig = px.bar(membership_counts, x='Membership', y='Count', 
    #                 hover_data=['Membership', 'Count'], color='Count')
    #     st.plotly_chart(fig)

# TAB_3
with tab3:
    st.divider()
    left_column, right_column = st.columns(2)
    with left_column:
        st.metric(label="Total Table", value=15)
    with right_column:
        st.metric(label="Metric", value="")
        desc_stats = df['Duration(hour)'].describe()
        st.write(desc_stats)
    
    def highlight_PS5(val):
        color = '#FFA7FD' if val == 16 or val == 17 else ''
        return f'color: {color}'
    
    df_style = df.style.map(highlight_PS5, subset=['Table_Id'])
    # st.dataframe(df_style, width=650)



# ----------------- OCCUPANCY RATE -----------------
    
    # Make a copy of the dataframe
    df_occupancy = df.copy()
    # df_occupancy['Check_In'] = pd.to_datetime(df_occupancy['Check_In'], format='%H:%M:%S')
    # df_occupancy['Duration(hour)'] = pd.to_datetime(df_occupancy['Duration(hour)']).dt.hour
    
    # Assuming 'total_tables' is the total number of tables at the pool hall
    total_tables = 15

    # Group the data by 'Date' and count the number of occupied tables
    df_occupancy = df_occupancy.groupby(['PurchaseDate']).size().reset_index(name='Occupied_Table_Hours')

    # Calculate the occupancy rate by dividing the occupied table hours by the total potential table hours in a day
    df_occupancy['Rate (%)'] = ((df_occupancy['Occupied_Table_Hours'] / (total_tables * 18)) * 100).round().astype(int)

    # Sort by date in descending order
    df_occupancy = df_occupancy.sort_values(by=["PurchaseDate"], ascending=False)

    # Print the resulting dataframe and datatypes
    # print(df_occupancy, df_occupancy.dtypes)

    st.title('Occupancy Rate')
    left_column, right_column = st.columns([3,2])
    with left_column:
        # The result is a DataFrame with the occupancy rate for each day
        st.dataframe(df_occupancy, width=650)
    with right_column: 
        # Display a metric, for example the average occupancy rate
        st.metric(label="Metric", value="")
        # Display descriptive statistics for the occupancy rate
        desc_stats = df_occupancy['Rate (%)'].describe()
        st.write(desc_stats)


    # # Initialize the session state variable if it's not already set
    # if 'show_plot' not in st.session_state:
    #     st.session_state.show_plot = False

    # # Define a button and its callback function to toggle the plot visibility
    # if st.button('Show/Hide Plot', key='occupacy_rate'):
    #     # Toggle the boolean value
    #     st.session_state.show_plot = not st.session_state.show_plot

    # # Check the state variable and display the plot accordingly
    # if st.session_state.show_plot:
    #     sns.set_theme()
    #     # Pivot the table to get 'Hour' as columns and 'Date' as rows
    #     occupancy_pivot = df_occupancy.pivot(index="Date", columns="Hour", values="Rate (%)")

    #     # Plot the heatmap
    #     fig, ax = plt.subplots(figsize=(20, 10))
    #     # Set the color of the figure background
    #     fig.patch.set_facecolor('#FFFAF0')
    #     # Set the color of the axes background
    #     ax.set_facecolor('#FFFAF0')
    #     # Rotate the yticks with a 35-degree angle
    #     plt.yticks(rotation=35)
    #     # Create the heatmap with annotations in white color
    #     sns.heatmap(occupancy_pivot, annot=True, annot_kws={"size": 6, "color": "white"}, fmt=".0f", cmap="Oranges", ax=ax)

    #     # Set the title and labels with white color for visibility on a dark background
    #     ax.set_title("Occupancy Rate Heatmap")
    #     ax.set_xlabel("Hour")
    #     ax.set_ylabel("Date")
    #     st.pyplot(fig)

        



        

