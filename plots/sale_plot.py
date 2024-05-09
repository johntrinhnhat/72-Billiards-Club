        
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objs as go

def sale_plot(df_selection):
        ### Line Chart ( Peak Houly Sales Trend)
        st.title('Sales Trend')

         # Ensure 'Hour' and 'DayOfWeek' are in the correct format
        df_selection['Time'] = df_selection['Time'].astype(int)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_selection['DayOfWeek'] = pd.Categorical(df_selection['DayOfWeek'], categories=days_order, ordered=True)

        # Aggregate sales by hour, day of the week, and purchase date
        hourly_sales = df_selection.groupby('Time')['Sales'].max().reset_index()
        dayofweek_sales = df_selection.groupby('DayOfWeek', observed=True)['Sales'].sum().reset_index()
        df_selection['PurchaseDate'] = pd.to_datetime(df_selection['PurchaseDate'])
        purchasedate_sales = df_selection.groupby(df_selection['PurchaseDate'].dt.date)['Sales'].sum().reset_index()
        purchasedate_sales = purchasedate_sales.sort_values('PurchaseDate')
        # hour_sale = df_selection[['Hour','Sales']]

        # Create a subplot figure
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Peak Hourly Sales Trend", "Sales Trend by Day of the Week", "Sales Trend by Purchase Date"))

        # Add Hourly Sales trace
        fig.add_trace(
            go.Scatter(x=hourly_sales['Time'], y=hourly_sales['Sales'], name="Hourly Sales", marker_color='#FFA500'),
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

        # Update x-axis titles and y-axis titles
        fig.update_xaxes(title_text="Hour", row=1, col=1)
        fig.update_xaxes(title_text="Day of Week", row=2, col=1)
        fig.update_xaxes(title_text="Purchase Date", row=3, col=1)

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
        df_guest = df_selection[df_selection['Customer_Name'] == 'khách lẻ'].copy()
        # print(df_guest)

        # Aggregate sales by purchase date, day of the week and hour
        df_guest['DayOfWeek'] = pd.Categorical(df_guest['DayOfWeek'], categories=days_order, ordered=True)
        hourly_sales_guest = df_guest.groupby('Time')['Sales'].sum().reset_index()
        dayofweek_sales_guest = df_guest.groupby('DayOfWeek', observed=True)['Sales'].sum().reset_index()
        purchasedate_sales_guest = df_guest.groupby(df_guest['PurchaseDate'].dt.date)['Sales'].sum().reset_index()

        # Create a subplot figure
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Purchasing Behavior by Purchase Date", "Purchasing Behavior by Day of the Week", "Purchasing Behavior by Hour"))

        # Add Purchase Date Sales trace
        fig.add_trace(
            go.Scatter(x=purchasedate_sales_guest['PurchaseDate'], y=purchasedate_sales_guest['Sales'], name="Purchase Date Sales", marker_color='#00BFFF'),
            row=1, col=1,
        )

        # Add Day of Week Sales trace
        fig.add_trace(
            go.Scatter(x=dayofweek_sales_guest['DayOfWeek'], y=dayofweek_sales_guest['Sales'], name="Day of Week Sales", marker_color='#ED64A6'),
            row=2, col=1,
        )

        # Add Hourly Sales trace
        fig.add_trace(
            go.Scatter(x=hourly_sales_guest['Time'], y=hourly_sales_guest['Sales'], name="Hourly Sales", marker_color='#FFA500'),
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