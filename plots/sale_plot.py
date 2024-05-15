        
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objs as go

def sale_plot(df_selection):
        ### Line Chart ( Peak Houly Sales Trend)
        st.title('Check Out Trends')

         # Ensure 'Hour' and 'DayOfWeek' are in the correct format
        df_selection['check_Out'] = df_selection['check_Out'].astype(int)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_selection['dayOfWeek'] = pd.Categorical(df_selection['dayOfWeek'], categories=days_order, ordered=True)

        # Aggregate sales by hour, day of the week, and purchase date
        hourly_sales = df_selection.groupby('check_Out')['revenue'].max().reset_index()
        dayofweek_sales = df_selection.groupby('dayOfWeek', observed=True)['revenue'].sum().reset_index()
        df_selection['purchase_Date'] = pd.to_datetime(df_selection['purchase_Date'])
        purchasedate_sales = df_selection.groupby(df_selection['purchase_Date'].dt.date)['revenue'].sum().reset_index()
        purchasedate_sales = purchasedate_sales.sort_values('purchase_Date')
        # hour_sale = df_selection[['Hour','Sales']]

        # Create a subplot figure
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Check-Out by Hour", "Check-Out by Day of the Week", "Check-Out by Purchase Date"))

        # Add Hourly Sales trace
        fig.add_trace(
            go.Scatter(x=hourly_sales['check_Out'], y=hourly_sales['revenue'], name="Hourly Sales", marker_color='#FFA500'),
            row=1, col=1,
        )

        # Add Day of Week Sales trace
        fig.add_trace(
            go.Scatter(x=dayofweek_sales['dayOfWeek'], y=dayofweek_sales['revenue'], name="Day of Week Sales", marker_color='#ED64A6'),
            row=2, col=1,
        )

        # Add Purchase Date Sales trace
        fig.add_trace(
            go.Scatter(x=purchasedate_sales['purchase_Date'], y=purchasedate_sales['revenue'], name="Purchase Date Sales", marker_color='#00BFFF'),
            row=3, col=1,
        )

        # Update x-axis titles and y-axis titles
        fig.update_xaxes(title_text="Hour", row=1, col=1)
        fig.update_xaxes(title_text="Day of Week", row=2, col=1)
        fig.update_xaxes(title_text="Purchase Date", row=3, col=1)

        fig.update_yaxes(title_text="revenue", row=1, col=1)
        fig.update_yaxes(title_text="revenue", row=2, col=1)
        fig.update_yaxes(title_text="revenue", row=3, col=1)

        # Update layout
        fig.update_layout(height=900, showlegend=False)

        # Display the figure in Streamlit
        st.plotly_chart(fig)

        ### Line Chart (Purchasing Pattern)
        st.title("Purchasing Behavior of 'khách lẻ'")
        # Filter transactions for 'khách lẻ'
        df_guest = df_selection[df_selection['customer_Name'] == 'Khách lẻ'].copy()
        print(df_guest)
        # Aggregate sales by purchase date, day of the week and hour
        df_guest['dayOfWeek'] = pd.Categorical(df_guest['dayOfWeek'], categories=days_order, ordered=True)
        hourly_sales_guest = df_guest.groupby('check_Out')['revenue'].sum().reset_index()
        dayofweek_sales_guest = df_guest.groupby('dayOfWeek', observed=True)['revenue'].sum().reset_index()
        purchasedate_sales_guest = df_guest.groupby(df_guest['purchase_Date'].dt.date)['revenue'].sum().reset_index()

        # Create a subplot figure
        fig = make_subplots(rows=3, cols=1, subplot_titles=("Purchasing Behavior by Purchase Date", "Purchasing Behavior by Day of the Week", "Purchasing Behavior by Hour"))

        # Add Purchase Date Sales trace
        fig.add_trace(
            go.Scatter(x=purchasedate_sales_guest['purchase_Date'], y=purchasedate_sales_guest['revenue'], name="Purchase Date Sales", marker_color='#00BFFF'),
            row=1, col=1,
        )

        # Add Day of Week Sales trace
        fig.add_trace(
            go.Scatter(x=dayofweek_sales_guest['dayOfWeek'], y=dayofweek_sales_guest['revenue'], name="Day of Week Sales", marker_color='#ED64A6'),
            row=2, col=1,
        )

        # Add Hourly Sales trace
        fig.add_trace(
            go.Scatter(x=hourly_sales_guest['check_Out'], y=hourly_sales_guest['revenue'], name="Hourly Sales", marker_color='#FFA500'),
            row=3, col=1,
        )

        # Update x-axis titles
        fig.update_xaxes(title_text="Purchase Date", row=1, col=1)
        fig.update_xaxes(title_text="Day of the Week", row=2, col=1)
        fig.update_xaxes(title_text="Hour", row=3, col=1)

        # Update y-axis titles
        fig.update_yaxes(title_text="revenue", row=1, col=1)
        fig.update_yaxes(title_text="revenue", row=2, col=1)
        fig.update_yaxes(title_text="revenue", row=3, col=1)

        # Update layout
        fig.update_layout(height=900, showlegend=False)

        # Display the figure in Streamlit
        st.plotly_chart(fig)