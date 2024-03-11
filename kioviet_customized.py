from datetime import datetime
import os
import subprocess
import re
import requests
import csv
import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file to ensure sensitive information is not hard-coded into the script
load_dotenv()
retailer = os.getenv('retailer')
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
access_token_url = os.getenv('access_token_url')
invoices_url = os.getenv('url')
url_customer = os.getenv('url_customer')

# def get_api_data(url, headers):
    # """Perform an API request to the specified URL with given headers and return the 'Data' part of the JSON response."""
    # response = requests.get(url, headers=headers)
    # response.raise_for_status()  # Raise an exception for HTTP error responses
    # return response.json()["data"]

# def get_api_data_each_invoice(url, headers):

#     response = requests.get(url_invoice, headers=headers)
#     response.raise_for_status()
#     return response.json()["Data"]

# def process_invoices_data(data):
#     """
#     Process the raw invoices data into a structured pandas DataFrame.

#     - Renames columns for better readability.
#     - Handles missing values and data transformations.
#     - Extracts additional date/time features from existing columns.
#     """
#     # Convert raw data into a pandas DataFrame
#     df = pd.DataFrame(data)
#     # Rename columns for clarity
#     df = df.rename(columns={
#         'TableId': 'Table_Id',
#         'CustomerName': 'Customer_Name',
#         'EntryDate': 'Check_In',
#         'TotalPayment': 'Sales',
#         'StatusValue': 'Status'
#     })

#     # Replace missing customer names with a default value
#     df['Customer_Name'] = df['Customer_Name'].replace('', 'khách lẻ')
#     # Normalize status values
#     df['Status'] = df['Status'].replace({'Hoàn thành': 'Done'})

#     # Map numerical table IDs to a more readable format
#     replacement_dict = {
#         # Mapping from original ID to a more friendly identifier
#         # Add more mappings as needed
#     }
#     df['Table_Id'] = df['Table_Id'].replace(replacement_dict).fillna('Unknown')

#     # Convert dates to datetime objects for easier manipulation
#     df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
#     df['DayOfWeek'] = df['PurchaseDate'].dt.day_name()
#     df['Check_In'] = pd.to_datetime(df['Check_In']).dt.strftime('%H:%M:%S')
#     df['Check_Out'] = pd.to_datetime(df['PurchaseDate']).dt.strftime('%H:%M:%S')
#     df['PurchaseDate'] = df['PurchaseDate'].dt.date

#     # Categorize data based on the day of the week
#     df['TimeOfWeek'] = df['DayOfWeek'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')
#     # Categorize data based on the time of day
#     df['HourOfDay'] = df['Check_Out'].apply(lambda x: 'Morning' if 9 <= pd.to_datetime(x).hour <= 12 else ('Evening' if 13 <= pd.to_datetime(x).hour <= 18 else ('Noon' if 19 <= pd.to_datetime(x).hour <= 23 or 0 <= pd.to_datetime(x).hour <= 6 else 'unknown')))
#     # Exclude outlier sales values
#     df = df[~df['Sales'].isin([6555000, 2836000, 0])]

#     # Finalize the DataFrame structure
#     df = df[['Table_Id', 'Customer_Name', 'PurchaseDate', 'DayOfWeek', 'TimeOfWeek', 'HourOfDay', 'Check_In', 'Check_Out', 'Discount', 'Sales']]
   

#     return df

# def main():
#     headers = {'Retailer': retailer, 'Authorization': f'Bearer {access_token}'}
#     # fetch Invoices & Customers data
#     invoices_data = get_api_data(invoices_url, headers)

    
#     # each_invoice_data = get_api_data_each_invoice(url_invoice, headers)
#     customers_data = get_api_data(url_customer, headers)

#     # process invoices data
#     df = process_invoices_data(invoices_data)
#     # df_invoice = pd.DataFrame(each_invoice_data)
#     # df_invoice = df_invoice[['InvoiceId', 'CategoryTree', 'ProductName', 'Note', 'Quantity', 'Price',]]

#     print(df)
#     # print(df_invoice.columns)

#     # print(df)


# if __name__ == "__main__":
#     main()