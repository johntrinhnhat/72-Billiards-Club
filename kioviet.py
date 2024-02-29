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
Retailer = os.getenv('Retailer')
access_token = os.getenv('access_token')
url_invoices = os.getenv('url')
url_customer = os.getenv('url_customer')
url_invoice = os.getenv('url_invoice')

# def get_api_data(url, headers):
#     """Perform an API request to the specified URL with given headers and return the 'Data' part of the JSON response."""
#     response = requests.get(url, headers=headers)
#     response.raise_for_status()  # Raise an exception for HTTP error responses
#     return response.json()["Data"]

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
#     headers = {'Retailer': Retailer, 'Authorization': f'Bearer {access_token}'}
#     # fetch Invoices & Customers data
#     invoices_data = get_api_data(url_invoices, headers)

    
#     each_invoice_data = get_api_data_each_invoice(url_invoice, headers)
#     customers_data = get_api_data(url_customer, headers)

#     # process invoices data
#     df = process_invoices_data(invoices_data)
#     df_invoice = pd.DataFrame(each_invoice_data)
#     df_invoice = df_invoice[['InvoiceId', 'CategoryTree', 'ProductName', 'Note', 'Quantity', 'Price',]]

#     print(df_invoice)
#     print(df_invoice.columns)

#     # print(df)


# if __name__ == "__main__":
#     main()

""""""""""""""""""" INVOICES DATA """""""""""""""""""

# Set up API request details
invoices_url = url_invoices
invoices_headers = {
    'Retailer': f'{Retailer}', 
    'Authorization': f'Bearer {access_token}',
}

# Perform the API request
response = requests.get(invoices_url, headers=invoices_headers)
response_data = response.json()
data = response_data["Data"]
invoice_data = []
invoice_id=[]
# Process each item in the response data
for item in data:

    # Extract date and hour using regular expression
    purchase_date_match = re.search(r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})', item.get("PurchaseDate", ""))
    purchase_date = purchase_date_match.group(1) if purchase_date_match else None
    purchase_hour = purchase_date_match.group(2) if purchase_date_match else None

    entry_date_match =re.search(r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})', item.get("EntryDate", ""))
    entry_hour = entry_date_match.group(2) if entry_date_match else None
    
    # Define data schema for each invoice
    invoice_schema = {
        'Id': item["Id"],
        'Table_Id': item.get("TableId", None),
        'Customer_Name': item["CustomerName"],
        'PurchaseDate': purchase_date,
        'EntryHour': entry_hour,
        'PurchaseHour': purchase_hour,
        'Discount': item["Discount"],
        'Total_Payment': item["TotalPayment"],
        'Status': item["StatusValue"],
    }

    id_schema = item["Id"]
    # Add invoice to list if BranchId is not 0
    if invoice_schema["Id"] != -1:
        invoice_data.append(invoice_schema)
        invoice_id.append(id_schema)
    
# Print total number of invoice_data processed
print(f"Invoice_ID: {invoice_id}")
print(f"Total invoices: {len(invoice_data)}")
# Define CSV field names
invoice_data_fieldnames=['Id', 'Table_Id', 'Customer_Name', 'PurchaseDate', 'EntryHour', 'PurchaseHour', 'Discount', 'Total_Payment', 'Status']

# Write invoices data to a CSV file
with open ('kioviet.csv', 'w', encoding='utf-8') as kioviet_file:
    writer = csv.DictWriter(kioviet_file, fieldnames=invoice_data_fieldnames)
    writer.writeheader()

    writer.writerows(invoice_data)



""""""""""""""""""" CUSTOMERS DATA """""""""""""""""""

# Set up API request details
customers_url = url_customer
customers_headers = {
    'Retailer': f'{Retailer}', 
    'Authorization': f'Bearer {access_token}',
}

# Perform the API request
response = requests.get(customers_url, headers=customers_headers)
response_data = response.json()
customers_data = response_data["Data"]

customers = []

for customer in customers_data:

    # Extract date and hour using regular expression
    create_date_match = re.search(r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})', customer.get("CreatedDate", ""))
    create_date = create_date_match.group(1) if create_date_match else None

    customers_data_schema = {
        'Id': customer["Id"],
        'BranchId': customer["BranchId"],
        'Name': customer.get("Name", None),
        'Contact_Number': customer.get("ContactNumber", None),
        'Membership': customer.get("Groups",None),
        'Created_Date': create_date,
        'Debt': customer.get("Debt", None),
        'Total_Revenue': customer.get("TotalRevenue"),
        'Is_Active': customer["IsActive"],
        'Last_Trading_Date': customer.get("LastTradingDate", None)
    }

    # Add invoice to list if BranchId is not 0
    if customers_data_schema["BranchId"] != 0:
        customers.append(customers_data_schema)

print(f"Total Customers: {len(customers)}")

# Define CSV field names
customer_fieldnames=['Id', 'BranchId', 'Name', 'Contact_Number', 'Membership', 'Created_Date', 'Debt', 'Total_Revenue', 'Is_Active', 'Last_Trading_Date']

# Write data to a CSV file
with open ('kioviet_customer.csv', 'w', encoding='utf-8') as kioviet_customer_file:
    writer = csv.DictWriter(kioviet_customer_file, fieldnames=customer_fieldnames)
    writer.writeheader()

    writer.writerows(customers)

"""
INVOICES DATA PROCESS
"""
all_data_fieldnames=['Id', 'Table_Id', 'Customer_Name', 'PurchaseDate', 'EntryHour', 'PurchaseHour', 'Discount', 'Total_Payment', 'Status']
# Load data into a DataFrame
df = pd.read_csv('kioviet.csv')
# Change column name 
df.rename(columns={
    'EntryHour':'Check_In',
    'PurchaseHour': 'Check_Out',
    'Total_Payment': 'Sales'},
    inplace=True)
# Replace missing values in 'Customer_Name' with 'khách lẻ'
df['Customer_Name'].fillna('khách lẻ', inplace=True)
# Change value of `Status` from `hoàn thành` to `done`
df['Status'].replace({'Hoàn thành': 'Done'}, inplace=True)
# Drop bias value 
df = df[~df['Sales'].isin([6555000, 2836000, 0])]
# Drop any column that have `status` values `Đã hủy`
df = df[df['Status'] != 'Đã hủy']
# Convert `PurchaseDate` to datetime object
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
# Extract features from `PurchaseDate`
df['DayOfWeek'] = df['PurchaseDate'].dt.day_name()
replacement_dict = {
    1000056: 1,
    1000057: 2,
    1000058: 3,
    1000059: 4,
    1000060: 5,
    1000061: 6,
    1000062: 7,
    1000063: 8,
    1000064: 9,
    1000065:10,
    1000066:11,
    1000067:12,
    1000068:13,
    1000069:14,
    1000070:15,
    1010514: 16,
    1010515: 17,
    # Add more mappings as needed
}
df['Table_Id'] = df['Table_Id'].replace(replacement_dict).fillna('Unknown')
df['Check_In'] = pd.to_datetime(df['Check_In'], format='%H:%M').dt.time
df['Check_Out'] = pd.to_datetime(df['Check_Out'], format='%H:%M').dt.time

# Function to calculate the duration in minutes
def calculate_duration(entry, exit):
    # Convert time objects back to strings
    entry_str = entry.strftime('%H:%M')
    exit_str = exit.strftime('%H:%M')
    # Parse strings to datetime objects
    entry_time = datetime.strptime(entry_str, '%H:%M')
    exit_time = datetime.strptime(exit_str, '%H:%M')
    # Calculate duration
    duration = (exit_time - entry_time).total_seconds() / 60
    # If duration is negative, assume ExitHour is on the next day and add 24 hours worth of minutes
    return duration if duration >= 0 else duration + 24*60

# Apply the function to calculate duration
df['Duration(min)'] = df.apply(lambda row: calculate_duration(row['Check_In'], row['Check_Out']), axis=1)

"""
CUSTOMERS DATA PROCESS
"""
# Load data into a DataFrame
df_customer = pd.read_csv('kioviet_customer.csv')
# Replace missing value in debt with
columns_to_fill = ['Debt', 'Membership', 'Last_Trading_Date']
df_customer[columns_to_fill] = df_customer[columns_to_fill].fillna('None')
# Replace 0 in 'Debt' with 'None' using a vectorized operation
df_customer['Debt'] = df_customer['Debt'].replace({0: 'None'})
# Ensure 'Contact_Number' is treated as strings, remove any '.0', and add leading '0' if not present
df_customer['Contact_Number'] = df_customer['Contact_Number'].astype(str).str.replace('.0', '', regex=False)
df_customer['Contact_Number'] = df_customer['Contact_Number'].apply(lambda x: '0' + x.lstrip('0'))
# Ensure consistent formatting: ###-###-####
df_customer['Contact_Number'] = df_customer['Contact_Number'].apply(lambda x: x[:3] + '-' + x[3:6] + '-' + x[6:])



# Select only the desired columns
df = df[['Table_Id', 'Customer_Name', 'PurchaseDate', 'DayOfWeek', 'Check_In', 'Check_Out', 'Duration(min)', 'Discount', 'Sales', 'Status']]
df_customer = df_customer[['Name', 'Contact_Number', 'Membership', 'Created_Date', 'Debt', 'Total_Revenue', 'Last_Trading_Date']]

# Print dataframes and their types
print(df, df.dtypes)
print(df_customer, df_customer.dtypes)

# Save DataFrame to CSV file
df.to_csv('kioviet.csv', index=False)
df_customer.to_csv('kioviet_customer.csv', index=False)


""""""""""""""""""" IMPORT DATA TO GOOGLE SHEET """""""""""""""""""

# Defind the scope of the application
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
# Add credential to account
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json')
# Authorize the clientsheet
client = gspread.authorize(creds)
# Open the sheet
sheet = client.open('72BilliardsClub')
# Get sheets
sheet_1 = sheet.get_worksheet(0)
sheet_2= sheet.get_worksheet(1)


# Convert 'Check_In' and 'Check_Out' from time to string format to avoid serialization issues
df['Check_In'] = df['Check_In'].apply(lambda x: x.strftime('%H:%M') if not pd.isnull(x) else None)
df['Check_Out'] = df['Check_Out'].apply(lambda x: x.strftime('%H:%M') if not pd.isnull(x) else None)
# Convert api dataframe to a list of lists, ensuring dates are in string format for serialization
df['PurchaseDate'] = df['PurchaseDate'].dt.strftime('%Y-%m-%d')

# Use 'set_with_dataframe' for a direct transfer
try:
    # Update the first worksheet with df
    set_with_dataframe(sheet_1, df, include_column_header=True, resize=False)
    # Update the second worksheet with df_customer
    set_with_dataframe(sheet_2, df_customer, include_column_header=True, resize=True)
    print("\nSuccessfully imported data to Google Sheets ✅\n")
except gspread.exceptions.APIError as e:
    print(f"Failed to update Google Sheets due to an API error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


""""""""""""""""""" AUTOMATION GITHUB UPDATE """""""""""""""""""
def run_git_commands():
    try:
        # Navigate to the directory containing your repository
        os.chdir('C:\\Users\\Khoi\\Desktop\\BilliardsClub')
        # Git commands
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Daily update'], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("Changes pushed to GitHub ✅.")
    except subprocess.CalledProcessError as e:
        print(f"Error in Git operation: {e}")

run_git_commands()