from datetime import datetime
import os
import subprocess
import re
import requests
import csv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import base64


# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
Retailer  = os.getenv('Retailer')
access_token = os.getenv('access_token')
url = os.getenv('url')
url_customer=os.getenv('url_customer')


""""""""""""""""""" INVOICES DATA """""""""""""""""""

# Set up API request details
invoices_url = url
invoices_headers = {
    'Retailer': f'{Retailer}', 
    'Authorization': f'Bearer {access_token}',
}

invoices_params = {
    'PurchaseDate': 'datetime',
    'ToDate': 'datetime'
}

# Perform the API request
response = requests.get(invoices_url, headers=invoices_headers, params=invoices_params)
response_data = response.json()
data = response_data["Data"]
all_data = []
# Process each item in the response data
for item in data:

    # Extract date and hour using regular expression
    purchase_date_match = re.search(r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})', item.get("PurchaseDate", ""))
    purchase_date = purchase_date_match.group(1) if purchase_date_match else None
    purchase_hour = purchase_date_match.group(2) if purchase_date_match else None

    entry_date_match =re.search(r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})', item.get("EntryDate", ""))
    entry_hour = entry_date_match.group(2) if entry_date_match else None
    # Define data schema for each invoice
    data_schema = {
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
    # Add invoice to list if BranchId is not 0
    if data_schema["Id"] != -1:
        all_data.append(data_schema)
    
# Print total number of all_data processed
print(f"Total invoices: {len(all_data)}")
# Define CSV field names
all_data_fieldnames=['Id', 'Table_Id', 'Customer_Name', 'PurchaseDate', 'EntryHour', 'PurchaseHour', 'Discount', 'Total_Payment', 'Status']

# Write invoices data to a CSV file
with open ('kioviet.csv', 'w', encoding='utf-8') as kioviet_file:
    writer = csv.DictWriter(kioviet_file, fieldnames=all_data_fieldnames)
    writer.writeheader()

    writer.writerows(all_data)



""""""""""""""""""" CUSTOMERS DATA """""""""""""""""""

# Set up API request details
customers_url = url_customer
customers_headers = {
    'Retailer': f'{Retailer}', 
    'Authorization': f'Bearer {access_token}',
}

customers_params = {
    'name': 'string',
    'contactNumber': 'string',
}

# Perform the API request
response = requests.get(customers_url, headers=customers_headers, params=customers_params)
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
CUSTOMERS DATA PROCESS
"""
# Load data into a DataFrame
df_customer = pd.read_csv('kioviet_customer.csv')
# Replace missing value in debt with
df_customer['Debt'] = df_customer['Debt'].fillna('None')
df_customer['Membership'] = df_customer['Membership'].fillna('None')
df_customer['Debt'] = df_customer['Debt'].replace(0,'None')
df_customer['Last_Trading_Date'] = df_customer['Last_Trading_Date'].fillna('None')
# First ensure that the 'contact_number' column is treated as strings
df_customer['Contact_Number'] = df_customer['Contact_Number'].astype(str)
# Remove any '.0' that comes from floating point representation
df_customer['Contact_Number'] = df_customer['Contact_Number'].str.replace('.0', '', regex=False)
# Now add the leading '0' if it's not already there
df_customer['Contact_Number'] = df_customer['Contact_Number'].apply(lambda x: '0' + x if not x.startswith('0') else x)
# Ensure consistent formatting: ###-###-####
df_customer['Contact_Number'] = df_customer['Contact_Number'].apply(lambda x: x[:3] + '-' + x[3:6] + '-' + x[6:])


"""
INVOICES DATA PROCESS
"""
# Load data into a DataFrame
df = pd.read_csv('kioviet.csv')
# Change column name `Total_Payment` to `Sales`
df.rename(columns={'Total_Payment': 'Sales', 'PurchaseHour': 'Hour'}, inplace=True)
# 
# df['Discount'] = df['Discount'].replace(0, None)
# Replace missing values in 'Customer_Name' with 'khách lẻ'
df['Customer_Name'] = df['Customer_Name'].fillna('khách lẻ')
# Change value of `Status` from `hoàn thành` to `done`
df.loc[df['Status'] == 'Hoàn thành', 'Status'] = 'Done'
# Drop bias value 
df = df[df['Sales'] != 6555000]
df = df[df['Sales'] != 2836000]
# Remove 0 Sales
df = df[df['Sales'] != 0]
# Drop any column that have `status` values `Đã hủy`
df = df[df['Status'] != 'Đã hủy']
# Convert `PurchaseDate` to datetime object
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
# Extract features from `PurchaseDate`
df['DayOfWeek'] = df['PurchaseDate'].dt.day_name()
# df['Hour'] = df['Hour'].apply(lambda x: datetime.strptime(x, '%H:%M').hour)
# df['Hour'] = df['PurchaseDate'].dt.hour
# df['Year'] = df['PurchaseDate'].dt.year
# df['Month'] = df['PurchaseDate'].dt.month
# df['Day'] = df['PurchaseDate'].dt.day
# Assuming 'purchasehour' is in 'HH:MM' format
# Convert it to a datetime object and then extract the hour

"""
POOL_TALBE DATA PROCESS
"""
df_pool = df.copy()
print(f"df_pool: {df_pool}")
df_pool = df_pool.rename(columns={"Hour": "Check_Out", "EntryHour": "Check_In", "PurchaseDate": "Date"})
# Convert EntryHour and Check_Out to datetime, assuming they are strings in the format "HH:MM"
df_pool['Check_In'] = pd.to_datetime(df_pool['Check_In'], format='%H:%M').dt.time
df_pool['Check_Out'] = pd.to_datetime(df_pool['Check_Out'], format='%H:%M').dt.time

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
df_pool['Duration(min)'] = df_pool.apply(lambda row: calculate_duration(row['Check_In'], row['Check_Out']), axis=1)
df_pool = df_pool[~df_pool['Duration(min)'].isin([1437, 1438, 1439])]
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
df_pool['Table_Id'] = df_pool['Table_Id'].replace(replacement_dict)
df_pool = df_pool[df_pool['Table_Id'] != 1000071] 

df_pool = df_pool.dropna(subset=['Table_Id'])
# df_pool = df_pool.sort_values(by=['PurchaseDate'], ascending=False)


# Select only the desired columns
df = df[['Customer_Name', 'PurchaseDate', 'Hour', 'DayOfWeek', 'Discount', 'Sales', 'Status']]
df_customer = df_customer[['Name', 'Contact_Number', 'Membership', 'Created_Date', 'Debt', 'Total_Revenue', 'Last_Trading_Date']]
df_pool = df_pool[['Table_Id', 'Date', 'Check_In', 'Check_Out', 'Duration(min)']]

# Print dataframes and their types
print(df, df.dtypes)
print(df_customer, df_customer.dtypes)
print(df_pool, df_pool.dtypes)

# Save DataFrame to CSV file
df.to_csv('kioviet.csv', index=False)
df_customer.to_csv('kioviet_customer.csv', index=False)
df_pool.to_csv('kioviet_pool.csv', index=False)


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
# Convert api dataframe to a list of lists
df['PurchaseDate'] = df['PurchaseDate'].dt.strftime('%Y-%m-%d')
data_sheet = df.values.tolist()
customer_data_sheet = df_customer.values.tolist()
# Include the header
header = df.columns.tolist()
customer_data_header = df_customer.columns.tolist()
# Insert header
data_sheet.insert(0, header)
customer_data_sheet.insert(0,customer_data_header)

try:
    # Update the first worksheet starting at the first cell
    sheet_1_updated = sheet_1.update(values=data_sheet, range_name='A1')
    # Update the second worksheet starting at the first cell
    sheet_2_updated = sheet_2.update(values=customer_data_sheet, range_name='A1')
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