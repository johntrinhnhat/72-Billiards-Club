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

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()
Retailer  = os.getenv('Retailer')
access_token = os.getenv('access_token')
url = os.getenv('url')

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

invoices = []

# Process each item in the response data
for item in data:

    # Extract date and hour using regular expression
    purchase_date_match = re.search(r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})', item.get("PurchaseDate", ""))
    purchase_date = purchase_date_match.group(1) if purchase_date_match else None
    purchase_hour = purchase_date_match.group(2) if purchase_date_match else None

    # Define data schema for each invoice
    data_schema = {
        'Id': item["Id"],
        'BranchId': item["BranchId"],
        'Customer_Name': item["CustomerName"],
        'PurchaseDate': purchase_date,
        'PurchaseHour': purchase_hour,
        'Total_Payment': item["TotalPayment"],
        'Status': item["StatusValue"],
    }
    # Add invoice to list if BranchId is not 0
    if data_schema["BranchId"] != 0:
        invoices.append(data_schema)
    
# Print total number of invoices processed
print(f"Total Invoices: {len(invoices)}")

# Define CSV field names
fieldnames=['Id', 'BranchId', 'Customer_Name', 'PurchaseDate', 'PurchaseHour', 'Total_Payment', 'Status']

# Write data to a CSV file
with open ('kioviet.csv', 'w', encoding='utf-8') as kioviet_file:
    writer = csv.DictWriter(kioviet_file, fieldnames=fieldnames)
    writer.writeheader()

    writer.writerows(invoices)


# Load data into a DataFrame
df = pd.read_csv('kioviet.csv')

# Replace missing values in 'Customer_Name' with 'khách lẻ'
df['Customer_Name'] = df['Customer_Name'].fillna('khách lẻ')

# Convert `PurchaseDate` to datetime object
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
# Extract features from `PurchaseDate`
df['Year'] = df['PurchaseDate'].dt.year
df['Month'] = df['PurchaseDate'].dt.month
df['Day'] = df['PurchaseDate'].dt.day

day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['DayOfWeek'] = df['PurchaseDate'].dt.dayofweek.map(day_map)

# Assuming 'purchasehour' is in 'HH:MM' format
# Convert it to a datetime object and then extract the hour
df['PurchaseHour'] = df['PurchaseHour'].apply(lambda x: datetime.strptime(x, '%H:%M').hour)

# The columns want to keep
columns_to_keep = ['Id', 'Customer_Name', 'Year', 'Month', 'Day', 'PurchaseHour', 'DayOfWeek', 'Total_Payment', 'Status']

# Select only the desired columns
df = df[columns_to_keep]

# Change title `Total_Payment` to `Sales`
df.rename(columns={'Total_Payment': 'Sales', 'PurchaseHour': 'Hour'}, inplace=True)

# Change value of `Status` from `hoàn thành` to `done`
df.loc[df['Status'] == 'Hoàn thành', 'Status'] = 'Done'

# Drop bias value 
df = df[df['Sales'] != 6555000]
df = df[df['Sales'] != 2836000]

# Drop any column that have `status` values `Đã hủy`
df = df[df['Status'] != 'Đã hủy']

print(df.head(), df.shape)

# Save DataFrame to CSV file
df.to_csv('kioviet.csv', index=False)


### IMPORT DATA TO GOOGLE SHEET

# Defind the scope of the application
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

# Add credential to account
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json')

# Authorize the clientsheet
client = gspread.authorize(creds)

# Open the sheet
sheet = client.open('kioviet_api_data_live')

# Get sheets
sheet_1 = sheet.get_worksheet(0)


# Convert api dataframe to a list of lists
data_sheet = df.values.tolist()

# Include the header
header = df.columns.tolist()
data_sheet.insert(0, header)

try:
    # Update the new worksheet starting at the first cell
    sheet_1_updated = sheet_1.update(range_name='A1', values=data_sheet)
finally:
    print(f"\nSuccessfuly import data to Google Sheet ✅\n")

def run_git_commands():
    try:
        # Navigate to the directory containing your repository
        os.chdir('C:\\Users\\Khoi\\Desktop\\BilliardsClub')
        # Git commands
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Daily update'], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("Changes pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Error in Git operation: {e}")

# Call the function at the end of your script
run_git_commands()