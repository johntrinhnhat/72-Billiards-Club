from concurrent.futures import ThreadPoolExecutor, as_completed
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
from colorama import Fore, init
init(autoreset=True)


# Load environment variables from .env
load_dotenv()
retailer = os.getenv('retailer')
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
access_token_url = os.getenv('access_token_url')
url = os.getenv('url')
customer_url = os.getenv('customer_url')
# Create a session for network requests
session = requests.Session()

# Function to get access token and reuse if still valid
access_token = None
def get_access_token():
    global access_token
    if access_token is not None:
        return access_token
    access_token_request = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    response = session.post(access_token_url, data=access_token_request)
    if response.status_code == 200:
        access_token = response.json().get("access_token")
        return access_token
    else:
        raise Exception(f"Failed to get access token: {response.text}")

# Function to fetch invoices and customers in parallel
def fetch_invoices(page, page_size):
    headers = {
        'Retailer': retailer, 
        'Authorization': f'Bearer {get_access_token()}',
    }
    params = {
        'branchId': 245409,
        'pageSize': page_size,
        'currentItem': page * page_size + 1,
        'fromPurchaseDate': '2023-11-07',
    }
    response = session.get(url, headers=headers, params=params)
    data = response.json().get('data', [])
    print(f'Fetching {page_size} invoices in page: {page} ...')
    return data

def process_invoices_data(invoices_data):
    # """"""""""""""""""" INVOICE SCHEMA """""""""""""""""""
    df_invoice = []
    for invoice in invoices_data:
        for detail in invoice['invoiceDetails']:
            duration = detail.get('quantity', '')
            discount = detail.get('discount', '')
            food = detail.get('productName','' )

        # Extract date and hour using regular expression
        purchase_date_match = re.search(r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})', invoice.get("purchaseDate", ""))
        purchase_date = purchase_date_match.group(1) if purchase_date_match else None
        purchase_hour = purchase_date_match.group(2) if purchase_date_match else None
        
        #Define data schema for each invoice
        invoice_schema = {
            'Id': invoice["id"],
            'Cashier': invoice["soldByName"],
            'Customer_Name': invoice["customerName"],
            'PurchaseDate': purchase_date,
            'Time': purchase_hour,
            'Duration(hour)': duration,
            'Discount': discount,
            'Payment': invoice["totalPayment"],
            'Status': invoice["statusValue"],
            'Service': food,
        }
        # Add invoice to list if BranchId is not 0
        if invoice_schema["Id"] != -1:
            df_invoice.append(invoice_schema)
    # """"""""""""""""""" CSV EXPORT_1 """""""""""""""""""
     # Define CSV field names
    invoices_schema=['Id', 'Cashier', 'Customer_Name', 'PurchaseDate', 'Time', 'Service', 'Duration(hour)', 'Discount', 'Payment', 'Status']
    # Write invoices data to a CSV file
    with open ('kioviet1.csv', 'w', encoding='utf-8') as kioviet_file:
        writer = csv.DictWriter(kioviet_file, fieldnames=invoices_schema)
        writer.writeheader()
        writer.writerows(df_invoice)
    
    # """"""""""""""""""" CSV IMPORT """""""""""""""""""
    df_invoice = pd.read_csv('kioviet1.csv')
    # Change column name 
    df_invoice.rename(columns={'Payment': 'Sales'},
        inplace=True)
    # Replace missing values in 'Customer_Name' with 'khách lẻ'
    df_invoice['Customer_Name'] = df_invoice['Customer_Name'].fillna('khách lẻ')
    # Change value of `Status` from `hoàn thành` to `done`
    df_invoice['Status'].replace({'Hoàn thành': 'Done'}, inplace=True)
    # # Drop bias value 
    df_invoice = df_invoice[~df_invoice['Sales'].isin([2836000, 0])]
    # Drop any column that have `status` values `Đã hủy`
    df_invoice = df_invoice[df_invoice['Status'] != 'Đã hủy']
    # Convert `PurchaseDate` to datetime object
    df_invoice['PurchaseDate'] = pd.to_datetime(df_invoice['PurchaseDate'])
    # # Extract features from `PurchaseDate`
    df_invoice['DayOfWeek'] = df_invoice['PurchaseDate'].dt.day_name()

    # """"""""""""""""""" CSV EXPORT_2 """""""""""""""""""
    df_invoice = df_invoice[['Id', 'Cashier', 'Customer_Name', 'PurchaseDate', 'DayOfWeek', 'Time', 'Duration(hour)', 'Service', 'Discount', 'Sales', 'Status']]
    df_invoice.to_csv('kioviet.csv', index=False)
    return df_invoice

def fetch_customers(page, page_size):
    headers = {
        'Retailer': retailer, 
        'Authorization': f'Bearer {get_access_token()}',
    }
    params = {
        'branchId': 245409,
        'pageSize': page_size,
        'currentItem': page * page_size + 1,
        'fromPurchaseDate': '2023-11-07',
    }
    response = session.get(customer_url, headers=headers, params=params)
    data = response.json().get('data', [])
    print(f'Fetching {page_size} customers in page: {page} ...')
    return data

def process_customers_data(customers_data):
    # """"""""""""""""""" CUSTOMER SCHEMA """""""""""""""""""
    df_customer = []
    for customer in customers_data:
        # Extract date and hour using regular expression
        create_date_match = re.search(r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})', customer.get("createdDate", ""))
        create_date = create_date_match.group(1) if create_date_match else None

        # Check the value for gender (assuming "male" and "female" are Boolean attributes)
        gender_bool = customer.get("gender", None)
        if gender_bool is True:
            gender = "Nam"
        elif gender_bool is False:
            gender = "Nữ"
        else:
            gender = "Unknown"  

        customers_data_schema = {
            'Id': customer["id"],
            'Name': customer.get("name", None),
            'Gender': gender,
            'Contact_Number': customer.get("contactNumber", None),
            'Debt': customer.get("debt", None),
            'Created_Date': create_date,
            # 'Total_Revenue': customer.get("totalRevenue"),
            # 'Membership': customer.get("Groups",None),
            #'Last_Trading_Date': customer.get("LastTradingDate", None)
        }

        # Add invoice to list if BranchId is not 0
        if customers_data_schema["Id"] != 0:
            df_customer.append(customers_data_schema)

    # """"""""""""""""""" CSV EXPORT_1 """""""""""""""""""
    # Define CSV field names
    customer_schema=['Id', 'Name', 'Gender', 'Contact_Number', 'Created_Date', 'Debt']
    # Write customers data to a CSV file
    with open ('kioviet_customer1.csv', 'w', encoding='utf-8') as kioviet_customer_file:
        writer = csv.DictWriter(kioviet_customer_file, fieldnames=customer_schema)
        writer.writeheader()
        writer.writerows(df_customer)

    # """"""""""""""""""" CSV IMPORT """""""""""""""""""
    # Load data into a DataFrame
    df_customer = pd.read_csv('kioviet_customer1.csv')
    # Replace missing value in debt with
    df_customer['Debt'] = df_customer['Debt'].fillna('None')
    # Replace 0 in 'Debt' with 'None' using a vectorized operation
    # df_customer['Debt'] = df_customer['Debt'].replace({0: 'None'})
    # Ensure 'Contact_Number' is treated as strings, remove any '.0', and add leading '0' if not present
    df_customer['Contact_Number'] = df_customer['Contact_Number'].astype(str).str.replace('.0', '', regex=False)
    df_customer['Contact_Number'] = df_customer['Contact_Number'].apply(lambda x: '0' + x.lstrip('0'))
    # Ensure consistent formatting: ###-###-####
    df_customer['Contact_Number'] = df_customer['Contact_Number'].apply(lambda x: x[:3] + '-' + x[3:6] + '-' + x[6:])
    # """"""""""""""""""" CSV EXPORT_2 """""""""""""""""""

    # Select only the desired columns
    df_customer = df_customer[['Id', 'Name', 'Gender', 'Contact_Number', 'Created_Date', 'Debt']]
    # Save DataFrame to CSV file
    df_customer.to_csv('kioviet_customer.csv', index=False)

    return df_customer

def google_sheet_import(df_invoice, df_customer):
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

    # Convert api dataframe to a list of lists, ensuring dates are in string format for serialization
    df_invoice['PurchaseDate'] = df_invoice['PurchaseDate'].dt.strftime('%Y-%m-%d')

    # Use 'set_with_dataframe' for a direct transfer
    try:
        # Update the first worksheet with df
        set_with_dataframe(sheet_1, df_invoice, include_column_header=True, resize=False)
        # Update the second worksheet with df_customer
        set_with_dataframe(sheet_2, df_customer, include_column_header=True, resize=True)
        print("\nSuccessfully imported data to Google Sheets ✅\n")
    except gspread.exceptions.APIError as e:
        print(f"Failed to update Google Sheets due to an API error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    page_size = 100  # Adjust as maximum allowed if possible
    pages = 210
    invoices_data = []
    customers_data =[]

# """"""""""""""""""" THREADPOOLEXCECUTOR TO FETCH DATA """""""""""""""""""

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_invoices = [executor.submit(fetch_invoices, page, page_size) for page in range(pages)]
        futures_customers = [executor.submit(fetch_customers, page, page_size) for page in range(pages)]
        
        for future in as_completed(futures_invoices):
            invoices_data.extend(future.result())
        
        for future in as_completed(futures_customers):
            customers_data.extend(future.result())

# """"""""""""""""""" DATA PROCESS TO DATAFRAME"""""""""""""""""""

    df_invoice = process_invoices_data(invoices_data)
    df_customer = process_customers_data(customers_data)

# """"""""""""""""""" PRINT DATAFRAME """""""""""""""""""

    print(df_invoice, df_invoice.dtypes)
    print(df_customer, df_customer.dtypes)

    print(f"{Fore.BLUE}Total_Customers: {len(df_customer)}")
    print(f"{Fore.BLUE}Total_Invoices: {len(df_invoice)}")

# """"""""""""""""""" IMPORT DATA TO GOOGLE SHEET """""""""""""""""""
    google_sheet_import(df_invoice, df_customer)

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
if __name__ == "__main__":
    main()



