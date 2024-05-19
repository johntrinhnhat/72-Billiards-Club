import subprocess
import os
import re
import aiohttp
import asyncio
import time
import gspread
from dotenv import load_dotenv
from gspread_dataframe import set_with_dataframe
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from colorama import Fore, init

init(autoreset=True)
start_time = time.time()  # Record the start time

# Initialize and load environment variables
load_dotenv()
ENV_VARS = ["retailer", "client_id", "client_secret", "access_token_url", "url", "customer_url"]
retailer, client_id, client_secret, access_token_url, url, customer_url = [os.getenv(var) for var in ENV_VARS]

# Caching access token with timeout handling
access_token = None
last_token_time = datetime.min
TOKEN_EXPIRY = 3500  # slightly less than one hour to handle edge cases

async def get_access_token():
    global access_token, last_token_time
    if access_token and (datetime.now() - last_token_time).seconds < TOKEN_EXPIRY:
        return access_token
    async with aiohttp.ClientSession() as session:
        response = await session.post(access_token_url, data={
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
        })
        response_json = await response.json()
        access_token = response_json["access_token"]
        last_token_time = datetime.now()
        return access_token

# Compiled regular expression for date and time extraction
date_time_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})')
note_pattern = re.compile(r'Từ \d{2}/\d{2}/\d{4} (\d{2}:\d{2}) đến \d{2}/\d{2}/\d{4} \d{2}:\d{2} \(\d+ giờ \d+ phút\)')

async def fetch_data(session, page, page_size, data_type):
    headers = {
        'Retailer': retailer, 
        'Authorization': f'Bearer {await get_access_token()}',
    }
    params = {
        'branchId': 245409,
        'pageSize': page_size,
        'currentItem': page * page_size + 1,
        'fromPurchaseDate': '2023-11-07',
    }
    url_map = {
        'invoices': url,
        'customers': customer_url,
    }
    async with session.get(url_map[data_type], headers=headers, params=params) as response:
        data = await response.json()
        print(f'Fetching {page_size} {data_type} in page: {page} ...')
        return data.get('data', [])

async def fetch_all_data(pages, page_size):
    async with aiohttp.ClientSession() as session:
        tasks_invoices = [fetch_data(session, page, page_size, 'invoices') for page in range(pages)]
        tasks_customers = [fetch_data(session, page, page_size, 'customers') for page in range(pages)]
        
        all_invoices = await asyncio.gather(*tasks_invoices)
        all_customers = await asyncio.gather(*tasks_customers)
        
        return all_invoices, all_customers

def process_invoices_data(invoices_data):
    df_invoice = []
    df_goods = []
    product_code = ['SP000090', 'SP000091', 'SP000092', 'SP000093', 'SP000094', 'SP000096', 'SP000097', 'ComboK', 'ComboS', 'SP000076', 'SP000067', 'SP000066']
    for invoice in invoices_data:
        purchase_date_match = date_time_pattern.search(invoice.get("purchaseDate", ""))
        date, hour = purchase_date_match.groups() if purchase_date_match else (None, None)

        for detail in invoice.get("invoiceDetails", []):
            if detail.get("productCode") in product_code:
                duration = detail.get("quantity", "")
                note = detail.get("note", "")
            else:
                duration = ""
                note = ""
            note_match = note_pattern.search(note)
            if note_match:
                check_In = note_match.group(1)
            else:
                check_In = ""
            
            goods_schema = {
                'id': invoice["id"],
                'purchase_Date': date,
                'product_Name': detail.get('productName').capitalize(),
                'quantity': detail.get('quantity'),
                'discount': detail.get('discount'),
                'revenue': detail.get('subTotal'),
            }

            df_goods.append(goods_schema)
        
        invoice_schema = {
                'id': invoice["id"],
                'customer_Name': invoice.get("customerName", "").title(),
                'purchase_Date': date,
                'check_In': check_In,
                'check_Out': hour,
                'duration_Hour': duration,
                'discount': invoice.get("discount"),
                'revenue': invoice.get("totalPayment"),
                'status': invoice.get("status"),
        }
        df_invoice.append(invoice_schema)

    df_invoice = pd.DataFrame(df_invoice)
    df_invoice['status'].replace({1: 'Done'}, inplace=True)
    df_invoice['purchase_Date'] = pd.to_datetime(df_invoice['purchase_Date'])
    df_invoice['dayOfWeek'] = df_invoice['purchase_Date'].dt.day_name()
    df_invoice['customer_Name'].replace({"": "Khách lẻ"}, inplace=True)

    df_invoice = df_invoice.query("id != 114200880 and revenue != 0 and status != 'Đã hủy'")
    df_invoice = df_invoice.sort_values(by='purchase_Date', ascending=False) 

    df_goods = pd.DataFrame(df_goods)
    df_goods = df_goods.query("id != 114200880 and revenue != 0")
    df_goods = df_goods.sort_values(by='purchase_Date', ascending=False)

    df_goods.to_csv('goods.csv', index=False)
    df_invoice.to_csv('invoices.csv', index=False)
    return df_invoice, df_goods

def process_customers_data(customers_data):
    df_customer = []
    for customer in customers_data:
        created_date_match = date_time_pattern.search(customer.get("createdDate", ""))
        date, _ = created_date_match.groups() if created_date_match else (None, None)
        
        gender_bool = customer.get("gender", None)
        if gender_bool is True:
            gender = "nam"
        elif gender_bool is False:
            gender = "nữ"
        else:
            gender = "-"  

        customers_data_schema = {
            'id': customer["id"],
            'name': customer.get("name").title(),
            'gender': gender,
            'contact_Number': customer.get("contactNumber"),
            'debt': customer.get("debt"),
            'created_Date': date,
        }

        df_customer.append(customers_data_schema)

    df_customer = pd.DataFrame(df_customer)
    df_customer['contact_Number'] = df_customer['contact_Number'].astype(str).str.replace('.0', '', regex=False).apply(lambda x: '0' + x.lstrip('0')[:3] + '-' + x.lstrip('0')[3:6] + '-' + x.lstrip('0')[6:])
    df_customer = df_customer.sort_values(by='created_Date', ascending=False) 

    df_customer.to_csv('kioviet_customer.csv', index=False)
    return df_customer

def google_sheet_import(df_invoice, df_customer, df_invoice_details):
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json')
    client = gspread.authorize(creds)
    sheet = client.open('72BilliardsClub')

    try:
        set_with_dataframe(sheet.get_worksheet(0), df_invoice, include_column_header=True, resize=False)
        set_with_dataframe(sheet.get_worksheet(1), df_customer, include_column_header=True, resize=True)
        set_with_dataframe(sheet.get_worksheet(2), df_invoice_details, include_column_header=True, resize=True)
        print("\nSuccessfully imported data to Google Sheets ✅\n")
    except gspread.exceptions.APIError as e:
        print(f"Failed to update Google Sheets due to an API error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def run_git_commands():
    try:
        os.chdir('C:\\Users\\Khoi\\Desktop\\BilliardsClub')
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Daily update'], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("Changes pushed to GitHub ✅.")
    except subprocess.CalledProcessError as e:
        print(f"Error in Git operation: {e}")

async def main(pages, page_size):
    page_size = page_size
    pages = pages
    
    all_invoices, all_customers = await fetch_all_data(pages, page_size)
    
    # Flatten the list of lists
    all_invoices = [item for sublist in all_invoices for item in sublist]
    all_customers = [item for sublist in all_customers for item in sublist]

    df_invoice, df_goods = process_invoices_data(all_invoices)
    df_customer = process_customers_data(all_customers)

    print(df_invoice, df_invoice.dtypes)
    print(df_goods, df_goods.dtypes)
    print(df_customer, df_customer.dtypes)

    print(f"{Fore.BLUE}Customers: {len(df_customer)}")
    print(f"{Fore.BLUE}Invoices: {len(df_invoice)}")
    print(f"{Fore.BLUE}Invoice_Detail: {len(df_goods)}")

    google_sheet_import(df_invoice, df_customer, df_goods)
    run_git_commands()

if __name__ == "__main__":
    asyncio.run(main(pages=215, page_size=100))

end_time = time.time()  # Record the end time
runtime = end_time - start_time  # Calculate the runtime

print(f"The script took {runtime} seconds to run.")