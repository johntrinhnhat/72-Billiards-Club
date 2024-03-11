import requests

access_token_url = 'https://id.kiotviet.vn/connect/token'
access_token_request = {
    'scopes': 'PublicApi.Access', # Phạm vi truy cập (Public API)
    'grant_type': 'client_credentials', 
    'client_id': '440897d4-4cb8-4f20-ab67-15a923cb008f', 
    'client_secret': '7BF8CD52E498A610E80C72A550EBCED1F70C3FCF', 
    "Content-Type":"application/x-www-form-urlencoded",
}

access_token_response = requests.post(access_token_url, access_token_request)
response_data = access_token_response.json()
access_token = response_data["access_token"]
print(access_token)
# data = response_data['data']