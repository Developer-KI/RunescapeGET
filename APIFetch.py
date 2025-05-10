#%%
import requests
import pandas as pd
from datetime import datetime
import json

headers = {
    'User-Agent': 'Price/Volume Tracker and Scraper- NoHFT',
    'From': 'mstavreff@outlook.com'
}

def fetch_latest(item_id: str):
    url = f"https://api.weirdgloop.org/exchange/history/osrs/latest?id={item_id}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        item = data[item_id]

        
    else:
        print("Failed to fetch data. Check the item ID or API status.")

def fetch_historical(item_id: str):
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id={item_id}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = []
        for item_id, entries in data.items():
            for entry in entries:
                records.append(entry)

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Amsterdam')

        print(df)
    else:
        print("Failed to fetch data. Check the item ID or API status.")

fetch_historical("561")
#%%