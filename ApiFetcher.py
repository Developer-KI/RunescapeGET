import requests
import pandas as pd

headers = {
    'User-Agent': 'Price/Volume Tracker and Scraper- NoHFT',
    'From': 'mstavreff@outlook.com, discord: shrimpsalad'
}

def fetch_latest(item_id: list[int]):
    item_id = list(map(str, item_id))
    item_call = '|'.join([i for i in item_id])
    
    url = f"https://api.weirdgloop.org/exchange/history/osrs/latest?id={item_call}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
    else:
        print("Failed to fetch data. Check the item ID or API status.")

def fetch_5min(timestamp: int = 0):
    if timestamp == 0:
        url = f"https://prices.runescape.wiki/api/v1/osrs/5m"
    else:
        url = f"https://prices.runescape.wiki/api/v1/osrs/5m?timestamp={timestamp}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame.from_dict(data['data'], orient='index')
        return df
    else:
        print("Failed to fetch data. Check the item ID or API status.")

def fetch_historical(item_id: int):
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

        return df
    else:
        print("Failed to fetch data. Check the item ID or API status.")