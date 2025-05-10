import requests
import pandas as pd
from datetime import datetime
import time

headers = {
    'User-Agent': 'Price/Volume Tracker and Scraper- NoHFT',
    'From': 'mstavreff@outlook.com, discord: shrimpsalad'
}

def fetch_latest_deprecated(item_ids: list[int]) -> pd.DataFrame:
    item_ids = list(map(str, item_ids))
    item_call = '|'.join([i for i in item_ids])
    
    url = f"https://api.weirdgloop.org/exchange/history/osrs/latest?id={item_call}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        records = list(data.values())
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_latest() -> pd.DataFrame:
    url = f"https://prices.runescape.wiki/api/v1/osrs/latest"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame.from_dict(data['data'], orient='index')
        df['highTime'] = pd.to_datetime(df['highTime'], unit='s')
        df['lowTime'] = pd.to_datetime(df['lowTime'], unit='s')
        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_5min(timestamp: int = 0) -> pd.DataFrame:
    if timestamp == 0:
        url = f"https://prices.runescape.wiki/api/v1/osrs/5m"
    else:
        url = f"https://prices.runescape.wiki/api/v1/osrs/5m?timestamp={timestamp}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame.from_dict(data['data'], orient='index')
        df = df.reset_index()
        df['timestamp'] = timestamp
        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_historical(item_id: int) -> pd.DataFrame:
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
        raise Exception("Failed to fetch data")

def fetch_historical_5m(n = 10, mins=5, waits=1.1) -> pd.DataFrame:
    unix_timestamp_seconds = int(datetime.now().timestamp())
    unix_timestamp_seconds = unix_timestamp_seconds - unix_timestamp_seconds % 300
    df = pd.DataFrame(columns=['index', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp'])

    for t in range(0, n):
        df_t = fetch_5min(unix_timestamp_seconds - (mins * 60) * t)
        df = pd.concat([df, df_t], ignore_index=True)
        time.sleep(waits)

    return df

if __name__ == "__main__":
    print(fetch_historical_5m())
