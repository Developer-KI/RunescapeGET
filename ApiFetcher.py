from numpy import void
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
        df = df.reset_index()
        df = df.rename(columns={"index": "item_id"})
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
        df["item_id"] = item_id
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
        df = df.reset_index()
        df = df.rename(columns={"index": "item_id"})
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
        df = df.rename(columns={"index": "item_id"})
        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_historical_5m(n = 10, mins=5, waits=1.1, timestamp: int = 0) -> pd.DataFrame:
    if timestamp != 0:
        unix_timestamp_seconds = timestamp
    else:
        unix_timestamp_seconds = int(datetime.now().timestamp())

    unix_timestamp_seconds = unix_timestamp_seconds - unix_timestamp_seconds % 300
    df = fetch_5min(unix_timestamp_seconds)

    for t in range(1, n):
        df_t = fetch_5min(unix_timestamp_seconds - (mins * 60) * t)
        df = pd.concat([df, df_t], ignore_index=True)
        time.sleep(waits)

    return df[['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp']]

def writing_returns(filepath: str = "./data.csv", n: int = 100, p: int= 10) -> void:
    timestampt_start = int(datetime.now().timestamp())
    timestampt_start = timestampt_start - timestampt_start % 300
    series_lenght = 0

    with open("./data_properties.txt", "r") as file:
            lines = file.readlines()
    if lines != list():
        timestampt_start = int(lines[0].replace("\n", ""))
        series_lenght = int(lines[1].replace("\n", ""))

    print(f"Initialized process. Expected mining time: {round(n * p * 1.1 / 60, 3)} minutes")
    for t in range(1, p):
        df_t = fetch_historical_5m(n = n, timestamp=timestampt_start - ((t * n) * 300))
        last_call_timestamp = df_t.at[df_t.index[-1], 'timestamp']
        df_t = df_t[['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp']]
        df_t.to_csv(filepath, mode='a', header=False, index=False)
        series_lenght = series_lenght + n
        with open("./data_properties.txt", "w") as file:
            file.write(f"{last_call_timestamp}\n")
            file.write(f"{series_lenght - 1}\n")
        print(f"{(t + 1) * n} queries added!")
    print("Success!")

if __name__ == "__main__":
    writing_returns(n=100, p=6)

def fetch_historical_common_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Common%20Trade%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Common Trade Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_historical_food_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Food%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Food Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")
    
def fetch_historical_herb_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Herb%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Herb Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")

def fetch_historical_log_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Log%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Log Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")
        
def fetch_historical_metal_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Metal%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Metal Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")
    
def fetch_historical_rune_index() -> pd.DataFrame:
    url = f"https://api.weirdgloop.org/exchange/history/osrs/all?id=GE%20Rune%20Index"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        records = data["GE Rune Index"]
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df
    else:
        raise Exception("Failed to fetch data")
    
def fetch_latest_idex_df():
    index_calls = {"Common%20Trade": "GE Common Trade Index", "Food": "GE Food Index", "Herb": "GE Herb Index", "Log": "GE Log Index", "Metal": "GE Metal Index", "Rune": "GE Rune Index"}
    df = pd.DataFrame(columns=[])

    for index in index_calls.keys():
        url = f"https://api.weirdgloop.org/exchange/history/osrs/latest?id=GE%20{index}%20Index"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            df_t = pd.DataFrame.from_dict(data[index_calls[index]], orient='index').T
            df = pd.concat([df, df_t], axis=0)
        else:
            raise Exception("Failed to fetch data")
        
        time.sleep(1.1)
            
    df = df.reset_index()
    del df['index']
    return df
