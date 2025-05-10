#%%
import requests
url = 'SOME URL'

headers = {
    'User-Agent': 'Price/Volume Tracker and Scraper- NoHFT',
    'From': 'mstavreff@outlook.com'  # This is another valid field
}

def fetch(item_id):
    url = f"https://services.runescape.com/m=itemdb_oldschool/api/catalogue/detail.json?item={item_id}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        item = data["item"]
        print(f"Item: {item['name']}")
        print(f"Current Price: {item['current']['price']}")
        print(f"Today's Change: {item['today']['price']}")
        print(f"30-Day Change: {item['day30']['trend']}")
        print(f"90-Day Change: {item['day90']['trend']}")
        print(f"180-Day Change: {item['day180']['trend']}")
    else:
        print("Failed to fetch data. Check the item ID or API status.")

# Example usage
item_id = 561  # Nature Rune
fetch(item_id)

#%%