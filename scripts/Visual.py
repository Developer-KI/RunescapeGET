# %%
#Script Innit
import pandas as pd
import matplotlib.pyplot as plt
import APIFetcher as fetcher
import DataPipeline as pipeline

plt.rcParams.update({
    'axes.facecolor': '#2E2E2E',
    'axes.titlecolor': 'white',
    'figure.facecolor': '#1E1E1E',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#444444',
    'axes.edgecolor': 'white'
})

# %%
#Read/Write processed data
price_data = pipeline.data_preprocess(read=True)
reference = pipeline.alchemy_preprocess(read=True)

# %%
#Price and Volume Time series
price_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="wprice")
volume_matrix_items = price_data.pivot(index='timestamp', columns='item_id', values='totalvol')

corr_price_items = price_matrix_items.corr()
corr_volume_items = volume_matrix_items.corr()

price_volatility, start = pipeline.volatility_market(price_data)

# %%
#Volatility Plot
plt.figure(figsize=(10, 5))
plt.plot(pd.to_datetime(price_volatility.iloc[start:].index, unit='s'), price_volatility.iloc[start:], marker="o", markersize='2', linestyle="-", label="Standard Deviation")

plt.xlabel("Time")
plt.ylabel("Standard Deviation (SD)")
plt.title("OSRS Market Volatility")
plt.legend()
plt.xticks(rotation=45)
plt.grid()

plt.show()

# %%
#Item Price vs Alchemy Price Plot
#219, 12934, 
graphID = 12934
if graphID in reference.index:
    nature = fetcher.fetch_historical(graphID)
    plt.figure(figsize=(10, 5))
    plt.plot(nature['timestamp'], nature['price'], marker="o", markersize='2', linestyle="-", label=f"{reference.loc[graphID,'item']} Price")
    plt.axhline(y=reference.loc[graphID, 'price'], color='cyan', linestyle='-', label='High Alchemy Price')

    plt.xlabel("Time")
    plt.ylabel("Price (GP)")
    plt.title("OSRS Grand Exchange Price Trend")
    plt.legend()
    plt.xticks(rotation=45)  # Rotate timestamps for clarity
    plt.grid()

    plt.show()
else: print('Invalid ID')
# %%