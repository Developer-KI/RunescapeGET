# %%
#Script Innit
import pandas as pd
import matplotlib.pyplot as plt
import DataPipeline as pipeline
import ModelTools as tools

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

start = 20
price_volatility = pipeline.volatility_market(price_data, smoothing=start)

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
tools.plot_historical_alch_vs_price(12934)
tools.plot_recent_alch_vs_price(12934)
# %%