# %%
import pandas as pd
import matplotlib.pyplot as plt
import APIFetcher as API
import json

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
raw_pricedata = pd.read_csv('../data/data.csv', names=['item_id', 'avgHighPrice', 'highPriceVolume', 'avgLowPrice', 'lowPriceVolume', 'timestamp'])

with open("../data/data_properties.txt", "r") as file:
            lines = file.readlines()
series_length = int(lines[1].replace("\n", ""))
group_raw_pricedata = raw_pricedata.groupby('item_id').nunique()

with open("../data/namealchemy.json", "r") as file:
    data = json.load(file)

# Convert dictionary to DataFrame
high_alchemy = pd.DataFrame(list(data.items()), columns=["Item", "Price"])

with open("../data/nameID.json", "r") as file:
    data = json.load(file)

# Convert dictionary to DataFrame
nameID = pd.DataFrame(list(data.items()), columns=["Item", "ID"])

reference = pd.merge(nameID, high_alchemy, on="Item", how="inner")
reference= reference.drop([0,1])

reference.set_index('ID', inplace=True)  # Use the 'ID' column as the row index
print(reference.loc[561,'Item']) #Nature Rune

# %%
#n-1 to ensure proper time series ranging
filtered_indexes = group_raw_pricedata[group_raw_pricedata['timestamp'] != series_length].index
raw_pricedata = raw_pricedata[~raw_pricedata['item_id'].isin(filtered_indexes)]
# interpolate missing values
raw_pricedata = raw_pricedata.interpolate()
#Weighted average of High/Low Price by High/Low Volume
raw_pricedata['totalvol'] = raw_pricedata['highPriceVolume'] + raw_pricedata['lowPriceVolume']
raw_pricedata['wprice'] = (raw_pricedata['highPriceVolume']/raw_pricedata['totalvol']) * (raw_pricedata['avgHighPrice'] - raw_pricedata['avgLowPrice']) + raw_pricedata['avgLowPrice']
#transforming panel data to price and volume matrices
price_matrix_items = raw_pricedata.pivot(index="timestamp", columns="item_id", values="wprice")
volume_matrix_items = raw_pricedata.pivot(index='timestamp', columns='item_id', values='totalvol')
corr_price_items = price_matrix_items.corr()
corr_volume_items = volume_matrix_items.corr()

# %%
#volatility smoothing
volativity_sensitivity = 30
volatilityitems = price_matrix_items.rolling(window=volativity_sensitivity).std()
#Aggregate volatility
volatilitymarket = volatilityitems.sum(axis=1)
#Dividing by shape as count of row/column length
volatilitymarket = volatilitymarket/corr_price_items.shape[1]

# %%
plt.figure(figsize=(10, 5))
plt.plot(pd.to_datetime(volatilitymarket.iloc[volativity_sensitivity:].index, unit='s'),volatilitymarket.iloc[volativity_sensitivity:], marker="o", markersize='2', linestyle="-", label="Standard Deviation")

plt.xlabel("Time")
plt.ylabel("Standard Deviation (SD)")
plt.title("OSRS Market Volatility")
plt.legend()
plt.xticks(rotation=45)
plt.grid()

plt.show()

# %%
graphID = 45
if graphID in reference.index:
    nature = API.fetch_historical(graphID)
    plt.figure(figsize=(10, 5))
    plt.plot(nature['timestamp'], nature['price'], marker="o", markersize='2', linestyle="-", label=f"{reference.loc[graphID,'Item']} Price")
    plt.axhline(y=reference.loc[graphID,'Price'], color='cyan', linestyle='-', label='High Alchemy Price')

    plt.xlabel("Time")
    plt.ylabel("Price (GP)")
    plt.title("OSRS Grand Exchange Price Trend")
    plt.legend()
    plt.xticks(rotation=45)  # Rotate timestamps for clarity
    plt.grid()

    plt.show()
else: print('Invalid ID')
# %%


import numpy as np
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_squared_error
# %%
np.random.seed(42)
date_range = pd.date_range(start="2020-01-01", periods=100, freq="D")  # 100 days
dfa = pd.DataFrame({
    "date": date_range,
    "value": np.random.randn(100) * 10 + 50  # Simulated time series values
})
print(dfa)
dfa['shift1']=dfa['value'].shift(1)
dfa['shift2']=dfa['value'].shift(2)
print(dfa)
dfa.dropna(inplace=True)
print(dfa)


import statsmodels as sm

#440 1605 item id
def TSRandomForest(regressors_main: pd.DataFrame, regressors_external: pd.DataFrame, window: int, n_splits, target_col, merge_on='timestamp', lag_features=[1,2],n_tree_estimators=100):

    for ext_df in regressors_external:
        regressors_main = regressors_main.merge(ext_df, on=merge_on, how='left') 
    for lag in lag_features:
         regressors_main[f'lag_{lag}'] = regressors_main[target_col].shift(lag)
    
    regressors_main=regressors_main.dropna(inplace=True)

    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=window) 
    mae_scores = []
    mse_scores = []
    rmse_scores = []
    aic_scores = []
    bic_scores = []
    
    for train_index, test_index in tscv.split(regressors_main):
        train_df, test_df = regressors_main.iloc[train_index], regressors_main.iloc[test_index]

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=n_tree_estimators, random_state=123)
        rf_model.fit(X_train, y_train)
        # Predict on test data
        y_pred = rf_model.predict(X_test)

        # Evaluate model performance
        mae = mean_absolute_error(y_test, y_pred)
        mae_scores.append(mae)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        rmse = root_mean_squared_error(y_test, y_pred)
        rmse_scores.append(rmse)


        print(f"Split {len(mae_scores)} - MAE: {mae:.2f}")
        print(f"Split {len(mae_scores)} - MAE: {mse:.2f}")
        print(f"Split {len(mae_scores)} - MAE: {rmse:.2f}")

    print(f"\nAverage MAE across splits: {sum(mae_scores) / len(mae_scores):.2f}")
    print(f"\nAverage MSE across splits: {sum(mse_scores) / len(mse_scores):.2f}")
    print(f"\nAverage RMSE across splits: {sum(rmse_scores) / len(rmse_scores):.2f}")
    return rf_model
