# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import APIFetcher as API

# %%
n=100
snapfifty = API.fetch_historical_5m(n)

# %%
#grab ID's with rows with NaN values == TRUE
NAindex=snapfifty[snapfifty.isna().any(axis=1)]['item_id'].unique()
snapfiftyprune = snapfifty[~snapfifty["item_id"].isin(NAindex)]
#removing low volume items
snapfiftyprunegroup= snapfiftyprune.groupby('item_id').nunique()
#n-1 to ensure proper ranging
filtered_indexes = snapfiftyprunegroup[snapfiftyprunegroup['timestamp'] != n-1].index
snapfiftyprune=snapfiftyprune[~snapfiftyprune['item_id'].isin(filtered_indexes)]
#rechecking groupings for n timestamps
snapfiftyprunegroup= snapfiftyprune.groupby('item_id').nunique()
#Weighted average of High/Low Price by High/Low Volume
snapfiftyprune['totalvol']=snapfiftyprune['highPriceVolume']+snapfiftyprune['lowPriceVolume']
snapfiftyprune['wprice']=(snapfiftyprune['highPriceVolume']/snapfiftyprune['totalvol'])*(snapfiftyprune['avgHighPrice']-snapfiftyprune['avgLowPrice'])+snapfiftyprune['avgLowPrice']
#transforming panel data to price and volume matrices
price_matrix_snapfifty=snapfiftyprune.pivot(index="timestamp", columns="item_id", values="wprice")
vol_matrix_snapfifty=snapfiftyprune.pivot(index='timestamp', columns='item_id', values='totalvol')
corr_price_snapfifty=price_matrix_snapfifty.corr()
corr_vol_snapfifty=vol_matrix_snapfifty.corr()
print(corr_price_snapfifty[corr_price_snapfifty > 0.7])


# %%
#volatility
volatilityitems = price_matrix_snapfifty.rolling(window=4).std()
volatilitymarket= volatilityitems.sum(axis=1)
#Dividing by shape as count of row/column length
volatilitymarket=volatilitymarket/price_matrix_snapfifty.shape[1]
volatilitymarket
print(snapfiftyprune['totalvol'].shape)

# %%
#440 1605 item id

features= vol_matrix_snapfifty['1605'].iloc[0:98]
features=pd.concat([features, price_matrix_snapfifty["1605"]], axis=1).iloc[0:98]
features=pd.concat([features, vol_matrix_snapfifty["440"]], axis=1).iloc[0:98]
features=pd.concat([features, price_matrix_snapfifty["440"]], axis=1).iloc[0:98]
features.columns=['ID 1605 Volume','ID 1605 Price','ID 440 Volume','ID 440 Price']
target=price_matrix_snapfifty['440'].iloc[1:100]

# %%
#440 1605 item id


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
print(y_pred)

y_pred = rf_model.predict(X_test)  # Predicted values
df_pred = pd.DataFrame({"timestamp": df_actual["timestamp"], "predicted_price": y_pred})


