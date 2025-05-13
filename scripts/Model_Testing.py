# %%
#Script Init
import pandas as pd
import numpy as np
import DataPipeline as pipeline
import ModelTools as tools
import Models.RFTS as my_models

# %%
# Aggregate features for model
price_data = pipeline.data_preprocess(read=False, interp_method='nearest')
price_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="wprice")
vol_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="totalvol")

target_item = 12934

price_items_reg = price_matrix_items[[target_item]].iloc[20:]
vol_items_reg = vol_matrix_items[[target_item]].iloc[20:]
price_items_reg.columns = [f'{target_item}']
vol_items_reg.columns = [f'{target_item}_vol']
volatility_index = pipeline.volatility_market(price_data, smoothing=20)[20:]
reg_data = pd.concat([price_items_reg, vol_items_reg, volatility_index], axis=1)

df_time = tools.target_time_features(reg_data, f'{target_item}', 5)
df_roll = tools.target_rolling_features(reg_data, f'{target_item}', 10)
df = pd.merge(df_time, df_roll, on='timestamp', how='inner').dropna()
# %%
#Run the RFTS model
model, test_idx = my_models.RFTS(data=df, target_col=f'{target_item}', estimators=200)
# %%
#Plot RFTS predictions vs realized price
X = df.drop(f'{target_item}', axis=1)
Y = df[f'{target_item}']

tools.plot_pred_vs_price(Y.iloc[test_idx[:100]], X.iloc[test_idx[:100]], model=model)
# %%
from hmmlearn.hmm import MultinomialHMM
#Paramters for price differences governing regime change
#Window must be >0, 1= no window
window = 300
diffpercent = 1
rolling_mean = price_matrix_items.rolling(window).mean()

shifted_mean = rolling_mean.shift(window)
upper_threshold = shifted_mean * (1 + diffpercent / 100)
lower_threshold = shifted_mean * (1 - diffpercent / 100)

booleanprice = np.select([
    rolling_mean > upper_threshold,
    rolling_mean < lower_threshold
], [2, 0], default=1)

from sklearn.preprocessing import OneHotEncoder



booleandf = pd.DataFrame(booleanprice, columns=price_matrix_items.columns)
X=booleandf[12934].values.reshape(-1,1)
X[0,0]=2

encoder = OneHotEncoder(sparse_output=False, categories='auto')
X_encoded = encoder.fit_transform(X).astype(int)  # Shape will now be (2499, 3)


iter = 100
startprob = [1/6,2/3,1/6] #reasonable to keep up/down always less than sideways
transprob = [[1/6,2/3,1/6],[1/6,2/3,1/6],[1/6,2/3,1/6]]
emissionprob = [[1/6,2/3,1/6],[1/6,2/3,1/6],[1/6,2/3,1/6]]
HMMmodel = MultinomialHMM(n_components=3, n_iter=iter, init_params='') #leave init_params empty to self-select probabilities
HMMmodel.n_features= len(np.unique(booleandf))
HMMmodel.startprob_ = np.array(startprob)
HMMmodel.transmat_ = np.array(transprob)
HMMmodel.emissionprob_ = np.array(emissionprob)
HMMmodel.fit(X_encoded)
hidden_states= HMMmodel.predict(X_encoded)
print(hidden_states)
