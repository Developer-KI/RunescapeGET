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
import hmmlearn as hmm
#Paramter for price differences governing regime change
window = 1
diffpercent = 0.2
rolling_mean = price_matrix_items.rolling(window).mean()

shifted_mean = rolling_mean.shift(window)
upper_threshold = shifted_mean * (1 + diffpercent / 100)
lower_threshold = shifted_mean * (1 - diffpercent / 100)

booleanprice = np.select([
    rolling_mean > upper_threshold,
    rolling_mean < lower_threshold
], [1, -1], default=0)

booleandf = pd.DataFrame(booleanprice, columns=price_matrix_items.columns)


# iter = 100
# startprob = [1/3,1/3,1/3] #reasonable to keep equal
# transprob = [[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3]]
# emissionprob = [[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3]]
# model = hmm.MultinomialHMM(n_components=3, n_iter=iter)
# model.startprob_ = np.array([])
# model.transmat_ = np.array([[], [], []])
# model.emissionprob_ = np.array([[], [], []])