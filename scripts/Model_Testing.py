# %%
#Script Innit
import pandas as pd
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

priceboolean = pd.DataFrame(columns=[''])
priceboolean.loc[0]=1
priceboolean.loc[0]=priceboolean.loc[0].fillna(1)
for col_name, col_values in price_matrix_items.items():
    for j in range(1,price_data.shape[0]+1):
        boolean_col = pd.Series()
        if col_values.iloc[j]>col_values.iloc[j-1]:
            boolean_col.loc[len(priceboolean)] = 1
        else:
            boolean_col.loc[len(priceboolean)] = 0
        priceboolean[col_name]=boolean_col
