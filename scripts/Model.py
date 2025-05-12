# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_squared_error
import DataPipeline as pipeline
np.random.seed(42)
# %%
price_data = pipeline.data_preprocess(read=True)
reference = pipeline.alchemy_preprocess(read=True)

price_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="wprice")
volume_matrix_items = price_data.pivot(index='timestamp', columns='item_id', values='totalvol')
price_items_reg = price_matrix_items[[219, 12934]]
price_items_reg.columns = ['219', '12934']

#%%
#219 12934 item id RF
def TSRandomForest(regressors_main: pd.DataFrame, regressors_external: list[pd.DataFrame], target_col: int, lag_features: list[int] = [1,2], window: int = 50, n_splits: int = 50, merge_on: str ='timestamp', n_tree_estimators: int =100):

    for ext_df in regressors_external:
        regressors_main = regressors_main.merge(ext_df, on=merge_on, how='left')
    for lag in lag_features:
         regressors_main[f'lag_{lag}'] = regressors_main[[f'{target_col}']].shift(lag)
    
    regressors_main.dropna(inplace=True)

    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=window) 
    mae_scores = []
    mse_scores = []
    rmse_scores = []
    aic_scores = []
    bic_scores = []
    
    for train_index, test_index in tscv.split(regressors_main):
        train_df, test_df = regressors_main.iloc[train_index], regressors_main.iloc[test_index]

        X_train = train_df.drop(columns=[f'{target_col}'])
        y_train = train_df[f'{test_index}']
        X_test = test_df.drop(columns=[f'{target_col}'])
        y_test = test_df[f'{test_index}']

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
        rmse = np.sqrt(mse)
        rmse_scores.append(rmse)

        rss = np.sum((y_test - y_pred) ** 2)

        # Number of parameters (including intercept-current price)
        k = regressors_main.shape[1] - 1 + 1  
        n = len(y_train)  # Number of observations

        # Compute AIC and BIC
        aic = 2 * k + n * np.log(rss / n)
        bic = k * np.log(n) + n * np.log(rss / n)

        aic_scores.append(aic)
        bic_scores.append(bic)

        print(f"Split {len(mae_scores)} - AIC: {aic:.2f}")
        print(f"Split {len(mae_scores)} - BIC: {bic:.2f}")
        print(f"Split {len(mae_scores)} - MAE: {mae:.2f}")
        print(f"Split {len(mae_scores)} - MAE: {mse:.2f}")
        print(f"Split {len(mae_scores)} - MAE: {rmse:.2f}")

    print(f"\nAverage MAE across splits: {sum(mae_scores) / len(mae_scores):.2f}")
    print(f"\nAverage MSE across splits: {sum(mse_scores) / len(mse_scores):.2f}")
    print(f"\nAverage RMSE across splits: {sum(rmse_scores) / len(rmse_scores):.2f}")
    return rf_model
# %%
