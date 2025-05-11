# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_squared_error
import DataPipeline
# %%
np.random.seed(42)

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