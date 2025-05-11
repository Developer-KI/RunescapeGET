# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import DataPipeline as pipeline

# %%
def target_time_features(y: pd.DataFrame, feature_col: str, time_feature: int = 2):
    data = y.copy()
    for t in range(1, time_feature + 1):
        data[f'lag{t}'] = data.loc[:,feature_col].shift(t)
    return data

def target_rolling_features(y: pd.DataFrame, feature_col: str, window: int = 2):
    data = y.copy()
    data['rolling_mean'] = data[feature_col].rolling(window).mean()
    data['rolling_std'] = data[feature_col].rolling(window).std()
    return data[['rolling_mean', 'rolling_std']]

def RFTS(data: pd.DataFrame, feature_col: str,lag: int = 2, window: int = 0, seed: int = 42) -> RandomForestRegressor:
    df_time = target_time_features(data, feature_col, lag)
    if window > 0:
        df_roll = target_rolling_features(data, feature_col, window)
        df = pd.merge(df_time, df_roll, on='timestamp', how='inner').dropna()
    else:
        df = df_time

    print(df)

    X = df.drop(feature_col, axis=1)
    Y = df[feature_col]

    tscv = TimeSeriesSplit(n_splits=5)
    cv_mses = []
    cv_meas = []
    output_model = None

    for train_idx, test_idx in tscv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = Y.iloc[train_idx], Y.iloc[test_idx]
    
        model_cv = RandomForestRegressor(n_estimators=100, random_state=seed)
        model_cv.fit(X_train_cv, y_train_cv)
    
        preds_cv = model_cv.predict(X_test_cv)
        model_mse = mean_squared_error(y_test_cv, preds_cv)
        model_mea = mean_absolute_error(y_test_cv, preds_cv)
        
        if model_mse < min(cv_mses, default=0):
            output_model = model_cv

        cv_mses.append(model_mse)
        cv_meas.append(model_mea)
    
    if output_model != None:
        print(f"Cross-validated MSE: {np.mean(cv_mses):.4f} (±{np.std(cv_mses):.4f})")
        print(f"Lowest MSE of output model: {min(cv_mses, default=-1)}")
        return output_model
    else:
        raise Exception("Failed to choose model")

# %%
price_data = pipeline.data_preprocess(read=True)
price_matrix_items = price_data.pivot(index="timestamp", columns="item_id", values="wprice")

price_items_reg = price_matrix_items[[219, 12934]]
price_items_reg.columns = ['219', '12934']

model = RFTS(price_items_reg, '219', 3, 10)
# %%
