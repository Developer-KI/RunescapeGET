import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def RFTS(data: pd.DataFrame, feature_col: str, splits: int = 5, estimators: int = 100, seed: int = 42) -> RandomForestRegressor:
    X = data.drop(feature_col, axis=1)
    Y = data[feature_col]

    tscv = TimeSeriesSplit(n_splits=splits)
    cv_mses = []
    output_model = None

    for train_idx, test_idx in tscv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = Y.iloc[train_idx], Y.iloc[test_idx]
    
        model_cv = RandomForestRegressor(n_estimators=estimators, max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=seed, n_jobs=-1)
        model_cv.fit(X_train_cv, y_train_cv)
    
        preds_cv = model_cv.predict(X_test_cv)

        model_mse = mean_squared_error(y_test_cv, preds_cv)
        
        if model_mse < min(cv_mses, default=0):
            output_model = model_cv

        cv_mses.append(model_mse)
    
    if output_model != None:
        print(f"Cross-validated RMSE: {np.sqrt(np.mean(cv_mses)):.4f} (±{np.sqrt(np.std(cv_mses)):.4f})")
        return output_model
    else:
        raise Exception("Failed to choose model")