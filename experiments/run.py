
# Limit numpy"s number of threads
import os

# Base imports
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Code imports
import sys
sys.path.insert(0, "../skmaple")
from skmaple import MAPLE
from Misc import load_normalize_data

np.random.seed(42)

datasets = ["autompgs","happiness", "winequality-red", "housing", "day", "music", "crimes", "communities"]
for dataset in datasets:   
    # Output
    out = {}

    lr_rmses = []
    rf_rmses = []
    maple_rmses = []
    for _ in range(1):

        # Load Data
        X_train, y_train, X_val, y_val, X_test, y_test, train_mean, train_stddev = load_normalize_data("Data/" + dataset + ".csv")

        # Linear model
        lr = Ridge(alpha=0.001)
        lr.fit(X_train, y_train)
        predictions = lr.predict(X_test)
        lr_rmses.append(np.sqrt(mean_squared_error(y_test, predictions)))

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, max_features=0.5, min_samples_leaf=10)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        rf_rmses.append(np.sqrt(mean_squared_error(y_test, predictions)))

        # MAPLE
        maple_rf = MAPLE(rf, lr)
        maple_rf.fit(X_train, y_train)
        predictions = maple_rf.predict(X_test)
        maple_rmses.append(np.sqrt(mean_squared_error(y_test, predictions)))
    
    out["lm_rmse"] = (np.mean(lr_rmses), np.std(lr_rmses))
    out["rf_rmse"] = (np.mean(rf_rmses), np.std(rf_rmses))
    out["maple_rmse"] = (np.mean(maple_rmses), np.std(maple_rmses))

    print(dataset, out)