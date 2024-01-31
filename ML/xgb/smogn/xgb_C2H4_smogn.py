import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import wandb
from datetime import datetime
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from numpy import absolute
import wandb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from functools import partial

sweep_config = {
    "method": "bayes",
    "metric": {"name": "rmse", "goal": "minimize"},
    "parameters": {
        "colsample_bytree": {"values": [0.3, 0.4, 0.5, 0.6, 0.8, 1]},
        "learning_rate": {"values": [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        "max_depth": {"values": [2, 4, 6, 8, 10, 12, 15]},
        "alpha": {"values": [0, 2, 4, 6, 8, 10]},
        "n_estimators": {"values": [100, 300, 500, 1000, 1200, 1500]},
        "subsample": {"values": [0.1, 0.2, 0.3, 0.5, 0.7, 1]},
        "num_parallel_tree": {"values": [1, 10, 30, 50, 70, 100, 125, 150]},
    },
}


def train(X, Y):
    current_time = datetime.now().strftime("%m/%d/%Y_%H:%M")

    config_defaults = {
        "booster": "gbtree",
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 1,
        "seed": 117,
        "test_size": 0.33,
    }

    wandb.init(
        project="bsc_project",
        name=current_time,
        save_code=True,
        tags=[
            "model:xgb_rf_r",
            "train:data_ex_Ni_Tl_Ti_Ir_Rh_Ru_Hg_Pd_COOH-Incl_Vnhe",
            "test:data_ex_Ni_Tl_Ti_Ir_Rh_Ru_Hg_Pd_COOH-Incl_Vnhe",
            f"target:{Y.name}",
            "CV:LOOCV",
            "Scaling:None",
            "Subsample:smogn",
        ],
        config=config_defaults,
    )  # defaults are over-ridden during the sweep
    config = wandb.config

    # fit model on train
    model = XGBRegressor(
        booster="gbtree",
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        alpha=config.alpha,
        n_estimators=config.n_estimators,
        num_parallel_tree=config.num_parallel_tree,
    )

    test_results = []
    n_samples = len(X)

    for i in range(n_samples):
        # Splitting the data into training and test sets
        X_train = pd.concat([X.iloc[:i], X.iloc[i + 1 :]])
        y_train = pd.concat([Y.iloc[:i], Y.iloc[i + 1 :]])
        X_test = X.iloc[i : i + 1]
        y_test = Y.iloc[i : i + 1].values
        # print(f'X_train: {X_train.shape}')
        # print(f'y_train: {y_train.shape}')
        # print(f'X_test: {X_test.shape}')
        # print(f'y_test: {y_test.shape}')
        # print(f'X_train: {X_train}')
        # print(f'y_train: {y_train}')
        # print(f'X_test: {X_test}')
        # print(f'y_test: {y_test}')
        # print(f'y_test: {type(y_test)}')

        # Training the model
        model.fit(X_train, y_train)

        # Testing the model
        test_pred = model.predict(X_test).flatten()

        # RMSE
        rmse = np.sqrt(mean_squared_error(y_test, [test_pred[0]]))

        # Storing the result
        test_results.append((test_pred[0], y_test[0], rmse))

    rmse_mean = np.mean(np.abs([t[2] for t in test_results]))

    print(f"RMSE: {rmse_mean}")
    print(test_results)
    wandb.log({"rmse": rmse_mean})


df = pd.read_csv("../../../Data/Datasets/BSc/smogn/smogn_C2H4", index_col=False)

X = df.iloc[:, 1:8]
Y = df.iloc[:, -1:]


target_Y = Y.iloc[:, 0]
print(f"Target: {target_Y}")

sweep_id = wandb.sweep(sweep_config, project="bsc_project")

wandb_train_func = partial(train, X, target_Y)

wandb.agent(sweep_id, function=wandb_train_func, count=100)
