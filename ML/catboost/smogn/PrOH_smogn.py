import numpy as np
import math
from catboost import Pool, CatBoostRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
import wandb
from functools import partial

df = pd.read_csv("../../../Data/Datasets/BSc/smogn/smogn_PrOH", index_col=False)

X = df.iloc[:, 1:8]
Y = df.iloc[:, -1:]

sweep_config = {
    "method": "bayes",
    "metric": {"name": "rmse", "goal": "minimize"},
    "parameters": {
        "rsm": {"values": [0.3, 0.4, 0.5, 0.6, 0.8, 1]},
        "learning_rate": {"values": [0.01, 0.05, 0.08, 0.1, 0.3, 0.5]},
        "depth": {"values": [2, 4, 6, 8, 10, 12, 15]},
        "l2_leaf_reg": {"values": [0, 2, 4, 6, 8, 10]},
        "iteration": {"values": [100, 300, 500, 1000, 1200, 1500]},
        "bagging_temperature": {"values": [0, 1, 10]},
    },
}


def train(X, Y):
    current_time = datetime.now().strftime("%m/%d/%Y_%H:%M")

    config_defaults = {
        "rsm": 1,
        "learning_rate": 0.1,
        "depth": 8,
        "l2_leaf_reg": 4,
        "iteration": 1000,
        "bagging_temperature": 1,
    }

    wandb.init(
        project="bsc_project",
        name=current_time,
        save_code=True,
        tags=[
            "model:catboost",
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
    model = CatBoostRegressor(
        rsm=config.rsm,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        iterations=config.iteration,
        bagging_temperature=config.bagging_temperature,
        loss_function="RMSEWithUncertainty",
        verbose=False,
        random_seed=0,
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
        train_pool = Pool(X_train, y_train)

        # Training the model
        model.fit(train_pool)

        # Testing the model
        test_pred = model.predict(X_test).flatten()

        # RMSE
        rmse = np.sqrt(mean_squared_error(y_test, [test_pred[0]]))

        # Storing the result
        test_results.append((test_pred[0], y_test[0], test_pred[1], rmse))

    rmse_mean = np.mean(np.abs([t[3] for t in test_results]))

    print(f"RMSE: {rmse_mean}")
    print(test_results)
    wandb.log({"rmse": rmse_mean})


target_Y = Y.iloc[:, 0]
print(f"Target: {target_Y}")

sweep_id = wandb.sweep(sweep_config, project="bsc_project")

wandb_train_func = partial(train, X, target_Y)

wandb.agent(sweep_id, function=wandb_train_func, count=100)
