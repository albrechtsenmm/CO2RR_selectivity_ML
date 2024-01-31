import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
import wandb
import random
from datetime import datetime
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn import linear_model


df = pd.read_csv("../../../Data/Datasets/BSc/data", index_col=False)

X = df.iloc[:, 4:-7]
Y = df.iloc[:, -7:]


current_time = datetime.now().strftime("%m/%d/%Y_%H:%M")

for i in range(len(Y.columns)):
    target_y = Y.iloc[:, i]

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="bsc_project",
        name=current_time,
        save_code=True,
        tags=[
            "model:mlr",
            "train:data_ex_Ni_Tl_Ti_Ir_Rh_Ru_Hg_Pd_COOH-Incl_Vnhe",
            "test:data_ex_Ni_Tl_Ti_Ir_Rh_Ru_Hg_Pd_COOH-Incl_Vnhe",
            f"target:{target_y.name}",
            "CV:LOOCV",
            "Scaling:None",
            "Subsample:None",
        ],
    )

    # build multiple linear regression model
    model = linear_model.LinearRegression()

    test_results = []
    n_samples = len(X)

    for i in range(n_samples):
        # Splitting the data into training and test sets
        X_train = pd.concat([X.iloc[:i], X.iloc[i + 1 :]])
        y_train = pd.concat([target_y.iloc[:i], target_y.iloc[i + 1 :]])
        X_test = X.iloc[i : i + 1]
        y_test = target_y.iloc[i : i + 1].values
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
        test_results.append((df["metal_facet"][i], test_pred[0], y_test[0], rmse))

    rmse_mean = np.mean(np.abs([t[3] for t in test_results]))

    print(f"RMSE: {rmse_mean}")
    print(test_results)
    wandb.log({"rmse": rmse_mean})

    wandb.finish()
