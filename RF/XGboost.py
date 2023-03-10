import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def XGboost_regressor_params_validate(x_dataset: np.ndarray, y_dataset: np.ndarray, parameters: list, itr_round:int, fold_num:int):

    dfull = xgb.DMatrix(x_dataset, y_dataset)
    cv_res1 = xgb.cv(parameters, dfull, num_boost_round=itr_round, nfold=fold_num, metrics='rmse')

    fig, ax = plt.subplots(figsize =(15,8), constrained_layout=True)
    ax.plot(range(1, itr_round), cv_res1.iloc[:, 0], c='red', label='train.RMSE.loss_function')
    ax.plot(range(1, itr_round), cv_res1.iloc[:, 2], c='blue', label='test.RMSE.loss_function')
    plt.show()

def XGB(x_dataset: np.ndarray, y_dataset: np.ndarray):

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.3, random_state=None)
    xgb_local = xgb.XGBRegressor(max_depth=8,
                            learning_rate=0.05,
                            n_estimators=100,
                            silent=True,
                            objective='reg:squarederror',
                            nthread=-1,
                            gamma=0,
                            min_child_weight=1,
                            max_delta_step=0,
                            subsample=0.85,
                            colsample_bytree=0.7,
                            colsample_bylevel=1,
                            reg_alpha=0,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            seed=1440,
                            missing=None)

    xgb_local.fit(x_train, y_train, eval_metric='rmse', verbose=True, eval_set=[(x_test, y_test)], early_stopping_rounds=100)
    y_pred = xgb_local.predict(x_test)

if __name__ == '__main__':
    paras = {'silent': True,
             'max_depth': 8,
             'learning_rate': 0.05,
             'n_estimators': 100,
             'objective': 'reg:squarederror',
             'nthread': -1,'gamma': 0,
             'min_child_weight':1,
             'max_delta_step':0,
             'subsample':0.85,
             'colsample_bytree':0.7,
             'colsample_bylevel':1,
             'reg_alpha':0,
             'reg_lambda':1,
             'scale_pos_weight':1
             }