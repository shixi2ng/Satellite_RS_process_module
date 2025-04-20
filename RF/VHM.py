###################################################################################################################################
# This program is built to construct a vegetation height model using GEDI and Sentinel-2 RS data
# Three sections are involved:
# (1) The GEDI-derived vegetation height and Sentinel-2-derived indices from 2019 to 2022 were combined to train a XGB or RFR model,
#      while four-folds cross validation was implement to evaluate the accuracy of the VHM
# (2) The GEDI-derived vegetation height and Sentinel-2-derived indices from all four years were used to train the ultimate XGB or RFR model
# (3) The well-tuned model was then used to obtain the annual peak vegetation height for floodplains in the MYR of year 2021 and 2022
#
# Copyright R Coded by Xi SHI
####################################################################################################################################
import copy
import os
import RF
import matplotlib.pyplot as plt
import numpy as np
from GEDI_toolbox.GEDI_main import GEDI_df
from RF.XGboost import *
import cupy as cp
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import traceback
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import shap
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest
from datetime import datetime


# 自定义评估函数 (Weighted MAE)
# def weighted_mae_metric(y_true, y_pred):
#     residual = np.abs(y_true - y_pred)
#     weights = np.ones_like(y_true)
#     weights[y_true < 3] = 1000000000000
#     weights[(y_true >= 3) & (y_true <= 4)] = 100
#     weights[y_true > 4] = 1
#     weighted_mae_value = np.sum(weights * residual) / np.sum(weights)
#     return  weighted_mae_value

def weighted_mae_obj(y_pred, dtrain):
    y_true = dtrain.get_label()
    residual = y_pred - y_true
    weights = 1 / (1 + np.exp((3.5 - y_true) * 1.7))
    grad = weights * np.sign(residual)
    hess = np.ones_like(y_true)
    return grad, hess

def weighted_mae_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    residual = np.abs(y_pred - y_true)
    weights = 1 / (1 + np.exp((3.5 - y_true) * 1.5 ))
    weighted_mae = np.sqrt(np.sum(weights * (residual ** 2)) / np.sum(weights))
    return 'weighted_mae', weighted_mae

def sort_by_average_position(lists):
    # Dictionary to track positions of each element
    positions = {}

    # Record position of each element in each list
    for list_idx, lst in enumerate(lists):
        for pos, element in enumerate(lst):
            if element not in positions:
                positions[element] = []
            # Extend positions list if needed (handles any number of lists)
            while len(positions[element]) <= list_idx:
                positions[element].append(0)
            positions[element][list_idx] = pos

    # Calculate average position for each element
    avg_positions = {elem: sum(pos_list) / len(pos_list) for elem, pos_list in positions.items()}

    # Sort elements by their average position
    sorted_elements = sorted(avg_positions.keys(), key=lambda x: avg_positions[x])

    return sorted_elements


def XGBoost(xy_dataset, output_folder, mode, max_d_, min_child_, gamma_, subs_, lamda_, alpha_, x_feature, y_feature, learning_ratio, data_index, print_info=True):

    cmap = plt.cm.YlOrBr
    try:
        # Generate the result dictionary
        if os.path.isdir(output_folder):
            res_csv_folder = os.path.join(output_folder, 'res_csv\\')
            res_plot_folder = os.path.join(output_folder, 'res_plot\\')
            train_plot_folder = os.path.join(output_folder, 'train_plot\\')
            bf.create_folder(res_csv_folder)
            bf.create_folder(res_plot_folder)
            bf.create_folder(train_plot_folder)
        else:
            raise Exception('The output folder does not exist')
        n_est = int(50/learning_ratio)

        # OUT NAME
        ex_filename = os.path.join(res_csv_folder, f'res_mode{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lamda_)}_alpha{str(alpha_)}.csv')

        # list for print
        original_train_y_list, predict_train_y_list = [], []
        original_test_y_list, predict_test_y_list = [], []
        train_index_list, test_index_list = [], []

        start_time = time.time()
        if print_info or not os.path.exists(ex_filename):

            x_train = xy_dataset[f'x_train_mode{str(mode)}']
            y_train = xy_dataset[f'y_train_mode{str(mode)}']
            x_test = xy_dataset[f'x_test_mode{str(mode)}']
            y_test = xy_dataset[f'y_test_mode{str(mode)}']
            train_index = data_index[f'train_mode{str(mode)}']
            test_index = data_index[f'test_mode{str(mode)}']

            measure_dic = {'train_score': [], 'test_score': [], 'train_mae': [], 'test_mae': [],
                           'train_mse': [], 'test_mse': [], 'train_rmse': [], 'test_rmse': [],
                           'train_r2': [], 'test_r2': [], 'OE': [], 'lr': [], 'rank': []}

            custom_f = True
            if isinstance(x_train, list):

                # Kfold
                fold_ = 0
                for x_train_, y_train_, x_test_, y_test_, train_index_, test_index_ in zip(x_train, y_train, x_test, y_test, train_index, test_index):

                    #
                    y_train_ori = copy.deepcopy(y_train_)
                    y_test_ori = copy.deepcopy(y_test_)

                    if custom_f is False:
                        # # normalise y
                        # pt = PowerTransformer(method = 'yeo-johnson', standardize=True)
                        # test_size = y_test_.shape[0]
                        # arr = np.concatenate((y_test_, y_train_)).reshape(-1, 1)
                        # arr = np.log((arr - 2) / (5.5 - arr))
                        # arr = pt.fit_transform(arr)
                        # # plt.hist(arr)
                        # # plt.show()
                        # y_test_ = arr[:test_size].flatten()
                        # y_train_ = arr[test_size:].flatten()

                        y_train_ = cp.array(y_train_)
                        y_test_ = cp.array(y_test_)
                        x_train_ = cp.array(x_train_)
                        x_test_ = cp.array(x_test_)
                        model = xgb.XGBRegressor(max_depth=max_d_,
                                                 max_leaves=0,
                                                 n_estimators=n_est,
                                                 learning_rate=learning_ratio,
                                                 reg_lambda=lamda_,
                                                 reg_alpha=alpha_,
                                                 colsample_bylevel=1,
                                                 # booster = 'gblinear',
                                                 min_child_weight=min_child_,
                                                 gamma=gamma_,
                                                 subsample=subs_,
                                                 colsample_bytree=subs_,
                                                 # eval_metric= 'rmse',
                                                 objective= weighted_mae_obj,
                                                 # quantile_alpha=0.5,
                                                 early_stopping_rounds=2,
                                                 device='cuda:0'
                                                 ).fit(x_train_, y_train_, eval_set=[(x_train_, y_train_), (x_test_, y_test_)])

                        y_train_pred = model.predict(x_train_)
                        y_test_pred = model.predict(x_test_)

                        # # reverse normalise
                        # y_train_pred = (5 * np.exp(y_train_pred) + 2) / (1 + np.exp(y_train_pred))
                        # y_test_pred = (5 * np.exp(y_test_pred) + 2) / (1 + np.exp(y_test_pred))
                        # y_train_pred = y_train_pred ** 2
                        # y_test_pred = y_test_pred ** 2
                        # y_train_pred = pt.inverse_transform(y_train_pred.reshape(-1,1))
                        # y_test_pred = pt.inverse_transform(y_test_pred.reshape(-1,1))
                        # y_train_pred = y_train_pred.flatten()
                        # y_test_pred = y_test_pred.flatten()

                        results = model.evals_result()
                        plt.figure(figsize=(10, 6))
                        plt.plot(results['validation_0']['mphe'], label='Train MAE', color ='blue')
                        plt.plot(results['validation_1']['mphe'], label='Validation MAE', color ='red')
                        plt.xlabel('Boosting Round')
                        plt.ylabel('MAE')
                        plt.title('Training and Validation MAE')
                        plt.legend()
                        plt.savefig(os.path.join(train_plot_folder, f'res_mode{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lamda_)}_alpha{str(alpha_)}_fold{str(fold_)}.png'), dpi=300)
                        plt.close()
                        fold_ += 1

                        measure_dic['train_score'].append(model.score(x_train_.get(), y_train_.get()))
                        measure_dic['test_score'].append(model.score(x_test_.get(), y_test_.get()))
                        measure_dic['train_mae'].append(metrics.mean_absolute_error(y_train_ori, y_train_pred))
                        measure_dic['test_mae'].append(metrics.mean_absolute_error(y_test_ori, y_test_pred))
                        measure_dic['train_rmse'].append(metrics.root_mean_squared_error(y_train_ori, y_train_pred))
                        measure_dic['test_rmse'].append(metrics.root_mean_squared_error(y_test_ori, y_test_pred))
                        measure_dic['train_mse'].append(metrics.mean_squared_error(y_train_ori, y_train_pred))
                        measure_dic['test_mse'].append(metrics.mean_squared_error(y_test_ori, y_test_pred))
                        measure_dic['train_r2'].append(metrics.r2_score(y_train_ori, y_train_pred))
                        measure_dic['test_r2'].append(metrics.r2_score(y_test_ori, y_test_pred))
                        measure_dic['lr'].append(learning_ratio)
                        measure_dic['OE'].append(model.get_booster().best_iteration)

                        # Importance ranking
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        measure_dic['rank'].append([x_feature[f] for f in indices])

                    else:
                        dtrain = xgb.DMatrix(x_train_, label=y_train_)
                        dtest = xgb.DMatrix(x_test_, label=y_test_)
                        evals_result = {}

                        # 参数设定
                        params = {
                            'max_depth': max_d_,
                            'max_leaves': 0,
                            'eta': learning_ratio,
                            'lambda': lamda_,
                            'alpha': alpha_,
                            'colsample_bylevel': 1,
                            'min_child_weight': min_child_,
                            'gamma': gamma_,
                            'subsample': subs_,
                            'colsample_bytree': subs_,
                            'device': 'cuda:0',
                            # 'booster': 'dart',
                        }

                        # 训练模型
                        model = xgb.train(
                            params=params,
                            dtrain=dtrain,
                            num_boost_round=n_est,
                            obj=weighted_mae_obj,
                            feval=weighted_mae_metric,
                            evals=[(dtrain, 'train'), (dtest, 'test')],
                            evals_result=evals_result,
                            early_stopping_rounds=2
                        )

                        y_train_pred = model.predict(dtrain)
                        y_test_pred = model.predict(dtest)

                        results = evals_result
                        plt.figure(figsize=(10, 6))
                        plt.plot(results['train']['weighted_mae'], label='Train MAE', color ='blue')
                        plt.plot(results['test']['weighted_mae'], label='Validation MAE', color ='red')
                        plt.xlabel('Boosting Round')
                        plt.ylabel('MAE')
                        plt.title('Training and Validation MAE')
                        plt.legend()
                        plt.savefig(os.path.join(train_plot_folder, f'res_mode{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lamda_)}_alpha{str(alpha_)}_fold{str(fold_)}.png'), dpi=300)
                        plt.close()
                        fold_ += 1

                        # 性能指标计算并记录
                        measure_dic['train_score'].append(1)
                        measure_dic['test_score'].append(0)

                        measure_dic['train_mae'].append(metrics.mean_absolute_error(y_train_ori, y_train_pred))
                        measure_dic['test_mae'].append(metrics.mean_absolute_error(y_test_ori, y_test_pred))
                        measure_dic['train_rmse'].append(metrics.mean_squared_error(y_train_ori, y_train_pred))
                        measure_dic['test_rmse'].append(metrics.mean_squared_error(y_test_ori, y_test_pred))
                        measure_dic['train_mse'].append(metrics.mean_squared_error(y_train_ori, y_train_pred))
                        measure_dic['test_mse'].append(metrics.mean_squared_error(y_test_ori, y_test_pred))
                        measure_dic['train_r2'].append(metrics.r2_score(y_train_ori, y_train_pred))
                        measure_dic['test_r2'].append(metrics.r2_score(y_test_ori, y_test_pred))
                        measure_dic['lr'].append(learning_ratio)
                        measure_dic['OE'].append(model.best_iteration)

                        # 特征重要性排序
                        model.feature_names = x_feature
                        importances = model.get_score(importance_type='weight')
                        indices = sorted(importances, key=importances.get, reverse=True)
                        measure_dic['rank'].append(indices)


                    if print_info:
                        original_test_y_list.extend(y_test_ori.tolist())
                        original_train_y_list.extend(y_train_ori.tolist())
                        predict_test_y_list.extend(y_test_pred.tolist())
                        predict_train_y_list.extend(y_train_pred.tolist())
                        train_index_list.extend(train_index_)
                        test_index_list.extend(test_index_)

                for _ in measure_dic.keys():
                    if _ == 'rank':
                        measure_dic[_].append(sort_by_average_position(measure_dic[_]))
                    else:
                        measure_dic[_].append(np.mean(measure_dic[_]))

            else:

                x_train_, y_train_, x_test_, y_test_ = x_train, y_train, x_test, y_test
                start_time = time.time()
                model = xgb.XGBRegressor(max_depth=max_d_,
                                         max_leaves=0,
                                         learning_rate=learning_ratio,
                                         n_estimators=n_est,
                                         reg_lambda=lamda_,
                                         reg_alpha=alpha_,
                                         colsample_bylevel=1,
                                         min_child_weight=min_child_,
                                         gamma=gamma_,
                                         subsample=subs_,
                                         colsample_bytree=subs_,
                                         eval_metric= meas_,
                                         early_stopping_rounds=3,
                                         device='cuda:0').fit(x_train_, y_train_, eval_set=[(x_train_, y_train_), (x_test_, y_test_)])

                y_train_pred = model.predict(x_train_)
                y_test_pred = model.predict(x_test_)

                measure_dic['train_score'].append(model.score(x_train_.get(), y_train_.get()))
                measure_dic['test_score'].append(model.score(x_test_.get(), y_test_.get()))
                measure_dic['train_mae'].append(metrics.mean_absolute_error(y_train_.get(), y_train_pred))
                measure_dic['test_mae'].append(metrics.mean_absolute_error(y_test_.get(), y_test_pred))
                measure_dic['train_rmse'].append(metrics.root_mean_squared_error(y_train_.get(), y_train_pred))
                measure_dic['test_rmse'].append(metrics.root_mean_squared_error(y_test_.get(), y_test_pred))
                measure_dic['train_mse'].append(metrics.mean_squared_error(y_train_.get(), y_train_pred))
                measure_dic['test_mse'].append(metrics.mean_squared_error(y_test_.get(), y_test_pred))
                measure_dic['train_r2'].append(metrics.r2_score(y_train_.get(), y_train_pred))
                measure_dic['test_r2'].append(metrics.r2_score(y_test_.get(), y_test_pred))
                measure_dic['lr'].append(learning_ratio)
                measure_dic['OE'].append(model.get_booster().best_iteration)

                if print_info:
                    original_test_y_list.extend(y_test_.tolist())
                    original_train_y_list.extend(y_train_.tolist())
                    predict_test_y_list.extend(model.predict(x_test_).tolist())
                    predict_train_y_list.extend(model.predict(x_train_).tolist())

                # Importance ranking
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                rank = [x_feature[f] for f in indices]

            df = pd.DataFrame(measure_dic)
            df.to_csv(ex_filename)
            con_model_f = True
        else:
            df = pd.read_csv(ex_filename)
            measure_dict = df.to_dict()
            con_model_f = False
        print(f'End time 4 XGboost mode:{str(mode)} dep:{str(max_d_)} minchild:{str(min_child_)} gamma:{str(gamma_)} subs:{str(subs_)} lambda:{str(lamda_)} alpha{str(alpha_)} Time:{str(time.time() - start_time)}')

        if print_info:
            train_test_y_df = {'ori': copy.deepcopy(original_test_y_list), 'pre': copy.deepcopy(predict_test_y_list), 'index': copy.deepcopy(test_index_list)}
            train_test_y_df['ori'].extend(original_train_y_list)
            train_test_y_df['pre'].extend(predict_train_y_list)
            train_test_y_df['index'].extend(train_index_list)
            train_test_y_df = pd.DataFrame(train_test_y_df)
            train_test_y_df.to_csv(os.path.join(res_plot_folder, f'fig_test_{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lamda_)}_alpha{str(alpha_)}_fold{str(fold_)}.csv'))

            print('----------------------------------------------------------------------------------------------')
            print(f'X_feature:{str(x_feature)}, Y_feature:{str(y_feature)}, Max depth:{str(max_d_)}')
            print('XGB模型Train得分: ', str(df['train_score'].iloc[-1]))
            print('Train MAE:', str(df['train_mae'].iloc[-1]))
            print('Train MSE:', str(df['train_mse'].iloc[-1]))
            print('Train RMSE:', str(df['train_rmse'].iloc[-1]))

            print('XGB模型Test得分: ', str(df['test_score'].iloc[-1]))
            print('Test MAE:', str(df['test_mae'].iloc[-1]))
            print('Test MSE:', str(df['test_mse'].iloc[-1]))
            print('Test RMSE:', str(df['test_rmse'].iloc[-1]))
            print('----------------------------------------------------------------------------------------------')

            plt.rcParams['font.family'] = ['Arial', 'SimHei']
            plt.rc('font', size=14)
            fig3, ax3 = plt.subplots(figsize=(5, 5), constrained_layout=True)

            qq = ax3.hist2d([_ -0.15 for _ in predict_test_y_list], original_test_y_list, bins=60, range=[[2, 6], [2, 6]], cmap=plt.cm.YlOrBr, vmin=5, vmax=200)
            ax3.plot(np.linspace(0, 6, 6), np.linspace(0, 6, 6), c=(0, 0, 0), lw=1.5, zorder=3, ls='--')
            # ax3.plot(np.linspace(0, 6, 6), np.linspace(0.93, chl + 0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
            # ax3.plot(np.linspace(0, 6, 6), np.linspace(-0.93, chl - 0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
            # ax3.fill_between(np.linspace(0, chl, chl), np.linspace(-0.93, chl - 0.93, chl), np.linspace(+0.93, chl + 0.93, chl), color=(0.3, 0.3, 0.3), alpha=0.1)
            ax3.set_xlabel('Landsat-modelled Poeacea vegetation height/m', fontsize=12, fontweight='bold')
            ax3.set_ylabel('GEDI RH100/m', fontsize=12, fontweight='bold')
            ax3.text(2.3, 6 - 0.5, f"MAE={str(measure_dic['test_mae'][-1])[0: 4]}m", c=(0, 0, 0), fontsize=16, fontweight='bold')
            ax3.text(2.3, 6 - 0.8, f"RMSE={str(measure_dic['test_rmse'][-1])[0: 4]}m", c=(0, 0, 0), fontsize=16, fontweight='bold')
            ax3.set_ylim(2, 6)
            ax3.set_xlim(2, 6)
            plt.savefig(os.path.join(res_plot_folder, f'fig_test_{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lamda_)}_alpha{str(alpha_)}.png'), dpi=300)
            plt.close()

            plt.rcParams['font.family'] = ['Arial', 'SimHei']
            plt.rc('font', size=14)
            fig3, ax3 = plt.subplots(figsize=(5, 5), constrained_layout=True)

            qq = ax3.hist2d([_ - 0.15 for _ in predict_train_y_list], original_train_y_list, bins=50, range=[[2, 6], [2, 6]], cmap=plt.cm.YlOrBr, vmin=5, vmax=800)
            ax3.plot(np.linspace(0, 6, 6), np.linspace(0, 6, 6), c=(0, 0, 0), lw=1.5, zorder=3, ls='--')
            # ax3.plot(np.linspace(0, chl, chl), np.linspace(0.93, chl + 0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
            # ax3.plot(np.linspace(0, chl, chl), np.linspace(-0.93, chl - 0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
            # ax3.fill_between(np.linspace(0, chl, chl), np.linspace(-0.93, chl - 0.93, chl),
            #                  np.linspace(+0.93, chl + 0.93, chl),
            #                  color=(0.3, 0.3, 0.3), alpha=0.1)
            ax3.set_xlabel('Landsat-modelled Poeacea vegetation height/m', fontsize=12, fontweight='bold')
            ax3.set_ylabel('GEDI RH100/m', fontsize=12, fontweight='bold')
            ax3.text(2.3, 6 - 0.5, f"MAE={str(measure_dic['train_mae'][-1])[0: 4]}m", c=(0, 0, 0), fontsize=18, fontweight='bold')
            ax3.text(2.3, 6 - 0.8, f"RMSE={str(measure_dic['train_rmse'][-1])[0: 4]}m", c=(0, 0, 0), fontsize=18, fontweight='bold')
            ax3.set_ylim(2, 6)
            ax3.set_xlim(2, 6)
            plt.savefig(os.path.join(res_plot_folder, f'fig_train_{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lamda_)}_alpha{str(alpha_)}.png'), dpi=300)
            plt.close()

            # # Shap explainer
            # explainer = shap.Explainer(model)
            # shap_values = explainer(x_train_)
            #
            # x_feature = [_.replace('_area_average', '') for _ in x_feature]
            # x_feature = [_.replace('noninun_', '') for _ in x_feature]
            # x_feature = [_.replace('Denv_', '') for _ in x_feature]
            # x_feature = [_.replace('Pheme_', '') for _ in x_feature]
            # shap_values.feature_names = x_feature
            # # # visualize the first prediction's explanation
            # shap.plots.bar(shap_values, max_display=100, show=False, clustering_cutoff=0.5)
            # plt.savefig(os.path.join(res_plot_folder, f'shap_v_{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lamda_)}_alpha{str(alpha_)}.png'),
            #             dpi=300)
            # plt.close()
            #
            # shap.summary_plot(shap_values, x_train_, plot_type='bar', max_display=100, show=False,)
            # plt.savefig(os.path.join(res_plot_folder, f'shap_sum_{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lamda_)}_alpha{str(alpha_)}.png'),
            #             dpi=300)
            # plt.close()

        return df.iloc[-1:]
    except:
        print(traceback.print_exc())
        # print(f'x_train_: {str(np.isnan(x_train_).any)}')
        # print(f'x_test_: {str(np.isnan(x_test_).any)}')
        # print(f'y_train_: {str(np.isnan(y_train_).any)}')
        # print(f'y_test_: {str(np.isnan(y_test_).any)}')
        # print(f'x_train_: {str(np.isinf(x_train_.get()).any)}')
        # print(f'x_test_: {str(np.isinf(x_test_.get()).any)}')
        # print(f'y_train_: {str(np.isinf(y_train_.get()).any)}')
        # print(f'y_test_: {str(np.isinf(y_test_.get()).any)}')
        raise Exception('Error during XGBOOST handling above err')

def RandomForest(x_train_, y_train_, x_test_, y_test_, max_d_, n_est_):
    model = RandomForestRegressor(max_depth=max_d_,
                                  n_estimators=n_est_,
                                  max_features=1,
                                  min_samples_leaf=4,
                                  n_jobs=-1).fit(x_train_, y_train_, eval_set=[(x_train_, y_train_), (x_test_, y_test_)])


class VHM(object):

    def __init__(self, model_type):

        # Define the parameters
        self.work_env = None
        self.gedi_linked_RS_df = None
        self.gedi_linked_RS_df_ = None
        self.cross_validation_factor = None
        self.mode_list = None
        self.model_type = None
        self.model = None
        self.normalised_dataset = False
        self.res_df = None

        # Define the var
        self.train_test_ds_dic = {}
        self.data_index_dic = {}
        self.hyperparameters = {}
        self.hyperparameters_df = {}
        self.mode_dic = {}
        self.learnig_ratio = 0.05

        # Define the support mode
        self._support_mode = [0, 1, 2, 3, 4, 5, 6]
        self._support_hyperpara = ['max_d', 'child_weight',  'gamma', 'subsample', 'lamda', 'alpha', 'n_est']
        self._default_hyperpara = {'max_d': [8, 9, 10], 'child_weight': [0.5, 1, 1.5],  'gamma': [0], 'subsample': [0.79], 'lamda': [10], 'alpha': [4], 'n_est': 1000}

        # Define the model type
        if isinstance(model_type, str) and model_type in ['XGB', 'RFR']:
            self.model_type = model_type
        else:
            raise Exception('The model type should be either XGB or RFR')

    def input_dataset(self, gedi_linked_csv, mode_list=None, cross_validation_factor=True, year_factor = True, work_env=None, normalised_dataset=False, print_dis=False):

        # User-defined indicator
        if not isinstance(cross_validation_factor, bool):
            raise Exception('The cross validation factor should be a boolean')
        else:
            self.cross_validation_factor = cross_validation_factor

        # User-defined indicator
        if not isinstance(year_factor, bool):
            raise Exception('Thyear_factor should be a boolean')
        else:
            self.year_factor = year_factor

        if mode_list is None:
            self.mode_list = copy.deepcopy(self._support_mode)
        elif not isinstance(mode_list, list):
            raise Exception('The mode list should be a list')
        else:
            self.mode_list = [_ for _ in mode_list if _ in self._support_mode]
            if len(self.mode_list) == 0:
                raise Exception('The mode list is not supported!')

        if not isinstance(normalised_dataset, bool):
            raise Exception('The normalised dataset should be a boolean')
        else:
            self.normalised_dataset = normalised_dataset

        # Input the dataset
        if not os.path.exists(gedi_linked_csv):
            raise Exception('The GEDI linked csv does not exist')
        else:
            self.gedi_linked_RS_df = pd.read_csv(gedi_linked_csv)

        # Output folder
        if work_env is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.work_env = os.path.join(os.path.dirname(gedi_linked_csv), f'VHM_{str(timestamp)}')
            bf.create_folder(self.work_env)
        elif not os.path.isdir(self.work_env):
            raise Exception('The work environment is not a folder')
        else:
            bf.create_folder(self.work_env)

        # Give the overall index
        self.gedi_linked_RS_df.reset_index(drop=False, inplace=True, names='ori_index')

        # Preselect the dataset
        self.gedi_linked_RS_df['DOY'] = (self.gedi_linked_RS_df['Date'] % 1000) / 365.25
        self.gedi_linked_RS_df = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['VegType_area_average'] < 3]
        self.gedi_linked_RS_df = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['Canopy Height (rh100)'] <= 5]
        self.gedi_linked_RS_df = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['Canopy Height (rh100)'] > 2]
        self.gedi_linked_RS_df = self.gedi_linked_RS_df.replace(np.inf, np.nan)
        self.gedi_linked_RS_df = self.gedi_linked_RS_df.replace(-np.inf, np.nan)

        # if os.path.exists('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\data_isolation.csv'):
        #     df_exclude = pd.read_csv('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\data_isolation.csv')
        #     df_exclude['per'] = (df_exclude['ori'] - df_exclude['pre']) / df_exclude['pre']
        #     df_exclude = np.unique(np.array(df_exclude[df_exclude['per'] > 0.9]['ori']))
        #     self.gedi_linked_RS_df = self.gedi_linked_RS_df[~self.gedi_linked_RS_df['Canopy Height (rh100)'].isin(df_exclude)].reset_index(drop=True)

        # self.gedi_linked_RS_df = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['Canopy Height (rh100)'] >= 2]
        # self.gedi_linked_RS_df = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['Landsat water rate'] < 0.1]

        # Normalised dataset
        if self.normalised_dataset:
            peak_doy_key = [_ for _ in self.gedi_linked_RS_df.keys() if 'peak_doy' in _][0]
            peak_vi_key = [_ for _ in self.gedi_linked_RS_df.keys() if 'peak_vi' in _][0]
            
            #
            for key in self.gedi_linked_RS_df.keys():

                # Remove 0
                if 'Denv_AGB' in key:
                    self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df[key] == 0, key] = np.nan

                if 'ratio_gedi_fast-growing' in key:
                    arr = np.array(self.gedi_linked_RS_df[key]).reshape(-1, 1)
                    arr = np.log(arr + 1)
                    self.gedi_linked_RS_df[key] = arr

                if 'reliability' not in key:
                    if 'OSAVI' in key:
                        pass
                    elif 'vi' in key or 'VI' in key or 'GREENESS' in key:
                        pass
                    elif 'noninun' in key:
                        arr = np.abs(np.array(self.gedi_linked_RS_df[key]).flatten())
                        arr = arr[~np.isnan(arr)]
                        q = np.sort(arr)[::-1][0]
                        self.gedi_linked_RS_df[key] = self.gedi_linked_RS_df[key] / q
                    elif 'Pheme' in key and ('SOS' in key or 'peak_doy' in key or 'trough_doy' in key or 'EOS' in key or 'EOM' in key):
                        self.gedi_linked_RS_df[key] = self.gedi_linked_RS_df[key] / 365.25
                    elif 'Pheme' in key and ('DR' in key or 'GR' in key):
                        self.gedi_linked_RS_df[key] = np.arctan(self.gedi_linked_RS_df[key]) / (np.pi / 4)
                    elif key == 'DOY':
                        self.gedi_linked_RS_df[key] = self.gedi_linked_RS_df[key] / self.gedi_linked_RS_df[peak_doy_key]
                        self.gedi_linked_RS_df[key] = self.gedi_linked_RS_df[key].clip(upper=1)
                else:
                    pass

                if 'reliability' not in key and 'Denv' in key and 'ratio' not in key:

                    # pt = PowerTransformer(method = 'yeo-johnson', standardize=True)
                    arr = np.array(self.gedi_linked_RS_df[key]).reshape(-1, 1)
                    arr = np.log(arr - np.nanmin(arr) + 1)
                    # # arr = pt.fit_transform(arr)
                    self.gedi_linked_RS_df[key] = arr

                    # arr = np.abs(np.array(self.gedi_linked_RS_df[key]).flatten())
                    # arr = arr[~np.isnan(arr)]
                    # q = np.sort(arr)[::-1][0]
                    # self.gedi_linked_RS_df[key] = self.gedi_linked_RS_df[key] / q

        # if reliability:
        #     reliability_key = [_ for _ in self.gedi_linked_RS_df.keys() if 'reliability' in _][0]
        #     self.gedi_linked_RS_df = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df[reliability_key] > 0.5]


        # Define the index
        index_area_average = [_ for _ in self.gedi_linked_RS_df.keys() if 'noninun' in _ and 'area_average' in _ and 'reliability' not in _]
        pheindex_area_average = [_ for _ in self.gedi_linked_RS_df.keys() if 'Pheme' in _ and 'area_average' in _ and 'reliability' not in _ and 'EOS' not in _]
        denv_area_average = [_ for _ in self.gedi_linked_RS_df.keys() if 'Denv' in _ and 'area_average' in _ and 'reliability' not in _ and 'RHU' not in _]

        index_focal = [_ for _ in self.gedi_linked_RS_df.keys() if 'noninun' in _ and 'focal' in _ and 'reliability' not in _]
        pheindex_focal = [_ for _ in self.gedi_linked_RS_df.keys() if 'Pheme' in _ and 'focal' in _ and 'reliability' not in _ and 'EOS' not in _]
        denv_focal = [_ for _ in self.gedi_linked_RS_df.keys() if 'Denv' in _ and 'focal' in _ and 'reliability' not in _ and 'RHU' not in _]

        # Generate the mode

        for mode in self.mode_list:
            # Def the mode of model
            if mode == 0:
                mode_index = copy.copy(index_area_average)
            elif mode == 1:
                mode_index = copy.copy(index_area_average)
                mode_index.extend(pheindex_area_average)
            elif mode == 2:
                mode_index = copy.copy(index_area_average)
                mode_index.extend(denv_area_average)
            elif mode == 3:
                mode_index = copy.copy(index_area_average)
                mode_index.extend(pheindex_area_average)
                mode_index.extend(denv_area_average)

            elif mode == 4:
                mode_index = copy.copy(index_focal)
                mode_index.extend(pheindex_focal)
                mode_index.extend(denv_focal)

            elif mode == 5:
                mode_index = copy.copy(pheindex_focal)
                mode_index.extend(denv_focal)

            elif mode == 6:
                mode_index = copy.copy(pheindex_focal)
            else:
                raise Exception('Error Mode')
            self.mode_dic[mode] = mode_index

            self.train_test_ds_dic[f'x_train_mode{str(mode)}'] = []
            self.train_test_ds_dic[f'y_train_mode{str(mode)}'] = []
            self.train_test_ds_dic[f'x_test_mode{str(mode)}'] = []
            self.train_test_ds_dic[f'y_test_mode{str(mode)}'] = []

            self.data_index_dic[f'train_mode{str(mode)}'] = []
            self.data_index_dic[f'test_mode{str(mode)}'] = []

            self.gedi_linked_RS_df_ = self.gedi_linked_RS_df.dropna(subset=mode_index).reset_index(drop=True)
            print(f'Mode:{str(mode)}, The number of samples:{str(self.gedi_linked_RS_df_.shape[0])}, drop ratio:{str(1 - self.gedi_linked_RS_df_.shape[0] / self.gedi_linked_RS_df.shape[0])}')

            if print_dis:
                input_dis_folder = os.path.join(self.work_env, f'input_distribution\\mode_{str(mode)}\\')
                bf.create_folder(input_dis_folder)
                self.gedi_linked_RS_df_.to_csv(os.path.join(self.work_env, f'input_distribution\\{str(mode)}.csv'))
                for mode_index_ in mode_index:
                    plt.hist(self.gedi_linked_RS_df_[mode_index_], bins=100)
                    plt.savefig(os.path.join(input_dis_folder, f'{mode_index_}.png'), dpi=300)
                    plt.close()

            iso = IsolationForest(contamination=0.01, random_state=42)
            yhat = iso.fit_predict(self.gedi_linked_RS_df_[mode_index])
            self.gedi_linked_RS_df_ = self.gedi_linked_RS_df_[yhat != -1].reset_index(drop=True)

            y_binned = pd.qcut(self.gedi_linked_RS_df_['Canopy Height (rh100)'], q=20, labels=False)
            # self.gedi_linked_RS_df_['Canopy Height (rh100)'] = self.gedi_linked_RS_df_['Canopy Height (rh100)'] * 10

            # Construct the train test dataset
            if cross_validation_factor and year_factor:
                for year in [2019, 2020, 2021, 2022, 2023]:
                    x_train = self.gedi_linked_RS_df_[self.gedi_linked_RS_df_['Date'] // 1000 != year][mode_index]
                    y_train = self.gedi_linked_RS_df_[self.gedi_linked_RS_df_['Date'] // 1000 != year]['Canopy Height (rh100)']
                    x_test = self.gedi_linked_RS_df_[self.gedi_linked_RS_df_['Date'] // 1000 == year][mode_index]
                    y_test = self.gedi_linked_RS_df_[self.gedi_linked_RS_df_['Date'] // 1000 == year]['Canopy Height (rh100)']

                    if x_train.shape[0] != 0 and x_test.shape[0] != 0:
                        self.train_test_ds_dic[f'x_train_mode{str(mode)}'].append(x_train)
                        self.train_test_ds_dic[f'y_train_mode{str(mode)}'].append(y_train)
                        self.train_test_ds_dic[f'x_test_mode{str(mode)}'].append(x_test)
                        self.train_test_ds_dic[f'y_test_mode{str(mode)}'].append(y_test)

            elif cross_validation_factor and not year_factor:
                kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                for train_index, test_index in kf.split(self.gedi_linked_RS_df_[mode_index], y_binned):
                    x_train_ = self.gedi_linked_RS_df_.loc[train_index, mode_index]
                    y_train_ = self.gedi_linked_RS_df_.loc[train_index, 'Canopy Height (rh100)']
                    x_test_ = self.gedi_linked_RS_df_.loc[test_index, mode_index]
                    y_test_ = self.gedi_linked_RS_df_.loc[test_index, 'Canopy Height (rh100)']
                    train_index_ = self.gedi_linked_RS_df_.loc[train_index, 'ori_index']
                    test_index_ = self.gedi_linked_RS_df_.loc[test_index, 'ori_index']
                    if x_train_.shape[0] != 0 and x_test_.shape[0] != 0:
                        self.train_test_ds_dic[f'x_train_mode{str(mode)}'].append(np.array(x_train_))
                        self.train_test_ds_dic[f'y_train_mode{str(mode)}'].append(np.array(y_train_))
                        self.train_test_ds_dic[f'x_test_mode{str(mode)}'].append(np.array(x_test_))
                        self.train_test_ds_dic[f'y_test_mode{str(mode)}'].append(np.array(y_test_))
                        self.data_index_dic[f'train_mode{str(mode)}'].append(np.array(train_index_))
                        self.data_index_dic[f'test_mode{str(mode)}'].append(np.array(test_index_))

            else:
                x_train_, x_test_, y_train_, y_test_ = train_test_split(self.gedi_linked_RS_df_[mode_index], self.gedi_linked_RS_df_['Canopy Height (rh100)'], test_size=0.2, random_state=41)
                if x_train_.shape[0] != 0 and x_test_.shape[0] != 0:
                    self.train_test_ds_dic[f'x_train_mode{str(mode)}'] = x_train_
                    self.train_test_ds_dic[f'y_train_mode{str(mode)}'] = y_train_
                    self.train_test_ds_dic[f'x_test_mode{str(mode)}'] = x_test_
                    self.train_test_ds_dic[f'y_test_mode{str(mode)}'] = y_test_

    def train_VHM(self, learning_ratio=0.1, bulk_train=True, print_info=False, **hyperparameters):

        if len(list(self.train_test_ds_dic.keys())) == 0:
            raise Exception('The train test dataset is not defined')

        # Check the hyperparameters
        for hyperparameter in hyperparameters.keys():
            if hyperparameter not in self._support_hyperpara:
                raise Exception(f'The hyperparameter {hyperparameter} is not supported')
            elif not isinstance(hyperparameters[hyperparameter], list):
                raise Exception(f'The hyperparameter {hyperparameter} should be a list')
            else:
                self.hyperparameters[hyperparameter] = hyperparameters[hyperparameter]

        # Reassign the hyperpara
        for hyperparameter in self._support_hyperpara:
            if hyperparameter not in self.hyperparameters.keys():
                self.hyperparameters[hyperparameter] = self._default_hyperpara[hyperparameter]


        # Create outputfolder
        outputpath = os.path.join(self.work_env, f'VHM_res\\')
        bf.create_folder(outputpath)

        # Generate the Hyperparameters df
        if not os.path.exists(os.path.join(outputpath, 'res.csv')):
            exist_hyperparameters = None
        else:
            self.res_df = pd.read_csv(os.path.join(outputpathv, 'res.csv'))
            exist_hyperparameters = self.res_df.loc[:, ['max_d', 'child_weight', 'gamma', 'subsample', 'lamda', 'alpha', 'mode']].values.tolist()

        self.hyperparameters_df = {'max_d': [], 'child_weight': [], 'gamma': [], 'subsample': [], 'lamda': [], 'alpha': [], 'mode': [], 'x_feature': []}
        for max_d_, min_child_,gamma_, subs_, lamda_, alpha_, mode_ in itertools.product(self.hyperparameters['max_d'],
                                                                                                      self.hyperparameters['child_weight'],
                                                                                                      self.hyperparameters['gamma'],
                                                                                                      self.hyperparameters['subsample'],
                                                                                                      self.hyperparameters['lamda'],
                                                                                                      self.hyperparameters['alpha'],
                                                                                                      self.mode_list):
            if exist_hyperparameters is None or [max_d_, min_child_,gamma_, subs_, lamda_, alpha_, mode_] not in exist_hyperparameters:
                self.hyperparameters_df['max_d'].append(max_d_)
                self.hyperparameters_df['child_weight'].append(min_child_)
                self.hyperparameters_df['gamma'].append(gamma_)
                self.hyperparameters_df['subsample'].append(subs_)
                self.hyperparameters_df['lamda'].append(lamda_)
                self.hyperparameters_df['alpha'].append(alpha_)
                self.hyperparameters_df['mode'].append(mode_)
                self.hyperparameters_df['x_feature'].append(self.mode_dic[mode_])
        self.hyperparameters_df = pd.DataFrame(self.hyperparameters_df)

        if self.hyperparameters_df.shape[0] != 0:
            if bulk_train:
                with ProcessPoolExecutor(max_workers=int(os.cpu_count() * RF.configuration['multiprocess_ratio'])) as exe:
                    res = exe.map(XGBoost, repeat(self.train_test_ds_dic), repeat(outputpath), self.hyperparameters_df['mode'],
                                  self.hyperparameters_df['max_d'], self.hyperparameters_df['child_weight'], self.hyperparameters_df['gamma'],
                                  self.hyperparameters_df['subsample'], self.hyperparameters_df['lamda'], self.hyperparameters_df['alpha'],
                                  self.hyperparameters_df['x_feature'], repeat('Canopy Height (rh100)'), repeat(learning_ratio), repeat(self.data_index_dic), repeat(print_info))

            else:
                res = []
                for max_d_, min_child_,gamma_, subs_, lamda_, alpha_, mode_ in zip(self.hyperparameters_df['max_d'],
                                                                                   self.hyperparameters_df['child_weight'],
                                                                                   self.hyperparameters_df['gamma'],
                                                                                   self.hyperparameters_df['subsample'],
                                                                                   self.hyperparameters_df['lamda'],
                                                                                   self.hyperparameters_df['alpha'],
                                                                                   self.hyperparameters_df['mode']):
                    res.append(XGBoost(self.train_test_ds_dic, outputpath, mode_, max_d_, min_child_, gamma_, subs_, lamda_, alpha_, self.mode_dic[mode_], 'Canopy Height (rh100)', learning_ratio, self.data_index_dic, print_info=print_info))

            res = list(res)
            res_df = None
            for _ in res:
                if res_df is not None:
                    res_df = pd.concat([res_df, _], axis=0)
                else:
                    res_df = _

            self.res_df = pd.concat([self.hyperparameters_df, res_df.reset_index(drop=True)], axis=1)
            self.res_df.to_csv(os.path.join(self.work_env, 'res.csv'))

    def analyse_model(self, mode_, learning_ratio=0.1,  bulk_train=True, **hyperparameters):

        # Check the dataset
        if len(list(self.train_test_ds_dic.keys())) == 0:
            raise Exception('The train test dataset is not defined')

        # Check the hyperparameters
        for hyperparameter in hyperparameters.keys():
            if hyperparameter not in self._support_hyperpara:
                raise Exception(f'The hyperparameter {hyperparameter} is not supported')
            elif not isinstance(hyperparameters[hyperparameter], (np.float_, float, int, np.int_)):
                raise Exception(f'The hyperparameter {hyperparameter} should be a list')
            else:
                self.hyperparameters[hyperparameter] = hyperparameters[hyperparameter]

        for hyperparameter in self._support_hyperpara:
            if hyperparameter not in self.hyperparameters.keys():
                self.hyperparameters[hyperparameter] = self._default_hyperpara[hyperparameter]

        # Create folder
        output_folder = os.path.join(self.work_env, 'analyse_fig\\')
        bf.create_folder(output_folder)

        XGBoost(self.train_test_ds_dic, output_folder, mode_, self.hyperparameters['max_d'], self.hyperparameters['child_weight'],
                self.hyperparameters['gamma'], self.hyperparameters['subsample'], self.hyperparameters['lamda'],
                self.hyperparameters['alpha'], self.mode_dic[mode_], 'Canopy Height (rh100)', learning_ratio, print_info=True)


    def grid_search_best_para(self):

        # Find the best hyperparameters
        best_para = self.res_df.loc[self.res_df['test_rmse'].idxmin()]
        print(f"The best hyperparameters are: {[_ + ':' + str(best_para[_]) for _ in list(best_para.keys())]}")

        bf.create_folder(f'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\VHM\\VHM_hyperpara_plot\\')

        para_dic = {}
        for keys in list(self.hyperparameters_df.keys()):
            if keys != 'x_feature':
                unique_values = self.res_df[keys].unique().tolist()
                unique_values.sort()
                para_temp = []
                para2_temp = []
                for value in unique_values:
                    average_test_rmse = np.mean(self.res_df['test_rmse'][self.res_df[keys] == value])
                    average_train_rmse = np.mean(self.res_df['train_rmse'][self.res_df[keys] == value])
                    para_temp.append(average_test_rmse)
                    para2_temp.append(average_train_rmse)
                para_dic[keys] = [unique_values, para_temp, para2_temp]

        for keys in para_dic:
            plt.plot(para_dic[keys][0], para_dic[keys][1], label=keys)
            plt.plot(para_dic[keys][0], para_dic[keys][2], label=keys)
            plt.legend()
            plt.savefig(f'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\VHM\\VHM_hyperpara_plot\\{keys}.png', dpi=300)
            plt.close()


if __name__ == '__main__':

    # Potential solution 1: RF
    # Potential solution 2

    # bf.merge_csv_files("G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\GEDI_link_RS\\", 'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_GEDI_2019_2023_for_model_high_quality.csv',
    #                    merge_keys = ['Quality Flag', 'Canopy Elevation (m)', 'index', 'EPSG_lat', 'Elevation (m)', 'Beam', 'Landsat water rate', 'Latitude', 'Shot Number', 'Sensitivity', 'EPSG_lon', 'Tandem-X DEM', 'Longitude', 'Degrade Flag', 'Date', 'RH 98', 'RH 25', 'Urban rate', 'Canopy Height (rh100)', 'Leaf off flag'])
    VHM = VHM('XGB')
    VHM.input_dataset('G:\A_GEDI_Floodplain_vegh\GEDI_MYR\L2_vegh\\floodplain_GEDI_2019_2023_for_model_high_quality.csv', mode_list=[3,],
                      cross_validation_factor=True, normalised_dataset=True, year_factor=False)
    # VHM.analyse_model(3, learning_ratio=0.05,  max_d=5, child_weight=0, gamma=0, subsample=0.9, lamda= 5, alpha=30, print_info = False)
    VHM.train_VHM(learning_ratio=0.01, bulk_train=False, max_d=[4, 5], child_weight=[0.1], gamma=[0.2], subsample=[0.8], lamda= [8], alpha=[3], print_info = True)
    VHM.grid_search_best_para()
    # a = 1
    # pass

