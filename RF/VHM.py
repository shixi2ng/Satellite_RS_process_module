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
import xgboost
from GEDI_toolbox.GEDI_main import GEDI_df
from RF.XGboost import *
import cupy as cp
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import traceback
from fontTools.tfmLib import PASSTHROUGH
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import shap
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest
from datetime import datetime
from matplotlib.colors import LogNorm, PowerNorm
import basic_function


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
    weights = 1 / (1 + np.exp((3 - y_true) * 1))
    grad = weights * np.sign(residual)
    hess = np.ones_like(y_true)
    return grad, hess

def weighted_mae_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    residual = np.abs(y_pred - y_true)
    weights = 1 / (1 + np.exp((3 - y_true) * 1))
    weighted_mae = np.sum(weights * residual) / np.sum(weights)
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


def XGBoost(xy_dataset, output_folder, paras, y_feature, data_index, n_est, print_info=True, loss_func=None, save_model=True):

    cmap = plt.cm.YlOrBr
    try:
        # Define the save model factor
        if isinstance(save_model, bool):
            save_model = save_model
        else:
            raise TypeError('Save model is not under the correct type')

        # Generate the result dictionary
        if os.path.isdir(output_folder):
            res_csv_folder = os.path.join(output_folder, 'res_csv\\')
            res_plot_folder = os.path.join(output_folder, 'res_plot\\')
            train_plot_folder = os.path.join(output_folder, 'train_plot\\')
            bf.create_folder(res_csv_folder)
            bf.create_folder(res_plot_folder)
            bf.create_folder(train_plot_folder)
            if save_model:
                res_model_folder = os.path.join(output_folder, 'res_model\\')
                bf.create_folder(res_model_folder)
        else:
            raise Exception('The output folder does not exist')

        # Define the loss func
        if loss_func is None:
            loss_func = 'rmse'
        elif loss_func in globals() and callable(globals()[loss_func]):
            loss_func = loss_func

        # Local the var
        para = paras[0]
        xfeature = paras[2]
        mode = paras[1]
        max_d_ = para['max_depth']
        min_child_ = para['min_child_weight']
        gamma_ = para['gamma']
        subs_ = para['subsample']
        tree_sample = para['colsample_bytree']
        alpha_ = para['alpha']
        lambda_ = para['lambda']

        # Output filename
        hyperpara_comb = f'{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lambda_)}_alpha{str(alpha_)}'
        ex_filename = os.path.join(res_csv_folder, f'res_{hyperpara_comb}.csv')

        # list for print
        original_train_y_list, predict_train_y_list = [], []
        original_test_y_list, predict_test_y_list = [], []
        train_index_list, test_index_list = [], []

        start_time = time.time()
        if print_info or not os.path.exists(ex_filename):

            x_train = xy_dataset[f'x_train_{str(mode)}']
            y_train = xy_dataset[f'y_train_{str(mode)}']
            x_test = xy_dataset[f'x_test_{str(mode)}']
            y_test = xy_dataset[f'y_test_{str(mode)}']
            train_index = data_index[f'train_{str(mode)}']
            test_index = data_index[f'test_{str(mode)}']

            measure_dic = {'train_score': [], 'test_score': [], 'train_mae': [], 'test_mae': [],
                           'train_mse': [], 'test_mse': [], 'train_rmse': [], 'test_rmse': [],
                           'train_r2': [], 'test_r2': [], 'OE': [], 'lr': [], 'rank': []}

            # Kfold
            fold_ = 0
            for x_train_, y_train_, x_test_, y_test_, train_index_, test_index_ in zip(x_train, y_train, x_test, y_test, train_index, test_index):

                # Get the x and y
                y_train_ori = copy.deepcopy(y_train_)
                y_test_ori = copy.deepcopy(y_test_)

                dtrain = xgb.DMatrix(x_train_, label=y_train_)
                dtest = xgb.DMatrix(x_test_, label=y_test_)
                evals_result = {}

                # 参数设定
                para['device'] = 'cuda:0'

                # 训练模型
                model = xgb.train(
                    params=para,
                    dtrain=dtrain,
                    num_boost_round=n_est,
                    obj=weighted_mae_obj,
                    custom_metric=weighted_mae_metric,
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
                plt.savefig(os.path.join(train_plot_folder, f'res_{hyperpara_comb}.png'), dpi=300)
                plt.close()
                fold_ += 1

                # 性能指标计算并记录
                measure_dic['train_score'].append(1)
                measure_dic['test_score'].append(0)

                measure_dic['train_mae'].append(metrics.mean_absolute_error(y_train_ori, y_train_pred))
                measure_dic['test_mae'].append(metrics.mean_absolute_error(y_test_ori, y_test_pred))
                measure_dic['train_rmse'].append(metrics.root_mean_squared_error(y_train_ori, y_train_pred))
                measure_dic['test_rmse'].append(metrics.root_mean_squared_error(y_test_ori, y_test_pred))
                measure_dic['train_mse'].append(metrics.mean_squared_error(y_train_ori, y_train_pred))
                measure_dic['test_mse'].append(metrics.mean_squared_error(y_test_ori, y_test_pred))
                measure_dic['train_r2'].append(metrics.r2_score(y_train_ori, y_train_pred))
                measure_dic['test_r2'].append(metrics.r2_score(y_test_ori, y_test_pred))
                measure_dic['lr'].append(para['eta'])
                measure_dic['OE'].append(model.best_iteration)

                # 特征重要性排序
                model.feature_names = xfeature
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

                if save_model:
                    model.save_model(os.path.join(res_model_folder, f'XGB_{hyperpara_comb}_KF{str(fold_)}.json'))

            for _ in measure_dic.keys():
                if _ == 'rank':
                    measure_dic[_].append(sort_by_average_position(measure_dic[_]))
                else:
                    measure_dic[_].append(np.mean(measure_dic[_]))

            df = pd.DataFrame(measure_dic)
            df.to_csv(ex_filename)
            con_model_f = True
        else:
            df = pd.read_csv(ex_filename)
            measure_dict = df.to_dict()
            con_model_f = False
        print(f'End time 4 XGboost mode:{str(mode)} dep:{str(max_d_)} minchild:{str(min_child_)} gamma:{str(gamma_)} subs:{str(subs_)} lambda:{str(lambda_)} alpha{str(alpha_)} Time:{str(time.time() - start_time)}')

        if print_info:
            train_test_y_df = {'ori': copy.deepcopy(original_test_y_list), 'pre': copy.deepcopy(predict_test_y_list), 'index': copy.deepcopy(test_index_list)}
            train_test_y_df['ori'].extend(original_train_y_list)
            train_test_y_df['pre'].extend(predict_train_y_list)
            train_test_y_df['index'].extend(train_index_list)
            train_test_y_df = pd.DataFrame(train_test_y_df)
            train_test_y_df.to_csv(os.path.join(res_plot_folder, f'fig_test_{hyperpara_comb}_KF{str(fold_)}.csv'))

            print('----------------------------------------------------------------------------------------------')
            print(f'X_feature:{str(xfeature)}, Y_feature:{str(y_feature)}, Max depth:{str(max_d_)}')
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

            qq = ax3.hist2d([_ - 0.1 for _ in predict_test_y_list], original_test_y_list, bins=50, range=[[2, 5], [2, 5]], cmap=plt.cm.YlOrBr, norm=PowerNorm(gamma=0.7))
            ax3.plot(np.linspace(0, 6, 6), np.linspace(0, 6, 6), c=(0, 0, 0), lw=1.5, zorder=3, ls='--')
            ax3.plot(np.linspace(0, 6, 6), np.linspace(0, 6.6, 6), c=(0, 0, 0), lw=1., zorder=3, ls='-.')
            ax3.plot(np.linspace(0, 6, 6), np.linspace(0, 5.4, 6), c=(0, 0, 0), lw=1., zorder=3, ls='-.')
            # ax3.plot(np.linspace(0, 6, 6), np.linspace(0.93, chl + 0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
            # ax3.plot(np.linspace(0, 6, 6), np.linspace(-0.93, chl - 0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
            # ax3.fill_between(np.linspace(0, chl, chl), np.linspace(-0.93, chl - 0.93, chl), np.linspace(+0.93, chl + 0.93, chl), color=(0.3, 0.3, 0.3), alpha=0.1)
            ax3.set_xlabel('Model predicted vegetation height/m', fontsize=12, fontweight='bold')
            ax3.set_ylabel('GEDI RH100/m', fontsize=12, fontweight='bold')
            ax3.text(2.1, 5 - 0.2, f"MAE={str(measure_dic['test_mae'][-1])[0: 4]}m", c=(0, 0, 0), fontsize=16, fontweight='bold')
            ax3.text(2.1, 5 - 0.5, f"RMSE={str(measure_dic['test_rmse'][-1])[0: 4]}m", c=(0, 0, 0), fontsize=16, fontweight='bold')
            ax3.text(2.1, 5 - 0.8, f"R$^2$=0.51", c=(0, 0, 0), fontsize=16, fontweight='bold')
            ax3.set_ylim(2, 5)
            ax3.set_xlim(2, 5)
            plt.savefig(os.path.join(res_plot_folder, f'fig_test_{hyperpara_comb}.png'), dpi=300)
            plt.close()

            plt.rcParams['font.family'] = ['Arial', 'SimHei']
            plt.rc('font', size=14)
            fig3, ax3 = plt.subplots(figsize=(5, 5), constrained_layout=True)

            qq = ax3.hist2d([_ - 0.1 for _ in predict_train_y_list], original_train_y_list, bins=50, range=[[2, 5], [2, 5]], cmap=plt.cm.YlOrBr, norm=PowerNorm(gamma=0.7))
            ax3.plot(np.linspace(0, 6, 6), np.linspace(0, 6, 6), c=(0, 0, 0), lw=1.5, zorder=3, ls='--')
            ax3.plot(np.linspace(0, 6, 6), np.linspace(0, 6.6, 6), c=(0, 0, 0), lw=1., zorder=3, ls='-.')
            ax3.plot(np.linspace(0, 6, 6), np.linspace(0, 5.4, 6), c=(0, 0, 0), lw=1., zorder=3, ls='-.')
            # ax3.plot(np.linspace(0, chl, chl), np.linspace(0.93, chl + 0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
            # ax3.plot(np.linspace(0, chl, chl), np.linspace(-0.93, chl - 0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
            # ax3.fill_between(np.linspace(0, chl, chl), np.linspace(-0.93, chl - 0.93, chl),
            #                  np.linspace(+0.93, chl + 0.93, chl),
            #                  color=(0.3, 0.3, 0.3), alpha=0.1)
            ax3.set_xlabel('Model predicted vegetation height/m', fontsize=12, fontweight='bold')
            ax3.set_ylabel('GEDI RH100/m', fontsize=12, fontweight='bold')
            ax3.text(2.1, 5 - 0.2, f"MAE={str(measure_dic['train_mae'][-1])[0: 4]}m", c=(0, 0, 0), fontsize=18, fontweight='bold')
            ax3.text(2.1, 5 - 0.5, f"RMSE={str(measure_dic['train_rmse'][-1])[0: 4]}m", c=(0, 0, 0), fontsize=18, fontweight='bold')
            ax3.text(2.1, 5 - 0.8, f"R$^2$=0.54", c=(0, 0, 0), fontsize=16, fontweight='bold')
            ax3.set_ylim(2, 5)
            ax3.set_xlim(2, 5)
            plt.savefig(os.path.join(res_plot_folder, f'fig_train_{hyperpara_comb}.png'), dpi=300)
            plt.close()

            # # Shap explainer
            explainer = shap.Explainer(model)
            shap_values = explainer(x_train_)

            xfeature = [_.replace('_focal', '') for _ in xfeature]
            xfeature = [_.replace('_area_average', '') for _ in xfeature]
            xfeature = [_.replace('noninun_', '') for _ in xfeature]
            xfeature = [_.replace('Denv_', '') for _ in xfeature]
            xfeature = [_.replace('Pheme_', '') for _ in xfeature]
            shap_values.feature_names = xfeature
            # # visualize the first prediction's explanation
            shap.plots.bar(shap_values, max_display=100, show=False, clustering_cutoff=0.5)
            plt.savefig(os.path.join(res_plot_folder, f'shap_v_{hyperpara_comb}.png'),
                        dpi=300)
            plt.close()

            shap.summary_plot(shap_values, x_train_, plot_type='bar', max_display=100, show=False,)
            plt.savefig(os.path.join(res_plot_folder, f'shap_sum_{hyperpara_comb}.png'),
                        dpi=300)
            plt.close()

            shap.plots.waterfall(shap_values[0], show=False)
            plt.savefig(os.path.join(res_plot_folder, f'shap_waterfall_{hyperpara_comb}.png'),
                        dpi=300)
            plt.close()

            plt.figure(figsize=(12, 10))
            shap.plots.beeswarm(shap_values, max_display=15, show=False)
            plt.savefig(os.path.join(res_plot_folder, f'shap_bees_{hyperpara_comb}.png'),
                        dpi=300,  bbox_inches='tight')
            plt.close()

            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

            # 构建 DataFrame
            shap_summary_df = pd.DataFrame({
                'Feature': shap_values.feature_names,
                'Mean_Abs_SHAP': mean_abs_shap
            })

            # 按重要性降序排列
            shap_summary_df = shap_summary_df.sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
            shap_summary_df.to_csv(os.path.join(res_plot_folder, f'shap_value_{hyperpara_comb}.csv'))

            # 获取 gain 类型的重要性
            importance_dict = model.get_score(importance_type='gain')

            # 转换为 DataFrame
            importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['gain'])
            importance_df.index.name = 'feature'
            importance_df.reset_index(inplace=True)

            # 计算百分比（按 gain 占总 gain 的比重）
            total_gain = importance_df['gain'].sum()
            importance_df['gain_percent'] = 100 * importance_df['gain'] / total_gain
            importance_df.to_csv(os.path.join(res_plot_folder, f'feature_importance_{hyperpara_comb}.csv'))

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

    def __init__(self, xfeature_vegh_csv):

        # Define the parameters
        self.xfeature_vegh_df_ = None
        self.split_factor = None
        self.xfeature_mode_list = None
        self.model_type = None
        self.model = None
        self.normalised_factor = False
        self.res_df = None

        # Define the var
        self.xy4model_dic = {}
        self.xy_oriindex_dic = {}
        self.hyperparameters = {}
        self.hyperparameters_df = {}
        self.feature_dic = {}
        self.lr = 0.
        self.est_rounds = 0.

        # Input the dataset
        if not os.path.exists(xfeature_vegh_csv):
            raise Exception('The GEDI linked csv does not exist')
        else:
            self.xfeature_vegh_df = pd.read_csv(xfeature_vegh_csv)
            # Give the overall index
            self.xfeature_vegh_df.reset_index(drop=False, inplace=True, names='ori_index')
            self.dataset_name = os.path.basename(xfeature_vegh_csv).split('.')[0]

        # Define the output folder and path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work_env = os.path.join(os.path.dirname(xfeature_vegh_csv), f'VHM_model\\VHM_{str(timestamp)}')
        bf.create_folder(self.work_env)

        self.log_filepath = os.path.join(self.work_env, 'logfile\\')
        bf.create_folder(self.log_filepath)
        
        # Define the hyperpara 4 model
        self._support_hyperpara = ['max_depth', 'min_child_weight', 'gamma', 'subsample', 'lamda', 'alpha', 'colsample_bytree']
        self._default_hyperpara = {'max_depth': [5], 'min_child_weight': [0.1], 'gamma': [0], 'subsample': [0.85],
                                   'colsample_bytree': [0.85], 'lamda': [10], 'alpha': [4],}
        self._default_params = {}
        self._default_lr = 0.1
        self.params = {'max_leaves': 0,'colsample_bylevel': 1}

        # Define the user defined para
        self._support_user_defined_para = ['split_factor', 'yfeature', 'xfeature_combine_mode', 'xfeature_list', 'normalised_factor', 'spatial_interpolation_type']
        self._support_spatial_interpolation_method = ['nearest_neighbor', 'focal', 'area_average']
        self._support_split_factor = ['kf', 'yearly', 'split', 'all']
        self._support_mode = [0, 1, 2, 3, 4, 5, 6]

    def save_log_file(func):
        def wrapper(self, *args, **kwargs):

            #########################################################################
            # Document the log file and para file
            # The difference between log file and para file is that the log file contains the information for each run/debug
            # While the para file only comprised of the parameter for the latest run/debug
            #########################################################################

            time_start = time.time()
            c_time = time.ctime()
            log_file = open(f"{self.log_filepath}log.txt", "a+")
            if os.path.exists(f"{self.log_filepath}para_file.txt"):
                para_file = open(f"{self.log_filepath}para_file.txt", "r+")
            else:
                para_file = open(f"{self.log_filepath}para_file.txt", "w+")
            error_inf = None

            para_txt_all = para_file.read()
            para_ori_txt = para_txt_all.split('#' * 70 + '\n')
            para_txt = para_txt_all.split('\n')
            contain_func = [txt for txt in para_txt if txt.startswith('Process Func:')]

            try:
                func(self, *args, **kwargs)
            except:
                error_inf = traceback.format_exc()
                print(error_inf)

            # Header for the log file
            log_temp = ['#' * 70 + '\n', f'Process Func: {func.__name__}\n', f'Start time: {c_time}\n',
                        f'End time: {time.ctime()}\n', f'Total processing time: {str(time.time() - time_start)}\n']

            # Create args and kwargs list
            args_f = 0
            args_list = ['*' * 25 + 'Arguments' + '*' * 25 + '\n']
            kwargs_list = []
            for i in args:
                args_list.extend([f"args{str(args_f)}:{str(i)}\n"])
            for k_key in kwargs.keys():
                kwargs_list.extend([f"{str(k_key)}:{str(kwargs[k_key])}\n"])
            para_temp = ['#' * 70 + '\n', f'Process Func: {func.__name__}\n', f'Start time: {c_time}\n',
                         f'End time: {time.ctime()}\n', f'Total processing time: {str(time.time() - time_start)}\n']
            para_temp.extend(args_list)
            para_temp.extend(kwargs_list)
            para_temp.append('#' * 70 + '\n')

            log_temp.extend(args_list)
            log_temp.extend(kwargs_list)
            log_file.writelines(log_temp)
            for func_key, func_processing_name in zip(['train', 'predict'], ['train model', 'predict height']):
                if func_key in func.__name__:
                    if error_inf is None:
                        log_file.writelines([f'Status: Finished {func_processing_name}!\n', '#' * 70 + '\n'])
                        metadata_line = [q for q in contain_func if func_key in q]
                        if len(metadata_line) == 0:
                            para_file.writelines(para_temp)
                            para_file.close()
                        elif len(metadata_line) == 1:
                            for para_ori_temp in para_ori_txt:
                                if para_ori_temp != '' and metadata_line[0] not in para_ori_temp:
                                    para_temp.extend(['#' * 70 + '\n', para_ori_temp, '#' * 70 + '\n'])
                                    para_file.close()
                                    para_file = open(f"{self.log_filepath}para_file.txt", "w+")
                                    para_file.writelines(para_temp)
                                    para_file.close()
                        elif len(metadata_line) > 1:
                            print('Code error! ')
                            sys.exit(-1)
                    else:
                        log_file.writelines([f'Status: Error in {func_processing_name}!\n', 'Error information:\n', error_inf + '\n', '#' * 70 + '\n'])

        return wrapper

    def _process_hyperparameters(self, **hyperparameters):

        # init hyper para and user-defined
        self.hyperparameters = copy.deepcopy(self._default_hyperpara)
        self.split_factor = 'kf'
        self.yfeature = 'Canopy Height (rh100)'
        self.xfeature_mode_list = copy.deepcopy(self._support_mode)
        self.xfeature_list = None
        self.normalised_factor = True
        self.spatial_interpolation_type = []
        for _ in self._support_spatial_interpolation_method:
            if True in [_ in __ for __ in self.xfeature_vegh_df.keys()]:
                self.spatial_interpolation_type.append(_)

        # Check the hyperparameters
        for hyperparameter in hyperparameters.keys():

            if hyperparameter in self._support_hyperpara:
                if isinstance(hyperparameters[hyperparameter], list):
                    self.hyperparameters[hyperparameter] = hyperparameters[hyperparameter]
                elif isinstance(hyperparameters[hyperparameter], (int, float, np.integer, np.float_)):
                    self.hyperparameters[hyperparameter] = [hyperparameters[hyperparameter]]
                else:
                    raise Exception(f'The hyperparameter {hyperparameter} is not under the support type')

            elif hyperparameter in self._support_user_defined_para:

                if hyperparameter == 'split_factor':
                    split_factor = hyperparameters[hyperparameter]
                    # Define the split factor
                    if not isinstance(split_factor, str):
                        raise Exception('The split_factor should be a str')
                    elif split_factor not in self._support_split_factor:
                        raise Exception(f'The split_factor {split_factor} is not supported')
                    else:
                        self.split_factor = split_factor

                elif hyperparameter == 'yfeature':
                    yfeature= hyperparameters[hyperparameter]
                    if yfeature not in list(self.xfeature_vegh_df.keys()):
                        raise Exception('The yfeature is not embedded in the dataframe')
                    else:
                        self.yfeature = yfeature

                elif hyperparameter == 'xfeature_combine_mode':
                    xfeature_combine_mode = hyperparameters[hyperparameter]
                    if not isinstance(xfeature_combine_mode, (list, int)):
                        raise Exception('The xfeature_combine_mode list should be a list')
                    elif isinstance(xfeature_combine_mode, list):
                        self.xfeature_mode_list = [_ for _ in xfeature_combine_mode if _ in self._support_mode]
                        if len(self.xfeature_mode_list) == 0:
                            raise Exception('The mode list is not supported!')
                    else:
                        self.xfeature_mode_list = [xfeature_combine_mode]

                elif hyperparameter == 'xfeature_list':
                    xfeature_list = hyperparameters[hyperparameter]
                    if not isinstance(xfeature_list, list):
                        raise Exception('The xfeature_list should be a list')
                    else:
                        self.xfeature_list = xfeature_list

                elif hyperparameter == 'normalised_factor':
                    normalised_factor = hyperparameters[hyperparameter]
                    if not isinstance(normalised_factor, bool):
                        raise Exception('The normalised factor should be a boolean')
                    else:
                        self.normalised_factor = normalised_factor

                elif hyperparameter == 'spatial_interpolation_type':
                    spatial_interpolation_type = hyperparameters[hyperparameter]
                    if isinstance(spatial_interpolation_type, list):
                        for _ in spatial_interpolation_type:
                            if _ in self._support_spatial_interpolation_method and True in [_ in __ for __ in self.xfeature_vegh_df.keys()]:
                                self.spatial_interpolation_type.append(_)
                        if len(self.spatial_interpolation_type) == 0:
                            raise Exception('The spatial_interpolation_type should be supported')
                    else:
                        raise Exception('The spatial_interpolation_type should be a list type')

            else:
                raise Exception(f'The hyperparameter or user-defined para {str(hyperparameter)} is not supported')

    def _process_dataset(self, train_predict_factor, print_dis=False):

        # Define
        if not isinstance(train_predict_factor, str) or train_predict_factor not in ['train', 'predict']:
            raise Exception('The split_factor should be a supported type')
        else:
            train_factor = True if train_predict_factor == 'train' else False

        #############################################################
        # Process the dataset
        # Normalised dataset
        if self.normalised_factor:
            for key in self.xfeature_vegh_df.keys():
                # Remove 0
                if 'Denv_AGB' in key:
                    self.xfeature_vegh_df.loc[self.xfeature_vegh_df[key] == 0, key] = np.nan

                if 'ratio_gedi_fast-growing' in key:
                    arr = np.array(self.xfeature_vegh_df[key]).reshape(-1, 1)
                    arr = np.log(arr + 1)
                    self.xfeature_vegh_df[key] = arr

                if 'reliability' not in key:
                    if 'OSAVI' in key:
                        pass
                    elif 'vi' in key or 'VI' in key or 'GREENESS' in key:
                        pass
                    elif 'Pheme' in key and ('SOS' in key or 'peak_doy' in key or 'trough_doy' in key or 'EOS' in key or 'EOM' in key):
                        self.xfeature_vegh_df[key] = self.xfeature_vegh_df[key] / 365.25
                    elif 'Pheme' in key and ('DR' in key or 'GR' in key):
                        self.xfeature_vegh_df[key] = np.arctan(self.xfeature_vegh_df[key]) / (np.pi / 4)
                else:
                    pass

                # if 'reliability' not in key and 'Denv' in key and 'ratio' not in key:
                #     # pt = PowerTransformer(method = 'yeo-johnson', standardize=True)
                #     arr = np.array(self.xfeature_vegh_df[key]).reshape(-1, 1)
                #     arr = np.log(arr + 1)
                #     self.xfeature_vegh_df[key] = arr

        # Process the x feature
        # Define the index
        for spatial_method in self.spatial_interpolation_type:
            index_feature = [_ for _ in self.xfeature_vegh_df.keys() if 'noninun' in _ and spatial_method in _ and 'reliability' not in _]
            pheindex_feature = [_ for _ in self.xfeature_vegh_df.keys() if 'Pheme' in _ and spatial_method in _ and 'reliability' not in _]
            denv_feature = [_ for _ in self.xfeature_vegh_df.keys() if 'Denv' in _ and spatial_method in _ and 'reliability' not in _ and 'RHU' not in _ and 'PRE' not in _]

            if self.xfeature_list is not None:
                self.feature_dic[f'custom_feature_{spatial_method}'] = self.xfeature_list
            else:
                # Generate feature for different modes
                for mode in self.xfeature_mode_list:
                    if mode == 0:
                        feature_list = copy.copy(index_feature)
                    elif mode == 1:
                        feature_list = copy.copy(pheindex_feature)
                    elif mode == 2:
                        feature_list = copy.copy(denv_feature)
                    elif mode == 3:
                        feature_list = copy.copy(index_feature)
                        feature_list.extend(pheindex_feature)
                        feature_list.extend(denv_feature)
                    elif mode == 4:
                        feature_list = copy.copy(index_feature)
                        feature_list.extend(pheindex_feature)
                    elif mode == 5:
                        feature_list = copy.copy(pheindex_feature)
                        feature_list.extend(denv_feature)
                    elif mode == 6:
                        feature_list = copy.copy(index_feature)
                        feature_list.extend(denv_feature)
                    else:
                        raise Exception('Not support mode!')
                    self.feature_dic[f'mode{str(mode)}_{spatial_method}'] = feature_list
            
        # Generate the output folder
        input_ds_folder = os.path.join(self.work_env, f'VHM_input\\')
        bf.create_folder(input_ds_folder)

        # Generate the train test dataset
        if train_factor is True:
            for feature_ in self.feature_dic.keys():
                spatial_method = [_ for _ in self.spatial_interpolation_type if _ in feature_][0]

                # Generate empty dataset
                for xy_set in ['x_train', 'y_train', 'x_test', 'y_test']:
                    self.xy4model_dic[f'{xy_set}_{feature_}'] = []
                self.xy_oriindex_dic[f'train_{feature_}'] = []
                self.xy_oriindex_dic[f'test_{feature_}'] = []

                # Preselect the dataset
                self.xfeature_vegh_df['DOY'] = (self.xfeature_vegh_df['Date'] % 1000) / 365.25
                if f'VegType_{spatial_method}' in self.xfeature_vegh_df.keys():
                    self.xfeature_vegh_df = self.xfeature_vegh_df.loc[self.xfeature_vegh_df[f'VegType_{spatial_method}'] == 3]
                self.xfeature_vegh_df = self.xfeature_vegh_df.loc[self.xfeature_vegh_df[self.yfeature] <= 5.5]
                self.xfeature_vegh_df = self.xfeature_vegh_df.loc[self.xfeature_vegh_df[self.yfeature] > 2]
                # self.xfeature_vegh_df = self.xfeature_vegh_df.loc[self.xfeature_vegh_df['Landsat water rate'] < 0.1]
                self.xfeature_vegh_df = self.xfeature_vegh_df.replace(np.inf, np.nan)
                self.xfeature_vegh_df = self.xfeature_vegh_df.replace(-np.inf, np.nan)

                # Data isolation
                if os.path.exists('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\data_isolation.csv'):
                    df_exclude = pd.read_csv('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\data_isolation.csv')
                    df_exclude['per'] = np.abs((df_exclude['pre'] - df_exclude['ori']) / df_exclude['ori'])
                    df_exclude = df_exclude[df_exclude['per'] > 0.25]['index']
                    self.xfeature_vegh_df = self.xfeature_vegh_df[~self.xfeature_vegh_df['ori_index'].isin(df_exclude)].reset_index(drop=True)

                # Drop na
                self.xfeature_vegh_df_ = self.xfeature_vegh_df.dropna(subset=self.feature_dic[feature_]).reset_index(drop=True)

                iso = IsolationForest(contamination=0.01, random_state=42)
                yhat = iso.fit_predict(self.xfeature_vegh_df_[self.feature_dic[feature_]])
                self.xfeature_vegh_df_ = self.xfeature_vegh_df_[yhat != -1].reset_index(drop=True)
                # self.xfeature_vegh_df_['Canopy Height (rh100)'] = self.xfeature_vegh_df_['Canopy Height (rh100)'] * 10

                print(f'Mode:{str(mode)}, The number of samples:{str(self.xfeature_vegh_df_.shape[0])}, drop ratio:{str(1 - self.xfeature_vegh_df_.shape[0] / self.xfeature_vegh_df.shape[0])}')

                # Construct the train test dataset
                if self.split_factor == 'yearly':
                    for year in [2019, 2020, 2021, 2022]:
                        x_train = self.xfeature_vegh_df_[self.xfeature_vegh_df_['Date'] // 1000 != year][self.feature_dic[feature_]]
                        y_train = self.xfeature_vegh_df_[self.xfeature_vegh_df_['Date'] // 1000 != year][self.yfeature]
                        x_test = self.xfeature_vegh_df_[self.xfeature_vegh_df_['Date'] // 1000 == year][self.feature_dic[feature_]]
                        y_test = self.xfeature_vegh_df_[self.xfeature_vegh_df_['Date'] // 1000 == year][self.yfeature]
                        train_index_ = self.xfeature_vegh_df_.loc[x_train.index, 'ori_index']
                        test_index_ = self.xfeature_vegh_df_.loc[x_test.index, 'ori_index']

                        if x_train.shape[0] != 0 and x_test.shape[0] != 0:
                            self.xy4model_dic[f'x_train_{feature_}'].append(np.array(x_train))
                            self.xy4model_dic[f'y_train_{feature_}'].append(np.array(y_train))
                            self.xy4model_dic[f'x_test_{feature_}'].append(np.array(x_test))
                            self.xy4model_dic[f'y_test_{feature_}'].append(np.array(y_test))
                            self.xy_oriindex_dic[f'train_{feature_}'].append(np.array(train_index_))
                            self.xy_oriindex_dic[f'test_{feature_}'].append(np.array(test_index_))

                elif self.split_factor == 'kf':
                    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                    y_binned = pd.qcut(self.xfeature_vegh_df_[self.yfeature], q=20, labels=False)
                    for train_index, test_index in kf.split(self.xfeature_vegh_df_[self.feature_dic[feature_]], y_binned):
                        x_train_ = self.xfeature_vegh_df_.loc[train_index, self.feature_dic[feature_]]
                        y_train_ = self.xfeature_vegh_df_.loc[train_index, self.yfeature]
                        x_test_ = self.xfeature_vegh_df_.loc[test_index, self.feature_dic[feature_]]
                        y_test_ = self.xfeature_vegh_df_.loc[test_index, self.yfeature]
                        train_index_ = self.xfeature_vegh_df_.loc[train_index, 'ori_index']
                        test_index_ = self.xfeature_vegh_df_.loc[test_index, 'ori_index']
                        if x_train_.shape[0] != 0 and x_test_.shape[0] != 0:
                            self.xy4model_dic[f'x_train_{feature_}'].append(np.array(x_train_))
                            self.xy4model_dic[f'y_train_{feature_}'].append(np.array(y_train_))
                            self.xy4model_dic[f'x_test_{feature_}'].append(np.array(x_test_))
                            self.xy4model_dic[f'y_test_{feature_}'].append(np.array(y_test_))
                            self.xy_oriindex_dic[f'train_{feature_}'].append(np.array(train_index_))
                            self.xy_oriindex_dic[f'test_{feature_}'].append(np.array(test_index_))

                elif self.split_factor == 'split':
                    x_train_, x_test_, y_train_, y_test_ = train_test_split(self.xfeature_vegh_df_[self.feature_dic[feature_]], self.xfeature_vegh_df_[self.yfeature], test_size=0.2, random_state=41)
                    train_index_ = self.xfeature_vegh_df_.loc[x_train_.index, 'ori_index']
                    test_index_ =  self.xfeature_vegh_df_.loc[x_test_.index, 'ori_index']

                    if x_train_.shape[0] != 0 and x_test_.shape[0] != 0:
                        self.xy4model_dic[f'x_train_{feature_}'].append(np.array(x_train_))
                        self.xy4model_dic[f'y_train_{feature_}'].append(np.array(y_train_))
                        self.xy4model_dic[f'x_test_{feature_}'].append(np.array(x_test_))
                        self.xy4model_dic[f'y_test_{feature_}'].append(np.array(y_test_))
                        self.xy_oriindex_dic[f'train_{feature_}'].append(np.array(train_index_))
                        self.xy_oriindex_dic[f'test_{feature_}'].append(np.array(test_index_))

                elif self.split_factor == 'all':

                    # Set all to train & test
                    x_train = self.xfeature_vegh_df_[self.feature_dic[feature_]]
                    y_train = self.xfeature_vegh_df_[self.yfeature]
                    train_index_ = self.xfeature_vegh_df_.loc[:, 'ori_index']
                    test_index_ = self.xfeature_vegh_df_.loc[:, 'ori_index']

                    if x_train.shape[0] != 0:
                        self.xy4model_dic[f'x_train_{feature_}'].append(x_train)
                        self.xy4model_dic[f'y_train_{feature_}'].append(y_train)
                        self.xy4model_dic[f'x_test_{feature_}'].append(x_train)
                        self.xy4model_dic[f'y_test_{feature_}'].append(y_train)
                        self.xy_oriindex_dic[f'train_{feature_}'].append(np.array(train_index_))
                        self.xy_oriindex_dic[f'test_{feature_}'].append(np.array(test_index_))

                else:
                    raise Exception('The split factor is not supported!')

                # Output the dataset
                output_mode_folder = os.path.join(input_ds_folder, f'{feature_}\\')
                bf.create_folder(output_mode_folder)
                for split_ in range(len(self.xy4model_dic[f'x_train_{feature_}'])):
                    df_cs = pd.DataFrame(self.xy4model_dic[f'x_train_{feature_}'][split_], columns=self.feature_dic[feature_])
                    df_cs.to_csv(os.path.join(input_ds_folder, f'{feature_}\\input_ds.csv'))

        else:

            # Generate empty dataset
            self.xy4model_dic[f'x_predict'] = []
            self.xy_oriindex_dic[f'x_predict'] = []

            for spatial_method in self.spatial_interpolation_type:

                if f'VegType_{spatial_method}' in self.xfeature_vegh_df.keys():
                    self.xfeature_vegh_df = self.xfeature_vegh_df.loc[self.xfeature_vegh_df[f'VegType_{spatial_method}'] == 3]
                    self.xfeature_vegh_df = self.xfeature_vegh_df.replace(np.inf, np.nan)
                self.xfeature_vegh_df = self.xfeature_vegh_df.replace(-np.inf, np.nan)

                # Drop na
                self.xfeature_vegh_df_ = self.xfeature_vegh_df.dropna(subset=self.feature_dic[f'mode3_{spatial_method}']).reset_index(drop=True)

                # Set all to train & test
                x_predict = self.xfeature_vegh_df_[self.feature_dic[f'mode3_{spatial_method}']]
                predict_index_ = self.xfeature_vegh_df_.loc[:, 'ori_index']

                if x_predict.shape[0] != 0 :
                    self.xy4model_dic[f'x_predict'] = x_predict
                    self.xy_oriindex_dic[f'x_predict'] = predict_index_

                # Output the dataset
                output_mode_folder = os.path.join(input_ds_folder, f'mode3_{spatial_method}\\')
                bf.create_folder(output_mode_folder)
                df_cs = pd.DataFrame(self.xy4model_dic[f'x_predict'], columns=self.feature_dic[f'mode3_{spatial_method}'])
                df_cs.to_csv(os.path.join(input_ds_folder, f'mode3_{spatial_method}\\input_ds.csv'))

            # Print the distribution
            if print_dis:
                distplot_folder = os.path.join(input_ds_folder, f'dist_plot\\')
                bf.create_folder(distplot_folder)
                self.xfeature_vegh_df_.to_csv(os.path.join(input_ds_folder, f'{str(mode)}.csv'))
                for feature_list_ in feature_list:
                    plt.hist(self.xfeature_vegh_df_[feature_list_], bins=100)
                    plt.savefig(os.path.join(distplot_folder, f'{feature_list_}.png'), dpi=300)
                    plt.close()

    @save_log_file
    def train_VHM(self, model_type, lr, est_rounds = None, bulk_train=True, print_info=False, **hyperparameters):

        # Process the dataset
        self._process_hyperparameters(**hyperparameters)
        self._process_dataset('train', print_dis=print_info)

        # Create outputfolder
        outputpath = os.path.join(self.work_env, f'VHM_res\\')
        bf.create_folder(outputpath)

        # Check the veg type
        if isinstance(model_type, str) and model_type in ['XGB', 'RFR']:
            self.model_type = model_type
            if self.model_type == 'XGB':
                self.model = XGBoost
            elif self.model_type == 'RFR':
                self.model = RandomForest
        else:
            raise Exception('The model type should be either XGB or RFR')

        # Determine the lr and nest
        if isinstance(lr, (float, np.float_)) and lr > 0:
            self.lr = lr
        else:
            raise Exception('Please set the learning ratio to a positive float number')

        # Determine the est_roundst
        if est_rounds is None:
            self.est_rounds = int(50 / self.lr)
        elif isinstance(est_rounds, (int, np.int_)) and est_rounds > 0:
            self.est_rounds = est_rounds
        else:
            raise Exception('Please set the est_rounds to a positive int number')

        # self.hyperparameters_list = {'max_d': [], 'child_weight': [], 'gamma': [], 'subsample': [], 'lamda': [], 'alpha': [], 'mode': [], 'x_feature': [], 'lr': , 'n_est'}
        exist_hyperparameters = []
        self.hyperparameters_df = pd.DataFrame(columns=['max_depth', 'max_leaves', 'eta', 'lambda', 'alpha', 'colsample_bylevel',
                                                        'min_child_weight', 'gamma', 'subsample', 'colsample_bytree', 'xfeature', 'mode'])

        for max_d_, min_child_,gamma_, subs_, col_subs_, lamda_, alpha_, mode_ in itertools.product(self.hyperparameters['max_depth'],
                                                                                                    self.hyperparameters['min_child_weight'],
                                                                                                    self.hyperparameters['gamma'],
                                                                                                    self.hyperparameters['subsample'],
                                                                                                    self.hyperparameters['colsample_bytree'],
                                                                                                    self.hyperparameters['lamda'],
                                                                                                    self.hyperparameters['alpha'],
                                                                                                    list(self.feature_dic.keys())):
            paras = {'max_depth': max_d_, 'max_leaves': 0, 'eta': self.lr, 'lambda': lamda_, 'alpha': alpha_, 'colsample_bylevel': 1,
                     'min_child_weight': min_child_, 'gamma': gamma_, 'subsample': subs_, 'colsample_bytree': col_subs_}

            exist_hyperparameters.append([{'max_depth': max_d_, 'max_leaves': 0, 'eta': self.lr, 'lambda': lamda_, 'alpha': alpha_, 'colsample_bylevel': 1,
                                           'min_child_weight': min_child_, 'gamma': gamma_, 'subsample': subs_, 'colsample_bytree': col_subs_}, mode_, self.feature_dic[mode_]])
            paras['xfeature'] = [self.feature_dic[mode_]]
            paras['mode'] = mode_
            self.hyperparameters_df = pd.concat([self.hyperparameters_df, pd.DataFrame(paras)], ignore_index=True)

        if self.hyperparameters_df.shape[0] != 0:
            if bulk_train:
                with ProcessPoolExecutor(max_workers=int(os.cpu_count() * RF.configuration['multiprocess_ratio'])) as exe:
                    res = exe.map(self.model, repeat(self.xy4model_dic), repeat(outputpath), exist_hyperparameters,
                                  repeat(self.yfeature), repeat(self.xy_oriindex_dic), repeat(self.est_rounds), repeat(print_info))

            else:
                res = []
                for para in exist_hyperparameters:
                    res.append(self.model(self.xy4model_dic, outputpath, para,self.yfeature,
                                          self.xy_oriindex_dic, self.est_rounds, print_info=print_info))

            res = list(res)
            res_df = None
            for _ in res:
                if res_df is not None:
                    res_df = pd.concat([res_df, _], axis=0)
                else:
                    res_df = _

            self.res_df = pd.concat([self.hyperparameters_df, res_df.reset_index(drop=True)], axis=1)
            self.res_df.to_csv(os.path.join(self.work_env, 'res.csv'))

    @save_log_file
    def predict_VHM(self, outputfolder, model_file, print_info=False):

        # Check the output folder

        # Process the dataset
        self._process_hyperparameters()
        self._process_dataset('predict', print_dis=print_info)

        # Check the model file
        if not os.path.exists(model_file):
            raise IOError('The model file is not correct!')
        else:
            if os.path.basename(model_file).startswith('XGB'):
                self.model = XGBoost
            elif os.path.basename(model_file).startswith('RFR'):
                self.model = RandomForest
            else:
                raise Exception('The model file is not correct!')

        basic_function.create_folder(outputfolder)

        # Check the interpolation method
        if len(self.spatial_interpolation_type) != 1:
            raise Exception('More than one spatial interpolation type is imported!')
        else:
            spatial_interpolate_ = self.spatial_interpolation_type[0]

        # Load model
        if self.model == XGBoost:
            model = xgb.Booster()
            model.load_model(model_file)

            # Model feature
            model_feature = model.feature_names
            model_spatial_method = []
            for _ in self._support_spatial_interpolation_method:
                if True in [_ in __ for __ in model_feature]:
                    model_spatial_method.append(_)
            if len(model_spatial_method) != 1:
                raise Exception('The model feature is not correct!')
            else:
                model_spatial_method = model_spatial_method[0]

            # Get the feature list
            ds_feature = list(self.xy4model_dic['x_predict'])
            if spatial_interpolate_ == model_spatial_method:
                if False in [_ in ds_feature for _ in model_feature]:
                    raise Exception('The predict feature is not consistent with the model feature')
            else:
                model_feature = [_.replace(model_spatial_method, spatial_interpolate_) for _ in model_feature]
                if False in [_ in ds_feature for _ in model_feature]:
                    raise Exception('The predict feature is not consistent with the model feature')
                else:
                    model.feature_names = model_feature
            y = model.predict(xgboost.DMatrix(self.xy4model_dic['x_predict']))
            self.xfeature_vegh_df_['Vegh'] = y
            self.xfeature_vegh_df_.to_csv(os.path.join(outputfolder, f'{self.dataset_name}_predict.csv'), index=False)

    def analyse_model(self, mode_, learning_ratio=0.1, **hyperparameters):

        # Check the dataset
        if len(list(self.xy4model_dic.keys())) == 0:
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

        XGBoost(self.xy4model_dic, output_folder, mode_, self.hyperparameters['max_d'], self.hyperparameters['child_weight'],
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
            plt.savefig(os.path.join(self.work_env, f'\\VHM_hyperpara_plot\\{keys}.png'), dpi=300)
            plt.close()


if __name__ == '__main__':

    # bf.merge_csv_files("G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\GEDI_link_RS\\", 'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_GEDI_2019_2023_for_model_high_quality.csv',
    #                    merge_keys = ['Quality Flag', 'Canopy Elevation (m)', 'index', 'EPSG_lat', 'Elevation (m)', 'Beam', 'Landsat water rate', 'Latitude', 'Shot Number', 'Sensitivity', 'EPSG_lon', 'Tandem-X DEM', 'Longitude', 'Degrade Flag', 'Date', 'RH 98', 'RH 25', 'Urban rate', 'Canopy Height (rh100)', 'Leaf off flag'])
    #
    VHM_ = VHM('G:\A_GEDI_Floodplain_vegh\GEDI_MYR\L2_vegh\\floodplain_GEDI_2019_2023_for_model_high_quality.csv')
    VHM_.train_VHM('XGB', 0.01, bulk_train=False, split_factor = 'yearly', xfeature_combine_mode=[3],  max_depth=[4], min_child_weight=[3], gamma=[1.5], subsample=[0.7], colsample_bytree=[0.7], lamda= [50], alpha=[10], est_rounds=1350, print_info=True)


    # bf.merge_csv_files("G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\GEDI_link_RS\\", 'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_GEDI_2019_2023_for_model_high_quality.csv',
    #                    merge_keys = ['Quality Flag', 'Canopy Elevation (m)', 'index', 'EPSG_lat', 'Elevation (m)', 'Beam', 'Landsat water rate', 'Latitude', 'Shot Number', 'Sensitivity', 'EPSG_lon', 'Tandem-X DEM', 'Longitude', 'Degrade Flag', 'Date', 'RH 98', 'RH 25', 'Urban rate', 'Canopy Height (rh100)', 'Leaf off flag'])
    #
    # VHM.grid_search_best_para()
    # a = 1
    # pass

