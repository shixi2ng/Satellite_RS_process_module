import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import basic_function as bf
from GEDI_toolbox.GEDI_main import GEDI_list
import copy
from sklearn import metrics
import os


def XGboost_regressor_params_validate(x_dataset: np.ndarray, y_dataset: np.ndarray, parameters: list, itr_round:int, fold_num:int):

    dfull = xgb.DMatrix(x_dataset, y_dataset)
    cv_res1 = xgb.cv(parameters, dfull, num_boost_round=itr_round, nfold=fold_num, metrics='rmse')

    fig, ax = plt.subplots(figsize=(15,8), constrained_layout=True)
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

    # cur_dic = {}
    # cur_dic[f'test3_rmse'] = []
    # cur_dic[f'test1_rmse'] = []
    # cur_dic[f'test2_rmse'] = []
    # cur_dic[f'test_rmse'] = []
    # cur_dic[f'train3_rmse'] = []
    # cur_dic[f'train1_rmse'] = []
    # cur_dic[f'train2_rmse'] = []
    # cur_dic[f'train_rmse'] = []
    # for max_d in range(1, 50):
    #     _ = bf.file_filter('G:\\A_veg\\S2_all\\GEDI_v3\\XGB\\change2\\', ['_nes300_', f'_maxd{str(max_d)}_'], and_or_factor='and', subfolder_detection=False)
    #     if len(_) == 0:
    #         cur_dic[f'test3_rmse'].append(np.nan)
    #         cur_dic[f'test2_rmse'].append(np.nan)
    #         cur_dic[f'test1_rmse'].append(np.nan)
    #         cur_dic[f'test_rmse'].append(np.nan)
    #         cur_dic[f'train3_rmse'].append(np.nan)
    #         cur_dic[f'train2_rmse'].append(np.nan)
    #         cur_dic[f'train1_rmse'].append(np.nan)
    #         cur_dic[f'train_rmse'].append(np.nan)
    #     else:
    #         pd_temp = pd.read_csv(_[0])
    #         cur_dic[f'test3_rmse'].append(pd_temp.loc[4, 'Test_index_at_RMSE'])
    #         cur_dic[f'test2_rmse'].append(pd_temp.loc[4, 'Test_index_RMSE'])
    #         cur_dic[f'test1_rmse'].append(pd_temp.loc[4, 'Test_index_phe_RMSE'])
    #         cur_dic[f'test_rmse'].append(pd_temp.loc[4, 'Test_index_phe_at_RMSE'])
    #         cur_dic[f'train3_rmse'].append(pd_temp.loc[4, 'Train_index_at_RMSE'])
    #         cur_dic[f'train2_rmse'].append(pd_temp.loc[4, 'Train_index_RMSE'])
    #         cur_dic[f'train1_rmse'].append(pd_temp.loc[4, 'Train_index_phe_RMSE'])
    #         cur_dic[f'train_rmse'].append(pd_temp.loc[4, 'Train_index_phe_at_RMSE'])
    #
    # fig4, ax4 = plt.subplots(figsize=(12, 8), constrained_layout=True)
    # ax4.plot(np.linspace(1, 49, 49), cur_dic[f'test3_rmse'], label='Test_index_at_RMSE')
    # ax4.plot(np.linspace(1, 49, 49), cur_dic[f'test2_rmse'], label='Test_index_RMSE')
    # ax4.plot(np.linspace(1, 49, 49), cur_dic[f'test1_rmse'], label='Test_index_phe_RMSE')
    # ax4.plot(np.linspace(1, 49, 49), cur_dic[f'test_rmse'], label='Test_index_phe_at_RMSE')
    # ax4.plot(np.linspace(1, 49, 49), cur_dic[f'train_rmse'], label='Train_index_RMSE')
    # ax4.plot(np.linspace(1, 49, 49), cur_dic[f'train1_rmse'], label='Train_index_phe_RMSE')
    # ax4.plot(np.linspace(1, 49, 49), cur_dic[f'train2_rmse'], label='Train_index_phe_at_RMSE')
    # ax4.plot(np.linspace(1, 49, 49), cur_dic[f'train2_rmse'], label='Train_index_at_RMSE')
    # plt.legend()
    # plt.show()

    gedi_ds = GEDI_list('G:\\A_veg\\S2_all\\GEDI_v3\\floodplain_2020_high_quality_merged.csv')
    pre_df = pd.read_csv('G:\\A_veg\\S2_all\\20220704\\all_merged2.csv')

    for hei_limit in range(4, 15):
        for mode in range(3):

            gedi_df = copy.copy(gedi_ds.GEDI_df)
            gedi_df = gedi_df.dropna()
            pre_df_temp = copy.copy(pre_df.dropna())
            index_list = ['S2_OSAVI_20m_noninun_linear_interpolation', 'S2_MNDWI_linear_interpolation',
                          'S2_B8A_noninun_linear_interpolation', 'S2_B9_noninun_linear_interpolation',
                          'S2_B7_noninun_linear_interpolation', 'S2_B6_noninun_linear_interpolation',
                          'S2_B5_noninun_linear_interpolation',
                          'S2_B4_noninun_linear_interpolation', 'S2_B3_noninun_linear_interpolation',
                          'S2_B2_noninun_linear_interpolation', ]
            index_list2 = ['S2_OSAVI_20m_noninun_linear_interpolation',
                           'S2_B8A_noninun_linear_interpolation', 'S2_B9_noninun_linear_interpolation',
                           'S2_B7_noninun_linear_interpolation', 'S2_B6_noninun_linear_interpolation',
                           'S2_B5_noninun_linear_interpolation',
                           'S2_B4_noninun_linear_interpolation', 'S2_B3_noninun_linear_interpolation',
                           'S2_B2_noninun_linear_interpolation', ]
            phe_list = ['S2phemetric_DR', 'S2phemetric_DR2', 'S2phemetric_EOS', 'S2phemetric_GR', 'S2phemetric_peak_doy', 'S2phemetric_peak_vi', 'S2phemetric_SOS', 'S2phemetric_trough_vi',]
            at_list = ['S2phemetric_static_TEMP', 'S2phemetric_static_DPAR', 'S2phemetric_peak_TEMP', 'S2phemetric_peak_DPAR', 'S2_accumulated_TEMP_relative']
            gedi_df = gedi_df.loc[gedi_df['Canopy Height (rh100)'] < hei_limit]
            gedi_df['S2_OSAVI_20m_noninun_linear_interpolation'] = (gedi_df['S2_OSAVI_20m_noninun_linear_interpolation'] - 32768) / 10000
            gedi_df['S2_NDVI_20m_noninun_linear_interpolation'] = (gedi_df['S2_NDVI_20m_noninun_linear_interpolation'] - 32768) / 10000
            gedi_df['S2_MNDWI_linear_interpolation'] = (gedi_df['S2_MNDWI_linear_interpolation'] - 32768) / 10000
    #
            res_dic = {}

            if mode == 0:
                temp = copy.copy(index_list)
                temp.extend(phe_list)
                temp.extend(at_list)

                for q, p in zip(['OSAVI_20m_noninun', 'MNDWI', 'B8A_noninun', 'B9_noninun', 'B7_noninun', 'B6_noninun', 'B5_noninun', 'B4_noninun', 'B3_noninun', 'B2_noninun', 'DR', 'DR2', 'EOS', 'GR', 'peak_doy', 'peak_vi', 'SOS', 'trough_vi', 'static_TEMP', 'static_DPAR', 'peak_TEMP', 'peak_DPAR', 'TEMP_relative'], temp):
                    pre_df_temp.rename(columns={q: p}, inplace=True)
                pre_df_temp['S2_MNDWI_linear_interpolation'] = (pre_df_temp['S2_MNDWI_linear_interpolation'] - 32768) / 10000

            elif mode == 1:
                temp = copy.copy(index_list)
                temp.extend(phe_list)

                for q, p in zip(['OSAVI_20m_noninun', 'MNDWI', 'B8A_noninun', 'B9_noninun', 'B7_noninun', 'B6_noninun',
                                 'B5_noninun', 'B4_noninun', 'B3_noninun', 'B2_noninun', 'DR', 'DR2', 'EOS', 'GR',
                                 'peak_doy', 'peak_vi', 'SOS', 'trough_vi'], temp):
                    pre_df_temp.rename(columns={q: p}, inplace=True)
                pre_df_temp['S2_MNDWI_linear_interpolation'] = (pre_df_temp['S2_MNDWI_linear_interpolation'] - 32768) / 10000

            elif mode == 2:

                temp = copy.copy(index_list2)
                temp.extend(phe_list)
                temp.extend(['S2phemetric_static_TEMP', 'S2phemetric_static_DPAR', 'S2phemetric_peak_TEMP', 'S2phemetric_peak_DPAR'])

                for q, p in zip(['OSAVI_20m_noninun',  'B8A_noninun', 'B9_noninun', 'B7_noninun', 'B6_noninun', 'B5_noninun', 'B4_noninun', 'B3_noninun', 'B2_noninun', 'DR', 'DR2', 'EOS', 'GR', 'peak_doy', 'peak_vi', 'SOS', 'trough_vi', 'static_TEMP', 'static_DPAR', 'peak_TEMP', 'peak_DPAR', ], temp):
                    pre_df_temp.rename(columns={q: p}, inplace=True)

            pre_df_temp['S2_OSAVI_20m_noninun_linear_interpolation'] = (pre_df_temp['S2_OSAVI_20m_noninun_linear_interpolation'] - 32768) / 10000


            x_train = gedi_df[temp]
            y_train = gedi_df['Canopy Height (rh100)']
            x_pre = pre_df_temp[temp]
            # x_test = gedi_df[temp]
            # y_test = gedi_df['Canopy Height (rh100)']
            feat_labels = x_train.columns[0:]

            XGB = xgb.XGBRegressor(max_depth=21,
                                   n_estimators=300,
                                   learning_rate=0.05,
                                   min_child_weight=1,
                                   max_delta_step=0,
                                   subsample=1,
                                   gamma=2,
                                   sampling_method='gradient_based',
                                   colsample_bytree=1,
                                   colsample_bylevel=1,
                                   reg_alpha=0,
                                   tree_method='gpu_hist', )
            XGB.fit(x_train, y_train, eval_metric='mae')
            # y_pred = XGB.predict(x_train)
            y_pred2 = XGB.predict(x_pre)

            out = pre_df_temp.loc[:, ['x', 'y']]
            out.insert(out.shape[1], 'ch', list(y_pred2))
            out.to_csv(f'G:\\A_veg\\S2_all\\20220704\\prediction\\out_mod{str(mode)}_heil{str(hei_limit)}.csv')

            # print('----------------------------------------------------------------------------------------------')
            # print(f' Index:{str(name)}, Max depth:{str(max_d)}, N_est:{str(n_est)}')
            # print('随机森林模型得分： ', score)
            # print('Train MAE:', metrics.mean_absolute_error(y_train, y_pred))
            # res_dic[f'Train_{name}_MAE'].append(metrics.mean_absolute_error(y_train, y_pred))
            # print('Train MSE:', metrics.mean_squared_error(y_train, y_pred))
            # res_dic[f'Train_{name}_MSE'].append(metrics.mean_squared_error(y_train, y_pred))
            # print('Train RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
            # res_dic[f'Train_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

            # print('Test MAE:', metrics.mean_absolute_error(y_test, y_pred2))
            # res_dic[f'Test_{name}_MAE'].append(metrics.mean_absolute_error(y_test, y_pred2))
            # print('Test MSE:', metrics.mean_squared_error(y_test, y_pred2))
            # res_dic[f'Test_{name}_MSE'].append(metrics.mean_squared_error(y_test, y_pred2))
            # print('Test RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
            # res_dic[f'Test_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
            # importances = XGB.feature_importances_
            # indices = np.argsort(importances)[::-1]  # [::-1]表示将各指标按权重大小进行排序输出
            # for f in range(x_train.shape[1]):
            #     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
            #
            # print('----------------------------------------------------------------------------------------------')
    #

            # res_dic = {}
            # for model in range(6):
            #     if model == 0:
            #         temp = index_list
            #         name = 'index'
            #     elif model == 1:
            #         temp = phe_list
            #         name = 'phe'
            #     elif model == 2:
            #         temp = at_list
            #         name = 'at'
            #     elif model == 3:
            #         temp = copy.copy(index_list)
            #         temp.extend(phe_list)
            #         name = 'index_phe'
            #     elif model == 4:
            #         temp = copy.copy(index_list)
            #         temp.extend(at_list)
            #         name = 'index_at'
            #     elif model == 5:
            #         temp = copy.copy(index_list)
            #         temp.extend(phe_list)
            #         temp.extend(at_list)
            #         name = 'index_phe_at'
            #
            #     res_dic[f'Test_{name}_MAE'], res_dic[f'Test_{name}_MSE'], res_dic[f'Test_{name}_RMSE'] = [], [], []
            #     res_dic[f'Train_{name}_MAE'], res_dic[f'Train_{name}_MSE'], res_dic[f'Train_{name}_RMSE'] = [], [], []
            #
            #     for year in [2019, 2020, 2021, 2022]:
            #         x_train = gedi_df.loc[gedi_df['Date'] // 1000 != year][temp]
            #         y_train = gedi_df.loc[gedi_df['Date'] // 1000 != year]['Canopy Height (rh100)']
            #         x_test = gedi_df.loc[gedi_df['Date'] // 1000 == year][temp]
            #         y_test = gedi_df.loc[gedi_df['Date'] // 1000 == year]['Canopy Height (rh100)']
            #         feat_labels = x_train.columns[0:]
            #
            #         XGB = xgb.XGBRegressor(max_depth=max_d,
            #                                n_estimators=n_est,
            #                                learning_rate=0.05,
            #                                min_child_weight=1,
            #                                max_delta_step=0,
            #                                subsample=1,
            #                                gamma =2,
            #                                sampling_method = 'gradient_based',
            #                                colsample_bytree=1,
            #                                colsample_bylevel=1,
            #                                reg_alpha=0,
            #                                tree_method='gpu_hist',)
            #         XGB.fit(x_train, y_train, eval_metric='mae')
            #         y_pred = XGB.predict(x_train)
            #
            #         y_pred2 = XGB.predict(x_test)
            #         score = XGB.score(x_train, y_train)
            #         print('----------------------------------------------------------------------------------------------')
            #         print(f'Year:{str(year)}, Index:{str(name)}, Max depth:{str(max_d)}, N_est:{str(n_est)}')
            #         print('随机森林模型得分： ', score)
            #         print('Train MAE:', metrics.mean_absolute_error(y_train, y_pred))
            #         res_dic[f'Train_{name}_MAE'].append(metrics.mean_absolute_error(y_train, y_pred))
            #         print('Train MSE:', metrics.mean_squared_error(y_train, y_pred))
            #         res_dic[f'Train_{name}_MSE'].append(metrics.mean_squared_error(y_train, y_pred))
            #         print('Train RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
            #         res_dic[f'Train_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
            #
            #         print('Test MAE:', metrics.mean_absolute_error(y_test, y_pred2))
            #         res_dic[f'Test_{name}_MAE'].append(metrics.mean_absolute_error(y_test, y_pred2))
            #         print('Test MSE:', metrics.mean_squared_error(y_test, y_pred2))
            #         res_dic[f'Test_{name}_MSE'].append(metrics.mean_squared_error(y_test, y_pred2))
            #         print('Test RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
            #         res_dic[f'Test_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
            #         importances = XGB.feature_importances_
            #         indices = np.argsort(importances)[::-1]  # [::-1]表示将各指标按权重大小进行排序输出
            #         for f in range(x_train.shape[1]):
            #             print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
            #
            #         print('----------------------------------------------------------------------------------------------')
            #
            #     res_dic[f'Train_{name}_MAE'].append(np.mean(np.array(res_dic[f'Train_{name}_MAE'])))
            #     res_dic[f'Train_{name}_MSE'].append(np.mean(np.array(res_dic[f'Train_{name}_MSE'])))
            #     res_dic[f'Train_{name}_RMSE'].append(np.mean(np.array(res_dic[f'Train_{name}_RMSE'])))
            #     res_dic[f'Test_{name}_MAE'].append(np.mean(np.array(res_dic[f'Test_{name}_MAE'])))
            #     res_dic[f'Test_{name}_MSE'].append(np.mean(np.array(res_dic[f'Test_{name}_MSE'])))
            #     res_dic[f'Test_{name}_RMSE'].append(np.mean(np.array(res_dic[f'Test_{name}_RMSE'])))
            #
            # res_df = pd.DataFrame(res_dic)
            # for _ in range(10000):
            #     if not os.path.exists(
            #             f'G:\\A_veg\\S2_all\\GEDI_v3\\XGB\\RF_res_maxd{str(max_d)}_nes{str(n_est)}_v{str(_)}.csv'):
            #         res_df.to_csv(
            #             f'G:\\A_veg\\S2_all\\GEDI_v3\\XGB\\RF_res_maxd{str(max_d)}_nes{str(n_est)}_v{str(_)}.csv')
            #         break


    # paras = {'silent': True,
    #          'max_depth': 8,
    #          'learning_rate': 0.05,
    #          'n_estimators': 100,
    #          'objective': 'reg:squarederror',
    #          'nthread': -1,'gamma': 0,
    #          'min_child_weight':1,
    #          'max_delta_step':0,
    #          'subsample':0.85,
    #          'colsample_bytree':0.7,
    #          'colsample_bylevel':1,
    #          'reg_alpha':0,
    #          'reg_lambda':1,
    #          'scale_pos_weight':1
    #          }