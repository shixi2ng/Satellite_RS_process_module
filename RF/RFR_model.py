import os.path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import pandas as pd
import basic_function as bf
from GEDI_toolbox.GEDI_main import GEDI_list
import copy


cur_dic = {}
cur_dic[f'test3_rmse'] = []
cur_dic[f'test1_rmse'] = []
cur_dic[f'test2_rmse'] = []
cur_dic[f'test_rmse'] = []
cur_dic[f'train3_rmse'] = []
cur_dic[f'train1_rmse'] = []
cur_dic[f'train2_rmse'] = []
cur_dic[f'train_rmse'] = []
for max_d in range(5, 50):
    _ = bf.file_filter('G:\A_veg\S2_all\XGB_vhm\GEDI_ds_V3\RFR\\', ['_nes500_', f'_maxd{str(max_d)}_'], and_or_factor='and')
    if len(_) == 0:
        cur_dic[f'test3_rmse'].append(np.nan)
        cur_dic[f'test2_rmse'].append(np.nan)
        cur_dic[f'test1_rmse'].append(np.nan)
        cur_dic[f'test_rmse'].append(np.nan)
        cur_dic[f'train3_rmse'].append(np.nan)
        cur_dic[f'train2_rmse'].append(np.nan)
        cur_dic[f'train1_rmse'].append(np.nan)
        cur_dic[f'train_rmse'].append(np.nan)
    else:
        pd_temp = pd.read_csv(_[0])
        cur_dic[f'test3_rmse'].append(pd_temp.loc[4, 'Test_index_at_RMSE'])
        cur_dic[f'test2_rmse'].append(pd_temp.loc[4, 'Test_index_RMSE'])
        cur_dic[f'test1_rmse'].append(pd_temp.loc[4, 'Test_index_phe_RMSE'])
        cur_dic[f'test_rmse'].append(pd_temp.loc[4, 'Test_index_phe_at_RMSE'])
        cur_dic[f'train3_rmse'].append(pd_temp.loc[4, 'Train_index_at_RMSE'])
        cur_dic[f'train2_rmse'].append(pd_temp.loc[4, 'Train_index_RMSE'])
        cur_dic[f'train1_rmse'].append(pd_temp.loc[4, 'Train_index_phe_RMSE'])
        cur_dic[f'train_rmse'].append(pd_temp.loc[4, 'Train_index_phe_at_RMSE'])

fig4, ax4 = plt.subplots(figsize=(12, 8), constrained_layout=True)
ax4.plot(np.linspace(5, 49, 45), cur_dic[f'test3_rmse'], label='Test_index_at_RMSE')
ax4.plot(np.linspace(5, 49, 45), cur_dic[f'test2_rmse'], label='Test_index_RMSE')
ax4.plot(np.linspace(5, 49, 45), cur_dic[f'test1_rmse'], label='Test_index_phe_RMSE')
ax4.plot(np.linspace(5, 49, 45), cur_dic[f'test_rmse'], label='Test_index_phe_at_RMSE')
ax4.plot(np.linspace(5, 49, 45), cur_dic[f'train_rmse'], label='Train_index_RMSE')
ax4.plot(np.linspace(5, 49, 45), cur_dic[f'train1_rmse'], label='Train_index_phe_RMSE')
ax4.plot(np.linspace(5, 49, 45), cur_dic[f'train2_rmse'], label='Train_index_phe_at_RMSE')
ax4.plot(np.linspace(5, 49, 45), cur_dic[f'train2_rmse'], label='Train_index_at_RMSE')
plt.legend()
plt.show()


for n_est in range(500, 1000):
    for max_d in range(20, 200):
        gedi_ds = GEDI_list('G:\\A_veg\\S2_all\\XGB_vhm\\GEDI_ds_V3\\floodplain_2020_high_quality_merged.csv')
        gedi_df = gedi_ds.GEDI_df
        gedi_df = gedi_df.dropna()
        index_list = ['S2_OSAVI_20m_noninun_linear_interpolation',
                      'S2_NDVI_20m_noninun_linear_interpolation', 'Tandem-X DEM',
                      'S2_B8A_noninun_linear_interpolation',  'S2_B9_noninun_linear_interpolation', 'S2_B8_noninun_linear_interpolation',
                      'S2_B7_noninun_linear_interpolation',  'S2_B6_noninun_linear_interpolation',
                      'S2_B5_noninun_linear_interpolation', ''
                      'S2_B4_noninun_linear_interpolation', 'S2_B3_noninun_linear_interpolation', 'S2_B2_noninun_linear_interpolation',]
        # phe_list = ['S2phemetric_DR', 'S2phemetric_DR2', 'S2phemetric_EOS', 'S2phemetric_GR', 'S2phemetric_peak_doy', 'S2phemetric_peak_vi', 'S2phemetric_SOS', 'S2phemetric_trough_vi']
        phe_list = ['S2phemetric_EOS', 'S2phemetric_peak_doy', 'S2phemetric_peak_vi', 'S2phemetric_SOS', 'S2phemetric_trough_vi', 'S2phemetric_static_TEMP', 'S2phemetric_static_DPAR']
        at_list = ['S2_accumulated_TEMP_relative', 'S2_accumulated_DPAR_relative', 'S2phemetric_peak_TEMP', 'S2phemetric_peak_DPAR', ]
        gedi_df = gedi_df.loc[gedi_df['Canopy Height (rh100)'] < 5]

        res_dic = {}
        for model in range(6):
            if model == 0:
                temp = index_list
                name = 'index'
            elif model == 1:
                temp = phe_list
                name = 'phe'
            elif model == 2:
                temp = at_list
                name = 'at'
            elif model == 3:
                temp = copy.copy(index_list)
                temp.extend(phe_list)
                name = 'index_phe'
            elif model == 4:
                temp = copy.copy(index_list)
                temp.extend(at_list)
                name = 'index_at'
            elif model == 5:
                temp = copy.copy(index_list)
                temp.extend(phe_list)
                temp.extend(at_list)
                name = 'index_phe_at'

            res_dic[f'Test_{name}_MAE'], res_dic[f'Test_{name}_MSE'], res_dic[f'Test_{name}_RMSE'] = [], [], []
            res_dic[f'Train_{name}_MAE'], res_dic[f'Train_{name}_MSE'], res_dic[f'Train_{name}_RMSE'] = [], [], []

            for year in [2019, 2020, 2021, 2022]:
                x_train = gedi_df.loc[gedi_df['Date'] // 1000 != year][temp]
                y_train = gedi_df.loc[gedi_df['Date'] // 1000 != year]['Canopy Height (rh100)']
                x_test = gedi_df.loc[gedi_df['Date'] // 1000 == year][temp]
                y_test = gedi_df.loc[gedi_df['Date'] // 1000 == year]['Canopy Height (rh100)']
                feat_labels = x_train.columns[0:]

                RFR = RandomForestRegressor(max_depth=max_d,
                                            n_estimators=n_est,
                                            max_features=1,
                                            min_samples_leaf=4,
                                            n_jobs=-1)
                RFR.fit(x_train, y_train)
                y_pred = RFR.predict(x_train)
                y_pred2 = RFR.predict(x_test)
                score = RFR.score(x_train, y_train)
                print('----------------------------------------------------------------------------------------------')
                print(f'Year:{str(year)}, Index:{str(name)}, Max depth:{str(max_d)}, N_est:{str(n_est)}')
                print('随机森林模型得分： ', score)
                print('Train MAE:', metrics.mean_absolute_error(y_train, y_pred))
                res_dic[f'Train_{name}_MAE'].append(metrics.mean_absolute_error(y_train, y_pred))
                print('Train MSE:', metrics.mean_squared_error(y_train, y_pred))
                res_dic[f'Train_{name}_MSE'].append(metrics.mean_squared_error(y_train, y_pred))
                print('Train RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
                res_dic[f'Train_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

                print('Test MAE:', metrics.mean_absolute_error(y_test, y_pred2))
                res_dic[f'Test_{name}_MAE'].append(metrics.mean_absolute_error(y_test, y_pred2))
                print('Test MSE:', metrics.mean_squared_error(y_test, y_pred2))
                res_dic[f'Test_{name}_MSE'].append(metrics.mean_squared_error(y_test, y_pred2))
                print('Test RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
                res_dic[f'Test_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
                importances = RFR.feature_importances_
                indices = np.argsort(importances)[::-1]  # [::-1]表示将各指标按权重大小进行排序输出
                for f in range(x_train.shape[1]):
                    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

                print('----------------------------------------------------------------------------------------------')

            res_dic[f'Train_{name}_MAE'].append(np.mean(np.array(res_dic[f'Train_{name}_MAE'])))
            res_dic[f'Train_{name}_MSE'].append(np.mean(np.array(res_dic[f'Train_{name}_MSE'])))
            res_dic[f'Train_{name}_RMSE'].append(np.mean(np.array(res_dic[f'Train_{name}_RMSE'])))
            res_dic[f'Test_{name}_MAE'].append(np.mean(np.array(res_dic[f'Test_{name}_MAE'])))
            res_dic[f'Test_{name}_MSE'].append(np.mean(np.array(res_dic[f'Test_{name}_MSE'])))
            res_dic[f'Test_{name}_RMSE'].append(np.mean(np.array(res_dic[f'Test_{name}_RMSE'])))

        res_df = pd.DataFrame(res_dic)
        for _ in range(10000):
            if not os.path.exists(f'G:\\A_veg\\S2_all\\GEDI_v3\\RFR\\RF_res_maxd{str(max_d)}_nes{str(n_est)}_v{str(_)}.csv'):
                res_df.to_csv(f'G:\\A_veg\\S2_all\\GEDI_v3\\RFR\\RF_res_maxd{str(max_d)}_nes{str(n_est)}_v{str(_)}.csv')
                break

cur_dic = {}
for minleaf in range(2, 5):
    cur_dic[f'train_rmse_{str(minleaf)}'] = []
    cur_dic[f'test_rmse_{str(minleaf)}'] = []
    for max_d in range(10, 501):
        _ = bf.file_filter('G:\\A_veg\\S2_all\\GEDI_v3\\', [f'ml{str(minleaf)}', '_nes500_', f'_maxd{str(max_d)}_'], and_or_factor='and')
        if len(_) == 0:
            cur_dic[f'train_rmse_{str(minleaf)}'].append(np.nan)
            cur_dic[f'test_rmse_{str(minleaf)}'].append(np.nan)
        else:
            pd = pd.read_csv(_[0])
            cur_dic[f'train_rmse_{str(minleaf)}'].append(pd.iloc[4, 'Test_index_RMSE'])
            cur_dic[f'train_rmse_{str(minleaf)}'].append(pd.iloc[4, 'Test_index_phe_at_RMSE'])

fig4, ax4 = plt.subplots(figsize=(12, 8), constrained_layout=True)
ax4.plot(np.linspace(10, 501, 492), cur_dic[f'train_rmse_2'])
ax4.plot(np.linspace(10, 501, 492), cur_dic[f'train_rmse_3'])
ax4.plot(np.linspace(10, 501, 492), cur_dic[f'train_rmse_4'])
ax4.plot(np.linspace(10, 501, 492), cur_dic[f'test_rmse_2'])
ax4.plot(np.linspace(10, 501, 492), cur_dic[f'test_rmse_3'])
ax4.plot(np.linspace(10, 501, 492), cur_dic[f'test_rmse_4'])

plt.show()

