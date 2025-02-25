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
from GEDI_toolbox.GEDI_main import GEDI_df
from RF.XGboost import *
import cupy as cp
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import traceback
from sklearn.model_selection import KFold


def XGBoost(xy_dataset, output_folder, mode, max_d_, min_child_, gamma_, subs_, lamda_, alpha_, x_feature, y_feature, learning_ratio, print_info=False):

    try:
        # Generate the result dictionary
        if os.path.isdir(output_folder):
            res_csv_folder = os.path.join(output_folder, 'res_csv\\')
            bf.create_folder(res_csv_folder)
        else:
            raise Exception('The output folder does not exist')

        start_time = time.time()
        if not os.path.exists(os.path.join(res_csv_folder, f'res_mode{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lamda_)}_alpha{str(alpha_)}.csv')):
            x_train = xy_dataset[f'x_train_mode{str(mode)}']
            y_train = xy_dataset[f'y_train_mode{str(mode)}']
            x_test = xy_dataset[f'x_test_mode{str(mode)}']
            y_test = xy_dataset[f'y_test_mode{str(mode)}']

            if isinstance(x_train, list):
                train_score_l, test_score_l, train_mae_l, test_mae_l, train_mse_l, test_mse_l, train_rmse_l, test_rmse_l, OE_l, rank_l = [], [], [], [], [], [], [], [], [], []
                for x_train_, y_train_, x_test_, y_test_ in zip(x_train, y_train, x_test, y_test):
                    model = xgb.XGBRegressor(max_depth=max_d_,
                                             max_leaves=0,
                                             n_estimators=2000,
                                             learning_rate=learning_ratio,
                                             reg_lambda=lamda_,
                                             reg_alpha=alpha_,
                                             colsample_bylevel=1,
                                             min_child_weight=min_child_,
                                             gamma=gamma_,
                                             subsample=subs_,
                                             colsample_bytree=subs_,
                                             eval_metric=metrics.mean_squared_error,
                                             early_stopping_rounds=20,
                                             device='cuda').fit(x_train_, y_train_, eval_set=[(x_train_, y_train_), (x_test_, y_test_)])

                    y_train_pred = model.predict(x_train_)
                    y_test_pred = model.predict(x_test_)
                    train_score_l.append(model.score(x_train_.get(), y_train_.get()))
                    test_score_l.append(model.score(x_test_.get(), y_test_.get()))
                    train_mae_l.append(metrics.mean_absolute_error(y_train_.get(), y_train_pred))
                    test_mae_l.append(metrics.mean_absolute_error(y_test_.get(), y_test_pred))
                    train_mse_l.append(metrics.mean_squared_error(y_train_.get(), y_train_pred))
                    test_mse_l.append(metrics.mean_squared_error(y_test_.get(), y_test_pred))
                    train_rmse_l.append(np.sqrt(metrics.mean_squared_error(y_train_.get(), y_train_pred)))
                    test_rmse_l.append(np.sqrt(metrics.mean_squared_error(y_test_.get(), y_test_pred)))
                    OE_l.append(model.get_booster().best_iteration)

                    # Importance ranking
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    rank_l.append([x_feature[f] for f in indices])

                train_score = np.mean(train_score_l)
                test_score = np.mean(test_score_l)
                train_mae = np.mean(train_mae_l)
                test_mae = np.mean(test_mae_l)
                train_mse = np.mean(train_mse_l)
                test_mse = np.mean(test_mse_l)
                train_rmse = np.mean(train_rmse_l)
                test_rmse = np.mean(test_rmse_l)
                OE = np.mean(OE_l)
                rank = list(itertools.chain(*rank_l)).reverse()

                train_score_l.append(train_score)
                test_score_l.append(test_score)
                train_mae_l.append(train_mae)
                test_mae_l.append(test_mae)
                train_mse_l.append(train_mse)
                test_mse_l.append(test_mse)
                train_rmse_l.append(train_rmse)
                test_rmse_l.append(test_rmse)
                OE_l.append(OE)
                rank_l.append(rank)

                dic = {'train_score': train_score_l, 'test_score': test_score_l, 'train_mae': train_mae_l,
                       'test_mae': test_mae_l, 'train_mse': train_mse_l, 'test_mse': test_mse_l,
                       'train_rmse': train_rmse_l, 'test_rmse': test_rmse_l, 'OE': OE_l, 'rank': rank_l}


            else:
                x_train_, y_train_, x_test_, y_test_ = x_train, y_train, x_test, y_test
                start_time = time.time()
                model = xgb.XGBRegressor(max_depth=max_d_,
                                         max_leaves=0,
                                         learning_rate=learning_ratio,
                                         n_estimators=2000,
                                         reg_lambda=lamda_,
                                         reg_alpha=alpha_,
                                         colsample_bylevel=1,
                                         min_child_weight=min_child_,
                                         gamma=gamma_,
                                         subsample=subs_,
                                         colsample_bytree=subs_,
                                         eval_metric=metrics.mean_squared_error,
                                         early_stopping_rounds=20,
                                         device='cuda').fit(x_train_, y_train_, eval_set=[(x_train_, y_train_), (x_test_, y_test_)])

                y_train_pred = model.predict(x_train_)
                y_test_pred = model.predict(x_test_)
                train_score = model.score(x_train_.get(), y_train_.get())
                test_score = model.score(x_test_.get(), y_test_.get())
                train_mae = metrics.mean_absolute_error(y_train_.get(), y_train_pred)
                test_mae = metrics.mean_absolute_error(y_test_.get(), y_test_pred)
                train_mse = metrics.mean_squared_error(y_train_.get(), y_train_pred)
                test_mse = metrics.mean_squared_error(y_test_.get(), y_test_pred)
                train_rmse = np.sqrt(metrics.mean_squared_error(y_train_.get(), y_train_pred))
                test_rmse = np.sqrt(metrics.mean_squared_error(y_test_.get(), y_test_pred))
                OE = model.get_booster().best_iteration

                # Importance ranking
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                rank = [x_feature[f] for f in indices]

                dic = {'train_score': train_score, 'test_score': test_score, 'train_mae': train_mae, 'test_mae': test_mae,
                       'train_mse': train_mse, 'test_mse': test_mse, 'train_rmse': train_rmse, 'test_rmse': test_rmse,
                       'OE': OE, 'rank': rank}
            pd.DataFrame(dic).to_csv(os.path.join(res_csv_folder, f'res_mode{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lamda_)}_alpha{str(alpha_)}.csv'))

        else:
            df = pd.read_csv(os.path.join(res_csv_folder, f'res_mode{str(mode)}_dep{str(max_d_)}_minchild{str(min_child_)}_gamma{str(gamma_)}_subs{str(subs_)}_lamda{str(lamda_)}_alpha{str(alpha_)}.csv'))
            train_score = df['train_score'][df.shape[0] - 1]
            test_score = df['test_score'][df.shape[0] - 1]
            train_mae = df['train_mae'][df.shape[0] - 1]
            test_mae = df['test_mae'][df.shape[0] - 1]
            train_mse = df['train_mse'][df.shape[0] - 1]
            test_mse = df['test_mse'][df.shape[0] - 1]
            train_rmse = df['train_rmse'][df.shape[0] - 1]
            test_rmse = df['test_rmse'][df.shape[0] - 1]
            OE = df['OE'][df.shape[0] - 1]
            rank = df['rank'][df.shape[0] - 1]

        print(f'End time 4 XGboost mode:{str(mode)} dep:{str(max_d_)} minchild:{str(min_child_)} gamma:{str(gamma_)} subs:{str(subs_)} lambda:{str(lamda_)} alpha{str(alpha_)} Time:{str(time.time() - start_time)}')
        if print_info:
            print('----------------------------------------------------------------------------------------------')
            print(f'X_feature:{str(x_feature)}, Y_feature:{str(y_feature)}, Max depth:{str(max_d_)}')
            print('XGB模型Train得分: ', str(train_score))
            print('Train MAE:', str(train_mae))
            print('Train MSE:', str(train_mse))
            print('Train RMSE:', str(train_rmse))

            print('XGB模型Test得分: ', str(test_score))
            print('Test MAE:', str(test_mae))
            print('Test MSE:', str(test_mse))
            print('Test RMSE:', str(test_rmse))
            print('----------------------------------------------------------------------------------------------')

            for f in range(x_train_.get().shape[1]):
                print("%2d) %-*s %f" % (f + 1, 30, x_feature[indices[f]], importances[indices[f]]))
                print('----------------------------------------------------------------------------------------------')

        return [train_score, test_score, train_mae, test_mae, train_mse, test_mse, train_rmse, test_rmse, OE, rank]
    except:
        print(traceback.print_exc())
        # raise Exception('Error during handling above err')


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
        self.hyperparameters = {}
        self.hyperparameters_df = {}
        self.learnig_ratio = 200

        # Define the support mode
        self._support_mode = [0, 1, 2, 3, 4, 5, 6]
        self._support_hyperpara = ['max_d', 'child_weight',  'gamma', 'subsample', 'lamda', 'alpha']
        self._default_hyperpara = {'max_d': [8, 9, 10], 'child_weight': [0.5, 1, 1.5],  'gamma': [0], 'subsample': [0.79], 'lamda': [10], 'alpha': [4]}

        # Define the model type
        if isinstance(model_type, str) and model_type in ['XGB', 'RFR']:
            self.model_type = model_type
        else:
            raise Exception('The model type should be either XGB or RFR')

    def input_dataset(self, gedi_linked_csv, mode_list=None, cross_validation_factor=True, work_env=None, normalised_dataset=False):

        # User-defined indicator
        if not isinstance(cross_validation_factor, bool):
            raise Exception('The cross validation factor should be a boolean')
        else:
            self.cross_validation_factor = cross_validation_factor

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
            self.work_env = os.path.join(os.path.dirname(gedi_linked_csv), 'VHM')
            bf.create_folder(self.work_env)
        elif not os.path.isdir(self.work_env):
            raise Exception('The work environment is not a folder')
        else:
            bf.create_folder(self.work_env)

        # Preselect the dataset
        self.gedi_linked_RS_df['DOY'] = self.gedi_linked_RS_df['Date'] % 1000
        self.gedi_linked_RS_df = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['VegType_area_average'] < 3]
        self.gedi_linked_RS_df = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['Canopy Height (rh100)'] < 6]
        # self.gedi_linked_RS_df = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['Canopy Height (rh100)'] >= 2]
        # self.gedi_linked_RS_df = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['Landsat water rate'] < 0.1]

        # Normalised dataset
        if self.normalised_dataset:
            peak_doy_key = [_ for _ in self.gedi_linked_RS_df.keys() if 'peak_doy' in _][0]
            peak_vi_key = [_ for _ in self.gedi_linked_RS_df.keys() if 'peak_vi' in _][0]
            
            # create feature concerning the peak ratio
            for key in self.gedi_linked_RS_df.keys():
                if 'Denv' in key and ('TEM' in key or 'DPAR' in key) and 'reliability ' not in key:
                    if 'SOS2DOY' in key:
                        self.gedi_linked_RS_df[key.replace('SOS2DOY', 'ratio')] = self.gedi_linked_RS_df[key] / self.gedi_linked_RS_df[key.replace('SOS2DOY', 'SOS2PEAK')]
                        self.gedi_linked_RS_df[key.replace('SOS2DOY', 'ratio')] = self.gedi_linked_RS_df[key.replace('SOS2DOY', 'ratio')].clip(upper=1)
                    elif 'SOY2DOY' in key:
                        self.gedi_linked_RS_df[key.replace('SOY2DOY', 'ratio')] = self.gedi_linked_RS_df[key] / self.gedi_linked_RS_df[key.replace('SOY2DOY', 'SOY2PEAK')]
                        self.gedi_linked_RS_df[key.replace('SOY2DOY', 'ratio')] = self.gedi_linked_RS_df[key.replace('SOY2DOY', 'ratio')].clip(upper=1)

            for key in self.gedi_linked_RS_df.keys():
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
                    elif 'SOS' in key or 'peak_doy' in key or 'trough_doy' in key or 'EOS' in key:
                        self.gedi_linked_RS_df[key] = self.gedi_linked_RS_df[key] / 365.25
                    elif 'Denv' in key and 'SOS2DOY' in key:
                        self.gedi_linked_RS_df[key] = self.gedi_linked_RS_df[key] / self.gedi_linked_RS_df[key.replace('SOS2DOY', 'SOS2PEAK')]
                    elif 'Denv' in key and 'SOY2DOY' in key:
                        self.gedi_linked_RS_df[key] = self.gedi_linked_RS_df[key] / self.gedi_linked_RS_df[key.replace('SOY2DOY', 'SOY2PEAK')]
                    elif 'Denv' in key and 'DR' in key:
                        self.gedi_linked_RS_df[key] = np.arctan(self.gedi_linked_RS_df[key]) / (np.pi / 4)
                    elif 'Denv' in key and 'GR' in key:
                        self.gedi_linked_RS_df[key] = np.arctan(self.gedi_linked_RS_df[key]) / (np.pi / 4)
                    elif key == 'DOY':
                        self.gedi_linked_RS_df[key] = self.gedi_linked_RS_df[key] / self.gedi_linked_RS_df[peak_doy_key]
                        self.gedi_linked_RS_df[key] = self.gedi_linked_RS_df[key].clip(upper=1)
                else:
                    pass

            for key in self.gedi_linked_RS_df.keys():
                if 'reliability' not in key:
                    if 'Denv' in key and 'PEAK' in key:
                        arr = np.abs(np.array(self.gedi_linked_RS_df[key]).flatten())
                        arr = arr[~np.isnan(arr)]
                        q = np.sort(arr)[::-1][0]
                        self.gedi_linked_RS_df[key] = self.gedi_linked_RS_df[key] / q

        # if reliability:
        #     reliability_key = [_ for _ in self.gedi_linked_RS_df.keys() if 'reliability' in _][0]
        #     self.gedi_linked_RS_df = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df[reliability_key] > 0.5]

        # Define the index
        index_area_average_linear_interpolation = [_ for _ in self.gedi_linked_RS_df.keys() if 'noninun' in _ and 'area_average' in _ and 'linear_interpolation' in _ and 'reliability' not in _]
        index_area_average_24days_ave = [_ for _ in self.gedi_linked_RS_df.keys() if 'noninun' in _ and 'area_average' in _ and '24days_ave' in _ and 'reliability' not in _]
        pheindex_area_average = [_ for _ in self.gedi_linked_RS_df.keys() if 'Pheme' in _ and 'area_average' in _ and 'reliability' not in _]
        denv_area_average = [_ for _ in self.gedi_linked_RS_df.keys() if 'Denv' in _ and 'area_average' in _ and 'reliability' not in _]

        index_focal_linear_interpolation = [_ for _ in self.gedi_linked_RS_df.keys() if 'noninun' in _ and 'focal' in _ and 'linear_interpolation' in _ and 'reliability' not in _]
        pheindex_focal = [_ for _ in self.gedi_linked_RS_df.keys() if 'Pheme' in _ and 'focal' in _ and 'reliability' not in _]
        denv_focal = [_ for _ in self.gedi_linked_RS_df.keys() if 'Denv' in _ and 'focal' in _ and 'reliability' not in _]

        # Generate the mode
        self.mode_dic = {}
        for mode in self.mode_list:
            # Def the mode of model
            if mode == 0:
                mode_index = copy.copy(index_area_average_linear_interpolation)
            elif mode == 1:
                mode_index = copy.copy(index_area_average_linear_interpolation)
                mode_index.extend(pheindex_area_average)
            elif mode == 2:
                mode_index = copy.copy(index_area_average_linear_interpolation)
                mode_index.extend(pheindex_area_average)
                mode_index.extend(denv_area_average)
            elif mode == 3:
                mode_index = copy.copy(index_area_average_24days_ave)
                mode_index.extend(pheindex_area_average)
                mode_index.extend(denv_area_average)
            elif mode == 4:
                mode_index = copy.copy(index_focal_linear_interpolation)
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

            self.gedi_linked_RS_df_ = self.gedi_linked_RS_df.dropna(subset=mode_index).reset_index(drop=True)
            print(f'Mode:{str(mode)}, The number of samples:{str(self.gedi_linked_RS_df_.shape[0])}, drop ratio:{str(1 - self.gedi_linked_RS_df_.shape[0] / self.gedi_linked_RS_df.shape[0])}')

            # Construct the train test dataset
            if cross_validation_factor:
                for year in [2019, 2020, 2021, 2022, 2023]:
                    x_train = self.gedi_linked_RS_df_[self.gedi_linked_RS_df_['Date'] // 1000 != year][mode_index]
                    y_train = self.gedi_linked_RS_df_[self.gedi_linked_RS_df_['Date'] // 1000 != year]['Canopy Height (rh100)']
                    x_test = self.gedi_linked_RS_df_[self.gedi_linked_RS_df_['Date'] // 1000 == year][mode_index]
                    y_test = self.gedi_linked_RS_df_[self.gedi_linked_RS_df_['Date'] // 1000 == year]['Canopy Height (rh100)']

                    if x_train.shape[0] != 0 and x_test.shape[0] != 0:
                        if self.model_type == 'XGB':
                            self.train_test_ds_dic[f'x_train_mode{str(mode)}'].append(cp.array(x_train))
                            self.train_test_ds_dic[f'y_train_mode{str(mode)}'].append(cp.array(y_train))
                            self.train_test_ds_dic[f'x_test_mode{str(mode)}'].append(cp.array(x_test))
                            self.train_test_ds_dic[f'y_test_mode{str(mode)}'].append(cp.array(y_test))
                        elif self.model_type == 'RFR':
                            self.train_test_ds_dic[f'x_train_mode{str(mode)}'].append(x_train)
                            self.train_test_ds_dic[f'y_train_mode{str(mode)}'].append(y_train)
                            self.train_test_ds_dic[f'x_test_mode{str(mode)}'].append(x_test)
                            self.train_test_ds_dic[f'y_test_mode{str(mode)}'].append(y_test)

            else:
                # x_train_, x_test_, y_train_, y_test_ = train_test_split(self.gedi_linked_RS_df_[mode_index], self.gedi_linked_RS_df_['RH 98'], test_size=0.3, random_state=41)
                kf = KFold(n_splits=4, shuffle=True, random_state=42)
                for train_index, test_index in kf.split(self.gedi_linked_RS_df_[mode_index]):
                    x_train_ = self.gedi_linked_RS_df_.loc[train_index, mode_index]
                    y_train_ = self.gedi_linked_RS_df_.loc[train_index, 'RH 98']
                    x_test_ = self.gedi_linked_RS_df_.loc[test_index, mode_index]
                    y_test_ = self.gedi_linked_RS_df_.loc[test_index, 'RH 98']
                    if self.model_type == 'XGB':
                        self.train_test_ds_dic[f'x_train_mode{str(mode)}'].append(cp.array(x_train_))
                        self.train_test_ds_dic[f'y_train_mode{str(mode)}'].append(cp.array(y_train_))
                        self.train_test_ds_dic[f'x_test_mode{str(mode)}'].append(cp.array(x_test_))
                        self.train_test_ds_dic[f'y_test_mode{str(mode)}'].append(cp.array(y_test_))
                    elif self.model_type == 'RFR':
                        self.train_test_ds_dic[f'x_train_mode{str(mode)}'].append(x_train_)
                        self.train_test_ds_dic[f'y_train_mode{str(mode)}'].append(y_train_)
                        self.train_test_ds_dic[f'x_test_mode{str(mode)}'].append(x_test_)
                        self.train_test_ds_dic[f'y_test_mode{str(mode)}'].append(y_test_)

    def train_VHM(self, learning_ratio=0.1, bulk_train=True, **hyperparameters):

        # Check the dataset
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

        for hyperparameter in self._support_hyperpara:
            if hyperparameter not in self.hyperparameters.keys():
                self.hyperparameters[hyperparameter] = self._default_hyperpara[hyperparameter]

        # Generate the Hyperparameters df
        if not os.path.exists(os.path.join(self.work_env, 'res.csv')):
            exist_hyperparameters = None
        else:
            self.res_df = pd.read_csv(os.path.join(self.work_env, 'res.csv'))
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
                with ProcessPoolExecutor() as exe:
                    res = exe.map(XGBoost, repeat(self.train_test_ds_dic), repeat(self.work_env), self.hyperparameters_df['mode'],
                                  self.hyperparameters_df['max_d'], self.hyperparameters_df['child_weight'], self.hyperparameters_df['gamma'],
                                  self.hyperparameters_df['subsample'], self.hyperparameters_df['lamda'], self.hyperparameters_df['alpha'],
                                  self.hyperparameters_df['x_feature'], repeat('Canopy Height (rh100)'), repeat(learning_ratio), repeat(False))

            else:
                res = []
                for max_d_, min_child_,gamma_, subs_, lamda_, alpha_, mode_ in zip(self.hyperparameters_df['max_d'],
                                                                                   self.hyperparameters_df['child_weight'],
                                                                                   self.hyperparameters_df['gamma'],
                                                                                   self.hyperparameters_df['subsample'],
                                                                                   self.hyperparameters_df['lamda'],
                                                                                   self.hyperparameters_df['alpha'],
                                                                                   self.hyperparameters_df['mode']):
                    res.append(XGBoost(self.train_test_ds_dic, self.work_env, mode_, max_d_, min_child_, gamma_, subs_, lamda_, alpha_, self.mode_dic[mode_], 'Canopy Height (rh100)', learning_ratio, print_info=False))

            res = list(res)
            res_df = pd.DataFrame(res, columns=['train_score', 'test_score', 'train_mae', 'test_mae', 'train_mse', 'test_mse', 'train_rmse', 'test_rmse', 'OE', 'rank'])
            res_df = pd.concat([self.hyperparameters_df, res_df], axis=1)
            if self.res_df is not None:
                self.res_df = pd.concat([self.res_df, res_df], axis=0)
            else:
                self.res_df = res_df
            self.res_df.to_csv(os.path.join(self.work_env, 'res.csv'))

    def grid_search_best_para(self):

        # Find the best hyperparameters
        best_para = self.res_df.loc[self.res_df['test_rmse'].idxmin()]
        print(f"The best hyperparameters are: {[_ + ':' + str(best_para[_]) for _ in list(best_para.keys())]}")

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
            plt.savefig(f'G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\VHM\\res_plot\\{keys}.png')
            plt.close()


if __name__ == '__main__':

    VHM = VHM('XGB')
    VHM.input_dataset('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_GEDI_2019_2023_for_model_high_quality.csv', mode_list=[0, 1, 2, 3, 4, 5, 6], cross_validation_factor=False, normalised_dataset=True)
    VHM.train_VHM(learning_ratio=0.1, bulk_train=True, max_d=[2, 3, 4, 5, 6, 7], child_weight=[0, 0.1], gamma=[0], subsample=[0.75, 0.8, 0.85, 0.90, 0.95, 1], lamda=[5, 10, 20, 30, 40], alpha=[1, 2, 3])
    VHM.grid_search_best_para()
    a = 1
    pass

    # for max_d in range(8, 10):
    #     for n_est in range(500, 501, 1):

    #         # Four-fold cross validation
    #         acc_dic = {}
    #         acc_dic[f'Validation_year'] = []
    #         acc_dic[f'Test_Mode{str(mode)}_RMSE'], acc_dic[f'Test_Mode{str(mode)}_MSE'], acc_dic[f'Test_Mode{str(mode)}_MAE'] = [], [], []
    #         acc_dic[f'Train_Mode{str(mode)}_RMSE'], acc_dic[f'Train_Mode{str(mode)}_MSE'], acc_dic[f'Train_Mode{str(mode)}_MAE'] = [], [], []
    #
    #         if model == 'XGB':
    #             # Train the model
    #             XGB = xgb.XGBRegressor(max_depth=max_d,
    #                                    n_estimators=n_est,
    #                                    learning_rate=0.05,
    #                                    min_child_weight=0.5,
    #                                    max_delta_step=0,
    #                                    subsample=0.85,
    #                                    gamma=5,
    #                                    sampling_method='gradient_based',
    #                                    colsample_bytree=1,
    #                                    colsample_bylevel=1,
    #                                    reg_alpha=0,
    #                                    tree_method='gpu_hist', )
    #             XGB.fit(x_train, y_train, eval_metric='rmse')
    #             y_train_pred = XGB.predict(x_train)
    #
    #             # Evaluate the acc
    #             y_train_pred_train = XGB.predict(x_train)
    #             y_train_pred_test = XGB.predict(x_test)
    #             train_score = XGB.score(x_train, y_train)
    #
    #         elif model == 'RFR':
    #             RFR = RandomForestRegressor(max_depth=max_d,
    #                                         n_estimators=n_est,
    #                                         max_features=1,
    #                                         min_samples_leaf=4,
    #                                         n_jobs=-1)
    #             RFR.fit(x_train, y_train)
    #
    #             # Evaluate the acc
    #             y_train_pred_train = RFR.predict(x_train)
    #             y_train_pred_test = RFR.predict(x_test)
    #             score = RFR.score(x_train, y_train)
    #
    #
    #
    #     acc_dic[f'Train_Mode{str(mode)}_MAE'].append(np.mean(np.array(acc_dic[f'Train_Mode{str(mode)}_MAE'])))
    #     acc_dic[f'Train_Mode{str(mode)}_MSE'].append(np.mean(np.array(acc_dic[f'Train_Mode{str(mode)}_MSE'])))
    #     acc_dic[f'Train_Mode{str(mode)}_RMSE'].append(np.mean(np.array(acc_dic[f'Train_Mode{str(mode)}_RMSE'])))
    #     acc_dic[f'Test_Mode{str(mode)}_MAE'].append(np.mean(np.array(acc_dic[f'Test_Mode{str(mode)}_MAE'])))
    #     acc_dic[f'Test_Mode{str(mode)}_MSE'].append(np.mean(np.array(acc_dic[f'Test_Mode{str(mode)}_MSE'])))
    #     acc_dic[f'Test_Mode{str(mode)}_RMSE'].append(np.mean(np.array(acc_dic[f'Test_Mode{str(mode)}_RMSE'])))
    #     acc_dic[f'Validation_year'].append(str('Average'))
    #
    #     # Output the acc dic
    #     bf.create_folder(f'G:\\A_veg\\S2_all\\XGB_vhm\\Model_acc_CrossVal\\')
    #     acc_df = pd.DataFrame(acc_dic)
    #     if not os.path.exists(
    #             f'G:\\A_veg\\S2_all\\XGB_vhm\\Model_acc_CrossVal\\{str(model)}_VHM_Mod{str(mode)}_Nest{str(n_est)}_maxd{str(max_d)}.csv'):
    #         acc_df.to_csv(
    #             f'G:\\A_veg\\S2_all\\XGB_vhm\\Model_acc_CrossVal\\{str(model)}_VHM_Mod{str(mode)}_Nest{str(n_est)}_maxd{str(max_d)}r.csv')
    #
    # if model_prediction_factor:
    #
    #     pre_df_list = [pd.read_csv(f'G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\{str(_)}\\{str(_)}_merged.csv') for _ in ['peak_2021', 'peak_2022']]
    #     output_location = [f'G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\{str(_)}\\predicted_feature_table\\' for _ in ['peak_2021', 'peak_2022']]
    #
    #     # Train the Vegetation Height Model(VHM)
    #     acc_dic = {}
    #     x_train = self.gedi_linked_RS_df[temp]
    #     y_train = self.gedi_linked_RS_df['Canopy Height (rh100)']
    #
    #     if model == 'XGB':
    #         VHM = xgb.XGBRegressor(max_depth=10,
    #                                n_estimators=300,
    #                                learning_rate=0.05,
    #                                min_child_weight=1,
    #                                max_delta_step=0,
    #                                subsample=1,
    #                                gamma=2,
    #                                sampling_method='gradient_based',
    #                                colsample_bytree=1,
    #                                colsample_bylevel=1,
    #                                reg_alpha=0,
    #                                tree_method='gpu_hist', )
    #         VHM.fit(x_train, y_train, eval_metric='rmse')
    #
    #     elif model == 'RFR':
    #         VHM = RandomForestRegressor(max_depth=6,
    #                                     n_estimators=300,
    #                                     max_features=1,
    #                                     min_samples_leaf=4,
    #                                     n_jobs=-1)
    #         VHM.fit(x_train, y_train)
    #
    #         # Evaluate the acc
    #         y_train_pred_train = VHM.predict(x_train)
    #         score = VHM.score(x_train, y_train)
    #
    #     # Print the accuracy of VHM
    #     print('----------------------------------------------------------------------------------------------')
    #     print(f' Mode:{str(mode)}, Max depth:{str(21)}, N_est:{str(300)}')
    #     print('随机森林模型得分： ', score)
    #     print('Train MAE:', metrics.mean_absolute_error(y_train, y_train_pred_train))
    #     acc_dic[f'Train_{str(mode)}_MAE'].append(metrics.mean_absolute_error(y_train, y_train_pred_train))
    #     print('Train MSE:', metrics.mean_squared_error(y_train, y_train_pred_train))
    #     acc_dic[f'Train_{str(mode)}_MSE'].append(metrics.mean_squared_error(y_train, y_train_pred_train))
    #     print('Train RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred_train)))
    #     acc_dic[f'Train_{str(mode)}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred_train)))
    #
    #     importances = VHM.feature_importances_
    #     indices = np.argsort(importances)[::-1]
    #     for f in range(x_train.shape[1]):
    #         print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    #     print('----------------------------------------------------------------------------------------------')
    #
    #     for pre_df, output_t in zip(pre_df_list, output_location):
    #
    #         bf.create_folder(output_t)
    #         pre_df_temp = copy.copy(pre_df.dropna())
    #
    #         if mode == 0:
    #             temp = copy.copy(index_list)
    #             temp.extend(phe_list)
    #             temp.extend(at_list)
    #
    #             for q, p in zip(
    #                     ['OSAVI_20m_noninun', 'MNDWI', 'B8A_noninun', 'B9_noninun', 'B7_noninun', 'B6_noninun',
    #                      'B5_noninun', 'B4_noninun', 'B3_noninun', 'B2_noninun', 'DR', 'DR2', 'EOS', 'GR',
    #                      'peak_doy', 'peak_vi', 'SOS', 'trough_vi', 'static_TEMP', 'static_DPAR', 'peak_TEMP',
    #                      'peak_DPAR', 'accumulated_TEMP', 'accumulated_DPAR'], temp):
    #                 pre_df_temp.rename(columns={q: p}, inplace=True)
    #             pre_df_temp['S2_MNDWI_linear_interpolation'] = (pre_df_temp[
    #                                                                 'S2_MNDWI_linear_interpolation'] - 32768) / 10000
    #
    #         elif mode == 1:
    #             temp = copy.copy(index_list)
    #             temp.extend(phe_list)
    #
    #             for q, p in zip(
    #                     ['OSAVI_20m_noninun', 'MNDWI', 'B8A_noninun', 'B9_noninun', 'B7_noninun', 'B6_noninun',
    #                      'B5_noninun', 'B4_noninun', 'B3_noninun', 'B2_noninun', 'DR', 'DR2', 'EOS', 'GR',
    #                      'peak_doy', 'peak_vi', 'SOS', 'trough_vi'], temp):
    #                 pre_df_temp.rename(columns={q: p}, inplace=True)
    #             pre_df_temp['S2_MNDWI_linear_interpolation'] = (pre_df_temp[
    #                                                                 'S2_MNDWI_linear_interpolation'] - 32768) / 10000
    #
    #         elif mode == 2:
    #
    #             temp = copy.copy(index_list2)
    #             temp.extend(phe_list)
    #             temp.extend(['S2phemetric_static_TEMP', 'S2phemetric_static_DPAR', 'S2phemetric_peak_TEMP',
    #                          'S2phemetric_peak_DPAR'])
    #
    #             for q, p in zip(['OSAVI_20m_noninun', 'B8A_noninun', 'B9_noninun', 'B7_noninun', 'B6_noninun',
    #                              'B5_noninun', 'B4_noninun', 'B3_noninun', 'B2_noninun', 'DR', 'DR2', 'EOS',
    #                              'GR', 'peak_doy', 'peak_vi', 'SOS', 'trough_vi', 'static_TEMP', 'static_DPAR',
    #                              'peak_TEMP', 'peak_DPAR', ], temp):
    #                 pre_df_temp.rename(columns={q: p}, inplace=True)
    #
    #         pre_df_temp['S2_OSAVI_20m_noninun_linear_interpolation'] = (pre_df_temp[
    #                                                                         'S2_OSAVI_20m_noninun_linear_interpolation'] - 32768) / 10000
    #
    #         x_pre = pre_df_temp[temp]
    #         # x_test = self.gedi_linked_RS_df[temp]
    #         # y_test = self.gedi_linked_RS_df['Canopy Height (rh100)']
    #         feat_labels = x_train.columns[0:]
    #
    #         # y_train_pred = XGB.predict(x_train)
    #         y_train_pred_test = VHM.predict(x_pre)
    #
    #         out = pre_df_temp.loc[:, ['x', 'y']]
    #         out.insert(out.shape[1], 'ch', list(y_train_pred_test))
    #         out.to_csv(f'{output_t}out_mod{str(mode)}_heil6.csv')

    # print('----------------------------------------------------------------------------------------------')
    # print(f' Index:{str(name)}, Max depth:{str(max_d)}, N_est:{str(n_est)}')
    # print('随机森林模型得分： ', score)
    # print('Train MAE:', metrics.mean_absolute_error(y_train, y_train_pred))
    # res_dic[f'Train_{name}_MAE'].append(metrics.mean_absolute_error(y_train, y_train_pred))
    # print('Train MSE:', metrics.mean_squared_error(y_train, y_train_pred))
    # res_dic[f'Train_{name}_MSE'].append(metrics.mean_squared_error(y_train, y_train_pred))
    # print('Train RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
    # res_dic[f'Train_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

    # print('Test MAE:', metrics.mean_absolute_error(y_test, y_test_pred))
    # res_dic[f'Test_{name}_MAE'].append(metrics.mean_absolute_error(y_test, y_test_pred))
    # print('Test MSE:', metrics.mean_squared_error(y_test, y_test_pred))
    # res_dic[f'Test_{name}_MSE'].append(metrics.mean_squared_error(y_test, y_test_pred))
    # print('Test RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
    # res_dic[f'Test_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
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
    #         x_train = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['Date'] // 1000 != year][temp]
    #         y_train = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['Date'] // 1000 != year]['Canopy Height (rh100)']
    #         x_test = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['Date'] // 1000 == year][temp]
    #         y_test = self.gedi_linked_RS_df.loc[self.gedi_linked_RS_df['Date'] // 1000 == year]['Canopy Height (rh100)']
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
    #         y_train_pred = XGB.predict(x_train)
    #
    #         y_test_pred = XGB.predict(x_test)
    #         score = XGB.score(x_train, y_train)
    #         print('----------------------------------------------------------------------------------------------')
    #         print(f'Year:{str(year)}, Index:{str(name)}, Max depth:{str(max_d)}, N_est:{str(n_est)}')
    #         print('随机森林模型得分： ', score)
    #         print('Train MAE:', metrics.mean_absolute_error(y_train, y_train_pred))
    #         res_dic[f'Train_{name}_MAE'].append(metrics.mean_absolute_error(y_train, y_train_pred))
    #         print('Train MSE:', metrics.mean_squared_error(y_train, y_train_pred))
    #         res_dic[f'Train_{name}_MSE'].append(metrics.mean_squared_error(y_train, y_train_pred))
    #         print('Train RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
    #         res_dic[f'Train_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
    #
    #         print('Test MAE:', metrics.mean_absolute_error(y_test, y_test_pred))
    #         res_dic[f'Test_{name}_MAE'].append(metrics.mean_absolute_error(y_test, y_test_pred))
    #         print('Test MSE:', metrics.mean_squared_error(y_test, y_test_pred))
    #         res_dic[f'Test_{name}_MSE'].append(metrics.mean_squared_error(y_test, y_test_pred))
    #         print('Test RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
    #         res_dic[f'Test_{name}_RMSE'].append(np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
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
    #          'reg_lamda':1,
    #          'scale_pos_weight':1
    #          }

# Learning RMSE FIG
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