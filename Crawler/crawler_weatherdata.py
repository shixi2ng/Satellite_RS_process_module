import copy
import os.path
import pandas as pd
import requests
from bs4 import BeautifulSoup
import traceback
import random
import ast
from tqdm import tqdm
from datetime import timedelta, datetime
import pytz
import time
import basic_function as bf
import concurrent.futures
from itertools import repeat
import numpy as np
from io import StringIO


def crawler_qweather_data(station_, date_list, crawler_folder, log_folder, web_url):

    try:
        if not os.path.exists(f'{log_folder}\\{str(station_)}_log.csv'):
            log_df = pd.DataFrame({'Station': [station_ for _ in range(len(date_list))], 'Date': date_list, 'State': ['Unknown' for _ in range(len(date_list))]})
        else:
            log_df2 = pd.read_csv(f'{log_folder}\\{str(station_)}_log.csv')
            log_df = pd.DataFrame({'Station': [station_ for _ in range(len(date_list))], 'Date': date_list, 'State': ['Unknown' for _ in range(len(date_list))]})
            for _ in range(log_df.shape[0]):
                if log_df2[(log_df2['Station'] == log_df['Station'][_]) & (log_df2['Date'] == log_df['Date'][_])].index.shape[0] == 1:
                    log_df.loc[_, 'State'] = log_df2[(log_df2['Station'] == log_df['Station'][_]) & (log_df2['Date'] == log_df['Date'][_])]['State'].values
                elif log_df2[(log_df2['Station'] == log_df['Station'][_]) & (log_df2['Date'] == log_df['Station'][_])].index.shape[0] > 1:
                    raise Exception('Code Error')
                else:
                    pass

        with tqdm(total=len(date_list), desc=f'CRAWLER Qweather data 4 Station {str(station_)}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            for date_ in date_list:
                if not os.path.exists(f'{crawler_folder}\\{str(station_)}_{str(date_)}.csv') and (log_df.loc[log_df['Date'] == date_, 'State'].iloc[0] == 'Unknown' or log_df.loc[log_df['Date'] == date_, 'State'].iloc[0] == 'Issued'):
                    station_data_url = f'{web_url}{str(station_)}/history/?date={str(date_)}'
                    response = requests.get(station_data_url)
                    if response.status_code == 200:
                        try:
                            page_content = response.text
                            # 使用BeautifulSoup解析页面内容
                            soup = BeautifulSoup(page_content, 'html.parser')
                            content = str(soup)
                            if "无历史数据" not in content:
                                table_temp = pd.read_html(StringIO(content), flavor='lxml')[0]
                                table_temp.to_csv(f'{crawler_folder}\\{str(station_)}_{str(date_)}.csv', encoding='GB18030')
                                log_df.loc[log_df['Date'] == date_, 'State'] = 'Downloaded'
                                time.sleep(0.1)
                            else:
                                log_df.loc[log_df['Date'] == date_, 'State'] = 'NoData'
                        except:
                            log_df.loc[log_df['Date'] == date_, 'State'] = 'Issued'
                    else:
                        log_df.loc[log_df['Date'] == date_, 'State'] = 'Issued'
                else:
                    log_df.loc[log_df['Date'] == date_, 'State'] = 'Downloaded'
                pbar.update()
        log_df.to_csv(f'{log_folder}\\{str(station_)}_log.csv', index=False)
    except:
        print(traceback.format_exc())
        pass


class Qweather_dataset(object):

    def __init__(self, work_env=None):

        # Define constant
        module_path = os.path.dirname(__file__)
        self.web_url = 'https://q-weather.info/weather/'
        self.station_inform = pd.read_csv(os.path.join(module_path, 'station_inform.csv'))
        self.support_station_list = list(pd.read_csv(os.path.join(module_path, 'station_inform.csv'))['Station_id'])
        self.support_date_range = [datetime(1950, 1, 1), datetime.now()]
        self.feature_dic = {'TEM': ['瞬时温度', 2, '12001', 'SURF_CLI_CHN_MUL_DAY-TEM-12001-'],
                            'PRS': ['地面气压', 3, '10004', 'SURF_CLI_CHN_MUL_DAY-PRS-10004-'],
                            'WIN': ['瞬时风速', 6, '11002', 'SURF_CLI_CHN_MUL_DAY-WIN-11002-'],
                            'RHU': ['相对湿度', 4, '13003', 'SURF_CLI_CHN_MUL_DAY-RHU-13003-'],
                            'PRE': ['1小时降水', 7, '13011', 'SURF_CLI_CHN_MUL_DAY-PRE-13011-']}

        # Define the path
        self.work_env = work_env
        self.crawler_folder = None
        self.log_folder = None
        self.cma_folder = None

        # Define the variable
        self.date_range = None
        self.station_list = None
        self.Qweather_df = None

        # Update the dataset
        self._update_dataset()

    def _update_dataset(self):

        # Update the path
        if self.work_env is not None:
            if os.path.isdir(self.work_env):
                self.crawler_folder = os.path.join(self.work_env, 'Qweather_crawler\\Original\\')
                self.log_folder = os.path.join(self.work_env, 'Qweather_crawler\\Log\\')
                bf.create_folder(self.crawler_folder)
                bf.create_folder(self.log_folder)
            else:
                raise ValueError('The work env is invalid')

        # Update the dataset
        file_list = bf.file_filter(self.crawler_folder, ['.csv'])
        if len(file_list) > 0:
            station_list, date_list, Qweather_dic = [], [], {'File_name': [], 'Station_id': [], 'Date': [], 'Year': [], 'Month': [], 'Day': []}
            for _ in file_list:
                station_list.append(int(os.path.basename(_).split('_')[0]))
                date_temp = os.path.basename(_).split('_')[1].split('.csv')[0]
                date_list.append(int(date_temp.split('-')[0]) * 10000 + int(date_temp.split('-')[1]) * 100 + int(date_temp.split('-')[2]))
                Qweather_dic['File_name'].append(_)
                Qweather_dic['Station_id'].append(int(os.path.basename(_).split('_')[0]))
                Qweather_dic['Date'].append(int(date_temp.split('-')[0]) * 10000 + int(date_temp.split('-')[1]) * 100 + int(date_temp.split('-')[2]))
                Qweather_dic['Year'].append(int(date_temp.split('-')[0]))
                Qweather_dic['Month'].append(int(date_temp.split('-')[1]))
                Qweather_dic['Day'].append(int(date_temp.split('-')[2]))

            self.date_range = np.unique(date_list).tolist()
            self.station_list = np.unique(station_list).tolist()
            self.Qweather_df = pd.DataFrame(Qweather_dic)

    def to_standard_cma_file(self):

        # Create standard cma folder
        self.cma_folder = os.path.join(self.work_env, 'Qweather_CMA_standard\\')
        bf.create_folder(self.cma_folder)

        if self.Qweather_df is not None:
            feature_path_dic = {}
            # Create feature path
            for _ in self.feature_dic.keys():
                feature_path_dic[_] = os.path.join(self.work_env, f'Qweather_CMA_standard\\{self.feature_dic[_][2]}\\')
                bf.create_folder(feature_path_dic[_])

            month_range = np.unique(np.array(self.date_range) // 100).tolist()
            with (tqdm(total=len(month_range), desc=f'Process Qweather data to standard CMA file', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar):

                for month_ in month_range:
                    feature_state = [os.path.exists(os.path.join(feature_path_dic[feature_], f'{self.feature_dic[feature_][3]}{str(month_)}.TXT')) for feature_ in self.feature_dic.keys()]
                    if False in feature_state:

                        # Create weather df dic
                        weather_df_dic = dict.fromkeys(self.feature_dic, None)
                        for _ in weather_df_dic.keys():
                            weather_df_dic[_] = []

                        for index_ in range(self.Qweather_df.shape[0]):
                            if int(self.Qweather_df['Year'][index_]) * 100 + int(self.Qweather_df['Month'][index_]) == month_:

                                try:
                                    # Read csv file
                                    csv_file = pd.read_csv(self.Qweather_df['File_name'][index_], encoding='GB18030')

                                    # Retrieve station inform
                                    station_ = self.Qweather_df['Station_id'][index_]
                                    if station_ not in self.station_list:
                                        print(f'Station {str(station_)} is not in the station inform.csv, it could be a new station!')
                                    else:
                                        station_df = self.station_inform[self.station_inform['Station_id'] == station_]
                                        if month_ > max(station_df['End_YYYYMM']):
                                            station_inform_ = station_df.loc[station_df['End_YYYYMM'].idxmax()].iloc[:4].values.tolist()
                                        elif month_ < min(station_df['Start_YYYYMM']):
                                            station_inform_ = station_df.loc[station_df['Start_YYYYMM'].idxmin()].iloc[:4].values.tolist()
                                        else:
                                            station_inform_ = station_df[(station_df['End_YYYYMM'] >= month_) & (station_df['Start_YYYYMM'] <= month_)].iloc[:, :4].values.tolist()[0]

                                        # Retrieve the date inform
                                        day_list = [self.Qweather_df['Year'][index_], self.Qweather_df['Month'][index_], self.Qweather_df['Day'][index_]]

                                        # Gather as base inform
                                        station_inform_.extend(day_list)
                                        if len(station_inform_) != 7:
                                            raise Exception('The base inform is wrong!')

                                        # Retrieve the feature data
                                        for feature_ in weather_df_dic.keys():
                                            feature_list, combine_inform = [], copy.deepcopy(station_inform_)
                                            if self.feature_dic[feature_][0] not in csv_file.keys():
                                                data_arr = np.array(csv_file[csv_file.keys()[self.feature_dic[feature_][1]]])
                                            else:
                                                data_arr = np.array(csv_file[self.feature_dic[feature_][0]])
                                            if data_arr.dtype == 'object':
                                                data_arr_new = []
                                                for _ in data_arr:
                                                    try:
                                                        data_arr_new.append(float(_))
                                                    except:
                                                        data_arr_new.append(np.nan)
                                                data_arr = np.array(data_arr_new)

                                            if np.isnan(data_arr).all():
                                                daily_mean, daily_max, daily_min, daily_acc,  = 32766, 32766, 32766, 3276.6
                                            else:
                                                daily_mean, daily_max, daily_min, daily_acc = np.nanmean(data_arr), np.nanmax(data_arr), np.nanmin(data_arr), np.nansum(data_arr)

                                            if np.isnan(data_arr[0:12]).all():
                                                daily_acc1 = 3276.6
                                            else:
                                                daily_acc1 = np.nansum(data_arr[0:12])

                                            if np.isnan(data_arr[12:]).all():
                                                daily_acc2 = 3276.6
                                            else:
                                                daily_acc2 = np.nansum(data_arr[12:])

                                            if feature_ == 'TEM' or feature_ == 'PRS':
                                                if np.isnan(data_arr).all():
                                                    feature_list = [32766, 32766, 32766, 0, 0, 0]
                                                else:
                                                    feature_list = [int(daily_mean * 10), int(daily_max * 10), int(daily_min * 10), 0, 0, 0]

                                            elif feature_ == 'RHU':
                                                feature_list = [int(daily_mean), int(daily_min), 0, 0]

                                            elif feature_ == 'PRE':
                                                feature_list = [int(daily_acc1 * 10), int(daily_acc2 * 10), int(daily_acc * 10), 0, 0, 0]

                                            elif feature_ == 'WIN':
                                                if np.isnan(data_arr).all():
                                                    feature_list = [32766, 32766, 0, 0, 0, 0, 0, 0, 0, 0]
                                                else:
                                                    feature_list = [int(daily_mean * 10), int(daily_max * 10), 0, 0, 0, 0, 0, 0, 0, 0]
                                            else:
                                                raise Exception('Code Error')
                                            combine_inform.extend(feature_list)
                                            weather_df_dic[feature_].append(combine_inform)
                                except:
                                    print(f'Failed {str(month_)} {feature_} {str(station_)}')

                        for feature_ in weather_df_dic.keys():
                            df_ = pd.DataFrame(weather_df_dic[feature_])
                            df_.to_csv(os.path.join(feature_path_dic[feature_], f'SURF_CLI_CHN_MUL_DAY-{feature_}-{self.feature_dic[feature_][2]}-{str(month_)}.TXT'), sep=' ', index=False, header=False)
                    pbar.update()

    def crawler_weather_data(self, output_folder, station_list: list = None, date_range=None, batch_download=True):

        # Create folders
        self.work_env = output_folder
        bf.create_folder(output_folder)

        # Batch download factor
        if isinstance(batch_download, bool):
            batch_download = batch_download
        else:
            raise TypeError('The batch download factor should be a bool')

        # Check the station list
        if isinstance(station_list, list):
            station_list = [_ for _ in station_list if _ in self.support_station_list]
            if len(station_list) == 0:
                raise ValueError('Please input the station supported in ./station_inform.csv')
        elif station_list is None:
            station_list = self.support_station_list
            print('All station is downloaded! Please mention it could be an extremely large dataset!')
        else:
            raise TypeError('The station list should be a list')

        # Check the date range
        if isinstance(date_range, list) and len(date_range) == 2:
            if date_range[0] > date_range[1]:
                raise ValueError('End date should late than start date!')
            else:
                date_range = [date_range[0], date_range[1]]
        elif date_range is None:
            date_range = self.support_date_range
            print('All data to now is downloaded! Please mention it could be an extremely large dataset!')
        else:
            raise TypeError('The date list should be a list')

        try:
            start_date = datetime(int(date_range[0] // 10000), int(np.mod(date_range[0], 10000) // 100), int(np.mod(date_range[0], 100)))
            end_date = datetime(int(date_range[1] // 10000), int(np.mod(date_range[1], 10000) // 100), int(np.mod(date_range[1], 100)))
            date_list = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end_date - start_date).days + 1)]
        except:
            raise ValueError('Invalid start or end date')

        # Batch download
        if batch_download:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(crawler_qweather_data, station_list, repeat(date_list), repeat(self.crawler_folder), repeat(self.log_folder), repeat(self.web_url))
        else:
            for station_ in station_list:
                crawler_qweather_data(station_, date_list, self.crawler_folder, self.log_folder, self.web_url)


if __name__ == '__main__':
    QW_ds = Qweather_dataset('G:\\A_Climatology_dataset\\station_dataset\\Qweather_dataset\\')
    QW_ds.to_standard_cma_file()
    # QW_ds.crawler_weather_data('G:\\A_Climatology_dataset\\station_dataset\\Qweather_dataset\\',
    #                            station_list=list(pd.read_csv('G:\\A_Climatology_dataset\\station_dataset\\station_profile\\shpfile\\Station_MYR.csv')['Station_id']),
    #                            date_range=[20160101, 20231231])