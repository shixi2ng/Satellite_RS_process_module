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
            log_df = pd.read_csv(f'{log_folder}\\{str(station_)}_log.csv')

        with tqdm(total=len(date_list), desc=f'CRAWLER Qweather data 4 Station {str(station_)}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            for date_ in date_list:
                if not os.path.exists(f'{crawler_folder}\\{str(station_)}_{str(date_)}.csv') and log_df.loc[log_df['Date'] == date_, 'State'].iloc[0] == 'Unknown':
                    station_data_url = f'{web_url}{str(station_)}/history/?date={str(date_)}'
                    response = requests.get(station_data_url)
                    if response.status_code == 200:
                        try:
                            page_content = response.text
                            # 使用BeautifulSoup解析页面内容
                            soup = BeautifulSoup(page_content, 'html.parser')
                            content = str(soup)
                            table_temp = pd.read_html(StringIO(content), flavor='lxml')[0]
                            table_temp.to_csv(f'{crawler_folder}\\{str(station_)}_{str(date_)}.csv', encoding='GB18030')
                            log_df.loc[log_df['Date'] == date_, 'State'] = 'Downloaded'
                            time.sleep(0.1)
                        except:
                            log_df.loc[log_df['Date'] == date_, 'State'] = 'Issued'
                    else:
                        log_df.loc[log_df['Date'] == date_, 'State'] = 'Issued'
                else:
                    log_df.loc[log_df['Date'] == date_, 'State'] = 'Downloaded'
                pbar.update()
        log_df.to_csv(f'{log_folder}\\{str(station_)}_log.csv')
    except:
        print(traceback.format_exc())
        pass


class Qweather_dataset(object):

    def __init__(self, work_env=None):

        # Define constant
        self.web_url = 'https://q-weather.info/weather/'
        self.station_inform = pd.read_csv('./station_inform.csv')
        self.support_station_list = list(pd.read_csv('./station_inform.csv')['Station_id'])
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
            for month_ in month_range:
                feature_state = [os.path.exists(os.path.join(feature_path_dic[feature_], f'{self.feature_dic[feature_][3]}{str(month_)}.TXT')) for feature_ in self.feature_dic.keys()]
                if False in feature_state:
                    weather_df_dic = dict.fromkeys(self.feature_dic, [])
                    for index_ in range(self.Qweather_df.shape[0]):
                        if int(self.Qweather_df['Year'][index_]) * 100 + int(self.Qweather_df['Month'][index_]) == month_:
                            csv_file = pd.read_csv(self.Qweather_df['File_name'][index_], encoding='GB18030')
                            for feature_ in weather_df_dic.keys():
                                data_arr = np.array(csv_file[self.feature_dic[feature_][0]])
                                daily_mean, daily_max, daily_min, daily_acc = np.nanmean(data_arr), np.nanmax(data_arr),\
                                                                              np.nanmin(data_arr), np.nansum(data_arr)
                                station_id = int(self.Qweather_df['Station_id'][index_])
                                station_inform = np.array(self.station_inform[self.station_inform['Station_id'] == station_id]).tolist()
                                lat = 1
                                lon = 1
                                alt = 1
                                year_ = int(self.Qweather_df['Year'][index_])
                                month_ = int(self.Qweather_df['Month'][index_])
                                day_ = int(self.Qweather_df['Day'][index_])

                                if feature_ == 'TEM':
                                    weather_df_dic[feature_].append([int(),
                                                                     int(self.Qweather_df['Station'][index_] * 100),
                                                                     int(self.Qweather_df['Station'][index_] * 100),])

                                elif feature_ == ['PRS', 'WIN', 'RHU']:
                                    pass

                                elif feature_ in ['PRE']:
                                    pass


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
    #                            date_range=[20200101, 20231231])