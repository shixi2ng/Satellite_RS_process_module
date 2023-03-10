import h5py
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point,LineString, Polygon, shape
# import geoviews as gv
# from geoviews import opts, tile_sources as gvts
import basic_function as bf
import sys
import requests as r
import os
import time
import concurrent.futures
from subprocess import call


def import_station_id(file: str):

    if not os.path.exists(file) or not (file.endswith('.xlsx') or file.endswith('.xls') or file.endswith('.csv')):
        raise ValueError('The input station id should store in an excel file!')
    else:
        df = pd.read_excel(file)
        station_id = list(df.iloc[:, 0])
        station_id_list = [str(id) for id in station_id]
        return station_id_list


def download_NCEIfiles_IDM(idList: list, database_link: str, year_range: tuple, IDM_path: str, Download_path: str):

    failure_file = []

    if not database_link.endswith('/'):
        database_link = database_link + '/'

    if not IDM_path.endswith('.exe') or not os.path.exists(IDM_path):
        raise ValueError('Please input the IDM env before initiate the download')

    if not os.path.exists(Download_path):
        raise ValueError('The download path is not valid! Create it before or change one!')

    for year in range(year_range[0], year_range[1] + 1):
        stationid_Lib = r.get(database_link + f'{str(year)}/').text
        stationid_Lib = [temp.split('.')[0] for temp in stationid_Lib.split('<a href="')]
        yearly_downpath = Download_path + str(year) + '\\'
        bf.create_folder(yearly_downpath)

        for stationid in idList:
            if str(stationid) not in stationid_Lib:
                failure_file.append(f'{database_link}{str(year)}/{str(stationid)}.csv')
            else:
                if not os.path.exists(yearly_downpath + f'{str(stationid)}_{str(year)}.csv'):
                    print(f'File name is ：' + f'{str(year)}/{str(stationid)}.csv')
                    print('Add to IDM: ' f'{database_link}{str(year)}/{str(stationid)}.csv')
                    print('---------------------Start to download-----------------------')
                    try:
                        call([IDM_path, '/d', f'{database_link}{str(year)}/{str(stationid)}.csv', '/p', yearly_downpath, '/f', f'{str(stationid)}_{str(year)}.csv', '/a'])
                        call([IDM_path, '/s'])
                    except:
                        failure_file.append(f'{database_link}{str(year)}/{str(stationid)}.csv')

        # for file in failure_file:
        #     try:
        #         call([IDM_path, '/d', file, '/p', yearly_downpath, '/a'])
        #         call([IDM_path, '/s'])
        #     except:
        #         print(f'The {file} cannot be downloaded!')


if __name__ == '__main__':
    IDM = "C:\\Program Files (x86)\\Internet Download Manager\\IDMan.exe"
    idList = import_station_id('G:\\A_veg\\NCEI\\Station_id.xlsx')
    bf.create_folder('G:\\A_veg\\NCEI\\download\\')
    download_NCEIfiles_IDM(idList, 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/', [2004, 2022], IDM, 'G:\\A_veg\\NCEI\\download\\')