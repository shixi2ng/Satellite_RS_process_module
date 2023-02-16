from subprocess import call
from sentinelsat import SentinelAPI
import time
from tqdm import tqdm
from xml.dom.minidom import parse
from subprocess import call
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt, LTAError, ServerError
from requests import HTTPError
from datetime import date
import time
import xlrd
from tqdm import tqdm
import geopandas as gp
import basic_function as bf
import os
import sys
import numpy as np
import pandas as pd
import json
import concurrent.futures
from collections import OrderedDict

# This Program is mainly used to batch download Sentinel-2 image using IDM
# Several things you need to make sure before the download
# (1) Successfully download and install IDM as well as sentinelsat package
# (2) Add username and password for the scihub url in the setting interface of IDM


class Queried_Sentinel_ds(object):
    def __init__(self, username, password, work_env, IDM_path=None):
        self.define_IDM_path(IDM_path)
        # Construct work env
        if os.path.exists(work_env):
            self.work_env = bf.Path(work_env).path_name
        else:
            print('Please input a valid work environment')
            sys.exit(-1)

        # Define variable
        self.req_products = None
        self.queried_folder = None
        self.download_path = None
        self.downloaded_file = []
        self.req_products_df = pd.DataFrame()
        self.online_file_ID_list = []
        self.offline_file_ID_list = []
        self.failure_file = []

        # Initialise API
        self.api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')

    def queried_with_ROI(self, shpfile_path, date_range, platform, product_type, cloud_cover_range, overwritten_factor=False):
        shpfile_path = bf.Path(shpfile_path).path_name
        # Create queried folder
        self.queried_folder = self.work_env + shpfile_path.split('\\')[-1].split('.shp')[0] + '_' + date_range[0] + '_' + date_range[1] + '_' + product_type + '\\queried_log\\'
        self.download_path = self.queried_folder.split('queried_log')[0] + 'download_files\\'
        bf.create_folder(self.queried_folder)
        bf.create_folder(self.download_path)
        self.downloaded_file = bf.file_filter(self.download_path, ['.zip'])
        # Construct req product
        if not os.path.exists(self.queried_folder + 'queried_products.csv') or overwritten_factor:
            foot_print = geojson_to_wkt(read_geojson(bf.shp2geojson(shpfile_path)))
            try:
                self.req_products = self.api.query(foot_print, date=date_range, platformname=platform, producttype=product_type, cloudcoverpercentage=cloud_cover_range)
            except:
                print('The query failed!')
                sys.exit(-1)
            self.req_products_df = self.api.to_dataframe(self.req_products)
            self.req_products_df.to_csv(self.queried_folder + 'queried_products.csv')
            self.req_products_df = pd.read_csv(self.queried_folder + 'queried_products.csv')
        else:
            self.req_products_df = pd.read_csv(self.queried_folder + 'queried_products.csv')

        if not os.path.exists(self.queried_folder + 'online_products.npy') or overwritten_factor:
            self.failure_file = []
            self.update_online_status()
            np.save(self.queried_folder + 'online_products.npy', self.online_file_ID_list)
            np.save(self.queried_folder + 'offline_products.npy', self.offline_file_ID_list)
        else:
            self.online_file_ID_list = np.load(self.queried_folder + 'online_products.npy', allow_pickle=True).tolist()
            self.offline_file_ID_list = np.load(self.queried_folder + 'offline_products.npy', allow_pickle=True).tolist()

    def convert_from_meta4_file(self, meta4file):
        pass

    def define_IDM_path(self, IDM_path):
        if os.path.exists(IDM_path) and IDM_path.endswith('.exe'):
            self.IDM_path = IDM_path
        else:
            print('The IDM path is not INPUT!')
            self.IDM_path = None

    def update_online_status(self):
        if self.req_products_df is None:
            print('Please query the files before the download!')
            return

        self.req_products_df.loc[:, 'status'] = np.nan
        for i in range(self.req_products_df.shape[0]):
            title_name = self.req_products_df.loc[i, 'title']
            if True in [title_name in file_name_temp for file_name_temp in self.downloaded_file]:
                self.req_products_df.loc[i, 'status'] = 'Downloaded'

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     for product_info, file_ID in zip(executor.map(self.api.get_product_odata, self.req_products.keys()), self.req_products.keys()):
        #         if product_info['Online']:
        #             self.online_file_ID_list.append(file_ID)
        #         else:
        #             self.offline_file_ID_list.append(file_ID)

        for i in range(self.req_products_df.shape[0]):
            if self.req_products_df.loc[i, 'status'] != 'Downloaded':
                product_info = self.api.get_product_odata(self.req_products_df[self.req_products_df.keys()[0]][i])
                if product_info['Online']:
                    self.online_file_ID_list.append(self.req_products_df[self.req_products_df.keys()[0]][i])
                    self.req_products_df.loc[i, 'status'] = 'Online'
                else:
                    self.offline_file_ID_list.append(self.req_products_df[self.req_products_df.keys()[0]][i])
                    self.req_products_df.loc[i, 'status'] = 'Offline'
        self.req_products_df.to_csv(self.queried_folder + 'queried_products.csv')

    def download_with_IDM(self, output_path=None):
        if self.req_products_df is None:
            print('Please query the files before the download!')
            sys.exit(-1)

        if self.IDM_path is None:
            print('Please input the IDM env before initiate the download')
            return
        if output_path is not None:
            bf.check_file_path(output_path)
            bf.create_folder(output_path)
            self.download_path = output_path

        for file_ID in self.online_file_ID_list:
            print(file_ID + 'is online')
            print('File name is ：' + self.req_products_df[self.req_products_df[self.req_products_df.keys()[0]] == file_ID]['title'].tolist()[0])
            print('Add to IDM: ' + self.req_products_df[self.req_products_df[self.req_products_df.keys()[0]] == file_ID]['link'].tolist()[0])
            print('---------------------Start to download-----------------------')
            try:
                call([self.IDM_path, '/d', self.req_products_df[self.req_products_df[self.req_products_df.keys()[0]] == file_ID]['link'].tolist()[0], '/p', self.download_path, '/n', '/a'])
                call([self.IDM_path, '/s'])
            except:
                self.failure_file.append(file_ID)
            # remove download files

        internal_file_list = []
        timers = 1
        while self.offline_file_ID_list != []:
            if timers < 4:
                file_num = 0
                while file_num < len(self.offline_file_ID_list) + 1 and len(self.offline_file_ID_list) != 0:
                    # for file_ID in self.offline_file_ID_list:
                    file_ID = self.offline_file_ID_list[file_num]
                    try:
                        print('Retrieving ' + self.req_products_df[self.req_products_df[self.req_products_df.keys()[0]] == file_ID]['title'].tolist()[0])
                        file_status = self.api.trigger_offline_retrieval(file_ID)
                        if file_status:
                            internal_file_list.append(file_ID)
                            self.offline_file_ID_list.remove(file_ID)
                            print('Succeed!')
                        else:
                            product_info = self.api.get_product_odata(file_ID)
                            if product_info['Online']:
                                self.offline_file_ID_list.remove(file_ID)
                                internal_file_list.append(file_ID)
                            else:
                                self.offline_file_ID_list.remove(file_ID)
                                self.failure_file.append(file_ID)
                                print('Failed for unknown reason!')

                    except (LTAError):
                        timers = 4
                        break

                    except:
                        self.offline_file_ID_list.remove(file_ID)
                        self.failure_file.append(file_ID)
                        print('Failed for unknown reason!')

            else:
                for _ in tqdm(range(int(1500)), ncols=100):
                    time.sleep(2)

                for file_ID in internal_file_list:
                    try:
                        call([self.IDM_path, '/d', self.req_products_df[self.req_products_df[self.req_products_df.keys()[0]] == file_ID]['link'].tolist()[0], '/p', self.download_path, '/n', '/a'])
                        call([self.IDM_path, '/s'])
                    except:
                        self.failure_file.append(file_ID)
                timers = 1
                internal_file_list = []
        np.save(self.queried_folder + 'failure_file.npy', self.failure_file)
        self.process_failure_file()

    def process_failure_file(self):
        self.update_online_status()
        failure_file = bf.file_filter(self.work_env, containing_word_list=['failure_file.npy'], subfolder_detection=True)
        self.failure_file = np.load(failure_file[0])


if __name__ == "__main__":
    # Parameters
    IDM = "C:\\Program Files (x86)\\Internet Download Manager\\IDMan.exe"
    DownPath = 'g:\\sentinel2_download\\'
    shpfile_path = 'E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain\\Floodplain_2020_simplified4.shp'

    S2_MID_YZR = Queried_Sentinel_ds('shixi2ng', 'shixi2nG', DownPath, IDM_path=IDM)
    S2_MID_YZR.queried_with_ROI(shpfile_path, ('20190101', '20191231'),'Sentinel-2', 'S2MSI2A',(0, 95), overwritten_factor=True)
    S2_MID_YZR.download_with_IDM()

