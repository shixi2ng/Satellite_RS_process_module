import traceback
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
import requests
import json
import concurrent.futures
from collections import OrderedDict
from Sentinel2_toolbox.utils import bulk_download_sentinel_files
from itertools import repeat

# This Program is mainly used to batch download Sentinel-2 image using IDM
# Several things you need to make sure before the download
# (1) Successfully download and install IDM as well as sentinelsat package
# (2) Add username and password for the scihub url in the setting interface of IDM


class Queried_Sentinel_ds_ODATA(object):
    def __init__(self, username, password, work_env, IDM_path=None):

        self.define_IDM_path(IDM_path)
        # Construct work env
        if os.path.exists(work_env):
            self.work_env = bf.Path(work_env).path_name
        else:
            print('Please input a valid work environment')
            sys.exit(-1)

        # Define the
        self._data_collection = ('SENTINEL-1', 'SENTINEL-2', 'SENTINEL-3', 'SENTINEL-5P')
        self._product_types = ['S2MSI2A']

        # Define variable
        self.req_products = None
        self.queried_folder = None
        self.download_path = None
        self.downloaded_file = []
        self.req_products_df = None
        self.online_file_ID_list = []
        self.offline_file_ID_list = []
        self.failure_file = []

        # Initialise API
        self.username = username
        self.password = password
        self.access_token = self._get_access_token_()

    def define_IDM_path(self, IDM_path):
        if os.path.exists(IDM_path) and IDM_path.endswith('.exe'):
            self.IDM_path = IDM_path
        else:
            print('The IDM path is not INPUT!')
            self.IDM_path = None

    def _get_access_token_(self):
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }
        try:
            r = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data=data,
            )
            r.raise_for_status()
        except Exception as e:
            raise Exception(
                f"Access token creation failed. Reponse from the server was: {r.json()}"
            )
        return r.json()["access_token"]

    def queried_with_ROI(self, shpfile_path, date_range, data_collection, product_type, cloud_cover_range, overwritten_factor=False):

        if data_collection not in self._data_collection:
            raise Exception(f'The data collection {str(data_collection)} is not supported!')

        if product_type not in self._product_types:
            raise Exception(f'The product type {str(product_type)} is not supported!')

        try:
            start_date = f'{str(date_range[0])[0:4]}-{str(date_range[0])[4:6]}-{str(date_range[0])[6:]}'
            end_date = f'{str(date_range[1])[0:4]}-{str(date_range[1])[4:6]}-{str(date_range[1])[6:]}'
        except:
            raise TypeError('The data range should under the format like (YYYYMMDD, YYYYMMDD)')

        aoi = str(gp.read_file(shpfile_path)['geometry'][0])
        aoi = 'POLYGON' + aoi[8:]

        shpfile_path = bf.Path(shpfile_path).path_name
        # Create queried folder
        self.queried_folder = self.work_env + shpfile_path.split('\\')[-1].split('.shp')[0] + '_' + date_range[0] + '_' + date_range[1] + '_' + data_collection + '\\queried_log\\'
        self.download_path = self.queried_folder.split('queried_log')[0] + 'download_files\\'
        bf.create_folder(self.queried_folder)
        bf.create_folder(self.download_path)
        self.downloaded_file = bf.file_filter(self.download_path, ['.zip'])

        # Construct req product
        if not os.path.exists(self.queried_folder + 'queried_products.csv') or overwritten_factor:
            json = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}') and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{str(product_type)}') and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {str(cloud_cover_range[1])})&$top=1000").json()
            self.req_products_df = pd.DataFrame.from_dict(json['value']).head(1000)

    def download_with_request(self, output_path=None, bulk_size=4):

        bulk_size = min(bulk_size, os.cpu_count())
        if self.req_products_df is None:
            raise Exception('Please query the files before the download!')

        if self.IDM_path is None:
            raise Exception('Please input the IDM env before initiate the download')

        if output_path is not None:
            bf.check_file_path(output_path)
            bf.create_folder(output_path)
            self.download_path = output_path

        dfs = []
        size = int(np.ceil(self.req_products_df.shape[0] / bulk_size))
        for _ in range(bulk_size):
            if _ == bulk_size - 1:
                dfs.append(self.req_products_df[size * _: ].reset_index(drop=True))
            else:
                dfs.append(self.req_products_df[size * _: size * (_ + 1)].reset_index(drop=True))

        with concurrent.futures.ProcessPoolExecutor(max_workers = bulk_size) as exe:
            res = exe.map(bulk_download_sentinel_files, dfs, repeat(self.access_token), repeat(self.download_path))

    def process_failure_file(self):
        failure_file = bf.file_filter(self.work_env, containing_word_list=['failure_file.npy'], subfolder_detection=True)
        self.failure_file = np.load(failure_file[0])


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
        self.api = SentinelAPI(username, password, 'https://apihub.copernicus.eu/apihub')

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
                print(traceback.format_exc())
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
        #     for product_info, file_ID in zip(executor.map(thalweg_temp.api.get_product_odata, thalweg_temp.req_products.keys()), thalweg_temp.req_products.keys()):
        #         if product_info['Online']:
        #             thalweg_temp.online_file_ID_list.append(file_ID)
        #         else:
        #             thalweg_temp.offline_file_ID_list.append(file_ID)

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
                call([self.IDM_path, '/d', self.req_products_df[self.req_products_df[self.req_products_df.keys()[0]] == file_ID]['link'].tolist()[0], '/p', self.download_path,  '/a'])
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
                    # for file_ID in thalweg_temp.offline_file_ID_list:
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

