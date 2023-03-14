# coding=utf-8
import gdal
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import shutil
import scipy.sparse as sp
from datetime import date
import rasterio
import math
import copy
import seaborn as sns
from scipy.optimize import curve_fit
import time
from scipy import ndimage
from basic_function import Path
import basic_function as bf
from functools import wraps
import concurrent.futures
from itertools import repeat
from zipfile import ZipFile
import traceback
import GEDI_process as gedi
import pywt
import psutil
import pickle
import geopandas as gp
from scipy import sparse as sm
from utils import no_nan_mean, log_para, retrieve_srs, write_raster, union_list, remove_all_file_and_folder, create_circle_polygon, extract_value2shpfile


# Input Snappy data style
np.seterr(divide='ignore', invalid='ignore')


class NDSparseMatrix:
    def __init__(self, *args, **kwargs):
        self.SM_group = {}
        self._matrix_type = None
        self._cols = -1
        self._rows = -1
        self._height = -1
        self.shape = [self._rows, self._cols, self._height]
        self.SM_namelist = None
        self.file_size = 0

        for kw_temp in kwargs.keys():
            if kw_temp not in ['SM_namelist']:
                raise KeyError(f'Please input the a right kwargs!')
            else:
                self.__dict__[kw_temp] = kwargs[kw_temp]

        if 'SM_namelist' in self.__dict__.keys():
            if self.SM_namelist is not None:
                if type(self.SM_namelist) is list:
                    if len(args) == len(self.SM_namelist):
                        raise ValueError(f'Please make sure the sm name list is consistent with the SM_group')
                else:
                    raise TypeError(f'Please input the SM_namelist under the list type!')
            else:
                if len(args) == 0:
                    self.SM_namelist = []
                else:
                    self.SM_namelist = [i for i in range(len(args))]
        else:
            if len(args) == 0:
                self.SM_namelist = []
            else:
                self.SM_namelist = [i for i in range(len(args))]
        i = 0
        for ele in args:
            if type(ele) not in (sm.spmatrix, sm.csr_matrix, sm.csc_matrix, sm.coo_matrix, sm.bsr_matrix, sm.dia_matrix, sm.dok_matrix):
                raise TypeError(f'Please input the {str(ele)} under sparse matrix type!')
            else:
                if self._matrix_type is None:
                    self._matrix_type = type(ele)
                elif self._matrix_type != type(ele):
                    raise TypeError(f'The {str(ele)} is not under the common type {str(self._matrix_type)}!')
                self.SM_group[self.SM_namelist[i]] = ele
            i += 1

        self._update_size_para()

    def _update_size_para(self):
        self.size = 0
        for ele in self.SM_group.values():
            if self._cols == -1 or self._rows == -1:
                self._cols, self._rows = ele.shape[1], ele.shape[0]
            elif ele.shape[1] != self._cols or ele.shape[0] != self._rows:
                raise Exception(f'Consistency Error for the {str(ele)}')
            self.size += len(pickle.dumps(ele))
        self._height = len(self.SM_namelist)
        self.shape = [self._rows, self._cols, self._height]

    def append(self, sm_matrix, name=None, pos=-1):
        if type(sm_matrix) not in (sm.spmatrix, sm.csr_matrix, sm.csc_matrix, sm.coo_matrix, sm.bsr_matrix, sm.dia_matrix, sm.dok_matrix):
            raise TypeError(f'The new sm_matrix is not a sm_matrix')
        elif type(sm_matrix) != self._matrix_type:
            if self._matrix_type is None:
                self._matrix_type = type(sm_matrix)
            else:
                raise TypeError(f'The new sm_matirx is not under the same type within the 3d sm matrix')

        if name is None:
            try:
                name = int(self.SM_namelist[-1]) + 1
            except:
                name = 0

        if pos == -1:
            self.SM_namelist.append(name)
        elif pos not in range(len(self.SM_namelist)):
            print(f'The pos{str(pos)} is not in the range')
            self.SM_namelist.append(name)
        else:
            self.SM_namelist.insert(pos, name)

        self.SM_group[name] = sm_matrix
        self._update_size_para()

    def extend(self, sm_matrix_list, name=None):
        if type(sm_matrix_list) is not list:
            raise TypeError(f'Please input the sm_matrix_list as a list!')

        if name is not None and type(name) is not list:
            raise TypeError(f'Please input the name list as a list!')
        elif len(name) != len(sm_matrix_list):
            raise Exception(f'Consistency error')

        i = 0
        for sm_matrix in sm_matrix_list:
            if name is None:
                self.append(sm_matrix, name=name)
            else:
                self.append(sm_matrix, name=name[i])
            i += 1

    def save(self, output_path):
        bf.create_folder(output_path)
        output_path = bf.Path(output_path).path_name
        i = 0
        for sm_name in self.SM_namelist:
            sm.save_npz(output_path + sm_name + '.npz', self.SM_group[sm_name])
            i += 1
        np.save(output_path + 'SMsequence.npz', np.array(self.SM_namelist))

    def load(self, input_path):

        input_path = bf.Path(input_path).path_name
        file_list = bf.file_filter(input_path, ['SMsequence.npz'])

        if len(file_list) == 0:
            raise ValueError('The header file is missing！')
        elif len(file_list) > 1:
            raise ValueError('There are more than one header file！')
        else:
            try:
                header_file = np.load(file_list[0], allow_pickle=True)
            except:
                raise Exception('file cannot be loaded')

        self.SM_namelist = header_file.tolist()
        self.SM_group = {}

        for SM_name in self.SM_namelist:
            SM_arr_path = bf.file_filter(input_path, ['.npz', str(SM_name)], and_or_factor='and')
            if len(SM_arr_path) == 0:
                raise ValueError(f'The file {str(SM_name)} is missing！')
            elif len(SM_arr_path) > 1:
                raise ValueError(f'There are more than one file sharing name {str(SM_name)}')
            else:
                try:
                    SM_arr_temp = sm.load_npz(SM_arr_path[0])
                except:
                    raise Exception(f'file {str(SM_name)} cannot be loaded')
            self.SM_group[SM_name] = SM_arr_temp

        self._update_size_para()
        return self

    def replace_layer(self, ori_layer_name, new_layer, new_layer_name=None):

        if type(new_layer) not in (sm.spmatrix, sm.csr_matrix, sm.csc_matrix, sm.coo_matrix, sm.bsr_matrix, sm.dia_matrix, sm.dok_matrix):
            raise TypeError(f'The new sm_matrix is not a sm_matrix')
        elif type(new_layer) != self._matrix_type:
            raise TypeError(f'The new sm_matirx is not under the same type within the 3d sm matrix')

        if new_layer_name is None:
            if ori_layer_name not in self.SM_namelist:
                raise ValueError(f'The {ori_layer_name} cannot be found')
            else:
                self.SM_group[ori_layer_name] = new_layer
                self._update_size_para()
        else:
            self.SM_group[ori_layer_name] = new_layer
            self.SM_namelist[self.SM_namelist.index(ori_layer_name)] = new_layer_name
            self._update_size_para()

    def remove_layer(self, layer_name):
        if layer_name not in self.SM_namelist:
            raise ValueError(f'The {layer_name} cannot be found')
        else:
            self.SM_group.pop(layer_name)
            self.SM_namelist.remove(layer_name)

    def _understand_range(self, list_temp: list, range_temp: range):

        if len(list_temp) == 1:
            if list_temp[0] == 'all':
                return [min(range_temp), max(range_temp) + 1]
            elif type(list_temp[0]) is int and list_temp[0] in range_temp:
                return [list_temp[0], list_temp[0] + 1]
            else:
                raise ValueError('Please input a supported type!')

        elif len(list_temp) == 2:
            if type(list_temp[0]) is int and type(list_temp[1]) is int:
                if list_temp[0] in range_temp and list_temp[1] in range_temp and list_temp[0] <= list_temp[1]:
                    return [list_temp[0], list_temp[1] + 1]
            else:
                raise ValueError('Please input a supported type!')

        elif len(list_temp) >=2 :
            raise ValueError('Please input a supported type!')

    def slice_matrix(self, tuple_temp: tuple):
        if len(tuple_temp) != 3 or type(tuple_temp) != tuple:
            raise TypeError(f'Please input the index array in a 3D tuple')
        else:
            rows_range = self._understand_range(tuple_temp[0], range(self._rows))
            cols_range = self._understand_range(tuple_temp[1], range(self._cols))
            heights_range = self._understand_range(tuple_temp[2], range(self._height))

        try:
            output_array = np.zeros([rows_range[1]-rows_range[0], cols_range[1]-cols_range[0], heights_range[1]- heights_range[0]], dtype=np.float16)
        except MemoryError:
            return None

        for height in range(heights_range[0], heights_range[1]):
            array_temp = self.SM_group[self.SM_namelist[height]]
            array_out = array_temp[rows_range[0]:rows_range[1], cols_range[0]: cols_range[1]]
            output_array[:, :, height - heights_range[0]] = array_out.toarray()
        return output_array


class Sentinel2_ds(object):

    def __init__(self, ori_zipfile_folder, work_env=None):
        # Define var
        self.S2_metadata = None
        self._subset_failure_file = []
        self.output_bounds = np.array([])
        self.raw_10m_bounds = np.array([])
        self.ROI = None
        self.ROI_name = None
        self.ori_folder = Path(ori_zipfile_folder).path_name
        self.S2_metadata_size = np.nan
        self.date_list = []
        self.main_coordinate_system = None

        # Define key variables (kwargs)
        self._size_control_factor = False
        self._cloud_removal_para = False
        self._vi_clip_factor = False
        self._sparsify_matrix_factor = False
        self._cloud_clip_seq = None
        self._pansharp_factor = True

        # Remove all the duplicated data
        dup_data = bf.file_filter(self.ori_folder, ['.1.zip'])
        for dup in dup_data:
            os.remove(dup)

        # Generate the original zip file list
        self.orifile_list = bf.file_filter(self.ori_folder, ['.zip', 'S2'], and_or_factor='and', subfolder_detection=True)
        self.orifile_list = [i for i in self.orifile_list if 'S2' in i.split('\\')[-1] and '.zip' in i.split('\\')[-1]]
        if not self.orifile_list:
            print('There has no Sentinel zipfiles in the input dir')
            sys.exit(-1)

        # Initialise the work environment
        if work_env is None:
            try:
                self.work_env = Path(os.path.dirname(os.path.dirname(self.ori_folder)) + '\\').path_name
            except:
                print('There has no base dir for the ori_folder and the ori_folder will be treated as the work env')
                self.work_env = self.ori_folder
        else:
            self.work_env = Path(work_env).path_name

        # Create cache path
        self.cache_folder = self.work_env + 'cache\\'
        bf.create_folder(self.cache_folder)
        self.trash_folder = self.work_env + 'trash\\'
        bf.create_folder(self.trash_folder)
        bf.create_folder(self.work_env + 'Corrupted_S2_file\\')

        # Create output path
        self.output_path = f'{self.work_env}Sentinel2_L2A_Output\\'
        self.shpfile_path = f'{self.work_env}shpfile\\'
        self.log_filepath = f'{self.work_env}logfile\\'
        bf.create_folder(self.output_path)
        bf.create_folder(self.log_filepath)
        bf.create_folder(self.shpfile_path)

        # define 2dc para
        self.dc_vi = {}
        self._dc_overwritten_para = False
        self._inherit_from_logfile = None
        self._remove_nan_layer = False
        self._manually_remove_para = False
        self._manually_remove_datelist = None

        # Constant
        self.band_name_list = ['B01_60m.jp2', 'B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B05_20m.jp2', 'B06_20m.jp2',
                               'B07_20m.jp2', 'B08_10m.jp2', 'B8A_20m.jp2', 'B09_60m.jp2', 'B11_20m.jp2', 'B12_20m.jp2']
        self.band_output_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        self.all_supported_index_list = ['RGB', 'QI', 'all_band', '4visual', 'NDVI', 'MNDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI',
                                         'NDVI_RE', 'NDVI_RE2','AWEI', 'AWEInsh', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9',
                                         'B11', 'B12', 'NDVI_20m', 'OSAVI_20m']

    def save_log_file(func):
        def wrapper(self, *args, **kwargs):

            #########################################################################
            # Document the log file and para file
            # The log file contained the information for each run and para file documented the args of each func
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
            for func_key, func_processing_name in zip(['metadata', 'subset', 'ds2sdc'], ['constructing metadata', 'executing subset and clip', '2sdc']):
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

    def _retrieve_para(self, required_para_name_list, **kwargs):

        if not os.path.exists(f'{self.log_filepath}para_file.txt'):
            print('The para file is not established yet')
            sys.exit(-1)
        else:
            para_file = open(f"{self.log_filepath}para_file.txt", "r+")
            para_raw_txt = para_file.read().split('\n')

        for para in required_para_name_list:
            for q in para_raw_txt:
                para = str(para)
                if q.startswith(para + ':'):
                    if q.split(para + ':')[-1] == 'None':
                        self.__dict__[para] = None
                    elif q.split(para + ':')[-1] == 'True':
                        self.__dict__[para] = True
                    elif q.split(para + ':')[-1] == 'False':
                        self.__dict__[para] = False
                    elif q.split(para + ':')[-1].startswith('['):
                        self.__dict__[para] = list(q.split(para + ':')[-1][1: -1])
                    elif q.split(para + ':')[-1].startswith('('):
                        self.__dict__[para] = tuple(q.split(para + ':')[-1][1: -1])
                    else:
                        try:
                            t = float(q.split(para + ':')[-1])
                            self.__dict__[para] = float(q.split(para + ':')[-1])
                        except:
                            self.__dict__[para] = q.split(para + ':')[-1]

    @save_log_file
    def construct_metadata(self):

        # Start constructing metadata
        print('---------------------------- Start the construction of Metadata ----------------------------')
        start_temp = time.time()

        # process input files
        if os.path.exists(self.work_env + 'Metadata.xlsx'):
            metadata_num = pd.read_excel(self.work_env + 'Metadata.xlsx').shape[0]
        else:
            metadata_num = 0

        if not os.path.exists(self.work_env + 'Metadata.xlsx') or metadata_num != len(self.orifile_list):
            corrupted_ori_file, corrupted_file_date, product_path, product_name, sensor_type, sensing_date, orbit_num, tile_num, width, height = (
                [] for i in range(10))
            corrupted_factor = 0
            for ori_file in self.orifile_list:
                try:
                    unzip_file = zipfile.ZipFile(ori_file)
                    unzip_file.close()
                    file_name = ori_file.split('\\')[-1]
                    product_path.append(ori_file)
                    sensing_date.append(file_name[file_name.find('_20') + 1: file_name.find('_20') + 9])
                    orbit_num.append(file_name[file_name.find('_R') + 2: file_name.find('_R') + 5])
                    tile_num.append(file_name[file_name.find('_T') + 2: file_name.find('_T') + 7])
                    sensor_type.append(file_name[file_name.find('S2'): file_name.find('S2') + 10])
                    # print(file_information)
                except:
                    if (not os.path.exists(self.work_env + 'Corrupted_S2_file')) and corrupted_factor == 0:
                        os.makedirs(self.work_env + 'Corrupted_S2_file')
                        corrupted_factor = 1
                    print(f'This file is corrupted {ori_file}!')
                    file_name = ori_file.split('\\')[-1]
                    corrupted_ori_file.append(file_name)
                    corrupted_file_date.append(file_name[file_name.find('_20') + 1: file_name.find('_20') + 9])

            # Move the corrupted files
            for corrupted_file_name in corrupted_ori_file:
                shutil.move(self.ori_folder + corrupted_file_name, self.work_env + 'Corrupted_S2_file\\' + corrupted_file_name)

            # Construct corrupted metadata
            Corrupted_metadata = pd.DataFrame({'Corrupted_file_name': corrupted_ori_file, 'File_Date': corrupted_file_date})
            if not os.path.exists(self.work_env + 'Corrupted_metadata.xlsx'):
                Corrupted_metadata.to_excel(self.work_env + 'Corrupted_metadata.xlsx')
            else:
                Corrupted_metadata_old_version = pd.read_excel(self.work_env + 'Corrupted_metadata.xlsx')
                Corrupted_metadata_old_version.append(Corrupted_metadata, ignore_index=True)
                Corrupted_metadata_old_version.drop_duplicates()
                Corrupted_metadata_old_version.to_excel(self.work_env + 'Corrupted_metadata.xlsx')

            self.S2_metadata = pd.DataFrame({'Product_Path': product_path, 'Sensing_Date': sensing_date,
                                             'Orbit_Num': orbit_num, 'Tile_Num': tile_num, 'Sensor_Type': sensor_type})

            # Process duplicate file
            duplicate_file_list = []
            i = 0
            while i <= self.S2_metadata.shape[0] - 1:
                file_inform = str(self.S2_metadata['Sensing_Date'][i]) + '_' + self.S2_metadata['Tile_Num'][i]
                q = i + 1
                while q <= self.S2_metadata.shape[0] - 1:
                    file_inform2 = str(self.S2_metadata['Sensing_Date'][q]) + '_' + self.S2_metadata['Tile_Num'][q]
                    if file_inform2 == file_inform:
                        if len(self.S2_metadata['Product_Path'][i]) > len(self.S2_metadata['Product_Path'][q]):
                            duplicate_file_list.append(self.S2_metadata['Product_Path'][i])
                        elif len(self.S2_metadata['Product_Path'][i]) < len(self.S2_metadata['Product_Path'][q]):
                            duplicate_file_list.append(self.S2_metadata['Product_Path'][q])
                        else:
                            if int(os.path.getsize(self.S2_metadata['Product_Path'][i])) > int(os.path.getsize(self.S2_metadata['Product_Path'][q])):
                                duplicate_file_list.append(self.S2_metadata['Product_Path'][q])
                            else:
                                duplicate_file_list.append(self.S2_metadata['Product_Path'][i])
                        break
                    q += 1
                i += 1

            duplicate_file_list = list(dict.fromkeys(duplicate_file_list))
            if duplicate_file_list != []:
                for file in duplicate_file_list:
                    shutil.move(file, self.work_env + 'Corrupted_S2_file\\' + file.split('\\')[-1])
                self.construct_metadata()
            else:
                self.S2_metadata.to_excel(self.work_env + 'Metadata.xlsx')
                self.S2_metadata = pd.read_excel(self.work_env + 'Metadata.xlsx')
        else:
            self.S2_metadata = pd.read_excel(self.work_env + 'Metadata.xlsx')
        self.S2_metadata.sort_values(by=['Sensing_Date'], ascending=True)
        self.S2_metadata_size = self.S2_metadata.shape[0]
        self.output_bounds = np.zeros([self.S2_metadata_size, 4]) * np.nan
        self.raw_10m_bounds = np.zeros([self.S2_metadata_size, 4]) * np.nan
        self.date_list = self.S2_metadata['Sensing_Date'].drop_duplicates().to_list()
        print(f'Finish in {str(time.time() - start_temp)} sec!')
        print('----------------------------  End the construction of Metadata  ----------------------------')

    def _qi_remove_cloud(self, processed_filepath, qi_filepath, serial_num, dst_nodata=0, **kwargs):
        # Determine the process parameter
        sensing_date = self.S2_metadata['Sensing_Date'][serial_num]
        tile_num = self.S2_metadata['Tile_Num'][serial_num]
        if kwargs['cloud_removal_strategy'] == 'QI_all_cloud':
            cloud_indicator = [0, 1, 2, 3, 8, 9, 10, 11]
        else:
            print('Cloud removal strategy is not supported!')
            return

        # Input ds
        try:
            qi_ds = gdal.Open(qi_filepath)
            processed_ds = gdal.Open(processed_filepath, gdal.GA_Update)
            qi_array = qi_ds.GetRasterBand(1).ReadAsArray()
            processed_array = processed_ds.GetRasterBand(1).ReadAsArray()
        except:
            print(f'Cannot read the qi array or processed array of {str(sensing_date)} {str(tile_num)}!')
            return

        if qi_array is None or processed_array is None:
            print(f'the qi array or processed array of {str(sensing_date)} {str(tile_num)} dont exist!')
            return

        elif qi_array.shape[0] != processed_array.shape[0] or qi_array.shape[1] != processed_array.shape[1]:
            print(f'Consistency issue between the qi array and processed array of {str(sensing_date)} {str(tile_num)}!')
            return

        for indicator in cloud_indicator:
            qi_array[qi_array == indicator] = 64
        qi_array[qi_array == 255] = 64
        processed_array[qi_array == 64] = dst_nodata

        processed_ds.GetRasterBand(1).WriteArray(processed_array)
        processed_ds.FlushCache()
        processed_ds = None
        qi_ds = None

    def _check_metadata_availability(self):
        if self.S2_metadata is None:
            try:
                self.construct_metadata()
            except:
                print('Please manually construct the S2_metadata before further processing!')
                sys.exit(-1)

    def _check_output_band_statue(self, band_name, tiffile_serial_num, *args, **kwargs):

        # Define local var
        sensing_date = self.S2_metadata['Sensing_Date'][tiffile_serial_num]
        tile_num = self.S2_metadata['Tile_Num'][tiffile_serial_num]

        # Factor configuration
        if True in [band_temp not in self.band_output_list for band_temp in band_name]:
            print(f'Band {band_name} is not valid!')
            sys.exit(-1)

        if self._vi_clip_factor:
            output_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\all_band\\'
        else:
            output_path = f'{self.output_path}Sentinel2_constructed_index\\all_band\\'
        bf.create_folder(output_path)

        # Detect whether the required band was generated before
        try:
            if False in [os.path.exists(f'{output_path}{str(sensing_date)}_{str(tile_num)}_{band_temp}.TIF') for band_temp in band_name]:
                if 'combine_band_factor' in kwargs.keys():
                    kwargs['combine_band_factor'] = False
                self.subset_tiffiles(band_name, tiffile_serial_num, **kwargs)

            # Return output
            if False in [os.path.exists(f'{output_path}{str(sensing_date)}_{str(tile_num)}_{band_temp}.TIF') for band_temp in band_name]:
                print(f'Something error processing {band_name}!')
                return None
            else:
                return [gdal.Open(f'{output_path}{str(sensing_date)}_{str(tile_num)}_{band_temp}.TIF') for band_temp in band_name]

        except:
            return None

    @save_log_file
    def mp_subset(self, *args, **kwargs):

        if 'chunk_size' in kwargs.keys() and type(kwargs['chunk_size']) is int:
            chunk_size = min(os.cpu_count(), kwargs['chunk_size'])
            del kwargs['chunk_size']
        else:
            chunk_size = os.cpu_count()

        if self.S2_metadata is None:
            print('Please construct the S2_metadata before the subset!')
            sys.exit(-1)

        i = range(self.S2_metadata.shape[0])
        # mp process
        with concurrent.futures.ProcessPoolExecutor(max_workers=chunk_size) as executor:
            results = executor.map(self.subset_tiffiles, repeat(args[0]), i, repeat(False), repeat(kwargs))

        results = list(results)
        for result in results:
            if result is not None and result[0] is not None:
                self._subset_failure_file.append(result)

        if self._subset_failure_file != []:
            self._process_subset_failure_file(args[0])

    @save_log_file
    def sequenced_subset(self, *args, **kwargs):

        if self.S2_metadata is None:
            print('Please construct the S2_metadata before the subset!')
            sys.exit(-1)

        # sequenced process
        for i in range(self.S2_metadata.shape[0]):
            [index, tiffile_serial_num, sensing_date, tile_num] = self.subset_tiffiles(args[0], i, **kwargs)
            if index is not None:
                self._subset_failure_file.append([index, tiffile_serial_num, sensing_date, tile_num])

        if self._subset_failure_file != []:
            self._process_subset_failure_file(args[0])

    def _wavelet_pansharpen(self, processed_image, pan_image_list, wavelet_level: int = 1, wavelet_type: str = 'haar'):

        # DataType check
        if type(pan_image_list) is not list:
            raise Exception('please input the pan image list as a list of array')
        elif type(processed_image) is not np.ndarray:
            raise Exception('please input the processed image as an array')

        # Consistency check
        for array_temp in pan_image_list:
            if array_temp.shape != processed_image.shape:
                raise Exception('Consistency Error')

        # Require the most correlated image
        # time1 = time.time()
        correlation_list = []
        temp = processed_image.reshape(-1)

        if np.nansum(temp) == 0:
            return processed_image
        else:
            correlation_list = []
            for array_temp in pan_image_list:
                array_temp[array_temp == 0] = np.nan
                array_temp = array_temp.reshape(-1)
                msk = (~np.ma.masked_invalid(temp).mask & ~np.ma.masked_invalid(array_temp).mask)
                correlation_list.append(np.corrcoef(temp[msk], array_temp[msk])[0][1])
            # print(f'Correlation analysis consumes {str(time.time() - time1)} s')
            most_correlated_image = pan_image_list[np.argmax(np.array(correlation_list))]
            # time2 = time.time()
            # print(f'Correlation analysis consumes {str(time2-time1)} s')

            # Pan sharpening
            pan_coef = pywt.wavedec2(most_correlated_image, wavelet_type, mode='sym', level=wavelet_level)
            processed_coef = pywt.wavedec2(processed_image, wavelet_type, mode='sym', level=wavelet_level)
            # time3 = time.time()
            # print(f'Wavelet analysis consumes {str(time3 - time2)} s')

            pan_coef[0] = processed_coef[0]
            sharped_image = pywt.waverec2(pan_coef, wavelet_type)
            # print(f'Wavelet reconstruction consumes {str(time.time() - time3)} s')
            return sharped_image

    def _subset_indicator_process(self, **kwargs):
        # Detect whether all the indicator are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ['ROI', 'ROI_name', 'size_control_factor', 'cloud_removal_strategy', 'combine_band_factor', 'pansharp_factor']:
                print(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process clip parameter
        if self.ROI is None:
            if 'ROI' in kwargs.keys():
                self._vi_clip_factor = True
                if '.shp' in kwargs['ROI'] and os.path.exists(kwargs['ROI']):
                    self.ROI = kwargs['ROI']
                else:
                    print('Please input valid shp file for clip!')
                    sys.exit(-1)

                if 'ROI_name' in kwargs.keys():
                    self.ROI_name = kwargs['ROI_name']
                else:
                    self.ROI_name = self.ROI.split('\\')[-1].split('.')[0]
            else:
                self._vi_clip_factor = False

        # process size control parameter
        if 'size_control_factor' in kwargs.keys():
            if type(kwargs['size_control_factor']) is bool:
                self._size_control_factor = kwargs['size_control_factor']
            else:
                print('Please mention the size_control_factor should be bool type!')
                self._size_control_factor = False
        else:
            self._size_control_factor = False

        # process cloud removal parameter
        if 'cloud_removal_strategy' in kwargs.keys():
            self._cloud_removal_para = True
        else:
            self._cloud_removal_para = False

        # process main_coordinate_system
        if 'main_coordinate_system' in kwargs.keys():
            self.main_coordinate_system = kwargs['main_coordinate_system']
        else:
            self.main_coordinate_system = 'EPSG:32649'

        # process combine band factor
        # This factor determine whether a TIFFILE with all bands integrated is created
        if 'combine_band_factor' in kwargs.keys():
            if type(kwargs['combine_band_factor']) is bool:
                self._combine_band_factor = kwargs['combine_band_factor']
            else:
                raise Exception(f'combine band factor is not under the bool type!')
        else:
            self._combine_band_factor = False

        # Process pansharp_factor
        # This factor control whether low-resolution(60m/120m) bands are sharpened to a high-resolution(10m)
        if 'pansharp_factor' in kwargs.keys():
            if type(kwargs['pansharp_factor']) is bool:
                self._pansharp_factor = kwargs['pansharp_factor']
            else:
                raise Exception(f'pansharp_factor is not under the bool type!')
        else:
            self._pansharp_factor = False

    def generate_10m_output_bounds(self, tiffile_serial_num, **kwargs):

        # Define local var
        topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])
        sensing_date = self.S2_metadata['Sensing_Date'][tiffile_serial_num]
        tile_num = self.S2_metadata['Tile_Num'][tiffile_serial_num]
        VI = 'all_band'

        # Define the output path
        if self._vi_clip_factor:
            output_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\{VI}\\'
        else:
            output_path = f'{self.output_path}Sentinel2_constructed_index\\{VI}\\'
        bf.create_folder(output_path)

        # Create the output bounds based on the 10-m Band2 images
        if self.output_bounds.shape[0] > tiffile_serial_num:
            if True in np.isnan(self.output_bounds[tiffile_serial_num, :]):
                temp_S2file_path = self.S2_metadata.iat[tiffile_serial_num, 1]
                zfile = ZipFile(temp_S2file_path, 'r')
                b2_band_file_name = f'{str(sensing_date)}_{str(tile_num)}_B2'
                if not os.path.exists(output_path + b2_band_file_name + '.TIF'):
                    b2_file = [zfile_temp for zfile_temp in zfile.namelist() if 'B02_10m.jp2' in zfile_temp]
                    if len(b2_file) != 1:
                        print(f'Data issue for the B2 file of all_cloud data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata.shape[0])})')
                        return
                    else:
                        try:
                            ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, b2_file[0]))
                            ulx_temp, xres_temp, xskew_temp, uly_temp, yskew_temp, yres_temp = ds_temp.GetGeoTransform()
                            self.output_bounds[tiffile_serial_num, :] = np.array([ulx_temp, uly_temp + yres_temp * ds_temp.RasterYSize, ulx_temp + xres_temp * ds_temp.RasterXSize, uly_temp])
                            band_output_limit = (int(self.output_bounds[tiffile_serial_num, 0]), int(self.output_bounds[tiffile_serial_num, 1]),
                                                 int(self.output_bounds[tiffile_serial_num, 2]), int(self.output_bounds[tiffile_serial_num, 3]))
                            if self._vi_clip_factor:
                                gdal.Warp('/vsimem/' + b2_band_file_name + '.vrt', ds_temp,
                                          dstSRS=self.main_coordinate_system, xRes=10, yRes=10, cutlineDSName=self.ROI,
                                          outputType=gdal.GDT_UInt16, dstNodata=65535, outputBounds=band_output_limit)
                            else:
                                gdal.Warp('/vsimem/' + b2_band_file_name + '.vrt', ds_temp,
                                          dstSRS=self.main_coordinate_system, xRes=10, yRes=10,
                                          outputType=gdal.GDT_UInt16, dstNodata=65535, outputBounds=band_output_limit)
                            gdal.Translate(output_path + b2_band_file_name + '.TIF', '/vsimem/' + b2_band_file_name + '.vrt', options=topts, noData=65535)
                            gdal.Unlink('/vsimem/' + b2_band_file_name + '.vrt')
                        except:
                            print(f'The B2 of {str(sensing_date)}_{str(tile_num)} is not valid')
                            return
                else:
                    ds4bounds = gdal.Open(output_path + b2_band_file_name + '.TIF')
                    ulx, xres, xskew, uly, yskew, yres = ds4bounds.GetGeoTransform()
                    self.output_bounds[tiffile_serial_num, :] = np.array(
                        [ulx, uly + yres * ds4bounds.RasterYSize, ulx + xres * ds4bounds.RasterXSize, uly])
                    ds4bounds = None
        else:
            print('The output bounds has some logical issue!')
            sys.exit(-1)

    def _process_subset_failure_file(self, index_list, **kwargs):
        if self._subset_failure_file != []:
            subset_failure_file_folder = self.work_env + 'Corrupted_S2_file\\subset_failure_file\\'
            bf.create_folder(subset_failure_file_folder)
            for subset_failure_file in self._subset_failure_file:
                # remove all the related file
                related_output_file = bf.file_filter(self.output_path, [str(subset_failure_file[2]), str(subset_failure_file[3])], and_or_factor='and', subfolder_detection=True)
                for file in related_output_file:
                    file_name = file.split('\\')[-1]
                    shutil.move(file, f'{self.trash_folder}{file_name}')

            shutil.rmtree(self.trash_folder)
            bf.create_folder(self.trash_folder)

            for subset_failure_file in self._subset_failure_file:
                # remove all the zip file
                time = 1
                cannot_remove_file = []
                while True:
                    try:
                        zipfile = bf.file_filter(self.ori_folder, [str(subset_failure_file[2]), str(subset_failure_file[3])], and_or_factor='and')
                        for file in zipfile:
                            file_name = file.split('\\')[-1]
                            shutil.move(file, f'{subset_failure_file_folder}{file_name}')
                        break
                    except:
                        time += 1
                        if time >= 3:
                            cannot_remove_file.extend(zipfile)
                            break
                        pass

            if cannot_remove_file != []:
                raise Exception(f'Please remove the files manually {cannot_remove_file}')

            self.construct_metadata()

    def subset_tiffiles(self, processed_index_list, tiffile_serial_num, overwritten_para=False, *args, **kwargs):
        """
        :type processed_index_list: list
        :type tiffile_serial_num: int
        :type overwritten_para: bool

        """
        # subset_tiffiles is the core function in subsetting, resampling, clipping images as well as extracting VI and removing clouds.
        # The argument includes
        # ROI = define the path of a .shp file using for clipping all the sentinel-2 images
        # ROI_name = using to generate the roi-specified output folder, the default value is setting as the name of the ROI shp file
        # cloud_remove_strategy = method using to remove clouds, supported methods include QI_all_cloud

        # Define local args
        topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])
        time1, time2, time3 = 0, 0, 0

        # Retrieve kwargs from args using the mp
        if args != () and type(args[0]) == dict:
            kwargs = copy.copy(args[0])

        # determine the subset indicator
        self._check_metadata_availability()
        self._subset_indicator_process(**kwargs)

        # Process subset index list
        processed_index_list = union_list(processed_index_list, self.all_supported_index_list)
        processed_index_list_temp = copy.copy(processed_index_list)
        for len_t in range(len(processed_index_list)):
            if processed_index_list[len_t] in ['B2', 'B3', 'B4', 'B8', 'all_band', '4visual', 'RGB']:
                processed_index_list_temp.remove(processed_index_list[len_t])
                temp_list = [processed_index_list[len_t]]
                temp_list.extend(processed_index_list_temp)
                processed_index_list_temp = copy.copy(temp_list)
        processed_index_list = copy.copy(processed_index_list_temp)

        combine_index_list = []
        if self._combine_band_factor:
            array_cube = None
            for q in processed_index_list:
                if q in ['NDVI', 'NDVI_20m', 'OSAVI_20m', 'AWEI', 'AWEInsh', 'MNDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI', 'NDVI_RE', 'NDVI_RE2', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'QI']:
                    combine_index_list.append(q)
                elif q in ['RGB', 'all_band', '4visual']:
                    combine_index_list.extend([['B2', 'B3', 'B4'], ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'], ['B2', 'B3', 'B4', 'B5', 'B8', 'B11']][['RGB', 'all_band', '4visual'].index(q)])
            combine_index_array_list = copy.copy(combine_index_list)

        if processed_index_list != []:
            # Generate the output boundary
            sensing_date = self.S2_metadata['Sensing_Date'][tiffile_serial_num]
            tile_num = self.S2_metadata['Tile_Num'][tiffile_serial_num]

            try:
                self.generate_10m_output_bounds(tiffile_serial_num, **kwargs)
            except:
                return 'B2', tiffile_serial_num, sensing_date, tile_num

            temp_S2file_path = self.S2_metadata.iat[tiffile_serial_num, 1]
            zfile = ZipFile(temp_S2file_path, 'r')
            band_output_limit = (int(self.output_bounds[tiffile_serial_num, 0]), int(self.output_bounds[tiffile_serial_num, 1]),
                                 int(self.output_bounds[tiffile_serial_num, 2]), int(self.output_bounds[tiffile_serial_num, 3]))

            for index in processed_index_list:
                start_temp = time.time()
                print(f'Start processing \033[1;31m{index}\033[0m data of \033[3;34m{str(sensing_date)} {str(tile_num)}\033[0m ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')

                # Generate output folder
                if self._vi_clip_factor:
                    subset_output_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\{index}\\'
                    if index in self.band_output_list or index in ['4visual', 'RGB']:
                        subset_output_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\all_band\\'
                else:
                    subset_output_path = f'{self.output_path}Sentinel2_constructed_index\\{index}\\'
                    if index in self.band_output_list or index in ['4visual', 'RGB']:
                        subset_output_path = f'{self.output_path}Sentinel2_constructed_index\\all_band\\'

                # Generate qi output folder
                if self._cloud_clip_seq or not self._vi_clip_factor:
                    qi_path = f'{self.output_path}Sentinel2_constructed_index\\QI\\'
                else:
                    qi_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\QI\\'

                # Combine band to a single
                if self._combine_band_factor:
                    folder_name = ''
                    for combine_index_temp in combine_index_list:
                        folder_name = folder_name + str(combine_index_temp) + '_'
                    if self._cloud_clip_seq or not self._vi_clip_factor:
                        combine_band_folder = f'{self.output_path}Sentinel2_constructed_index\\' + folder_name[0:-1] + '\\'
                    else:
                        combine_band_folder = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\' + folder_name[0:-1] + '\\'
                    bf.create_folder(combine_band_folder)
                    if os.path.exists(f'{combine_band_folder}{str(sensing_date)}_{str(tile_num)}.npz'):
                        self._combine_band_factor, combine_index_list = False, []

                bf.create_folder(subset_output_path)
                bf.create_folder(qi_path)

                # Define the file name for VI
                file_name = f'{str(sensing_date)}_{str(tile_num)}_{index}'

                # Generate QI layer
                if (index == 'QI' and overwritten_para) or (index == 'QI' and not overwritten_para and not os.path.exists(qi_path + file_name + '.TIF')):
                    band_all = [zfile_temp for zfile_temp in zfile.namelist() if 'SCL_20m.jp2' in zfile_temp]
                    if len(band_all) != 1:
                        print(f'Something error during processing {index} data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')
                        return [index, tiffile_serial_num, sensing_date, tile_num]
                    else:
                        for band_temp in band_all:
                            try:
                                ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, band_temp))

                                if self._vi_clip_factor:
                                    gdal.Warp('/vsimem/' + file_name + 'temp.vrt', ds_temp, xRes=10, yRes=10, dstSRS=self.main_coordinate_system, outputBounds=band_output_limit, outputType=gdal.GDT_Byte, dstNodata=255)
                                    gdal.Warp('/vsimem/' + file_name + '.vrt', '/vsimem/' + file_name + 'temp.vrt', xRes=10, yRes=10, outputBounds=band_output_limit, cutlineDSName=self.ROI)
                                else:
                                    gdal.Warp('/vsimem/' + file_name + '.vrt', ds_temp, xRes=10, yRes=10, dstSRS=self.main_coordinate_system, outputBounds=band_output_limit, outputType=gdal.GDT_Byte, dstNodata=255)
                                gdal.Translate(qi_path + file_name + '.TIF', '/vsimem/' + file_name + '.vrt', options=topts, noData=255, outputType=gdal.GDT_Byte)

                                if self._combine_band_factor and 'QI' in combine_index_list:
                                    temp_ds = gdal.Open(qi_path + file_name + '.TIF')
                                    temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
                                    temp_array = temp_array.astype(np.float)
                                    temp_array[temp_array == 255] = np.nan
                                    if array_cube is None:
                                        array_cube = np.zeros([temp_array[0].shape[0], temp_array[0].shape[1], len(combine_index_array_list)], dtype=np.int16)

                                    if array_cube.shape[0] == temp_array.shape[0] and array_cube.shape[1] == temp_array.shape[1]:
                                        array_cube[:, :, combine_index_list.index('QI')] = temp_array
                                    else:
                                        print('consistency issuse')
                                        return
                                gdal.Unlink('/vsimem/' + file_name + '.TIF')

                            except:
                                print(f'The {index} of {str(sensing_date)}_{str(tile_num)} is not valid')
                                return [index, tiffile_serial_num, sensing_date, tile_num]

                elif index == 'QI' and os.path.exists(qi_path + file_name + '.TIF') and 'QI' in combine_index_list:
                    temp_ds = gdal.Open(qi_path + file_name + '.TIF')
                    temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
                    temp_array = temp_array.astype(np.float)
                    temp_array[temp_array == 255] = np.nan
                    if array_cube is None:
                        array_cube = np.zeros([temp_array[0].shape[0], temp_array[0].shape[1], len(combine_index_array_list)], dtype=np.int16)

                    if array_cube.shape[0] == temp_array.shape[0] and array_cube.shape[1] == temp_array.shape[1]:
                        array_cube[:, :, combine_index_list.index('QI')] = temp_array
                    else:
                        print('consistency issuse')
                        return

                # Subset band images
                elif index == 'all_band' or index == '4visual' or index == 'RGB' or index in self.band_output_list:
                    # Check the output band
                    if index == 'all_band':
                        band_name_list, band_output_list = self.band_name_list, self.band_output_list
                    elif index == '4visual':
                        band_name_list, band_output_list = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B08_10m.jp2', 'B05_20m.jp2', 'B11_20m.jp2'], ['B2', 'B3', 'B4', 'B8', 'B5', 'B11']
                    elif index == 'RGB':
                        band_name_list, band_output_list = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2'], ['B2', 'B3', 'B4']
                    elif index in self.band_output_list:
                        band_name_list, band_output_list = [self.band_name_list[self.band_output_list.index(index)]], [index]
                    else:
                        print('Code error!')
                        sys.exit(-1)

                    if overwritten_para or False in [os.path.exists(subset_output_path + str(sensing_date) + '_' + str(tile_num) + '_' + str(band_temp) + '.TIF') for band_temp in band_output_list] or (self._combine_band_factor and True in [band_index_temp in band_output_list for band_index_temp in combine_index_list]):
                        for band_name, band_output in zip(band_name_list, band_output_list):
                            if band_output != 'B2':
                                if self._pansharp_factor and band_output in ['B1', 'B5', 'B6', 'B7', 'B8A', 'B9', 'B11', 'B12'] and False in [os.path.exists(f'{subset_output_path}{str(sensing_date)}_{str(tile_num)}_{str(band_t)}.TIF') for band_t in ['B2', 'B3', 'B4', 'B8']]:
                                    band4sharpen = []
                                    for band_t in ['B2', 'B3', 'B4', 'B8']:
                                        if not os.path.exists(f'{subset_output_path}{str(sensing_date)}_{str(tile_num)}_{str(band_t)}.TIF'):
                                            band4sharpen.append(band_t)
                                    if band4sharpen == []:
                                        raise Exception('Code Error')
                                    else:
                                        if 'combine_band_factor' in kwargs.keys():
                                            kwargs['combine_band_factor'] = False
                                        self.subset_tiffiles(band4sharpen, tiffile_serial_num, *args, **kwargs)
                                else:
                                    all_band_file_name = f'{str(sensing_date)}_{str(tile_num)}_{str(band_output)}'
                                    if not os.path.exists(subset_output_path + all_band_file_name + '.TIF') or overwritten_para:
                                        band_all = [zfile_temp for zfile_temp in zfile.namelist() if band_name in zfile_temp]
                                        if len(band_all) != 1:
                                            print(f'Something error during processing {band_output} of {index} data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')
                                            return [index, tiffile_serial_num, sensing_date, tile_num]
                                        else:
                                            for band_temp in band_all:
                                                try:
                                                    ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, band_temp))
                                                    t1 = time.time()
                                                    if band_output in ['B1', 'B9', 'B11', 'B10', 'B12']:
                                                        if self._vi_clip_factor:
                                                            gdal.Warp('/vsimem/' + all_band_file_name + '.vrt', ds_temp,
                                                                      xRes=10, yRes=10, dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI,
                                                                      outputBounds=band_output_limit, outputType=gdal.GDT_UInt16,
                                                                      dstNodata=65535, resampleAlg=gdal.GRA_Bilinear)
                                                        else:
                                                            gdal.Warp('/vsimem/' + all_band_file_name + '.vrt', ds_temp,
                                                                      xRes=10, yRes=10, dstSRS=self.main_coordinate_system,
                                                                      outputBounds=band_output_limit, outputType=gdal.GDT_UInt16,
                                                                      dstNodata=65535, resampleAlg=gdal.GRA_Bilinear)
                                                    else:
                                                        if self._vi_clip_factor:
                                                            gdal.Warp('/vsimem/' + all_band_file_name + '.vrt', ds_temp,
                                                                      xRes=10, yRes=10, dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI,
                                                                      outputBounds=band_output_limit, outputType=gdal.GDT_UInt16,
                                                                      dstNodata=65535)
                                                        else:
                                                            gdal.Warp('/vsimem/' + all_band_file_name + '.vrt', ds_temp,
                                                                      xRes=10, yRes=10, dstSRS=self.main_coordinate_system,
                                                                      outputBounds=band_output_limit, outputType=gdal.GDT_UInt16,
                                                                      dstNodata=65535)
                                                    time1 = time.time() - t1

                                                    t2 = time.time()
                                                    if self._cloud_removal_para:
                                                        qi_file_path = f'{qi_path}{str(sensing_date)}_{str(tile_num)}_QI.TIF'
                                                        if not os.path.exists(qi_file_path):
                                                            if 'combine_band_factor' in kwargs.keys():
                                                                kwargs['combine_band_factor'] = False
                                                            self.subset_tiffiles(['QI'], tiffile_serial_num, **kwargs)
                                                        try:
                                                            self._qi_remove_cloud('/vsimem/' + all_band_file_name + '.vrt',
                                                                            qi_file_path, tiffile_serial_num, dst_nodata=65535,
                                                                            sparse_matrix_factor=self._sparsify_matrix_factor,
                                                                            **kwargs)
                                                        except:
                                                            return [None, None, None, None]
                                                    time2 = time.time() - t1

                                                    t3 = time.time()
                                                    if self._pansharp_factor and band_output in ['B11', 'B12']:
                                                        high_resolution_image_list = []
                                                        for high_res_band in ['B2', 'B3', 'B4', 'B8']:
                                                            if os.path.exists(f'{subset_output_path}{str(sensing_date)}_{str(tile_num)}_{high_res_band}.TIF'):
                                                                ds_t = gdal.Open(f'{subset_output_path}{str(sensing_date)}_{str(tile_num)}_{high_res_band}.TIF')
                                                                array_t = ds_t.GetRasterBand(1).ReadAsArray()
                                                                array_t = array_t.astype(np.float)
                                                                array_t[array_t == 65535] = 0
                                                                high_resolution_image_list.append(array_t)

                                                        if high_resolution_image_list == []:
                                                            print('Something went wrong for the code in pan sharpening')
                                                        else:
                                                            process_ds = gdal.Open('/vsimem/' + all_band_file_name + '.vrt', gdal.GA_Update)
                                                            process_image = process_ds.GetRasterBand(1).ReadAsArray()
                                                            process_image = process_image.astype(np.float)
                                                            process_image[process_image == 65535] = 0
                                                            output_image = self._wavelet_pansharpen(process_image, high_resolution_image_list)
                                                            # output_image = ATPRK.ATPRK_PANsharpen(process_image, high_resolution_image_list)
                                                            output_image = np.round(output_image)
                                                            output_image = output_image.astype(np.int)
                                                            output_image[output_image == 0] = 65535
                                                            process_ds.GetRasterBand(1).WriteArray(output_image)
                                                            process_ds.FlushCache()
                                                            process_ds = None
                                                    gdal.Translate(subset_output_path + all_band_file_name + '.TIF',
                                                                   '/vsimem/' + all_band_file_name + '.vrt', options=topts,
                                                                   noData=65535)
                                                    time3 = time.time() - t3

                                                    if self._combine_band_factor and band_output in combine_index_list:
                                                        temp_ds = gdal.Open(subset_output_path + all_band_file_name + '.TIF')
                                                        temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
                                                        temp_array = temp_array.astype(np.float)
                                                        temp_array[temp_array == 65535] = np.nan
                                                        if array_cube is None:
                                                            array_cube = np.zeros(
                                                                [temp_array[0].shape[0], temp_array[0].shape[1],
                                                                 len(combine_index_array_list)], dtype=np.int16)

                                                        if array_cube.shape[0] == temp_array.shape[0] and \
                                                                array_cube.shape[1] == temp_array.shape[1]:
                                                            array_cube[:, :,
                                                            combine_index_list.index(band_output)] = temp_array
                                                        else:
                                                            print('consistency issuse')
                                                            return

                                                    gdal.Unlink('/vsimem/' + all_band_file_name + '.TIF')
                                                    time2 = time.time() - t2
                                                    print(f'Subset {band_output} of {str(sensing_date)}_{str(tile_num)} consumes {str(time1)[0:5]}s, remove cloud consumes {str(time2)[0:5]}s, pan-sharpening consumes {str(time3)[0:5]}s!')
                                                except:
                                                    print(f'The {band_output} of {str(sensing_date)}_{str(tile_num)} is not valid')
                                                    return [index, tiffile_serial_num, sensing_date, tile_num]

                                    elif os.path.exists(subset_output_path + all_band_file_name + '.TIF') and self._combine_band_factor and band_output in combine_index_list:
                                        temp_ds = gdal.Open(subset_output_path + all_band_file_name + '.TIF')
                                        temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
                                        temp_array = temp_array.astype(np.float)
                                        temp_array[temp_array == 65535] = np.nan
                                        if array_cube is None:
                                            array_cube = np.zeros([temp_array[0].shape[0], temp_array[0].shape[1], len(combine_index_array_list)], dtype=np.int16)

                                        if array_cube.shape[0] == temp_array.shape[0] and array_cube.shape[1] == temp_array.shape[1]:
                                            array_cube[:, :, combine_index_list.index(band_output)] = temp_array
                                        else:
                                            print('consistency issuse')
                                            return

                            else:
                                if not os.path.exists(f'{subset_output_path}\\{str(sensing_date)}_{str(tile_num)}_B2.TIF'):
                                    print('Code error for B2!')
                                    sys.exit(-1)
                                else:
                                    if False in [os.path.exists(
                                        subset_output_path + str(sensing_date) + '_' + str(tile_num) + '_' + str(
                                            band_temp) + '.TIF') for band_temp in band_output_list]:
                                        if self._cloud_removal_para:
                                            qi_file_path = f'{qi_path}{str(sensing_date)}_{str(tile_num)}_QI.TIF'
                                            if not os.path.exists(qi_file_path):
                                                if 'combine_band_factor' in kwargs.keys():
                                                    kwargs['combine_band_factor'] = False
                                                self.subset_tiffiles(['QI'], tiffile_serial_num, **kwargs)
                                            self._qi_remove_cloud(
                                                f'{subset_output_path}\\{str(sensing_date)}_{str(tile_num)}_B2.TIF',
                                                qi_file_path, tiffile_serial_num, dst_nodata=65535,
                                                sparse_matrix_factor=self._sparsify_matrix_factor, **kwargs)

                                    elif self._combine_band_factor and 'B2' in combine_index_list:
                                        temp_ds = gdal.Open(f'{subset_output_path}\\{str(sensing_date)}_{str(tile_num)}_B2.TIF')
                                        temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
                                        temp_array = temp_array.astype(np.float)
                                        temp_array[temp_array == 65535] = np.nan
                                        if array_cube is None:
                                            array_cube = np.zeros([temp_array[0].shape[0], temp_array[0].shape[1], len(combine_index_array_list)], dtype=np.int16)

                                        if array_cube.shape[0] == temp_array.shape[0] and array_cube.shape[1] == temp_array.shape[1]:
                                            array_cube[:, :, combine_index_list.index('B2')] = temp_array
                                        else:
                                            print('consistency issuse')
                                            return

                    if index == 'RGB':
                        gamma_coef = 1.5
                        if self._vi_clip_factor:
                            rgb_output_path = f'{self.output_path}Sentinel2_{self.ROI_name}_index\\RGB\\'
                        else:
                            rgb_output_path = f'{self.output_path}Sentinel2_constructed_index_index\\RGB\\'
                        bf.create_folder(rgb_output_path)

                        if not os.path.exists(f'{rgb_output_path}{str(sensing_date)}_{str(tile_num)}_RGB.tif') or overwritten_para:
                            b2_file = bf.file_filter(subset_output_path, containing_word_list=[f'{str(sensing_date)}_{str(tile_num)}_B2'])
                            b3_file = bf.file_filter(subset_output_path, containing_word_list=[f'{str(sensing_date)}_{str(tile_num)}_B3'])
                            b4_file = bf.file_filter(subset_output_path, containing_word_list=[f'{str(sensing_date)}_{str(tile_num)}_B4'])
                            b2_ds = gdal.Open(b2_file[0])
                            b3_ds = gdal.Open(b3_file[0])
                            b4_ds = gdal.Open(b4_file[0])
                            b2_array = b2_ds.GetRasterBand(1).ReadAsArray()
                            b3_array = b3_ds.GetRasterBand(1).ReadAsArray()
                            b4_array = b4_ds.GetRasterBand(1).ReadAsArray()

                            b2_array = ((b2_array / 10000) ** (1/gamma_coef) * 255).astype(np.int)
                            b3_array = ((b3_array / 10000) ** (1/gamma_coef) * 255).astype(np.int)
                            b4_array = ((b4_array / 10000) ** (1/gamma_coef) * 255).astype(np.int)

                            b2_array[b2_array > 255] = 0
                            b3_array[b3_array > 255] = 0
                            b4_array[b4_array > 255] = 0

                            dst_ds = gdal.GetDriverByName('GTiff').Create(f'{rgb_output_path}{str(sensing_date)}_{str(tile_num)}_RGB.tif', xsize=b2_array.shape[1], ysize=b2_array.shape[0], bands=3, eType=gdal.GDT_Byte, options=['COMPRESS=LZW', 'PREDICTOR=2'])
                            dst_ds.SetGeoTransform(b2_ds.GetGeoTransform())  # specify coords
                            dst_ds.SetProjection(b2_ds.GetProjection())  # export coords to file
                            dst_ds.GetRasterBand(1).WriteArray(b4_array)
                            dst_ds.GetRasterBand(1).SetNoDataValue(0)# write r-band to the raster
                            dst_ds.GetRasterBand(2).WriteArray(b3_array)
                            dst_ds.GetRasterBand(2).SetNoDataValue(0) # write g-band to the raster
                            dst_ds.GetRasterBand(3).WriteArray(b2_array)
                            dst_ds.GetRasterBand(3).SetNoDataValue(0)# write b-band to the raster
                            dst_ds.FlushCache()  # write to disk
                            dst_ds = None

                elif not (index == 'QI' or index == 'all_band' or index == '4visual' or index in self.band_output_list):
                    index_construction_indicator = False
                    if not overwritten_para and not os.path.exists(subset_output_path + file_name + '.TIF'):
                        if index == 'NDVI':
                            # time1 = time.time()
                            ds_list = self._check_output_band_statue(['B8', 'B4'], tiffile_serial_num, **kwargs)
                            # print('process b8 and b4' + str(time.time() - time1))
                            if ds_list is not None:
                                try:
                                    if self._sparsify_matrix_factor:
                                        B8_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B4_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                    else:
                                        B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    # print(time.time()-time1)
                                    B8_array[B8_array == 65535] = np.nan
                                    B4_array[B4_array == 65535] = np.nan
                                    output_array = (B8_array - B4_array) / (B8_array + B4_array)
                                    B4_array = None
                                    B8_array = None
                                    # print(time.time()-time1)
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'NDVI_20m':
                            time1 = time.time()
                            ds_list = self._check_output_band_statue(['B8A', 'B4'], tiffile_serial_num, **kwargs)
                            print('reload b8 and b4' + str(time.time() - time1))
                            if ds_list is not None:
                                time2 = time.time()
                                try:
                                    if self._sparsify_matrix_factor:
                                        B8A_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B4_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                    else:
                                        B8A_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    # print(time.time()-time1)
                                    B8A_array[B8A_array == 65535] = np.nan
                                    B4_array[B4_array == 65535] = np.nan
                                    output_array = (B8A_array - B4_array) / (B8A_array + B4_array)
                                    B4_array = None
                                    B8A_array = None
                                    print('generate output array' + str(time.time() - time2))
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'OSAVI_20m':
                            # time1 = time.time()
                            ds_list = self._check_output_band_statue(['B8A', 'B4'], tiffile_serial_num, **kwargs)
                            # print('process b8 and b4' + str(time.time() - time1))
                            if ds_list is not None:
                                try:
                                    if self._sparsify_matrix_factor:
                                        B8A_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B4_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                    else:
                                        B8A_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    # print(time.time()-time1)
                                    B8A_array[B8A_array == 65535] = np.nan
                                    B4_array[B4_array == 65535] = np.nan
                                    output_array = 1.16 * (B8A_array - B4_array) / (B8A_array + B4_array + 0.16)
                                    B4_array = None
                                    B8A_array = None
                                    # print(time.time()-time1)
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'AWEI':
                            ds_list = self._check_output_band_statue(['B3', 'B11', 'B8', 'B12'], tiffile_serial_num, **kwargs)
                            if ds_list is not None:
                                try:
                                    if self._sparsify_matrix_factor:
                                        B3_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B11_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B8_array = sp.csr_matrix(ds_list[2].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B12_array = sp.csr_matrix(ds_list[3].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                    else:
                                        B3_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B11_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B8_array = ds_list[2].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B12_array = ds_list[3].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B3_array[B3_array == 65535] = np.nan
                                    B11_array[B11_array == 65535] = np.nan
                                    B8_array[B8_array == 65535] = np.nan
                                    B12_array[B12_array == 65535] = np.nan
                                    output_array = 4 * (B3_array - B11_array) - (0.25 * B8_array + 2.75 * B12_array)
                                    B3_array = None
                                    B11_array = None
                                    B8_array = None
                                    B12_array = None
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'AWEInsh':
                            ds_list = self._check_output_band_statue(['B3', 'B11', 'B8', 'B12', 'B2'], tiffile_serial_num,
                                                                     **kwargs)
                            if ds_list is not None:
                                try:
                                    if self._sparsify_matrix_factor:
                                        B3_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B11_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B8_array = sp.csr_matrix(ds_list[2].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B12_array = sp.csr_matrix(ds_list[3].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B2_array = sp.csr_matrix(ds_list[4].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                    else:
                                        B3_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B11_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B8_array = ds_list[2].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B12_array = ds_list[3].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B2_array = ds_list[4].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B3_array[B3_array == 65535] = np.nan
                                    B11_array[B11_array == 65535] = np.nan
                                    B8_array[B8_array == 65535] = np.nan
                                    B12_array[B12_array == 65535] = np.nan
                                    B2_array[B2_array == 65535] = np.nan
                                    output_array = B2_array + 2.5 * B3_array - 0.25 * B12_array - 1.5 * (B8_array + B11_array)
                                    B3_array = None
                                    B11_array = None
                                    B8_array = None
                                    B2_array = None
                                    B12_array = None
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'MNDWI':
                            ds_list = self._check_output_band_statue(['B3', 'B11'], tiffile_serial_num, **kwargs)
                            if ds_list is not None:
                                try:
                                    if self._sparsify_matrix_factor:
                                        B3_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B11_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                    else:
                                        B3_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B11_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B3_array[B3_array == 65535] = np.nan
                                    B11_array[B11_array == 65535] = np.nan
                                    output_array = (B3_array - B11_array) / (B3_array + B11_array)
                                    B3_array = None
                                    B11_array = None
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'EVI':
                            ds_list = self._check_output_band_statue(['B2', 'B4', 'B8'], tiffile_serial_num, **kwargs)
                            if ds_list is not None:
                                try:
                                    B2_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B8_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B8_array[B8_array == 65535] = np.nan
                                    B4_array[B4_array == 65535] = np.nan
                                    B2_array[B2_array == 65535] = np.nan
                                    output_array = 2.5 * (B8_array - B4_array) / (B8_array + 6 * B4_array - 7.5 * B2_array + 1)
                                    B4_array = None
                                    B8_array = None
                                    B2_array = None
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'EVI2':
                            ds_list = self._check_output_band_statue(['B8', 'B4'], tiffile_serial_num, **kwargs)
                            # print('process b8 and b4' + str(time.time() - time1))
                            if ds_list is not None:
                                try:
                                    B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B8_array[B8_array == 65535] = np.nan
                                    B4_array[B4_array == 65535] = np.nan
                                    output_array = 2.5 * (B8_array - B4_array) / (B8_array + 2.4 * B4_array + 1)
                                    B4_array = None
                                    B8_array = None
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'GNDVI':
                            ds_list = self._check_output_band_statue(['B8', 'B3'], tiffile_serial_num, **kwargs)
                            # print('process b8 and b4' + str(time.time() - time1))
                            if ds_list is not None:
                                try:
                                    if self._sparsify_matrix_factor:
                                        B8_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B3_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                    else:
                                        B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B3_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B8_array[B8_array == 65535] = np.nan
                                    B3_array[B3_array == 65535] = np.nan
                                    output_array = (B8_array - B3_array) / (B8_array + B3_array)
                                    B3_array = None
                                    B8_array = None
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'NDVI_RE':
                            ds_list = self._check_output_band_statue(['B7', 'B5'], tiffile_serial_num, **kwargs)
                            # print('process b8 and b4' + str(time.time() - time1))
                            if ds_list is not None:
                                try:
                                    if self._sparsify_matrix_factor:
                                        B7_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B5_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                    else:
                                        B7_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B5_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B7_array[B7_array == 65535] = np.nan
                                    B5_array[B5_array == 65535] = np.nan
                                    output_array = (B7_array - B5_array) / (B7_array + B5_array)
                                    B5_array = None
                                    B7_array = None
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'NDVI_RE2':
                            ds_list = self._check_output_band_statue(['B8', 'B5'], tiffile_serial_num, **kwargs)
                            # print('process b8 and b4' + str(time.time() - time1))
                            if ds_list is not None:
                                try:
                                    if self._sparsify_matrix_factor:
                                        B8_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                        B5_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                    else:
                                        B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                        B5_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B8_array[B8_array == 65535] = np.nan
                                    B5_array[B5_array == 65535] = np.nan
                                    output_array = (B8_array - B5_array) / (B8_array + B5_array)
                                    B5_array = None
                                    B8_array = None
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'OSAVI':
                            ds_list = self._check_output_band_statue(['B8', 'B4'], tiffile_serial_num, **kwargs)
                            # print('Process B8 and B4 in' + str(time.time() - time1))
                            if ds_list is not None:
                                try:
                                    time1 = time.time()
                                    B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B8_array[B8_array == 65535] = np.nan
                                    B4_array[B4_array == 65535] = np.nan
                                    output_array = 1.16 * (B8_array - B4_array) / (B8_array + B4_array + 0.16)
                                    B4_array = None
                                    B8_array = None
                                    # print(time.time()-time1)
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        elif index == 'IEI':
                            ds_list = self._check_output_band_statue(['B8', 'B4'], tiffile_serial_num, **kwargs)
                            # print('Process B8 and B4 in' + str(time.time() - time1))
                            if ds_list is not None:
                                try:
                                    time1 = time.time()
                                    B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                    B8_array[B8_array == 65535] = np.nan
                                    B4_array[B4_array == 65535] = np.nan
                                    output_array = 1.5 * (B8_array - B4_array) / (B8_array + B4_array + 0.5)
                                    B4_array = None
                                    B8A_array = None
                                except:
                                    index_construction_indicator = True
                            else:
                                index_construction_indicator = True
                        else:
                            print(f'{index} is not supported!')
                            sys.exit(-1)

                        if self._combine_band_factor:
                            if array_cube is None:
                                array_cube = np.zeros([output_array.shape[0], output_array.shape[1], len(combine_index_array_list)], dtype=np.int16)

                            if array_cube.shape[0] == output_array.shape[0] and array_cube.shape[1] == output_array.shape[1]:
                                array_cube[:, :, combine_index_list.index(index)] = output_array
                            else:
                                print('consistency issuse')
                                return

                    elif self._combine_band_factor and os.path.exists(subset_output_path + file_name + '.TIF'):
                        time3 = time.time()
                        gdal.Warp('/vsimem/' + file_name + '.vrt', subset_output_path + file_name + '.TIF', outputBounds=band_output_limit, xRes=10, yRes=10)
                        output_ds = gdal.Open('/vsimem/' + file_name + '.vrt')
                        output_array = output_ds.GetRasterBand(1).ReadAsArray()

                        if array_cube is None:
                            array_cube = np.zeros([output_array.shape[0], output_array.shape[1], len(combine_index_array_list)], dtype=np.int16)

                        if array_cube.shape[0] == output_array.shape[0] and array_cube.shape[1] == output_array.shape[1]:
                            array_cube[:, :, combine_index_list.index(index)] = output_array
                        else:
                            print('consistency issuse')
                            return
                        print(f'copy array to list consumes {str(time.time() - time3)}')
                        gdal.Unlink('/vsimem/' + file_name + '.TIF')
                        output_ds = None

                    if index_construction_indicator:
                        return [index, tiffile_serial_num, sensing_date, tile_num]

                    if not os.path.exists(subset_output_path + file_name + '.TIF'):
                        # Output the VI
                        # output_array[np.logical_or(output_array > 1, output_array < -1)] = np.nan
                        if self._size_control_factor is True:
                            output_array[np.isnan(output_array)] = -3.2768
                            output_array = output_array * 10000
                            write_raster(ds_list[0], output_array, '/vsimem/', file_name + '.vrt', raster_datatype=gdal.GDT_Int16)
                            data_type = gdal.GDT_Int16
                        else:
                            write_raster(ds_list[0], output_array, '/vsimem/', file_name + '.vrt', raster_datatype=gdal.GDT_Float32)
                            data_type = gdal.GDT_Float32

                        if self._vi_clip_factor:
                            gdal.Warp('/vsimem/' + file_name + '2.vrt', '/vsimem/' + file_name + '.vrt', xRes=10, yRes=10, cutlineDSName=self.ROI, cropToCutline=True, outputType=data_type)
                        else:
                            gdal.Warp('/vsimem/' + file_name + '2.vrt', '/vsimem/' + file_name + '.vrt', xRes=10, yRes=10, outputType=data_type)
                        gdal.Translate(subset_output_path + file_name + '.TIF', '/vsimem/' + file_name + '2.vrt', options=topts)
                        gdal.Unlink('/vsimem/' + file_name + '.vrt')
                        gdal.Unlink('/vsimem/' + file_name + '2.vrt')

                print(f'Finish processing \033[1;31m{index}\033[0m data of \033[3;34m{str(sensing_date)} {str(tile_num)}\033[0m in \033[1;31m{str(time.time() - start_temp)}\033[0m s ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')

                # Generate SA map
                if not os.path.exists(self.output_path + 'ROI_map\\' + self.ROI_name + '_map.npy'):
                    if self._vi_clip_factor:
                        file_list = bf.file_filter(f'{self.output_path}Sentinel2_{self.ROI_name}_index\\NDVI\\', ['.TIF'], and_or_factor='and')
                    else:
                        file_list = bf.file_filter(f'{self.output_path}Sentinel2_constructed_index\\NDVI\\', ['.TIF'], and_or_factor='and')
                    bf.create_folder(self.output_path + 'ROI_map\\')
                    ds_temp = gdal.Open(file_list[0])
                    array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                    array_temp[:, :] = 1
                    write_raster(ds_temp, array_temp, self.cache_folder, 'temp_' + self.ROI_name + '.TIF',
                                 raster_datatype=gdal.GDT_Int16)
                    if retrieve_srs(ds_temp) != self.main_coordinate_system:
                        gdal.Warp('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                  self.cache_folder + 'temp_' + self.ROI_name + '.TIF',
                                  dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI, cropToCutline=True,
                                  xRes=10, yRes=10, dstNodata=-32768)
                    else:
                        gdal.Warp('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                  self.cache_folder + 'temp_' + self.ROI_name + '.TIF', cutlineDSName=self.ROI,
                                  cropToCutline=True, dstNodata=-32768, xRes=10, yRes=10)
                    ds_ROI_array = gdal.Open('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF')
                    ds_sa_array = ds_ROI_array.GetRasterBand(1).ReadAsArray()

                    if (ds_sa_array == -32768).all() == False:
                        np.save(self.output_path + 'ROI_map\\' + self.ROI_name + '_map.npy', ds_sa_array)
                        if retrieve_srs(ds_temp) != self.main_coordinate_system:
                            gdal.Warp(self.output_path + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                      self.cache_folder + 'temp_' + self.ROI_name + '.TIF',
                                      dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI, cropToCutline=True,
                                      xRes=10, yRes=10, dstNodata=-32768)
                        else:
                            gdal.Warp(self.output_path + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                      self.cache_folder + 'temp_' + self.ROI_name + '.TIF', cutlineDSName=self.ROI,
                                      cropToCutline=True, dstNodata=-32768, xRes=10, yRes=10)
                    gdal.Unlink('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF')
                    ds_temp = None
                    ds_ROI_array = None
                    remove_all_file_and_folder(bf.file_filter(self.cache_folder, ['temp', '.TIF'], and_or_factor='and'))

            if self._combine_band_factor:
                time4 = time.time()
                np.savez_compressed(f'{combine_band_folder}{str(sensing_date)}_{str(tile_num)}.npz', array_cube)
                print(f'Write band combination .npz consumes \033[1;31m{str(time.time() - time4)}\033[0m s')
        else:
            print('Caution! the input variable VI_list should be a list and make sure all of them are in Capital Letter')
            sys.exit(-1)
        return [None, None, None, None]

    def check_subset_intergrality(self, indicator, **kwargs):
        if self.ROI_name is None and ('ROI' not in kwargs.keys() and 'ROI_name' not in kwargs.keys()):
            check_path = f'{self.output_path}{str(indicator)}\\'
        elif self.ROI_name is None and ('ROI' in kwargs.keys() or 'ROI_name' in kwargs.keys()):
            self._subset_indicator_process(**kwargs)
            if self.ROI_name is None:
                print()

    def temporal_mosaic(self, indicator, date, **kwargs):
        self._check_metadata_availability()
        self.check_subset_intergrality(indicator)
        pass

    def _process_2sdc_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('inherit_from_logfile', 'ROI', 'ROI_name', 'dc_overwritten_para', 'remove_nan_layer', 'manually_remove_datelist', 'size_control_factor'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process clipped_overwritten_para
        if 'dc_overwritten_para' in kwargs.keys():
            if type(kwargs['dc_overwritten_para']) is bool:
                self._dc_overwritten_para = kwargs['dc_overwritten_para']
            else:
                raise TypeError('Please mention the dc_overwritten_para should be bool type!')
        else:
            self._clipped_overwritten_para = False

        # process inherit from logfile
        if 'inherit_from_logfile' in kwargs.keys():
            if type(kwargs['inherit_from_logfile']) is bool:
                self._inherit_from_logfile = kwargs['inherit_from_logfile']
            else:
                raise TypeError('Please mention the dc_overwritten_para should be bool type!')
        else:
            self._inherit_from_logfile = False

        # process remove_nan_layer
        if 'remove_nan_layer' in kwargs.keys():
            if type(kwargs['remove_nan_layer']) is bool:
                self._remove_nan_layer = kwargs['remove_nan_layer']
            else:
                raise TypeError('Please mention the remove_nan_layer should be bool type!')
        else:
            self._remove_nan_layer = False

        # process remove_nan_layer
        if 'manually_remove_datelist' in kwargs.keys():
            if type(kwargs['manually_remove_datelist']) is list:
                self._manually_remove_datelist = kwargs['manually_remove_datelist']
                self._manually_remove_para = True
            else:
                raise TypeError('Please mention the manually_remove_datelist should be list type!')
        else:
            self._manually_remove_datelist = False
            self._manually_remove_para = False

        # process ROI_NAME
        if 'ROI_name' in kwargs.keys():
            self.ROI_name = kwargs['ROI_name']
        elif self.ROI_name is None and self._inherit_from_logfile:
            self._retrieve_para(['ROI_name'])
        elif self.ROI_name is None:
            raise Exception('Notice the ROI name was missed!')

        # process ROI
        if 'ROI' in kwargs.keys():
            self.ROI = kwargs['ROI']
        elif self.ROI is None and self._inherit_from_logfile:
            self._retrieve_para(['ROI'])
        elif self.ROI is None:
            raise Exception('Notice the ROI was missed!')

        # Retrieve size control factor
        if 'size_control_factor' in kwargs.keys():
            if type(kwargs['size_control_factor']) is bool:
                self._size_control_factor = kwargs['size_control_factor']
            else:
                raise TypeError('Please mention the size_control_factor should be bool type!')
        elif self._inherit_from_logfile:
            self._retrieve_para(['size_control_factor'])
        else:
            self._size_control_factor = False

    @save_log_file
    def seq_ds2sdc(self, index_list, *args, **kwargs):

        # sequenced process
        for index in index_list:
            self._ds2sdc(index, *args, **kwargs)

    @save_log_file
    def mp_ds2sdc(self, index_list, *args, **kwargs):

        if 'chunk_size' in kwargs.keys() and type(kwargs['chunk_size']) is int:
            chunk_size = min(os.cpu_count(), kwargs['chunk_size'])
            del kwargs['chunk_size']
        else:
            chunk_size = os.cpu_count()

        # mp process
        with concurrent.futures.ProcessPoolExecutor(max_workers=chunk_size) as executor:
            executor.map(self._ds2sdc, index_list, repeat(kwargs))

    def _ds2sdc(self, index, *args, **kwargs):

        # for the MP
        if args != () and type(args[0]) == dict:
            kwargs = copy.copy(args[0])

        # process clip parameter
        self._process_2sdc_para(**kwargs)

        # Remove all files which not meet the requirements
        band_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11']
        ROI_midname = 'constructed' if self.ROI_name is None else self.ROI_name
        self.dc_vi[index + 'input_path'] = self.output_path + f'Sentinel2_{ROI_midname}_index\\all_band\\' if index in band_list else self.output_path + f'Sentinel2_{ROI_midname}_index\\{index}\\'

        # path check
        if not os.path.exists(self.dc_vi[index + 'input_path']):
            raise Exception('Please validate the roi name and vi for datacube output!')

        self.dc_vi[index] = self.output_path + f'Sentinel2_{ROI_midname}_datacube\\' + index + '_sequenced_datacube\\'
        bf.create_folder(self.dc_vi[index])

        if len(bf.file_filter(self.dc_vi[index + 'input_path'], [f'{index}.TIF'], and_or_factor='and')) != self.S2_metadata_size:
            raise ValueError(f'{index} of the {self.ROI_name} is not consistent')

        print(f'Start convert the dataset of \033[0;31m{index}\033[0m to sdc.')
        start_time = time.time()

        if self._dc_overwritten_para or not os.path.exists(self.dc_vi[index] + 'doy.npy') or not os.path.exists(self.dc_vi[index] + 'header.npy'):

            if self.ROI_name is None:
                print('Start processing ' + index + ' datacube.')
                header_dic = {'ROI_name': None, 'index': index, 'Datatype': 'float', 'ROI': None, 'ROI_array': None,
                              'sdc_factor': True, 'coordinate_system': self.main_coordinate_system,'size_control_factor': self._size_control_factor,
                              'oritif_folder': self.dc_vi[index + 'input_path'], 'dc_group_list': None, 'tiles': None}
            else:
                print('Start processing ' + index + ' datacube of the ' + self.ROI_name + '.')
                sa_map = np.load(bf.file_filter(self.output_path + 'ROI_map\\', [self.ROI_name, '.npy'], and_or_factor='and')[0], allow_pickle=True)
                header_dic = {'ROI_name': self.ROI_name, 'index': index, 'Datatype': 'float', 'ROI': self.ROI, 'ROI_array': self.output_path + 'ROI_map\\' + self.ROI_name + '_map.npy',
                              'sdc_factor': True, 'coordinate_system': self.main_coordinate_system, 'size_control_factor': self._size_control_factor,
                              'oritif_folder': self.dc_vi[index + 'input_path'], 'dc_group_list': None, 'tiles': None}

            VI_stack_list = bf.file_filter(self.dc_vi[index + 'input_path'], [index, '.TIF'], and_or_factor='and')
            VI_stack_list.sort()
            cols, rows = sa_map.shape[1], sa_map.shape[0]
            doy_list = [int(filepath_temp.split('\\')[-1][0:8]) for filepath_temp in VI_stack_list]
            doy_list = np.unique(np.array(doy_list))
            doy_list = doy_list.tolist()

            # Evaluate the size and sparsity of sdc
            mem = psutil.virtual_memory()
            dc_max_size = int((mem.free) * 0.90)
            nodata_value = -32768 if sa_map.dtype == np.int16 else np.nan
            sparsify = np.sum(sa_map == nodata_value) / (sa_map.shape[0] * sa_map.shape[1])
            _huge_matrix = True if len(doy_list) * cols * rows * 2 > dc_max_size else False
            _sparse_matrix = True if sparsify > 0.9 else False

            if _huge_matrix:
                if _sparse_matrix:
                    data_cube = NDSparseMatrix()
                    nodata_value = -32768 if self._size_control_factor else np.nan
                    data_valid_array = np.zeros([len(doy_list)], dtype=np.int16)
                    i = 0
                    while i < len(doy_list):
                        t1 = time.time()

                        if index in band_list:
                            index_req_list = bf.file_filter(self.dc_vi[index + 'input_path'], [str(doy_list[i]), f'{index}.TIF'], and_or_factor='and')
                            vrt = gdal.BuildVRT(f'/vsimem/{str(doy_list[i])}_{index}.vrt', index_req_list)
                            gdal.Warp(f'/vsimem/{str(doy_list[i])}_{index}.tif', vrt, cutlineDSName=self.ROI, cropToCutline=True, xRes=10, yRes=10, dstNodata=0)
                            vrt = None
                            tiffile_list_temp = [f'/vsimem/{str(doy_list[i])}_{index}.tif']
                        else:
                            tiffile_list_temp = [temp for temp in VI_stack_list if str(doy_list[i]) in temp]

                        if len(tiffile_list_temp) == 1:
                            ds_temp = gdal.Open(tiffile_list_temp[0], gdal.GA_ReadOnly)
                            array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                            if self._size_control_factor and index not in band_list:
                                array_temp = array_temp.astype(np.uint16) + 32768
                            else:
                                array_temp[np.isnan(array_temp)] = 0

                        elif len(tiffile_list_temp) > 1:
                            j, array_list = 0, []
                            while j < len(tiffile_list_temp):
                                ds_temp = gdal.Open(tiffile_list_temp[j], gdal.GA_ReadOnly)
                                array_list.append(ds_temp.GetRasterBand(1).ReadAsArray())
                                j += 1
                            array_temp = np.stack(array_list, axis=2)
                            if self._size_control_factor and index not in band_list:
                                array_temp = np.max(array_temp.astype(np.uint16) + 32768, axis=2)
                            else:
                                array_temp = np.nanmax(array_temp, axis=2)
                                array_temp[np.isnan(array_temp)] = 0

                        else:
                            raise Exception('Code error')

                        if index in band_list:
                            gdal.Unlink(f'/vsimem/{str(doy_list[i])}_{index}.tif')

                        sm_temp = sm.coo_matrix(array_temp)
                        data_cube.append(sm_temp, name=doy_list[i])
                        data_valid_array[i] = 1 if sm_temp.data.shape[0] == 0 else 0
                        print(f'Input the {str(doy_list[i])} into the sdc using {str(time.time()-t1)[0:5]}s (layer {str(i)} of {str(len(doy_list))})')
                        i += 1

                    # remove nan layer
                    if self._manually_remove_para is True and self._manually_remove_datelist is not None:
                        i_temp = 0
                        while i_temp < len(doy_list):
                            if doy_list[i_temp] in self._manually_remove_datelist:
                                data_valid_array[i] = 1
                                self._manually_remove_datelist.remove(doy_list[i_temp])
                            i_temp += 1

                    elif self._manually_remove_para is True and self._manually_remove_datelist is None:
                        raise ValueError('Please correctly input the manual input date list')

                    if self._remove_nan_layer or self._manually_remove_para:
                        i_temp = 0
                        while i_temp < len(doy_list):
                            if data_valid_array[i_temp]:
                                if doy_list[i_temp] in data_cube.SM_namelist:
                                    data_cube.remove_layer(doy_list[i_temp])
                                doy_list.remove(doy_list[i_temp])
                                data_valid_array = np.delete(data_valid_array, i_temp, 0)
                                i_temp -= 1
                            i_temp += 1

                    # Save the sdc
                    np.save(self.dc_vi[index] + f'doy.npy', doy_list)
                    bf.create_folder(f'{self.dc_vi[index]}{str(index)}_sequenced_datacube\\')
                    data_cube.save(f'{self.dc_vi[index]}{str(index)}_sequenced_datacube\\')

                else:

                    if len(doy_list) * cols * rows * 2 > dc_max_size:
                        _huge_matrix = True
                        dc_len = int(np.floor(dc_max_size / (rows * cols * 2)))
                        dc_num = int(np.ceil(len(doy_list) / dc_len))
                        dc_len_list = [int(np.ceil(len(doy_list) / dc_num)) for i_temp in range(dc_num - 1)]
                        dc_len_list.append(len(doy_list) - np.sum(np.array(dc_len_list)))
                    else:
                        dc_num = 1
                        dc_len_list = [len(doy_list)]

                    # Method 1 create multiple array
                    # Define var
                    dtype_temp = np.int16 if self._size_control_factor else np.float16
                    nodata_value = -32768 if self._size_control_factor else np.nan
                    band_nodata_value, dc_initial, dc_len_list_final = np.nan, 0, []
                    sa_map[sa_map == -32768] = 0

                    for dc_num_temp in range(dc_num):
                        doy_list_temp = doy_list[dc_initial: dc_initial + dc_len_list[dc_num_temp]]
                        data_cube_temp = np.zeros((rows, cols, dc_len_list[dc_num_temp]), dtype=np.float16)
                        data_valid_array = np.zeros([dc_len_list[dc_num_temp]], dtype=np.int16)
                        i = 0
                        while i < len(doy_list_temp):
                            tiffile_list_temp = [temp for temp in VI_stack_list if str(doy_list_temp[i]) in temp]
                            if len(tiffile_list_temp) == 1:
                                t1 = time.time()
                                ds_temp = gdal.Open(tiffile_list_temp[0])
                                array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                                data_cube_temp[:, :, i] = np.ma.masked_array(array_temp, mask=[sa_map])
                                t2 = time.time()
                                print(f'input array consumes {str(t2 - t1)}s')
                                data_valid_array[i] = np.all(array_temp.flatten() == nodata_value)
                                print(f'detect valid or not {str(time.time() - t2)}s')
                            elif len(tiffile_list_temp) > 1:
                                t1 = time.time()
                                data_cube_temp_temp = np.zeros((rows, cols, len(tiffile_list_temp)),
                                                               dtype=np.float16)
                                j = 0
                                while j < len(tiffile_list_temp):
                                    ds_temp = gdal.Open(tiffile_list_temp[j])
                                    array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                                    data_cube_temp_temp[:, :, j] = array_temp
                                    j += 1
                                data_cube_temp[:, :, i] = np.nanmax(data_cube_temp_temp, axis=2)
                                data_cube_temp_temp = None
                                t2 = time.time()
                                print(f'input array consumes {str(t2 - t1)}s')
                                data_valid_array[i] = np.all(data_cube_temp[:, :, i].flatten() == nodata_value)
                                print(f'detect valid or not {str(time.time() - t2)}s')
                            else:
                                raise Exception('Code error')
                            i += 1
                        dc_initial += dc_len_list[dc_num_temp]

                    if self._size_control_factor:
                        data_cube_temp[data_cube_temp == -32768] = np.nan
                        data_cube_temp = data_cube_temp / 10000

                    if self._manually_remove_para is True and self._manually_remove_datelist is not None:
                        i_temp = 0
                        while i_temp < len(doy_list_temp):
                            if str(doy_list_temp[i_temp]) in self._manually_remove_datelist:
                                data_valid_array[i] = 1
                                self._manually_remove_datelist.remove(str(doy_list_temp[i_temp]))
                            i_temp += 1

                    elif self._manually_remove_para is True and self._manually_remove_datelist is None:
                        raise ValueError('Please correctly input the manual input date list')

                    if self._remove_nan_layer or self._manually_remove_para:
                        i_temp = 0
                        while i_temp < len(doy_list_temp):
                            if data_valid_array[i_temp]:
                                doy_list_temp = doy_list_temp.remove(doy_list_temp[i_temp])
                                data_cube_temp = np.delete(data_cube_temp, i_temp, 2)
                                data_valid_array = np.delete(data_valid_array, i_temp, 0)
                                i_temp -= 1
                            i_temp += 1

                    dc_len_list_final.append(doy_list_temp)

                    # Save the sdc
                    np.save(self.dc_vi[index] + f'date{str(dc_num_temp)}.npy', doy_list_temp)
                    np.savez_compressed(self.dc_vi[index] + str(index) + f'_sequenced_datacube{str(dc_num_temp)}.npz', data_cube_temp)
                    header_dic['dc_group_list'], header_dic['tiles'] = dc_len_list_final, dc_num

            else:
                nodata_value = -32768 if self._size_control_factor else np.nan
                data_cube_temp = np.zeros((rows, cols, len(doy_list)), dtype=np.float16)
                data_valid_array = np.zeros([len(doy_list)], dtype=np.int16)
                i = 0
                while i < len(doy_list):
                    tiffile_list_temp = [temp for temp in VI_stack_list if str(doy_list[i]) in temp]
                    if len(tiffile_list_temp) == 1:
                        ds_temp = gdal.Open(tiffile_list_temp[0])
                        array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                        data_cube_temp[:, :, i] = array_temp
                        data_valid_array[i] = np.all(array_temp.flatten() == nodata_value)
                    elif len(tiffile_list_temp) > 1:
                        data_cube_temp_temp = np.zeros((rows, cols, len(tiffile_list_temp)), dtype=np.float16)
                        j = 0
                        while j < len(tiffile_list_temp):
                            ds_temp = gdal.Open(tiffile_list_temp[j])
                            array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                            data_cube_temp_temp[:, :, j] = array_temp
                            j += 1
                        data_cube_temp[:, :, i] = np.nanmax(data_cube_temp_temp, axis=2)
                        data_cube_temp_temp = None
                        data_valid_array[i] = np.all(data_cube_temp[:, :, i].flatten() == nodata_value)
                    else:
                        raise Exception('Code error')
                    i += 1

                if self._size_control_factor:
                    data_cube_temp[data_cube_temp == -32768] = np.nan
                    data_cube_temp = data_cube_temp / 10000

                if self._manually_remove_para is True and self._manually_remove_datelist is not None:
                    i_temp = 0
                    while i_temp < len(doy_list):
                        if str(doy_list[i_temp]) in self._manually_remove_datelist:
                            data_valid_array[i] = 1
                            self._manually_remove_datelist.remove(str(doy_list[i_temp]))
                        i_temp += 1

                elif self._manually_remove_para is True and self._manually_remove_datelist is None:
                    raise ValueError('Please correctly input the manual input date list')

                if self._remove_nan_layer or self._manually_remove_para:
                    i_temp = 0
                    while i_temp < len(doy_list):
                        if data_valid_array[i_temp]:
                            doy_list = doy_list.remove(doy_list[i_temp])
                            data_cube_temp = np.delete(data_cube_temp, i_temp, 2)
                            data_valid_array = np.delete(data_valid_array, i_temp, 0)
                            i_temp -= 1
                        i_temp += 1

                # Write the datacube
                np.save(self.dc_vi[index] + f'date.npy', doy_list)
                np.save(self.dc_vi[index] + str(index) + f'_sequenced_datacube.npz', data_cube_temp)
                end_time = time.time()

            header_dic['ds_file'], header_dic['sparse_matrix'], header_dic['huge_matrix'] = self.output_path + 'ROI_map\\' + self.ROI_name + '_map.TIF', _sparse_matrix, _huge_matrix
            np.save(self.dc_vi[index] + 'header.npy', header_dic)

        print(f'Finished writing the sdc in \033[1;31m{str(time.time() - start_time)} s\033[0m.')


class Sentinel2_dc(object):
    def __init__(self, dc_filepath, work_env=None):
        # define var
        if os.path.exists(dc_filepath) and os.path.isdir(dc_filepath):
            self.dc_filepath = dc_filepath
        else:
            raise ValueError('Please input a valid dc filepath')
        # eliminating_all_not_required_file(self.dc_filepath, filename_extension=['npy'])

        # Read header
        header_file = bf.file_filter(self.dc_filepath, ['header.npy'])
        if len(header_file) == 0:
            raise ValueError('There has no valid sdc or the header file of the sdc was missing!')
        elif len(header_file) > 1:
            raise ValueError('There has more than one header file in the dir')
        else:
            try:
                self.dc_header = np.load(header_file[0], allow_pickle=True).item()
                if type(self.dc_header) is not dict:
                    raise Exception('Please make sure the header file is a dictionary constructed in python!')

                for dic_name in ['ROI_name', 'index', 'Datatype', 'ROI', 'ROI_array', 'sdc_factor', 'coordinate_system',
                                 'oritif_folder', 'ds_file', 'sparse_matrix', 'huge_matrix', 'size_control_factor', 'dc_group_list', 'tiles']:
                    if dic_name not in self.dc_header.keys():
                        raise Exception(f'The {dic_name} is not in the dc header, double check!')
                    else:
                        self.__dict__[dic_name] = self.dc_header[dic_name]
            except:
                raise Exception('Something went wrong when reading the header!')

        start_time = time.time()
        print(f'Start loading the sdc of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')
        # Read doy or date file of the Datacube
        try:
            if self.sdc_factor is True:
                # Read doylist
                doy_file = bf.file_filter(self.dc_filepath, ['doy.npy'])
                if len(doy_file) == 0:
                    raise ValueError('There has no valid doy file or file was missing!')
                elif len(doy_file) > 1:
                    raise ValueError('There has more than one doy file in the dc dir')
                else:
                    sdc_doylist = np.load(doy_file[0], allow_pickle=True)
                    self.sdc_doylist = [int(sdc_doy) for sdc_doy in sdc_doylist]
            else:
                raise TypeError('Please input as a sdc')
        except:
            raise Exception('Something went wrong when reading the doy and date list!')

        # Read datacube
        try:
            if self.sparse_matrix and self.huge_matrix:
                if os.path.exists(self.dc_filepath + f'{self.index}_sequenced_datacube\\'):
                    self.dc = NDSparseMatrix().load(self.dc_filepath + f'{self.index}_sequenced_datacube\\')
                else:
                    raise Exception('Please double check the code if the sparse huge matrix is generated properply')
            elif not self.huge_matrix:
                self.dc_filename = bf.file_filter(self.dc_filepath, ['sequenced_datacube.npy'])
                if len(self.dc_filename) == 0:
                    raise ValueError('There has no valid dc or the dc was missing!')
                elif len(self.dc_filename) > 1:
                    raise ValueError('There has more than one date file in the dc dir')
                else:
                    self.dc = np.load(self.dc_filename[0], allow_pickle=True)
            elif self.huge_matrix and not self.sparse_matrix:
                self.dc_filename = bf.file_filter(self.dc_filepath, ['sequenced_datacube', '.npy'], and_or_factor='and')
        except:
            raise Exception('Something went wrong when reading the datacube!')

        self.dc_XSize, self.dc_YSize, self.dc_ZSize = self.dc.shape[0], self.dc.shape[1], self.dc.shape[2]
        print(f'Finish loading the sdc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time()-start_time)}\033[0ms' )

        # Check work env
        if work_env is not None:
            self.work_env = Path(work_env).path_name
        else:
            self.work_env = Path(os.path.dirname(os.path.dirname(self.dc_filepath))).path_name
        self.root_path = Path(os.path.dirname(os.path.dirname(self.work_env))).path_name

        # Inundation parameter process
        self._DSWE_threshold = None
        self._flood_month_list = None
        self.flood_mapping_method = []
        
    def save(self, output_path):

        start_time = time.time()
        print(f'Start saving the sdc of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        if not os.path.exists(output_path):
            bf.create_folder(output_path)
        output_path = bf.Path(output_path).path_name

        header_dic = {'ROI_name': self.ROI_name, 'index': self.index, 'Datatype': self.Datatype, 'ROI': self.ROI, 'ROI_array': self.ROI_array,
                      'sdc_factor': self.sdc_factor, 'coordinate_system': self.coordinate_system, 'ds_file': self.ds_file,
                      'size_control_factor': self.size_control_factor, 'sparse_matrix': self.sparse_matrix, 'huge_matrix': self.huge_matrix,
                      'oritif_folder': self.oritif_folder, 'dc_group_list': self.dc_group_list, 'tiles': self.tiles}

        doy = self.sdc_doylist
        np.save(f'{output_path}doy.npy', doy)
        np.save(f'{output_path}header.npy', header_dic)

        if self.sparse_matrix:
            self.dc.save(f'{output_path}{str(self.index)}_sequenced_datacube\\')
        else:
            np.save(f'{output_path}{str(self.index)}_sequenced_datacube.npy', self.dc)

        print(f'Finish saving the sdc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')


class Sentinel2_dcs(object):
    def __init__(self, *args, work_env=None, auto_harmonised=True):

        # Generate the datacubes list
        self._dcs_backup_ = []
        for args_temp in args:
            if type(args_temp) != Sentinel2_dc:
                raise TypeError('The Landsat datacubes was a bunch of Landsat datacube!')
            else:
                self._dcs_backup_.append(args_temp)

        # Validation and consistency check
        if len(self._dcs_backup_) == 0:
            raise ValueError('Please input at least one valid Landsat datacube')

        if type(auto_harmonised) != bool:
            raise TypeError('Please input the auto harmonised factor as bool type!')
        else:
            harmonised_factor = False

        self.index_list, ROI_list, ROI_name_list, Datatype_list, ds_list, ROI_array_list, sdc_factor_list, doy_list, coordinate_system_list, oritif_folder_list = [], [], [], [], [], [], [], [], [], []
        self.dcs, huge_matrix_list, sparse_matrix_list, self.size_control_factor_list = [], [], [], []
        x_size, y_size, z_size = 0, 0, 0
        for dc_temp in self._dcs_backup_:
            if x_size == 0 and y_size == 0 and z_size == 0:
                x_size, y_size, z_size = dc_temp.dc_XSize, dc_temp.dc_YSize, dc_temp.dc_ZSize
            elif x_size != dc_temp.dc_XSize or y_size != dc_temp.dc_YSize:
                raise Exception('Please make sure all the datacube share the same size!')
            elif z_size != dc_temp.dc_ZSize:
                if auto_harmonised:
                    harmonised_factor = True
                else:
                    raise Exception('The datacubes is not consistent in the date dimension! Turn auto harmonised fator as True if wanna avoid this problem!')

            self.index_list.append(dc_temp.index)
            ROI_name_list.append(dc_temp.ROI_name)
            sdc_factor_list.append(dc_temp.sdc_factor)
            ROI_list.append(dc_temp.ROI)
            ds_list.append(dc_temp.ds_file)
            ROI_array_list.append(dc_temp.ROI_array)
            Datatype_list.append(dc_temp.Datatype)
            coordinate_system_list.append(dc_temp.coordinate_system)
            oritif_folder_list.append(dc_temp.oritif_folder)
            huge_matrix_list.append(dc_temp.huge_matrix)
            sparse_matrix_list.append(dc_temp.sparse_matrix)
            self.size_control_factor_list.append(dc_temp.size_control_factor)
            # self.dcs.append(dc_temp.dc)

        if x_size != 0 and y_size != 0 and z_size != 0:
            self.dcs_XSize, self.dcs_YSize, self.dcs_ZSize = x_size, y_size, z_size
        else:
            raise Exception('Please make sure all the datacubes was not void')

        # Check the consistency of the roi list
        if len(ROI_list) == 0 or False in [len(ROI_list) == len(self.index_list),
                                           len(self.index_list) == len(sdc_factor_list),
                                           len(ROI_name_list) == len(sdc_factor_list),
                                           len(ROI_name_list) == len(ds_list), len(ds_list) == len(ROI_array_list),
                                           len(ROI_array_list) == len(Datatype_list),
                                           len(coordinate_system_list) == len(Datatype_list),
                                           len(oritif_folder_list) == len(coordinate_system_list)]:
            raise Exception('The ROI list or the index list for the datacubes were not properly generated!')
        elif False in [roi_temp == ROI_list[0] for roi_temp in ROI_list]:
            raise Exception('Please make sure all datacubes were in the same roi!')
        elif False in [sdc_temp == sdc_factor_list[0] for sdc_temp in sdc_factor_list]:
            raise Exception('Please make sure all dcs were consistent!')
        elif False in [roi_name_temp == ROI_name_list[0] for roi_name_temp in ROI_name_list]:
            raise Exception('Please make sure all dcs were consistent!')
        elif False in [(ROI_array == ROI_array_list[0]) for ROI_array in ROI_array_list]:
            raise Exception('Please make sure all dcs were consistent!')
        elif False in [dt_temp == Datatype_list[0] for dt_temp in Datatype_list]:
            raise Exception('Please make sure all dcs were consistent!')
        elif False in [coordinate_system_temp == coordinate_system_list[0] for coordinate_system_temp in coordinate_system_list]:
            raise Exception('Please make sure all coordinate system were consistent!')
        elif False in [sparse_matrix_temp == sparse_matrix_list[0] for sparse_matrix_temp in sparse_matrix_list]:
            raise Exception('Please make sure all sparse_matrix were consistent!')
        elif False in [huge_matrix_temp == huge_matrix_list[0] for huge_matrix_temp in huge_matrix_list]:
            raise Exception('Please make sure all huge_matrix were consistent!')

        # Define the field
        self.ROI = ROI_list[0]
        self.ROI_name = ROI_name_list[0]
        self.sdc_factor = sdc_factor_list[0]
        self.Datatype = Datatype_list[0]
        self.ROI_array = ROI_array_list[0]
        self.ds_file = ds_list[0]
        self.coordinate_system = coordinate_system_list[0]
        self.oritif_folder = oritif_folder_list
        self.sparse_matrix = sparse_matrix_list[0]
        self.huge_matrix = huge_matrix_list[0]

        # Read the doy or date list
        if self.sdc_factor is False:
            raise Exception('Please sequenced the datacubes before further process!')
        else:
            doy_list = [temp.sdc_doylist for temp in self._dcs_backup_]
            if False in [len(temp) == len(doy_list[0]) for temp in doy_list] or False in [(temp == doy_list[0]) for temp in doy_list]:
                if auto_harmonised:
                    harmonised_factor = True
                else:
                    raise Exception('The datacubes is not consistent in the date dimension! Turn auto harmonised factor as True if wanna avoid this problem!')
            else:
                self.doy_list = self._dcs_backup_[0].sdc_doylist

        # Harmonised the dcs
        if harmonised_factor:
            self._auto_harmonised_dcs()

        #  Define the output_path
        if work_env is None:
            self.work_env = Path(os.path.dirname(os.path.dirname(self._dcs_backup_[0].dc_filepath))).path_name
        else:
            self.work_env = work_env

        # Define var for the flood mapping
        self.inun_det_method_dic = {}
        self._variance_num = 2
        self._inundation_overwritten_factor = False
        self._DEM_path = None
        self._DT_std_fig_construction = False
        self._construct_inundated_dc = True
        self._flood_mapping_accuracy_evaluation_factor = False
        self._sample_rs_link_list = None
        self._sample_data_path = None
        self._flood_mapping_method = ['Unet', 'MNDWI_thr', 'DT']

        # Define var for the phenological analysis
        self._curve_fitting_algorithm = None
        self._flood_removal_method = None
        self._curve_fitting_dic = {}

        # Define var for NIPY reconstruction
        self._add_NIPY_dc = True
        self._NIPY_overwritten_factor = False

        # Define var for phenology metrics generation
        self._phenology_index_all = ['annual_ave_VI', 'flood_ave_VI', 'unflood_ave_VI', 'max_VI', 'max_VI_doy',
                                     'bloom_season_ave_VI', 'well_bloom_season_ave_VI']
        self._curve_fitting_dic = {}
        self._all_quantify_str = None

        # Define var for flood_free_phenology_metrics
        self._flood_free_pm = ['annual_max_VI', 'average_VI_between_max_and_flood']
    
    def _auto_harmonised_dcs(self):

        doy_all = np.array([])
        for dc_temp in self._dcs_backup_:
            doy_all = np.concatenate([doy_all, dc_temp.sdc_doylist], axis=0)
        doy_all = np.sort(np.unique(doy_all))

        i = 0
        while i < len(self._dcs_backup_):
            m_factor = False
            for doy in doy_all:
                if doy not in self._dcs_backup_[i].sdc_doylist:
                    m_factor = True
                    if not self.sparse_matrix:
                        self._dcs_backup_[i].dc = np.insert(self._dcs_backup_[i].dc, np.argwhere(doy_all == doy).flatten()[0], np.nan * np.zeros([self._dcs_backup_[i].dc_XSize, self._dcs_backup_[i].dc_YSize, 1]), axis=2)
                    else:
                        self._dcs_backup_[i].dc.append(sm.coo_matrix(np.zeros([self.dcs_XSize, self.dcs_YSize])), name=doy, pos=np.argwhere(doy_all == doy).flatten()[0])
            if m_factor:
                self._dcs_backup_[i].sdc_doylist = copy.copy(doy_all)
                self._dcs_backup_[i].dc_ZSize = self._dcs_backup_[i].dc.shape[2]
            i += 1

        z_size, doy_list = 0, []
        for dc_temp in self._dcs_backup_:
            if z_size == 0:
                z_size = dc_temp.dc_ZSize
            elif z_size != dc_temp.dc_ZSize:
                raise Exception('Auto harmonised failure!')
            doy_list.append(dc_temp.sdc_doylist)

        if False in [temp.shape[0] == doy_list[0].shape[0] for temp in doy_list] or False in [(temp == doy_list[0]).all() for temp in doy_list]:
            raise Exception('Auto harmonised failure!')

        self.dcs_ZSize = z_size
        self.doy_list = doy_list[0]

    def append(self, dc_temp: Sentinel2_dc) -> None:
        if type(dc_temp) is not Sentinel2_dc:
            raise TypeError('The appended data should be a Sentinel2_dc!')

        for indicator in ['ROI', 'ROI_name', 'sdc_factor', 'coordinate_system', 'sparse_matrix', 'huge_matrix']:
            if dc_temp.__dict__[indicator] != self.__dict__[indicator]:
                raise ValueError('The appended datacube is not consistent with the original datacubes')

        if self.dcs_XSize != dc_temp.dc_XSize or self.dcs_YSize != dc_temp.dc_YSize or self.dcs_ZSize != dc_temp.dc_ZSize:
            raise ValueError('The appended datacube has different size compared to the original datacubes')

        if self.doy_list != dc_temp.sdc_doylist:
            raise ValueError('The appended datacube has doy list compared to the original datacubes')

        self.index_list.append(dc_temp.index)
        self.oritif_folder.append(dc_temp.oritif_folder)
        self._dcs_backup_.append(dc_temp)

    def extend(self, dcs_temp) -> None:
        if type(dcs_temp) is not Sentinel2_dcs:
            raise TypeError('The appended data should be a Sentinel2_dcs!')

        for indicator in ['ROI', 'ROI_name', 'sdc_factor', 'dcs_XSize', 'dcs_YSize', 'dcs_ZSize', 'doy_list', 'coordinate_system']:
            if dcs_temp.__dict__[indicator] != self.__dict__[indicator]:
                raise ValueError('The appended datacube is not consistent with the original datacubes')

        self.index_list.extend(dcs_temp.index_list)
        self.oritif_folder.extend(dcs_temp.oritif_folder)
        self._dcs_backup_.extend(dcs_temp)

    def _inun_det_para(self, **kwargs):
        self._construct_inundated_dc = True
        self._inundated_ow_para = False

        for key_temp in kwargs.keys():
            if key_temp not in ['construct_inundated_dc', 'overwritten_para']:
                raise ValueError(f'The {str(key_temp)} for func inundation detection is not supported!')

            if key_temp == 'construct_inundated_dc' and type(kwargs['construct_inundated_dc']) is bool:
                self._construct_inundated_dc = kwargs['construct_inundated_dc']
            elif key_temp == 'overwritten_para' and type(kwargs['overwritten_para']) is bool:
                self._inundated_ow_para = kwargs['overwritten_para']

    def _inundation_detection(self, method: str, **kwargs):
        # process inundation detection method
        self._inun_det_para(**kwargs)

        start_time = time.time()
        print(f'Start detecting the inundation area in the \033[1;34m{self.ROI_name}\033[0m')

        # Method 1 MNDWI static threshold
        if method not in self._flood_mapping_method:
            raise ValueError(f'The inundation detection method {str(method)} is not supported')

        if method == 'MNDWI_thr':
            if self.size_control_factor_list[self.index_list.index('MNDWI')]:
                MNDWI_static_thr = 33768
            else:
                MNDWI_static_thr = 0.1

            if 'inundation_MNDWI_thr' not in self.index_list or self._inundated_ow_para:
                if not os.path.exists(self.work_env + 'inundation_MNDWI_thr_sequenced_datacube\\header.npy'):
                    bf.create_folder(self.work_env + 'inundation_MNDWI_thr_sequenced_datacube\\')
                    if 'MNDWI' in self.index_list:
                        if self.sparse_matrix:
                            namelist = self._dcs_backup_[self.index_list.index('MNDWI')].dc.SM_namelist
                            inundation_sm = NDSparseMatrix()
                            for z_temp in range(self.dcs_ZSize):
                                inundation_array = self._dcs_backup_[self.index_list.index('MNDWI')].dc.SM_group[namelist[z_temp]].toarray()
                                inundation_array = np.logical_and(inundation_array <= MNDWI_static_thr, inundation_array != 0)
                                inundation_array = sm.coo_matrix(inundation_array)
                                inundation_sm.append(inundation_array, name=namelist[z_temp])
    
                            if self._construct_inundated_dc:
                                inundation_dc = copy.copy(self._dcs_backup_[self.index_list.index('MNDWI')])
                                inundation_dc.dc = inundation_sm
                                inundation_dc.index = 'inundation_' + method
                                self.append(inundation_dc)
    
                        else:
                            inundation_array = copy.copy(self._dcs_backup_[self.index_list.index('MNDWI')].dc)
                            inundation_array = inundation_array >= MNDWI_static_thr
    
                            if self._construct_inundated_dc:
                                inundation_dc = copy.copy(self._dcs_backup_[self.index_list.index('MNDWI')])
                                inundation_dc.dc = inundation_array
                                self.append(inundation_dc)
                        inundation_dc.save(self.work_env + 'inundation_MNDWI_thr_sequenced_datacube\\')
                    else:
                        raise ValueError('Please construct a valid datacube with MNDWI sdc inside!')
                else:
                    inundation_dc = Sentinel2_dc(self.work_env + 'inundation_MNDWI_thr_sequenced_datacube\\')
                    self.append(inundation_dc)
                
        print(f'Finish detecting the inundation area in the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time()-start_time)}\033[0m s!')

    def _process_inundation_removal_para(self, **kwargs):
        pass

    def _inundation_removal(self, processed_index, inundation_method, construct_new_dc=True, **kwargs):

        # Identify the inundation pixel
        self._inundation_detection(inundation_method, construct_inundated_dc=True, **kwargs)

        inundation_index = 'inundation_' + inundation_method
        self._process_inundation_removal_para(**kwargs)

        if processed_index not in self.index_list or inundation_index not in self.index_list:
            raise ValueError('The inudnation removal or vegetaion index is not properly generated!')

        inundation_dc = self._dcs_backup_[self.index_list.index(inundation_index)]
        processed_dc = copy.copy(self._dcs_backup_[self.index_list.index(processed_index)])
        
        if not os.path.exists(self.work_env + processed_dc.index + '_noninun_sequenced_datacube\\header.npy'):
            bf.create_folder(self.work_env + processed_dc.index + '_noninun_sequenced_datacube\\')
            if self.sparse_matrix:
                for height in range(self.dcs_ZSize):
                    processed_dc.dc.SM_group[processed_dc.dc.SM_namelist[height]] = processed_dc.dc.SM_group[processed_dc.dc.SM_namelist[height]].multiply(inundation_dc.dc.SM_group[inundation_dc.dc.SM_namelist[height]])

                processed_dc.index = processed_dc.index + '_noninun'
                processed_dc.save(self.work_env + processed_dc.index + '_sequenced_datacube\\')

                if not construct_new_dc:
                    self._dcs_backup_[self.index_list.index(processed_index)] = processed_dc
                    self.size_control_factor_list[self.index_list.index(processed_index)] = processed_dc.size_control_factor
                    self.index_list[self.index_list.index(processed_index)] = processed_dc.index
                else:
                    self.append(processed_dc)
            else:
               pass
        else:
            processed_dc = Sentinel2_dc(self.work_env + processed_dc.index + '_noninun_sequenced_datacube\\')
            self.append(processed_dc)

    def generate_phenology_metric(self):
        pass

    def link_GEDI_S2_phenology_inform(self):
        pass

    def _process_link_GEDI_S2_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('retrieval_method'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process clipped_overwritten_para
        if 'retrieval_method' in kwargs.keys():
            if type(kwargs['retrieval_method']) is str and kwargs['retrieval_method'] in ['nearest_neighbor', 'linear_interpolation']:
                self._GEDI_link_S2_retrieval_method = kwargs['retrieval_method']
            else:
                raise TypeError('Please mention the dc_overwritten_para should be str type!')
        else:
            self._GEDI_link_S2_retrieval_method = 'nearest_neighbor'

    def link_GEDI_S2_inform(self, GEDI_xlsx_file, index_list, **kwargs):

        # Two different method0 Nearest data and linear interpolation
        self._process_link_GEDI_S2_para(**kwargs)

        # Retrieve GEDI inform
        self.GEDI_list_temp = gedi.GEDI_list(GEDI_xlsx_file)

        # Retrieve the S2 inform
        raster_gt = gdal.Open(self.ds_file).GetGeoTransform()

        dc_dic = {}
        for index_temp in index_list:

            if index_temp not in self.index_list:
                raise Exception(f'The {str(index_temp)} is not a valid index or is not inputted into the dcs!')

            if self._GEDI_link_S2_retrieval_method == 'nearest_neighbor':
                self.GEDI_list_temp.GEDI_df.insert(loc=len(self.GEDI_list_temp.GEDI_df.columns), column=f'S2_nearest_{index_temp}_value', value=np.nan)
                self.GEDI_list_temp.GEDI_df.insert(loc=len(self.GEDI_list_temp.GEDI_df.columns), column=f'S2_nearest_{index_temp}_date', value=np.nan)
            elif self._GEDI_link_S2_retrieval_method == 'linear_interpolation':
                self.GEDI_list_temp.GEDI_df.insert(loc=len(self.GEDI_list_temp.GEDI_df.columns), column=f'S2_{index_temp}_linear_interpolation', value=np.nan)

            dc_dic[index_temp] = self._dcs_backup_[self.index_list.index(index_temp)].dc

        data_num = [num_temp for num_temp in range(self.GEDI_list_temp.df_size)]

        ### Since the dc_dic was too big for most conditions, it will be defined as a global var,
        ### py3.8 or higher version is required by announcing the dc_dc as a global variable before the mp process
        with concurrent.futures.ProcessPoolExecutor(initializer=init_globe, initargs=(dc_dic,), max_workers=5) as executor:
            result = executor.map(link_GEDI_inform, data_num, list(self.GEDI_list_temp.GEDI_df['Latitude']),
                          list(self.GEDI_list_temp.GEDI_df['Longitude']), list(self.GEDI_list_temp.GEDI_df['Date']),
                          repeat(bf.date2doy(self.doy_list)), repeat(raster_gt), repeat(index_list), repeat(self.GEDI_list_temp.df_size), repeat(self._GEDI_link_S2_retrieval_method),
                          repeat(self.sparse_matrix))

        result = list(result)
        for result_temp in result:
            for index_temp in index_list:
                try:
                    self.GEDI_list_temp.GEDI_df[f'S2_{index_temp}_{self._GEDI_link_S2_retrieval_method}'][result_temp['id']] = result_temp[index_temp]
                except:
                    self.GEDI_list_temp.GEDI_df[f'S2_{index_temp}_{self._GEDI_link_S2_retrieval_method}'][result_temp['id']] = np.nan

        self.GEDI_list_temp.save(GEDI_xlsx_file.split('.')[0] + '_append.csv')


def init_globe(var):
    global dc4link_GEDI
    dc4link_GEDI = var


def link_GEDI_inform(*args):
    [i, lat, lon, date_temp, doy_list, raster_gt, index_list, df_size, GEDI_link_S2_retrieval_method, sparse_matrix] = [args[i] for i in range(9)]

    # Get the basic inform of the i GEDI point
    year_temp = int(date_temp) // 1000

    # Reprojection
    point_temp = gp.points_from_xy([lon], [lat], crs='epsg:4326')
    point_temp = point_temp.to_crs(crs='epsg:32649')
    point_coords = [point_temp[0].coords[0][0], point_temp[0].coords[0][1]]

    # Draw a circle around the central point
    polygon = create_circle_polygon(point_coords, 25)
    index_dic = {'id': i}

    for index_temp in index_list:

        t1 = time.time()
        print(f'Start linking the {index_temp} value with the GEDI dataframe!({str(i)} of {str(df_size)})')

        if GEDI_link_S2_retrieval_method == 'nearest_neighbor':

            # Link GEDI and S2 inform using nearest_neighbor
            pass

        elif GEDI_link_S2_retrieval_method == 'linear_interpolation':

            # Link GEDI and S2 inform using linear_interpolation
            data_postive, date_postive, data_negative, date_negative = None, None, None, None
            index_dic[index_temp] = np.nan

            for date_interval in range(0, 30):

                if date_interval == 0 and date_interval + date_temp in doy_list:
                    array_temp = dc4link_GEDI[index_temp].SM_group[str(bf.doy2date(date_temp))]
                    if sparse_matrix:
                        info_temp = extract_value2shpfile(array_temp, raster_gt, polygon, 32649, nodatavalue=0)
                        info_temp = (float(info_temp) - 32768) / 10000

                    if ~np.isnan(info_temp):
                        index_dic[index_temp] = info_temp
                        break

                else:
                    if data_negative is None and date_temp - date_interval in doy_list:
                        date_temp_temp = date_temp - date_interval
                        array_temp = dc4link_GEDI[index_temp].SM_group[str(bf.doy2date(date_temp_temp))]
                        if sparse_matrix:
                            info_temp = extract_value2shpfile(array_temp, raster_gt, polygon, 32649, nodatavalue=0)
                            info_temp = (float(info_temp) - 32768) / 10000

                        if ~np.isnan(info_temp):
                            data_negative = info_temp
                            date_negative = date_temp_temp

                    if data_negative is None and date_temp + date_interval in doy_list:
                        date_temp_temp = date_temp + date_interval
                        array_temp = dc4link_GEDI[index_temp].SM_group[str(bf.doy2date(date_temp_temp))]
                        if sparse_matrix:
                            info_temp = extract_value2shpfile(array_temp, raster_gt, polygon, 32649, nodatavalue=0)
                            info_temp = (float(info_temp) - 32768) / 10000

                        if ~np.isnan(info_temp):
                            data_postive = info_temp
                            date_postive = date_temp_temp

                    if data_postive is not None and data_negative is not None:
                        index_dic[index_temp] = data_negative + (date_temp - date_negative) * (data_postive - data_negative) / (date_postive - date_negative)
                        break

        print(f'Finish linking the {index_temp} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
    return index_dic


def link_GEDI_inform4mp(*args):
    i, lat, lon, dc4link_GEDI, date_temp, doy_list = args[0], args[1], args[2], args[3], args[4], args[5]
    raster_gt, index_list, df_size, GEDI_link_S2_retrieval_method, sparse_matrix = args[6], args[7], args[8], args[9], args[10]

    # Get the basic inform of the i GEDI point
    year_temp = int(date_temp) // 1000

    # Reprojection
    point_temp = gp.points_from_xy([lon], [lat], crs='epsg:4326')
    point_temp = point_temp.to_crs(crs='epsg:32649')
    point_coords = [point_temp[0].coords[0][0], point_temp[0].coords[0][1]]

    # Draw a circle around the central point
    polygon = create_circle_polygon(point_coords, 25)
    index_dic = {'id': i}

    for index_temp in index_list:

        t1 = time.time()
        print(f'Start linking the {index_temp} value with the GEDI dataframe!({str(i)} of {str(df_size)})')

        if GEDI_link_S2_retrieval_method == 'nearest_neighbor':

            # Link GEDI and S2 inform using nearest_neighbor
            pass

        elif GEDI_link_S2_retrieval_method == 'linear_interpolation':

            # Link GEDI and S2 inform using linear_interpolation
            data_postive, date_postive, data_negative, date_negative = None, None, None, None
            index_dic[index_temp] = np.nan

            for date_interval in range(0, 60):

                if date_interval == 0 and date_interval + date_temp in doy_list:
                    array_temp = dc4link_GEDI[index_temp].SM_group[str(bf.doy2date(date_temp))]
                    if sparse_matrix:
                        info_temp = extract_value2shpfile(array_temp, raster_gt, polygon, 32649, nodatavalue=0)
                        info_temp = (float(info_temp) - 32768) / 10000

                    if ~np.isnan(info_temp):
                        index_dic[index_temp] = info_temp
                        break

                else:
                    if data_negative is None and date_temp - date_interval in doy_list:
                        date_temp_temp = date_temp - date_interval
                        array_temp = dc4link_GEDI[index_temp].SM_group[str(bf.doy2date(date_temp_temp))]
                        if sparse_matrix:
                            info_temp = extract_value2shpfile(array_temp, raster_gt, polygon, 32649, nodatavalue=0)
                            info_temp = (float(info_temp) - 32768) / 10000

                        if ~np.isnan(info_temp):
                            data_negative = info_temp
                            date_negative = date_temp_temp

                    if data_negative is None and date_temp + date_interval in doy_list:
                        date_temp_temp = date_temp + date_interval
                        array_temp = dc4link_GEDI[index_temp].SM_group[str(bf.doy2date(date_temp_temp))]
                        if sparse_matrix:
                            info_temp = extract_value2shpfile(array_temp, raster_gt, polygon, 32649, nodatavalue=0)
                            info_temp = (float(info_temp) - 32768) / 10000

                        if ~np.isnan(info_temp):
                            data_postive = info_temp
                            date_postive = date_temp_temp

                    if data_postive is not None and data_negative is not None:
                        index_dic[index_temp] = data_negative + (date_temp - date_negative) * (data_postive - data_negative) / (date_postive - date_negative)
                        break

        print(f'Finish linking the {index_temp} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')


if __name__ == '__main__':

    #### Download Sentinel-2 data with IDM
    # IDM = "C:\\Program Files (x86)\\Internet Download Manager\\IDMan.exe"
    # DownPath = 'g:\\sentinel2_download\\'
    # shpfile_path = 'E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain\\Floodplain_2020_simplified4.shp'
    #
    # S2_MID_YZR = Queried_Sentinel_ds('shixi2ng', 'shixi2nG', DownPath, IDM_path=IDM)
    # S2_MID_YZR.queried_with_ROI(shpfile_path, ('20190101', '20191231'),'Sentinel-2', 'S2MSI2A',(0, 95), overwritten_factor=True)
    # S2_MID_YZR.download_with_IDM()

    # Test
    # filepath = 'G:\A_veg\S2_test\\Orifile\\'
    # s2_ds_temp = Sentinel2_ds(filepath)
    # s2_ds_temp.construct_metadata()
    # s2_ds_temp.sequenced_subset(['all_band', 'MNDWI'], ROI='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp',
    #                             ROI_name='MYZR_FP_2020', cloud_removal_strategy='QI_all_cloud',
    #                             size_control_factor=True, combine_band_factor=False, pansharp_factor=False)
    # s2_ds_temp.ds2sdc([ 'MNDWI'], inherit_from_logfile=True, remove_nan_layer=True, size_control_factor=True)

    ######  Main procedure for GEDI Sentinel-2 link
    ######  Subset and 2sdc
    # filepath = 'G:\A_veg\S2_all\\Original_file\\'
    # s2_ds_temp = Sentinel2_ds(filepath)
    # s2_ds_temp.construct_metadata()
    #
    # s2_ds_temp.mp_subset(['NDVI_20m', 'OSAVI_20m', 'MNDWI', 'AWEI', 'AWEInsh'], ROI='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp',
    #                      ROI_name='MYZR_FP_2020', cloud_removal_strategy='QI_all_cloud', size_control_factor=True, combine_band_factor=False)
    # s2_ds_temp.mp_ds2sdc(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9'], inherit_from_logfile=True, remove_nan_layer=True, chunk_size=9)

    ##### Constuct dcs
    dc_temp_dic = {}
    dc_temp_dic['MNDWI'] = Sentinel2_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\MNDWI_sequenced_datacube\\')
    for index in ['B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'OSAVI_20m']:
        dc_temp = Sentinel2_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\{index}_sequenced_datacube\\')
        dcs_temp = Sentinel2_dcs(dc_temp_dic['MNDWI'], dc_temp)
        dcs_temp._inundation_removal(index, 'MNDWI_thr')
        dcs_temp = None

    ###### Link S2 inform
    dc_temp_dic = {}
    for index in ['NDVI_20m_noninun', 'B2_noninun', 'B3_noninun', 'B4_noninun', 'B5_noninun']:
        dc_temp_dic[index] = Sentinel2_dc(
            f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\{index}_sequenced_datacube\\')
    dcs_temp = Sentinel2_dcs(dc_temp_dic['NDVI_20m_noninun'], dc_temp_dic['B2_noninun'], dc_temp_dic['B3_noninun'], dc_temp_dic['B4_noninun'], dc_temp_dic['B5_noninun'])
    dcs_temp.link_GEDI_S2_inform('G:\A_veg\S2_all\GEDI\\MID_YZR_high_quality.xlsx', ['NDVI_20m_noninun', 'B2_noninun', 'B3_noninun', 'B4_noninun', 'B5_noninun'], retrieval_method='linear_interpolation')

    file_path = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\Original_Zipfile\\'
    output_path = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\'
    l2a_output_path = output_path + 'Sentinel2_L2A_output\\'
    QI_output_path = output_path + 'Sentinel2_L2A_output\\QI\\'
    bf.create_folder(l2a_output_path)
    bf.create_folder(QI_output_path)


