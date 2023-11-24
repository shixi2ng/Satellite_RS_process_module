# coding=utf-8
import gdal
import sys
import collections
# import snappy
# from snappy import PixelPos, Product, File, ProductData, ProductIO, ProductUtils, ProgressMonitor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import shutil
import datetime
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

# Input Snappy data style
np.seterr(divide='ignore', invalid='ignore')


def read_tiffile_size(tif_file: str) -> list:
    if type(tif_file) is str and tif_file.lower().endswith('.tif'):
        ds_temp = gdal.Open(tif_file)
        return [ds_temp.RasterXSize, ds_temp.RasterYSize]
    else:
        print('Invalid file type for the read_tiffile_size function')
        sys.exit(-1)


def get_tiffile_nodata(tif_file: str) -> int:
    if type(tif_file) is str and tif_file.lower().endswith('.tif'):
        ds_temp = gdal.Open(tif_file)
        no_data_value = ds_temp.GetRasterBand(1).GetNoDataValue()
        if str(no_data_value).endswith('.0'):
            no_data_value = int(no_data_value)
        return no_data_value
    else:
        print('Invalid file type for the read_tiffile_size function')
        sys.exit(-1)


def check_kwargs(kwargs_dic, key_list, func_name=None) -> None:
    for key_temp in key_list:
        if key_temp not in kwargs_dic.keys():
            if func_name is not None:
                print(f'The {key_temp} is not available for the {str(func_name)}!')
                sys.exit(-1)
            else:
                print(f'The {key_temp} is not available')
                sys.exit(-1)


def write_raster(ori_ds, new_array, file_path_f, file_name_f, raster_datatype=None, nodatavalue=None) -> None:
    if raster_datatype is None and nodatavalue is None:
        raster_datatype = gdal.GDT_Float32
        nodatavalue = np.nan
    elif raster_datatype is not None and nodatavalue is None:
        if raster_datatype is gdal.GDT_UInt16 or raster_datatype == 'UInt16':
            raster_datatype = gdal.GDT_UInt16
            nodatavalue = 65535
        elif raster_datatype is gdal.GDT_Int16 or raster_datatype == 'Int16':
            raster_datatype = gdal.GDT_Int16
            nodatavalue = -32768
        else:
            nodatavalue = 0
    elif raster_datatype is None and nodatavalue is not None:
        raster_datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    gt = ori_ds.GetGeoTransform()
    proj = ori_ds.GetProjection()
    if os.path.exists(file_path_f + file_name_f):
        os.remove(file_path_f + file_name_f)
    outds = driver.Create(file_path_f + file_name_f, xsize=new_array.shape[1], ysize=new_array.shape[0], bands=1,
                          eType=raster_datatype, options=['COMPRESS=LZW', 'PREDICTOR=2'])
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(new_array)
    outband.SetNoDataValue(nodatavalue)
    outband.FlushCache()
    outband = None
    outds = None


def union_list(small_list, big_list) -> list:
    union_list_temp = []
    if type(small_list) != list or type(big_list) != list:
        raise TypeError('Please input list for union list')

    for i in small_list:
        if i not in big_list:
            print(f'{i} is not supported!')
        else:
            union_list_temp.append(i)
    return union_list_temp


def qi_remove_cloud(processed_filepath, qi_filepath, dst_nodata=0, **kwargs):
    # Determine the process parameter
    if kwargs['cloud_removal_strategy'] == 'QI_all_cloud':
        cloud_indicator = [1, 2, 3, 8, 9, 10, 11]
    else:
        raise ValueError(f'The input Cloud removal strategy is not supported!')

    ### Due to the resample issue the sparse matrix is not appicable
    # if 'sparse_matrix_factor' in kwargs.keys():
    #     sparse_matrix_factor = kwargs['sparse_matrix_factor']
    # else:
    #     sparse_matrix_factor = False

    # process cloud removal and clipping sequence
    if 'cloud_clip_priority' in kwargs.keys():
        if kwargs['cloud_clip_priority'] == 'cloud':
            cloud_clip_seq = True
        elif kwargs['cloud_clip_priority'] == 'clip':
            cloud_clip_seq = False
        else:
            cloud_clip_seq = False
            raise ValueError(f'The input Cloud clip sequence para is not supported!')
    else:
        cloud_clip_seq = False

    # time1 = time.time()
    # qi_ds = gdal.Open(qi_filepath)
    # processed_ds = gdal.Open(processed_filepath, gdal.GA_Update)
    # qi_array = sp.csr_matrix(qi_ds.GetRasterBand(1).ReadAsArray())
    # processed_array = sp.csr_matrix(processed_ds.GetRasterBand(1).ReadAsArray())
    #
    # if qi_array.shape[0] != processed_array.shape[0] or qi_array.shape[1] != processed_array.shape[1]:
    #     print('Consistency error')
    #     sys.exit(-1)
    #
    # for indicator in cloud_indicator:
    #     qi_array[qi_array == indicator] = 64
    # processed_array[qi_array == 64] = dst_nodata
    # qi_ds = None
    # time1_all = time.time() - time1
    # processed_array2 = processed_ds.GetRasterBand(1).ReadAsArray()
    # r1 = np.sum(np.sum(processed_array2 - processed_array.toarray()))

    # Remove the cloud
    # time2 = time.time()
    qi_ds = gdal.Open(qi_filepath)
    processed_ds = gdal.Open(processed_filepath, gdal.GA_Update)

    # if sparse_matrix_factor and not cloud_clip_seq:
    #     qi_array = sp.lil_matrix(qi_ds.GetRasterBand(1).ReadAsArray())
    #     processed_array = sp.lil_matrix(processed_ds.GetRasterBand(1).ReadAsArray())
    # else:
    #     qi_array = qi_ds.GetRasterBand(1).ReadAsArray()
    #     processed_array = processed_ds.GetRasterBand(1).ReadAsArray()

    qi_array = qi_ds.GetRasterBand(1).ReadAsArray()
    processed_array = processed_ds.GetRasterBand(1).ReadAsArray()

    if qi_array.shape[0] != processed_array.shape[0] or qi_array.shape[1] != processed_array.shape[1]:
        raise Exception('Consistency error!')

    for indicator in cloud_indicator:
        qi_array[qi_array == indicator] = 64
    processed_array[qi_array == 64] = dst_nodata
    processed_array[np.logical_and(qi_array == dst_nodata, processed_array != dst_nodata)] = dst_nodata

    # time2_all = time.time() - time2
    # r2 = np.sum(np.sum(processed_array2 - processed_array.toarray()))

    # time3 = time.time()
    # qi_ds = gdal.Open(qi_filepath)
    # processed_ds = gdal.Open(processed_filepath, gdal.GA_Update)
    # qi_array = sp.csc_matrix(qi_ds.GetRasterBand(1).ReadAsArray())
    # processed_array = sp.csc_matrix(processed_ds.GetRasterBand(1).ReadAsArray())
    #
    # if qi_array.shape[0] != processed_array.shape[0] or qi_array.shape[1] != processed_array.shape[1]:
    #     print('Consistency error')
    #     sys.exit(-1)
    #
    # for indicator in cloud_indicator:
    #     qi_array[qi_array == indicator] = 64
    # processed_array[qi_array == 64] = dst_nodata
    # qi_ds = None
    # time3_all = time.time() - time3
    # r3 = np.sum(np.sum(processed_array2 - processed_array.toarray()))
    # print(time1_all, time2_all, time3_all, r1, r2, r3)
    # if sparse_matrix_factor and not cloud_clip_seq:
    #     processed_ds.GetRasterBand(1).WriteArray(processed_array.toarray())
    # else:
    #     processed_ds.GetRasterBand(1).WriteArray(processed_array)
    processed_ds.GetRasterBand(1).WriteArray(processed_array)
    processed_ds.FlushCache()
    processed_ds = None
    qi_ds = None


class Sentinel2_ds(object):

    def __init__(self, ori_zipfile_folder, work_env=None):
        """


        """
        # Define var
        self.S2_metadata = None
        self.subset_failure_file = []
        self.output_bounds = np.array([])
        self.raw_10m_bounds = np.array([])
        self.ROI = None
        self.ROI_name = None
        self.ori_folder = Path(ori_zipfile_folder).path_name
        self.orifile_list = None
        self.S2_metadata_size = np.nan
        self.date_list = []
        self.corrupted_file_list = []

        # Define key variables for subset (kwargs)
        self.size_control_factor = False
        self.cloud_removal_para = False
        self.vi_clip_factor = False
        self.sparsify_matrix_factor = False
        self.large_roi = False
        self.dst_coord = None

        # Define var for merge
        self.merge_cor_pathlist = []
        self.merge_indicator_list = []

        # Define key variables for merge (kwargs)

        # Remove all the duplicated data
        dup_data = bf.file_filter(self.ori_folder, ['.1.zip'])
        for dup in dup_data:
            os.remove(dup)

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

        # Create output path
        self.output_path = f'{self.work_env}Sentinel2_L2A_Output\\'
        self.shpfile_path = f'{self.work_env}shpfile\\'
        self.log_filepath = f'{self.work_env}log\\'
        bf.create_folder(self.output_path)
        bf.create_folder(self.log_filepath)
        bf.create_folder(self.shpfile_path)

        # Constant
        self.band_name_list = ['B01_60m.jp2', 'B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B05_20m.jp2', 'B06_20m.jp2',
                               'B07_20m.jp2', 'B8A_20m.jp2', 'B09_60m.jp2', 'B11_20m.jp2', 'B12_20m.jp2']
        self.band_output_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B11', 'B12']
        self.all_supported_index_list = ['QI', 'all_band', '4visual', 'NDVI', 'MNDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI',
                                         'NDVI_RE', 'NDVI_RE2', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',
                                         'B11', 'B12']
        self.process_steps = ('metadata', 'subset', 'merge')
        self.all_supported_datacube_list = copy.copy(self.all_supported_index_list)
        self.all_supported_datacube_list.extend([f'{i}_merged' for i in self.all_supported_index_list])

    def save_log_file(func):
        def wrapper(self, *args, **kwargs):

            #########################################################################
            # Document the log file and para file
            # The difference between log file and para file is that the log file contains the information for each run/debug
            # While the para file only comprises of the parameter for the latest run/debug
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
            for func_key, func_processing_name in zip(['metadata', 'subset', 'merge'], ['constructing metadata', 'executing subset', 'executing merge']):
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

    @save_log_file
    def construct_metadata(self):
        print('---------------------------- Start the construction of Metadata ----------------------------')
        start_temp = time.time()

        # Read the ori S2 zip file
        self.orifile_list = bf.file_filter(self.ori_folder, ['.zip', 'S2'], and_or_factor='and', subfolder_detection=True)
        self.orifile_list = [i for i in self.orifile_list if 'S2' in i.split('\\')[-1] and '.zip' in i.split('\\')[-1]]
        if not self.orifile_list:
            raise ValueError('There has no valid Sentinel-2 zip files in the input folder')

        # Read the corrupted S2 zip file
        corrupted_file_folder = os.path.join(self.work_env, 'Corrupted_S2_file\\')
        bf.create_folder(corrupted_file_folder)
        corrupted_file_list = bf.file_filter(corrupted_file_folder, ['.zip', 'S2'], and_or_factor='and', subfolder_detection=True)
        corrupted_file_list = [i for i in corrupted_file_list if 'S2' in i.split('\\')[-1] and '.zip' in i.split('\\')[-1]]

        # Get the detail of current metadata file
        if os.path.exists(self.work_env + 'Metadata.xlsx'):
            metadata_num = pd.read_excel(self.work_env + 'Metadata.xlsx').shape[0]
        else:
            metadata_num = 0

        # Get the detail of current corrupted metadata file
        if os.path.exists(self.work_env + 'Corrupted_metadata.xlsx'):
            corrupted_metadata_num = pd.read_excel(self.work_env + 'Corrupted_metadata.xlsx').shape[0]
        else:
            corrupted_metadata_num = 0

        ### Two possible conditions:
        # 1) The metadata is not generated when the DS is constructed for the first time.
        # 2) The metadata is not consistent with the real files since corrupted files may be found during the subset.
        #

        if not os.path.exists(self.work_env + 'Metadata.xlsx') or not os.path.exists(self.work_env + 'Corrupted_metadata.xlsx'):
            unzip_check_para = True
            update_factor = True
        elif metadata_num != len(self.orifile_list) or corrupted_metadata_num != len(corrupted_file_list):
            unzip_check_para = False
            update_factor = True
        else:
            update_factor = False

        if update_factor:
            corrupted_ori_file, corrupted_file_date, product_path, product_name, sensor_type, sensing_date, orbit_num, tile_num, width, height = ([] for i in range(10))
            for ori_file in self.orifile_list:
                if os.path.join(corrupted_file_folder, ori_file.split('\\')[-1]) in corrupted_file_list:
                    try:
                        os.remove(ori_file)
                    except:
                        pass
                else:
                    try:
                        if unzip_check_para:
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
                        file_name = ori_file.split('\\')[-1].split('.zip')[0]
                        corrupted_ori_file.append(file_name)
                        corrupted_file_date.append(file_name[file_name.find('_20') + 1: file_name.find('_20') + 9])
                        os.rename(ori_file, self.work_env + 'Corrupted_S2_file\\' + file_name)

            for corrupted_file in corrupted_file_list:
                file_name = corrupted_file.split('\\')[-1].split('.zip')[0]
                corrupted_ori_file.append(file_name)
                corrupted_file_date.append(file_name[file_name.find('_20') + 1: file_name.find('_20') + 9])

            Corrupted_metadata = pd.DataFrame({'Corrupted_file_name': corrupted_ori_file, 'File_Date': corrupted_file_date})
            Corrupted_metadata.to_excel(self.work_env + 'Corrupted_metadata.xlsx')

            self.S2_metadata = pd.DataFrame(
                {'Product_Path': product_path, 'Sensing_Date': sensing_date, 'Orbit_Num': orbit_num,
                 'Tile_Num': tile_num, 'Sensor_Type': sensor_type})
            self.S2_metadata.to_excel(self.work_env + 'Metadata.xlsx')
            self.S2_metadata = pd.read_excel(self.work_env + 'Metadata.xlsx')
        else:
            self.S2_metadata = pd.read_excel(self.work_env + 'Metadata.xlsx')
        self.S2_metadata.sort_values(by=['Sensing_Date'], ascending=True)
        self.S2_metadata_size = self.S2_metadata.shape[0]
        self.output_bounds = np.zeros([self.S2_metadata_size, 4]) * np.nan
        self.raw_10m_bounds = np.zeros([self.S2_metadata_size, 4]) * np.nan
        self.date_list = self.S2_metadata['Sensing_Date'].drop_duplicates().sort_values().tolist()
        print(f'Finish in {str(time.time() - start_temp)} sec!')
        print('----------------------------  End the construction of Metadata  ----------------------------')

    def process_corrupted_files(self, corrupted_file_list):
        if type(corrupted_file_list) != list:
            raise TypeError('argument 0 should be a list')

        for corrupted_file in corrupted_file_list:
            if os.path.exists(corrupted_file) and self.ori_folder in corrupted_file:
                bf.create_folder(os.path.join(self.work_env, 'Corrupted_S2_file\\'))
                if os.path.exists(os.path.join(self.work_env, 'Corrupted_S2_file\\' + os.path.basename(corrupted_file))):
                    os.remove(corrupted_file)
                else:
                    try:
                        os.rename(corrupted_file, os.path.join(self.work_env, 'Corrupted_S2_file\\' + os.path.basename(corrupted_file)))
                    except:
                        try:
                            shutil.copyfile(corrupted_file, os.path.join(self.work_env, 'Corrupted_S2_file\\' + os.path.basename(corrupted_file)))
                        except:
                            zip1 = zipfile.ZipFile(os.path.join(self.work_env, 'Corrupted_S2_file\\' + os.path.basename(corrupted_file)), 'w')
                            zip1.close()
            else:
                raise ValueError('Please input an existing corrupted file!')
        self.construct_metadata()

    def check_metadata_availability(self):
        # Check metadata availability
        if self.S2_metadata is None:
            try:
                self.construct_metadata()
            except:
                print('Please manually construct the S2_metadata before further processing!')
                sys.exit(-1)

    def check_output_band_statue(self, band_name, tiffile_serial_num, *args, **kwargs):

        # Define local var
        sensing_date = 'Sensing_Date'
        tile_num = 'Tile_Num'

        # Factor configuration
        if True in [band_temp not in self.band_output_list for band_temp in band_name]:
            raise ValueError(f'Band {band_name} is not valid!')

        if self.vi_clip_factor:
            output_path = f'{self.output_path}{self.ROI_name}_all_band\\'
        else:
            output_path = f'{self.output_path}_all_band\\'
        bf.create_folder(output_path)

        # Detect whether the required band was generated before
        try:
            if False in [os.path.exists(
                    f'{output_path}{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_{band_temp}.tif')
                         for band_temp in band_name]:
                self.subset_tiffiles(band_name, tiffile_serial_num, **kwargs)

            # Return output
            if False in [os.path.exists(
                    f'{output_path}{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_{band_temp}.tif')
                         for band_temp in band_name]:
                print(f'Something error processing {band_name}!')
                return None
            else:
                return [gdal.Open(
                    f'{output_path}{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_{band_temp}.tif')
                        for band_temp in band_name]

        except:
            return None

    @save_log_file
    def mp_subset(self, *args, **kwargs):
        if self.S2_metadata is None:
            print('Please construct the S2_metadata before the subset!')
            sys.exit(-1)
        i = range(self.S2_metadata.shape[0])
        # mp process
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.subset_tiffiles, repeat(args[0]), i, repeat(False), repeat(kwargs))
        self.process_corrupted_files(self.corrupted_file_list)

    @save_log_file
    def sequenced_subset(self, *args, **kwargs):
        if self.S2_metadata is None:
            print('Please construct the S2_metadata before the subset!')
            sys.exit(-1)
        # sequenced process
        for i in range(self.S2_metadata.shape[0]):
            self.subset_tiffiles(args[0], i, **kwargs)
        self.process_corrupted_files(self.corrupted_file_list)

    def process_subset_para(self, **kwargs):
        # Detect whether all the indicator are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('ROI', 'ROI_name', 'size_control_factor', 'cloud_removal_strategy', 'large_roi', 'dst_coord'):
                print(f'{kwarg_indicator} is not supported kwargs! Please double check!')
                sys.exit(-1)

        # process clip parameter
        if self.ROI is None:
            if 'ROI' in kwargs.keys():
                self.vi_clip_factor = True
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
                self.vi_clip_factor = False

        # process size control parameter
        if 'size_control_factor' in kwargs.keys():
            if type(kwargs['size_control_factor']) is bool:
                self.size_control_factor = kwargs['size_control_factor']
            else:
                print('Please mention the size_control_factor should be bool type!')
                self.size_control_factor = False
        else:
            self.size_control_factor = False

        # process cloud removal parameter
        if 'cloud_removal_strategy' in kwargs.keys():
            self.cloud_removal_para = True
        else:
            self.cloud_removal_para = False

        # process sparse matrix parameter
        # if 'sparsify_matrix_factor' in kwargs.keys():
        #     thalweg_temp.sparsify_matrix_factor = kwargs['sparsify_matrix_factor']
        # else:
        #     thalweg_temp.sparsify_matrix_factor = False

        # process cloud removal and clipping sequence
        if 'large_roi' in kwargs.keys():
            if kwargs['large_roi'] is True:
                if self.vi_clip_factor:
                    self.large_roi = True
                else:
                    print('Please input the ROI if large_roi is True')
                    sys.exit(-1)
            elif kwargs['large_roi'] is False:
                self.large_roi = False
            else:
                print('Large ROI factor need bool type input!')
                sys.exit(-1)
        else:
            self.large_roi = False

        # Process dst coordinate
        if 'dst_coord' in kwargs.keys():
            self.dst_coord = kwargs['dst_coord']
        else:
            self.dst_coord = False

    def generate_10m_output_bounds(self, tiffile_serial_num, **kwargs):

        time1 = time.time()
        # Define local var
        topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])
        sensing_date = 'Sensing_Date'
        tile_num = 'Tile_Num'
        VI = 'all_band'
        ds_temp = None

        # Define the output path
        if self.vi_clip_factor:
            b2_output_path = f'{self.output_path}{self.ROI_name}_{VI}\\'
        else:
            b2_output_path = f'{self.output_path}{VI}\\'
        bf.create_folder(b2_output_path)

        # Create the output bounds based on the 10-m Band2 images
        if self.output_bounds.shape[0] > tiffile_serial_num:
            if True in np.isnan(self.output_bounds[tiffile_serial_num, :]):
                temp_S2file_path = self.S2_metadata.iat[tiffile_serial_num, 1]
                b2_band_file_name = f'{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_B2'
                if not os.path.exists(b2_output_path + b2_band_file_name + '.tif'):
                    zfile = ZipFile(temp_S2file_path, 'r')
                    b2_file = [zfile_temp for zfile_temp in zfile.namelist() if 'B02_10m.jp2' in zfile_temp]
                    if len(b2_file) != 1:
                        print(
                            f'Data issue for the B2 file of all_cloud data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata.shape[0])})')
                        self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                        return
                    else:

                        try:
                            ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, b2_file[0]))
                        except:
                            zfile = None
                            self.corrupted_file_list.append(temp_S2file_path)
                            return

                        if ds_temp is None:
                            zfile = None
                            self.corrupted_file_list.append(temp_S2file_path)
                            return

                        if self.large_roi and self.vi_clip_factor:
                            if self.dst_coord is False:
                                gdal.Warp('/vsimem/' + b2_band_file_name + '.tif', ds_temp, xRes=10, yRes=10,
                                          cutlineDSName=self.ROI,
                                          outputType=gdal.GDT_UInt16, dstNodata=0)
                            else:
                                gdal.Warp('/vsimem/' + b2_band_file_name + '.tif', ds_temp, dstSRS=self.dst_coord,
                                          xRes=10,
                                          yRes=10, cutlineDSName=self.ROI, outputType=gdal.GDT_UInt16, dstNodata=0)
                        elif not self.large_roi and self.vi_clip_factor:
                            if self.dst_coord is False:
                                gdal.Warp('/vsimem/' + b2_band_file_name + '.tif', ds_temp, dstSRS=self.dst_coord,
                                          cutlineDSName=self.ROI,
                                          cropToCutline=True, xRes=10, yRes=10, outputType=gdal.GDT_UInt16, dstNodata=0)
                            else:
                                gdal.Warp('/vsimem/' + b2_band_file_name + '.tif', ds_temp, xRes=10, yRes=10,
                                          cutlineDSName=self.ROI,
                                          cropToCutline=True, outputType=gdal.GDT_UInt16, dstNodata=0)
                        elif not self.vi_clip_factor:
                            if self.dst_coord is False:
                                gdal.Warp('/vsimem/' + b2_band_file_name + '.tif', ds_temp, xRes=10, yRes=10,
                                          outputType=gdal.GDT_UInt16, dstNodata=0)
                            else:
                                gdal.Warp('/vsimem/' + b2_band_file_name + '.tif', ds_temp, dstSRS=self.dst_coord,
                                          xRes=10,
                                          yRes=10, outputType=gdal.GDT_UInt16, dstNodata=0)
                        gdal.Translate(b2_output_path + b2_band_file_name + '.tif',
                                       '/vsimem/' + b2_band_file_name + '.tif', options=topts, noData=0)
                        gdal.Unlink('/vsimem/' + b2_band_file_name + '.tif')
                    zfile.close()

                ds4bounds = gdal.Open(b2_output_path + b2_band_file_name + '.tif')
                ulx, xres, xskew, uly, yskew, yres = ds4bounds.GetGeoTransform()
                self.output_bounds[tiffile_serial_num, :] = np.array(
                    [ulx, uly + yres * ds4bounds.RasterYSize, ulx + xres * ds4bounds.RasterXSize, uly])
                ds4bounds = None

            # if thalweg_temp.cloud_clip_seq and True in np.isnan(thalweg_temp.raw_10m_bounds[tiffile_serial_num, :]):
            #     temp_S2file_path = thalweg_temp.S2_metadata.iat[tiffile_serial_num, 1]
            #     zfile = ZipFile(temp_S2file_path, 'r')
            #     b2_file = [zfile_temp for zfile_temp in zfile.namelist() if 'B02_10m.jp2' in zfile_temp]
            #     if len(b2_file) != 1:
            #         print(
            #             f'Data issue for the B2 file of all_cloud data ({str(tiffile_serial_num + 1)} of {str(thalweg_temp.S2_metadata.shape[0])})')
            #         thalweg_temp.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
            #         return
            #     else:
            #         ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, b2_file[0]))
            #         if thalweg_temp.cloud_clip_seq:
            #             ulx_temp, xres_temp, xskew_temp, uly_temp, yskew_temp, yres_temp = ds_temp.GetGeoTransform()
            #             thalweg_temp.raw_10m_bounds[tiffile_serial_num, :] = np.array(
            #                 [ulx_temp, uly_temp + yres_temp * ds_temp.RasterYSize,
            #                  ulx_temp + xres_temp * ds_temp.RasterXSize, uly_temp])
        else:
            print('The output bounds has some logical issue!')
            sys.exit(-1)
        # print(
        #     f' Generate 10m bounds of {str(thalweg_temp.S2_metadata[sensing_date][tiffile_serial_num])}_{str(thalweg_temp.S2_metadata[tile_num][tiffile_serial_num])} consume {time.time() - time1}s')

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
        ds_temp = None
        topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])
        time1, time2, time3 = 0, 0, 0
        processed_index_list = union_list(processed_index_list, self.all_supported_index_list)

        # Retrieve kwargs from args using the mp
        if args != () and type(args[0]) == dict:
            kwargs = copy.copy(args[0])

        # determine the subset indicator
        self.process_subset_para(**kwargs)
        self.check_metadata_availability()

        if processed_index_list != []:
            temp_S2file_path = self.S2_metadata.iat[tiffile_serial_num, 1]
            for VI in processed_index_list:
                start_temp = time.time()
                sensing_date = 'Sensing_Date'
                tile_num = 'Tile_Num'
                print(f'Start processing {VI} data for {str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')

                # Generate output folder
                if self.vi_clip_factor:
                    subset_output_path = f'{self.output_path}{self.ROI_name}_{VI}\\'
                    qi_path = f'{self.output_path}{self.ROI_name}_QI\\'
                    if VI in self.band_output_list:
                        subset_output_path = f'{self.output_path}{self.ROI_name}_all_band\\'
                else:
                    subset_output_path = f'{self.output_path}{VI}\\'
                    qi_path = f'{self.output_path}QI\\'
                    if VI in self.band_output_list:
                        subset_output_path = f'{self.output_path}_all_band\\'
                bf.create_folder(subset_output_path)

                # Define the file name for VI
                file_name = f'{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_{VI}'

                # Generate QI layer
                if (VI == 'QI' and overwritten_para) or (VI == 'QI' and not overwritten_para and not os.path.exists(qi_path + file_name)):
                    zfile = ZipFile(temp_S2file_path, 'r')
                    # Process subset indicator
                    self.generate_10m_output_bounds(tiffile_serial_num, **kwargs)
                    output_limit = (
                        int(self.output_bounds[tiffile_serial_num, 0]), int(self.output_bounds[tiffile_serial_num, 1]),
                        int(self.output_bounds[tiffile_serial_num, 2]), int(self.output_bounds[tiffile_serial_num, 3]))
                    band_all = [zfile_temp for zfile_temp in zfile.namelist() if 'SCL_20m.jp2' in zfile_temp]
                    if len(band_all) != 1:
                        print(
                            f'Something error during processing {VI} data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')
                        self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                    else:
                        for band_temp in band_all:

                            # Method 1
                            try:
                                ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, band_temp))
                            except:
                                zfile = None
                                self.corrupted_file_list.append(temp_S2file_path)
                                return

                            if ds_temp is None:
                                zfile = None
                                self.corrupted_file_list.append(temp_S2file_path)
                                return

                            if self.large_roi and self.vi_clip_factor:
                                if self.dst_coord is False:
                                    gdal.Warp('/vsimem/' + file_name + '.tif', ds_temp, xRes=10, yRes=10,
                                              cutlineDSName=self.ROI,
                                              outputType=gdal.GDT_UInt16, dstNodata=0, outputBounds=output_limit)
                                else:
                                    gdal.Warp('/vsimem/' + file_name + '.tif', ds_temp, xRes=10, yRes=10,
                                              cutlineDSName=self.ROI, dstSRS=self.dst_coord,
                                              outputType=gdal.GDT_UInt16, dstNodata=0, outputBounds=output_limit)
                            elif not self.large_roi and self.vi_clip_factor:
                                if self.dst_coord is False:
                                    gdal.Warp('/vsimem/' + file_name + '.tif', ds_temp, xRes=10, yRes=10,
                                              outputBounds=output_limit, cutlineDSName=self.ROI,
                                              cropToCutline=True, outputType=gdal.GDT_UInt16, dstNodata=0)
                                else:
                                    gdal.Warp('/vsimem/' + file_name + '.tif', ds_temp, xRes=10, yRes=10,
                                              dstSRS=self.dst_coord, outputBounds=output_limit, cutlineDSName=self.ROI,
                                              cropToCutline=True, outputType=gdal.GDT_UInt16, dstNodata=0)
                            elif not self.vi_clip_factor:
                                if self.dst_coord is False:
                                    gdal.Warp('/vsimem/' + file_name + '.tif', ds_temp, xRes=10, yRes=10,
                                              outputBounds=output_limit,
                                              outputType=gdal.GDT_UInt16, dstNodata=0)
                                else:
                                    gdal.Warp('/vsimem/' + file_name + '.tif', ds_temp, xRes=10, yRes=10,
                                              dstSRS=self.dst_coord, outputBounds=output_limit, cropToCutline=True,
                                              outputType=gdal.GDT_UInt16, dstNodata=0)
                            gdal.Translate(qi_path + file_name + '.tif', '/vsimem/' + file_name + '.tif', options=topts,
                                           noData=0, outputType=gdal.GDT_UInt16)
                            gdal.Unlink('/vsimem/' + file_name + '.tif')
                    zfile.close()
                    # Method 2

                    # ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, band_temp))
                    # if thalweg_temp.dst_coord is False:
                    #     gdal.Warp('/vsimem/' + file_name + '.tif', ds_temp, xRes=10, yRes=10, outputBounds=output_limit, outputType=gdal.GDT_UInt16, dstNodata=0)
                    # else:
                    #     gdal.Warp('/vsimem/' + file_name + '.tif', ds_temp, xRes=10, yRes=10,
                    #               outputBounds=output_limit, outputType=gdal.GDT_UInt16, dstNodata=0, dstSRS=thalweg_temp.dst_coord)
                    #
                    # if thalweg_temp.large_roi and thalweg_temp.vi_clip_factor:
                    #     gdal.Warp('/vsimem/' + file_name + '2.tif', '/vsimem/' + file_name + '.tif', xRes=10, yRes=10,
                    #               cutlineDSName=thalweg_temp.ROI,  dstNodata=0)
                    # elif not thalweg_temp.large_roi and thalweg_temp.vi_clip_factor:
                    #     gdal.Warp('/vsimem/' + file_name + '2.tif', '/vsimem/' + file_name + '.tif', xRes=10, yRes=10,
                    #               cutlineDSName=thalweg_temp.ROI,
                    #               cropToCutline=True, dstNodata=0)
                    # elif not thalweg_temp.vi_clip_factor:
                    #     gdal.Warp('/vsimem/' + file_name + '2.tif', '/vsimem/' + file_name + '.tif', xRes=10, yRes=10,
                    #               outputType=gdal.GDT_UInt16, dstNodata=0)
                    #
                    # gdal.Unlink('/vsimem/' + file_name + '.tif')
                    # gdal.Translate(qi_path + file_name + '.tif', '/vsimem/' + file_name + '2.tif', options=topts,
                    #                noData=0, outputType=gdal.GDT_UInt16)
                    # gdal.Unlink('/vsimem/' + file_name + '2.tif')

                # Subset band images
                elif VI == 'all_band' or VI == '4visual' or VI in self.band_output_list:

                    # Check the output band
                    if VI == 'all_band':
                        band_name_list, band_output_list = self.band_name_list, self.band_output_list
                    elif VI == '4visual':
                        band_name_list, band_output_list = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B05_20m.jp2', 'B8A_20m.jp2', 'B11_20m.jp2'],\
                                                           ['B2', 'B3', 'B4', 'B5', 'B8', 'B11']
                    elif VI in self.band_output_list:
                        band_name_list, band_output_list = [self.band_name_list[self.band_output_list.index(VI)]], [VI]
                    else:
                        print('Code error!')
                        sys.exit(-1)

                    if overwritten_para or False in [os.path.exists(subset_output_path + str(self.S2_metadata[sensing_date][tiffile_serial_num]) + '_' + str(
                                    self.S2_metadata[tile_num][tiffile_serial_num]) + '_' + str(band_temp) + '.tif') for band_temp in band_output_list]:

                        # Process subset indicator
                        self.generate_10m_output_bounds(tiffile_serial_num, **kwargs)
                        output_limit = (
                            int(self.output_bounds[tiffile_serial_num, 0]),
                            int(self.output_bounds[tiffile_serial_num, 1]),
                            int(self.output_bounds[tiffile_serial_num, 2]),
                            int(self.output_bounds[tiffile_serial_num, 3]))

                        # read zipfile
                        zfile = ZipFile(temp_S2file_path, 'r')

                        for band_name, band_output in zip(band_name_list, band_output_list):
                            if band_output != 'B2':
                                all_band_file_name = f'{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_{str(band_output)}'
                                if not os.path.exists(subset_output_path + all_band_file_name + '.tif') or overwritten_para:
                                    band_all = [zfile_temp for zfile_temp in zfile.namelist() if band_name in zfile_temp]
                                    if len(band_all) != 1:
                                        print(
                                            f'Something error during processing {band_output} of {VI} data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')
                                        self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                                    else:
                                        for band_temp in band_all:
                                            t1 = time.time()

                                            try:
                                                ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, band_temp))
                                            except:
                                                zfile = None
                                                self.corrupted_file_list.append(temp_S2file_path)
                                                return

                                            if ds_temp is None:
                                                zfile = None
                                                self.corrupted_file_list.append(temp_S2file_path)
                                                return

                                            time1 = time.time() - t1
                                            t2 = time.time()
                                            if self.large_roi and self.vi_clip_factor:
                                                if self.dst_coord is not False:
                                                    gdal.Warp('/vsimem/' + all_band_file_name + '.tif', ds_temp,
                                                              xRes=10, yRes=10, dstSRS=self.dst_coord,
                                                              cutlineDSName=self.ROI,
                                                              outputBounds=output_limit, outputType=gdal.GDT_UInt16,
                                                              dstNodata=0)
                                                else:
                                                    gdal.Warp('/vsimem/' + all_band_file_name + '.tif', ds_temp,
                                                              xRes=10, yRes=10,
                                                              cutlineDSName=self.ROI,
                                                              outputBounds=output_limit,
                                                              outputType=gdal.GDT_UInt16,
                                                              dstNodata=0)
                                            elif not self.large_roi and self.vi_clip_factor:
                                                if self.dst_coord is not False:
                                                    gdal.Warp('/vsimem/' + all_band_file_name + '.tif', ds_temp,
                                                              xRes=10, yRes=10,
                                                              cutlineDSName=self.ROI, dstSRS=self.dst_coord,
                                                              cropToCutline=True, outputBounds=output_limit,
                                                              outputType=gdal.GDT_UInt16,
                                                              dstNodata=0)
                                                else:
                                                    gdal.Warp('/vsimem/' + all_band_file_name + '.tif', ds_temp,
                                                              xRes=10, yRes=10,
                                                              cutlineDSName=self.ROI,
                                                              cropToCutline=True, outputBounds=output_limit,
                                                              outputType=gdal.GDT_UInt16,
                                                              dstNodata=0)
                                            elif not self.vi_clip_factor:
                                                if self.dst_coord is not False:
                                                    gdal.Warp('/vsimem/' + all_band_file_name + '.tif', ds_temp,
                                                              xRes=10, yRes=10, dstSRS=self.dst_coord,
                                                              outputBounds=output_limit, outputType=gdal.GDT_UInt16,
                                                              dstNodata=0)
                                                else:
                                                    gdal.Warp('/vsimem/' + all_band_file_name + '.tif', ds_temp,
                                                              xRes=10, yRes=10,
                                                              outputBounds=output_limit, outputType=gdal.GDT_UInt16,
                                                              dstNodata=0)
                                            time2 = time.time() - t2
                                            t3 = time.time()
                                            if self.cloud_removal_para:
                                                qi_file_path = f'{qi_path}{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_QI.tif'
                                                if not os.path.exists(qi_file_path):
                                                    self.subset_tiffiles(['QI'], tiffile_serial_num, **kwargs)
                                                qi_remove_cloud('/vsimem/' + all_band_file_name + '.tif',
                                                                qi_file_path, dst_nodata=0,
                                                                **kwargs)
                                            time3 = time.time() - t3
                                            t4 = time.time()
                                            gdal.Translate(subset_output_path + all_band_file_name + '.tif',
                                                           '/vsimem/' + all_band_file_name + '.tif', options=topts,
                                                           noData=0)
                                            gdal.Unlink('/vsimem/' + all_band_file_name + '.tif')
                                            time4 = time.time() - t4
                                            # print(time1, time2, time3, time4)
                            else:
                                if not os.path.exists(
                                        f'{subset_output_path}\\{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_B2.tif'):
                                    print('Code error for B2!')
                                    sys.exit(-1)
                                else:
                                    if self.cloud_removal_para:
                                        qi_file_path = f'{qi_path}{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_QI.tif'
                                        if not os.path.exists(qi_file_path):
                                            self.subset_tiffiles(['QI'], tiffile_serial_num, **kwargs)
                                        qi_remove_cloud(
                                            f'{subset_output_path}\\{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_B2.tif',
                                            qi_file_path, dst_nodata=0, **kwargs)
                        zfile.close()

                elif not overwritten_para and not os.path.exists(subset_output_path + file_name + '.tif') and not (VI == 'QI' or VI == 'all_band' or VI == '4visual' or VI in self.band_output_list):
                    if VI == 'NDVI':
                        # time1 = time.time()
                        ds_list = self.check_output_band_statue(['B8', 'B4'], tiffile_serial_num, **kwargs)
                        if ds_list is not None:
                            # time1 = time.time()
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B4_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                            B8_array[B4_array == 0] = 0
                            B4_array[B8_array == 0] = 0
                            output_array = (B8_array - B4_array) / (B8_array + B4_array)
                            B4_array = None
                            B8_array = None
                            # print(time.time()-time1)
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'MNDWI':
                        ds_list = self.check_output_band_statue(['B3', 'B11'], tiffile_serial_num, **kwargs)
                        if ds_list is not None:
                            if self.sparsify_matrix_factor:
                                B3_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B11_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B3_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                B11_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                            B3_array[B11_array == 0] = 0
                            B11_array[B3_array == 0] = 0
                            output_array = (B3_array - B11_array) / (B3_array + B11_array)
                            B3_array = None
                            B11_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'EVI':
                        ds_list = self.check_output_band_statue(['B8', 'B4', 'B22'], tiffile_serial_num, **kwargs)
                        if ds_list is not None:
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B4_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B2_array = sp.csr_matrix(ds_list[2].GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                                B2_array = ds_list[2].GetRasterBand(1).ReadAsArray().astype(np.float)
                            output_array = 2.5 * (B8_array - B4_array) / (B8_array + 6 * B4_array - 7.5 * B2_array + 1)
                            B8_array[B4_array == 0] = 0
                            B4_array[B8_array == 0] = 0
                            B4_array = None
                            B8_array = None
                            B2_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'EVI2':
                        ds_list = self.check_output_band_statue(['B8', 'B4'], tiffile_serial_num, **kwargs)
                        if ds_list is not None:
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B4_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                            B8_array[B4_array == 0] = 0
                            B4_array[B8_array == 0] = 0
                            output_array = 2.5 * (B8_array - B4_array) / (B8_array + 2.4 * B4_array + 1)
                            B4_array = None
                            B8_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'GNDVI':
                        ds_list = self.check_output_band_statue(['B8', 'B3'], tiffile_serial_num, **kwargs)
                        if ds_list is not None:
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B3_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                B3_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                            B8_array[B3_array == 0] = 0
                            B3_array[B8_array == 0] = 0
                            output_array = (B8_array - B3_array) / (B8_array + B3_array)
                            B3_array = None
                            B8_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'NDVI_RE':
                        ds_list = self.check_output_band_statue(['B7', 'B5'], tiffile_serial_num, **kwargs)
                        if ds_list is not None:
                            if self.sparsify_matrix_factor:
                                B7_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B5_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B7_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                B5_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                            B7_array[B5_array == 0] = 0
                            B5_array[B7_array == 0] = 0
                            output_array = (B7_array - B5_array) / (B7_array + B5_array)
                            B7_array = None
                            B5_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'NDVI_RE2':
                        ds_list = self.check_output_band_statue(['B8', 'B5'], tiffile_serial_num, **kwargs)
                        if ds_list is not None:
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B5_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                B5_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                            B8_array[B5_array == 0] = 0
                            B5_array[B8_array == 0] = 0
                            output_array = (B8_array - B5_array) / (B8_array + B5_array)
                            B8_array = None
                            B5_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'OSAVI':
                        ds_list = self.check_output_band_statue(['B8', 'B4'], tiffile_serial_num, **kwargs)
                        if ds_list is not None:
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B4_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                            B8_array[B4_array == 0] = 0
                            B4_array[B8_array == 0] = 0
                            output_array = 1.16 * (B8_array - B4_array) / (B8_array + B4_array + 0.16)
                            B4_array = None
                            B8_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'IEI':
                        ds_list = self.check_output_band_statue(['B8', 'B4'], tiffile_serial_num, **kwargs)
                        if ds_list is not None:
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B4_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                            B8_array[B4_array == 0] = 0
                            B4_array[B8_array == 0] = 0
                            output_array = 1.5 * (B8_array - B4_array) / (B8_array + B4_array + 0.5)
                            B4_array = None
                            B8_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    else:
                        print(f'{VI} is not supported!')
                        sys.exit(-1)
                    output_array[np.logical_or(output_array > 1, output_array < -1)] = np.nan
                    time_1 = time.time()
                    if self.size_control_factor is True:
                        output_array[np.isnan(output_array)] = -3.2768
                        output_array = output_array * 10000
                        write_raster(ds_list[0], output_array, subset_output_path, file_name + '.tif',
                                     raster_datatype=gdal.GDT_Int16)
                    else:
                        write_raster(ds_list[0], output_array, subset_output_path, file_name + '.tif',
                                     raster_datatype=gdal.GDT_Float32)
                    print(f'** Writing {VI} consume about {time.time() - time_1}')
                print(
                    f'Finish processing {VI} data in {str(time.time() - start_temp)}s ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')
        else:
            raise TypeError('Caution! the input variable VI_list should be a list and make sure all of them are in Capital Letter')
        return

    def band_composition(self):
        pass

    def check_subset_integrality(self, indicator: list, **kwargs) -> None:
        self.check_metadata_availability()
        indicator = union_list(indicator, self.all_supported_index_list)
        if 'QI' in indicator:
            indicator.remove('QI')

        for indicator_temp in indicator:
            if indicator_temp not in self.merge_indicator_list:
                try:
                    if 'ROI_name' in kwargs.keys():
                        roi = kwargs['ROI_name']
                    elif self.ROI_name is not None:
                        roi = self.ROI_name
                    elif 'ROI' in kwargs.keys():
                        roi = kwargs['ROI'].split('\\')[-1].split('.')[0]
                    elif self.ROI is not None:
                        roi = self.ROI.split('\\')[-1].split('.')[0]
                    else:
                        if os.path.exists(self.log_filepath + 'para_file.txt'):
                            para_file = open(f"{self.log_filepath}para_file.txt")
                            para_txt = para_file.read()
                            ROI_all = [i for i in para_txt.split('\n') if i.startswith('ROI')]
                            if len(ROI_all) == 0:
                                roi = ''
                            else:
                                if ROI_all[-1].split(':')[0] == 'ROI':
                                    roi = ROI_all[-1].split(':')[-1]
                                elif ROI_all[-1].split(':')[0] == 'ROI_name':
                                    roi = ROI_all[-1].split(':')[-1].split('\\')[-1].split('.')[0]
                                else:
                                    roi = ''
                        else:
                            roi = ''
                except:
                    raise TypeError('The type of ROI is incorrect!')

                # create check path
                if roi == '':
                    check_path = f'{self.output_path}{str(indicator_temp)}\\'
                else:
                    check_path = f'{self.output_path}{str(roi)}_{str(indicator_temp)}\\'

                # consistency check
                if os.path.exists(check_path):
                    valid_file = bf.file_filter(check_path, [indicator_temp])
                    if self.S2_metadata_size == len(valid_file):
                        self.merge_indicator_list.append(indicator_temp)
                        self.merge_cor_pathlist.append(check_path)
                    elif self.S2_metadata_size < len(valid_file):
                        raise Exception(f'Please make sure the {indicator_temp} was not manually modified!')
                    elif self.S2_metadata_size > len(valid_file):
                        all_files = [str(self.S2_metadata['Sensing_Date'][i]) + '_' + self.S2_metadata['Tile_Num'][i] for i in range(self.S2_metadata_size)]
                        for temp_valid in valid_file:
                            if temp_valid.split('\\')[-1].split('.')[0][0:14] in all_files:
                                all_files.remove(temp_valid.split('\\')[-1].split('.')[0][0:14])
                        raise Exception(f'Mentioned! These files were missing！{all_files}')

    def retrieve_para_from_para_file(self, required_para_name_list, **kwargs):

        if not os.path.exists(f'{self.log_filepath}para_file.txt'):
            print('The para file is not established yet')
            sys.exit(-1)
        else:
            para_file = open(f"{self.log_filepath}para_file.txt", "r+")
            para_raw_txt = para_file.read().split('\n')

        for para in required_para_name_list:
            if para in self.__dir__():
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
                                t = float(q.split(para +':')[-1])
                                self.__dict__[para] = float(q.split(para + ':')[-1])
                            except:
                                self.__dict__[para] = q.split(para + ':')[-1]

    def check_merge_para(self, **kwargs):

        # Detect whether all the indicator are valid
        merge_para = ['ROI', 'ROI_name', 'queried_from_parafile']
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in merge_para:
                print(f'{kwarg_indicator} is not supported kwargs! Please double check!')
                sys.exit(-1)

        # Get the ROI or outbounds
        if 'ROI' not in kwargs.keys() and self.ROI is None:
            if 'queried_from_parafile' in kwargs.keys():
                self.retrieve_para_from_para_file(['ROI'])
            else:
                self.ROI = None
        elif 'ROI' in kwargs.keys():
            if '.shp' in kwargs['ROI'] and os.path.exists(kwargs['ROI']):
                self.ROI = kwargs['ROI']
            else:
                print('Please input valid shp file for clip!')
                sys.exit(-1)

        if 'ROI_name' not in kwargs.keys() and self.ROI_name is None:
            if 'queried_from_parafile' in kwargs.keys():
                self.retrieve_para_from_para_file(['ROI_name'])
            else:
                self.ROI = None
        elif 'ROI_name' in kwargs.keys() and self.ROI_name is None:
            self.ROI_name = kwargs['ROI_name']
        elif self.ROI is not None:
            self.ROI_name = self.ROI.split('\\')[-1].split('.')[0]

    @save_log_file
    def sequenced_merge(self, *args, **kwargs):
        self.check_subset_integrality(args[0], **kwargs)
        if not self.date_list:
            print('No valid date!')
            sys.exit(-1)
        for i in range(len(self.date_list)):
            self.merge_by_date(args[0], i, **kwargs)

    @save_log_file
    def mp_merge(self, *args, **kwargs):
        self.check_subset_integrality(args[0], **kwargs)
        if not self.date_list:
            print('No valid date!')
            sys.exit(-1)
        i = range(len(self.date_list))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.merge_by_date, repeat(args[0]), i, repeat(kwargs))

    def merge_by_date(self, processed_indicator_list: list, date_index: int, *args, **kwargs) -> None:
        print(f'Start merging the {str(processed_indicator_list)} of {self.date_list[date_index]} ({str(date_index)} of {str(len(self.date_list))})')

        # Retrieve kwargs from args using the mp
        if args != () and type(args[0]) == dict:
            kwargs = copy.copy(args[0])

        # initial check
        self.check_merge_para(**kwargs)

        # merge by date
        if len(self.merge_indicator_list) == 0:
            print('Please input valid processed indicator list for merge!')
            sys.exit(-1)
        elif len(self.merge_indicator_list) != len(self.merge_cor_pathlist):
            print('Code error')
            sys.exit(-1)

        # Check whether it is essential for merge
        date_temp = self.date_list[date_index]
        date_sum = 0
        for q in self.S2_metadata['Sensing_Date']:
            if q == date_temp:
                date_sum += 1

        for indicator_temp, indicator_filepath in zip(self.merge_indicator_list, self.merge_cor_pathlist):
            time_start = time.time()
            print(f'Start merging {indicator_temp} of {str(date_temp)} ({str(date_index + 1)} of {str(len(self.date_list))})')
            # Start processing
            valid_file_list = bf.file_filter(indicator_filepath, [indicator_temp, str(date_temp)], and_or_factor='and')
            if self.ROI is not None and self.ROI_name is not None:
                merge_file_path = f'{self.output_path}{self.ROI_name}_{indicator_temp}_merged\\'
            else:
                merge_file_path = f'{self.output_path}{indicator_temp}_merged\\'
            bf.create_folder(merge_file_path)

            if date_sum == 0:
                print('Code error!')
                sys.exit(-1)
            elif date_sum == 1 and len(valid_file_list) == 1:
                if self.ROI is not None:
                    gdal.Warp(f'{merge_file_path}{str(date_temp)}_{indicator_temp}.tif', valid_file_list[0], xRes=10, yRes=10,
                              cutlineDSName=self.ROI, cropToCutline=True)
                else:
                    gdal.Warp(f'{merge_file_path}{str(date_temp)}_{indicator_temp}.tif', valid_file_list[0], xRes=10,
                              yRes=10)
            elif date_sum > 1 and date_sum == len(valid_file_list):
                valid_vrt_file = gdal.BuildVRT('temp.vrt', valid_file_list, xRes=10, yRes=10)
                if self.ROI is not None:
                    gdal.Warp(f'{merge_file_path}{str(date_temp)}_{indicator_temp}.tif', valid_vrt_file, xRes=10, yRes=10, cutlineDSName=self.ROI,
                              cropToCutline=True)
                else:
                    gdal.Translate(f'{merge_file_path}{str(date_temp)}_{indicator_temp}.tif', valid_vrt_file, xRes=10, yRes=10)
            else:
                print('Integrality check failed!')
                sys.exit(-1)
            valid_vrt_file = None
            print(f'Finish merging {indicator_temp} of {str(date_temp)} in {str(time.time()-time_start)}s ({str(date_index + 1)} of {str(len(self.date_list))})')

    def check_temporal_consistency(self, *args, **kwargs):

        check_kwargs(kwargs, ['ROI', 'ROI_name'], func_name='check_temporal_consistency')

        if type(args[0]) == list:
            processed_list = union_list(args[0], self.all_supported_datacube_list)
            if len(processed_list) == 0:
                print('None valid index for temporal process')
                return
            else:
                for index_temp in processed_list:
                    check_folder_temp = bf.file_filter(self.output_path, [str(index_temp)])
                    if len(check_folder_temp) == 0:
                        print('Please generate the index before the temporal process!')
                        break
                    elif len(check_folder_temp) == 1:
                        index_name = check_folder_temp[0].split('\\')[-1]
                        print(f"The {index_name} was regarded as the temporal process object!")
                        processed_file_all = bf.file_filter(check_folder_temp, ['.tif'])
                        raster_size_list = []
                        if len(processed_file_all) > 200:
                            with concurrent.futures.ProcessPoolExecutor() as executor:
                                for result in executor.map(read_tiffile_size, processed_file_all):
                                    raster_size_list.append(result)
                        else:
                            for tif_file in processed_file_all:
                                raster_size_list.append(read_tiffile_size(tif_file))
                    else:
                        if self.ROI_name is not None:
                            roi_temp = self.ROI_name
                        elif self.ROI is not None:
                            roi_temp = self.ROI.split('\\')[-1].split('.')[0]
                        elif 'ROI_name' in kwargs.keys():
                            roi_temp = kwargs['ROI_name']
                        elif 'ROI' in kwargs.keys():
                            roi_temp = kwargs['ROI'].split('\\')[-1].split('.')[0]
                        else:
                            roi_temp = ''

                        if roi_temp == '':
                            check_folder_temp = [path for path in check_folder_temp if f'\\{str(index_temp)}' in path]
                        else:
                            check_folder_temp = [path for path in check_folder_temp if f'\\{roi_temp}_{str(index_temp)}' in path]

                        if len(check_folder_temp) != 1:
                            print('Please input tht correct ROI name for the temporal analysis!')
                            sys.exit(-1)
                        else:
                            index_name = check_folder_temp[0].split('\\')[-1]
                            print(f"The {index_name} was regarded as the temporal process object!")
                            processed_file_all = bf.file_filter(check_folder_temp, ['.tif'])
                            raster_size_list = []
                            if len(processed_file_all) > 200:
                                with concurrent.futures.ProcessPoolExecutor() as executor:
                                    for result in executor.map(read_tiffile_size, processed_file_all):
                                        raster_size_list.append(result)
                            else:
                                for tif_file in processed_file_all:
                                    raster_size_list.append(read_tiffile_size(tif_file))

                    if False in [com == raster_size_list[0] for com in raster_size_list]:
                        print('The dataset is temporal inconsistency, please double check!')
                        sys.exit(-1)
                    else:
                        return check_folder_temp[0]

        elif type(args[0]) == str:
            if os.path.exists(args[0]):
                raster_size_list = []
                processed_file_all = bf.file_filter(bf.Path(args[0]).path_name, ['.tif'])
                if len(processed_file_all) > 200:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        for result in executor.map(read_tiffile_size, processed_file_all):
                            raster_size_list.append(result)
                else:
                    for tif_file in processed_file_all:
                        raster_size_list.append(read_tiffile_size(tif_file))

                if False in [ras_size_temp == raster_size_list[0] for ras_size_temp in raster_size_list]:
                    print('The dataset is temporal inconsistency, please double check!')
                    sys.exit(-1)
                else:
                    return bf.Path(args[0]).path_name

            else:
                print('Please input a valid path for the arguments!')
                sys.exit(-1)

        else:
            print('Please input the indicator name or the valid file path for temporal analysis')
            sys.exit(-1)

    def composition(self, *args, **kwargs):
        self.check_temporal_consistency()

    @save_log_file
    def export2datacube(self, vi: str, *args, **kwargs) -> None:

        # check the temporal consistency and completeness of the index series
        input_index_path = self.check_temporal_consistency(vi, **kwargs)
        input_files = bf.file_filter(input_index_path, ['.tif'])
        if len(input_files) != len(self.date_list):
            print('Consistency issue between the metadata date list and the index date list!')
            sys.exit(-1)

        # define var
        header_dic = {'DC_origin': input_index_path.split('\\')[-2],'Date_list': self.date_list}
        datacube_output_path = os.path.dirname(os.path.dirname(input_index_path)) + input_index_path.split('\\')[-2] + '_dc\\'
        raster_x_size, raster_y_size = read_tiffile_size(input_files[0])[0], read_tiffile_size(input_files[0])[1]
        nodata_value = get_tiffile_nodata(input_files[0])
        datacube = np.zeros([raster_y_size, raster_x_size, len(input_files)]) * nodata_value

        # Generate the datacube
        date_index = 0
        for date_temp in self.date_list:
            for file_name in input_files:
                if str(date_temp) in file_name:
                    break
                elif file_name == input_files[-1] and str(date_temp) not in file_name:
                    print('Date list has discrepancy with the file list')
                    sys.exit(-1)
            ds_temp = gdal.Open(file_name)
            raster_temp = ds_temp.GetRasterBand(1).ReadAsArray()
            datacube[:, :, date_index] = raster_temp.reshape(raster_temp.shape[0], raster_temp.shape[1], 1)
            date_index += 1

        # Output the datacube
        np.save(datacube_output_path + input_index_path.split('\\')[-2] + '_dc.npy', datacube)
        np.save(datacube_output_path + input_index_path.split('\\')[-2] + '_header.npy', header_dic)


class RS_datacube(object):
    def __init__(self, datacube_path: str, *args, **kwargs) -> None:
        pass

    def read_datacube(self):
        pass

    def lsp_extraction(self):
        pass

    def phenology_process(self):
        pass

    def vi_process(self, l2a_output_path_f, mask_path, index_list, study_area_f, specific_name_list_f,
                   overwritten_para_clip_f, overwritten_para_cloud_f, overwritten_para_dc_f, overwritten_para_sdc):
        # VI_LIST check
        try:
            vi_list_temp = copy.copy(index_list)
            vi_list_temp.remove('QI')
        except:
            print('QI is obligatory file')
            sys.exit(-1)
        if len(vi_list_temp) == 0:
            print('There can not only have QI one single dataset')
            sys.exit(-1)
        # create folder
        for vi in index_list:
            if not os.path.exists(l2a_output_path_f + vi):
                print(vi + 'folders are missing')
                sys.exit(-1)
            else:
                temp_output_path = l2a_output_path_f + vi + '_' + study_area_f + '\\'
                bf.create_folder(temp_output_path)
            for name in specific_name_list_f:
                temp_created_output_path = temp_output_path + name + '\\'
                bf.create_folder(temp_created_output_path)

        # clip the image
        if 'clipped' in specific_name_list_f:
            print('Start clipping all VI image')
            bf.create_folder(l2a_output_path_f + 'temp')
            if 'QI' not in index_list:
                print('Please notice that QI dataset is necessary for further process')
                sys.exit(-1)
            else:
                VI_list_temp = copy.copy(index_list)
                VI_list_temp.remove('QI')
            containing_word = ['.tif']
            eliminating_all_non_tif_file(l2a_output_path_f + 'QI' + '\\')
            QI_temp_file_list = bf.file_filter(l2a_output_path_f + 'QI' + '\\', containing_word)
            for qi_temp_file in QI_temp_file_list:
                file_information = [qi_temp_file[qi_temp_file.find('202'):qi_temp_file.find('202') + 14]]
                TEMP_QI_DS = gdal.Open(qi_temp_file)
                QI_cols = TEMP_QI_DS.RasterXSize
                QI_rows = TEMP_QI_DS.RasterYSize
                temp_file_list = []
                for vi in index_list:
                    if overwritten_para_clip_f or not os.path.exists(
                            l2a_output_path_f + vi + '_' + study_area_f + '\\clipped\\' + file_information[
                                0] + '_' + vi + '_clipped.tif'):
                        eliminating_all_non_tif_file(l2a_output_path_f + vi + '\\')
                        temp_file_name = bf.file_filter(l2a_output_path_f + vi + '\\', file_information)
                        if len(temp_file_name) == 0 or len(temp_file_name) > 1:
                            print('VI File consistency problem occurred')
                            sys.exit(-1)
                        else:
                            temp_file_list.append(temp_file_name[0])
                            TEMP_VI_DS = gdal.Open(temp_file_name[0])
                            VI_cols = TEMP_VI_DS.RasterXSize
                            VI_rows = TEMP_VI_DS.RasterYSize
                            if VI_rows == QI_rows and VI_cols == QI_cols:
                                print(f'Start clip the ' + file_information[0] + vi + ' image')
                                if '49R' not in file_information[0]:
                                    TEMP_warp = gdal.Warp(l2a_output_path_f + 'temp\\temp.tif', TEMP_VI_DS,
                                                          dstSRS='EPSG:32649', xRes=10, yRes=10, dstNodata=np.nan)
                                    gdal.Warp(
                                        l2a_output_path_f + vi + '_' + study_area_f + '\\clipped\\' + file_information[
                                            0] + '_' + vi + '_clipped.tif', TEMP_warp, cutlineDSName=mask_path,
                                        cropToCutline=True, dstNodata=np.nan, xRes=10, yRes=10)
                                else:
                                    gdal.Warp(
                                        l2a_output_path_f + vi + '_' + study_area_f + '\\clipped\\' + file_information[
                                            0] + '_' + vi + '_clipped.tif', TEMP_VI_DS, cutlineDSName=mask_path,
                                        cropToCutline=True, dstNodata=np.nan, xRes=10, yRes=10)
                                print(f'Successfully clip the ' + file_information[0] + vi + ' image')
                            else:
                                print('VI File spatial consistency problem occurred')
                                sys.exit(-1)
            print('Finish clipping all VI image')
        else:
            print('The obligatory process is clipped')

        # Remove the pixel influenced by cloud
        if 'cloud_free' in specific_name_list_f:
            print('Start removing cloud in all VI image')
            if 'QI' not in index_list:
                print('Please notice that QI dataset is necessary for this process')
                sys.exit(-1)
            else:
                VI_list_temp = copy.copy(index_list)
                VI_list_temp.remove('QI')
            containing_word = ['_clipped.tif']
            eliminating_all_non_tif_file(l2a_output_path_f + 'QI_' + study_area_f + '\\clipped\\')
            QI_temp_file_list = bf.file_filter(l2a_output_path_f + 'QI_' + study_area_f + '\\clipped\\',
                                               containing_word)
            mndwi_threshold = -0.1
            for qi_temp_file in QI_temp_file_list:
                file_information = [qi_temp_file[qi_temp_file.find('202'):qi_temp_file.find('202') + 14]]
                TEMP_QI_DS = gdal.Open(qi_temp_file)
                QI_temp_array = TEMP_QI_DS.GetRasterBand(1).ReadAsArray()
                cloud_pixel_cor = np.argwhere(QI_temp_array >= 7)
                cloud_mask_pixel_cor = np.argwhere(QI_temp_array == 3)
                water_pixel_cor = np.argwhere(QI_temp_array == 6)
                QI_cols = TEMP_QI_DS.RasterXSize
                QI_rows = TEMP_QI_DS.RasterYSize
                temp_file_list = []
                for vi in VI_list_temp:
                    if overwritten_para_cloud_f or not os.path.exists(
                            l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\' + file_information[
                                0] + '_' + vi + '_clipped_cloud_free.tif'):
                        eliminating_all_non_tif_file(l2a_output_path_f + vi + '_' + study_area_f + '\\clipped\\')
                        temp_file_name = bf.file_filter(l2a_output_path_f + vi + '_' + study_area_f + '\\clipped\\',
                                                        file_information)
                        if len(temp_file_name) == 0 or len(temp_file_name) > 1:
                            print('VI File consistency problem occurred')
                            sys.exit(-1)
                        else:
                            temp_file_list.append(temp_file_name[0])
                            TEMP_VI_DS = gdal.Open(temp_file_name[0])
                            VI_temp_array = TEMP_VI_DS.GetRasterBand(1).ReadAsArray()
                            VI_cols = TEMP_VI_DS.RasterXSize
                            VI_rows = TEMP_VI_DS.RasterYSize
                            if VI_rows == QI_rows and VI_cols == QI_cols:
                                print(f'Start process the ' + file_information[0] + vi + ' image')
                                for cor in cloud_pixel_cor:
                                    VI_temp_array[cor[0], cor[1]] = -1
                                    VI_temp_array[cor[0], cor[1]] = -1
                                for cor in cloud_mask_pixel_cor:
                                    VI_temp_array[cor[0], cor[1]] = -1
                                    VI_temp_array[cor[0], cor[1]] = -1
                                for cor in water_pixel_cor:
                                    if VI_temp_array[cor[0], cor[1]] < mndwi_threshold:
                                        mndwi_threshold = VI_temp_array[cor[0], cor[1]]
                                write_raster(TEMP_VI_DS, VI_temp_array,
                                             l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\',
                                             file_information[0] + '_' + vi + '_clipped_cloud_free.tif')
                                print(
                                    f'Successfully removing cloud found in the ' + file_information[0] + vi + ' image')
                            else:
                                print('VI File spatial consistency problem occurred')
                                sys.exit(-1)
            print('Finish removing cloud in all VI image')

        # Create datacube
        if 'data_cube' in specific_name_list_f and 'cloud_free' in specific_name_list_f:
            print('Start creating datacube')
            VI_list_temp = copy.copy(index_list)
            VI_list_temp.remove('QI')
            containing_word = ['_clipped_cloud_free.tif']
            eliminating_all_non_tif_file(l2a_output_path_f + VI_list_temp[0] + '_' + study_area_f + '\\cloud_free\\')
            VI_temp_file_list = bf.file_filter(
                l2a_output_path_f + VI_list_temp[0] + '_' + study_area_f + '\\cloud_free\\',
                containing_word)
            file_amount_temp = len(VI_temp_file_list)
            # Double check file consistency
            for vi in VI_list_temp:
                eliminating_all_non_tif_file(l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\')
                VI_temp_file_list = bf.file_filter(l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\',
                                                   containing_word)
                if len(VI_temp_file_list) != file_amount_temp:
                    print('Some consistency error occurred during the datacube creation')
                    sys.exit(-1)
            # Generate datacube
            for vi in VI_list_temp:
                eliminating_all_non_tif_file(l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\')
                VI_temp_file_list = bf.file_filter(l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\',
                                                   containing_word)
                VI_temp_file_list.sort()
                output_path_temp = l2a_output_path_f + vi + '_' + study_area_f + '\\data_cube\\'
                if (not os.path.exists(output_path_temp + "date_cube.npy") or not os.path.exists(
                        output_path_temp + "data_cube.npy")) or overwritten_para_dc_f:
                    temp_ds = gdal.Open(VI_temp_file_list[0])
                    cols = temp_ds.RasterXSize
                    rows = temp_ds.RasterYSize
                    data_cube_temp = np.zeros((rows, cols, len(VI_temp_file_list)))
                    date_cube_temp = np.zeros((len(VI_temp_file_list)))

                    i = 0
                    while i < len(VI_temp_file_list):
                        date_cube_temp[i] = int(VI_temp_file_list[i][
                                                VI_temp_file_list[i].find('\\20') + 1: VI_temp_file_list[i].find(
                                                    '\\20') + 9])
                        i += 1

                    i = 0
                    while i < len(VI_temp_file_list):
                        temp_ds2 = gdal.Open(VI_temp_file_list[i])
                        data_cube_temp[:, :, i] = temp_ds2.GetRasterBand(1).ReadAsArray()
                        i += 1
                    np.save(output_path_temp + "date_cube.npy", date_cube_temp)
                    if vi == 'QI':
                        np.save(output_path_temp + "data_cube.npy", data_cube_temp)
                    else:
                        np.save(output_path_temp + "data_cube.npy", data_cube_temp.astype(np.float16))
            print('Finish creating datacube')
        else:
            print('Please notice that cloud must be removed before further process')

        # Create sequenced datacube
        if 'sequenced_data_cube' in specific_name_list_f and 'data_cube' in specific_name_list_f:
            print('Start creating sequenced datacube')
            VI_dic = {}
            VI_list_temp = copy.copy(index_list)
            VI_list_temp2 = copy.copy(index_list)
            VI_list_temp.remove('QI')
            VI_list_temp2.remove('QI')
            for vi in VI_list_temp2:
                output_path_temp = l2a_output_path_f + vi + '_' + study_area_f + '\\sequenced_data_cube\\'
                if overwritten_para_sdc or (not os.path.exists(output_path_temp + "doy_list.npy") or not os.path.exists(
                        output_path_temp + "sequenced_data_cube.npy")):
                    vi_data_cube_temp = np.load(
                        l2a_output_path_f + vi + '_' + study_area_f + '\\data_cube\\' + 'data_cube.npy')
                    vi_date_cube_temp = np.load(
                        l2a_output_path_f + vi + '_' + study_area_f + '\\data_cube\\' + 'date_cube.npy')
                    VI_dic[vi] = vi_data_cube_temp
                else:
                    VI_list_temp.remove(vi)
                VI_dic['doy'] = []

            if len(VI_list_temp) != 0:
                if vi_data_cube_temp.shape[2] == vi_date_cube_temp.shape[0]:
                    date_list = []
                    doy_list = []
                    for i in vi_date_cube_temp:
                        date_temp = int(i)
                        if date_temp not in date_list:
                            date_list.append(date_temp)
                    for i in date_list:
                        doy_list.append(datetime.date(int(i // 10000), int((i % 10000) // 100),
                                                      int(i % 100)).timetuple().tm_yday + int(i // 10000) * 1000)
                    if not VI_dic['doy']:
                        VI_dic['doy'] = doy_list
                    elif VI_dic['doy'] != doy_list:
                        print('date error')
                        sys.exit(-1)
                    for vi in VI_list_temp:
                        data_cube_inorder = np.zeros(
                            (vi_data_cube_temp.shape[0], vi_data_cube_temp.shape[1], len(doy_list)), dtype='float16')
                        VI_dic[vi + '_in_order'] = data_cube_inorder
                else:
                    print('datacube has different length with datecube')
                    sys.exit(-1)

                for date_t in date_list:
                    date_all = [z for z, z_temp in enumerate(vi_date_cube_temp) if z_temp == date_t]
                    if len(date_all) == 1:
                        for vi in VI_list_temp:
                            data_cube_temp = VI_dic[vi][:, :, np.where(vi_date_cube_temp == date_t)[0]]
                            data_cube_temp[data_cube_temp == -1] = np.nan
                            data_cube_temp = data_cube_temp.reshape(data_cube_temp.shape[0], -1)
                            VI_dic[vi + '_in_order'][:, :, date_list.index(date_t)] = data_cube_temp
                    elif len(date_all) > 1:
                        for vi in VI_list_temp:
                            if np.where(vi_date_cube_temp == date_t)[0][len(date_all) - 1] - \
                                    np.where(vi_date_cube_temp == date_t)[0][0] + 1 == len(date_all):
                                data_cube_temp = VI_dic[vi][:, :, np.where(vi_date_cube_temp == date_t)[0][0]:
                                                                  np.where(vi_date_cube_temp == date_t)[0][0] + len(
                                                                      date_all)]
                            else:
                                print('date long error')
                                sys.exit(-1)
                            data_cube_temp_factor = copy.copy(data_cube_temp)
                            data_cube_temp_factor[np.isnan(data_cube_temp_factor)] = -1
                            data_cube_temp_factor[data_cube_temp_factor != -1] = 1
                            data_cube_temp_factor[data_cube_temp_factor == -1] = 0
                            data_cube_temp_factor = data_cube_temp_factor.sum(axis=2)
                            data_cube_temp[data_cube_temp == -1] = 0
                            data_cube_temp[np.isnan(data_cube_temp)] = 0
                            data_cube_temp = data_cube_temp.sum(axis=2)
                            data_cube_temp_temp = data_cube_temp / data_cube_temp_factor
                            VI_dic[vi + '_in_order'][:, :, date_list.index(date_t)] = data_cube_temp_temp
                    else:
                        print('Something error during generate sequenced datecube')
                        sys.exit(-1)
                for vi in VI_list_temp:
                    output_path_temp = l2a_output_path_f + vi + '_' + study_area_f + '\\sequenced_data_cube\\'
                    np.save(output_path_temp + "doy_list.npy", VI_dic['doy'])
                    np.save(output_path_temp + "sequenced_data_cube.npy", VI_dic[vi + '_in_order'])
                else:
                    print('The data and date shows inconsistency')
            print('Finish creating sequenced datacube')
        else:
            print('Please notice that datacube must be generated before further process')


def eliminating_all_non_tif_file(file_path_f):
    filter_name = ['.tif']
    tif_file_list = bf.file_filter(file_path_f, filter_name)
    for file in tif_file_list:
        if file[-4:] != '.tif':
            try:
                os.remove(file)
            except:
                print('file cannot be removed')
                sys.exit(-1)


def remove_all_file_and_folder(filter_list):
    for file in filter_list:
        if os.path.isdir(str(file)):
            try:
                shutil.rmtree(file)
            except:
                print('folder cannot be removed')
        elif os.path.isfile(str(file)):
            try:
                os.remove(file)
            except:
                print('file cannot be removed')
        else:
            print(f'{str(file)} has been removed!')


# def s2_resample(temp_S2file):
#     parameters_resample = HashMap()
#     parameters_resample.put('targetResolution', 10)
#     temp_s2file_resample = snappy.GPF.createProduct('Resample', parameters_resample, temp_S2file)
#     temp_width = temp_s2file_resample.getSceneRasterWidth()
#     temp_height = temp_s2file_resample.getSceneRasterHeight()
#     ul_pos = temp_S2file.getSceneGeoCoding().getGeoPos(PixelPos(0, 0), None)
#     ur_pos = temp_S2file.getSceneGeoCoding().getGeoPos(PixelPos(0, temp_S2file.getSceneRasterWidth() - 1), None)
#     lr_pos = temp_S2file.getSceneGeoCoding().getGeoPos(
#         PixelPos(temp_S2file.getSceneRasterHeight() - 1, temp_S2file.getSceneRasterWidth() - 1), None)
#     ll_pos = temp_S2file.getSceneGeoCoding().getGeoPos(PixelPos(temp_S2file.getSceneRasterHeight() - 1, 0), None)
#     print(list(temp_s2file_resample.getBandNames()))
#     return temp_s2file_resample, temp_width, temp_height, ul_pos, ur_pos, lr_pos, ll_pos
#
#
# def s2_reprojection(product, crs):
#     parameters_reprojection = HashMap()
#     parameters_reprojection.put('crs', crs)
#     parameters_reprojection.put('resampling', 'Nearest')
#     product_reprojected = snappy.GPF.createProduct('Reproject', parameters_reprojection, product)
#     # ProductIO.writeProduct(product_reprojected, temp_filename, 'BEAM-DIMAP')
#     return product_reprojected
#
#
# def write_subset_band(temp_s2file_resample, band_name, subset_output_path, file_output_name):
#     parameters_subset_sd = HashMap()
#     parameters_subset_sd.put('sourceBands', band_name)
#     # parameters_subset_sd.put('copyMetadata', True)
#     temp_product_subset = snappy.GPF.createProduct('Subset', parameters_subset_sd, temp_s2file_resample)
#     subset_write_op = WriteOp(temp_product_subset, File(subset_output_path + file_output_name), 'GeoTIFF-BigTIFF')
#     subset_write_op.writeProduct(ProgressMonitor.NULL)
#
#     temp_product_subset.dispose()
#     del temp_product_subset
#     # temp_product_subset = None


def create_NDWI_NDVI_CURVE(NDWI_data_cube, NDVI_data_cube, doy_list, fig_path_f):
    if NDWI_data_cube.shape == NDVI_data_cube.shape and doy_list.shape[0] == NDWI_data_cube.shape[2]:
        start_year = doy_list[0] // 1000
        doy_num = []
        for doy in doy_list:
            doy_num.append((doy % 1000) + 365 * ((doy // 1000) - start_year))
        for y in range(NDVI_data_cube.shape[0] // 16, 9 * NDVI_data_cube.shape[0] // 16):
            for x in range(8 * NDVI_data_cube.shape[1] // 16, NDVI_data_cube.shape[1]):
                NDVI_temp_list = []
                NDWI_temp_list = []
                for z in range(NDVI_data_cube.shape[2]):
                    NDVI_temp_list.append(NDVI_data_cube[y, x, z])
                    NDWI_temp_list.append(NDWI_data_cube[y, x, z])

                plt.xlabel('DOY')
                plt.ylabel('ND*I')
                plt.xlim(xmax=max(doy_num), xmin=0)
                plt.ylim(ymax=1, ymin=-1)
                colors1 = '#006000'
                colors2 = '#87CEFA'
                area = np.pi * 3 ** 2
                plt.scatter(doy_num, NDVI_temp_list, s=area, c=colors1, alpha=0.4, label='NDVI')
                plt.scatter(doy_num, NDWI_temp_list, s=area, c=colors2, alpha=0.4, label='NDWI')
                plt.plot([0, 0.8], [max(doy_num), 0.8], linewidth='1', color='#000000')
                plt.legend()
                plt.savefig(fig_path_f + 'Scatter_plot_' + str(x) + '_' + str(y) + '.png', dpi=300)
                plt.close()
    else:
        print('The data and date shows inconsistency')


def cor_to_pixel(two_corner_coordinate, study_area_example_file_path):
    pixel_limitation_f = {}
    if len(two_corner_coordinate) == 2:
        UL_corner = two_corner_coordinate[0]
        LR_corner = two_corner_coordinate[1]
        if len(UL_corner) == len(LR_corner) == 2:
            upper_limit = UL_corner[1]
            lower_limit = LR_corner[1]
            right_limit = LR_corner[0]
            left_limit = UL_corner[0]
            dataset_temp_list = bf.file_filter(study_area_example_file_path, ['.tif'])
            temp_dataset = gdal.Open(dataset_temp_list[0])
            # TEMP_warp = gdal.Warp(study_area_example_file_path + '\\temp.tif', temp_dataset, dstSRS='EPSG:4326')
            # temp_band = temp_dataset.GetRasterBand(1)
            # temp_cols = temp_dataset.RasterXSize
            # temp_rows = temp_dataset.RasterYSize
            temp_transform = temp_dataset.GetGeoTransform()
            temp_xOrigin = temp_transform[0]
            temp_yOrigin = temp_transform[3]
            temp_pixelWidth = temp_transform[1]
            temp_pixelHeight = -temp_transform[5]
            pixel_limitation_f['x_max'] = max(int((right_limit - temp_xOrigin) / temp_pixelWidth),
                                              int((left_limit - temp_xOrigin) / temp_pixelWidth))
            pixel_limitation_f['y_max'] = max(int((temp_yOrigin - lower_limit) / temp_pixelHeight),
                                              int((temp_yOrigin - upper_limit) / temp_pixelHeight))
            pixel_limitation_f['x_min'] = min(int((right_limit - temp_xOrigin) / temp_pixelWidth),
                                              int((left_limit - temp_xOrigin) / temp_pixelWidth))
            pixel_limitation_f['y_min'] = min(int((temp_yOrigin - lower_limit) / temp_pixelHeight),
                                              int((temp_yOrigin - upper_limit) / temp_pixelHeight))
        else:
            print('Please make sure input all corner pixel with two coordinate in list format')
    else:
        print('Please mention the input coordinate should contain the coordinate of two corner pixel')
    try:
        # TEMP_warp.dispose()
        os.remove(study_area_example_file_path + '\\temp.tif')
    except:
        print('please remove the temp file manually')
    return pixel_limitation_f


def check_vi_file_consistency(l2a_output_path_f, index_list):
    vi_file = []
    c_word = ['.tif']
    r_word = ['.ovr']
    for vi in index_list:
        if not os.path.exists(l2a_output_path_f + vi):
            print(vi + 'folders are missing')
            sys.exit(-1)
        else:
            redundant_file_list = bf.file_filter(l2a_output_path_f + vi + '\\', r_word)
            remove_all_file_and_folder(redundant_file_list)
            tif_file_list = bf.file_filter(l2a_output_path_f + vi + '\\', c_word)
            vi_temp = []
            for tif_file in tif_file_list:
                vi_temp.append(tif_file[tif_file.find('\\20') + 2:tif_file.find('\\20') + 15])
            vi_file.append(vi_temp)
    for i in range(len(vi_file)):
        if not collections.Counter(vi_file[0]) == collections.Counter(vi_file[i]):
            print('VIs did not share the same file numbers')
            sys.exit(-1)


def f_two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x) + b2 * np.sin(2 * w * x)


def curve_fitting(l2a_output_path_f, index_list, study_area_f, pixel_limitation_f, fig_path_f, mndwi_threshold):
    # so, this is the Curve fitting Version 1, Generally it is used to implement two basic functions:
    # (1) Find the inundated pixel by introducing MNDWI with an appropriate threshold and remove it.
    # (2) Using the remaining data to fitting the vegetation growth curve
    # (3) Obtaining vegetation phenology information

    # Check whether the VI data cube exists or not
    VI_dic_sequenced = {}
    VI_dic_curve = {}
    doy_factor = False
    consistency_factor = True
    if 'NDWI' in index_list and os.path.exists(
            l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'sequenced_data_cube.npy') and os.path.exists(
        l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'doy_list.npy'):
        NDWI_sequenced_datacube_temp = np.load(
            l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'sequenced_data_cube.npy')
        NDWI_date_temp = np.load(
            l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'doy_list.npy')
        VI_list_temp = copy.copy(index_list)
        try:
            VI_list_temp.remove('QI')
        except:
            print('QI is not in the VI list')
        VI_list_temp.remove('NDWI')
        for vi in VI_list_temp:
            try:
                VI_dic_sequenced[vi] = np.load(
                    l2a_output_path_f + vi + '_' + study_area_f + '\\sequenced_data_cube\\' + 'sequenced_data_cube.npy')
                if not doy_factor:
                    VI_dic_sequenced['doy'] = np.load(
                        l2a_output_path_f + vi + '_' + study_area_f + '\\sequenced_data_cube\\' + 'doy_list.npy')
                    doy_factor = True
            except:
                print('Please make sure the forward programme has been processed')
                sys.exit(-1)

            if not (NDWI_date_temp == VI_dic_sequenced['doy']).all or not (
                    VI_dic_sequenced[vi].shape[2] == len(NDWI_date_temp)):
                consistency_factor = False
                print('Consistency problem occurred')
                sys.exit(-1)

        VI_dic_curve['VI_list'] = VI_list_temp
        for y in range(pixel_limitation_f['y_min'], pixel_limitation_f['y_max'] + 1):
            for x in range(pixel_limitation_f['x_min'], pixel_limitation_f['x_max'] + 1):
                VIs_temp = np.zeros((len(NDWI_date_temp), len(VI_list_temp) + 2))
                VIs_temp_curve_fitting = np.zeros((len(NDWI_date_temp), len(VI_list_temp) + 1))
                NDWI_threshold_cube = np.zeros(len(NDWI_date_temp))
                VIs_temp[:, 1] = copy.copy(NDWI_sequenced_datacube_temp[y, x, :])
                VIs_temp[:, 0] = ((VI_dic_sequenced['doy'] // 1000) - 2020) * 365 + VI_dic_sequenced['doy'] % 1000
                VIs_temp_curve_fitting[:, 0] = ((VI_dic_sequenced['doy'] // 1000) - 2020) * 365 + VI_dic_sequenced[
                    'doy'] % 1000

                NDWI_threshold_cube = copy.copy(VIs_temp[:, 1])
                NDWI_threshold_cube[NDWI_threshold_cube > mndwi_threshold] = np.nan
                NDWI_threshold_cube[NDWI_threshold_cube < mndwi_threshold] = 1
                NDWI_threshold_cube[np.isnan(NDWI_threshold_cube)] = np.nan

                i = 0
                for vi in VI_list_temp:
                    VIs_temp[:, i + 2] = copy.copy(VI_dic_sequenced[vi][y, x, :])
                    VIs_temp_curve_fitting[:, i + 1] = copy.copy(VI_dic_sequenced[vi][y, x, :]) * NDWI_threshold_cube
                    i += 1

                doy_limitation = np.where(VIs_temp_curve_fitting[:, 0] > 365)
                for i in range(len(doy_limitation)):
                    VIs_temp_curve_fitting = np.delete(VIs_temp_curve_fitting, doy_limitation[i], 0)

                nan_pos = np.where(np.isnan(VIs_temp_curve_fitting[:, 1]))
                for i in range(len(nan_pos)):
                    VIs_temp_curve_fitting = np.delete(VIs_temp_curve_fitting, nan_pos[i], 0)

                nan_pos2 = np.where(np.isnan(VIs_temp[:, 1]))
                for i in range(len(nan_pos2)):
                    VIs_temp = np.delete(VIs_temp, nan_pos2[i], 0)

                i_test = np.argwhere(np.isnan(VIs_temp_curve_fitting))
                if len(i_test) > 0:
                    print('consistency error')
                    sys.exit(-1)

                paras_temp = np.zeros((len(VI_list_temp), 6))

                curve_fitting_para = True
                for i in range(len(VI_list_temp)):
                    if VIs_temp_curve_fitting.shape[0] > 5:
                        paras, extras = curve_fit(f_two_term_fourier, VIs_temp_curve_fitting[:, 0],
                                                  VIs_temp_curve_fitting[:, i + 1], maxfev=5000,
                                                  p0=[0, 0, 0, 0, 0, 0.017], bounds=(
                                [-100, -100, -100, -100, -100, 0.014], [100, 100, 100, 100, 100, 0.020]))
                        paras_temp[i, :] = paras
                    else:
                        curve_fitting_para = False
                VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting_paras'] = paras_temp
                VI_dic_curve[str(y) + '_' + str(x) + 'ori'] = VIs_temp
                VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting'] = VIs_temp_curve_fitting

                x_temp = np.linspace(0, 365, 10000)
                # 'QI', 'NDVI', 'NDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI', 'NDVI_RE', 'NDVI_2', 'NDVI_RE2'
                colors = {'colors_NDVI': '#00CD00', 'colors_NDVI_2': '#00EE00', 'colors_NDVI_RE': '#CDBE70',
                          'colors_NDVI_RE2': '#CDC673', 'colors_GNDVI': '#7D26CD', 'colors_NDWI': '#0000FF',
                          'colors_EVI': '#FFFF00', 'colors_EVI2': '#FFD700', 'colors_OSAVI': '#FF3030'}
                markers = {'markers_NDVI': 'o', 'markers_NDWI': 's', 'markers_EVI': '^', 'markers_EVI2': 'v',
                           'markers_OSAVI': 'p', 'markers_NDVI_2': 'D', 'markers_NDVI_RE': 'x', 'markers_NDVI_RE2': 'X',
                           'markers_GNDVI': 'd'}
                plt.rcParams["font.family"] = "Times New Roman"
                plt.figure(figsize=(10, 6))
                ax = plt.axes((0.1, 0.1, 0.9, 0.8))
                plt.xlabel('DOY')
                plt.ylabel('ND*I')
                plt.xlim(xmax=max(((VI_dic_sequenced['doy'] // 1000) - 2020) * 365 + VI_dic_sequenced['doy'] % 1000),
                         xmin=1)
                plt.ylim(ymax=1, ymin=-1)
                ax.tick_params(axis='x', which='major', labelsize=15)
                plt.xticks([15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351, 380, 409, 440, 470, 501, 532],
                           ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan',
                            'Feb', 'Mar', 'Apr', 'May', 'Jun'])
                plt.plot(np.linspace(365, 365, 1000), np.linspace(-1, 1, 1000), linestyle='--', color=[0.5, 0.5, 0.5])
                area = np.pi * 3 ** 2

                plt.scatter(VIs_temp[:, 0], VIs_temp[:, 1], s=area, c=colors['colors_NDWI'], alpha=1, label='NDWI')
                for i in range(len(VI_list_temp)):
                    plt.scatter(VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting'][:, 0],
                                VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting'][:, i + 1], s=area,
                                c=colors['colors_' + VI_list_temp[i]], alpha=1, norm=0.8, label=VI_list_temp[i],
                                marker=markers['markers_' + VI_list_temp[i]])
                    # plt.show()
                    if curve_fitting_para:
                        a0_temp, a1_temp, b1_temp, a2_temp, b2_temp, w_temp = VI_dic_curve[str(y) + '_' + str(
                            x) + 'curve_fitting_paras'][i, :]
                        plt.plot(x_temp,
                                 f_two_term_fourier(x_temp, a0_temp, a1_temp, b1_temp, a2_temp, b2_temp, w_temp),
                                 linewidth='1.5', color=colors['colors_' + VI_list_temp[i]])
                plt.legend()
                plt.savefig(fig_path_f + 'Scatter_plot_' + str(x) + '_' + str(y) + '.png', dpi=300)
                plt.close()
                print('Finish plotting Figure ' + str(x) + '_' + str(y))
        np.save(fig_path_f + 'fig_data.npy', VI_dic_curve)
    else:
        print('Please notice that NDWI is essential for inundated pixel removal')
        sys.exit(-1)


if __name__ == '__main__':

    # Test
    filepath = 'E:\\Z_Phd_Other_stuff\\2022_08_09_Map\\Sentinel_2\\Original_files\\'
    s2_ds_temp = Sentinel2_ds(filepath)
    s2_ds_temp.construct_metadata()
    s2_ds_temp.mp_subset(['all_band', 'MNDWI'])

    # 4Main
    # filepath = 'G:\\Sample_Sentinel2\\Original_file\\'
    # s2_ds_temp = Sentinel2_ds(filepath)
    # s2_ds_temp.construct_metadata()
    # s2_ds_temp.mp_subset(['all_band', 'NDVI', 'MNDWI', 'OSAVI'], ROI='E:\\A_Veg_phase2\\Sentinel_2_test\\shpfile\\Floodplain_2020.shp',
    #                      ROI_name='MYZR_FP_2020', cloud_removal_strategy='QI_all_cloud', size_control_factor=True, large_roi=True, dst_coord='EPSG:32649')
    # s2_ds_temp.mp_merge(['NDVI', 'MNDWI', 'OSAVI'], queried_from_parafile=True)

    # 4TEST
    filepath = 'E:\A_Veg_phase2\Sentinel_2_test\Original_file\\'
    s2_ds_temp = Sentinel2_ds(filepath)
    s2_ds_temp.construct_metadata()
    s2_ds_temp.sequenced_subset(['all_band', 'NDVI', 'OSAVI'], ROI='E:\\A_Veg_phase2\\Sentinel_2_test\\shpfile\\Floodplain_2020.shp',
                                 ROI_name='MYZR_FP_2020', cloud_removal_strategy='QI_all_cloud',
                                size_control_factor=True, large_roi=True, dst_coord='EPSG:32649')
    s2_ds_temp.mp_merge(['NDVI'], queried_from_parafile=True)


    file_path = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\Original_Zipfile\\'
    output_path = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\'
    l2a_output_path = output_path + 'Sentinel2_L2A_output\\'
    QI_output_path = output_path + 'Sentinel2_L2A_output\\QI\\'
    bf.create_folder(l2a_output_path)
    bf.create_folder(QI_output_path)

    # Code built-in parameters Configuration
    overwritten_para_vis = False
    overwritten_para_clipped = False
    overwritten_para_cloud = True
    overwritten_para_datacube = True
    overwritten_para_sequenced_datacube = True

    # Generate VIs in GEOtiff format
    i = 0
    VI_list = ['NDVI', 'NDWI']
    # metadata_size = thalweg_temp.S2_metadata.shape[0]
    # while i < metadata_size:
    #     generate_vi_file(VI_list, i, l2a_output_path, metadata_size, overwritten_para_vis, thalweg_temp.S2_metadata)
    #     try:
    #         cache_output_path = 'C:\\Users\\sx199\\.snap\\var\\cache\\s2tbx\\l2a-reader\\8.0.7\\'
    #         cache_path = [cache_output_path + temp for temp in os.listdir(cache_output_path)]
    #         remove_all_file_and_folder(cache_path)
    #     except:
    #         print('process occupied')
    #     i += 1

    # # this allows GDAL to throw Python Exceptions
    # gdal.UseExceptions()
    # mask_path = 'E:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\Arcmap\\shp\\Huxianzhou.shp'
    # # Check VI file consistency
    # check_vi_file_consistency(l2a_output_path, VI_list)
    # study_area = mask_path[mask_path.find('\\shp\\') + 5: mask_path.find('.shp')]
    # specific_name_list = ['clipped', 'cloud_free', 'data_cube', 'sequenced_data_cube']
    # # Process files
    # VI_list = ['NDVI', 'NDWI']
    # vi_process(l2a_output_path, VI_list, study_area, specific_name_list, overwritten_para_clipped,
    #            overwritten_para_cloud, overwritten_para_datacube, overwritten_para_sequenced_datacube)

    # Inundated detection
    # Spectral unmixing
    # Curve fitting
    # mndwi_threshold = -0.15
    # fig_path = l2a_output_path + 'Fig\\'
    # pixel_limitation = cor_to_pixel([[778602.523, 3322698.324], [782466.937, 3325489.535]],
    #                                 l2a_output_path + 'NDVI_' + study_area + '\\cloud_free\\')
    # curve_fitting(l2a_output_path, VI_list, study_area, pixel_limitation, fig_path, mndwi_threshold)
    # Generate Figure
    # NDWI_DATA_CUBE = np.load(NDWI_data_cube_path + 'data_cube_inorder.npy')
    # NDVI_DATA_CUBE = np.load(NDVI_data_cube_path + 'data_cube_inorder.npy')
    # DOY_LIST = np.load(NDVI_data_cube_path + 'doy_list.npy')
    # fig_path = output_path + 'Sentinel2_L2A_output\\Fig\\'
    # create_folder(fig_path)
    # create_NDWI_NDVI_CURVE(NDWI_DATA_CUBE, NDVI_DATA_CUBE, DOY_LIST, fig_path)
