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
from Basic_function import Path
import Basic_function as bf
from functools import wraps
import concurrent.futures
from itertools import repeat
from zipfile import ZipFile

# global self.S2_metadata, mndwi_threshold, VI_dic

# Input Snappy data style
np.seterr(divide='ignore', invalid='ignore')


def log_para(func):
    def wrapper(*args, **kwargs):
        pass

    return wrapper


def write_raster(ori_ds, new_array, file_path_f, file_name_f, raster_datatype=None, nodatavalue=None):
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


def union_list(small_list, big_list):
    union_list_temp = []
    if type(small_list) != list or type(big_list) != list:
        print('Please input valid lists')
        sys.exit(-1)

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
        print('Cloud removal strategy is not supported!')
        sys.exit(-1)

    if 'sparse_matrix_factor' in kwargs.keys():
        sparse_matrix_factor = kwargs['sparse_matrix_factor']
    else:
        sparse_matrix_factor = False

    # process cloud removal and clipping sequence
    if 'cloud_clip_priority' in kwargs.keys():
        if kwargs['cloud_clip_priority'] == 'cloud':
            cloud_clip_seq = True
        elif kwargs['cloud_clip_priority'] == 'clip':
            cloud_clip_seq = False
        else:
            cloud_clip_seq = False
            print('Cloud clip sequence para need to input the specific process!')
            sys.exit(-1)
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

    if sparse_matrix_factor and not cloud_clip_seq:
        qi_array = sp.lil_matrix(qi_ds.GetRasterBand(1).ReadAsArray())
        processed_array = sp.lil_matrix(processed_ds.GetRasterBand(1).ReadAsArray())
    else:
        qi_array = qi_ds.GetRasterBand(1).ReadAsArray()
        processed_array = processed_ds.GetRasterBand(1).ReadAsArray()

    if qi_array.shape[0] != processed_array.shape[0] or qi_array.shape[1] != processed_array.shape[1]:
        print('Consistency error')
        sys.exit(-1)

    for indicator in cloud_indicator:
        qi_array[qi_array == indicator] = 64
    processed_array[qi_array == 64] = dst_nodata

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
    if sparse_matrix_factor and not cloud_clip_seq:
        processed_ds.GetRasterBand(1).WriteArray(processed_array.toarray())
    else:
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
        self.S2_metadata_size = np.nan
        self.date_list = []

        # Define key variables (kwargs)
        self.size_control_factor = False
        self.cloud_removal_para = False
        self.vi_clip_factor = False
        self.sparsify_matrix_factor = False
        self.cloud_clip_seq = None

        # Remove all the duplicated data
        dup_data = bf.file_filter(self.ori_folder, ['.1.zip'])
        for dup in dup_data:
            os.remove(dup)

        # Generate the original zip file list
        self.orifile_list = bf.file_filter(self.ori_folder, ['.zip', 'S2'], and_or_factor='and',
                                           subfolder_detection=True)
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

        # Create output path
        self.output_path = f'{self.work_env}Sentinel2_L2A_Output\\'
        self.shpfile_path = f'{self.work_env}shpfile\\'
        self.log_file = f'{self.work_env}logfile\\'
        bf.create_folder(self.output_path)
        bf.create_folder(self.log_file)
        bf.create_folder(self.shpfile_path)

        # Constant
        self.band_name_list = ['B01_60m.jp2', 'B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B05_20m.jp2', 'B06_20m.jp2',
                               'B07_20m.jp2', 'B8A_20m.jp2', 'B09_60m.jp2', 'B11_20m.jp2', 'B12_20m.jp2']
        self.band_output_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B11', 'B12']
        self.all_supported_index_list = ['QI', 'all_band', '4visual', 'NDVI', 'MNDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI',
                                         'NDVI_RE', 'NDVI_RE2', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',
                                         'B11', 'B12']

    # def save_log_file(func):
    #
    #     def wrapper(self, *args, **kwargs):
    #         if os.path.exists(self.log_file)
    #         log_file = fsave
    #         func(*args, **kwargs)
    #         consuming_time = time.time()-time1
    #
    #
    #     return wrapper

    # @save_log_file
    def construct_metadata(self):
        print('---------------------------- Start the construction of Metadata ----------------------------')
        start_temp = time.time()
        # Input the file and generate the metadata
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
                    shutil.move(ori_file, self.work_env + 'Corrupted_S2_file\\' + file_name)

            Corrupted_metadata = pd.DataFrame(
                {'Corrupted_file_name': corrupted_ori_file, 'File_Date': corrupted_file_date})
            if not os.path.exists(self.work_env + 'Corrupted_metadata.xlsx'):
                Corrupted_metadata.to_excel(self.work_env + 'Corrupted_metadata.xlsx')
            else:
                Corrupted_metadata_old_version = pd.read_excel(self.work_env + 'Corrupted_metadata.xlsx')
                Corrupted_metadata_old_version.append(Corrupted_metadata, ignore_index=True)
                Corrupted_metadata_old_version.drop_duplicates()
                Corrupted_metadata_old_version.to_excel(self.work_env + 'Corrupted_metadata.xlsx')

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
        self.date_list = self.S2_metadata['Sensing_Date'].drop_duplicates().to_list()
        print(f'Finish in {str(time.time() - start_temp)} sec!')
        print('----------------------------  End the construction of Metadata  ----------------------------')

    def check_metadata_availability(self):

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
            print(f'Band {band_name} is not valid!')
            sys.exit(-1)

        if self.vi_clip_factor:
            output_path = f'{self.output_path}{self.ROI_name}_all_band\\'
        else:
            output_path = f'{self.output_path}_all_band\\'
        bf.create_folder(output_path)

        # Detect whether the required band was generated before
        try:
            if False in [os.path.exists(f'{output_path}{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_{band_temp}.tif') for band_temp in band_name]:
                self.subset_tiffiles(band_name, tiffile_serial_num, **kwargs)

            # Return output
            if False in [os.path.exists(f'{output_path}{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_{band_temp}.tif') for band_temp in band_name]:
                print(f'Something error PROCESSIMG {band_name}!')
                return None
            else:
                return [gdal.Open(f'{output_path}{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_{band_temp}.tif') for band_temp in band_name]

        except:
            return None

    def mp_subset(self, *args, **kwargs):
        if self.S2_metadata is None:
            print('Please construct the S2_metadata before the subset!')
            sys.exit(-1)
        i = range(self.S2_metadata.shape[0])
        # mp process
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.subset_tiffiles, repeat(args[0]), i, repeat(False), repeat(kwargs))

    def sequenced_subset(self, *args, **kwargs):
        if self.S2_metadata is None:
            print('Please construct the S2_metadata before the subset!')
            sys.exit(-1)
        # sequenced process
        for i in range(self.S2_metadata.shape[0]):
            self.subset_tiffiles(args[0], i, **kwargs)

    def subset_indicator_process(self, **kwargs):
        # Detect whether all the indicator are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ['ROI', 'ROI_name', 'size_control_factor', 'cloud_removal_strategy',
                                       'sparsify_matrix_factor', 'cloud_clip_priority']:
                print(f'{kwarg_indicator} is not supported kwargs! Please double check!')

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
        if 'sparsify_matrix_factor' in kwargs.keys():
            self.sparsify_matrix_factor = True
        else:
            self.sparsify_matrix_factor = False

        # process cloud removal and clipping sequence
        if 'cloud_clip_priority' in kwargs.keys():
            if kwargs['cloud_clip_priority'] == 'cloud':
                self.cloud_clip_seq = True
            elif kwargs['cloud_clip_priority'] == 'clip':
                self.cloud_clip_seq = False
            else:
                self.cloud_clip_seq = False
                print('Cloud clip sequence para need to input the specific process!')
                sys.exit(-1)
        else:
            self.cloud_clip_seq = False

    def generate_10m_output_bounds(self, tiffile_serial_num, **kwargs):
        # determine the subset indicator
        self.subset_indicator_process(**kwargs)
        self.check_metadata_availability()

        time1 = time.time()
        # Define local var
        topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])
        sensing_date = 'Sensing_Date'
        tile_num = 'Tile_Num'
        VI = 'all_band'

        # Define the output path
        if self.vi_clip_factor:
            output_path = f'{self.output_path}{self.ROI_name}_{VI}\\'
        else:
            output_path = f'{self.output_path}{VI}\\'
        bf.create_folder(output_path)
        print(f' Generate 10m bounds define variable consume {time.time()-time1}s')

        # Create the output bounds based on the 10-m Band2 images
        if self.output_bounds.shape[0] > tiffile_serial_num:
            if True in np.isnan(self.output_bounds[tiffile_serial_num, :]):
                temp_S2file_path = self.S2_metadata.iat[tiffile_serial_num, 1]
                zfile = ZipFile(temp_S2file_path, 'r')
                b2_band_file_name = f'{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_B2'
                if not os.path.exists(output_path + b2_band_file_name + '.tif'):
                    b2_file = [zfile_temp for zfile_temp in zfile.namelist() if 'B02_10m.jp2' in zfile_temp]
                    if len(b2_file) != 1:
                        print(
                            f'Data issue for the B2 file of all_cloud data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata.shape[0])})')
                        self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                        return
                    else:
                        ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, b2_file[0]))
                        if self.vi_clip_factor:
                            gdal.Warp('/vsimem/' + b2_band_file_name + '.tif', ds_temp, dstSRS='EPSG:32649', xRes=10,
                                      yRes=10, cutlineDSName=self.ROI, cropToCutline=True, outputType=gdal.GDT_UInt16,
                                      dstNodata=0)
                        else:
                            gdal.Warp('/vsimem/' + b2_band_file_name + '.tif', ds_temp, dstSRS='EPSG:32649', xRes=10,
                                      yRes=10, outputType=gdal.GDT_UInt16, dstNodata=0)
                        gdal.Translate(output_path + b2_band_file_name + '.tif',
                                       '/vsimem/' + b2_band_file_name + '.tif', options=topts, noData=0)
                        gdal.Unlink('/vsimem/' + b2_band_file_name + '.tif')
                ds4bounds = gdal.Open(output_path + b2_band_file_name + '.tif')
                ulx, xres, xskew, uly, yskew, yres = ds4bounds.GetGeoTransform()
                self.output_bounds[tiffile_serial_num, :] = np.array(
                    [ulx, uly + yres * ds4bounds.RasterYSize, ulx + xres * ds4bounds.RasterXSize, uly])
                ds4bounds = None

            if self.cloud_clip_seq and True in np.isnan(self.raw_10m_bounds[tiffile_serial_num, :]):
                temp_S2file_path = self.S2_metadata.iat[tiffile_serial_num, 1]
                zfile = ZipFile(temp_S2file_path, 'r')
                b2_file = [zfile_temp for zfile_temp in zfile.namelist() if 'B02_10m.jp2' in zfile_temp]
                if len(b2_file) != 1:
                    print(
                        f'Data issue for the B2 file of all_cloud data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata.shape[0])})')
                    self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                    return
                else:
                    ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, b2_file[0]))
                    if self.cloud_clip_seq:
                        ulx_temp, xres_temp, xskew_temp, uly_temp, yskew_temp, yres_temp = ds_temp.GetGeoTransform()
                        self.raw_10m_bounds[tiffile_serial_num, :] = np.array(
                            [ulx_temp, uly_temp + yres_temp * ds_temp.RasterYSize,
                             ulx_temp + xres_temp * ds_temp.RasterXSize, uly_temp])
        else:
            print('The output bounds has some logical issue!')
            sys.exit(-1)

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
        processed_index_list = union_list(processed_index_list, self.all_supported_index_list)

        # Retrieve kwargs from args using the mp
        if args != () and type(args[0]) == dict:
            kwargs = copy.copy(args[0])

        # Process subset indicator
        self.generate_10m_output_bounds(tiffile_serial_num, **kwargs)

        if processed_index_list != []:
            temp_S2file_path = self.S2_metadata.iat[tiffile_serial_num, 1]
            zfile = ZipFile(temp_S2file_path, 'r')
            output_limit = (
                int(self.output_bounds[tiffile_serial_num, 0]), int(self.output_bounds[tiffile_serial_num, 1]),
                int(self.output_bounds[tiffile_serial_num, 2]), int(self.output_bounds[tiffile_serial_num, 3]))
            if self.cloud_clip_seq:
                raw_10m_bound = (
                int(self.raw_10m_bounds[tiffile_serial_num, 0]), int(self.raw_10m_bounds[tiffile_serial_num, 1]),
                int(self.raw_10m_bounds[tiffile_serial_num, 2]), int(self.raw_10m_bounds[tiffile_serial_num, 3]))
            for VI in processed_index_list:
                start_temp = time.time()
                print(f'Start processing {VI} data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')
                sensing_date = 'Sensing_Date'
                tile_num = 'Tile_Num'

                # Generate output folder
                if self.vi_clip_factor:
                    subset_output_path = f'{self.output_path}{self.ROI_name}_{VI}\\'
                    if VI in self.band_output_list:
                        subset_output_path = f'{self.output_path}{self.ROI_name}_all_band\\'
                else:
                    subset_output_path = f'{self.output_path}{VI}\\'
                    if VI in self.band_output_list:
                        subset_output_path = f'{self.output_path}_all_band\\'

                # Generate qi output folder
                if self.cloud_clip_seq or not self.vi_clip_factor:
                    qi_path = f'{self.output_path}QI\\'
                else:
                    qi_path = f'{self.output_path}{self.ROI_name}_QI\\'

                bf.create_folder(subset_output_path)
                bf.create_folder(qi_path)

                # Define the file name for VI
                file_name = f'{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_{VI}'

                # Generate QI layer
                if (VI == 'QI' and overwritten_para) or (
                        VI == 'QI' and not overwritten_para and not os.path.exists(qi_path + file_name)):
                    band_all = [zfile_temp for zfile_temp in zfile.namelist() if 'SCL_20m.jp2' in zfile_temp]
                    if len(band_all) != 1:
                        print(
                            f'Something error during processing {VI} data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')
                        self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                    else:
                        for band_temp in band_all:
                            ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, band_temp))
                            if self.cloud_clip_seq or not self.vi_clip_factor:
                                gdal.Warp('/vsimem/' + file_name + '.tif', ds_temp, xRes=10, yRes=10, outputType=gdal.GDT_UInt16, dstNodata=0, outputBounds=raw_10m_bound)
                            else:
                                gdal.Warp('/vsimem/' + file_name + '.tif', ds_temp, xRes=10, yRes=10,
                                          dstSRS='EPSG:32649', outputBounds=output_limit, cutlineDSName=self.ROI,
                                          cropToCutline=True, outputType=gdal.GDT_UInt16, dstNodata=0)
                            gdal.Translate(qi_path + file_name + '.tif', '/vsimem/' + file_name + '.tif', options=topts,
                                           noData=0, outputType=gdal.GDT_UInt16)
                            gdal.Unlink('/vsimem/' + file_name + '.tif')

                # Subset band images
                elif VI == 'all_band' or VI == '4visual' or VI in self.band_output_list:
                    # Check the output band
                    if VI == 'all_band':
                        band_name_list, band_output_list = self.band_name_list, self.band_output_list
                    elif VI == '4visual':
                        band_name_list, band_output_list = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B05_20m.jp2',
                                                            'B8A_20m.jp2', 'B11_20m.jp2'], ['B2', 'B3', 'B4', 'B5',
                                                                                            'B8', 'B11']
                    elif VI in self.band_output_list:
                        band_name_list, band_output_list = [self.band_name_list[self.band_output_list.index(VI)]], [VI]
                    else:
                        print('Code error!')
                        sys.exit(-1)

                    if overwritten_para or False in [os.path.exists(
                            subset_output_path + str(self.S2_metadata[sensing_date][tiffile_serial_num]) + '_' + str(
                                self.S2_metadata[tile_num][tiffile_serial_num]) + '_' + str(band_temp) + '.tif') for
                        band_temp in band_output_list]:
                        for band_name, band_output in zip(band_name_list, band_output_list):
                            if band_output != 'B2' or self.cloud_clip_seq:
                                all_band_file_name = f'{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_{str(band_output)}'
                                if not os.path.exists(all_band_file_name + '.tif') or overwritten_para:
                                    band_all = [zfile_temp for zfile_temp in zfile.namelist() if
                                                band_name in zfile_temp]
                                    if len(band_all) != 1:
                                        print(
                                            f'Something error during processing {band_output} of {VI} data ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)})')
                                        self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                                    else:
                                        for band_temp in band_all:
                                            if not self.cloud_clip_seq:
                                                t1 = time.time()
                                                ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, band_temp))
                                                time1 = time.time() - t1
                                                t2 = time.time()
                                                if self.vi_clip_factor:
                                                    gdal.Warp('/vsimem/' + all_band_file_name + '.tif', ds_temp,
                                                              xRes=10, yRes=10, dstSRS='EPSG:32649',
                                                              cutlineDSName=self.ROI, cropToCutline=True,
                                                              outputBounds=output_limit, outputType=gdal.GDT_UInt16,
                                                              dstNodata=0)
                                                else:
                                                    gdal.Warp('/vsimem/' + all_band_file_name + '.tif', ds_temp,
                                                              xRes=10, yRes=10, dstSRS='EPSG:32649',
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
                                                                    sparse_matrix_factor=self.sparsify_matrix_factor,
                                                                    **kwargs)
                                                time3 = time.time() - t3
                                                t4 = time.time()
                                                gdal.Translate(subset_output_path + all_band_file_name + '.tif',
                                                               '/vsimem/' + all_band_file_name + '.tif', options=topts,
                                                               noData=0)
                                                gdal.Unlink('/vsimem/' + all_band_file_name + '.tif')
                                                time4 = time.time() - t4
                                                print(time1, time2, time3, time4)
                                            else:
                                                t1 = time.time()
                                                if self.cloud_removal_para:
                                                    qi_file_path = f'{qi_path}{str(self.S2_metadata[sensing_date][tiffile_serial_num])}_{str(self.S2_metadata[tile_num][tiffile_serial_num])}_QI.tif'
                                                    if not os.path.exists(qi_file_path):
                                                        self.subset_tiffiles(['QI'], tiffile_serial_num, **kwargs)
                                                    gdal.Warp('/vsimem/' + all_band_file_name + '_temp.tif',
                                                              '/vsizip/%s/%s' % (temp_S2file_path, band_temp), xRes=10,
                                                              yRes=10, outputBounds=raw_10m_bound)
                                                    qi_remove_cloud('/vsimem/' + all_band_file_name + '_temp.tif',
                                                                    qi_file_path, dst_nodata=0,
                                                                    sparse_matrix_factor=self.sparsify_matrix_factor,
                                                                    **kwargs)
                                                    ds_temp = gdal.Open('/vsimem/' + all_band_file_name + '_temp.tif')
                                                else:
                                                    ds_temp = gdal.Open('/vsizip/%s/%s' % (temp_S2file_path, band_temp))
                                                time1 += time.time() - t1
                                                t2 = time.time()
                                                if self.vi_clip_factor:
                                                    gdal.Warp('/vsimem/' + all_band_file_name + '.tif', ds_temp,
                                                              xRes=10, yRes=10,
                                                              cutlineDSName=self.ROI, cropToCutline=True,
                                                              outputBounds=output_limit,
                                                              dstNodata=0)
                                                else:
                                                    gdal.Warp('/vsimem/' + all_band_file_name + '.tif', ds_temp,
                                                              xRes=10, yRes=10,
                                                              outputBounds=output_limit,
                                                              dstNodata=0)
                                                time2 += time.time() - t2
                                                t3 = time.time()
                                                gdal.Translate(subset_output_path + all_band_file_name + '.tif',
                                                               '/vsimem/' + all_band_file_name + '.tif', options=topts,
                                                               noData=0, dstSRS='EPSG:32649',)
                                                gdal.Unlink('/vsimem/' + all_band_file_name + '.tif')
                                                time3 += time.time() - t3
                                                print(time1, time2, time3)
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
                                            qi_file_path, dst_nodata=0,
                                            sparse_matrix_factor=self.sparsify_matrix_factor, **kwargs)

                elif not overwritten_para and not os.path.exists(subset_output_path + file_name + 'tif') and not (
                        VI == 'QI' or VI == 'all_band' or VI == '4visual' or VI in self.band_output_list):
                    if VI == 'NDVI':
                        time1 = time.time()
                        ds_list = self.check_output_band_statue(['B8', 'B4'], tiffile_serial_num, **kwargs)
                        print('process b8 and b4' + str(time.time() - time1))
                        if ds_list is not None:
                            time1 = time.time()
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(ds_list[0].GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B4_array = sp.csr_matrix(ds_list[1].GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = ds_list[0].GetRasterBand(1).ReadAsArray().astype(np.float)
                                B4_array = ds_list[1].GetRasterBand(1).ReadAsArray().astype(np.float)
                            # print(time.time()-time1)
                            output_array = (B8_array - B4_array) / (B8_array + B4_array)
                            B4_array = None
                            B8_array = None
                            # print(time.time()-time1)
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'MNDWI':
                        B11_statue, a_ds, B11_filepath = self.check_output_band_statue('B11', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        B3_statue, b_ds, B3_filepath = self.check_output_band_statue('B3', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        if B11_statue is not None and B3_statue is not None:
                            if self.sparsify_matrix_factor:
                                B3_array = sp.csr_matrix(a_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B11_array = sp.csr_matrix(b_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B3_array = a_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                                B11_array = b_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                            output_array = (B3_array - B11_array) / (B3_array + B11_array)
                            B3_array = None
                            B11_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'EVI':
                        B2_statue, a_ds, B2_filepath = self.check_output_band_statue('B2', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        B4_statue, b_ds, B4_filepath = self.check_output_band_statue('B4', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        B8_statue, c_ds, B8_filepath = self.check_output_band_statue('B8', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        if B2_statue is not None and B4_statue is not None and B8_statue is not None:
                            if self.sparsify_matrix_factor:
                                B2_array = sp.csr_matrix(a_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B4_array = sp.csr_matrix(b_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B8_array = sp.csr_matrix(c_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B2_array = a_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                                B4_array = b_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                                B8_array = c_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                            output_array = 2.5 * (B8_array - B4_array) / (B8_array + 6 * B4_array - 7.5 * B2_array + 1)
                            B4_array = None
                            B8_array = None
                            B2_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'EVI2':
                        B8_statue, a_ds, B8_filepath = self.check_output_band_statue('B8', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        B4_statue, b_ds, B4_filepath = self.check_output_band_statue('B4', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        if B8_statue is not None and B4_statue is not None:
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(a_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B4_array = sp.csr_matrix(b_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = a_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                                B4_array = b_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                            output_array = 2.5 * (B8_array - B4_array) / (B8_array + 2.4 * B4_array + 1)
                            B4_array = None
                            B8_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'GNDVI':
                        B8_statue, a_ds, B8_filepath = self.check_output_band_statue('B8', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        B3_statue, b_ds, B3_filepath = self.check_output_band_statue('B3', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        if B8_statue is not None and B3_statue is not None:
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(a_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B3_array = sp.csr_matrix(b_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = a_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                                B3_array = b_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                            output_array = (B8_array - B3_array) / (B8_array + B3_array)
                            B3_array = None
                            B8_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'NDVI_RE':
                        B7_statue, a_ds, B7_filepath = self.check_output_band_statue('B7', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        B5_statue, b_ds, B5_filepath = self.check_output_band_statue('B5', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        if B7_statue is not None and B5_statue is not None:
                            if self.sparsify_matrix_factor:
                                B7_array = sp.csr_matrix(a_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B5_array = sp.csr_matrix(b_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B7_array = a_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                                B5_array = b_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                            output_array = (B7_array - B5_array) / (B7_array + B5_array)
                            B5_array = None
                            B7_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'NDVI_RE2':
                        B8_statue, a_ds, B8_filepath = self.check_output_band_statue('B8', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        B5_statue, b_ds, B5_filepath = self.check_output_band_statue('B5', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        if B8_statue is not None and B5_statue is not None:
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(a_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B5_array = sp.csr_matrix(b_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = a_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                                B5_array = b_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                            output_array = (B8_array - B5_array) / (B8_array + B5_array)
                            B5_array = None
                            B8_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'OSAVI':
                        B8_statue, a_ds, B8_filepath = self.check_output_band_statue('B8', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        B4_statue, b_ds, B4_filepath = self.check_output_band_statue('B4', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        if B8_statue is not None and B4_statue is not None:
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(a_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B4_array = sp.csr_matrix(b_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = a_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                                B4_array = b_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                            output_array = 1.16 * (B8_array - B4_array) / (B8_array + B4_array + 0.16)
                            B4_array = None
                            B8_array = None
                        else:
                            self.subset_failure_file.append([VI, tiffile_serial_num, temp_S2file_path])
                            break
                    elif VI == 'IEI':
                        B8_statue, a_ds, B8_filepath = self.check_output_band_statue('B8', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        B4_statue, b_ds, B4_filepath = self.check_output_band_statue('B4', str(
                            self.S2_metadata[sensing_date][tiffile_serial_num]), str(
                            self.S2_metadata[tile_num][tiffile_serial_num]), zfile, temp_S2file_path)
                        if B8_statue is not None and B4_statue is not None:
                            if self.sparsify_matrix_factor:
                                B8_array = sp.csr_matrix(a_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                                B4_array = sp.csr_matrix(b_ds.GetRasterBand(1).ReadAsArray()).astype(np.float)
                            else:
                                B8_array = a_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                                B4_array = b_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
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
                    print(f'write NDVI consume about {time.time() - time1}')
                print(
                    f'Finish processing {VI} data in {str(time.time() - start_temp)}s ({str(tiffile_serial_num + 1)} of {str(self.S2_metadata_size)} )')
        else:
            print(
                'Caution! the input variable VI_list should be a list and make sure all of them are in Capital Letter')
            sys.exit(-1)
        return

    def check_subset_intergrality(self, indicator, **kwargs):
        if self.ROI_name is None and ('ROI' not in kwargs.keys() and 'ROI_name' not in kwargs.keys()):
            check_path = f'{self.output_path}{str(indicator)}\\'
        elif self.ROI_name is None and ('ROI' in kwargs.keys() or 'ROI_name' in kwargs.keys()):
            self.subset_indicator_process(**kwargs)
            if self.ROI_name is None:
                print()

    def temporal_mosaic(self, indicator, date, **kwargs):
        self.check_metadata_availability()
        self.check_subset_intergrality(indicator)

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
    # Create Output folder
    # filepath = 'E:\A_Veg_phase2\Sentinel_2_test\Original_file'
    filepath = 'G:\Sample_Sentinel2\Original_file\\'
    s2_ds_temp = Sentinel2_ds(filepath)
    s2_ds_temp.construct_metadata()
    # s2_ds_temp.subset_tiffiles(['all_band', 'NDVI'],0)
    # s2_ds_temp.sequenced_subset(['all_band'], ROI='E:\\A_Veg_phase2\\Sentinel_2_test\\shpfile\\Floodplain_2020.shp',
    #                             ROI_name='MYZR_FP_2020', cloud_removal_strategy='QI_all_cloud',
    #                             sparsify_matrix_factor=True, size_control_factor=True, cloud_clip_priority='clip')
    s2_ds_temp.mp_subset(['all_band'], ROI='E:\\A_Veg_phase2\\Sentinel_2_test\\shpfile\\Floodplain_2020.shp', ROI_name='MYZR_FP_2020', cloud_removal_strategy='QI_all_cloud', sparsify_matrix_factor=True, size_control_factor=True, cloud_clip_priority='clip')
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
    # metadata_size = self.S2_metadata.shape[0]
    # while i < metadata_size:
    #     generate_vi_file(VI_list, i, l2a_output_path, metadata_size, overwritten_para_vis, self.S2_metadata)
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
