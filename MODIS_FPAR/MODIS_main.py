import concurrent.futures
import numpy as np
import pandas as pd

import basic_function as bf
from osgeo import gdal, ogr, osr
import os
from itertools import repeat
import time
import traceback
import sys
import shutil
from NDsm import NDSparseMatrix
import scipy.sparse as sm
import psutil
import json
from datetime import datetime


global topts
topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])


class ISCCP_PAR(object):
    def __init__(self, file_path, work_env=None):
        if work_env is None:
            try:
                self.work_env = bf.Path(os.path.dirname(os.path.dirname(file_path)) + '\\').path_name
            except:
                print('There has no base dir for the ori_folder and the ori_folder will be treated as the work env')
                self.work_env = bf.Path(file_path).path_name
        else:
            self.work_env = bf.Path(work_env).path_name

        # Init key variable
        self.ROI, self.ROI_name = None, None
        self.main_coordinate_system = None
        self.nc_files = bf.file_filter(file_path, ['.nc'], subfolder_detection=True)

        # Generate the metadata
        self.metadata_df = {'FileDir': [], 'DOY': [], 'Hour': [], 'Datatype': []}
        for file in self.nc_files:
            try:
                filename = os.path.basename(file)
                if 'ISCCP_HXG_total_PAR' in file:
                    datatype = 'PAR'
                    doy = bf.date2doy(int(filename.split('_')[4]) * 10000 + int(filename.split('_')[5]) * 100 + int(filename.split('_')[6]))
                    hour = filename.split('_')[7]

                    self.metadata_df['FileDir'].append(file)
                    self.metadata_df['DOY'].append(doy)
                    self.metadata_df['Datatype'].append(datatype)
                    self.metadata_df['Hour'].append(hour)
                else:
                    print(f'The {file} is not a standard ISCCP file!')
            except:
                print(f'The {file} is not a standard ISCCP file!')

        self.doy_list = set(list(self.metadata_df['DOY']))
        self.year_range = set([int(np.floor(temp/1000)) for temp in self.doy_list])

        # Define cache folder
        self.cache_folder, self.trash_folder = self.work_env + 'cache\\', self.work_env + 'trash\\'
        bf.create_folder(self.cache_folder)
        bf.create_folder(self.trash_folder)

        # Create the output path
        self.output_path, self.shpfile_path, self.log_filepath = f'{self.work_env}ISCCP_Output\\', f'{self.work_env}shpfile\\', f'{self.work_env}logfile\\'
        bf.create_folder(self.output_path)
        bf.create_folder(self.log_filepath)
        bf.create_folder(self.shpfile_path)

    def save_log_file(func):
        def wrapper(self, *args, **kwargs):

            ############################################################################################################
            # Document the log file and para file
            # The log file contained the information for each run and para file documented the args of each func
            ############################################################################################################

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
            for func_key, func_processing_name in zip(['nc2tif', 'cal_dailyPAR', 'ds2sdc', 'extract_with_ROI'],
                                                      ['convert nc to tif file', 'calculate daily PAR', '2sdc', 'ROI extraction']):
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
                        log_file.writelines(
                            [f'Status: Error in {func_processing_name}!\n', 'Error information:\n', error_inf + '\n',
                             '#' * 70 + '\n'])

        return wrapper

    @save_log_file
    def seq_nc2tif(self,):
        for i in self.doy_list:
            self._nc2tif(self.output_path, i)

    @save_log_file
    def mp_nc2tif(self, ):
        # Create the file list based on the doy
        doyly_filelist = []
        for doy in self.doy_list:
            doy_filelist = []
            for file in self.nc_files:
                if f'.A{str(doy)}.' in file:
                    doy_filelist.append(file)
            doyly_filelist.append(doy_filelist)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._nc2tif, self.doy_list)

    def _nc2tif(self, doy: int, crs: str = None):

        # Process the nc file
        files = list(self.metadata_df[self.metadata_df['DOY'] == doy])

        if crs is None:
            crs = 'EPSG:32649'
        elif not isinstance(crs, str):
            raise TypeError(f'The csr should under str type!')

        # Create output folder
        ori_vrt_output_folder = f'{self.output_path}Ori_Denv_raster\\Hourly_VRT\\'
        ori_tif_output_folder = f'{self.output_path}Ori_Denv_raster\\Hourly_TIF\\'
        bf.create_folder(ori_vrt_output_folder)
        bf.create_folder(ori_tif_output_folder)

        roi_output_folder = f'{self.output_path}{str(self.ROI_name)}\\' if self.ROI_name is not None else None
        if roi_output_folder is not None:
            bf.create_folder(roi_output_folder)

        # Determine the sub-dataset
        src_ds = gdal.Open(files[0])

        # 获取投影信息，如果NetCDF中已有投影，可以直接使用
        projection = src_ds.GetProjection()
        if not projection:
            # 如果NetCDF中没有投影信息，手动设置WGS84
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)  # WGS84
            projection = srs.ExportToWkt()

        # 获取仿射变换参数
        geotransform = src_ds.GetGeoTransform()
        if not geotransform:
            # 如果NetCDF中没有仿射变换信息，手动定义
            # 假设数据覆盖全球，分辨率为0.1度
            geotransform = (-180, 0.1, 0, 90, 0, -0.1)

        # 定义驱动
        driver = gdal.GetDriverByName('GTiff')

        # 创建目标数据集
        dst_ds = driver.Create(
            output_tif,
            src_ds.RasterXSize,
            src_ds.RasterYSize,
            1,  # 波段数
            src_ds.GetRasterBand(1).DataType
        )

        # 设置地理参考
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)

        # 读取数据并写入
        data = src_ds.GetRasterBand(1).ReadAsArray()
        dst_ds.GetRasterBand(1).WriteArray(data)

        # 设置NoData值（可选）
        nodata = src_ds.GetRasterBand(1).GetNoDataValue()
        if nodata is not None:
            dst_ds.GetRasterBand(1).SetNoDataValue(nodata)

        # 关闭数据集
        dst_ds = None
        src_ds = None

        print(f"GeoTIFF已保存为 {output_tif}")


class MODIS_ds(object):

    def __init__(self, file_path, work_env=None):

        if work_env is None:
            try:
                self.work_env = bf.Path(os.path.dirname(os.path.dirname(file_path)) + '\\').path_name
            except:
                print('There has no base dir for the ori_folder and the ori_folder will be treated as the work env')
                self.work_env = bf.Path(file_path).path_name
        else:
            self.work_env = bf.Path(work_env).path_name

        # Init key variable
        self.ROI, self.ROI_name = None, None
        self.main_coordinate_system = None
        self.hdf_files = bf.file_filter(file_path, ['.hdf'], subfolder_detection=True)

        # ras2dc
        self._temporal_div_str = ['year', 'month']

        # Generate the df
        self.metadata_df = {'FileDir': [], 'DOY': [], 'Datatype': [], 'Tile': []}
        for file in self.hdf_files:
            try:
                filename = os.path.basename(file)
                datatype = filename.split('.')[0]
                doy = int(filename.split('.')[1].split('A')[-1])
                tile = filename.split('.')[2]

                self.metadata_df['FileDir'].append(file)
                self.metadata_df['DOY'].append(doy)
                self.metadata_df['Datatype'].append(datatype)
                self.metadata_df['Tile'].append(tile)

            except:
                print(f'The {file} is not a standard MODIS file!')
        self.metadata_df = pd.DataFrame(self.metadata_df)
        self.doy_list = set(list(self.metadata_df['DOY']))
        self.year_range = set([int(np.floor(temp/1000)) for temp in self.doy_list])

        # Define cache folder
        self.cache_folder, self.trash_folder = self.work_env + 'cache\\', self.work_env + 'trash\\'
        bf.create_folder(self.cache_folder)
        bf.create_folder(self.trash_folder)

        # Create the output path
        self.output_path, self.shpfile_path, self.log_filepath = f'{self.work_env}MODIS_Output\\', f'{self.work_env}shpfile\\', f'{self.work_env}logfile\\'
        bf.create_folder(self.output_path)
        bf.create_folder(self.log_filepath)
        bf.create_folder(self.shpfile_path)

    def _retrieve_para(self, required_para_name_list, protected_var=False, **kwargs):

        if not os.path.exists(f'{self.log_filepath}para_file.txt'):
            raise NameError('The para file is not established yet')
        else:
            para_file = open(f"{self.log_filepath}para_file.txt", "r+")
            para_raw_txt = para_file.read().split('\n')

        for para in required_para_name_list:
            for q in para_raw_txt:
                para = str(para)
                if q.startswith(para + ':'):
                    if q.split(para + ':')[-1] == 'None':
                        if not protected_var:
                            self.__dict__[para] = None
                        else:
                            self.__dict__['_' + para] = None
                    elif q.split(para + ':')[-1] == 'True':
                        if not protected_var:
                            self.__dict__[para] = True
                        else:
                            self.__dict__['_' + para] = True
                    elif q.split(para + ':')[-1] == 'False':
                        if not protected_var:
                            self.__dict__[para] = False
                        else:
                            self.__dict__['_' + para] = True
                    elif q.split(para + ':')[-1].startswith('['):
                        if not protected_var:
                            self.__dict__[para] = list(q.split(para + ':')[-1][1: -1])
                        else:
                            self.__dict__['_' + para] = list(q.split(para + ':')[-1][1: -1])
                    elif q.split(para + ':')[-1].startswith('('):
                        if not protected_var:
                            self.__dict__[para] = tuple(q.split(para + ':')[-1][1: -1])
                        else:
                            self.__dict__['_' + para] = tuple(q.split(para + ':')[-1][1: -1])
                    else:
                        try:
                            t = float(q.split(para + ':')[-1])
                            if not protected_var:
                                self.__dict__[para] = float(q.split(para + ':')[-1])
                            else:
                                self.__dict__['_' + para] = float(q.split(para + ':')[-1])
                        except:
                            if not protected_var:
                                self.__dict__[para] = q.split(para + ':')[-1]
                            else:
                                self.__dict__['_' + para] = q.split(para + ':')[-1]

    def save_log_file(func):
        def wrapper(self, *args, **kwargs):

            ############################################################################################################
            # Document the log file and para file
            # The log file contained the information for each run and para file documented the args of each func
            ############################################################################################################

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
            for func_key, func_processing_name in zip(['hdf2tif', 'cal_dailyPAR', 'ds2sdc', 'extract_with_ROI'],
                                                      ['convert hdf to tif file', 'calculate daily PAR', '2sdc', 'ROI extraction']):
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
                        log_file.writelines(
                            [f'Status: Error in {func_processing_name}!\n', 'Error information:\n', error_inf + '\n',
                             '#' * 70 + '\n'])

        return wrapper

    @save_log_file
    def seq_hdf2tif(self, subindex):
        for i in self.doy_list:
            self._hdf2tif(i, subindex)

    @save_log_file
    def mp_hdf2tif(self, subindex):
        # Create the file list based on the doy
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._hdf2tif, self.doy_list, repeat(subindex))

    def _hdf2tif(self, doy: int, subindex_list: list, crs: str = None):

        # Get the hdf file under doy
        files = list(self.metadata_df[self.metadata_df['DOY'] == doy]['FileDir'])

        if crs is None:
            crs = 'EPSG:32649'
        elif not isinstance(crs, str):
            raise TypeError(f'The csr should under str type!')

        # Create output folder
        ori_vrt_output_folder = f'{self.output_path}Ori_Denv_raster\\Hourly_VRT\\'
        ori_tif_output_folder = f'{self.output_path}Ori_Denv_raster\\Hourly_TIF\\'
        bf.create_folder(ori_vrt_output_folder)
        bf.create_folder(ori_tif_output_folder)

        roi_output_folder = f'{self.output_path}{str(self.ROI_name)}\\' if self.ROI_name is not None else None
        if roi_output_folder is not None:
            bf.create_folder(roi_output_folder)

        # Determine the sub-dataset
        ds_temp = gdal.Open(files[0])
        subset_ds = ds_temp.GetSubDatasets()
        subindex_supported = [subset[0].split(':')[-1] for subset in subset_ds]
        subindex_dic = {}
        for subindex in subindex_list:
            if subindex not in subindex_supported:
                raise TypeError(f'{subindex} is not supported!')
            else:
                subindex_dic[subindex] = []

        # Separate all the hdf files through the index
        for file in files:
            ds_temp = gdal.Open(file)
            subset_ds = ds_temp.GetSubDatasets()
            subindex_temp = [subset[0].split(':')[-1] for subset in subset_ds]
            for subindex in subindex_list:
                if subindex not in subindex_temp:
                    raise TypeError(f'The file {file} is not consistency compared to other files in the ds!')
                else:
                    subindex_dic[subindex].append(subset_ds[subindex_temp.index(subindex)][0])

        # Create the tiffiles based on each index
        for subindex in subindex_list:
            if not os.path.exists(f'{ori_tif_output_folder}{str(subindex)}_{str(doy)}.tif'):
                s_t = time.time()
                print(f'Start generating the {subindex} tiffile of {str(doy)}')
                tiffile_list = []
                for file in subindex_dic[subindex]:
                    # Generate tiffile_list
                    filename = file.split('\\')[-1].split('.hdf')[0]
                    tiffile_list.append(f'/vsimem/{filename}.TIF')

                    # Create tiffile
                    ds_temp = gdal.Open(file)
                    gdal.Warp(f'/vsimem/{filename}.TIF', ds_temp, dstSRS=crs)

                # BuildVRT
                vrt = gdal.BuildVRT(f'{ori_vrt_output_folder}{str(subindex)}_{str(doy)}.vrt', tiffile_list)
                gdal.Translate(f'{ori_tif_output_folder}{str(subindex)}_{str(doy)}.tif', vrt, options=topts)

                for file in tiffile_list:
                    gdal.Unlink(file)
                print(f'Finish processing the {subindex} tiffile of {str(doy)} in {str(time.time()-s_t)[:6]}s')

    @save_log_file
    def seq_cal_dailyPAR(self, method='mean'):
        for doy in self.doy_list:
            self._cal_dailyPAR(doy, method=method)

    @save_log_file
    def mp_cal_dailyPAR(self, method: str = 'mean'):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._cal_dailyPAR, self.doy_list, repeat(method))

    def _cal_dailyPAR(self, doy: int, method='mean', remove_low_quality_data=True) -> None:

        input_path = f'{self.output_path}Ori_Denv_raster\\Hourly_TIF\\'
        output_path = f'{self.output_path}Ori_Denv_raster\\DPAR\\'
        bf.create_folder(output_path)

        if not os.path.exists(f'{output_path}{str(doy)}_DPAR.TIF'):
            s_t = time.time()
            print(f'Start generating the {str(doy)}_DPAR.TIF')
            file_list = bf.file_filter(input_path, [str(doy), 'GMT'], and_or_factor='and')
            if file_list == []:
                raise Exception(f'The {str(doy)} files is not properly generated or the input folder is not correct!')

            array_temp, nodatavalue = None, None
            for file in file_list:
                ds_temp = gdal.Open(file)
                if array_temp is None:
                    array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                    array_temp = array_temp.reshape([array_temp.shape[0], array_temp.shape[1], 1])
                else:
                    array_t = ds_temp.GetRasterBand(1).ReadAsArray()
                    array_temp = np.concatenate((array_temp, array_t.reshape(array_t.shape[0], array_temp.shape[1], 1)), axis=2)

                if nodatavalue is None:
                    nodatavalue = ds_temp.GetRasterBand(1).GetNoDataValue()

            if method == 'mean':
                array_temp = np.nanmean(array_temp, axis=2) * 86.4
                bf.write_raster(ds_temp, array_temp, output_path, f'{str(doy)}_DPAR.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=nodatavalue)
            else:
                pass
            print(f'Finish generating the {str(doy)}_DPAR file in {str(time.time()-s_t)[0:5]}s')

    @save_log_file
    def seq_extract_with_ROI(self, ROI, zvalue: list, bounds: list = None, ras_res: list = None):

        for ztemp in zvalue:
            filelist = bf.file_filter(f'{self.output_path}Ori_Denv_raster\\{ztemp}\\', ['.TIF'])
            if len(filelist) == 0:
                self.seq_cal_dailyPAR()

            for file in filelist:
                self._extract_ras_with_ROI(file, ROI, bounds=bounds, ras_res=ras_res)

    @save_log_file
    def mp_extract_with_ROI(self, ROI, zvalue: list, bounds: list = None, ras_res: list = None):

        for ztemp in zvalue:
            filelist = bf.file_filter(f'{self.output_path}Ori_Denv_raster\\{ztemp}\\', ['.TIF'], exclude_word_list=['.aux', '.ovr'])
            if len(filelist) == 0:
                self.mp_cal_dailyPAR()

            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(self._extract_ras_with_ROI, filelist, repeat(ROI), repeat(bounds), repeat(ras_res))

    def _extract_ras_with_ROI(self, file: str, ROI, bounds: list = None, ras_res: list = None):

        # Process kwargs
        if isinstance(ROI, str):
            if ROI.endswith('.shp'):
                self.ROI, self.ROI_name = ROI, ROI.split('\\')[-1].split('.')[0]
            else:
                self.ROI, self.ROI_name = None, None
        else:
            raise TypeError(f'{str(ROI)} is not a supported ROI')

        if bounds is None and self.ROI is not None:
            shp_temp = ogr.Open(self.ROI)
            bounds = shp_temp.GetLayer(1).GetExtent()
            bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
        elif not isinstance(bounds, tuple):
            raise TypeError(f'The bounds should under tuple type!')
        elif len(bounds) != 4:
            raise TypeError(f'The bounds should under tuple type (Xmin, Ymin, Xmax, Ymax)!')

        if ras_res is None:
            ras_res = [10, 10]
        elif not isinstance(ras_res, list):
            raise TypeError(f'The bounds should under list type!')
        elif len(ras_res) != 2:
            raise TypeError(f'The ras res should under list type (Ysize, Xsize)!')

        bf.create_folder(f'{self.output_path}{self.ROI_name}_Denv_raster\\DPAR\\')

        # Cut using the para
        if self.ROI is not None:
            filename = file.split('\\')[-1]
            if not os.path.exists(f"{self.output_path}{self.ROI_name}_Denv_raster\\DPAR\\{filename}"):
                s_t = time.time()
                print(f'Start cutting the {str(filename)} using {self.ROI_name}.shp!')
                ds_temp = gdal.Open(file)
                gdal.Warp(f"/vsimem/{self.ROI_name}{filename.split('.TIF')[0]}_TEMP.TIF", ds_temp, xRes=ras_res[0], yRes=ras_res[1], resampleAlg=gdal.GRA_Bilinear, outputBounds=bounds)
                gdal.Warp(f"/vsimem/{self.ROI_name}{filename.split('.TIF')[0]}_TEMP2.TIF", f"/vsimem/{self.ROI_name}{filename.split('.TIF')[0]}_TEMP.TIF", xRes=ras_res[0], yRes=ras_res[1], cutlineDSName=self.ROI, outputBounds=bounds)
                gdal.Translate(f"{self.output_path}{self.ROI_name}_Denv_raster\\DPAR\\{filename}", f"/vsimem/{self.ROI_name}{filename.split('.TIF')[0]}_TEMP2.TIF", options=topts)
                gdal.Unlink(f"/vsimem/{self.ROI_name}{filename.split('.TIF')[0]}_TEMP2.TIF")
                gdal.Unlink(f"/vsimem/{self.ROI_name}{filename.split('.TIF')[0]}_TEMP.TIF")
                print(f'Finish cutting the {str(filename)} in {str(time.time()-s_t)}s!')

    def _process_raster2dc_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('inherit_from_logfile', 'ROI', 'ROI_name', 'dc_overwritten_para',
                                       'manually_remove_datelist', 'size_control_factor'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process clipped_overwritten_para
        if 'dc_overwritten_para' in kwargs.keys():
            if isinstance(kwargs['dc_overwritten_para'], bool):
                self._dc_overwritten_para = kwargs['dc_overwritten_para']
            else:
                raise TypeError('Please mention the dc_overwritten_para should be bool type!')
        else:
            self._clipped_overwritten_para = False

        # process inherit from logfile
        if 'inherit_from_logfile' in kwargs.keys():
            if isinstance(kwargs['inherit_from_logfile'], bool):
                self._inherit_from_logfile = kwargs['inherit_from_logfile']
            else:
                raise TypeError('Please mention the dc_overwritten_para should be bool type!')
        else:
            self._inherit_from_logfile = False

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
        if 'ds2ras_method' in kwargs.keys():
            if 'ds2ras_method' in self._ds2raster_method:
                self._ds2ras_method = kwargs['ds2ras_method']
            else:
                raise TypeError('Please mention the ds2ras_method should be supported!')
        elif self._inherit_from_logfile:
            self._retrieve_para(['ds2ras_method'], protected_var=True)
        else:
            raise TypeError('Please mention the ds2ras_method should be supported!')

    def raster2dc(self, index_list: list, temporal_division=None, ROI=None, ** kwargs):

        kwargs['inherit_from_logfile'] = True
        self._process_raster2dc_para(**kwargs)

        if temporal_division is not None and not isinstance(temporal_division, str):
            raise TypeError(f'The {temporal_division} should be a str!')
        elif temporal_division is None:
            temporal_division = 'all'
        elif temporal_division not in self._temporal_div_str:
            raise ValueError(f'The {temporal_division} is not supported!')

        if ROI is not None:
            self.ROI = ROI
            self.ROI_name = self.ROI.split('\\')[-1].split('.')[0]

        for index_temp in index_list:

            # Create the output path
            output_path = f'{self.output_path}{self.ROI_name}_Denv_datacube\\'
            bf.create_folder(output_path)

            # Obtain the input files
            input_folder = f'{self.output_path}{self.ROI_name}_Denv_raster\\{index_temp}\\'
            input_files = bf.file_filter(input_folder, ['.TIF'], exclude_word_list=['aux'])
            if self.main_coordinate_system is None:
                self.main_coordinate_system = bf.retrieve_srs(gdal.Open(input_files[0]))

            # Create the ROI map
            roi_map_folder = f'{self.output_path}ROI_map\\'
            bf.create_folder(roi_map_folder)
            ROI_tif_name, ROI_array_name = f'{roi_map_folder}{self.ROI_name}.TIF', f'{roi_map_folder}{self.ROI_name}.npy'
            if not os.path.exists(ROI_tif_name) or not os.path.exists(ROI_array_name):
                for i in range(len(input_files)):
                    try:
                        shutil.copyfile(input_files[i], ROI_tif_name)
                        break
                    except:
                        pass
                ds_temp = gdal.Open(ROI_tif_name, gdal.GA_Update)
                array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                nodata_value = ds_temp.GetRasterBand(1).GetNoDataValue()
                array_temp[array_temp != nodata_value] = 1
                np.save(ROI_array_name, array_temp)
                ds_temp.GetRasterBand(1).WriteArray(array_temp)
                ds_temp.FlushCache()
                ds_temp = None
            else:
                ds_temp = gdal.Open(ROI_tif_name)
                array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                nodata_value = ds_temp.GetRasterBand(1).GetNoDataValue()
            cols, rows = array_temp.shape[1], array_temp.shape[0]
            sparsify = np.sum(array_temp == nodata_value) / (array_temp.shape[0] * array_temp.shape[1])
            _sparse_matrix = True if sparsify > 0.9 else False

            # Create the header dic
            metadata_dic = {'ROI_name': self.ROI_name, 'index': index_temp, 'Datatype': 'float', 'ROI': self.ROI,
                            'ROI_array': ROI_array_name, 'ROI_tif': ROI_tif_name, 'sdc_factor': True, 'Denv_factor': True,
                            'coordinate_system': self.main_coordinate_system, 'size_control_factor': False,
                            'oritif_folder': input_folder, 'dc_group_list': None, 'tiles': None, 'timescale': temporal_division}

            if temporal_division == 'year':
                time_range = self.year_range
            elif temporal_division == 'month':
                time_range = []
                for year_temp in self.year_range:
                    for month_temp in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
                        time_range.append(str(year_temp) + str(month_temp))
            elif temporal_division == 'all':
                time_range = ['TIF']

            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                executor.map(self._raster2sdc, repeat(output_path), repeat(input_folder), time_range, repeat(index_temp),
                             repeat(metadata_dic), repeat(rows), repeat(cols), repeat(_sparse_matrix))

    def _raster2sdc(self, output_path, input_folder, time_temp, zvalue_temp, metadata_dic, rows, cols, _sparse_matrix, ):
        start_time = time.time()
        print(f'Start constructing the {str(time_temp)} {str(zvalue_temp)} Denv datacube of {self.ROI_name}.')
        # Construct the header dic
        nodata_value = None

        # Create the output path
        yearly_output_path = output_path + str(int(time_temp)) + '\\' if time_temp != 'TIF' else output_path + 'all\\'
        bf.create_folder(yearly_output_path)

        if not os.path.exists(f'{yearly_output_path}doy.npy') or not os.path.exists(f'{yearly_output_path}metadata.json'):

            # Determine the input files
            yearly_input_files = bf.file_filter(input_folder, ['.TIF', '\\' + str(time_temp)], exclude_word_list=['aux'], and_or_factor='and')
            if yearly_input_files == []:
                raise Exception('There are no valid input files, double check the temporal division!')

            if nodata_value is None:
                nodata_value = gdal.Open(yearly_input_files[0])
                nodata_value = nodata_value.GetRasterBand(1).GetNoDataValue()

            # Create the doy list
            doy_list = [int(filepath_temp.split('\\')[-1][0:7]) for filepath_temp in yearly_input_files]

            # Determine whether the output folder is huge and sparsify or not?
            mem = psutil.virtual_memory()
            dc_max_size = int(mem.free * 0.90)
            _huge_matrix = True if len(doy_list) * cols * rows * 2 > dc_max_size else False

            if _huge_matrix:
                if _sparse_matrix:
                    i = 0
                    data_cube = NDSparseMatrix()
                    data_valid_array = np.zeros([len(doy_list)], dtype=int)
                    while i < len(doy_list):

                        try:
                            t1 = time.time()
                            if not os.path.exists(f"{input_folder}{str(doy_list[i])}_{zvalue_temp}.TIF"):
                                raise Exception(f'The {str(doy_list[i])}_{zvalue_temp} is not properly generated!')
                            else:
                                array_temp = gdal.Open(f"{input_folder}{str(doy_list[i])}_{zvalue_temp}.TIF")
                                array_temp = array_temp.GetRasterBand(1).ReadAsArray()
                                array_temp[array_temp == nodata_value] = 0

                            sm_temp = sm.csr_matrix(array_temp.astype(np.uint16))
                            array_temp = None
                            data_cube.append(sm_temp, name=doy_list[i])
                            data_valid_array[i] = 1 if sm_temp.data.shape[0] == 0 else 0

                            print(f'Assemble the {str(doy_list[i])} into the sdc using {str(time.time() - t1)[0:5]}s (layer {str(i)} of {str(len(doy_list))})')
                            i += 1
                        except:
                            error_inf = traceback.format_exc()
                            print(error_inf)

                    # remove nan layers
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
                    np.save(f'{yearly_output_path}doy.npy', doy_list)
                    data_cube.save(f'{yearly_output_path}{zvalue_temp}_Denv_datacube\\')
                else:
                    pass
            else:
                i = 0
                data_cube = np.zeros([rows, cols, len(doy_list)])
                data_valid_array = np.zeros([len(doy_list)], dtype=int)
                while i < len(doy_list):

                    try:
                        t1 = time.time()
                        if not os.path.exists(f"{input_folder}{str(doy_list[i])}_{zvalue_temp}.TIF"):
                            raise Exception(f'The {str(doy_list[i])}_{zvalue_temp} is not properly generated!')
                        else:
                            array_temp = gdal.Open(f"{input_folder}{str(doy_list[i])}_{zvalue_temp}.TIF")
                            array_temp = array_temp.GetRasterBand(1).ReadAsArray()

                        data_cube[:, :, i] = array_temp
                        data_valid_array[i] = np.all(array_temp == nodata_value)

                        print(f'Assemble the {str(doy_list[i])} into the sdc using {str(time.time() - t1)[0:5]}s (layer {str(i)} of {str(len(doy_list))})')
                        i += 1
                    except:
                        error_inf = traceback.format_exc()
                        print(error_inf)

                # remove nan layers
                i_temp = 0
                while i_temp < len(doy_list):
                    if data_valid_array[i_temp]:
                        data_cube = np.delete(data_cube, i_temp, 2)
                        doy_list.remove(doy_list[i_temp])
                        data_valid_array = np.delete(data_valid_array, i_temp, 0)
                        i_temp -= 1
                    i_temp += 1

                np.save(f'{yearly_output_path}doy.npy', doy_list)
                np.save(f'{yearly_output_path}{zvalue_temp}_Denv_datacube.npy', data_cube)

                # Save the metadata dic
            metadata_dic['sparse_matrix'], metadata_dic['huge_matrix'] = _sparse_matrix, _huge_matrix
            metadata_dic['timerange'] = time_temp
            with open(f'{yearly_output_path}metadata.json', 'w') as js_temp:
                json.dump(metadata_dic, js_temp)

        print(f'Finish constructing the {str(time_temp)} {str(zvalue_temp)} Denv datacube of {self.ROI_name} in \033[1;31m{str(time.time() - start_time)} s\033[0m.')


if __name__ == '__main__':
    MD_ds = MODIS_ds('G:\\A_Climatology_dataset\\gridded_dataset\\MODIS_PAR_V6.2\\Ori\\')
    # MD_ds.mp_hdf2tif(['PAR_Quality', 'GMT_0000_PAR', 'GMT_0300_PAR', 'GMT_0600_PAR', 'GMT_0900_PAR', 'GMT_1200_PAR', 'GMT_1500_PAR', 'GMT_1800_PAR', 'GMT_2100_PAR'])
    # MD_ds.mp_cal_dailyPAR()
    # bounds_temp = bf.raster_ds2bounds('G:\\A_Climatology_dataset\\gridded_dataset\\MODIS_PAR_V6.2\\ROI_map\\floodplain_2020.TIF')
    # MD_ds.mp_extract_with_ROI('G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020.shp', ['DPAR'], bounds=bounds_temp, ras_res=[30, 30])
    MD_ds.raster2dc(['DPAR'], ROI='G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020.shp', temporal_division='year', inherit_from_logfile=True)
