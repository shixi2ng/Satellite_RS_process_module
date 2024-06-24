import shutil
from NDsm import NDSparseMatrix
import numpy as np
from osgeo import gdal, osr
import pandas as pd
import basic_function as bf
import datetime
import geopandas as gp
import time
import os
import concurrent.futures
from itertools import repeat
import copy
import traceback
import sys
from utils import shp2raster_idw
from osgeo import ogr
import psutil
import scipy.sparse as sm
import json
from shapely import Polygon, Point

global topts
topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])


class NCEI_ds(object):

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
        csv_files = bf.file_filter(file_path, ['.csv'], subfolder_detection=True)
        self.year_range = list(set([int(temp.split('.csv')[0].split('_')[-1]) for temp in csv_files]))
        self.station_list = list(set([int(temp.split('\\')[-1].split('_')[0]) for temp in csv_files]))
        self.files_content_dic, self.valid_inform, self.index = {}, [], []

        # Define cache folder
        self.cache_folder, self.trash_folder = self.work_env + 'cache\\', self.work_env + 'trash\\'
        bf.create_folder(self.cache_folder)
        bf.create_folder(self.trash_folder)

        # Create output path
        self.output_path, self.shpfile_path, self.log_filepath = f'{self.work_env}NCEI_Output\\', f'{self.work_env}shpfile\\', f'{self.work_env}logfile\\'
        bf.create_folder(self.output_path)
        bf.create_folder(self.log_filepath)
        bf.create_folder(self.shpfile_path)

        # ds2raster
        self._ds2raster_method_tup = ('IDW',)
        self._ds2raster_method = None

        # ras2dc
        self._temporal_div_str = ['year', 'month']

        for year in self.year_range:
            valid_inform_list = []
            self.files_content_dic[year] = []
            current_year_files = bf.file_filter(file_path, ['.csv', str(year)], and_or_factor='and', subfolder_detection=True)

            for csv_file_path in current_year_files:
                df_temp = pd.read_csv(csv_file_path)
                self.files_content_dic[year].append(df_temp)
                valid_inform_list.extend(list(df_temp.keys()))

            self.valid_inform.extend(valid_inform_list)

        self.valid_inform = list(set(self.valid_inform))

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
            for func_key, func_processing_name in zip(['ds2pointshp', 'ds2raster', 'raster2dc'], ['2point', '2raster', 'rs2dc']):
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
    def ds2pointshp(self, zvalue_list: list, output_path: str, main_coordinate_system: str):

        output_path = bf.Path(output_path).path_name
        bf.create_folder(output_path)

        for z in zvalue_list:
            if z not in self.valid_inform:
                raise ValueError(f'The zvalue {str(z)} is not valid!')

        index_all = ''
        for index in zvalue_list:
            index_all = f'{index_all}_{str(index)}'

        basic_inform4point = copy.copy(zvalue_list)
        for z in ['LATITUDE', 'LONGITUDE', 'STATION', 'DATE']:
            if z not in basic_inform4point:
                basic_inform4point.append(z)

        if not isinstance(main_coordinate_system, str):
            raise TypeError(f'Please input the {main_coordinate_system} as a string!')
        else:
            self.main_coordinate_system = main_coordinate_system

        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._ds2point, self.year_range, repeat(output_path), repeat(basic_inform4point), repeat(index_all), repeat(self.main_coordinate_system))

    def _ds2point(self, year, output_path, z_4point, index_all, crs):

        z_dic = {}
        current_year_files_content = self.files_content_dic[year]
        for date_temp in range(0, datetime.date(year, 12, 31).toordinal() - datetime.date(year, 1, 1).toordinal() + 1):

            date = datetime.datetime.fromordinal(datetime.date(year, 1, 1).toordinal() + date_temp)
            t1 = time.time()
            print(f'Start processing the climatology data of \033[1;31m{str(datetime.date.strftime(date, "%Y%m%d"))}\033[0m')
            if not os.path.exists(f'{output_path}{str(datetime.date.strftime(date, "%Y%m%d"))}{index_all}.shp'):

                for z in z_4point:
                    z_dic[z] = []

                for current_year_file_content in current_year_files_content:
                    for z in z_4point:
                        try:
                            if z == 'TEMP':
                                z_dic[z].append(int(current_year_file_content[current_year_file_content['DATE'] == datetime.date.strftime(date, "%Y-%m-%d")][z].values[0] * 10))
                            else:
                                z_dic[z].append(current_year_file_content[current_year_file_content['DATE'] == datetime.date.strftime(date, "%Y-%m-%d")][z].values[0])
                        except:
                            print(f"The {z} data of {str(date)} in STATION:{str(current_year_file_content.loc[0, 'STATION'])} was missing!!")

                geodf_temp = gp.GeoDataFrame(z_dic, geometry=gp.points_from_xy(z_dic['LONGITUDE'], z_dic['LATITUDE']), crs="EPSG:4326")
                geodf_temp = geodf_temp.to_crs(crs)

                if geodf_temp.size == 0:
                    print(f'There has no valid file for date \033[1;31m{str(datetime.date.strftime(date, "%Y%m%d"))}\033[0m')
                else:
                    geodf_temp.to_file(f'{output_path}{str(datetime.date.strftime(date, "%Y%m%d"))}{index_all}.shp', encoding='gbk')
                    print(f'Finish generating the shpfile of \033[1;31m{str(datetime.date.strftime(date, "%Y%m%d"))}\033[0m in \033[1;34m{str(time.time()-t1)[0:7]}\033[0m s')

    @save_log_file
    def ds2raster(self, zvalue_list: list, raster_size=None, ds2ras_method=None, bounds=None, ROI=None, crs=None):

        # Process ds2raster para
        if isinstance(zvalue_list, str):
            zvalue_list = [zvalue_list]
        elif not isinstance(zvalue_list, list):
            raise TypeError('The zvalue should be a list')

        if ds2ras_method is None:
            self._ds2raster_method = 'IDW'
        elif ds2ras_method not in self._ds2raster_method_tup:
            raise ValueError(f'The {ds2ras_method} is not supported for ds2raster!')
        else:
            self._ds2raster_method = ds2ras_method

        if isinstance(ROI, str):
            if not ROI.endswith('.shp'):
                raise TypeError(f'The ROI should be a valid shpfile!')
            else:
                self.ROI = ROI
                self.ROI_name = self.ROI.split('\\')[-1].split('.')[0]
        else:
            raise TypeError(f'The ROI should be a valid shpfile!')

        if isinstance(bounds, tuple):
            if len(bounds) != 4 and False in [type(temp) in [float, np.float, int, np.int16] for temp in bounds]:
                raise TypeError(f'bounds should be under the tuple type with num-coord in it!')
        elif bounds is not None:
            raise TypeError(f'bounds should be under the tuple type!')

        if raster_size is None and bounds is not None:
            raster_size = [int((bounds[3] - bounds[1]) / 10), int((bounds[2] - bounds[0]) / 10)]
        elif isinstance(raster_size, list) and len(raster_size) == 2:
            raster_size = raster_size
        elif raster_size is not None:
            raise TypeError(f'raster size should under the list type!')

        if crs is not None:
            self.main_coordinate_system = crs
        else:
            self.main_coordinate_system = 'EPSG:32649'

        # Create the point shpfiles
        shpfile_folder = self.output_path + 'Ori_shpfile\\'
        bf.create_folder(shpfile_folder)
        self.ds2pointshp(zvalue_list, shpfile_folder, self.main_coordinate_system)

        for zvalue_temp in zvalue_list:

            # Generate output bounds based on the point bounds
            if bounds is None:
                shp_temp = ogr.Open(bf.file_filter(shpfile_folder, ['.shp'])[0])
                bounds = shp_temp.GetLayer(1).GetExtent()
                bounds = (bounds[0], bounds[1], bounds[2], bounds[3])

            if raster_size is None:
                raster_size = [int((bounds[3] - bounds[1]) / 10), int((bounds[2] - bounds[0]) / 10)]

            # Retrieve all the point shpfiles
            shpfiles = bf.file_filter(shpfile_folder, ['.shp'])
            if shpfiles == []:
                raise ValueError(f'There are no valid shp files in the {str(shpfile_folder)}!')

            # Generate the raster
            if ds2ras_method == 'IDW':

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    executor.map(shp2raster_idw, shpfiles, repeat(self.output_path), repeat(zvalue_temp), repeat(raster_size), repeat(bounds), repeat(self.ROI), repeat(self.main_coordinate_system))

            else:
                pass

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
            if self._ds2ras_method is None:
                self._ds2ras_method = 'IDW'
        else:
            raise TypeError('Please mention the ds2ras_method should be supported!')

    @save_log_file
    def raster2dc(self, zvalue_list: list, temporal_division=None, ROI=None, **kwargs):

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

        for zvalue_temp in zvalue_list:

            # Create the output path
            output_path = f'{self.output_path}{self.ROI_name}_Denv_datacube\\' if self.ROI_name is not None else f'{self.output_path}Ori_Denv_datacube\\'
            bf.create_folder(output_path)

            # Obtain the input files
            input_folder = f'{self.output_path}{str(self.ROI_name)}_Denv_raster\\IDW_{zvalue_temp}\\' if self.ROI_name is not None else f'{self.output_path}Ori_Denv_raster\\IDW_{zvalue_temp}\\'
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
                array_temp = array_temp.astype(np.int16)

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

            # Create the metadata dic
            metadata_dic = {'ROI_name': self.ROI_name, 'index': zvalue_temp, 'Datatype': 'float', 'ROI': self.ROI,
                            'ROI_array': ROI_array_name, 'ROI_tif': ROI_tif_name, 'sdc_factor': True,
                            'coordinate_system': self.main_coordinate_system, 'size_control_factor': False,
                            'oritif_folder': input_folder, 'dc_group_list': None, 'tiles': None, 'timescale': temporal_division,
                            'Denv_factor': True}

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
                executor.map(self._raster2sdc, repeat(output_path), repeat(input_folder), time_range, repeat(zvalue_temp), repeat(metadata_dic), repeat(rows), repeat(cols), repeat(_sparse_matrix))

    def _raster2sdc(self, output_path, input_folder, time_temp, zvalue_temp, metadata_dic, rows, cols, _sparse_matrix,):

        start_time = time.time()
        print(f'Start constructing the {str(time_temp)} {str(zvalue_temp)} sdc of {self.ROI_name}.')
        nodata_value = None

        # Create the output path
        yearly_output_path = output_path + str(int(time_temp)) + '\\' if time_temp != 'TIF' else output_path + 'all\\'
        bf.create_folder(yearly_output_path)

        if not os.path.exists(f'{yearly_output_path}doy.npy') or not os.path.exists(f'{yearly_output_path}metadata.json'):

            # Determine the input files
            yearly_input_files = bf.file_filter(input_folder, ['.TIF', '\\' + str(time_temp)], exclude_word_list=['aux'],  and_or_factor='and')

            if nodata_value is None:
                nodata_value = gdal.Open(yearly_input_files[0])
                nodata_value = nodata_value.GetRasterBand(1).GetNoDataValue()

            # Create the doy list
            doy_list = bf.date2doy([int(filepath_temp.split('\\')[-1][0:8]) for filepath_temp in yearly_input_files])

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
                            if not os.path.exists(f"{input_folder}{str(bf.doy2date(doy_list[i]))}_{zvalue_temp}.TIF"):
                                raise Exception(f'The {str(doy_list[i])}_{zvalue_temp} is not properly generated!')
                            else:
                                array_temp = gdal.Open(f"{input_folder}{str(bf.doy2date(doy_list[i]))}_{zvalue_temp}.TIF")
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
                        data_valid_array[i] = [array_temp == nodata_value].all()

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

        print(f'Finish constructing the {str(time_temp)} {str(zvalue_temp)} sdc of {self.ROI_name} in \033[1;31m{str(time.time() - start_time)} s\033[0m.')


class CMA_ds(object):

    def __init__(self, file_path, work_env=None):

        if work_env is None:
            try:
                self.work_env = bf.Path(os.path.dirname(file_path) + '\\').path_name
            except:
                print('There has no base dir for the ori_folder and the ori_folder will be treated as the work env')
                self.work_env = bf.Path(file_path).path_name
        else:
            self.work_env = bf.Path(work_env).path_name

        # Define the metadata folder
        self.metadata_folder = self.work_env + 'metadata\\'
        bf.create_folder(self.metadata_folder)

        # Init _arg
        self._ds2raster_method_tup = ('IDW',)
        self._ds2raster_method = None

        # ras2dc
        self._temporal_div_str = ['year', 'month']

        # Init key variable
        self.ROI, self.ROI_name = None, None
        self.main_coordinate_system = None
        csv_files = bf.file_filter(file_path, ['SURF_CLI_CHN', '.txt', '.TXT'], subfolder_detection=True, and_or_factor='or')
        for _ in csv_files:
            if 'SURF_CLI_CHN_MUL_DAY' not in _:
                csv_files.remove(_)
                print(f'{str(_)} is not a valid SURF_CLI_CHN file!')

        # Valid index
        self._valid_index = ['EVP', 'PRS', 'WIN', 'TEM', 'GST', 'RHU', 'PRE', 'SSD']
        self._index_csvfile = {'EVP': [], 'PRS': [], 'WIN': [], 'TEM': [], 'GST': [], 'RHU': [], 'PRE': [], 'SSD': []}
        self._index_dic = {'EVP': [(7, 0.1, 'Minor EVP', 32766), (8, 0.1, 'Major EVP', 32766)],
                           'PRS': [(7, 0.1, 'DAve_Airp', 32766), (8, 0.1, 'DMax_Airp', 32766), (9, 0.1, 'DMin_Airp', 32766)],
                           'WIN': [(7, 0.1, 'DAve_Winds', 32766), (8, 0.1, 'DMax_Winds', 32766), (10, 0.1, 'DMin_Winds', 32766)],
                           'TEM': [(7, 0.1, 'DAve_AT', 32766), (8, 0.1, 'DMax_AT', 32766), (9, 0.1, 'DMin_AT', 32766)],
                           'GST': [(7, 0.1, 'DAve_LST', 32766), (8, 0.1, 'DMax_LST', 32766), (9, 0.1, 'DMin_LST', 32766)],
                           'RHU': [(7, 0.01, 'DAve_hum', 32766), (8, 0.01, 'Dmin_hum', 32766)],
                           'PRE': [(7, 0.1, '20-8_prec', 32766), (8, 0.1, '8-20_prec', 32766), (9, 0.1, 'Acc_prec', 32766)],
                           'SSD': [(7, 0.1, 'SS_hour', 32766)]}
        self._interpolate_index = {'EVP': (8, 0.1, 'Major EVP', 32766), 'PRS': (7, 0.1, 'DAve_Airp', 32766), 'WIN': (7, 0.1, 'DAve_Winds', 32766),
                                   'TEM': (7, 0.1, 'DAve_AT', 32766), 'GST': (7, 0.1, 'DAve_LST', 32766), 'RHU': (7, 0.01, 'DAve_hum', 32766),
                                   'PRE': (9, 0.1, 'Acc_prec', 32766), 'SSD': (7, 0.1, 'SS_hour', 32766)}
        self.index = list(set([_.split('SURF_CLI_CHN_MUL_DAY-')[1].split('-')[0] for _ in csv_files]))
        self._splin_conf = '{}\n1\n2\n1\n0\n0\n{}\n{}\n-400 9000 1 1\n1\n0\n3\n1\n0\n1\n1\n{}.dat\n300\n6\n(A6,2F14.6,F6.2,F6.2)\n{}.res\n{}.opt\n{}.sur\n{}.lis\n{}.cov\n\n\n\n\n\n'
        self._lapgrd_conf = '{}\n1\n1\n{}\n2\n3\n1\n1\n{}\n2\n{}\n2\n{}\n2\n{}\n2\n{}\n{}\n{}\n2\n{}\n{}\n{}\n\n\n\n\n\n'

        if False in [_ in self._valid_index for _ in self.index]:
            raise Exception(f'The index {str(self.index[[_ in self._valid_index for _ in self.index].index(False)])}')

        for __ in csv_files:
            if True not in [index_ in __ for index_ in self.index]:
                csv_files.remove(__)
            else:
                for index_ in self._valid_index:
                    if index_ in __:
                        self._index_csvfile[index_].append(__)

        # Get the index month and
        self._cma_header_ = ['Station_id', 'Lat', 'Lon', 'Alt', 'YYYY', 'MM', 'DD']
        station_inform_dic = {'Station_id': [], 'Lon': [], 'Lat': [], 'Alt': []}
        self.date_range = {}
        for index_ in self.index:
            self.date_range[index_] = []
            if not os.path.exists(os.path.join(self.metadata_folder, f'{index_}.csv')) or not os.path.exists(os.path.join(self.metadata_folder, f'station_inform.csv')):
                index_month_station = {'Station_id': [], 'Index': [], 'Month': []}
                for csv_ in csv_files:
                    if index_ in csv_:
                        month_ = int(csv_.split('-')[-1].split('.')[0])
                        df_temp = pd.read_table(csv_, delim_whitespace=True, header=None)
                        station_all = pd.unique(df_temp[0])
                        for station_ in station_all:
                            if station_ not in station_inform_dic['Station_id']:
                                station_inform_dic['Station_id'].append(station_)
                                station_inform_dic['Lon'].append(pd.unique(df_temp[df_temp[0] == station_][1])[0] // 100 + np.mod(pd.unique(df_temp[df_temp[0] == station_][1])[0], 100) / 60)
                                station_inform_dic['Lat'].append(pd.unique(df_temp[df_temp[0] == station_][2])[0] // 100 + np.mod(pd.unique(df_temp[df_temp[0] == station_][2])[0], 100) / 60)
                                station_inform_dic['Alt'].append(pd.unique(df_temp[df_temp[0] == station_][3])[0] / 100)

                        index_month_station['Station_id'].extend(station_all)
                        index_month_station['Index'].extend([index_ for _ in range(len(station_all))])
                        index_month_station['Month'].extend([month_ for _ in range(len(station_all))])
                index_month_station = pd.DataFrame(index_month_station)
                index_month_station.to_csv(os.path.join(self.metadata_folder, f'{index_}.csv'))
            else:
                index_month_station = pd.read_csv(os.path.join(self.metadata_folder, f'{index_}.csv'))
            self.date_range[index_] = pd.unique(index_month_station['Month'])

        if not os.path.exists(os.path.join(self.metadata_folder, f'station_inform.csv')):
            self.station_inform_df = pd.DataFrame(station_inform_dic)
            self.station_inform_df.to_csv(os.path.join(self.metadata_folder, f'station_inform.csv'))
        else:
            self.station_inform_df = pd.read_csv(os.path.join(self.metadata_folder, f'station_inform.csv'))

        self.csv_files = csv_files

        # Define cache folder
        # self.cache_folder, self.trash_folder = self.work_env + 'cache\\', self.work_env + 'trash\\'
        # bf.create_folder(self.cache_folder)
        # bf.create_folder(self.trash_folder)

        # Create output path
        self.output_path, self.log_filepath = f'{os.path.dirname(self.work_env)}\\CMA_OUTPUT\\',  f'{self.work_env}logfile\\'
        bf.create_folder(self.output_path)
        bf.create_folder(self.log_filepath)
        # bf.create_folder(self.shpfile_path)

        # column_name_list = ['None' for _ in range(len(df_temp.columns))]
        # for column_ in range(len(df_temp.columns)):
        #     if column_ < len(header_):
        #         column_name_list[column_] = header_[column_]
        #     elif column_ in [_[0] for _ in self._index_dic[index_]]:
        #         pos = [_[0] for _ in self._index_dic[index_]].index(column_)
        #         column_name_list[column_] = self._index_dic[index_][pos][2]
        #     else:
        #         column_name_list[column_] = 'None'
        # df_temp.columns = column_name_list

    def merge_with_NCEI_ds(self, NCEI_dataset):
        pass

    def anusplin(self, ROI, mask, DEM, index=None, date_range=None, cell_size=None, output_path=None, bulk=True):
        """
        :param ROI: ROI is used as CropContent in gdal.Warp to extract the ANUSPLIN_processed climatology data. It should
        be under the same coordinate system with the mask and DEM
        :param mask: Mask is used in the ANUSPLIN-LAPGRD program to extract the climatology data. It should be converted
        into the ASCII TXT file and share same bounds, cell size and coordinate system with the DEM.
        :param DEM: DEM is treated as the covariant in the ANUSPLINE. It should be converted into the ASCII TXT file and
        share same bounds, cell size and coordinate system with the mask.
        :param index: The index of climatology data for processing
        :param date_range: Start YYYYDOY - End YYYYDOY
        :param cell_size: Output cellsize
        :param output_path:
        :param bulk:
        """

        # Check the ANUSPLIN program
        if 'splina.exe' not in os.listdir(os.getcwd()) or 'lapgrd.exe' not in os.listdir(os.getcwd()):
            raise Exception('The splina.exe or lapgrd.exe was missing!')
        else:
            self._splina_program = os.path.join(os.getcwd(), 'splina.exe')
            self._lapgrd_program = os.path.join(os.getcwd(), 'lapgrd.exe')

        # Identify the DEM bounds
        if DEM.endswith('.txt') or DEM.endswith('.TXT'):
            with open(DEM, 'r') as f:
                dem_content = f.read()
                dem_content = dem_content.split('\n')[0:6]
                dem_content = [(str(_.split(' ')[0]), float(_.split(' ')[-1])) for _ in dem_content]
                dem_bound = [dem_content[2][1], dem_content[3][1] + dem_content[1][1] * dem_content[4][1],
                             dem_content[2][1] + dem_content[0][1] * dem_content[4][1], dem_content[3][1]]
                dem_cellsize = dem_content[4][1]
                f.close()
        else:
            raise TypeError('Please convert the dem into txt file')

        # Identify the mask bounds
        if mask.endswith('.txt') or mask.endswith('.TXT'):
            with open(mask, 'r') as f:
                mask_content = f.read()
                mask_content = mask_content.split('\n')[0:6]
                mask_content = [(str(_.split(' ')[0]), float(_.split(' ')[-1])) for _ in mask_content]
                mask_bound = [mask_content[2][1], mask_content[3][1] + mask_content[1][1] * mask_content[4][1],
                              mask_content[2][1] + mask_content[0][1] * mask_content[4][1], mask_content[3][1]]
                mask_cellsize = mask_content[4][1]
                f.close()
        else:
            raise TypeError('Please convert the MASK into ASCII txt file')

        # Check the consistency between the mask amd DEM
        if mask_bound != dem_bound:
            raise ValueError('The DEM and mask share inconsistency bound')
        else:
            bounds = copy.deepcopy(mask_bound)
        if mask_cellsize != dem_cellsize:
            raise ValueError('The DEM and mask share inconsistency bound')
        else:
            if cell_size is None:
                cell_size = copy.deepcopy(mask_cellsize)
            else:
                if cell_size <= mask_cellsize and mask_cellsize / cell_size == int(mask_cellsize / cell_size):
                    cell_size = copy.deepcopy(mask_cellsize)
                else:
                    raise Exception('Please refactor the cell size!')

        # Identify the ROI
        if isinstance(ROI, str) and os.path.exists(ROI) and (ROI.endswith('.shp') or ROI.endswith('.SHP')):
            driver = ogr.GetDriverByName('ESRI Shapefile')
            datasource = driver.Open(ROI, 0)
            layer = datasource.GetLayer(0)
            spatial_ref = layer.GetSpatialRef()
            bounds_crs = 'EPSG:' + spatial_ref.GetAttrValue("AUTHORITY", 1)
            extent = layer.GetExtent()
            bounds = [extent[0], extent[3], extent[1], extent[2]]
        elif isinstance(ROI, str) and os.path.exists(ROI) and (ROI.endswith('.tif') or ROIy.endswith('.TIF')):
            ROI_type = 'ras'
            bounds = bf.get_tif_border(ROI)
            ds = gdal.Open(ROI)
            proj = osr.SpatialReference(wkt=ds.GetProjection())
            bounds_crs = 'EPSG:' + proj.GetAttrValue('AUTHORITY', 1)
            ds = None
        else:
            raise TypeError('The input ROI is under wrong type!')

        # Check the consistency between ROI with mask and DEM


        # Identify the index
        if index is None:
            index = self.index
        elif isinstance(index, list):
            index = [index_ for index_ in index if index_ in self.index]
        elif isinstance(index, str):
            if index not in self.index:
                raise Exception(f'The {str(index)} is not supported!')
        else:
            raise TypeError('The input index is under wrong type!')

        # Identify the date_range
        if date_range is None:
            date_range = {key: self.date_range[key] for key in self.date_range if key in index}
        elif isinstance(date_range, list) and len(date_range) == 2:
            try:
                start_date, end_date = int(date_range[0]), int(date_range[1])
                start_month, end_month = int(start_date // 100), int(end_date // 100)
                if start_date > end_date:
                    raise ValueError('The end date is smaller than start date!')
                else:
                    date_range = {key: self.date_range[key] for key in self.date_range if key in index}
                    for _ in date_range.keys():
                        date_range_ = [date_ for date_ in date_range[_] if start_month <= date_ <= end_month]
                        date_range[_] = date_range_
            except:
                print(traceback.format_exc())
                raise Exception('The input date type is not correct')
        else:
            raise TypeError('The input date range is under wrong type!')

        # Check the output path
        if output_path is None:
            self.splin_path = self.output_path + 'ANUSPLIN_SPLIN\\'
            self.lapgrd_path = self.output_path + 'ANUSPLIN_LAPGRD\\'
        else:
            if not os.path.exists(output_path):
                raise ValueError('The output path does not exist')
            else:
                self.splin_path = os.path.join(self.output_path, 'ANUSPLIN_SPLIN\\')
                self.lapgrd_path = os.path.join(self.work_env, 'ANUSPLIN_LAPGRD\\')
        bf.create_folder(self.splin_path)
        bf.create_folder(self.lapgrd_path)

        for index_ in index:
            index_date_range = date_range[index_]
            # if bulk is True:
            #     with concurrent.futures.ProcessPoolExecutor() as exe:
            #         exe.map(self.execute_splin, repeat(index_), index_date_range, repeat(self.splin_path), repeat(bounds), repeat(bounds_crs))
            # else:
            #     for index_month in index_date_range:
            #         self.execute_splin(index_, index_month, self.splin_path, bounds, bounds_crs)

            self.execute_lapgrd(index_, index_date_range[0], DEM, ROI, self.splin_path, self.lapgrd_path, bounds_crs)
            if bulk is True:
                with concurrent.futures.ProcessPoolExecutor() as exe:
                    exe.map(self.execute_lapgrd, repeat(index_), index_date_range, repeat(DEM), repeat(ROI), repeat(self.splin_path), repeat(self.lapgrd_path), repeat(bounds_crs))
            else:
                for index_month in index_date_range:
                    self.execute_lapgrd(index_, index_month, DEM, ROI, self.splin_path, self.lapgrd_path, bounds_crs)

    def execute_splin(self, index, year_month, output_path, output_bounds, output_crs):

        # Identify the output path
        if not os.path.exists(output_path):
            raise ValueError(f'The {output_path} is not valid')
        index_output_path = os.path.join(output_path, f'{index}\\')
        bf.create_folder(index_output_path)

        # Copy the programme
        index_splina = os.path.join(index_output_path, 'splina.exe')
        if not os.path.exists(index_splina):
            shutil.copy(self._splina_program, index_splina)

        csv_file = [csv_ for csv_ in self.csv_files if str(year_month) in csv_ and index in csv_]
        df_name_list = ['Station_id', 'Lat', 'Lon', 'Alt', 'DOY']
        df_name_list.extend([__[2] for __ in self._index_dic[index]])
        df_temp = pd.read_table(csv_file[0], delim_whitespace=True, header=None)
        column_name_all = [_[2] for _ in self._index_dic[index]]
        column_name_list = ['None' for _ in range(len(df_temp.columns))]
        for column_ in range(len(df_temp.columns)):
            if column_ < len(self._cma_header_):
                column_name_list[column_] = self._cma_header_[column_]
            elif column_ in [_[0] for _ in self._index_dic[index]]:
                pos = [_[0] for _ in self._index_dic[index]].index(column_)
                column_name_list[column_] = self._index_dic[index][pos][2]
            else:
                column_name_list[column_] = 'None'
        df_temp.columns = column_name_list

        # Process DOY
        doy_list = []
        for row in range(df_temp.shape[0]):
            doy_list.append(df_temp['YYYY'][row] * 1000 + datetime.date(year=df_temp['YYYY'][row], month=df_temp['MM'][row], day=df_temp['DD'][row]).toordinal()
                            - datetime.date(year=df_temp['YYYY'][row], month=1, day=1).toordinal() + 1)
        df_temp['DOY'] = doy_list

        # Process lat lon and alt
        df_temp['Lon'] = df_temp['Lon'] // 100 + (np.mod(df_temp['Lon'], 100) / 60)
        df_temp['Lat'] = df_temp['Lat'] // 100 + (np.mod(df_temp['Lat'], 100) / 60)
        df_temp['Alt'] = df_temp['Alt'].astype(np.float32) / 100

        # Determine the header
        header = ['Station_id', 'Lon', 'Lat', 'Alt', self._interpolate_index[index][2]]

        # Get the geodf itr through date
        doy_list = pd.unique(df_temp['DOY'])
        for doy in doy_list:
            t1 = time.time()
            print(f'Start executing the SPLINA for {str(index)} of \033[1;31m{str(doy)}\033[0m!')
            if not os.path.exists(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.dat')):
                pd_temp = df_temp[df_temp['DOY'] == doy][header]
                pd_temp[self._interpolate_index[index][2]] = pd_temp[self._interpolate_index[index][2]].astype(np.float32)
                pd_temp[self._interpolate_index[index][2]] = pd_temp[self._interpolate_index[index][2]].replace(self._interpolate_index[index][3], np.nan)
                pd_temp[self._interpolate_index[index][2]] = pd_temp[self._interpolate_index[index][2]] * self._interpolate_index[index][1]
                pd_temp = pd_temp.reset_index(drop = True)
                geodf_temp = gp.GeoDataFrame(pd_temp, geometry=[Point(xy) for xy in zip(pd_temp['Lon'], pd_temp['Lat'])], crs='EPSG:4326')

                # Transform into destination crs
                if output_crs != 'EPSG:4326':
                    geodf_temp = geodf_temp.to_crs(crs=output_crs)
                    geodf_temp['Lon'] = [_.coords[0][0] for _ in geodf_temp['geometry']]
                    geodf_temp['Lat'] = [_.coords[0][1] for _ in geodf_temp['geometry']]

                # Use the bounds to extract the point
                geodf_temp = geodf_temp[(output_bounds[2] >= geodf_temp['Lon']) & (geodf_temp['Lon'] >= output_bounds[0])]
                geodf_temp = geodf_temp[(output_bounds[3] <= geodf_temp['Lat']) & (geodf_temp['Lat'] <= output_bounds[1])]
                df_doy= pd.DataFrame(geodf_temp)[header]

                # Format the dat file
                # df_temp['Station_id'] = df_temp['Station_id'].astype(str)
                # df_temp['Station_id'] = df_temp['Station_id'].apply(lambda x: x[:6].ljust(6, ' '))
                # df_temp['Lon'] = df_temp['Lon'].map('{:14.6f}'.format)
                # df_temp['Lat'] = df_temp['Lat'].map('{:14.6f}'.format)
                # df_temp['Alt'] = df_temp['Alt'].map('{:6.2f}'.format)
                # df_temp[self._interpolate_index[index][2]] = df_temp[self._interpolate_index[index][2]].map('{:6.1f}'.format)
                df_doy = df_doy.reset_index(drop=True)

                f = open(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.dat'), 'w')
                for i in range(df_doy.shape[0]):
                    sta, lon, lat, height, val = [df_doy[__][i] for __ in header]
                    text = '{:>5} {:>14.6f} {:>14.6f} {:>6.2f} {:>6.2f}\n'.format(str(sta), lon, lat, height, val)
                    f.write(text)
                f.close()

            file_name = os.path.join(index_output_path, f'{str(index)}_{str(doy)}')
            if not os.path.exists(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.conf')):
                f = open(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.conf'), 'w')
                lon_coord = f'{str(output_bounds[0])} {str(output_bounds[2])} 0 1'
                lat_coord = f'{str(output_bounds[3])} {str(output_bounds[1])} 0 1'

                f.writelines(self._splin_conf.format(file_name, lon_coord, lat_coord, file_name,file_name,file_name,file_name,file_name,file_name))
                f.close()

            if not os.path.exists(file_name + '.sur'):
                cmd_ = index_splina + ' <{}> {}.log'.format(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.c onf'), os.path.join(index_output_path, f'{str(index)}_{str(doy)}'))
                os.system(cmd_)
            print(f'Finish executing the SPLINA procedure for {str(index)} of \033[1;31m{str(doy)}\033[0m in \033[1;34m{str(time.time() - t1)[0:7]}\033[0m s')

    def execute_lapgrd(self, index, year_month, demfile, ROI, splin_path, output_path, crs):

        # Identify the output path
        if not os.path.exists(output_path):
            raise ValueError(f'The {output_path} is not valid')
        index_output_path = os.path.join(output_path, f'{index}\\')
        splin_path = os.path.join(splin_path, f'{index}\\')
        bf.create_folder(index_output_path)

        # Copy the programme
        index_lapgrd = os.path.join(index_output_path, 'lapgrd.exe')
        if not os.path.exists(index_lapgrd):
            shutil.copy(self._lapgrd_program, index_lapgrd)

        csv_file = [csv_ for csv_ in self.csv_files if str(year_month) in csv_ and index in csv_]
        df_name_list = ['Station_id', 'Lat', 'Lon', 'Alt', 'DOY']
        df_name_list.extend([__[2] for __ in self._index_dic[index]])
        df_temp = pd.read_table(csv_file[0], delim_whitespace=True, header=None)
        column_name_all = [_[2] for _ in self._index_dic[index]]
        column_name_list = ['None' for _ in range(len(df_temp.columns))]
        for column_ in range(len(df_temp.columns)):
            if column_ < len(self._cma_header_):
                column_name_list[column_] = self._cma_header_[column_]
            elif column_ in [_[0] for _ in self._index_dic[index]]:
                pos = [_[0] for _ in self._index_dic[index]].index(column_)
                column_name_list[column_] = self._index_dic[index][pos][2]
            else:
                column_name_list[column_] = 'None'
        df_temp.columns = column_name_list

        # Process DOY
        doy_list = []
        for row in range(df_temp.shape[0]):
            doy_list.append(df_temp['YYYY'][row] * 1000 + datetime.date(year=df_temp['YYYY'][row], month=df_temp['MM'][row], day=df_temp['DD'][row]).toordinal()
                            - datetime.date(year=df_temp['YYYY'][row], month=1, day=1).toordinal() + 1)
        df_temp['DOY'] = doy_list

        # Process DEM
        if demfile.endswith('.txt') or demfile.endswith('.TXT'):
            with open(demfile, 'r') as f:
                dem_content = f.read()
                dem_content = dem_content.split('\n')[0:6]
                dem_content = [(str(_.split(' ')[0]), float(_.split(' ')[-1])) for _ in dem_content]
                bounds_dem = [dem_content[2][1], dem_content[3][1] + dem_content[1][1] * dem_content[4][1], dem_content[2][1] + dem_content[0][1] * dem_content[4][1], dem_content[3][1]]
                cellsize = dem_content[4][1]
                nodata = dem_content[5][1]
                f.close()
        else:
            raise TypeError('Please convert the dem into txt file')

        # Get the geodf itr through date
        doy_list = pd.unique(df_temp['DOY'])
        for doy in doy_list:
            t1 = time.time()
            print(f'Start executing the LAPGRD for {str(index)} of \033[1;31m{str(doy)}\033[0m!')
            if (len(bf.file_filter(splin_path, [str(index), str(doy), '.cov'], and_or_factor='and')) < 1 or
                    len(bf.file_filter(splin_path, [str(index), str(doy), '.sur'], and_or_factor='and')) < 1):
                raise Exception('.cov file for {str(index)} of \033[1;31m{str(doy)}\033[0m is missing!')

            output_grd = os.path.join(index_output_path, f'{str(index)}_{str(doy)}.grd')
            output_res = os.path.join(index_output_path, f'{str(index)}_{str(doy)}_res.grd')
            # Generate the conf file
            if not os.path.exists(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.conf')):
                f = open(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.conf'), 'w')
                sur_file = bf.file_filter(splin_path, [str(index), str(doy), '.sur'], and_or_factor='and')[0]
                cov_file = bf.file_filter(splin_path, [str(index), str(doy), '.cov'], and_or_factor='and')[0]
                with open(sur_file, 'r') as surf_f:
                    surf_content = surf_f.read()
                    surf_content = surf_content.split('\n')[2:4]
                    if surf_content[0].startswith(' '):
                        x_limit = f"{surf_content[0].split(' ')[2]} {surf_content[0].split(' ')[4]} {cellsize}"
                    else:
                        x_limit = f"{surf_content[0].split(' ')[1]} {surf_content[0].split(' ')[3]} {cellsize}"

                    if surf_content[1].startswith(' '):
                        y_limit = f"{surf_content[1].split(' ')[2]} {surf_content[1].split(' ')[4]} {cellsize}"
                    else:
                        y_limit = f"{surf_content[1].split(' ')[1]} {surf_content[1].split(' ')[3]} {cellsize}"

                f.writelines(self._lapgrd_conf.format(sur_file, cov_file, x_limit, y_limit, ROI, demfile, nodata, output_grd, '(100f10.3)', nodata, output_res, '(1f8.2)'))
                f.close()

            # Execute the lapgrd
            if not os.path.exists(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.TIF')):
                cmd_ = index_lapgrd + ' <{}> {}.log'.format(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.conf'), os.path.join(index_output_path, f'{str(index)}_{str(doy)}'))
                os.system(cmd_)

                # Execute the lapgrd
                temp_ds2 = gdal.Warp('/vsimem/' + f'{str(index)}_{str(doy)}_temp.vrt', os.path.join(index_output_path, f'{str(index)}_{str(doy)}.grd'), srcSRS=crs,
                                     resampleAlg=gdal.GRA_NearestNeighbour,  cropToCutline=True, cutlineDSName='G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020.shp',
                                     outputType=gdal.GDT_Float32, dstNodata=np.nan)
                temp_ds3 = gdal.Translate(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.TIF'), f'/vsimem/{str(index)}_{str(doy)}_temp.vrt', options=topts)
                temp_ds2 = None
                temp_ds3 = None

            try:
                os.remove(output_res)
                os.remove(output_grd)
            except:
                print('Failed to delete file')
            print(f'Finish executing the SPLINA procedure for {str(index)} of \033[1;31m{str(doy)}\033[0m in \033[1;34m{str(time.time() - t1)[0:7]}\033[0m s')

    def generate_climate_ras(self, ROI=None, index=None, date_range=None, output_path=None, bulk=True, generate_shp=True, generate_ras=True):

        # Generate climate raster
        # Identify the index
        if index is None:
            index = self.index
        elif isinstance(index, list):
            index = [index_ for index_ in index if index_ in self.index]
        elif isinstance(index, str):
            if index not in self.index:
                raise Exception(f'The {str(index)} is not supported!')
        else:
            raise TypeError('The input index is under wrong type!')

        # Identify the ROI
        spa_bounds = [np.min(self.station_inform_df['Lat']), np.max(self.station_inform_df['Lon']),
                      np.max(self.station_inform_df['Lat']), np.min(self.station_inform_df['Lon'])]
        if ROI is None:
            ROI_type = None

            bounds = [np.min(self.station_inform_df['Lat']), np.max(self.station_inform_df['Lon']),
                      np.max(self.station_inform_df['Lat']), np.min(self.station_inform_df['Lon'])]
            Res = min(np.ceil(np.abs(bounds[0][0] - bounds[1][0]) / (2000 * 0.01)) * 0.01, np.ceil(np.abs(bounds[0][1] - bounds[1][1]) / (2000 * 0.01)) * 0.01)
            bounds[1][0] = bounds[0][0] + np.ceil(np.abs(bounds[1][0] - bounds[0][0]) / Res) * Res
            bounds[1][1] = bounds[0][1] - np.ceil(np.abs(bounds[1][1] - bounds[0][1]) / Res) * Res
            ROI_inform = [None, 'entire', 'EPSG:4369', Res, bounds, spa_bounds]

        elif isinstance(ROI, str) and os.path.exists(ROI) and (ROI.endswith('.shp') or ROI.endswith('.SHP')):
            ROI_type = 'shp'

            bounds = [np.min(self.station_inform_df['Lat']), np.max(self.station_inform_df['Lon']),
                      np.max(self.station_inform_df['Lat']), np.min(self.station_inform_df['Lon'])]
            Res = int(min(np.ceil(np.abs(bounds[0][0] - bounds[1][0]) / 20000),
                          np.ceil(np.abs(bounds[0][1] - bounds[1][1]) / 20000)))
            wid_height = []

            bounds[1][0] = bounds[0][0] + np.ceil(np.abs(bounds[1][0] - bounds[0][0]) / Res) * Res
            bounds[1][1] = bounds[0][1] - np.ceil(np.abs(bounds[1][1] - bounds[0][1]) / Res) * Res
            driver = ogr.GetDriverByName('ESRI Shapefile')
            datasource = driver.Open(ROI, 0)
            layer = datasource.GetLayer(0)
            spatial_ref = layer.GetSpatialRef()
            epsg_crs = 'EPSG:' + spatial_ref.GetAttrValue("AUTHORITY", 1)
            ROI_inform = [ROI, ROI.split('\\')[-1].split('.')[0], epsg_crs, wid_height, bounds, spa_bounds]

        elif isinstance(ROI, str) and os.path.exists(ROI) and (ROI.endswith('.tif') or ROI.endswith('.TIF')):
            ROI_type = 'ras'
            bounds = bf.get_tif_border(ROI)
            ds = gdal.Open(ROI)
            wid_height = [ds.RasterXSize, ds.RasterYSize]
            proj = osr.SpatialReference(wkt=ds.GetProjection())
            epsg_crs = 'EPSG:' + proj.GetAttrValue('AUTHORITY', 1)
            ds = None
            ROI_inform = [ROI, ROI.split('\\')[-1].split('.')[0], epsg_crs, wid_height, bounds, spa_bounds]
            spa_bounds = [bounds[0] - (bounds[2] - bounds[0]) / 2,  bounds[1] - (bounds[1] - bounds[3]) / 2,
                          bounds[2] + (bounds[2] - bounds[0]) / 2,  bounds[1] + (bounds[1] - bounds[3]) / 2]
        else:
            raise TypeError('The input ROI is under wrong type!')

        # Identify the date_range
        if date_range is None:
            date_range = {key: self.date_range[key] for key in self.date_range if key in index}
        elif isinstance(date_range, list) and len(date_range) == 2:
            try:
                start_date, end_date = int(date_range[0]), int(date_range[1])
                start_month, end_month = int(start_date // 100), int(end_date // 100)
                if start_date > end_date:
                    raise ValueError('The end date is smaller than start date!')
                else:
                    date_range = {key: self.date_range[key] for key in self.date_range if key in index}
                    for _ in date_range.keys():
                        date_range_ = [date_ for date_ in date_range[_] if start_month <= date_ <= end_month]
                        date_range[_] = date_range_
            except:
                print(traceback.format_exc())
                raise Exception('The input date type is not correct')
        else:
            raise TypeError('The input date range is under wrong type!')

        # Check the output path
        if output_path is None:
            self.raster_output_path = self.output_path + 'Raster\\'
            self.shp_output_path = self.output_path + 'Shpfile\\'
        else:
            if not os.path.exists(output_path):
                raise ValueError('The output path does not exist')
            else:
                self.raster_output_path = os.path.join(self.output_path, 'Raster\\')
                self.shp_output_path = os.path.join(self.work_env, 'Shpfile\\')
        bf.create_folder(self.raster_output_path)
        bf.create_folder(self.shp_output_path)

        for index_ in index:
            # Generate the shpfile
            index_date_range = date_range[index_]
            if generate_shp:
                if bulk:
                    with concurrent.futures.ProcessPoolExecutor() as exe:
                        exe.map(self._generate_shpfile, index_date_range, repeat(index_), repeat(self.shp_output_path))
                else:
                    for date_ in index_date_range:
                        self._generate_shpfile(date_, index, self.shp_output_path)

            # Generate the raster file
            if generate_ras:
                shpfile_list = bf.file_filter(os.path.join(self.shp_output_path, f'{index_}\\'), ['.shp'])
                shpfile_valid_list = [shp_ for shp_ in shpfile_list if int(np.floor(bf.doy2date(int(shp_.split(f'{index_}_')[-1].split('.')[0])) / 100)) in index_date_range]
                zvalue = [_[2] for _ in self._index_dic[index_]]

                if bulk:
                    with concurrent.futures.ProcessPoolExecutor() as exe:
                        exe.map(self._shp2ras, shpfile_valid_list, repeat(zvalue), repeat(ROI_inform), repeat(self.raster_output_path))
                else:
                    for shpfile_ in shpfile_valid_list:
                        self._shp2ras(shpfile_, zvalue, ROI_inform, self.raster_output_path)

                # Remove tif redundant
                cachepath = os.path.join(self.raster_output_path + 'cache\\')
                for filename in os.listdir(cachepath):
                    file_path = os.path.join(cachepath, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)  # This will remove the file or link
                        elif os.path.isdir(file_path):
                            os.rmdir(file_path)  # This will remove the directory (if it's empty)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')

    def _generate_shpfile(self, index_year_month, index, outputpath):

        try:
            # Check the output path
            if not os.path.exists(outputpath):
                raise ValueError(f'The {outputpath} is not valid')
            index_output_path = os.path.join(outputpath, f'{index}\\')
            bf.create_folder(index_output_path)

            csv_file = [csv_ for csv_ in self.csv_files if str(index_year_month) in csv_ and index in csv_]
            if len(csv_file) != 1:
                raise Exception('Code error!')

            df_name_list = ['Station_id', 'Lat', 'Lon', 'Alt', 'DOY']
            df_name_list.extend([__[2] for __ in self._index_dic[index]])
            df_temp = pd.read_table(csv_file[0], delim_whitespace=True, header=None)
            column_name_all = [_[2] for _ in self._index_dic[index]]
            column_name_list = ['None' for _ in range(len(df_temp.columns))]
            for column_ in range(len(df_temp.columns)):
                if column_ < len(self._cma_header_):
                    column_name_list[column_] = self._cma_header_[column_]
                elif column_ in [_[0] for _ in self._index_dic[index]]:
                    pos = [_[0] for _ in self._index_dic[index]].index(column_)
                    column_name_list[column_] = self._index_dic[index][pos][2]
                else:
                    column_name_list[column_] = 'None'
            df_temp.columns = column_name_list

            # Process DOY
            doy_list = []
            for row in range(df_temp.shape[0]):
                doy_list.append(df_temp['YYYY'][row] * 1000 + datetime.date(year=df_temp['YYYY'][row],
                                                                            month=df_temp['MM'][row],
                                                                            day=df_temp['DD'][row]).toordinal()
                                - datetime.date(year=df_temp['YYYY'][row], month=1, day=1).toordinal() + 1)
            df_temp['DOY'] = doy_list

            # Process lat lon and alt
            df_temp['Lon'] = df_temp['Lon'] // 100 + (np.mod(df_temp['Lon'], 100) / 60)
            df_temp['Lat'] = df_temp['Lat'] // 100 + (np.mod(df_temp['Lat'], 100) / 60)
            df_temp['Alt'] = df_temp['Alt'].astype(np.float32) / 100

            # Determine the header
            header = ['Station_id', 'Lon', 'Lat', 'Alt', 'DOY']
            for _ in self._index_dic[index]:
                header.append(_[2])

            # Get the geodf itr through date
            doy_list = pd.unique(df_temp['DOY'])
            for doy in doy_list:
                t1 = time.time()
                print(f'Start processing the {str(index)} data of \033[1;31m{str(doy)}\033[0m')
                if not os.path.exists(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.shp')):
                    pd_temp = df_temp[df_temp['DOY'] == doy][header]
                    for _ in self._index_dic[index]:
                        pd_temp[_[2]] = pd_temp[_[2]].astype(np.float32)
                        pd_temp[_[2]] = pd_temp[_[2]].replace(_[3], np.nan)
                        pd_temp[_[2]] = pd_temp[_[2]] * _[1]
                    geodf_temp = gp.GeoDataFrame(pd_temp, geometry=gp.points_from_xy(pd_temp['Lon'], pd_temp['Lat']), crs="EPSG:4326")
                    geodf_temp.to_file(os.path.join(index_output_path, f'{str(index)}_{str(doy)}.shp'), encoding='gbk')
                print(f'Finish generating the {str(index)} shpfile of \033[1;31m{str(doy)}\033[0m in \033[1;34m{str(time.time() - t1)[0:7]}\033[0m s')
        except:
            print(traceback.format_exc())

    def _shp2ras(self, shpfile: str, zvalue_list: list, ROI: list, output_path: str):

        # Process the ROI
        ROI_file, ROI_name, crs, width_height, bounds, spa_bounds = ROI
        file_name = shpfile.split('\\')[-1].split('.')[0]

        # Generate the raster
        for z in zvalue_list:
            t1 = time.time()
            bf.create_folder(os.path.join(output_path + f'{ROI_name}_Denv_raster\\IDW_{z}\\'))
            bf.create_folder(os.path.join(output_path + 'cache\\'))

            print(f"Start generating the raster of \033[1;31m{str(file_name)}\033[0m")
            if not os.path.exists(output_path + f'{ROI_name}_Denv_raster\\IDW_{z}\\' + file_name + '.TIF'):

                if ROI_file is None:
                    try:
                        temp_ds1 = gdal.Grid('/vsimem/' + file_name + '_temp.vrt', shpfile, zfield=z, outputSRS=crs,
                                             algorithm='invdist:power=2:min_points=4:max_points=15', outputBounds=bounds,
                                             spatFilter=spa_bounds, width=width_height[0], height=width_height[1], outputType=gdal.GDT_Float32,
                                             noData=np.nan)
                        temp_ds1 = None
                    except:
                        raise Exception(traceback.format_exc())

                    try:
                        temp_ds3 = gdal.Translate(output_path + f'{ROI_name}_Denv_raster\\IDW_{z}\\' + file_name + '.TIF',
                                                  '/vsimem/' + file_name + '_temp.vrt', options=topts)
                        temp_ds3 = None
                    except:
                        raise Exception(traceback.format_exc())
                else:
                    try:
                        st = time.time()
                        gdf = gp.read_file(shpfile)
                        crs_ori = 'EPSG:' + str(gdf.crs.to_epsg())
                        if crs_ori != crs:
                            gdf = gdf.to_crs(crs)
                            gdf.to_file(shpfile)
                            gdf = None
                        temp_ds1 = gdal.Grid('/vsimem/' + file_name + z + '.tif', shpfile, zfield=z, outputSRS=crs,
                                             algorithm='invdist:power=2:min_points=4:max_points=10', outputBounds=bounds,
                                             width=width_height[0], height=width_height[1], outputType=gdal.GDT_Float32,
                                             noData=np.nan)
                        temp_ds1 = None
                        print(f'Grid consumes {str(time.time() - st)} s')
                    except:
                        raise Exception(traceback.format_exc())

                    try:
                        st = time.time()
                        roi_ds = gdal.Open(ROI[0])
                        temp_ds2 = gdal.Warp('/vsimem/' + file_name + '_temp.vrt', '/vsimem/' + file_name + z + '.tif',
                                             resampleAlg=gdal.GRA_NearestNeighbour,  width=width_height[0], height=width_height[1],
                                             cropToCutline=True, cutlineDSName='G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020.shp', outputType=gdal.GDT_Float32,
                                             dstNodata=np.nan)
                        print(f'Warp consumes {str(time.time() - st)} s')

                        st = time.time()
                        temp_ds3 = gdal.Translate(output_path + f'{ROI_name}_Denv_raster\\IDW_{z}\\' + file_name + '.TIF',
                                                  '/vsimem/' + file_name + '_temp.vrt', options=topts)
                        print(f'Translate consumes {str(time.time() - st)} s')
                        temp_ds2, temp_ds3 = None, None
                    except:
                        raise Exception(traceback.format_exc())
                gdal.Unlink('/vsimem/' + file_name + '_temp.vrt')
                gdal.Unlink('/vsimem/' + file_name + z + '.tif')

            print(f'Finish generating the {z} raster of \033[1;31m{str(file_name)}\033[0m in \033[1;34m{str(time.time() - t1)[0:7]}\033[0m s')

        # try:
        #     for file in bf.file_filter(output_path + 'cache\\', ['.TIF']):
        #         os.remove(file)
        # except:
        #     pass

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
            for func_key, func_processing_name in zip(['ds2pointshp', 'ds2raster', 'raster2dc'], ['2point', '2raster', 'rs2dc']):
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
    def ds2pointshp(self, zvalue_list: list, output_path: str):

        output_path = bf.Path(output_path).path_name
        bf.create_folder(output_path)

        for z in zvalue_list:
            if z not in self.valid_inform:
                raise ValueError(f'The zvalue {str(z)} is not valid!')

        index_all = ''
        for index in zvalue_list:
            index_all = f'{index_all}_{str(index)}'

        basic_inform4point = copy.copy(zvalue_list)
        for z in ['LATITUDE', 'LONGITUDE', 'STATION', 'DATE']:
            if z not in basic_inform4point:
                basic_inform4point.append(z)

        if not isinstance(main_coordinate_system, str):
            raise TypeError(f'Please input the {main_coordinate_system} as a string!')
        else:
            self.main_coordinate_system = main_coordinate_system

        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._ds2point, self.year_range, repeat(output_path), repeat(basic_inform4point), repeat(index_all), repeat(self.main_coordinate_system))

    def _ds2point(self, year, output_path, z_4point, index_all, crs):

        z_dic = {}
        current_year_files_content = self.files_content_dic[year]
        for date_temp in range(0, datetime.date(year, 12, 31).toordinal() - datetime.date(year, 1, 1).toordinal() + 1):

            date = datetime.datetime.fromordinal(datetime.date(year, 1, 1).toordinal() + date_temp)
            t1 = time.time()
            print(f'Start processing the climatology data of \033[1;31m{str(datetime.date.strftime(date, "%Y%m%d"))}\033[0m')
            if not os.path.exists(f'{output_path}{str(datetime.date.strftime(date, "%Y%m%d"))}{index_all}.shp'):

                for z in z_4point:
                    z_dic[z] = []

                for current_year_file_content in current_year_files_content:
                    for z in z_4point:
                        try:
                            if z == 'TEMP':
                                z_dic[z].append(int(current_year_file_content[current_year_file_content['DATE'] == datetime.date.strftime(date, "%Y-%m-%d")][z].values[0] * 10))
                            else:
                                z_dic[z].append(current_year_file_content[current_year_file_content['DATE'] == datetime.date.strftime(date, "%Y-%m-%d")][z].values[0])
                        except:
                            print(f"The {z} data of {str(date)} in STATION:{str(current_year_file_content.loc[0, 'STATION'])} was missing!!")

                geodf_temp = gp.GeoDataFrame(z_dic, geometry=gp.points_from_xy(z_dic['LONGITUDE'], z_dic['LATITUDE']), crs="EPSG:4326")
                geodf_temp = geodf_temp.to_crs(crs)

                if geodf_temp.size == 0:
                    print(f'There has no valid file for date \033[1;31m{str(datetime.date.strftime(date, "%Y%m%d"))}\033[0m')
                else:
                    geodf_temp.to_file(f'{output_path}{str(datetime.date.strftime(date, "%Y%m%d"))}{index_all}.shp', encoding='gbk')
                    print(f'Finish generating the shpfile of \033[1;31m{str(datetime.date.strftime(date, "%Y%m%d"))}\033[0m in \033[1;34m{str(time.time()-t1)[0:7]}\033[0m s')

    @save_log_file
    def ds2raster(self, zvalue_list: list, raster_size=None, ds2ras_method=None, bounds=None, ROI=None, crs=None):

        # Process ds2raster para
        if isinstance(zvalue_list, str):
            zvalue_list = [zvalue_list]
        elif not isinstance(zvalue_list, list):
            raise TypeError('The zvalue should be a list')

        if ds2ras_method is None:
            self._ds2raster_method = 'IDW'
        elif ds2ras_method not in self._ds2raster_method_tup:
            raise ValueError(f'The {ds2ras_method} is not supported for ds2raster!')
        else:
            self._ds2raster_method = ds2ras_method

        if isinstance(ROI, str):
            if not ROI.endswith('.shp'):
                raise TypeError(f'The ROI should be a valid shpfile!')
            else:
                self.ROI = ROI
                self.ROI_name = self.ROI.split('\\')[-1].split('.')[0]
        else:
            raise TypeError(f'The ROI should be a valid shpfile!')

        if isinstance(bounds, tuple):
            if len(bounds) != 4 and False in [type(temp) in [float, np.float, int, np.int16] for temp in bounds]:
                raise TypeError(f'bounds should be under the tuple type with num-coord in it!')
        elif bounds is not None:
            raise TypeError(f'bounds should be under the tuple type!')

        if raster_size is None and bounds is not None:
            raster_size = [int((bounds[3] - bounds[1]) / 10), int((bounds[2] - bounds[0]) / 10)]
        elif isinstance(raster_size, list) and len(raster_size) == 2:
            raster_size = raster_size
        elif raster_size is not None:
            raise TypeError(f'raster size should under the list type!')

        if crs is not None:
            self.main_coordinate_system = crs
        else:
            self.main_coordinate_system = 'EPSG:32649'

        # Create the point shpfiles
        shpfile_folder = self.output_path + 'Ori_shpfile\\'
        bf.create_folder(shpfile_folder)
        self.ds2pointshp(zvalue_list, shpfile_folder, self.main_coordinate_system)

        for zvalue_temp in zvalue_list:

            # Generate output bounds based on the point bounds
            if bounds is None:
                shp_temp = ogr.Open(bf.file_filter(shpfile_folder, ['.shp'])[0])
                bounds = shp_temp.GetLayer(1).GetExtent()
                bounds = (bounds[0], bounds[1], bounds[2], bounds[3])

            if raster_size is None:
                raster_size = [int((bounds[3] - bounds[1]) / 10), int((bounds[2] - bounds[0]) / 10)]

            # Retrieve all the point shpfiles
            shpfiles = bf.file_filter(shpfile_folder, ['.shp'])
            if shpfiles == []:
                raise ValueError(f'There are no valid shp files in the {str(shpfile_folder)}!')

            # Generate the raster
            if ds2ras_method == 'IDW':

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    executor.map(shp2raster_idw, shpfiles, repeat(self.output_path), repeat(zvalue_temp), repeat(raster_size), repeat(bounds), repeat(self.ROI), repeat(self.main_coordinate_system))

            else:
                pass

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
            if self._ds2ras_method is None:
                self._ds2ras_method = 'IDW'
        else:
            raise TypeError('Please mention the ds2ras_method should be supported!')

    @save_log_file
    def raster2dc(self, zvalue_list: list, temporal_division=None, ROI=None, **kwargs):

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

        for zvalue_temp in zvalue_list:

            # Create the output path
            output_path = f'{self.output_path}{self.ROI_name}_Denv_datacube\\' if self.ROI_name is not None else f'{self.output_path}Ori_Denv_datacube\\'
            bf.create_folder(output_path)

            # Obtain the input files
            input_folder = f'{self.output_path}{str(self.ROI_name)}_Denv_raster\\IDW_{zvalue_temp}\\' if self.ROI_name is not None else f'{self.output_path}Ori_Denv_raster\\IDW_{zvalue_temp}\\'
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
                array_temp = array_temp.astype(np.int16)

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

            # Create the metadata dic
            metadata_dic = {'ROI_name': self.ROI_name, 'index': zvalue_temp, 'Datatype': 'float', 'ROI': self.ROI,
                            'ROI_array': ROI_array_name, 'ROI_tif': ROI_tif_name, 'sdc_factor': True,
                            'coordinate_system': self.main_coordinate_system, 'size_control_factor': False,
                            'oritif_folder': input_folder, 'dc_group_list': None, 'tiles': None, 'timescale': temporal_division,
                            'Denv_factor': True}

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
                executor.map(self._raster2sdc, repeat(output_path), repeat(input_folder), time_range, repeat(zvalue_temp), repeat(metadata_dic), repeat(rows), repeat(cols), repeat(_sparse_matrix))

    def _raster2sdc(self, output_path, input_folder, time_temp, zvalue_temp, metadata_dic, rows, cols, _sparse_matrix,):

        start_time = time.time()
        print(f'Start constructing the {str(time_temp)} {str(zvalue_temp)} sdc of {self.ROI_name}.')
        nodata_value = None

        # Create the output path
        yearly_output_path = output_path + str(int(time_temp)) + '\\' if time_temp != 'TIF' else output_path + 'all\\'
        bf.create_folder(yearly_output_path)

        if not os.path.exists(f'{yearly_output_path}doy.npy') or not os.path.exists(f'{yearly_output_path}metadata.json'):

            # Determine the input files
            yearly_input_files = bf.file_filter(input_folder, ['.TIF', '\\' + str(time_temp)], exclude_word_list=['aux'],  and_or_factor='and')

            if nodata_value is None:
                nodata_value = gdal.Open(yearly_input_files[0])
                nodata_value = nodata_value.GetRasterBand(1).GetNoDataValue()

            # Create the doy list
            doy_list = bf.date2doy([int(filepath_temp.split('\\')[-1][0:8]) for filepath_temp in yearly_input_files])

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
                            if not os.path.exists(f"{input_folder}{str(bf.doy2date(doy_list[i]))}_{zvalue_temp}.TIF"):
                                raise Exception(f'The {str(doy_list[i])}_{zvalue_temp} is not properly generated!')
                            else:
                                array_temp = gdal.Open(f"{input_folder}{str(bf.doy2date(doy_list[i]))}_{zvalue_temp}.TIF")
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
                        data_valid_array[i] = [array_temp == nodata_value].all()

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

        print(f'Finish constructing the {str(time_temp)} {str(zvalue_temp)} sdc of {self.ROI_name} in \033[1;31m{str(time.time() - start_time)} s\033[0m.')


if __name__ == '__main__':

    station_rec = [Polygon([(441536.34182, 3457208.02321), (1104536.34182, 3457208.02321), (1104536.34182, 3210608.02321), (441536.34182, 3210608.02321)])]
    geometry = gp.GeoDataFrame({'id':[1]}, geometry=station_rec, crs='EPSG:32649')
    geometry.to_file('G:\\A_Landsat_Floodplain_veg\\ROI_map\\weather_boundary.shp')
    ds_temp = CMA_ds('G:\\A_Landsat_Floodplain_veg\\Climatology_data\\Data_cma\\')
    ds_temp.anusplin('G:\\A_Landsat_Floodplain_veg\\Climatology_data\\Mask\\myr_floodplain.txt', 'G:\\A_Landsat_Floodplain_veg\\Climatology_data\DEM\\\DEM\\alos_dem.txt', index=['TEM'], date_range=[19850101, 20201231], boundary='G:\\A_Landsat_Floodplain_veg\\ROI_map\\weather_boundary.shp',)

    # ds_temp = gdal.Open('G:\\A_GEDI_Floodplain_vegh\\S2_all\\Sentinel2_L2A_Output\\ROI_map\\MYZR_FP_2020_map.TIF')
    # bounds_temp = bf.raster_ds2bounds('G:\\A_GEDI_Floodplain_vegh\\S2_all\\Sentinel2_L2A_Output\\ROI_map\\MYZR_FP_2020_map.TIF')
    # size = [ds_temp.RasterYSize, ds_temp.RasterXSize]
    ds_temp = CMA_ds('G:\\A_Landsat_Floodplain_veg\\Data_cma\\')
    ds_temp.generate_climate_ras(ROI='G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020_map.TIF', index=['TEM'], date_range=[19850101, 20201231], generate_shp=True, bulk=True)
    # ds_temp.ds2raster(['TEMP'], ROI='G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020.shp', raster_size=size, ds2ras_method='IDW', bounds=bounds_temp)
    # ds_temp.raster2dc(['TEMP'], temporal_division='year')