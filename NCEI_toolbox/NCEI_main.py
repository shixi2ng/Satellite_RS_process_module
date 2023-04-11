import shutil
from NDsm import NDSparseMatrix
import numpy as np
import gdal
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
        self._ds2raster_method_tup = ('idw',)
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
                            self.__dict__[para] = float(q.split(para + ':')[-1])
                        except:
                            self.__dict__[para] = q.split(para + ':')[-1]

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
            for func_key, func_processing_name in zip(['ds2pointshp', 'ds2raster_idw', 'raster2dc'], ['2point', '2raster', 'rs2dc']):
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

    @ save_log_file
    def ds2raster(self, zvalue_list: list, raster_size=None, ds2ras_method=None, bounds=None, ROI=None, crs=None):

        # Process ds2raster para
        if isinstance(zvalue_list, str):
            zvalue_list = [zvalue_list]
        elif not isinstance(zvalue_list, list):
            raise TypeError('The zvalue should be a list')

        if ds2ras_method is None:
            self._ds2raster_method = 'idw'
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
        shpfile_folder = self.output_path + 'shpfile\\'
        bf.create_folder(shpfile_folder)
        self.ds2pointshp(zvalue_list, shpfile_folder, self.main_coordinate_system)

        for zvalue_temp in zvalue_list:

            # Create output raster folders
            rasterfile_folder = f'{self.output_path}\\{str(self._ds2raster_method)}_{str(zvalue_temp)}_rsfile\\'
            bf.create_folder(rasterfile_folder)

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
            if ds2ras_method == 'idw':

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    executor.map(shp2raster_idw, shpfiles, repeat(rasterfile_folder), repeat(zvalue_temp), repeat(raster_size), repeat(bounds), repeat(self.ROI), repeat(self.main_coordinate_system))

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
        else:
            raise TypeError('Please mention the ds2ras_method should be supported!')

    @save_log_file
    def raster2dc(self, zvalue_list:list, temporal_division=None, ROI=None, **kwargs):

        kwargs['inherit_from_logfile'] = True
        self._process_raster2dc_para(**kwargs)

        if temporal_division is not None and isinstance(temporal_division, str):
            raise TypeError(f'The {temporal_division} should be a str!')
        elif temporal_division is None:
            temporal_division = 'year'
        elif temporal_division not in self._temporal_div_str:
            raise ValueError(f'The {temporal_division} is not supported!')

        if ROI is not None:
            self.ROI = ROI
            self.ROI_name = self.ROI.split('\\')[-1].split('.')[0]

        for zvalue_temp in zvalue_list:

            # Create the output path
            output_path = f'{self.output_path}\\{zvalue_temp}_datacube\\'
            bf.create_folder(output_path)

            # Obtain the input files
            input_folder = f'{self.output_path}{str(self._ds2raster_method)}_{str(zvalue_temp)}_rsfile\\{self.ROI_name}\\'
            input_files = bf.file_filter(input_folder, ['.TIF'], exclude_word_list=['aux'])
            if self.main_coordinate_system is None:
                self.main_coordinate_system = bf.retrieve_srs(gdal.Open(input_files[0]))

            # Create the ROI map
            roi_map_folder = f'{self.output_path}\\{str(self._ds2raster_method)}_{str(zvalue_temp)}_rsfile\\ROI_map\\'
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
                array_temp = array_temp.astype(np.int16)
                array_temp[array_temp != -32768] = 1
                np.save(ROI_array_name, array_temp)
                ds_temp.GetRasterBand(1).WriteArray(array_temp)
                ds_temp.FlushCache()
                ds_temp = None
            else:
                ds_temp = gdal.Open(ROI_tif_name)
                array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
            cols, rows = array_temp.shape[1], array_temp.shape[0]
            sparsify = np.sum(array_temp == -32768) / (array_temp.shape[0] * array_temp.shape[1])
            _sparse_matrix = True if sparsify > 0.9 else False

            # Create the header dic
            header_dic = {'ROI_name': self.ROI_name, 'index': zvalue_temp, 'Datatype': 'float', 'ROI': self.ROI,
                          'ROI_array': ROI_array_name, 'ROI_tif': ROI_tif_name, 'sdc_factor': True,
                          'coordinate_system': self.main_coordinate_system, 'size_control_factor': False,
                          'oritif_folder': input_folder, 'dc_group_list': None, 'tiles': None}

            if temporal_division == 'year':
                time_range = self.year_range
            elif temporal_division == 'month':
                time_range = []
                for year_temp in self.year_range:
                    for month_temp in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
                        time_range.append(str(year_temp) + str(month_temp))
            elif temporal_division is None:
                time_range = ['TIF']

            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                executor.map(self._raster2sdc, repeat(output_path), repeat(input_folder), time_range, repeat(zvalue_temp), repeat(header_dic), repeat(rows), repeat(cols), repeat(_sparse_matrix))

    def _raster2sdc(self, output_path, input_folder, time_temp, zvalue_temp, header_dic, rows, cols, _sparse_matrix,):
        start_time = time.time()
        print(f'Start constructing the {str(time_temp)} {str(zvalue_temp)} sdc of {self.ROI_name}.')
        # Construct the header dic
        nodata_value = None

        # Create the yearly output path
        yearly_output_path = output_path + str(int(time_temp)) + '\\'
        bf.create_folder(yearly_output_path)

        if not os.path.exists(f'{yearly_output_path}doy.npy') or not os.path.exists(f'{yearly_output_path}header.npy'):
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
                                array_temp = gdal.Open(
                                    f"{input_folder}{str(bf.doy2date(doy_list[i]))}_{zvalue_temp}.TIF")
                                array_temp = array_temp.GetRasterBand(1).ReadAsArray()
                                array_temp[array_temp == nodata_value] = 0

                            sm_temp = sm.csr_matrix(array_temp.astype(np.uint16))
                            array_temp = None
                            data_cube.append(sm_temp, name=doy_list[i])
                            data_valid_array[i] = 1 if sm_temp.data.shape[0] == 0 else 0

                            print(
                                f'Assemble the {str(doy_list[i])} into the sdc using {str(time.time() - t1)[0:5]}s (layer {str(i)} of {str(len(doy_list))})')
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
                    data_cube.save(f'{yearly_output_path}{zvalue_temp}_datacube\\')
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
                            raise Exception(
                                f'The {str(doy_list[i])}_{zvalue_temp} is not properly generated!')
                        else:
                            array_temp = gdal.Open(f"{input_folder}{str(doy_list[i])}_{zvalue_temp}.TIF")
                            array_temp = array_temp.GetRasterBand(1).ReadAsArray()

                        data_cube[:, :, i] = array_temp
                        data_valid_array[i] = [array_temp == nodata_value].all()

                        print(
                            f'Assemble the {str(doy_list[i])} into the sdc using {str(time.time() - t1)[0:5]}s (layer {str(i)} of {str(len(doy_list))})')
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

            # Save the header dic
            header_dic['sparse_matrix'], header_dic['huge_matrix'] = _sparse_matrix, _huge_matrix
            np.save(f'{yearly_output_path}header.npy', header_dic)

        print(f'Finish constructing the {str(time_temp)} {str(zvalue_temp)} sdc of {self.ROI_name} in \033[1;31m{str(time.time() - start_time)} s\033[0m.')


if __name__ == '__main__':
    ds_temp = gdal.Open('G:\A_veg\S2_all\Sentinel2_L2A_Output\ROI_map\\MYZR_FP_2020_map.TIF')
    bounds_temp = bf.raster_ds2bounds('G:\A_veg\S2_all\Sentinel2_L2A_Output\ROI_map\\MYZR_FP_2020_map.TIF')
    size = [ds_temp.RasterYSize, ds_temp.RasterXSize]
    ds_temp = NCEI_ds('G:\A_veg\\NDCI_temperature\\NCEI_19_22\\download\\')
    ds_temp.ds2raster(['TEMP'], ROI='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp', raster_size=size, ds2ras_method='idw', bounds=bounds_temp)
    ds_temp.raster2dc(['TEMP'])