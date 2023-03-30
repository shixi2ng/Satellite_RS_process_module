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

        # key variable
        self.ROI = None
        csv_files = bf.file_filter(file_path, ['.csv'], subfolder_detection=True)
        self.year_range = list(set([int(temp.split('.csv')[0].split('_')[-1]) for temp in csv_files]))
        self.station_list = list(set([int(temp.split('\\')[-1].split('_')[0]) for temp in csv_files]))
        self.files_content_dic = {}
        self.valid_inform = []
        self.index = []

        # Define cache folder
        self.cache_folder = self.work_env + 'cache\\'
        self.trash_folder = self.work_env + 'trash\\'
        bf.create_folder(self.cache_folder)
        bf.create_folder(self.trash_folder)

        # Create output path
        self.output_path = f'{self.work_env}NCEI_Output\\'
        self.shpfile_path = f'{self.work_env}shpfile\\'
        self.log_filepath = f'{self.work_env}logfile\\'
        bf.create_folder(self.output_path)
        bf.create_folder(self.log_filepath)
        bf.create_folder(self.shpfile_path)

        # ds2raster
        self._ds2raster_method_tup = ('idw',)
        self._ds2raster_method = None

        # ras2dc
        self._temporal_div_str = ['year', 'month']

        for year in self.year_range:
            ava_inform_list = []
            self.files_content_dic[year] = []
            current_year_files = bf.file_filter(file_path, ['.csv', str(year)], and_or_factor='and', subfolder_detection=True)

            for csv_file_path in current_year_files:
                df_temp = pd.read_csv(csv_file_path)
                self.files_content_dic[year].append(df_temp)
                ava_inform_list.extend(list(df_temp.keys()))

            self.valid_inform.extend(ava_inform_list)

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
    def ds2pointshp(self, zvalue: list, output_path: str):

        output_path = bf.Path(output_path).path_name
        bf.create_folder(output_path)

        for z in zvalue:
            if z not in self.valid_inform:
                raise ValueError(f'The zvalue {str(z)} is not valid!')

        index_all = ''
        for index in zvalue:
            index_all = f'{index_all}_{str(index)}'

        z_4point = copy.copy(zvalue)
        for z in ['LATITUDE', 'LONGITUDE', 'STATION', 'DATE']:
            if z not in z_4point:
                z_4point.append(z)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._ds2point, self.year_range, repeat(output_path), repeat(z_4point), repeat(index_all))

    def _ds2point(self, year, output_path, z_4point, index_all):

        for date_temp in range(0, datetime.date(year, 12, 31).toordinal() - datetime.date(year, 1, 1).toordinal() + 1):

            date = datetime.datetime.fromordinal(datetime.date(year, 1, 1).toordinal() + date_temp)
            current_year_files_content = self.files_content_dic[year]
            z_dic = {}

            t1 = time.time()
            print(f'Start processing the climatology data of \033[1;31m{str(datetime.date.strftime(date, "%Y_%m_%d"))}\033[0m')
            if not os.path.exists(f'{output_path}\\{str(datetime.date.strftime(date, "%Y_%m_%d"))}_{index_all}.shp'):

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
                            break

                geodf_temp = gp.GeoDataFrame(z_dic, geometry=gp.points_from_xy(z_dic['LONGITUDE'], z_dic['LATITUDE']), crs="EPSG:4326")
                geodf_temp = geodf_temp.to_crs('EPSG:32649')
                if geodf_temp.size == 0:
                    print(f'There has no valid file for date \033[1;31m{str(datetime.date.strftime(date, "%Y_%m_%d"))}\033[0m')
                else:
                    geodf_temp.to_file(f'{output_path}\\{str(datetime.date.strftime(date, "%Y_%m_%d"))}_{index_all}.shp', encoding='gbk')
                    print(f'Finish generating the shpfile of \033[1;31m{str(datetime.date.strftime(date, "%Y_%m_%d"))}\033[0m in \033[1;34m{str(time.time()-t1)[0:7]}\033[0m s')

    @ save_log_file
    def ds2raster(self, zvalue: list, ROI=None, raster_size=None, ds2ras_method=None):

        if ds2ras_method is None:
            self._ds2raster_method = 'idw'
        elif ds2ras_method not in self._ds2raster_method_tup:
            raise ValueError(f'The {ds2ras_method} is not supported for ds2raster!')
        else:
            self._ds2raster_method = ds2ras_method

        if raster_size is None:
            raster_size = [10, 10]

        if ds2ras_method == 'idw':
            output_folder = self.output_path + 'idw_raster\\'
            bf.create_folder(output_folder)
            shpfile_folder = output_folder + 'shpfile\\'
            rasterfile_folder = output_folder + 'idwfile\\'
            bf.create_folder(shpfile_folder)
            bf.create_folder(rasterfile_folder)
            self.ds2pointshp(zvalue, shpfile_folder)

            shpfiles = bf.file_filter(shpfile_folder, ['.shp'])
            if shpfiles == []:
                raise ValueError(f'There are no valid shp files in the {str(shpfile_folder)}!')

            with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                executor.map(self._shp2raster, shpfiles, repeat(rasterfile_folder), repeat(zvalue), repeat(ROI), repeat(raster_size))

    def _shp2raster(self, shpfile: str, output_f: str, zvalue: list, clip_shpfile: str, raster_size: list):

        t1 = time.time()
        bounds = [374666.34182, 3672978.02321, 1165666.34182, 2863978.02321]
        width = (bounds[2] - bounds[0]) / (raster_size[0] * 20)
        height = (bounds[1] - bounds[3]) / (raster_size[1] * 20)

        print(f"Start generating the raster of \033[1;31m{str(shpfile)}\033[0m")
        for z in zvalue:
            try:
                if not os.path.exists(output_f + 'ori\\' + shpfile.split('\\')[-1].split('.')[0] + '.TIF'):
                    bf.create_folder(output_f + 'ori\\')
                    temp1 = gdal.Grid(output_f + 'ori\\' + shpfile.split('\\')[-1].split('.')[0] + '.TIF', shpfile, zfield=z, algorithm='invdist:power=2:min_points=5:max_points=12', outputBounds=bounds, spatFilter=bounds, width=width, height=height, outputType=gdal.GDT_Int16, noData=-32768)
                    temp1 = None

                if not os.path.exists(output_f + clip_shpfile.split('\\')[-1].split('.')[0] + '\\' + shpfile.split('\\')[-1].split('.')[0] + '.TIF') and clip_shpfile is not None and type(clip_shpfile) is str and clip_shpfile.endswith('.shp'):
                    bf.create_folder(output_f + clip_shpfile.split('\\')[-1].split('.')[0] + '\\')
                    temp2 = gdal.Warp('/vsimem/' + shpfile.split('\\')[-1].split('.')[0] + '_temp.vrt', output_f + 'ori\\' + shpfile.split('\\')[-1].split('.')[0] + '.TIF', resampleAlg=gdal.GRA_NearestNeighbour, xRes=raster_size[0], yRes=raster_size[1],  cropToCutline=True, cutlineDSName=clip_shpfile, outputType=gdal.GDT_Int16, dstNodata=-32768)
                    temp3 = gdal.Translate(output_f + clip_shpfile.split('\\')[-1].split('.')[0] + '\\' + shpfile.split('\\')[-1].split('.')[0] + '.TIF', '/vsimem/' + shpfile.split('\\')[-1].split('.')[0] + '_temp.vrt', options=topts)
                    temp2 = None
                    temp3 = None
                    gdal.Unlink('/vsimem/' + shpfile.split('\\')[-1].split('.')[0] + '_temp.vrt')
                # else:
                #     # Unfinish part
                #     gdal.Translate(output_f + shpfile.split('\\')[-1].split('.')[0] + '.TIF', '/vsimem/' + shpfile.split('\\')[-1].split('.')[0] + '_temp.TIF', xRes=raster_size[0], yRes=raster_size[1], options=topts)
            except:
                pass
        print(f'Finish generating the raster of \033[1;31m{str(shpfile)}\033[0m in \033[1;34m{str(time.time() - t1)[0:7]}\033[0m s')

    def _process_raster2dc_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in (
            'inherit_from_logfile', 'ROI', 'ROI_name', 'dc_overwritten_para',
            'manually_remove_datelist', 'size_control_factor'):
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
    def raster2dc(self, temporal_division=None, ROI=None, **kwargs):

        self._process_raster2dc_para(**kwargs)

        if temporal_division is None:
            temporal_division = 'year'
        elif temporal_division not in self._temporal_div_str:
            raise ValueError(f'The {temporal_division} is not supported!')

        if ROI == 'inherit_from_logfile':
            self._retrieve_para(['ROI'])
        elif ROI is None:
            self.ROI = 'ori'
        else:
            self.ROI = ROI

        if temporal_division == 'year':
            for year in self.year_range:
                input_folder = ssss

        header_dic = {'ROI_name': None, 'index': index, 'Datatype': 'float', 'ROI': None, 'ROI_array': None,
                      'ROI_tif': None,
                      'sdc_factor': True, 'coordinate_system': self.main_coordinate_system,
                      'size_control_factor': self._size_control_factor,
                      'oritif_folder': self._dc_infr[index + 'input_path'], 'dc_group_list': None, 'tiles': None}


