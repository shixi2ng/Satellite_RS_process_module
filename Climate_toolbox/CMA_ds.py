import shutil
from tqdm import tqdm
from NDsm import NDSparseMatrix
import numpy as np
from osgeo import gdal, osr, ogr
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
import psutil
import scipy.sparse as sm
import json
from shapely import Polygon, Point
import inspect
global topts
import Climate_toolbox

topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])


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
        self.metadata_folder = self.work_env + 'Metadata\\'
        bf.create_folder(self.metadata_folder)

        # Init _arg
        self._ds2ras_method_tup = ('IDW', 'ANUSPLIN')
        self._ds2ras_method = None

        # ras2dc
        self._temporal_div_str = ['year', 'month', 'all']

        # Init key variable
        self.ROI, self.ROI_name = None, None
        self.main_coordinate_system = None
        csv_files = bf.file_filter(file_path, ['SURF_CLI_CHN', '.txt', '.TXT'], exclude_word_list=['log.txt', 'para_file.txt'], subfolder_detection=True, and_or_factor='or')
        csv_files = [_ for _ in csv_files if 'SURF_CLI_CHN_MUL_DAY' in _]

        # Valid zvalue
        self._valid_zvalue = ['EVP', 'PRS', 'WIN', 'TEM', 'GST', 'RHU', 'PRE', 'SSD']
        self._zvalue_csvfile = {'EVP': [], 'PRS': [], 'WIN': [], 'TEM': [], 'GST': [], 'RHU': [], 'PRE': [], 'SSD': []}
        self._zvalue_dic = {'EVP': [(7, 0.1, 'Minor EVP', 32766), (8, 0.1, 'Major EVP', 32766)],
                            'PRS': [(7, 0.1, 'DAve_Airp', 32766), (8, 0.1, 'DMax_Airp', 32766), (9, 0.1, 'DMin_Airp', 32766)],
                            'WIN': [(7, 0.1, 'DAve_Winds', 32766), (8, 0.1, 'DMax_Winds', 32766), (10, 0.1, 'DMin_Winds', 32766)],
                            'TEM': [(7, 0.1, 'DAve_AT', 32766), (8, 0.1, 'DMax_AT', 32766), (9, 0.1, 'DMin_AT', 32766)],
                            'GST': [(7, 0.1, 'DAve_LST', 32766), (8, 0.1, 'DMax_LST', 32766), (9, 0.1, 'DMin_LST', 32766)],
                            'RHU': [(7, 0.01, 'DAve_hum', 32766), (8, 0.01, 'Dmin_hum', 32766)],
                            'PRE': [(7, 0.1, '20-8_prec', 32766), (8, 0.1, '8-20_prec', 32766), (9, 0.1, 'Acc_prec', 32766)],
                            'SSD': [(7, 0.1, 'SS_hour', 32766)]}
        self._interpolate_zvalue = {'EVP': (8, 0.1, 'Major EVP', 32766), 'PRS': (7, 0.1, 'DAve_Airp', 32766), 'WIN': (7, 0.1, 'DAve_Winds', 32766),
                                    'TEM': (7, 0.1, 'DAve_AT', 32766), 'GST': (7, 0.1, 'DAve_LST', 32766), 'RHU': (7, 0.01, 'DAve_hum', 32766),
                                    'PRE': (9, 0.1, 'Acc_prec', 32766), 'SSD': (7, 0.1, 'SS_hour', 32766)}
        self.zvalue = list(set([_.split('SURF_CLI_CHN_MUL_DAY-')[1].split('-')[0] for _ in csv_files]))

        if False in [_ in self._valid_zvalue for _ in self.zvalue]:
            raise Exception(f'The zvalue {str(self.zvalue[[_ in self._valid_zvalue for _ in self.zvalue].index(False)])}')

        for __ in csv_files:
            if True not in [zvalue_ in __ for zvalue_ in self.zvalue]:
                csv_files.remove(__)
            else:
                for zvalue_ in self._valid_zvalue:
                    if zvalue_ in __:
                        self._zvalue_csvfile[zvalue_].append(__)

        # Get the zvalue month and
        self._cma_header_ = ['Station_id', 'Lat', 'Lon', 'Alt', 'YYYY', 'MM', 'DD']
        self.date_range = {}
        self.year_range = {}

        # Generate the station inform df and data range df
        station_inform_dic = {'Station_id': [], 'Lon': [], 'Lat': [], 'Alt': [], 'Month': [],}
        self.date_range = {zvalue_: {'Station_id': [], 'Index': [], 'Month': []} for zvalue_ in self.zvalue}
        zvalue_month_station = {}

        if False in [os.path.exists(os.path.join(self.metadata_folder, f'{zvalue_}.csv')) for zvalue_ in self.zvalue] or not os.path.exists(os.path.join(self.metadata_folder, f'station_inform.csv')):
            with tqdm(total=len(csv_files), desc=f'Extract the CMA metadata', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                for csv_ in csv_files:
                    zvalue_ = csv_.split('SURF_CLI_CHN_MUL_DAY-')[1].split('-')[0]
                    month_ = int(csv_.split('SURF_CLI_CHN_MUL_DAY-')[1].split('.')[0].split('-')[-1])
                    df_temp = pd.read_table(csv_, delim_whitespace=True, header=None)

                    # traverse the station inform
                    station_inform_df_ = df_temp.iloc[:, :4].drop_duplicates().reset_index(drop=True)
                    station_inform_dic['Station_id'].extend(list(station_inform_df_.iloc[:, 0]))
                    lon, lat, alt = list(station_inform_df_.iloc[:, 1]), list(station_inform_df_.iloc[:, 2]), list(station_inform_df_.iloc[:, 3])
                    station_inform_dic['Lon'].extend(lon)
                    station_inform_dic['Lat'].extend(lat)
                    station_inform_dic['Alt'].extend(alt)
                    station_inform_dic['Month'].extend([month_ for __ in range(len(lon))])

                    # traverse the date range
                    station_all = list(pd.unique(df_temp.iloc[:, 0]))
                    self.date_range[zvalue_]['Station_id'].extend(station_all)
                    self.date_range[zvalue_]['Index'].extend([zvalue_ for _ in range(len(station_all))])
                    self.date_range[zvalue_]['Month'].extend([month_ for _ in range(len(station_all))])
                    pbar.update()

            # Generate the station inform.csv
            if not os.path.exists(os.path.join(self.metadata_folder, f'station_inform.csv')):
                station_inform_df = pd.DataFrame(station_inform_dic)
                station_inform_df_ = station_inform_df.iloc[:, :4].drop_duplicates()
                station_inform_df_.sort_values(by='Station_id', inplace=True)
                station_inform_df_.reset_index(drop=True, inplace=True)
                station_inform_df_['Start_YYYYMM'] = None
                station_inform_df_['End_YYYYMM'] = None
                shape = len(station_inform_df_)
                for _ in range(shape):
                    month_list = list(pd.unique(station_inform_df.loc[(station_inform_df['Station_id'] == station_inform_df_.loc[_, 'Station_id']) &
                                                                      (station_inform_df['Lon'] == station_inform_df_.loc[_, 'Lon']) &
                                                                      (station_inform_df['Lat'] == station_inform_df_.loc[_, 'Lat']) &
                                                                      (station_inform_df['Alt'] == station_inform_df_.loc[_, 'Alt']), 'Month']))
                    if len(month_list) == (max(month_list) // 100 - min(month_list) // 100) * 12 + np.mod(max(month_list), 100) - np.mod(min(month_list), 100) + 1:
                        station_inform_df_.loc[_, 'Start_YYYYMM'] = int(min(month_list))
                        station_inform_df_.loc[_, 'End_YYYYMM'] = int(max(month_list))
                    else:
                        data_range = separate_month_range(month_list)
                        row2copy = station_inform_df_.loc[[_]].reset_index(drop=True)
                        for __ in range(len(data_range)):
                            if __ == 0:
                                station_inform_df_.loc[_, 'Start_YYYYMM'] = int(min(data_range[0]))
                                station_inform_df_.loc[_, 'End_YYYYMM'] = int(max(data_range[0]))
                            else:
                                row2copy_ = copy.deepcopy(row2copy)
                                row2copy_.loc[0, 'Start_YYYYMM'] = int(min(data_range[__]))
                                row2copy_.loc[0, 'End_YYYYMM'] = int(max(data_range[__]))
                                station_inform_df_ = pd.concat([station_inform_df_, row2copy_], ignore_index=True)

                station_inform_df_ = station_inform_df_.sort_values(by=['Station_id', 'Start_YYYYMM']).reset_index(drop=True)
                station_inform_df_.to_csv(os.path.join(self.metadata_folder, f'station_inform.csv'), index=False)
            else:
                station_inform_df_ = pd.read_csv(os.path.join(self.metadata_folder, f'station_inform.csv'))

            if False in [os.path.exists(os.path.join(self.metadata_folder, f'{zvalue_}.csv')) for zvalue_ in self.zvalue]:
                for zvalue_ in self.zvalue:
                    zvalue_month_station[zvalue_] = pd.DataFrame(self.date_range[zvalue_])
                    zvalue_month_station[zvalue_].to_csv(os.path.join(self.metadata_folder, f'{zvalue_}.csv'), index=False)
            else:
                for zvalue_ in self.zvalue:
                    zvalue_month_station[zvalue_] = pd.read_csv(os.path.join(self.metadata_folder, f'{zvalue_}.csv'))
        else:
            for zvalue_ in self.zvalue:
                zvalue_month_station[zvalue_] = pd.read_csv(os.path.join(self.metadata_folder, f'{zvalue_}.csv'))
            station_inform_df_ = pd.read_csv(os.path.join(self.metadata_folder, f'station_inform.csv'))

        station_inform_df_['Lat'] = station_inform_df_['Lat'] // 100 + np.mod(station_inform_df_['Lat'], 100) / 60
        station_inform_df_['Lon'] = station_inform_df_['Lon'] // 100 + np.mod(station_inform_df_['Lon'], 100) / 60
        station_inform_df_['Alt'] = station_inform_df_['Alt'] / 100
        self.station_inform_df = station_inform_df_

        # Generate the year range and date_range
        for zvalue_ in self.zvalue:
            self.date_range[zvalue_] = pd.unique(zvalue_month_station[zvalue_]['Month'])
            self.year_range[zvalue_] = pd.unique(np.array([_ // 100 for _ in self.date_range[zvalue_]])).tolist()

        self.csv_files = csv_files
        # Define cache folder
        # self.cache_folder, self.trash_folder = self.work_env + 'cache\\', self.work_env + 'trash\\'
        # bf.create_folder(self.cache_folder)
        # bf.create_folder(self.trash_folder)

        # Create output path
        self.output_path, self.log_filepath = f'{os.path.dirname(self.work_env)}\\CMA_OUTPUT\\',  f'{self.work_env}Logfile\\'
        bf.create_folder(self.output_path)
        bf.create_folder(self.log_filepath)
        # bf.create_folder(self.shpfile_path)

        # column_name_list = ['None' for _ in range(len(df_temp.columns))]
        # for column_ in range(len(df_temp.columns)):
        #     if column_ < len(header_):
        #         column_name_list[column_] = header_[column_]
        #     elif column_ in [_[0] for _ in self._zvalue_dic[zvalue_]]:
        #         pos = [_[0] for _ in self._zvalue_dic[zvalue_]].zvalue(column_)
        #         column_name_list[column_] = self._zvalue_dic[zvalue_][pos][2]
        #     else:
        #         column_name_list[column_] = 'None'
        # df_temp.columns = column_name_list

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
            sig = inspect.signature(func)
            args_name = [name for name, param in sig.parameters.items()]
            args_name.remove('self')
            args_list = ['*' * 25 + 'Arguments' + '*' * 25 + '\n']
            kwargs_list = []
            for i, name_ in zip(args, args_name[:len(args)]):
                args_list.extend([f"{str(name_)}:{str(i)}\n"])

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

            for func_key, func_processing_name in zip(['anusplin', 'ds2pointshp', 'ds2raster', 'raster2dc'], ['anusplin', '2point', '2raster', 'rs2dc']):
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

    def append_ds(self, new_cma_ds, output_folder, inplace=False):

        # Create folder
        bf.create_folder(output_folder)
        if os.path.exists(output_folder):
            new_ds_folder = os.path.join(output_folder, 'Original\\')
            bf.create_folder(new_ds_folder)
        else:
            raise ValueError('The output folder is not valid')

        # Get CMA ds
        if isinstance(new_cma_ds, CMA_ds):

            # Generate the original and new csv files
            orids_csv_files = self.csv_files
            newds_csv_files = new_cma_ds.csv_files
            ori_filename = [os.path.basename(_).split('.')[0] for _ in orids_csv_files]
            new_filename = [os.path.basename(_).split('.')[0] for _ in newds_csv_files]
            ori_filename.extend(new_filename)
            all_filename = list(set(ori_filename))

            with tqdm(total=len(all_filename), desc=f'Assemble the CMA dataset', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                for filename_ in all_filename:

                    # Generate the status
                    ori_status = [filename_ in _ for _ in orids_csv_files]
                    new_status = [filename_ in _ for _ in newds_csv_files]

                    # Create output folder
                    for index_ in self._valid_zvalue:
                        if index_ in filename_:
                            new_index_folder = os.path.join(new_ds_folder, f'{index_}\\')
                            bf.create_folder(new_index_folder)

                    if True in ori_status and True in new_status:
                        ori_df_temp = pd.read_table(orids_csv_files[ori_status.index(True)], delim_whitespace=True, header=None)
                        new_df_temp = pd.read_table(newds_csv_files[new_status.index(True)], delim_whitespace=True, header=None)
                        if len(ori_df_temp.keys()) != len(new_df_temp.keys()):
                            pass
                        new_df_ = pd.concat([ori_df_temp, new_df_temp], ignore_index=True)
                        new_df_ = new_df_.drop_duplicates()
                        new_df_ = new_df_.sort_values(by=[0,4,5,6])
                        new_df_ = new_df_.reset_index(drop=True)
                        new_df_.to_csv(os.path.join(new_index_folder, f'{filename_}.TXT'), sep=' ', index=False, header=False)
                    elif True in ori_status:
                        shutil.copy(orids_csv_files[ori_status.index(True)], os.path.join(new_index_folder, f'{filename_}.TXT'))
                    elif True in new_status:
                        shutil.copy(newds_csv_files[new_status.index(True)], os.path.join(new_index_folder, f'{filename_}.TXT'))
                    else:
                        raise Exception('Code error!')
                    pbar.update()

        else:
            raise TypeError('The append ds should under a CMA dataset!')

        if inplace:
            return CMA_ds(output_folder)
        else:
            self = CMA_ds(output_folder)

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

    @save_log_file
    def anusplin(self, ROI, DEM, mask=None, zvalue=None, date_range=None, cell_size=None,  bulk=True):
        """
        :param ROI: ROI is used as CropContent in gdal.Warp to extract the ANUSPLIN_processed climatology data. It should
        be under the same coordinate system with the mask and DEM
        :param mask: Mask is used in the ANUSPLIN-LAPGRD program to extract the climatology data. It should be converted
        into the ASCII TXT file and share same bounds, cell size and coordinate system with the DEM.
        :param DEM: DEM is treated as the covariant in the ANUSPLINE. It should be converted into the ASCII TXT file and
        share same bounds, cell size and coordinate system with the mask.
        :param zvalue: The zvalue of climatology data for processing
        :param date_range: Start YYYYDOY - End YYYYDOY
        :param cell_size: Output cellsize
        :param output_path:
        :param bulk:
        """

        # Check the ANUSPLIN program
        module_dir = os.path.dirname(__file__)
        if 'splina.exe' not in os.listdir(module_dir) or 'lapgrd.exe' not in os.listdir(module_dir):
            raise Exception('The splina.exe or lapgrd.exe was missing!')
        else:
            self._splina_program = os.path.join(module_dir, 'splina.exe')
            self._lapgrd_program = os.path.join(module_dir, 'lapgrd.exe')

        # Identify the DEM bounds
        if DEM.endswith('.txt') or DEM.endswith('.TXT'):
            try:
                with open(DEM, 'r') as f:
                    dem_content = f.read()
                    dem_content = dem_content.split('\n')[0:6]
                    dem_content = [(str(_.split(' ')[0]), float(_.split(' ')[-1])) for _ in dem_content]
                    dem_bound = [dem_content[2][1], dem_content[3][1] + dem_content[1][1] * dem_content[4][1],
                                 dem_content[2][1] + dem_content[0][1] * dem_content[4][1], dem_content[3][1]]
                    dem_cellsize = dem_content[4][1]
                    dem_nodata = dem_content[5][1]
                    f.close()
            except:
                print(traceback.format_exc())
                raise Exception('Invalid DEM during the anusplin')
        else:
            raise TypeError('Please convert the dem into txt file using ArcMap or other Geo-software!')

        # Identify the mask bounds
        if mask is not None:
            if mask.endswith('.txt') or mask.endswith('.TXT'):
                try:
                    with open(mask, 'r') as f:
                        mask_content = f.read()
                        mask_content = mask_content.split('\n')[0:6]
                        mask_content = [(str(_.split(' ')[0]), float(_.split(' ')[-1])) for _ in mask_content]
                        mask_bound = [mask_content[2][1], mask_content[3][1] + mask_content[1][1] * mask_content[4][1],
                                      mask_content[2][1] + mask_content[0][1] * mask_content[4][1], mask_content[3][1]]
                        mask_cellsize = mask_content[4][1]
                        mask_nodata = mask_content[5][1]
                        f.close()
                except:
                    print(traceback.format_exc())
                    raise Exception('Invalid mask during the anusplin')
            else:
                raise TypeError('Please convert the MASK into ASCII txt file')

            # Check the consistency between the mask amd DEM
            if mask_bound != dem_bound:
                raise ValueError('The DEM and mask share inconsistency bound')
            else:
                anusplin_bounds = copy.deepcopy(dem_bound)

            if mask_cellsize != dem_cellsize:
                raise ValueError('The DEM and mask share inconsistency cellsize')
            else:
                anusplin_cellsize = copy.deepcopy(dem_cellsize)

            if mask_nodata != dem_nodata:
                raise ValueError('The DEM and mask share inconsistency nodata')
            else:
                anusplin_nodata = copy.deepcopy(dem_nodata)
        else:
            anusplin_bounds = copy.deepcopy(dem_bound)
            anusplin_cellsize = copy.deepcopy(dem_cellsize)
            anusplin_nodata = copy.deepcopy(dem_nodata)

        # Identify the ROI
        if isinstance(ROI, str) and os.path.exists(ROI) and (ROI.endswith('.shp') or ROI.endswith('.SHP')):
            driver = ogr.GetDriverByName('ESRI Shapefile')
            datasource = driver.Open(ROI, 0)
            layer = datasource.GetLayer(0)
            spatial_ref = layer.GetSpatialRef()
            output_crs = 'EPSG:' + spatial_ref.GetAttrValue("AUTHORITY", 1)
            extent = layer.GetExtent()
            roi_bounds = [extent[0], extent[3], extent[1], extent[2]]
        elif isinstance(ROI, str) and os.path.exists(ROI) and (ROI.endswith('.tif') or ROI.endswith('.TIF')):
            roi_bounds = bf.get_tif_border(ROI)
            ds = gdal.Open(ROI)
            proj = osr.SpatialReference(wkt=ds.GetProjection())
            output_crs = 'EPSG:' + proj.GetAttrValue('AUTHORITY', 1)
            ds = None
        else:
            raise TypeError('The input ROI is under wrong type!')

        self.ROI = ROI
        self.ROI_name = self.ROI.split('\\')[-1].split('.')[0]

        # Check if the ROI within the range of DEM and Mask
        if (roi_bounds[0] < anusplin_bounds[0] or roi_bounds[1] > anusplin_bounds[1]
                or roi_bounds[2] > anusplin_bounds[2] or roi_bounds[3] < anusplin_bounds[3]):
            raise Exception('The ROI is outside the DEM and Mask')

        # Identify the output cell size
        if cell_size is None:
            output_cellsize = 30
        else:
            output_cellsize = cell_size

        # Identify the zvalue
        if zvalue is None:
            zvalue = self.zvalue
        elif isinstance(zvalue, list):
            zvalue = [zvalue_ for zvalue_ in zvalue if zvalue_ in self.zvalue]
        elif isinstance(zvalue, str):
            if zvalue not in self.zvalue:
                raise Exception(f'The {str(zvalue)} is not supported!')
        else:
            raise TypeError('The input zvalue is under wrong type!')

        # Identify the date_range
        if date_range is None:
            date_range = {key: self.date_range[key] for key in self.date_range if key in zvalue}
        elif isinstance(date_range, list) and len(date_range) == 2:
            try:
                start_date, end_date = int(date_range[0]), int(date_range[1])
                start_month, end_month = int(start_date // 100), int(end_date // 100)
                if start_date > end_date:
                    raise ValueError('The end date is smaller than start date!')
                else:
                    date_range = {key: self.date_range[key] for key in self.date_range if key in zvalue}
                    for _ in date_range.keys():
                        date_range_ = [date_ for date_ in date_range[_] if start_month <= date_ <= end_month]
                        date_range[_] = date_range_
            except:
                print(traceback.format_exc())
                raise Exception('The input date type is not correct')
        else:
            raise TypeError('The input date range is under wrong type!')

        # Execute the SPLIN and LAPGRD process
        for zvalue_ in zvalue:
            zvalue_date_range = date_range[zvalue_]
            if bulk is True:
                with concurrent.futures.ProcessPoolExecutor() as exe:
                    res = exe.map(self.execute_splin, repeat(zvalue_), zvalue_date_range, repeat(anusplin_bounds), repeat(output_crs))

                # Export the error res
                err_list = []
                res = list(res)
                for res_ in res:
                    err_list.extend(res_)
                err_df_ = pd.DataFrame(err_list)
                err_df_.to_csv(f'{self.output_path}{str(self.ROI_name)}_Denv_raster\\ANUSPLIN_SPLIN\\{zvalue_}\\err.csv')

            else:
                for zvalue_month in zvalue_date_range:
                    self.execute_splin(zvalue_, zvalue_month,  anusplin_bounds, output_crs)

            if bulk is True:
                with concurrent.futures.ProcessPoolExecutor() as exe:
                    exe.map(self.execute_lapgrd, repeat(zvalue_), zvalue_date_range, repeat(DEM), repeat(anusplin_cellsize), repeat(anusplin_nodata), repeat(ROI), repeat(output_cellsize), repeat(output_crs), repeat(mask))
            else:
                for zvalue_month in zvalue_date_range:
                    self.execute_lapgrd(zvalue_, zvalue_month, DEM, anusplin_cellsize, anusplin_nodata, ROI, output_cellsize, output_crs, lapgrd_maskfile=mask)

    def execute_splin(self, zvalue, year_month,  splin_bounds, output_crs):

        try:
            # Define the output path
            splin_path = self.output_path + f'{str(self.ROI_name)}_Denv_raster\\ANUSPLIN_SPLIN\\'
            bf.create_folder(splin_path)

            # Define the standard lapgrd conf file
            self._splin_conf = '{}\n1\n2\n1\n0\n0\n{}\n{}\n-400 9000 1 1\n1000.0\n0\n4\n1\n0\n1\n1\n{}.dat\n300\n6\n(a6,2f14.6,f6.2,f4.1)\n{}.res\n{}.opt\n{}.sur\n{}.lis\n{}.cov\n\n\n\n\n\n'
            self._splin_conf_rootsquare = '{}\n1\n2\n1\n0\n0\n{}\n{}\n-400 9000 1 1\n1000.0\n2\n4\n1\n0\n1\n1\n{}.dat\n300\n6\n(a6,2f14.6,f6.2,f4.1)\n{}.res\n{}.opt\n{}.sur\n{}.lis\n{}.cov\n\n\n\n\n\n'

            # Identify the output path
            if not os.path.exists(splin_path):
                raise ValueError(f'The {splin_path} is not valid')
            zvalue_splin_path = os.path.join(splin_path, f'{zvalue}\\')
            bf.create_folder(zvalue_splin_path)

            # Copy the programme
            zvalue_splina = os.path.join(zvalue_splin_path, 'splina.exe')
            if not os.path.exists(zvalue_splina):
                shutil.copy(self._splina_program, zvalue_splina)

            # Read the df from csv file
            csv_file = [csv_ for csv_ in self.csv_files if str(year_month) in csv_ and zvalue in csv_]
            df_temp = pd.read_table(csv_file[0], delim_whitespace=True, header=None)
            column_name_all = [_[2] for _ in self._zvalue_dic[zvalue]]
            column_name_list = ['None' for _ in range(len(df_temp.columns))]
            for column_ in range(len(df_temp.columns)):
                if column_ < len(self._cma_header_):
                    column_name_list[column_] = self._cma_header_[column_]
                elif column_ in [_[0] for _ in self._zvalue_dic[zvalue]]:
                    pos = [_[0] for _ in self._zvalue_dic[zvalue]].index(column_)
                    column_name_list[column_] = self._zvalue_dic[zvalue][pos][2]
                else:
                    column_name_list[column_] = 'None'
            df_temp.columns = column_name_list

            # Process DOY
            doy_list = []
            for row in range(df_temp.shape[0]):
                doy_list.append(df_temp['YYYY'][row] * 1000 + datetime.date(year=df_temp['YYYY'][row], month=df_temp['MM'][row], day=df_temp['DD'][row]).toordinal()
                                - datetime.date(year=df_temp['YYYY'][row], month=1, day=1).toordinal() + 1)
            df_temp['DOY'] = doy_list
            doy_list = pd.unique(df_temp['DOY'])

            # Process lat lon and alt
            df_temp['Lon'] = df_temp['Lon'] // 100 + (np.mod(df_temp['Lon'], 100) / 60)
            df_temp['Lat'] = df_temp['Lat'] // 100 + (np.mod(df_temp['Lat'], 100) / 60)
            df_temp['Alt'] = df_temp['Alt'].astype(np.float32) / 100

            # Determine the header
            header = ['Station_id', 'Lon', 'Lat', 'Alt', self._interpolate_zvalue[zvalue][2]]

            # Get the geodf itr through date
            # Create Res result
            err_list = []
            for doy in doy_list:
                try:
                    t1 = time.time()
                    print(f'Start executing the SPLINA for {str(zvalue)} of \033[1;31m{str(doy)}\033[0m!')
                    if not os.path.exists(os.path.join(zvalue_splin_path, f'{str(zvalue)}_{str(doy)}.dat')):
                        pd_temp = df_temp[df_temp['DOY'] == doy][header]

                        # Change the datatype
                        pd_temp[self._interpolate_zvalue[zvalue][2]] = pd_temp[self._interpolate_zvalue[zvalue][2]].astype(np.float32)

                        # Process the invalid value
                        if zvalue == 'PRE':
                            pd_temp.loc[pd_temp[self._interpolate_zvalue[zvalue][2]] > 30000, self._interpolate_zvalue[zvalue][2]] = np.nan
                        else:
                            pd_temp.loc[pd_temp[self._interpolate_zvalue[zvalue][2]] == self._interpolate_zvalue[zvalue][3], self._interpolate_zvalue[zvalue][2]] = np.nan

                        # Resize the value
                        pd_temp[self._interpolate_zvalue[zvalue][2]] = pd_temp[self._interpolate_zvalue[zvalue][2]] * self._interpolate_zvalue[zvalue][1]
                        pd_temp = pd_temp.reset_index(drop=True)
                        geodf_temp = gp.GeoDataFrame(pd_temp, geometry=[Point(xy) for xy in zip(pd_temp['Lon'], pd_temp['Lat'])], crs='EPSG:4326')

                        # Transform into destination crs
                        if output_crs != 'EPSG:4326':
                            geodf_temp = geodf_temp.to_crs(crs=output_crs)
                            geodf_temp['Lon'] = [_.coords[0][0] for _ in geodf_temp['geometry']]
                            geodf_temp['Lat'] = [_.coords[0][1] for _ in geodf_temp['geometry']]

                        # Use the bounds to extract the point
                        geodf_temp = geodf_temp[(splin_bounds[2] >= geodf_temp['Lon']) & (geodf_temp['Lon'] >= splin_bounds[0])]
                        geodf_temp = geodf_temp[(splin_bounds[3] <= geodf_temp['Lat']) & (geodf_temp['Lat'] <= splin_bounds[1])]
                        df_doy = pd.DataFrame(geodf_temp)[header]
                        df_doy = df_doy.reset_index(drop=True)

                        # Format and generate the dat file
                        f = open(os.path.join(zvalue_splin_path, f'{str(zvalue)}_{str(doy)}.dat'), 'w')
                        for i in range(df_doy.shape[0]):
                            sta, lon, lat, height, val = [df_doy[__][i] for __ in header]
                            if ~np.isnan(val):
                                text = '{:>5} {:>14.6f}{:>14.6f}{:>6.2f}{:>4.1f}\n'.format(str(sta), lon, lat, height, val)
                                f.write(text)
                            else:
                                pass
                        f.close()

                    file_name = os.path.join(zvalue_splin_path, f'{str(zvalue)}_{str(doy)}')
                    if not os.path.exists(os.path.join(zvalue_splin_path, f'{str(zvalue)}_{str(doy)}.conf')):
                        # Format and generate the conf file
                        f = open(os.path.join(zvalue_splin_path, f'{str(zvalue)}_{str(doy)}.conf'), 'w')
                        lon_coord = f'{str(splin_bounds[0])} {str(splin_bounds[2])} 0 1'
                        lat_coord = f'{str(splin_bounds[3])} {str(splin_bounds[1])} 0 1'
                        if zvalue != 'PRE':
                            f.writelines(self._splin_conf.format(file_name, lon_coord, lat_coord, file_name, file_name, file_name, file_name, file_name, file_name))
                        else:
                            f.writelines(self._splin_conf_rootsquare.format(file_name, lon_coord, lat_coord, file_name, file_name, file_name, file_name, file_name, file_name))
                        f.close()

                    # Generate the surf file
                    if not os.path.exists(file_name + '.sur'):
                        cmd_ = zvalue_splina + ' <{}> {}.log'.format(os.path.join(zvalue_splin_path, f'{str(zvalue)}_{str(doy)}.conf'), os.path.join(zvalue_splin_path, f'{str(zvalue)}_{str(doy)}'))
                        os.system(cmd_)
                    
                    # Generate the error
                    doy_err_ = [doy, zvalue, np.nan, np.nan, np.nan]
                    txt = open(os.path.join(zvalue_splin_path, f'{str(zvalue)}_{str(doy)}.log'))
                    log = txt.readlines()
                    for _ in range(len(log)):
                        if log[_].startswith('SURFACE STATISTICS'):
                            error_ = [__ for __ in log[_ + 3].split(' ') if isinstance(__, (float, np.floating))]
                            if len(error_) == 8:
                                doy_err_[2], doy_err_[3], doy_err_[4] = [float(error_[3]), float(error_[4]), float(error_[-1])]
                    print(f'Finish executing the SPLINA procedure for {str(zvalue)} of \033[1;31m{str(doy)}\033[0m in \033[1;34m{str(time.time() - t1)[0:7]}\033[0m s')
                except:
                    print(traceback.format_exc())
                    print(f'Failed to execute the SPLINA procedure for {str(zvalue)} of \033[1;31m{str(doy)}\033[0m.')
                    doy_err_ = [doy, zvalue, np.nan, np.nan, np.nan]
                
                # Append the daily error into the result list
                err_list.append(doy_err_)
            return err_list    
        except:
            print(traceback.format_exc())

    def execute_lapgrd(self, zvalue, year_month, lapgrd_demfile, lapgrd_cellsize, lapgrd_nodata, ROI, ROI_cellsize, crs, lapgrd_maskfile=None):

        try:
            # Define the input and output path
            splin_path = self.output_path + f'{str(self.ROI_name)}_Denv_raster\\ANUSPLIN_SPLIN\\'
            lapgrd_path = self.output_path + f'{str(self.ROI_name)}_Denv_raster\\ANUSPLIN_LAPGRD\\'
            bf.create_folder(lapgrd_path)

            # Define the standard lapgrd conf file
            if lapgrd_maskfile is None:
                self._lapgrd_conf = '{}\n1\n1\n{}\n2\n\n1\n1\n{}\n2\n{}\n0\n2\n{}\n2\n{}\n{}\n{}\n2\n{}\n{}\n{}\n\n\n\n\n\n'
                self._lapgrd_conf_rootsquare = '{}\n1\n1\n1\n{}\n2\n\n1\n1\n{}\n2\n{}\n0\n2\n{}\n2\n{}\n{}\n{}\n2\n{}\n{}\n{}\n\n\n\n\n\n'
            else:
                self._lapgrd_conf = '{}\n1\n1\n{}\n2\n\n1\n1\n{}\n2\n{}\n2\n{}\n2\n{}\n2\n{}\n{}\n{}\n2\n{}\n{}\n{}\n\n\n\n\n\n'
                self._lapgrd_conf_rootsquare = '{}\n1\n1\n1\n{}\n2\n\n1\n1\n{}\n2\n{}\n2\n{}\n2\n{}\n2\n{}\n{}\n{}\n2\n{}\n{}\n{}\n\n\n\n\n\n'

            # Identify the output LAPGRD path and Input splin path
            if not os.path.exists(lapgrd_path):
                raise ValueError(f'The {lapgrd_path} is not valid')
            if not os.path.exists(os.path.join(splin_path, f'{zvalue}\\')):
                raise ValueError(f'The {splin_path} is not valid')
            else:
                splin_path = os.path.join(splin_path, f'{zvalue}\\')

            zvalue_lapgrd_path = os.path.join(lapgrd_path, f'{zvalue}\\')
            bf.create_folder(zvalue_lapgrd_path)

            # Copy the programme
            zvalue_lapgrd = os.path.join(zvalue_lapgrd_path, 'lapgrd.exe')
            if not os.path.exists(zvalue_lapgrd):
                shutil.copy(self._lapgrd_program, zvalue_lapgrd)

            csv_file = [csv_ for csv_ in self.csv_files if str(year_month) in csv_ and zvalue in csv_]
            df_name_list = ['Station_id', 'Lat', 'Lon', 'Alt', 'DOY']
            df_name_list.extend([__[2] for __ in self._zvalue_dic[zvalue]])
            df_temp = pd.read_table(csv_file[0], delim_whitespace=True, header=None)
            column_name_all = [_[2] for _ in self._zvalue_dic[zvalue]]
            column_name_list = ['None' for _ in range(len(df_temp.columns))]
            for column_ in range(len(df_temp.columns)):
                if column_ < len(self._cma_header_):
                    column_name_list[column_] = self._cma_header_[column_]
                elif column_ in [_[0] for _ in self._zvalue_dic[zvalue]]:
                    pos = [_[0] for _ in self._zvalue_dic[zvalue]].index(column_)
                    column_name_list[column_] = self._zvalue_dic[zvalue][pos][2]
                else:
                    column_name_list[column_] = 'None'
            df_temp.columns = column_name_list

            # Process DOY
            doy_list = []
            for row in range(df_temp.shape[0]):
                doy_list.append(df_temp['YYYY'][row] * 1000 + datetime.date(year=df_temp['YYYY'][row], month=df_temp['MM'][row], day=df_temp['DD'][row]).toordinal()
                                - datetime.date(year=df_temp['YYYY'][row], month=1, day=1).toordinal() + 1)
            df_temp['DOY'] = doy_list

            # Get the geodf itr through date
            doy_list = pd.unique(df_temp['DOY'])
            for doy in doy_list:
                try:
                    t1 = time.time()
                    print(f'Start executing the LAPGRD for {str(zvalue)} of \033[1;31m{str(doy)}\033[0m!')
                    if (not os.path.exists(os.path.join(splin_path, f'{str(zvalue)}_{str(doy)}.cov')) or
                            not os.path.exists(os.path.join(splin_path, f'{str(zvalue)}_{str(doy)}.sur'))):
                        raise Exception(f'.cov file for {str(zvalue)} of \033[1;31m{str(doy)}\033[0m is missing!')

                    # Generate the conf file
                    output_grd = os.path.join(zvalue_lapgrd_path, f'{str(zvalue)}_{str(doy)}.grd')
                    output_res = os.path.join(zvalue_lapgrd_path, f'{str(zvalue)}_{str(doy)}_res.grd')
                    if not os.path.exists(os.path.join(zvalue_lapgrd_path, f'{str(zvalue)}_{str(doy)}.conf')):
                        f = open(os.path.join(zvalue_lapgrd_path, f'{str(zvalue)}_{str(doy)}.conf'), 'w')
                        sur_file = bf.file_filter(splin_path, [str(zvalue), str(doy), '.sur'], and_or_factor='and')[0]
                        cov_file = bf.file_filter(splin_path, [str(zvalue), str(doy), '.cov'], and_or_factor='and')[0]
                        with open(sur_file, 'r') as surf_f:
                            surf_content = surf_f.read()
                            surf_content = surf_content.split('\n')[2:4]
                            if surf_content[0].startswith(' '):
                                x_limit = f"{surf_content[0].split(' ')[2]} {surf_content[0].split(' ')[4]} {str(lapgrd_cellsize)}"
                            else:
                                x_limit = f"{surf_content[0].split(' ')[1]} {surf_content[0].split(' ')[3]} {str(lapgrd_cellsize)}"

                            if surf_content[1].startswith(' '):
                                y_limit = f"{surf_content[1].split(' ')[2]} {surf_content[1].split(' ')[4]} {str(lapgrd_cellsize)}"
                            else:
                                y_limit = f"{surf_content[1].split(' ')[1]} {surf_content[1].split(' ')[3]} {str(lapgrd_cellsize)}"

                        if lapgrd_maskfile is None:
                            if zvalue == 'PRE':
                                f.writelines(self._lapgrd_conf_rootsquare.format(sur_file, cov_file, x_limit, y_limit, lapgrd_demfile, str(lapgrd_nodata), output_grd, '(1f6.2)', str(lapgrd_nodata), output_res, '(1f8.2)'))
                            else:
                                f.writelines(self._lapgrd_conf.format(sur_file, cov_file, x_limit, y_limit, lapgrd_demfile, str(lapgrd_nodata), output_grd, '(1f6.2)', str(lapgrd_nodata), output_res, '(1f8.2)'))
                        else:
                            if zvalue == 'PRE':
                                f.writelines(self._lapgrd_conf_rootsquare.format(sur_file, cov_file, x_limit, y_limit, lapgrd_maskfile, lapgrd_demfile, str(lapgrd_nodata), output_grd, '(1f6.2)', str(lapgrd_nodata), output_res, '(1f8.2)'))
                            else:
                                f.writelines(self._lapgrd_conf.format(sur_file, cov_file, x_limit, y_limit, lapgrd_maskfile, lapgrd_demfile, str(lapgrd_nodata), output_grd, '(1f6.2)', str(lapgrd_nodata), output_res, '(1f8.2)'))
                        f.close()

                    # Execute the lapgrd
                    if not os.path.exists(os.path.join(zvalue_lapgrd_path, f'{str(zvalue)}_{str(doy)}.grd')):
                        cmd_ = zvalue_lapgrd + ' <{}> {}.log'.format(os.path.join(zvalue_lapgrd_path, f'{str(zvalue)}_{str(doy)}.conf'), os.path.join(zvalue_lapgrd_path, f'{str(zvalue)}_{str(doy)}'))
                        os.system(cmd_)

                    # Execute the gdal Warp
                    if not os.path.exists(os.path.join(zvalue_lapgrd_path, f'{str(zvalue)}_{str(doy)}.TIF')):
                        temp_ds = gdal.Warp('/vsimem/' + f'{str(zvalue)}_{str(doy)}_temp.vrt', os.path.join(zvalue_lapgrd_path, f'{str(zvalue)}_{str(doy)}.grd'), srcSRS=crs,
                                            resampleAlg=gdal.GRA_Bilinear, outputType=gdal.GDT_Float32, dstNodata=np.nan, xRes=ROI_cellsize, yRes=ROI_cellsize)
                        temp_ds2 = gdal.Warp('/vsimem/' + f'{str(zvalue)}_{str(doy)}_temp2.vrt', os.path.join('/vsimem/' + f'{str(zvalue)}_{str(doy)}_temp.vrt'), srcSRS=crs,
                                             cropToCutline=True, cutlineDSName=ROI, outputType=gdal.GDT_Float32, dstNodata=np.nan, xRes=ROI_cellsize, yRes=ROI_cellsize)
                        temp_ds3 = gdal.Translate(os.path.join(zvalue_lapgrd_path, f'{str(zvalue)}_{str(doy)}.TIF'), f'/vsimem/{str(zvalue)}_{str(doy)}_temp2.vrt', options=topts)
                        gdal.Unlink('/vsimem/' + f'{str(zvalue)}_{str(doy)}_temp.vrt')
                        gdal.Unlink('/vsimem/' + f'{str(zvalue)}_{str(doy)}_temp2.vrt')
                        temp_ds = None
                        temp_ds2 = None
                        temp_ds3 = None

                    # Remove redundant fileD
                    try:
                        os.remove(output_res)
                        os.remove(output_grd)
                    except:
                        print('Failed to delete file')

                    print(f'Finish executing the LAPGRD procedure for {str(zvalue)} of \033[1;31m{str(doy)}\033[0m in \033[1;34m{str(time.time() - t1)[0:7]}\033[0m s')
                except:
                    print(traceback.format_exc())
                    print(f'Failed to execute the LAPGRD procedure for {str(zvalue)} of \033[1;31m{str(doy)}\033[0m.')
        except:
            print(traceback.format_exc())

    def _process_raster2dc_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('inherit_from_logfile', 'ROI', 'ROI_name', 'dc_overwritten_para',
                                       'manually_remove_datelist', 'size_control_factor', 'ds2ras_method'):
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
        elif self.ROI_name is None and 'ROI' in kwargs.keys():
            self.ROI_name = kwargs['ROI'].split('\\')[-1].split('.')[0]
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
            if kwargs['ds2ras_method'] in self._ds2ras_method_tup:
                self._ds2ras_method = kwargs['ds2ras_method']
            else:
                raise TypeError('Please mention the ds2ras_method should be supported!')
        elif self._inherit_from_logfile:
            self._retrieve_para(['ds2ras_method'], protected_var=True)
            if self._ds2ras_method is None:
                self._ds2ras_method = 'IDW'
        else:
            raise TypeError('Please mention the ds2ras_method should be supported!')

    def raster2dc(self, zvalue_list: list, temporal_division=None, **kwargs):

        kwargs['inherit_from_logfile'] = True
        self._process_raster2dc_para(**kwargs)

        if temporal_division is not None and not isinstance(temporal_division, str):
            raise TypeError(f'The {temporal_division} should be a str!')
        elif temporal_division is None:
            temporal_division = 'all'
        elif temporal_division not in self._temporal_div_str:
            raise ValueError(f'The {temporal_division} is not supported!')

        for zvalue_temp in zvalue_list:
            # Create the output path
            output_path = f'{self.output_path}{self.ROI_name}_Denv_datacube\\' if self.ROI_name is not None else f'{self.output_path}Ori_Denv_datacube\\'
            bf.create_folder(output_path)

            # Obtain the input files
            if self._ds2ras_method == 'ANUSPLIN':
                input_folder = f'{self.output_path}{str(self.ROI_name)}_Denv_raster\\ANUSPLIN_LAPGRD\\{zvalue_temp}\\' if self.ROI_name is not None else f'{self.output_path}Ori_Denv_raster\\ANUSPLIN_LAPGRD\\{zvalue_temp}\\'
                input_files = bf.file_filter(input_folder, ['.TIF'], exclude_word_list=['aux', 'xml', 'ovr'])
            elif self._ds2ras_method == 'IDW':
                input_folder = f'{self.output_path}{str(self.ROI_name)}_Denv_raster\\IDW_{zvalue_temp}\\' if self.ROI_name is not None else f'{self.output_path}Ori_Denv_raster\\IDW_{zvalue_temp}\\'
                input_files = bf.file_filter(input_folder, ['.TIF'], exclude_word_list=['aux', 'xml', 'ovr'])
            else:
                raise Exception('The method is not supported!')

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

                if np.isnan(nodata_value):
                    array_temp[~np.isnan(array_temp)] = 1
                else:
                    array_temp[array_temp != nodata_value] = 1

                ds_temp.GetRasterBand(1).WriteArray(array_temp)
                ds_temp.FlushCache()
                ds_temp = None
            else:
                ds_temp = gdal.Open(ROI_tif_name)
                array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                nodata_value = ds_temp.GetRasterBand(1).GetNoDataValue()
            np.save(ROI_array_name, array_temp)

            # Create the metadata dic
            metadata_dic = {'ROI_name': self.ROI_name, 'index': zvalue_temp, 'Datatype': 'float', 'ROI': self.ROI,
                            'ROI_array': ROI_array_name, 'ROI_tif': ROI_tif_name, 'sdc_factor': True,
                            'coordinate_system': self.main_coordinate_system, 'size_control_factor': False,
                            'oritif_folder': input_folder, 'dc_group_list': None, 'tiles': None, 'timescale': temporal_division,
                            'Denv_factor': True, 'Zoffset': None, 'Nodata_value': None}

            year_range = [int(_.split(f'{str(zvalue_temp)}_')[1].split('.')[0]) // 1000 for _ in input_files]
            if temporal_division == 'year':
                time_range = np.unique(np.array([_ for _ in self.year_range[zvalue_temp] if _ in year_range])).tolist()
            elif temporal_division == 'month':
                time_range = []
                year_temp_list = np.unique(np.array([_ for _ in self.year_range[zvalue_temp] if _ in year_range])).tolist()
                for year_temp in year_temp_list:
                    for month_temp in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
                        time_range.append(str(year_temp) + str(month_temp))
            elif temporal_division == 'all':
                time_range = ['TIF']

            with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
                executor.map(self._raster2sdc, repeat(output_path), repeat(input_folder), time_range, repeat(zvalue_temp), repeat(metadata_dic))

    def _raster2sdc(self, output_path, input_folder, time_temp, zvalue_temp, metadata_dic, ):

        try:
            start_time = time.time()
            print(f'Start constructing the {str(time_temp)} {str(zvalue_temp)} sdc of {self.ROI_name}.')

            # Create the output path
            yearly_output_path = f'{output_path}{str(zvalue_temp)}\\{str(time_temp)}\\' if time_temp != 'TIF' else f'{output_path}{str(zvalue_temp)}\\'
            bf.create_folder(yearly_output_path)

            if not os.path.exists(f'{yearly_output_path}doy.npy') or not os.path.exists(f'{yearly_output_path}metadata.json'):

                # Determine the input files
                yearly_input_files = bf.file_filter(input_folder, ['.TIF', f'{str(zvalue_temp)}_{str(time_temp)}'], exclude_word_list=['aux', 'xml', 'ovr'],  and_or_factor='and')
                ds_ = gdal.Open(yearly_input_files[0])
                arr_ = ds_.GetRasterBand(1).ReadAsArray()
                nodata_value = ds_.GetRasterBand(1).GetNoDataValue()
                rows, cols = ds_.RasterXSize, ds_.RasterYSize
                arr_type = arr_.dtype
                size_control = False
                if arr_type in (np.float16, np.float32, np.float64):
                    size_control = True
                    resize_factor = 100

                # Create the doy list
                doy_list = [int(filepath_temp.split(f'\\{str(zvalue_temp)}_')[-1][0:7]) for filepath_temp in yearly_input_files]

                # Determine whether the output matrix is huge and sparsify or not?
                mem = psutil.virtual_memory()
                dc_max_size = int(mem.free * 0.90)
                _huge_matrix = True if len(doy_list) * cols * rows * 2 > dc_max_size else False

                # Calculate sparse matrix
                if np.isnan(nodata_value):
                    sparsify = np.sum(np.isnan(arr_)) / (arr_.shape[0] * arr_.shape[1])
                else:
                    sparsify = np.sum(arr_ == nodata_value) / (arr_.shape[0] * arr_.shape[1])
                _sparse_matrix = True if sparsify > 0.9 else False

                if _sparse_matrix:
                    data_cube = NDSparseMatrix()
                    invalid_doy_list = []
                    for doy_ in doy_list:
                        try:
                            t1 = time.time()
                            if not os.path.exists(f"{input_folder}{zvalue_temp}_{str(doy_)}.TIF"):
                                raise Exception(f'The {zvalue_temp}_{str(doy_)} is not properly generated!')
                            else:
                                ds_temp = gdal.Open(f"{input_folder}{zvalue_temp}_{str(doy_)}.TIF")
                                array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                                if np.isnan(nodata_value):
                                    array_temp[np.isnan(array_temp)] = 0
                                else:
                                    array_temp[array_temp == nodata_value] = 0

                                if size_control:
                                    array_temp = array_temp * resize_factor
                                    array_temp = array_temp.astype(np.int16)

                            sm_temp = sm.csr_matrix(array_temp)
                            data_cube.append(sm_temp, name=doy_)
                            if sm_temp.data.shape[0] == 0:
                                invalid_doy_list.append(doy_)
                            print(f'Assemble the {str(doy_)} into the sdc using {str(time.time() - t1)[0:5]}s (layer {str(doy_list.index(doy_))} of {str(len(doy_list))})')
                            array_temp = None
                        except:
                            error_inf = traceback.format_exc()
                            invalid_doy_list.append(doy_)
                            print(error_inf)

                    # remove invalid layers
                    for doy_ in invalid_doy_list:
                        if doy_ in data_cube.SM_namelist:
                            data_cube.remove_layer(doy_)
                        doy_list.remove(doy_)

                    # Save the sdc
                    np.save(f'{yearly_output_path}doy.npy', doy_list)
                    data_cube.save(f'{yearly_output_path}{zvalue_temp}_Denv_datacube\\')
                    metadata_dic['Nodata_value'] = 0
                else:
                    if _huge_matrix:
                        pass
                    else:
                        data_cube = NDSparseMatrix()
                        invalid_doy_list = []
                        for doy_ in doy_list:
                            try:
                                t1 = time.time()
                                if not os.path.exists(f"{input_folder}{zvalue_temp}_{str(doy_)}.TIF"):
                                    raise Exception(f'The {zvalue_temp}_{str(doy_)} is not properly generated!')
                                else:
                                    ds_temp = gdal.Open(f"{input_folder}{zvalue_temp}_{str(doy_)}.TIF")
                                    array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                                    if np.isnan(nodata_value):
                                        array_temp[np.isnan(array_temp)] = 0
                                    else:
                                        array_temp[array_temp == nodata_value] = 0

                                    if size_control:
                                        array_temp = array_temp * resize_factor
                                        array_temp = array_temp.astype(np.int16)

                                sm_temp = sm.csr_matrix(array_temp)
                                data_cube.append(sm_temp, name=doy_)
                                if sm_temp.data.shape[0] == 0:
                                    invalid_doy_list.append(doy_)
                                print(f'Assemble the {str(doy_)} into the sdc using {str(time.time() - t1)[0:5]}s (layer {str(doy_list.index(doy_))} of {str(len(doy_list))})')
                                array_temp = None
                            except:
                                error_inf = traceback.format_exc()
                                invalid_doy_list.append(doy_)
                                print(error_inf)

                        # remove invalid layers
                        for doy_ in invalid_doy_list:
                            if doy_ in data_cube.SM_namelist:
                                data_cube.remove_layer(doy_)
                            doy_list.remove(doy_)

                        # Save the sdc
                        np.save(f'{yearly_output_path}doy.npy', doy_list)
                        data_cube.save(f'{yearly_output_path}{zvalue_temp}_Denv_datacube\\')
                        metadata_dic['Nodata_value'] = 0

                # Save the metadata dic
                if size_control:
                    metadata_dic['size_control_factor'] = True
                metadata_dic['sparse_matrix'], metadata_dic['huge_matrix'] = _sparse_matrix, _huge_matrix
                metadata_dic['timerange'] = time_temp
                with open(f'{yearly_output_path}metadata.json', 'w') as js_temp:
                    json.dump(metadata_dic, js_temp)

            print(f'Finish constructing the {str(time_temp)} {str(zvalue_temp)} sdc of {self.ROI_name} in \033[1;31m{str(time.time() - start_time)} s\033[0m.')
        except:
            print(traceback.format_exc())
            raise Exception('Error during the datacube construction')

    @save_log_file
    def interpolate_gridded_climate_data(self, interpolate_method, ROI=None, zvalue=None, date_range=None, output_path=None, bulk=True, generate_shp=True, generate_ras=True):

        # Generate climate raster
        # Define the interpolate method
        if interpolate_method not in self._ds2ras_method_tup:
            raise ValueError(f'The {interpolate_method} is not supported for ds2raster!')
        else:
            self._ds2ras_method = interpolate_method

        # Identify the zvalue
        if zvalue is None:
            zvalue = self.zvalue
        elif isinstance(zvalue, list):
            zvalue = [zvalue_ for zvalue_ in zvalue if zvalue_ in self.zvalue]
        elif isinstance(zvalue, str):
            if zvalue not in self.zvalue:
                raise Exception(f'The {str(zvalue)} is not supported!')
        else:
            raise TypeError('The input zvalue is under wrong type!')

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
            date_range = {key: self.date_range[key] for key in self.date_range if key in zvalue}
        elif isinstance(date_range, list) and len(date_range) == 2:
            try:
                start_date, end_date = int(date_range[0]), int(date_range[1])
                start_month, end_month = int(start_date // 100), int(end_date // 100)
                if start_date > end_date:
                    raise ValueError('The end date is smaller than start date!')
                else:
                    date_range = {key: self.date_range[key] for key in self.date_range if key in zvalue}
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
            self.raster_output_path = f'{self.output_path}{interpolate_method}_gridfile\\'
            self.shp_output_path = f'{self.output_path}{interpolate_method}_shpfile\\'
        else:
            if not os.path.exists(output_path):
                raise ValueError('The output path does not exist')
            else:
                self.raster_output_path = os.path.join(self.output_path, f'{interpolate_method}_gridfile\\')
                self.shp_output_path = os.path.join(self.work_env, f'{interpolate_method}_shpfile\\')
        bf.create_folder(self.raster_output_path)
        bf.create_folder(self.shp_output_path)

        for zvalue_ in zvalue:
            # Generate the shpfile
            zvalue_date_range = date_range[zvalue_]
            if generate_shp:
                if bulk:
                    with concurrent.futures.ProcessPoolExecutor() as exe:
                        exe.map(self._generate_shpfile, zvalue_date_range, repeat(zvalue_), repeat(self.shp_output_path))
                else:
                    for date_ in zvalue_date_range:
                        self._generate_shpfile(date_, zvalue, self.shp_output_path)

            # Generate the gridded file
            if generate_ras:
                shpfile_list = bf.file_filter(os.path.join(self.shp_output_path, f'{zvalue_}\\'), ['.shp'])
                shpfile_valid_list = [shp_ for shp_ in shpfile_list if int(np.floor(bf.doy2date(int(shp_.split(f'{zvalue_}_')[-1].split('.')[0])) / 100)) in zvalue_date_range]
                zvalue = [_[2] for _ in self._zvalue_dic[zvalue_]]

                if bulk:
                    with concurrent.futures.ProcessPoolExecutor() as exe:
                        exe.map(self._shpfile2gridfile, shpfile_valid_list, repeat(zvalue), repeat(ROI_inform), repeat(self.raster_output_path))
                else:
                    for shpfile_ in shpfile_valid_list:
                        self._shpfile2gridfile(shpfile_, zvalue, ROI_inform, self.raster_output_path)

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

    def _generate_shpfile(self, zvalue_year_month, zvalue, outputpath):

        try:
            # Check the output path
            if not os.path.exists(outputpath):
                raise ValueError(f'The {outputpath} is not valid')
            zvalue_output_path = os.path.join(outputpath, f'{zvalue}\\')
            bf.create_folder(zvalue_output_path)

            csv_file = [csv_ for csv_ in self.csv_files if str(zvalue_year_month) in csv_ and zvalue in csv_]
            if len(csv_file) != 1:
                raise Exception('Code error!')

            df_name_list = ['Station_id', 'Lat', 'Lon', 'Alt', 'DOY']
            df_name_list.extend([__[2] for __ in self._zvalue_dic[zvalue]])
            df_temp = pd.read_table(csv_file[0], delim_whitespace=True, header=None)
            column_name_all = [_[2] for _ in self._zvalue_dic[zvalue]]
            column_name_list = ['None' for _ in range(len(df_temp.columns))]
            for column_ in range(len(df_temp.columns)):
                if column_ < len(self._cma_header_):
                    column_name_list[column_] = self._cma_header_[column_]
                elif column_ in [_[0] for _ in self._zvalue_dic[zvalue]]:
                    pos = [_[0] for _ in self._zvalue_dic[zvalue]].index(column_)
                    column_name_list[column_] = self._zvalue_dic[zvalue][pos][2]
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
            for _ in self._zvalue_dic[zvalue]:
                header.append(_[2])

            # Get the geodf itr through date
            doy_list = pd.unique(df_temp['DOY'])
            for doy in doy_list:
                t1 = time.time()
                print(f'Start processing the {str(zvalue)} data of \033[1;31m{str(doy)}\033[0m')
                if not os.path.exists(os.path.join(zvalue_output_path, f'{str(zvalue)}_{str(doy)}.shp')):
                    pd_temp = df_temp[df_temp['DOY'] == doy][header]
                    for _ in self._zvalue_dic[zvalue]:
                        pd_temp[_[2]] = pd_temp[_[2]].astype(np.float32)
                        pd_temp[_[2]] = pd_temp[_[2]].replace(_[3], np.nan)
                        pd_temp[_[2]] = pd_temp[_[2]] * _[1]
                    geodf_temp = gp.GeoDataFrame(pd_temp, geometry=gp.points_from_xy(pd_temp['Lon'], pd_temp['Lat']), crs="EPSG:4326")
                    geodf_temp.to_file(os.path.join(zvalue_output_path, f'{str(zvalue)}_{str(doy)}.shp'), encoding='gbk')
                print(f'Finish generating the {str(zvalue)} shpfile of \033[1;31m{str(doy)}\033[0m in \033[1;34m{str(time.time() - t1)[0:7]}\033[0m s')
        except:
            print(traceback.format_exc())

    def _shpfile2gridfile(self, shpfile: str, zvalue_list: list, ROI: list, output_path: str):

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


if __name__ == '__main__':

    # station_rec = [Polygon([(441536.34182, 3457208.02321), (1104536.34182, 3457208.02321), (1104536.34182, 3210608.02321), (441536.34182, 3210608.02321)])]
    # geometry = gp.GeoDataFrame({'id':[1]}, geometry=station_rec, crs='EPSG:32649')
    # geometry.to_file('G:\\A_Landsat_Floodplain_veg\\ROI_map\\weather_boundary.shp')

    # Climate ds
    # ds_temp2 = CMA_ds('G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2015\\')
    # ds_temp = CMA_ds('G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\400_control_station_1950_2020\\')
    # ds_temp.append_ds(ds_temp2, 'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2020\\')
    # ds_temp = CMA_ds('G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2020\\')
    # ds_temp2 = CMA_ds('G:\\A_Climatology_dataset\\station_dataset\\Qweather_dataset\\Qweather_CMA_standard\\')
    # ds_temp.append_ds(ds_temp2, 'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\\')

    ds_temp = CMA_ds('G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\\')
    # ds_temp.anusplin('G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020_UTM.shp', 'G:\\A_Climatology_dataset\\supplement_DEM\\DEM\\alos_dem300.txt',
    #                  mask=None, zvalue=['PRE'], date_range=[19850101, 20231231], bulk=True)
    ds_temp.raster2dc(['SSD'], ROI='G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020_UTM.shp', ds2ras_method='ANUSPLIN', temporal_division='year')

    # ds_temp = gdal.Open('G:\\A_GEDI_Floodplain_vegh\\S2_all\\Sentinel2_L2A_Output\\ROI_map\\MYZR_FP_2020_map.TIF')
    # bounds_temp = bf.raster_ds2bounds('G:\\A_GEDI_Floodplain_vegh\\S2_all\\Sentinel2_L2A_Output\\ROI_map\\MYZR_FP_2020_map.TIF')
    # size = [ds_temp.RasterYSize, ds_temp.RasterXSize]
    # ds_temp = CMA_ds('G:\\A_Landsat_Floodplain_veg\\Data_cma\\')
    # ds_temp.generate_climate_ras(ROI='G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020_map.TIF', zvalue=['TEM'], date_range=[19850101, 20201231], generate_shp=True, bulk=True)
    # ds_temp.ds2raster(['TEMP'], ROI='G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020.shp', raster_size=size, ds2ras_method='IDW', bounds=bounds_temp)
    # ds_temp.raster2dc(['TEMP'], temporal_division='year')