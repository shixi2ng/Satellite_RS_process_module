from River_GIS.utils import *
import time
import numpy as np
from osgeo import gdal, osr
import basic_function as bf
import pandas as pd
import geopandas as gp
import os
import traceback
from datetime import date
import copy
import scipy.sparse as sm
from NDsm import NDSparseMatrix
from Landsat_toolbox.Landsat_main_v2 import Landsat_dc
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import shapely
from shapely import LineString, Point, Polygon
import concurrent.futures
from itertools import repeat
import rasterio
import psutil
import json


topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])


def multiple_concept_model(year, thal, ):
    hydrodc1 = HydroDatacube()
    hydrodc1.from_hydromatrix(f'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\hydrodatacube\\{str(year)}\\')
    hydrodc1.simplified_conceptual_inundation_model(
        'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\ele_pretgd4model.TIF', thal,
        'G:\A_Landsat_Floodplain_veg\Water_level_python\inundation_status\\prewl_predem\\')


class RiverChannel2D():
    def __init__(self):
        # Define the dem factor
        self.DEM_arr = None

        # Define the work env
        self.work_env = None


class DEM(object):

    def __init__(self):
        # Define the dem factor
        self.DEM_arr = None

        # Define the work env
        self.work_env = None

    def from_tiffile(self, tiffile: str):

        # Path exists
        if os.path.exists(tiffile):
            pass


class HydrometricStationData(object):

    def __init__(self):

        # Define the work path
        self.work_env = None

        # Define the property
        self.hydrometric_id = []
        self.station_namelist = []
        self.cross_section_namelist = []
        self.water_level_offset = {}
        self.hydrological_inform_dic = {}

    def import_from_standard_excel(self, file_name: str, cross_section_name: str, water_level_offset = None):

        self.work_env = bf.Path(os.path.dirname(file_name)).path_name
        hydrometric_id, station_name, cs_name, wl_offset = None, None, None, None

        if isinstance(file_name, str) and os.path.exists(file_name):
            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                df_temp = pd.read_excel(file_name)
            elif file_name.endswith('.csv'):
                df_temp = pd.read_csv(file_name)
            else:
                raise TypeError('Please make sure the standard file is csv or excel file')

            file_name = file_name.split('\\')[-1]
            if '_' not in file_name or len(file_name.split('_')) < 3:
                raise ValueError('Please make sure the filename is under id_stationname_startyear_endyear.xlsx format!')
            else:
                if file_name.split('_')[0].isnumeric():
                    hydrometric_id = file_name.split('_')[0]
                else:
                    raise ValueError('Please make sure the hydrometric station id is firstly presented!')
                station_name = file_name.split('_')[1]
                cs_name = cross_section_name

            if water_level_offset is None:
                water_level_offset = np.nan
            elif isinstance(water_level_offset, float):
                water_level_offset = water_level_offset
            elif isinstance(water_level_offset, str) and water_level_offset.isnumeric():
                water_level_offset = np.float(water_level_offset)
            else:
                raise TypeError('The water level offset not under right type!')

            wl_start_index = df_temp.loc[df_temp[df_temp.keys()[0]] == station_name].index
            year_list_, month_list_, day_list_, doy_list_, flow_list_, sed_con_list_, sed_flux_list_, water_level_list_ = [], [], [], [], [], [], [], []
            for _ in wl_start_index:
                itr = 1
                year_list, month_list, day_list, doy_list, flow_list, sed_con_list, sed_flux_list, water_level_list = [], [], [], [], [], [], [], []
                while itr <= 366:
                    if _ + itr <= df_temp.shape[0] - 1 and df_temp[df_temp.keys()[0]][_ + itr] <= 366:
                        try:
                            if df_temp[df_temp.keys()[1]][_ + itr] in range(1900, 2100) and df_temp[df_temp.keys()[2]][_ + itr] in range(1, 13) and df_temp[df_temp.keys()[2]][_ + itr] in range(1, 32):
                                year_list.append(df_temp[df_temp.keys()[1]][_ + itr])
                                month_list.append(df_temp[df_temp.keys()[2]][_ + itr])
                                day_list.append(df_temp[df_temp.keys()[3]][_ + itr])
                                doy_list.append(df_temp[df_temp.keys()[1]][_ + itr] * 1000 + (datetime(year=df_temp[df_temp.keys()[1]][_ + itr], month=df_temp[df_temp.keys()[2]][_ + itr], day=df_temp[df_temp.keys()[3]][_ + itr]).toordinal()
                                                                                              - datetime(year=df_temp[df_temp.keys()[1]][_ + itr], month=1, day=1).toordinal() + 1))
                                try:
                                    value_up = float(df_temp[df_temp.keys()[4]][_ + itr - 1])
                                except (ValueError, KeyError):
                                    value_up = np.nan
                                try:
                                    value_down = float(df_temp[df_temp.keys()[4]][_ + itr + 1])
                                except (ValueError, KeyError):
                                    value_down = np.nan
                                try:
                                    value_mid = float(df_temp[df_temp.keys()[4]][_ + itr])
                                except ValueError:
                                    value_mid = np.nan

                                if value_mid == 0 and (value_up == 0 or value_down == 0) or np.isnan(value_mid) and (np.isnan(value_up) or np.isnan(value_down)):
                                    flow_list.append(np.nan)
                                else:
                                    flow_list.append(np.float32(value_mid))

                                try:
                                    value_up = float(df_temp[df_temp.keys()[5]][_ + itr - 1])
                                except (ValueError, KeyError):
                                    value_up = np.nan
                                try:
                                    value_down = float(df_temp[df_temp.keys()[5]][_ + itr + 1])
                                except (ValueError, KeyError):
                                    value_down = np.nan
                                try:
                                    value_mid = float(df_temp[df_temp.keys()[5]][_ + itr])
                                except ValueError:
                                    value_mid = np.nan

                                if value_mid == 0 and (value_up == 0 or value_down == 0) or np.isnan(value_mid) and (np.isnan(value_up) or np.isnan(value_down)):
                                    sed_con_list.append(np.nan)
                                else:
                                    sed_con_list.append(np.float32(value_mid))

                                try:
                                    value_up = float(df_temp[df_temp.keys()[6]][_ + itr - 1])
                                except (ValueError, KeyError):
                                    value_up = np.nan
                                try:
                                    value_down = float(df_temp[df_temp.keys()[6]][_ + itr + 1])
                                except (ValueError, KeyError):
                                    value_down = np.nan
                                try:
                                    value_mid = float(df_temp[df_temp.keys()[6]][_ + itr])
                                except ValueError:
                                    value_mid = np.nan
                                if value_mid == 0 and (value_up == 0 or value_down == 0) or np.isnan(value_mid) and (np.isnan(value_up) or np.isnan(value_down)):
                                    sed_flux_list.append(np.nan)
                                else:
                                    sed_flux_list.append(np.float32(value_mid))

                                try:
                                    value_up = float(df_temp[df_temp.keys()[7]][_ + itr - 1])
                                except (ValueError, KeyError):
                                    value_up = np.nan
                                try:
                                    value_down = float(df_temp[df_temp.keys()[7]][_ + itr + 1])
                                except (ValueError, KeyError):
                                    value_down = np.nan
                                try:
                                    value_mid = float(df_temp[df_temp.keys()[7]][_ + itr])
                                except ValueError :
                                    value_mid = np.nan
                                if value_mid == 0 and (value_up == 0 or value_down == 0) or np.isnan(value_mid) and (np.isnan(value_up) or np.isnan(value_down)):
                                    water_level_list.append(np.nan)
                                else:
                                    water_level_list.append(np.float32(value_mid))
                        except:
                            print(traceback.format_exc())
                            print(f'The column {str(_)} for {station_name} is not imported')

                    itr += 1
                year_list_.extend(year_list)
                month_list_.extend(month_list)
                day_list_.extend(day_list)
                doy_list_.extend(doy_list)
                flow_list_.extend(flow_list)
                sed_con_list_.extend(sed_con_list)
                sed_flux_list_.extend(sed_flux_list)
                water_level_list_.extend(water_level_list)
            dic = {'year': year_list_, 'month': month_list_, 'day': day_list_, 'doy': doy_list_,
                   'flow/m3/s': flow_list_, 'sediment_concentration/kg/m3': sed_con_list_, 'sediment_fluxes': sed_flux_list_,
                   'water_level/m': water_level_list_}
            hydrological_inform_df = pd.DataFrame(dic)
            self.hydrometric_id.append(hydrometric_id)
            self.station_namelist.append(station_name)
            self.cross_section_namelist.append(cs_name)
            self.water_level_offset[station_name] = water_level_offset
            self.hydrological_inform_dic[station_name] = hydrological_inform_df
        else:
            raise Exception('Please input an existing file')

    def to_csvs(self, output_path: str = None):

        # Export 2 shpfile
        if output_path is None:
            output_path = self.work_env + 'standard_csv\\'
        else:
            output_path = bf.Path(output_path).path_name
        bf.create_folder(output_path)

        for _ in range(len(self.hydrometric_id)):
            hydrometric_id_temp = self.hydrometric_id[_]
            station_name_temp = self.station_namelist[_]
            cross_section_temp = self.cross_section_namelist[_]
            water_level_offset = self.water_level_offset[station_name_temp]
            hydrological_inform = self.hydrological_inform_dic[station_name_temp]

            if isinstance(hydrological_inform, pd.DataFrame):
                hydrological_inform.to_csv(f'{output_path}{str(hydrometric_id_temp)}_{str(station_name_temp)}_{str(cross_section_temp)}_{str(water_level_offset)}.csv', encoding='utf-8', index=False)
            else:
                raise TypeError('Code input wrong type hydrological df!')

    def cs_wl(self, thal, cs_name, date_, ):

        if isinstance(date_, int) and 19000000 < date_ < 21000000:
            year_ = date_ // 10000
            month_ = (date_ - year_ * 10000) // 100
            day_ = date_ - year_ * 10000 - month_ * 100
            doy = date(year=year_, month=month_, day=day_).toordinal() - date(year=year_, month=1, day=1).toordinal() + 1
            doy = year_ * 1000 + doy
        else:
            raise Exception('Please input valid date')

        if cs_name not in thal.Thalweg_cs_namelist:
            raise Exception('Please input the right cross section!')
        else:
            cs_pos = thal.Thalweg_cs_namelist.index(cs_name)
            _ = copy.deepcopy(cs_pos) - 1

            while _ >= 0:
                if thal.Thalweg_cs_namelist[_] in self.cross_section_namelist:
                    station_name = self.station_namelist[self.cross_section_namelist.index(thal.Thalweg_cs_namelist[_])]
                    hydro_doy = np.array(self.hydrological_inform_dic[station_name]['doy'])
                    if doy in hydro_doy:
                        wl_st = np.array(self.hydrological_inform_dic[station_name][self.hydrological_inform_dic[station_name]['doy'] == doy]['water_level/m'])[0]
                        wl_st = wl_st + self.water_level_offset[station_name]
                        if thal.smoothed_Thalweg is not None:
                            wl_st_dis = dis2points_via_line(thal.smoothed_Thalweg.coords[thal.smoothed_cs_index[cs_pos]], thal.smoothed_Thalweg.coords[thal.smoothed_cs_index[_]], thal.smoothed_Thalweg)
                        else:
                            wl_st_dis = dis2points_via_line(thal.Thalweg_Linestring.coords[cs_pos],thal.Thalweg_Linestring.coords[_], thal.Thalweg_Linestring)
                        break
                _ -= 1

            _ = copy.deepcopy(cs_pos) + 1
            while _ <= len(thal.Thalweg_cs_namelist):
                if thal.Thalweg_cs_namelist[_] in self.cross_section_namelist:
                    station_name = self.station_namelist[self.cross_section_namelist.index(thal.Thalweg_cs_namelist[_])]
                    hydro_doy = np.array(self.hydrological_inform_dic[station_name]['doy'])
                    if doy in hydro_doy:
                        wl_ed = np.array(self.hydrological_inform_dic[station_name][self.hydrological_inform_dic[station_name]['doy'] == doy]['water_level/m'])[0]
                        wl_ed = wl_ed + self.water_level_offset[station_name]
                        if thal.smoothed_Thalweg is not None:
                            wl_ed_dis = dis2points_via_line(thal.smoothed_Thalweg.coords[thal.smoothed_cs_index[cs_pos]], thal.smoothed_Thalweg.coords[thal.smoothed_cs_index[_]], thal.smoothed_Thalweg)
                        else:
                            wl_ed_dis = dis2points_via_line(thal.Thalweg_Linestring.coords[cs_pos],thal.Thalweg_Linestring.coords[_], thal.Thalweg_Linestring)
                        break
                _ += 1

            wl_out = wl_st + (wl_ed - wl_st) * (wl_st_dis) / (wl_ed_dis + wl_st_dis)
            print(f'{str(cs_name)}__{str(date_)}')
            print(f'{str(wl_out)}')

    def linear_comparison(self, station_name, thal, year,):

        s = [int(_) for _ in self.hydrometric_id]

        if station_name not in self.station_namelist:
            raise Exception('Please input the right station!')
        elif int(self.hydrometric_id[self.station_namelist.index(station_name)]) == min(s) or int(self.hydrometric_id[self.station_namelist.index(station_name)]) == max(s):
            raise Exception('Linear comparison is not supported for start and end station!')
        else:

            interpolated_list = []
            guaged_list = []
            if isinstance(year, int) and year in self.hydrological_inform_dic[station_name]['year']:
                year = [year]
            elif isinstance(year, list):
                for _ in year:
                    if _ not in self.hydrological_inform_dic[station_name]['year']:
                        print('The year is not valid')
                        return

            for year_ in year:
                if year_ in np.array(self.hydrological_inform_dic[station_name]['year']):
                    cs = self.cross_section_namelist[self.station_namelist.index(station_name)]
                    cs_wl = np.array(self.hydrological_inform_dic[station_name][self.hydrological_inform_dic[station_name]['year'] == year_]['water_level/m'])
                else:
                    print('The year is not valid')
                    return

                station_id = int(self.hydrometric_id[self.station_namelist.index(station_name)])
                station_minr = [int(_) for _ in self.hydrometric_id if int(_) < station_id]
                station_maxr = [int(_) for _ in self.hydrometric_id if int(_) > station_id]
                station_minr = -np.sort(-np.array(station_minr))
                station_maxr = np.sort(np.array(station_maxr))

                for _ in station_minr:
                    station_name_ = self.station_namelist[self.hydrometric_id.index(str(_))]
                    if year_ in np.array(self.hydrological_inform_dic[station_name_]['year']):
                        start_cs = self.cross_section_namelist[self.station_namelist.index(station_name_)]
                        start_cs_wl = np.array(self.hydrological_inform_dic[station_name_][self.hydrological_inform_dic[station_name_]['year'] == year_]['water_level/m'])
                        break

                    if _ == station_minr[-1]:
                        raise Exception('No valid start cs')

                for _ in station_maxr:
                    station_name_ = self.station_namelist[self.hydrometric_id.index(str(_))]
                    if year_ in np.array(self.hydrological_inform_dic[station_name_]['year']):
                        end_cs = self.cross_section_namelist[self.station_namelist.index(station_name_)]
                        end_cs_wl = np.array(self.hydrological_inform_dic[station_name_][self.hydrological_inform_dic[station_name_]['year'] == year_]['water_level/m'])
                        break

                    if _ == station_maxr[-1]:
                        raise Exception('No valid end cs')

                if end_cs_wl.shape[0] != start_cs_wl.shape[0] or start_cs_wl.shape[0] != cs_wl.shape[0]:
                    return

                if cs in thal.Thalweg_cs_namelist:
                    if thal.smoothed_Thalweg is not None:
                        cs_index = thal.smoothed_cs_index[thal.Thalweg_cs_namelist.index(cs)]
                    else:
                        cs_index = thal.Thalweg_cs_namelist.index(cs)
                else:
                    raise Exception('Cross section is not in thalweg!')

                if start_cs in thal.Thalweg_cs_namelist:
                    if thal.smoothed_Thalweg is not None:
                        scs_index = thal.smoothed_cs_index[thal.Thalweg_cs_namelist.index(start_cs)]
                        start_dis = dis2points_via_line(thal.smoothed_Thalweg.coords[cs_index], thal.smoothed_Thalweg.coords[scs_index], thal.smoothed_Thalweg)
                    else:
                        scs_index = thal.Thalweg_cs_namelist.index(start_cs)
                        start_dis = dis2points_via_line(thal.Thalweg_Linestring.coords[cs_index],  thal.Thalweg_Linestring.coords[scs_index], thal.Thalweg_Linestring)
                else:
                    raise Exception('Start cross section is not in thalweg!')

                if end_cs in thal.Thalweg_cs_namelist:
                    if thal.smoothed_Thalweg is not None:
                        ecs_index = thal.smoothed_cs_index[thal.Thalweg_cs_namelist.index(end_cs)]
                        end_dis = dis2points_via_line(thal.smoothed_Thalweg.coords[cs_index], thal.smoothed_Thalweg.coords[ecs_index], thal.smoothed_Thalweg)
                    else:
                        ecs_index = thal.Thalweg_cs_namelist.index(end_cs)
                        end_dis = dis2points_via_line(thal.Thalweg_Linestring.coords[cs_index], thal.Thalweg_Linestring.coords[ecs_index], thal.Thalweg_Linestring)
                else:
                    raise Exception('End cross section is not in thalweg!')

                linear_interpolated_wl = start_cs_wl + (end_cs_wl - start_cs_wl) * (start_dis) / (end_dis + start_dis)

                for _ in range(0, 365):
                    if abs(linear_interpolated_wl[_] - cs_wl[_]) > 5:
                        linear_interpolated_wl[_] = cs_wl[_]

                q = np.nanmean(linear_interpolated_wl[0: 365] - np.array(cs_wl)[0: 365]) / 2
                linear_interpolated_wl = linear_interpolated_wl - q

                interpolated_list.extend(linear_interpolated_wl[0: 365].tolist())
                guaged_list.extend(np.array(cs_wl)[0: 365].tolist())

        plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
        plt.rc('font', size=50)
        plt.rc('axes', linewidth=5)

        fig_temp, ax_temp = plt.subplots(figsize=(8 * len(year), 10), constrained_layout=True)
        ax_temp.plot(np.linspace(1, 365 * len(year) + 1, 365 * len(year)), interpolated_list, color=(1, 0, 0), lw=3, label='Interpolated stage')
        ax_temp.plot(np.linspace(1, 365 * len(year) + 1, 365 * len(year)), guaged_list, color=(0, 0, 1), lw=3, label='Gauged stage')
        q = 0
        year_list = []
        pos_list = []
        min_v = np.floor(min(min(interpolated_list), min(guaged_list)) / 2) * 2
        max_v = np.ceil(max(max(interpolated_list), max(guaged_list)) / 2) * 2
        for _ in year:
            q += 1
            ax_temp.plot(np.linspace(365 * q + 1, 365 * q + 1, 365), np.linspace(min_v, max_v, 365), lw=4, color=(0, 0, 0), ls='--')
            year_list.append(str(_))
            pos_list.append((q - 1) * 365 + 182)

        print(f'{station_name}')
        print(str(np.sqrt(np.nanmean((np.array(interpolated_list) - np.nanmean(np.array(interpolated_list) - np.array(guaged_list)) - np.array(guaged_list)) ** 2))))
        print(f'{str(np.nanmean(np.abs(np.array(guaged_list)[0: -1] - np.array(guaged_list)[1: ])))}')
        rmse = np.sqrt(np.nanmean((np.array(interpolated_list) - np.array(guaged_list)) ** 2))
        # add text box for the statistics
        stats = (f'ME = {np.nanmean(np.array(interpolated_list) - np.array(guaged_list)):.2f}\n'
                 f'MAE = {np.nanmean(abs(np.array(interpolated_list) - np.array(guaged_list))):.2f}\n'
                 f'RMSE = {rmse:.2f}')
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
        ax_temp.text(0.13, 0.1, stats, fontsize=50, bbox=bbox, transform=ax_temp.transAxes, horizontalalignment='right')
        ax_temp.legend(fontsize=50)
        ax_temp.set_xlim(1, 365 * len(year) + 1)
        ax_temp.set_ylim(min_v, max_v)
        ax_temp.set_xticks(pos_list)
        ax_temp.set_xticklabels(year_list)
        plt.savefig(f"E:\\A_Vegetation_Identification\\Paper\\Moderate_rev\Fig\\Method1\\{station_name}_{str(max(year))}_{str(min(year))}_rmse{str(rmse)[0:4]}.png", dpi=300)
        plt.close()
        fig_temp = None
        ax_temp = None

    def load_csvs(self, csv_filelist):
        for _ in csv_filelist:
            pass


class HydroDatacube(object):

    def __init__(self):

        # Define the hydroinform
        self.hydro_inform_dic = None

        # Define the hydrodatacube
        self.hydrodatacube = None
        self.year = None
        self.sparse_factor = None

    def merge_hydro_inform(self, hydro_ds: HydrometricStationData):

        # Create the hydro inform dic
        self.hydro_inform_dic = {}

        # Detect the datatype
        if not isinstance(hydro_ds, HydrometricStationData):
            raise TypeError('The hydro ds is not a standard hydrometric station data!')

        # Merge hydro inform
        for _ in hydro_ds.cross_section_namelist:
            wl_offset = hydro_ds.water_level_offset[hydro_ds.station_namelist[hydro_ds.cross_section_namelist.index(_)]]
            self.hydro_inform_dic[_] = hydro_ds.hydrological_inform_dic[hydro_ds.station_namelist[hydro_ds.cross_section_namelist.index(_)]]
            self.hydro_inform_dic[_]['water_level/m'] = self.hydro_inform_dic[_]['water_level/m'] + wl_offset

    def hydrodc_csv2matrix(self, outputfolder, hydroinform_csv):

        # Check if hydro station data is import
        if self.hydro_inform_dic is None:
            raise Exception('Please input the hydro inform first')

        # Check the output folder
        if not os.path.exists(outputfolder):
            bf.create_folder(outputfolder)
            outputfolder = bf.Path(outputfolder).path_name

        # Import hydroinform_csv
        if isinstance(hydroinform_csv, str):
            if not os.path.exists(hydroinform_csv):
                raise ValueError(f'The {str(hydroinform_csv)} does not exist!')
            elif hydroinform_csv.endswith('.xlsx') or hydroinform_csv.endswith('.xls'):
                hydroinform_df = pd.read_excel(hydroinform_csv)
            elif hydroinform_csv.endswith('.csv'):
                hydroinform_df = pd.read_csv(hydroinform_csv)
            else:
                raise TypeError(f'The hydroinform_csv should be a xlsx!')
        else:
            raise TypeError(f'The {str(hydroinform_csv)} should be a str!')

        # Read the header
        if hydroinform_csv.split('\\')[-1].startswith('hydro_dc'):
            try:
                Xsize, Ysize = int(hydroinform_csv.split('\\')[-1].split('_X_')[1].split('_')[0]), int(hydroinform_csv.split('\\')[-1].split('_Y_')[1].split('_')[0])
                header = hydroinform_csv.split(str(Ysize))[1].split('.')[0]
            except:
                raise Exception('Please make sure the file name is not manually changed')
        else:
            raise Exception('Please make sure the file name is not manually changed')

        # Get the year list
        hydroinform_df_list = []
        cpu_amount = os.cpu_count()
        size = int(np.ceil(hydroinform_df.shape[0] / cpu_amount))
        for _ in range(cpu_amount):
            if _ != cpu_amount - 1:
                hydroinform_df_list.append(list(hydroinform_df['yearly_wl'][_ * size: (_ + 1) * size]))
            else:
                hydroinform_df_list.append(list(hydroinform_df['yearly_wl'][_ * size: ]))

        with concurrent.futures.ProcessPoolExecutor() as exe:
            res = exe.map(process_hydroinform_df, hydroinform_df_list)

        yearly_hydroinform_all = []
        res = list(res)
        for _ in res:
            yearly_hydroinform_all.extend(_)
        hydroinform_df['yearly_wl'] = yearly_hydroinform_all

        # Get the year list and array size
        x_list, y_list = [], []
        year_list = [int(_[0]) for _ in yearly_hydroinform_all[0]]
        hydro_inform = list(hydroinform_df['yearly_wl'])
        hydro_inform_list = []
        for year in year_list:
            yearly_hydro = []
            with tqdm(total=len(hydro_inform), desc=f'Relist hydro inform of year {str(year)}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                for _ in hydro_inform:
                    # xy_list = [hydroinform_df['y'][hydro_inform.index(_)], hydroinform_df['x'][hydro_inform.index(_)]]
                    year_temp = _[year_list.index(year)]
                    # year_list.extend(xy_list)
                    yearly_hydro.append(year_temp)
                    pbar.update()
                hydro_inform_list.append(yearly_hydro)
            x_list.append(list(hydroinform_df['x']))
            y_list.append(list(hydroinform_df['y']))

        hydro_list = []
        for year in year_list:
            hydro_temp = {}
            for _ in self.hydro_inform_dic.keys():
                hydro_temp[_] = list(self.hydro_inform_dic[_][self.hydro_inform_dic[_]['year'] == year]['water_level/m'])
            hydro_list.append(hydro_temp)

        # Define the matrix
        mem = psutil.virtual_memory().available
        if hydroinform_df.shape[0] / (4827 * 16357) < 0.2 or Xsize * Ysize * 4 * 365 > psutil.virtual_memory().available:
            for year, hydro_dic, hydro_inform, x_l, y_l in zip(year_list, hydro_list, hydro_inform_list, x_list, y_list):
                # Define the sparse matrix
                doy_list = [year * 1000 + _ for _ in range(1, datetime(year=year + 1, month=1, day=1).toordinal() - datetime(year=year, month=1, day=1).toordinal() + 1)]
                sm_list = [sm.lil_matrix((Ysize, Xsize)) for _ in range(len(doy_list))]

                with tqdm(total=len(hydro_inform), desc=f'Generate hydro datacube',
                          bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                    for _ in range(len(hydro_inform)):
                        y, x = int(y_l[_]), int(x_l[_])
                        # print(str(y) + str(x))
                        wl_start_series = np.array(hydro_dic[hydro_inform[_][1]])
                        wl_end_series = np.array(hydro_dic[hydro_inform[_][2]])
                        # print(str(wl_start_series))
                        wl_start_dis = hydro_inform[_][3]
                        wl_end_dis = hydro_inform[_][4]
                        wl_inter = wl_start_series + (wl_end_series - wl_start_series) * wl_start_dis / (
                                    wl_start_dis + wl_end_dis)
                        # print(str(wl_inter))
                        # print(str(wl_end_dis))
                        for __ in range(len(wl_inter)):
                            sm_list[__][y, x] = wl_inter[__]
                        pbar.update()

                print(f'Start saving the hydro datacube of year {str(year)}!')
                st = time.time()
                # for _ in range(len(sm_list)):
                #     sm_list[_] = sm_list[_].tocsr()
                ND_temp = NDSparseMatrix(*sm_list, SM_namelist=doy_list)
                ND_temp.save(f'{outputfolder}{str(year)}\\')
                print(f'Finish saving the hydro datacube of year {str(year)} in {str(time.time() - st)}!')

        else:
            for year, hydro_dic, hydro_inform in zip(year_list, hydro_list, hydro_inform_list):
                doy_list = [year * 1000 + _ for _ in range(1, datetime(year=year + 1, month=1, day=1).toordinal() - datetime(year=year, month=1, day=1).toordinal() + 1)]
                dc = np.zeros((Ysize, Xsize, len(doy_list))) * np.nan
                np.save(f'{outputfolder}{str(year)}\\', dc)

    def from_hydromatrix(self, filepath):

        # Extract year inform
        start_time = time.time()
        self.year = int(filepath.split('\\')[-2])
        print(f"Start loading the Hydrodatacube of \033[1;31m{str(self.year)}\033[0m")

        # Import datacube
        if filepath.endswith('.npy'):
            self.hydrodatacube = np.load(filepath)
            self.sparse_factor = False
        else:
            try:
                self.hydrodatacube = NDSparseMatrix()
                self.hydrodatacube = self.hydrodatacube.load(filepath)
                self.sparse_factor = True
            except:
                raise TypeError('Please input correct type of hydro datacube')

        print(f'Finish loading the Hydrodatacube of \033[1;31m{str(self.year)}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def simplified_conceptual_inundation_model(self, demfile, thalweg_temp, output_path, inun_factor=True, construct_inunfac_dc_factor=True, meta_dic=None):

        # Check the thalweg
        if isinstance(thalweg_temp,Thalweg):
            thelwag_linesting = thalweg_temp.Thalweg_Linestring
        else:
            raise TypeError('Please input the thalweg with right type')

        # Check the consistency
        if self.hydrodatacube is None:
            raise Exception('Please import the datacube before the comparison')

        # create folder
        bf.create_folder(output_path)

        # Compare with dem
        if self.sparse_factor:
            cpu_amount = os.cpu_count() - 2
            itr = int(np.floor(self.hydrodatacube.shape[2]/cpu_amount))
            sm_list, nm_list = [], []
            for _ in range(cpu_amount):
                if _ != cpu_amount - 1:
                    nm_list.append(self.hydrodatacube.SM_namelist[_ * itr: (_ + 1) * itr])
                else:
                    nm_list.append(self.hydrodatacube.SM_namelist[_ * itr:])
                sm_list.append([self.hydrodatacube.SM_group[__] for __ in nm_list[-1]])

            with concurrent.futures.ProcessPoolExecutor() as exe:
                exe.map(concept_inundation_model, nm_list, sm_list, repeat(demfile), repeat(thelwag_linesting), repeat(output_path))

            # Create inun factor
            if inun_factor:
                bf.create_folder(f'{output_path}\\inundation_factor\\{str(self.year)}\\')
                dem_ds = gdal.Open(demfile)
                dem_arr = dem_ds.GetRasterBand(1).ReadAsArray()
                dem_nodata = np.nan

                inun_duration = np.zeros([self.hydrodatacube.shape[0], self.hydrodatacube.shape[1]])
                inun_mean_waterlevel = np.zeros([self.hydrodatacube.shape[0], self.hydrodatacube.shape[1]])
                inun_max_waterlevel = np.zeros([self.hydrodatacube.shape[0], self.hydrodatacube.shape[1]])

                if np.isnan(dem_nodata):
                    inun_duration[np.isnan(dem_arr)] = np.nan
                    inun_mean_waterlevel[np.isnan(dem_arr)] = np.nan
                    inun_max_waterlevel[np.isnan(dem_arr)] = np.nan
                else:
                    inun_duration[inun_duration == dem_nodata] = np.nan
                    inun_mean_waterlevel[inun_mean_waterlevel == dem_nodata] = np.nan
                    inun_max_waterlevel[inun_max_waterlevel == dem_nodata] = np.nan

                cpu_amount = int(os.cpu_count() / 2)
                indi_list_amount = int(np.floor(len(self.hydrodatacube.SM_namelist) / cpu_amount))
                sm_name_list, sm_group_list = [], []
                for _ in range(cpu_amount):
                    if _ != cpu_amount - 1:
                        sm_name_list.append(self.hydrodatacube.SM_namelist[_ * indi_list_amount: (_ + 1) * indi_list_amount])
                    else:
                        sm_name_list.append(self.hydrodatacube.SM_namelist[_ * indi_list_amount:])
                    sm_group_list.append([self.hydrodatacube.SM_group[__] for __ in sm_name_list[-1]])

                with concurrent.futures.ProcessPoolExecutor() as exe:
                    res = exe.map(generate_inundation_indicator, sm_name_list, sm_group_list, repeat(dem_arr), repeat(dem_nodata), repeat(output_path))

                res = list(res)
                for _ in res:
                    inun_duration = inun_duration + _[0]
                    inun_mean_waterlevel = inun_mean_waterlevel + _[1]
                    inun_max_waterlevel = np.nanmax([inun_max_waterlevel, _[2]], axis=0)
                exe = None

                inun_mean_waterlevel = inun_mean_waterlevel / inun_duration
                bf.write_raster(dem_ds, inun_duration, f'{output_path}\\inundation_factor\\{str(self.year)}\\', 'inun_duration.tif')
                bf.write_raster(dem_ds, inun_mean_waterlevel, f'{output_path}\\inundation_factor\\{str(self.year)}\\', 'inun_mean_wl.tif')
                bf.write_raster(dem_ds, inun_max_waterlevel, f'{output_path}\\inundation_factor\\{str(self.year)}\\', 'inun_max_wl.tif')

                if construct_inunfac_dc_factor:
                    if meta_dic is not None and isinstance(meta_dic, str) and meta_dic.endswith('.json'):
                        with open(meta_dic) as js_temp:
                            dc_metadata = json.load(js_temp)
                        metadata_dic = {'ROI_name': dc_metadata['ROI_name'], 'index': 'Inundation_factor', 'Datatype': 'float', 'ROI': dc_metadata['ROI'],
                                        'ROI_array': dc_metadata['ROI_array'], 'ROI_tif': dc_metadata['ROI_tif'], 'Inunfac_factor': True,
                                        'coordinate_system': dc_metadata['coordinate_system'], 'size_control_factor': False,
                                        'oritif_folder': f'{output_path}\\inundation_factor\\{str(self.year)}\\', 'dc_group_list': None, 'tiles': None,
                                        'Zoffset': None, 'sparse_matrix': True, 'huge_matrix': True, 'Nodata_value': np.nan}
                    else:
                        raise Exception('Please input related metadata dic before the dc construction')

                    bf.create_folder(f'{output_path}\\inundation_dc\\')
                    input_arr = {'inun_duration': inun_duration, 'inun_max_wl': inun_max_waterlevel, 'inun_mean_wl': inun_mean_waterlevel}
                    construct_inunfac_dc(input_arr, f'{output_path}\\inundation_dc\\', self.year, metadata_dic)

    def seq_simplified_conceptual_inundation_model(self, demfile, thalweg_temp, output_path, inun_factor = True):

        # Check the thalweg
        if isinstance(thalweg_temp,Thalweg):
            thelwag_linesting = thalweg_temp.Thalweg_Linestring
        else:
            raise TypeError('Please input the thalweg with right type')

        if demfile.endswith('.tif') or demfile.endswith('.TIF'):
            dem_file_ds = gdal.Open(demfile)
            dem_file_arr = dem_file_ds.GetRasterBand(1).ReadAsArray()
        else:
            raise TypeError('Please input the dem file with right type')

        # Check the consistency
        if self.hydrodatacube is None:
            raise Exception('Please import the datacube before the comparison')

        # create folder
        bf.create_folder(output_path)

        # Compare with dem
        if self.sparse_factor:
            for _ in range(len(self.hydrodatacube.SM_namelist)):
                wl_sm_, wl_nm_ = self.hydrodatacube.SM_group[self.hydrodatacube.SM_namelist[_]], self.hydrodatacube.SM_namelist[_]
                try:
                    if not os.path.exists(f'{output_path}\\inundation_final\\{str(wl_nm_)}.tif'):
                        if wl_sm_.shape[0] != dem_file_arr.shape[0]:
                            wl_sm_ = np.row_stack((wl_sm_.toarray(), np.zeros([dem_file_arr.shape[0] - wl_sm_.shape[0], wl_sm_.shape[1]])))

                        if wl_sm_.shape[1] != dem_file_arr.shape[1]:
                            wl_sm_ = np.column_stack(
                                (wl_sm_.toarray(), np.zeros([wl_sm_.shape[0], dem_file_arr.shape[1] - wl_sm_.shape[1]])))

                        inun_arr = np.array(wl_sm_ > dem_file_arr).astype(np.uint8)
                        bf.create_folder(f'{output_path}\\inundation_temp\\')
                        bf.write_raster(dem_file_ds, inun_arr, f'{output_path}\\inundation_temp\\', str(wl_nm_) + '.tif',
                                        raster_datatype=gdal.GDT_Byte)

                        src_temp = rasterio.open(f'{output_path}\\inundation_temp\\{str(wl_nm_)}.tif')
                        shp_dic = ({'properties': {'raster_val': int(v)}, 'geometry': s} for i, (s, v) in
                                   enumerate(rasterio.features.shapes(inun_arr, connectivity=8, transform=src_temp.transform)) if
                                   ~np.isnan(v))
                        meta = src_temp.meta.copy()
                        meta.update(compress='lzw')

                        shp_list = list(shp_dic)
                        nw_shp_list = []
                        shp_file = gp.GeoDataFrame.from_features(shp_list)
                        for __ in range(shp_file.shape[0]):
                            if shp_file['raster_val'][__] == 1:
                                if thelwag_linesting.intersects(shp_file['geometry'][__]):
                                    nw_shp_list.append(shp_list[__])
                        nw_shp_file = gp.GeoDataFrame.from_features(nw_shp_list)

                        bf.create_folder(f'{output_path}\\inundation_final\\')
                        with rasterio.open(f'{output_path}\\inundation_final\\{str(wl_nm_)}.tif', 'w+', **meta) as out:
                            out_arr = out.read(1)

                            # this is where we create a generator of geom, value pairs to use in rasterizing
                            shapes = ((geom, value) for geom, value in zip(nw_shp_file.geometry, nw_shp_file.raster_val))

                            burned = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=src_temp.transform)
                            out.write_band(1, burned)
                    print(f'The {str(wl_nm_)} is generated!')

                except:
                    print(traceback.format_exc())
                    print(f'The {str(wl_nm_)} is not generated!!!!!')

    def remove_depression(self):
        pass


class Inunfac_dc(object):

    ####################################################################################################
    # Inunfactor_dc represents "Inundation factor Datacube"
    # It normally contains inundation factor like inundation duration, inundation water level derived from the simplified flood inundation model, etc
    # And stack into a 3-D datacube type.
    # Currently, it was integrated into the River_GIS as a output data for inunfactor generation.
    ####################################################################################################

    def __init__(self, inunfactor_filepath, work_env=None):

        # Check the phemetric path
        self.Inunfac_dc_filepath = bf.Path(inunfactor_filepath).path_name

        # Init key var
        self.ROI_name, self.ROI, self.ROI_tif, self.ROI_array = None, None, None, None
        self.index, self.Datatype, self.coordinate_system = None, None, None
        self.dc_group_list, self.tiles = None, None
        self.sdc_factor, self.sparse_matrix, self.size_control_factor, self.huge_matrix = False, False, False, False
        self.Inunfac_factor, self.inunyear = False, None
        self.curfit_dic = {}

        # Init protected var
        self._support_inunfac_list = ['inun_duration', 'inun_max_wl', 'inun_mean_wl']

        # Check work env
        if work_env is not None:
            self._work_env = bf.Path(work_env).path_name
        else:
            self._work_env = bf.Path(os.path.dirname(os.path.dirname(self.Inunfac_dc_filepath))).path_name
        self.root_path = bf.Path(os.path.dirname(os.path.dirname(self._work_env))).path_name

        # Define the basic var name
        self._fund_factor = ('ROI_name', 'index', 'Datatype', 'ROI', 'ROI_array', 'inunyear',
                             'coordinate_system', 'oritif_folder', 'ROI_tif', 'sparse_matrix',
                             'huge_matrix', 'size_control_factor', 'dc_group_list', 'tiles', 'Nodata_value', 'Zoffset')

        # Read the metadata file
        metadata_file = bf.file_filter(self.Inunfac_dc_filepath, ['metadata.json'])
        if len(metadata_file) == 0:
            raise ValueError('There has no valid sdc or the metadata file of the sdc was missing!')
        elif len(metadata_file) > 1:
            raise ValueError('There has more than one metadata file in the dir')
        else:
            try:
                with open(metadata_file[0]) as js_temp:
                    dc_metadata = json.load(js_temp)

                if not isinstance(dc_metadata, dict):
                    raise Exception('Please make sure the metadata file is a dictionary constructed in python!')
                else:
                    # Determine whether this datacube is Inunfac or not
                    if 'Inunfac_factor' not in dc_metadata.keys():
                        raise TypeError('The Inunfac_factor was lost or it is not a Inunfac datacube!')
                    elif dc_metadata['Inunfac_factor'] is False:
                        raise TypeError(f'{self.Inunfac_dc_filepath} is not a Inunfac datacube!')
                    elif dc_metadata['Inunfac_factor'] is True:
                        self.Inunfac_factor = dc_metadata['Inunfac_factor']
                    else:
                        raise TypeError('The Inunfac factor was under wrong type!')

                    for dic_name in self._fund_factor:
                        if dic_name not in dc_metadata.keys():
                            raise Exception(f'The {dic_name} is not in the dc metadata, double check!')
                        else:
                            self.__dict__[dic_name] = dc_metadata[dic_name]
            except:
                print(traceback.format_exc())
                raise Exception('Something went wrong when reading the metadata!')

        start_time = time.time()
        print(f'Start loading the Inunfac datacube of {str(self.inunyear)} \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        # Read paraname file of the Inunfac datacube
        try:
            if self.Inunfac_factor is True:
                # Read paraname
                paraname_file = bf.file_filter(self.Inunfac_dc_filepath, ['paraname.npy'])
                if len(paraname_file) == 0:
                    raise ValueError('There has no paraname file or file was missing!')
                elif len(paraname_file) > 1:
                    raise ValueError('There has more than one paraname file in the Inunfac datacube dir')
                else:
                    paraname_list = np.load(paraname_file[0], allow_pickle=True)
                    self.paraname_list = [paraname for paraname in paraname_list]
            else:
                raise TypeError('Please input as a Inunfac datacube')
        except:
            raise Exception('Something went wrong when reading the paraname list!')

        # Read func dic
        try:
            if self.sparse_matrix and self.huge_matrix:
                if os.path.exists(self.Inunfac_dc_filepath + f'Inunfac_datacube\\'):
                    self.dc = NDSparseMatrix().load(self.Inunfac_dc_filepath + f'Inunfac_datacube\\')
                else:
                    raise Exception('Please double check the code if the sparse huge matrix is generated properly')
            elif not self.huge_matrix:
                self.dc_filename = bf.file_filter(self.Inunfac_dc_filepath, ['Inunfac_datacube.npy'])
                if len(self.dc_filename) == 0:
                    raise ValueError('There has no valid Inunfac datacube or the dc was missing!')
                elif len(self.dc_filename) > 1:
                    raise ValueError('There has more than one data file in the dc dir')
                else:
                    self.dc = np.load(self.dc_filename[0], allow_pickle=True)
            elif self.huge_matrix and not self.sparse_matrix:
                self.dc_filename = bf.file_filter(self.Inunfac_dc_filepath, ['sequenced_datacube', '.npy'], and_or_factor='and')
        except:
            print(traceback.format_exc())
            raise Exception('Something went wrong when reading the Inunfac datacube!')

        # autotrans sparse matrix
        if self.sparse_matrix and self.dc._matrix_type == sm.coo_matrix:
            self._autotrans_sparse_matrix()

        # Drop duplicate layers
        self._drop_duplicate_layers()

        # Backdoor metadata check
        self._backdoor_metadata_check()

        # Size calculation and shape definition
        self.dc_XSize, self.dc_YSize, self.dc_ZSize = self.dc.shape[1], self.dc.shape[0], self.dc.shape[2]
        if self.dc_ZSize != len(self.paraname_list):
            raise TypeError('The Inunfac datacube is not consistent with the paraname file')

        print(f'Finish loading the Inunfac datacube of  {str(self.inunyear)} \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def _autotrans_sparse_matrix(self):

        if not isinstance(self.dc, NDSparseMatrix):
            raise TypeError('The autotrans sparse matrix is specified for the NDsm!')

        for _ in self.dc.SM_namelist:
            if isinstance(self.dc.SM_group[_], sm.coo_matrix):
                self.dc.SM_group[_] = sm.csr_matrix(self.dc.SM_group[_])
                self.dc._update_size_para()
        self.save(self.Inunfac_dc_filepath)

    def _drop_duplicate_layers(self):
        for _ in self.paraname_list:
            if len([t for t in self.paraname_list if t == _]) != 1:
                pos = [tt for tt in range(len(self.paraname_list)) if self.paraname_list[tt] == _]
                if isinstance(self.dc, NDSparseMatrix):
                    self.dc.SM_namelist.pop(pos[-1])
                    self.paraname_list.pop(pos[-1])
                    self.dc._update_size_para()
                else:
                    self.dc = np.delete(self.dc, pos[-1], axis=2)
                    self.paraname_list.pop(pos[-1])

    def _backdoor_metadata_check(self):

        backdoor_issue = False
        # Check if metadata is valid and timeliness
        # Problem 1 dir change between multi-devices ROI ROI arr ROI tif
        if not os.path.exists(self.ROI_tif):
            self.ROI_tif = retrieve_correct_filename(self.ROI_tif)
            backdoor_issue = True

        if not os.path.exists(self.ROI_array):
            self.ROI_array = retrieve_correct_filename(self.ROI_array)
            backdoor_issue = True

        if not os.path.exists(self.ROI):
            self.ROI = retrieve_correct_filename(self.ROI)
            backdoor_issue = True

        if self.ROI_tif is None or self.ROI_array is None or self.ROI is None:
            raise Exception('Please manually change the roi path in the phemetric dc')

        # Problem 2
        if backdoor_issue:
            self.save(self.Inunfac_dc_filepath)
            self.__init__(self.Inunfac_dc_filepath)

    def __sizeof__(self):
        return self.dc.__sizeof__() + self.paraname_list.__sizeof__()

    def _add_layer(self, array, layer_name: str):

        # Process the layer name
        if not isinstance(layer_name, str):
            raise TypeError('Please input the adding layer name as a string！')
        elif not layer_name.startswith(str(self.inunyear)):
            layer_name = str(self.inunyear) + '_' + layer_name

        if self.sparse_matrix:
            sparse_type = type(self.dc.SM_group[self.dc.SM_namelist[0]])
            if not isinstance(array, sparse_type):
                try:
                    array = type(self.dc.SM_group[self.dc.SM_namelist[0]])(array)
                except:
                    raise Exception(f'The adding layer {layer_name} cannot be converted to the data type')
            try:
                self.dc.add_layer(array, layer_name, self.dc.shape[2])
                self.paraname_list.append(layer_name)
            except:
                raise Exception('Some error occurred during the add layer within a phemetric dc')
        else:
            try:
                self.dc = np.concatenate((self.dc, array.reshape(array.shape[0], array.shape[1], 1)), axis=2)
                self.paraname_list.append(layer_name)
            except:
                raise Exception('Some error occurred during the add layer within a phemetric dc')

    def remove_layer(self, layer_name):

        # Process the layer name
        layer_ = []
        if isinstance(layer_name, str):
            if str(self.inunyear) + '_' + layer_name in self.paraname_list:
                layer_.append(str(self.inunyear) + '_' + layer_name)
            elif layer_name in self.paraname_list:
                layer_.append(layer_name)
        elif isinstance(layer_name, list):
            for layer_temp in layer_name:
                if str(self.inunyear) + '_' + layer_temp in self.paraname_list:
                    layer_.append(str(self.inunyear) + '_' + layer_temp)
                elif layer_temp in self.paraname_list:
                    layer_.append(layer_temp)

        # Remove selected layer
        for _ in layer_:
            if self.sparse_matrix:
                try:
                    self.dc.remove_layer(_)
                    self.paraname_list.remove(_)
                except:
                    raise Exception('Some error occurred during the removal of layer within a phemetric dc')
            else:
                try:
                    self.dc = np.delete(self.dc, self.paraname_list.index(_), axis=2)
                    self.paraname_list.remove(layer_name)
                except:
                    raise Exception('Some error occurred during the add layer within a phemetric dc')

        self.save(self.Inunfac_dc_filepath)
        self.__init__(self.Inunfac_dc_filepath)

    def save(self, output_path: str):
        start_time = time.time()
        print(f'Start saving the Inunfac datacube of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        if not os.path.exists(output_path):
            bf.create_folder(output_path)
        output_path = bf.Path(output_path).path_name

        metadata_dic = {'ROI_name': self.ROI_name, 'index': self.index, 'Datatype': self.Datatype, 'ROI': self.ROI,
                        'ROI_array': self.ROI_array, 'ROI_tif': self.ROI_tif, 'sdc_factor': self.sdc_factor,
                        'coordinate_system': self.coordinate_system, 'sparse_matrix': self.sparse_matrix, 'huge_matrix': self.huge_matrix,
                        'size_control_factor': self.size_control_factor, 'oritif_folder': self.oritif_folder, 'dc_group_list': self.dc_group_list, 'tiles': self.tiles,
                        'inunyear': self.inunyear, 'curfit_dic': self.curfit_dic, 'Inunfac_factor': self.Inunfac_factor,
                        'Nodata_value': self.Nodata_value, 'Zoffset': self.Zoffset}

        paraname = self.paraname_list
        np.save(f'{output_path}paraname.npy', paraname)
        with open(f'{output_path}metadata.json', 'w') as js_temp:
            json.dump(metadata_dic, js_temp)

        if self.sparse_matrix:
            self.dc.save(f'{output_path}{str(self.index)}_Inunfac_datacube\\')
        else:
            np.save(f'{output_path}{str(self.index)}_Inunfac_datacube.npy', self.dc)

        print(f'Finish saving the Inunfac datacube of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')


class Thalweg(object):

    def __init__(self):

        # Define work env
        self.work_env = None

        # Define the property of cross section
        self.Thalweg_cs_namelist = None
        self.Thalweg_Linestring = None
        self.Thalweg_geodf = None
        self.original_cs = None
        self.Thalweg_slope = None

        # Define smooth index
        self.smoothed_Thalweg = None
        self.smoothed_cs_index = None
        self.smoothed_Thalweg_slope = None

        # Define the property of cross section
        self.crs = None

    def _generate_thalweg_slope(self):

        # Generate the slope of each line
        if self.Thalweg_Linestring is None:
            raise IOError('Please input the thalweg linestring!')
        else:
            self.Thalweg_slope = []
            for _ in range(len(self.Thalweg_Linestring.coords) - 1):
                dis_ = distance_between_2points(self.Thalweg_Linestring.coords[_], self.Thalweg_Linestring.coords[_ + 1])
                x_diff, y_diff = self.Thalweg_Linestring.coords[_ + 1][0] - self.Thalweg_Linestring.coords[_][0], self.Thalweg_Linestring.coords[_ + 1][1] - self.Thalweg_Linestring.coords[_][1]
                self.Thalweg_slope.append((x_diff / dis_, y_diff / dis_))

        # Generate the slope of each smooth line
        if self.smoothed_Thalweg is None:
            pass
        else:
            self.smoothed_Thalweg_slope = []
            for _ in range(len(self.smoothed_Thalweg.coords) - 1):
                dis_ = distance_between_2points(self.smoothed_Thalweg.coords[_], self.smoothed_Thalweg.coords[_ + 1])
                x_diff, y_diff = self.smoothed_Thalweg.coords[_ + 1][0] - self.smoothed_Thalweg.coords[_][0], self.smoothed_Thalweg.coords[_ + 1][1] - self.smoothed_Thalweg.coords[_][1]
                self.smoothed_Thalweg_slope.append((x_diff / dis_, y_diff / dis_))

    def _extract_Thalweg_geodf(self):

        # Check the keys
        try:
            self.crs = self.Thalweg_geodf.crs
        except:
            raise ValueError('The crs of geodf is missing')

        if False in [_ in list(self.Thalweg_geodf.keys()) for _ in ['cs_namelist', 'geometry']]:
            missing_keys = [_ for _ in ['cs_namelist', 'geometry'] if _ not in list(self.Thalweg_geodf.keys())]
            raise KeyError(f'The key {str(missing_keys)} of geodf is missing!')

        # Restruct the thalweg
        if len(self.Thalweg_geodf['cs_namelist'][0]) == self.Thalweg_geodf['geometry'][0].coords.__len__():
            self.Thalweg_cs_namelist = self.Thalweg_geodf['cs_namelist'][0]
            self.Thalweg_Linestring = self.Thalweg_geodf['geometry'][0]
        else:
            raise Exception('The thalweg has inconsistent cross section name and linestring!')

    def _struct_Thalweg_geodf(self):

        if self.Thalweg_cs_namelist is None or self.Thalweg_Linestring is None or len(self.Thalweg_cs_namelist) == 0 or self.Thalweg_Linestring.coords.__len__() == 0:
            raise ValueError('No information concerning the thalweg is imported')
        else:
            if len(self.Thalweg_cs_namelist) == self.Thalweg_Linestring.coords.__len__():
                thalweg_dic = {'cs_namelist': self.Thalweg_cs_namelist}
                self.Thalweg_geodf = gp.GeoDataFrame(thalweg_dic)

                if self.crs is not None:
                    self.Thalweg_geodf = self.Thalweg_geodf.set_crs(self.crs)

    def straighten_river_through_thalweg(self, tiffile, itr=1, river_value=-200, resize_factor=5):

        gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'NO')

        # Read the tiffile
        ds_temp = gdal.Open(tiffile)
        srs_temp = retrieve_srs(ds_temp)
        if int(srs_temp.split(':')[-1]) != self.crs.to_epsg():
            gdal.Warp('/vsimem/temp1.TIF', ds_temp)
            ds_temp = gdal.Open('/vsimem/temp1.TIF')
        [ul_x, x_res, xt, ul_y, yt, y_res] = ds_temp.GetGeoTransform()
        arr = ds_temp.GetRasterBand(1).ReadAsArray()
        nodata = ds_temp.GetRasterBand(1).GetNoDataValue()

        # Process into the designed pixel
        tiffile_resize = tiffile.split('.TIF')[0] + f'_resized_{str(itr / resize_factor)}.TIF'
        if not os.path.exists(tiffile_resize):
            arr_new = np.repeat(np.repeat(arr, int(x_res / itr * resize_factor), axis=0), int(- y_res / itr * resize_factor), axis=1)
            driver = gdal.GetDriverByName('GTiff')
            driver.Register()
            outds = driver.Create(tiffile_resize, xsize=arr_new.shape[1], ysize=arr_new.shape[0], bands=1, eType=gdal.GDT_Float32, options=['COMPRESS=LZW', 'PREDICTOR=2'])
            outds.SetGeoTransform(ds_temp.GetGeoTransform())
            outds.SetProjection(ds_temp.GetProjection())
            outband = outds.GetRasterBand(1)
            outband.WriteArray(arr_new)
            outband.SetNoDataValue(nodata)
            outband.FlushCache()
            outband = None
            outds = None

        # Determine the thalweg
        if self.smoothed_Thalweg is None:
            thalweg_line = copy.deepcopy(self.Thalweg_Linestring)
            slope_ = copy.deepcopy(self.Thalweg_slope)
        else:
            thalweg_line = copy.deepcopy(self.smoothed_Thalweg)
            slope_ = copy.deepcopy(self.smoothed_Thalweg_slope)

        # Define the value
        itr_num = int(np.ceil(thalweg_line.length / itr)) + 1
        left_bank_v, right_bank_v = [], []
        coord, end_point = thalweg_line.coords[0], 1

        # For each itr
        # Strategies I and II

        online_factor, num = True, 0
        end_coord, mid_coord, end_slope, mid_slope = [(thalweg_line.coords[0][0], thalweg_line.coords[0][1])],[(thalweg_line.coords[0][0], thalweg_line.coords[0][1])], [slope_[0]], [slope_[0]]
        with tqdm(total=itr_num, desc=f'Get the slope and coord for each itr', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            while online_factor:
                dis_to_end = distance_between_2points(coord, thalweg_line.coords[end_point])
                if num == 0:
                    if dis_to_end >= itr:
                        end_coord.append((coord[0] + (itr * slope_[end_point - 1][0]), coord[1] + (itr * slope_[end_point - 1][1])))
                        end_slope.append(slope_[end_point - 1])
                        mid_coord.append((coord[0] + (itr * slope_[end_point - 1][0]), coord[1] + (itr * slope_[end_point - 1][1])))
                        mid_slope.append(slope_[end_point - 1])
                        coord1 = (coord[0] + (itr * slope_[end_point - 1][0]), coord[1] + (itr * slope_[end_point - 1][1]))
                        end_point1 = end_point
                    else:
                        dis_remain = itr - dis_to_end
                        for end__ in range(len(thalweg_line.coords)):
                            if dis_remain <= distance_between_2points(thalweg_line.coords[end_point + end__], thalweg_line.coords[end_point + 1 + end__]):
                                end_coord.append((thalweg_line.coords[end_point + end__][0] + dis_remain * slope_[end_point + end__ - 1][0],
                                                  thalweg_line.coords[end_point + end__][1] + dis_remain * slope_[end_point + end__ - 1][1]))
                                end_slope.append(slope_[end_point + end__ - 1])
                                mid_coord.append((thalweg_line.coords[end_point + end__][0] + dis_remain * slope_[end_point + end__ - 1][0],
                                                  thalweg_line.coords[end_point + end__][1] + dis_remain * slope_[end_point + end__ - 1][1]))
                                mid_slope.append(slope_[end_point + end__ - 1])
                                coord1 = (thalweg_line.coords[end_point + end__][0] + dis_remain * slope_[end_point + end__ - 1][0],
                                          thalweg_line.coords[end_point + end__][1] + dis_remain * slope_[end_point + end__ - 1][1])
                                end_point1 = end_point + end__ + 1
                                break
                            else:
                                dis_remain = dis_remain - distance_between_2points(thalweg_line.coords[end_point + end__], thalweg_line.coords[end_point + 1 + end__])

                    if dis_to_end >= 3 * itr / 2:
                        end_coord.append((coord[0] + (itr * slope_[end_point - 1][0]) * 3 / 2, coord[1] + (itr * slope_[end_point - 1][1]) * 3 / 2))
                        end_slope.append(slope_[end_point - 1])
                    else:
                        dis_remain = itr - dis_to_end
                        for end__ in range(len(thalweg_line.coords)):
                            if dis_remain <= distance_between_2points(thalweg_line.coords[end_point + end__], thalweg_line.coords[end_point + 1 + end__]):
                                end_coord.append((thalweg_line.coords[end_point + end__][0] + dis_remain * slope_[end_point + end__ - 1][0],
                                                  thalweg_line.coords[end_point + end__][1] + dis_remain * slope_[end_point + end__ - 1][1]))
                                end_slope.append(slope_[end_point + end__ - 1])
                                break
                            else:
                                dis_remain = dis_remain - distance_between_2points(thalweg_line.coords[end_point + end__], thalweg_line.coords[end_point + 1 + end__])
                    coord = coord1
                    end_point = end_point1

                elif end_point == len(thalweg_line.coords) - 1 and dis_to_end < itr:

                    mid_coord.append((thalweg_line.coords[len(thalweg_line.coords) - 1][0], thalweg_line.coords[len(thalweg_line.coords) - 1][1]))
                    mid_slope.append(slope_[len(slope_) - 1])
                    end_coord.append((thalweg_line.coords[len(thalweg_line.coords) - 1][0], thalweg_line.coords[len(thalweg_line.coords) - 1][1]))
                    end_slope.append(slope_[len(slope_) - 1])
                    online_factor = False

                else:

                    if dis_to_end >= 3 * itr / 2:
                        end_coord.append((coord[0] + (itr * slope_[end_point - 1][0]) * 3 / 2,
                                          coord[1] + (itr * slope_[end_point - 1][1]) * 3 / 2))
                        end_slope.append(slope_[end_point - 1])
                    else:
                        dis_remain = itr * 3 / 2 - dis_to_end
                        for end__ in range(len(thalweg_line.coords)):
                            if dis_remain <= distance_between_2points(thalweg_line.coords[end_point + end__], thalweg_line.coords[end_point + 1 + end__]):
                                end_coord.append((thalweg_line.coords[end_point + end__][0] + dis_remain * slope_[end_point + end__ - 1][0],
                                                  thalweg_line.coords[end_point + end__][1] + dis_remain * slope_[end_point + end__ - 1][1]))
                                end_slope.append(slope_[end_point + end__ - 1])
                                break
                            else:
                                dis_remain = dis_remain - distance_between_2points(thalweg_line.coords[end_point + end__], thalweg_line.coords[end_point + 1 + end__])

                    if dis_to_end >= itr:
                        mid_coord.append((coord[0] + (itr * slope_[end_point - 1][0]), coord[1] + (itr * slope_[end_point - 1][1])))
                        mid_slope.append(slope_[end_point - 1])
                        coord = (coord[0] + (itr * slope_[end_point - 1][0]), coord[1] + (itr * slope_[end_point - 1][1]))
                        end_point = end_point
                    else:
                        dis_remain = itr - dis_to_end
                        for end__ in range(len(thalweg_line.coords)):
                            if dis_remain <= distance_between_2points(thalweg_line.coords[end_point + end__], thalweg_line.coords[end_point + 1 + end__]):
                                mid_coord.append((thalweg_line.coords[end_point + end__][0] + dis_remain * slope_[end_point + end__ - 1][0],
                                                  thalweg_line.coords[end_point + end__][1] + dis_remain * slope_[end_point + end__ - 1][1]))
                                mid_slope.append(slope_[end_point + end__ - 1])
                                coord = (thalweg_line.coords[end_point + end__][0] + dis_remain * slope_[end_point + end__ - 1][0],
                                         thalweg_line.coords[end_point + end__][1] + dis_remain * slope_[end_point + end__ - 1][1])
                                end_point = end_point + end__ + 1
                                break
                            else:
                                dis_remain = dis_remain - distance_between_2points(thalweg_line.coords[end_point + end__], thalweg_line.coords[end_point + 1 + end__])
                num += 1
                pbar.update()

        # a = LineString(end_coord)
        # gdf = gp.GeoDataFrame({'geometry': a, 'index': [0]}, crs=srs_temp)
        # gdf.to_file('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11_nc\\temp\\atemp.shp')
        time1, time2, time3 = 0, 0, 0
        bf.create_folder('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11_nc\\temp\\')
        ds_temp, arr = None, None

        line_slope, line_coord, line_itr = [], [], []
        for __ in range(len(mid_coord)):
            line_coord.append([[end_coord[__][0], end_coord[__][1]], [end_coord[__ + 1][0], end_coord[__ + 1][1]]])
            line_slope.append([[end_slope[__][0], end_slope[__][1]], [end_slope[__ + 1][0], end_slope[__ + 1][1]]])
            line_itr.append(__)

        with concurrent.futures.ProcessPoolExecutor() as exe:
            res = exe.map(get_value_through_river_crosssection, line_itr, line_coord, line_slope, repeat(nodata), repeat(itr), repeat(tiffile_resize), repeat(len(mid_coord)))

        res = list(res)
        left_bank_v = [_[0] for _ in res]
        right_bank_v = [_[1] for _ in res]

        left_size, right_size = 0, 0
        for _ in left_bank_v:
            left_size = max(left_size, len(_))

        for _ in right_bank_v:
            right_size = max(right_size, len(_))

        new_arr = np.zeros([left_size + right_size - 1, len(left_bank_v)]) * np.nan
        for q in range(len(left_bank_v)):
            _ = left_bank_v[q]
            for __ in range(len(_)):
                new_arr[left_size - __, q] = _[__]

        for q in range(len(right_bank_v)):
            _ = right_bank_v[q]
            for __ in range(len(_)):
                new_arr[left_size + __ - 1, q] = _[__]

        np.save(f'G:\A_Landsat_Floodplain_veg\Paper\Fig11_nc\\arr_{str(left_size)}_{str(itr)}.npy', new_arr)
        fig, ax = plt.subplots(figsize=(60, 3))
        cax = ax.imshow(new_arr)
        plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig11_nc\\fig11_nc_{str(left_size)}_{str(itr)}.png', dpi=600)

    # def load_shapefile(thalweg_temp, shapefile):
    #
    #     # Process work env
    #     thalweg_temp.work_env = bf.Path(os.path.dirname(shapefile)).path_name
    #
    #     # Check the shapefile existence
    #     if isinstance(shapefile, str):
    #         if not os.path.exists(shapefile):
    #             raise ValueError(f'The {str(shapefile)} does not exist!')
    #         elif shapefile.endswith('.shp'):
    #             thalweg_temp.Thalweg_geodf = gp.read_file(shapefile, encoding='utf-8')
    #         else:
    #             raise TypeError(f'The geodf json file should be a json!')
    #     else:
    #         raise TypeError(f'The {str(shapefile)} should be a str!')
    #
    #     # Extract information from geodf
    #     cs_name_temp = thalweg_temp.Thalweg_geodf['cs_namelis'][0]
    #     cs_name = []
    #     for _ in cs_name_temp.split("',"):
    #         cs_name.append(_.split("'")[-1])
    #
    #     thalweg_temp._extract_Thalweg_geodf()
    #     return thalweg_temp

    def load_smooth_Thalweg_shp(self, shpfile):

        # Check the shapefile existence
        if isinstance(shpfile, str):
            if not os.path.exists(shpfile):
                raise ValueError(f'The {str(shpfile)} does not exist!')
            elif shpfile.endswith('.shp'):
                Thalweg_geodf_temp = gp.read_file(shpfile, encoding='utf-8')
            else:
                raise TypeError(f'The geodf json file should be a json!')
        else:
            raise TypeError(f'The {str(shpfile)} should be a str!')

        # Extract information from geodf
        if Thalweg_geodf_temp.shape[0] != 1 or type(Thalweg_geodf_temp['geometry'][0]) != LineString:
            raise ValueError('The shpfile should be a string type')
        else:
            self.smoothed_Thalweg = Thalweg_geodf_temp['geometry'][0]
            self.smoothed_cs_index = []

            for _ in range(len(self.Thalweg_cs_namelist)):
                simplified_thalweg_arr = np.array(self.smoothed_Thalweg.coords)
                if _ == 0:
                    self.smoothed_cs_index.append(0)
                elif _ == len(self.Thalweg_cs_namelist) - 1:
                    self.smoothed_cs_index.append(simplified_thalweg_arr.shape[0] - 1)
                else:
                    cs_line = list(self.original_cs.cross_section_geodf['geometry'][self.original_cs.cross_section_geodf['cs_name'] == self.Thalweg_cs_namelist[_]])[0]
                    intersect = shapely.intersection(cs_line, self.smoothed_Thalweg)
                    if not isinstance(intersect, Point):
                        raise Exception(f'Smooth line is not intersected with cross section {str(self.Thalweg_cs_namelist[_])}')

                    start_vertex = determin_start_vertex_of_point(intersect, self.smoothed_Thalweg)
                    arr_ = np.zeros([1, simplified_thalweg_arr.shape[1]])
                    arr_[0, 0], arr_[0, 1], arr_[0, 2] = intersect.coords[0][0], intersect.coords[0][1], intersect.coords[0][2]
                    simplified_thalweg_arr = np.insert(simplified_thalweg_arr, start_vertex + 1, arr_, axis=0)
                    self.smoothed_cs_index.append(start_vertex + 1)

                    if arr_.shape[1] == 2:
                        smoothed_thalweg_list = [(simplified_thalweg_arr[_, 0], simplified_thalweg_arr[_, 1]) for _ in range(simplified_thalweg_arr.shape[0])]
                    elif arr_.shape[1] == 3:
                        smoothed_thalweg_list = [(simplified_thalweg_arr[_, 0], simplified_thalweg_arr[_, 1], simplified_thalweg_arr[_, 2]) for _ in range(simplified_thalweg_arr.shape[0])]
                    else:
                        raise Exception('Code error!')
                    self.smoothed_Thalweg = LineString(smoothed_thalweg_list)
            # geodf = gp.GeoDataFrame(data=[{'a': 'b'}], geometry=[thalweg_temp.smoothed_Thalweg])
            # geodf.to_file('G:\A_Landsat_Floodplain_veg\Water_level_python\\a.shp')
        self._generate_thalweg_slope()

    def load_geojson(self, geodf_json):

        # Process work env
        self.work_env = bf.Path(os.path.dirname(os.path.dirname(geodf_json))).path_name

        # Check the df json existence
        if isinstance(geodf_json, str):
            if not os.path.exists(geodf_json):
                raise ValueError(f'The {str(geodf_json)} does not exist!')
            elif geodf_json.endswith('.json'):
                self.Thalweg_geodf = gp.read_file(geodf_json)
            else:
                raise TypeError(f'The geodf json file should be a json!')
        else:
            raise TypeError(f'The {str(geodf_json)} should be a str!')

        # Check the cs json existence
        try:
            cs_json = os.path.dirname(geodf_json) + '\\CrossSection.json'
        except:
            print(traceback.format_exc())
            raise Exception('The cs json can not be generated!')

        if isinstance(cs_json, str):
            if not os.path.exists(cs_json):
                raise ValueError(f"The CrossSection json does not exist!")
            elif cs_json.endswith('.json'):
                self.original_cs = CrossSection()
                self.original_cs = self.original_cs.load_geojson(cs_json)
            else:
                raise TypeError(f'The cs_json file should be a json!')
        else:
            raise TypeError(f'The {str(cs_json)} should be a str!')

        # Extract information from geodf
        self._extract_Thalweg_geodf()
        self._generate_thalweg_slope()
        return self

    def to_geojson(self,  output_path: str = None):

        # Define work path
        if output_path is None:
            output_path = self.work_env + 'output_geojson\\'
        else:
            output_path = bf.Path(output_path).path_name
        bf.create_folder(output_path)

        # To geojson
        if isinstance(self.Thalweg_geodf, gp.GeoDataFrame):
            self.Thalweg_geodf['cs_namelist'] = self.Thalweg_geodf['cs_namelist'].astype(str)
            self.Thalweg_geodf.to_file(output_path + 'thalweg.json', driver='GeoJSON')

        if isinstance(self.original_cs, CrossSection):
            self.original_cs.to_geojson(output_path)

    def to_shapefile(self,  output_path: str = None):

        # Define work path
        if output_path is None:
            output_path = self.work_env + 'output_shpfile\\'
        else:
            output_path = bf.Path(output_path).path_name
        bf.create_folder(output_path)

        # To shapefile
        if isinstance(self.Thalweg_geodf, gp.GeoDataFrame):
            self.Thalweg_geodf['cs_namelist'] = self.Thalweg_geodf['cs_namelist'].astype(str)
            self.Thalweg_geodf.to_file(output_path + 'thalweg.shp', encoding='utf-8')

        if isinstance(self.original_cs, CrossSection):
            self.original_cs.to_shpfile(output_path)

    def merged_hydro_inform(self, hydro_ds: HydrometricStationData):

        # Create the hydro inform dic
        self.hydro_inform_dic = {}

        # Detect the datatype
        if not isinstance(hydro_ds, HydrometricStationData):
            raise TypeError('The hydrods is not a standard hydrometric station data!')

        # Merge hydro inform
        for _ in hydro_ds.cross_section_namelist:
            if _ in self.Thalweg_cs_namelist and ~np.isnan(hydro_ds.water_level_offset[hydro_ds.station_namelist[hydro_ds.cross_section_namelist.index(_)]]):
                wl_offset = hydro_ds.water_level_offset[hydro_ds.station_namelist[hydro_ds.cross_section_namelist.index(_)]]
                self.hydro_inform_dic[_] = hydro_ds.hydrological_inform_dic[hydro_ds.station_namelist[hydro_ds.cross_section_namelist.index(_)]]
                self.hydro_inform_dic[_]['water_level/m'] = self.hydro_inform_dic[_]['water_level/m'] + wl_offset


class Flood_freq_based_hyspometry_method(object):

    def __init__(self, year_list: list, work_env=None):

        # Check the fitness
        if False in [np.logical_and(isinstance(_, int), _ in range(1990, 2100)) for _ in year_list]:
            raise Exception('Please double check the input year list for flood-frequency-based hyspometry method!')

        # Define method para
        self.year_list = year_list
        self.ref_ele_map = None
        self.work_env = None

        # Get the work env
        try:
            if work_env is not None:
                if not os.path.exists(work_env):
                    bf.create_folder(work_env)
                self.work_env = bf.Path(work_env).path_name
        except:
            print(traceback.format_exc())
            raise Exception('Cannot read the work env ')

    def perform_in_epoch(self, thalweg_temp: Thalweg, inundation_frequency_tif: str, hydro_datacube=True):

        if self.work_env is None:
            self.work_env = thalweg_temp.work_env

        if len(self.year_list) <= 5:
            print('Please mention that the Flood_freq_based_hyspometry_method might not perfrom well for data under 5 years!')

        # Check if the hydro inform is merged
        if 'hydro_inform_dic' not in thalweg_temp.__dict__.keys():
            raise Exception('Please merged standard hydrometric_station_ds before linkage!')

        # Process cross section inform
        cs_list, year_domain, hydro_pos = [], [], []
        if thalweg_temp.smoothed_Thalweg is None:
            for _ in range(len(thalweg_temp.Thalweg_cs_namelist)):
                if thalweg_temp.Thalweg_cs_namelist[_] in thalweg_temp.hydro_inform_dic.keys():
                    year_domain.append(np.unique(np.array(thalweg_temp.hydro_inform_dic[thalweg_temp.Thalweg_cs_namelist[_]]['year'])).tolist())
                    hydro_pos.append(_)
                    cs_list.append(thalweg_temp.Thalweg_cs_namelist[_])

        elif isinstance(thalweg_temp.smoothed_Thalweg, LineString):
            for _ in range(len(thalweg_temp.Thalweg_cs_namelist)):
                pos = thalweg_temp.smoothed_cs_index[_]
                if thalweg_temp.Thalweg_cs_namelist[_] in thalweg_temp.hydro_inform_dic.keys():
                    year_domain.append(np.unique(np.array(thalweg_temp.hydro_inform_dic[thalweg_temp.Thalweg_cs_namelist[_]]['year'])).tolist())
                    hydro_pos.append(pos)
                    cs_list.append(thalweg_temp.Thalweg_cs_namelist[_])

        else:
            raise TypeError('The smoothed thalweg is not under the correct type')

        # Import Inundation_frequency_tif
        if isinstance(inundation_frequency_tif, str):
            if not inundation_frequency_tif.endswith('.TIF') and not inundation_frequency_tif.endswith('.tif'):
                raise TypeError('The inundation frequency map should be a TIF file')
            else:
                try:
                    ds_temp = gdal.Open(inundation_frequency_tif)
                    srs_temp = retrieve_srs(ds_temp)
                    if int(srs_temp.split(':')[-1]) != thalweg_temp.crs.to_epsg():
                        gdal.Warp('/vsimem/temp1.TIF', ds_temp, )
                        ds_temp = gdal.Open('/vsimem/temp1.TIF')
                    [ul_x, x_res, xt, ul_y, yt, y_res] = ds_temp.GetGeoTransform()
                    arr = ds_temp.GetRasterBand(1).ReadAsArray()
                except:
                    raise ValueError('The inundation frequency tif file is problematic!')
        else:
            raise TypeError('The inundation frequency map should be a TIF file')

        # Process the inundation frequency map
        arr_pd = np.argwhere(~np.isnan(arr))
        v_list = []
        for _ in arr_pd:
            v_list.append(arr[_[0], _[1]])
        arr_pd = np.concatenate((arr_pd.astype(np.float32), np.array(v_list).reshape(len(v_list), 1)), axis=1)
        arr_pd = pd.DataFrame(arr_pd, columns=['y', 'x', 'if'])
        arr_pd = arr_pd.sort_values(['x', 'y'], ascending=[True, True])
        arr_pd = arr_pd.reset_index(drop=True)

        cpu_amount = os.cpu_count()
        arr_pd_list,  indi_size = [], int(np.ceil(arr_pd.shape[0] / cpu_amount))
        for i_size in range(cpu_amount):
            if i_size != cpu_amount - 1:
                arr_pd_list.append(arr_pd[indi_size * i_size: indi_size * (i_size + 1)])
            else:
                arr_pd_list.append(arr_pd[indi_size * i_size: -1])

        with concurrent.futures.ProcessPoolExecutor() as exe:
            res = exe.map(flood_frequency_based_hypsometry, arr_pd_list, repeat(thalweg_temp), repeat(self.year_list), repeat([ul_x, x_res, ul_y, y_res]), repeat(cs_list), repeat(year_domain), repeat(hydro_pos), repeat(hydro_datacube))

        if hydro_datacube:
            res_df = None
            res = list(res)
            for result_temp in res:
                if res_df is None:
                    res_df = copy.copy(result_temp)
                else:
                    res_df = pd.concat([res_df, result_temp])
            res_df.to_csv(thalweg_temp.work_env + f'hydro_dc_X_{str(arr.shape[1])}_Y_{str(arr.shape[0])}_' + inundation_frequency_tif.split('\\')[-1].split('.')[0] + '.csv')

        # Generate ele and inundation arr
        ele_arr = np.zeros_like(arr)
        ele_arr[np.isnan(arr)] = np.nan
        inun_arr = np.zeros_like(arr)
        inun_arr[np.isnan(arr)] = np.nan
        res_df = res_df.reset_index(drop=True)
        for _ in range(res_df.shape[0]):
            ele_arr[int(res_df['y'][_]), int(res_df['x'][_])] = res_df['wl'][_]
            inun_arr[int(res_df['y'][_]), int(res_df['x'][_])] = res_df['fr'][_]
        bf.write_raster(ds_temp, ele_arr, thalweg_temp.work_env, 'ele_' + inundation_frequency_tif.split('\\')[-1], raster_datatype=gdal.GDT_Float32)
        bf.write_raster(ds_temp, inun_arr, thalweg_temp.work_env, 'inun_' + inundation_frequency_tif.split('\\')[-1], raster_datatype=gdal.GDT_Float32)
        self.ref_ele_map = thalweg_temp.work_env, 'ele_' + inundation_frequency_tif.split('\\')[-1]

        # Generate water level dc
        # all_year_list = [year_ for year_ in range(year_range[0], year_range[1])]
        # for year in range(year_range[0], year_range[1]):
        #     doy_list = [year * 1000 + date for date in range(1, 1 + datetime(year=year + 1, month=1, day=1).toordinal() - datetime(year=year, month=1, day=1).toordinal())]
        #     matrix_list = [sm.lil_matrix((arr.shape[0], arr.shape[1])) for date in range(1, 1 + datetime(year=year + 1, month=1, day=1).toordinal() - datetime(year=year, month=1, day=1).toordinal())]
        #     bf.create_folder(thalweg_temp.work_env + f'yearly_wl\\{str(year)}\\')
        #     for _ in range(res_df.shape[0]):
        #         st_cs, ed_cs = res_df['yearly_wl'][_][int(all_year_list.index(year))][0], res_df['yearly_wl'][_][int(all_year_list.index(year))][1]
        #         st_dis, ed_dis = res_df['yearly_wl'][_][int(all_year_list.index(year))][2], res_df['yearly_wl'][_][int(all_year_list.index(year))][3]
        #         wl_st_series = thalweg_temp.hydro_inform_dic[st_cs]['year'] == year
        #         wl_end_series = thalweg_temp.hydro_inform_dic[ed_cs]['year'] == year
        #         wl_start = thalweg_temp.hydro_inform_dic[st_cs][wl_st_series]['water_level/m'].reset_index(drop=True)
        #         wl_end = thalweg_temp.hydro_inform_dic[ed_cs][wl_end_series]['water_level/m'].reset_index(drop=True)
        #         wl_pos = list(wl_start + (wl_end - wl_start) * st_dis / (st_dis + ed_dis))
        #         if len(wl_pos) == len(doy_list):
        #             for __ in range(len(wl_pos)):
        #                 matrix_list[__][int(res_df['y'][_]), int(res_df['x'][_])] = wl_pos[__]
        #     year_m = ndsm(*matrix_list, SM_namelist=doy_list)
        #     year_m.save(thalweg_temp.work_env + f'yearly_wl\\{str(year)}\\')

    def refine_annual_topography(self, thalweg_temp, inun_dc: Landsat_dc, hydro_dc: HydroDatacube, elevation_map: str = None):

        if self.work_env is None:
            self.work_env = thalweg_temp.work_env

        # Get the elevation map
        if self.ref_ele_map is None and elevation_map is None:
            raise IOError('Please input the elevation map before refinement!')
        elif elevation_map is not None:
            try:
                ele_ds = gdal.Open(elevation_map)
                ele_arr = ele_ds.GetRasterBand(1).ReadAsArray()
            except:
                print(traceback.format_exc())
                raise TypeError('Please mention the elevation map is not correctly imported!')
        else:
            try:
                ele_ds = gdal.Open(self.ref_ele_map)
                ele_arr = ele_ds.GetRasterBand(1).ReadAsArray()
            except:
                print(traceback.format_exc())
                raise TypeError('Please mention the elevation map is not correctly imported!')

        # thalweg extraction
        if thalweg_temp.smoothed_Thalweg is None:
            linestr = thalweg_temp.smoothed_Thalweg
        else:
            linestr = thalweg_temp.Thalweg_Linestring

        # Check the year
        if hydro_dc.year not in self.year_list:
            raise Exception('Please input the hydro datacube under right year!')
        elif hydro_dc.year not in np.unique(np.floor(np.array(inun_dc.sdc_doylist)/10000).astype(np.int32)):
            raise Exception('Please input the inundation dc under right year!')

        boundary_wl = np.zeros_like(ele_arr) * np.nan
        inundation_arr = np.zeros_like(ele_arr)
        for _ in inun_dc.sdc_doylist:
            if int(np.floor(_/10000)) == hydro_dc.year:

                # Retrieve the flood inundation map
                inun_arr = inun_dc.dc[:, :, inun_dc.sdc_doylist.index(_)]
                inun_arr_output = copy.deepcopy(inun_arr)
                inun_arr_output[inun_arr != 2] = 0
                inun_arr_output[inun_arr == 2] = 1
                inundation_arr = np.logical_or(inundation_arr, inun_arr_output)
                bf.create_folder(f'{self.work_env}inundation_temp\\')
                bf.write_raster(ele_ds, inun_arr_output, f'{self.work_env}inundation_temp\\', str(_) + '.tif', raster_datatype=gdal.GDT_Byte)

                src_temp = rasterio.open(f'{self.work_env}inundation_temp\\' + str(_) + '.tif')
                shp_dic = ({'properties': {'raster_val': int(v)}, 'geometry': s} for i, (s, v) in
                           enumerate(rasterio.features.shapes(inun_arr_output, connectivity=8, transform=src_temp.transform)) if
                           ~np.isnan(v))
                meta = src_temp.meta.copy()
                meta.update(compress='lzw')

                # Extract the flood inundation area (remove depressions)
                shp_list = list(shp_dic)
                nw_shp_list = []
                shp_file = gp.GeoDataFrame.from_features(shp_list)
                for __ in range(shp_file.shape[0]):
                    if shp_file['raster_val'][__] == 1:
                        if linestr.intersects(shp_file['geometry'][__]):
                            nw_shp_list.append(shp_list[__])
                nw_shp_file = gp.GeoDataFrame.from_features(nw_shp_list)

                bf.create_folder(f'{self.work_env}inundation_final\\')
                if os.path.exists(f'{self.work_env}inundation_final\\{str(_)}.tif'):
                    os.remove(f'{self.work_env}inundation_final\\{str(_)}.tif')

                with rasterio.open(f'{self.work_env}inundation_final\\{str(_)}.tif', 'w+', **meta) as out:
                    out_arr = out.read(1)
                    out_arr[inun_arr == 0] = 0
                    # this is where we create a generator of geom, value pairs to use in rasterizing
                    shapes = ((geom, value) for geom, value in zip(nw_shp_file.geometry, nw_shp_file.raster_val)) if nw_shp_file.shape[0] != 0 else None
                    burned = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=src_temp.transform) if shapes is not None else out_arr
                    out.write_band(1, burned)

                # Get the boundary pixel
                ds_temp = gdal.Open(f'{self.work_env}inundation_final\\{str(_)}.tif')
                arr_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                arr_temp[inun_arr == 1] = 0
                pos = np.argwhere(arr_temp == 1)

                # Get the water level map
                doy = bf.date2doy(_)
                if isinstance(hydro_dc.hydrodatacube, NDSparseMatrix):
                    hydro = hydro_dc.hydrodatacube.SM_group[doy].toarray()
                else:
                    pass

                # Get the boundary water level
                if pos.shape[0] != 0:
                    for __ in pos:
                        dc_temp = inun_arr[__[0] - 1: __[0] + 2, __[1] - 1: __[1] + 2]

                        if 1 in dc_temp:
                            wl_inundated_list, wl_noninun_list = [], []
                            for ___ in inun_dc.sdc_doylist:
                                if int(np.floor(___ / 10000)) == hydro_dc.year:
                                    doy_t = bf.date2doy(___)
                                    if inun_dc.dc[__[0], __[1], inun_dc.sdc_doylist.index(___)] == 2:
                                        wl_inundated_list.append(hydro_dc.hydrodatacube[int(__[0]), int(__[1]), hydro_dc.hydrodatacube.SM_namelist.index(doy_t)])
                                    elif inun_dc.dc[__[0], __[1], inun_dc.sdc_doylist.index(___)] == 1:
                                        wl_noninun_list.append(hydro_dc.hydrodatacube[__[0], __[1], hydro_dc.hydrodatacube.SM_namelist.index(doy_t)])

                            wl_noninun_list = np.array(wl_noninun_list)
                            if wl_noninun_list.shape[0] == 0 or np.sum(wl_noninun_list > hydro[__[0], __[1]]) / wl_noninun_list.shape[0] > 0.1:
                                pass
                            else:
                                if np.isnan(boundary_wl[__[0], __[1]]):
                                    boundary_wl[__[0], __[1]] = hydro[__[0], __[1]]
                                else:
                                    boundary_wl[__[0], __[1]] = min(hydro[__[0], __[1]], boundary_wl[__[0], __[1]])

        bias_arr = boundary_wl - ele_arr
        bf.write_raster(ele_ds, boundary_wl, self.work_env, f'{str(hydro_dc.year)}_wl_boundary.tif', nodatavalue=np.nan)
        bf.write_raster(ele_ds, bias_arr, self.work_env, f'{str(hydro_dc.year)}_bias.tif', nodatavalue=np.nan)

        with rasterio.open(self.work_env + f'{str(hydro_dc.year)}_bias.tif') as raster:
            # Read the specific band
            band = raster.read(1)  # assuming you're interested in the first band

            # Coordinate transformation
            transform = raster.transform

            # Extract points, here assuming we want all non-zero or non-nodata values
            rows, cols = np.where(band != raster.nodata) if ~np.isnan(raster.nodata) else np.where(~np.isnan(band))
            points = [transform * (col, row) for col, row in zip(cols, rows)]
            values = band[rows, cols]  # The corresponding raster values

            # Create geospatial points
            geometries = [Point(xy) for xy in points]
            gdf = gp.GeoDataFrame({'geometry': geometries, 'Raster Value': values})

            # Set the coordinate reference system (CRS) to the raster's CRS
            gdf.crs = raster.crs

            # Save to a Shapefile
            gdf.to_file(self.work_env + f'{str(hydro_dc.year)}_bias.shp')
            # Interpolate the bias
            # Method 1 Kriging

            # Method 2 Cubic interpolation

            # Method 3 Linear interpolation

            # Method 4


class CrossSection(object):

    def __init__(self, work_env=None):

        # Define work env
        if work_env is None:
            self.work_env = None
        else:
            try:
                bf.create_folder(work_env)
                self.work_env = bf.Path(work_env).path_name
            except:
                self.work_env = None

        # Define the property of cross section
        self.cross_section_name = None
        self.cross_section_dem = None
        self.cross_section_distance = None
        self.cross_section_tribu = None
        self.cross_section_bank_coord = None
        self.cross_section_geodf = None
        self.cross_section_num = 0
        self.crs = None
        self.issued_cross_section = []

        # Define the DEM extracted cross section inform
        self.cross_section_2D_dem = None

        # Define the differential dem
        self.cross_section_diff = None
        self.cross_section_diff_geometry = None

    def _consistent_cross_section_inform(self):

        # Consistent all inform imported from the standard xlsx
        if self.cross_section_name != list(self.cross_section_dem.keys()) or self.cross_section_dem.keys() != list(self.cross_section_distance.keys()):
            combined_list = [_ for _ in self.cross_section_name if _ in list(self.cross_section_dem.keys()) and _ in list(self.cross_section_distance.keys())]
            for _ in self.cross_section_name:
                if _ not in combined_list:
                    self.cross_section_name.remove(_)
                    print(f'Some information for the cross section {str(_)} is missing!')
            for _ in list(self.cross_section_dem.keys()):
                if _ not in self.cross_section_dem.keys():
                    self.cross_section_dem.pop(_)
                    print(f'Some information for the cross section {str(_)} is missing!')
            for _ in list(self.cross_section_distance.keys()):
                if _ not in list(self.cross_section_distance.keys()):
                    self.cross_section_distance.pop(_)
                    print(f'Some information for the cross section {str(_)} is missing!')

        # Consistent the tribu
        if self.cross_section_tribu is None:
            self.cross_section_tribu = {}
            for _ in self.cross_section_name:
                self.cross_section_tribu[_] = False
        elif list(self.cross_section_tribu.keys()) != self.cross_section_name:
            for _ in self.cross_section_name:
                if _ not in list(self.cross_section_tribu.keys()):
                    self.cross_section_tribu[_] = False
            for _ in list(self.cross_section_tribu.keys()):
                if _ not in self.cross_section_name:
                    self.cross_section_tribu.pop(_)

        # Consistent the geolocation
        if self.cross_section_bank_coord is None:
            self.cross_section_bank_coord = {}
            for _ in self.cross_section_name:
                self.cross_section_bank_coord[_] = [(np.nan, np.nan), (np.nan, np.nan)]
        elif list(self.cross_section_bank_coord.keys()) != self.cross_section_name:
            for _ in self.cross_section_name:
                if _ not in list(self.cross_section_bank_coord.keys()):
                    self.cross_section_bank_coord[_] = [(np.nan, np.nan), (np.nan, np.nan)]
            for _ in list(self.cross_section_bank_coord.keys()):
                if _ not in self.cross_section_name:
                    self.cross_section_bank_coord.pop(_)

    def _construct_geodf(self):

        # Consistent the dem, distance and name list
        self._consistent_cross_section_inform()
        self.cross_section_num = len(self.cross_section_name)

        # Generate the geodataframe
        tribu_temp, dem_temp, distance_temp, geometry_temp, bank_coord_temp = [], [], [], [], []
        for _ in self.cross_section_name:
            tribu_temp.append(self.cross_section_tribu[_])
            dem_temp.append(self.cross_section_dem[_])
            distance_temp.append(self.cross_section_distance[_])
            bank_coord_temp.append(self.cross_section_bank_coord[_])

            # Generate the geometry
            if bank_coord_temp[-1] == LineString() or np.isnan(bank_coord_temp[-1][0][0]):
                geometry_temp.append(LineString())
            else:
                line_coord = []
                lb_xcoord, lb_ycoord = bank_coord_temp[-1][0][0], bank_coord_temp[-1][0][1]
                dis_ = np.sqrt((bank_coord_temp[-1][1][0] - bank_coord_temp[-1][0][0]) ** 2 + (bank_coord_temp[-1][1][1] - bank_coord_temp[-1][0][1]) ** 2)
                itr_xdim, itr_ydim = (bank_coord_temp[-1][1][0] - bank_coord_temp[-1][0][0]) / dis_, (bank_coord_temp[-1][1][1] - bank_coord_temp[-1][0][1]) / dis_
                offset = self.cross_section_dem[_][0][0] if abs(dis_ - (self.cross_section_dem[_][-1][0] - self.cross_section_dem[_][0][0])) < abs(dis_ - self.cross_section_dem[_][-1][0]) else 0

                for dem_index in range(len(self.cross_section_dem[_])):
                    dem_dis = self.cross_section_dem[_][dem_index][0] - offset
                    line_coord.append((lb_xcoord + dem_dis * itr_xdim, lb_ycoord + dem_dis * itr_ydim, self.cross_section_dem[_][dem_index][1]))
                geometry_temp.append(LineString(line_coord))

        dic_temp = {'cs_name': self.cross_section_name, 'cs_tribu': tribu_temp,
                    'cs_dem': dem_temp, 'cs_distance2dam': distance_temp,
                    'cs_bank_coord': bank_coord_temp, 'geometry': geometry_temp}
        self.cross_section_geodf = gp.GeoDataFrame(dic_temp)

        # Set the crs
        if self.crs is not None:
            self.cross_section_geodf = self.cross_section_geodf.set_crs(self.crs)

    def _extract_from_geodf(self):

        # Detect if the geodf consists adequate information
        try:
            self.crs = self.cross_section_geodf.crs
        except:
            raise ValueError('The crs of geodf is missing')

        if False in [_ in list(self.cross_section_geodf.keys()) for _ in ['cs_name', 'cs_distance2dam', 'geometry', 'cs_tribu', 'cs_dem', 'cs_bank_coord']]:
            missing_keys = [_ for _ in ['cs_name', 'cs_distance2dam', 'geometry', 'cs_tribu', 'cs_dem', 'cs_bank_coord'] if _ not in list(self.cross_section_geodf.keys())]
            raise KeyError(f'The key {str(missing_keys)} of geodf is missing!')

        # Extract information from geodf
        self.cross_section_name = list(self.cross_section_geodf['cs_name'])
        self.cross_section_dem, self.cross_section_distance, self.cross_section_tribu, self.cross_section_bank_coord = {}, {}, {}, {}
        self.cross_section_num = len(self.cross_section_name)

        # Construct the dic
        dem4df = []
        for _ in self.cross_section_name:
            cs_index = self.cross_section_geodf[self.cross_section_geodf['cs_name'] == _].index[0]
            self.cross_section_distance[_] = self.cross_section_geodf['cs_distance2dam'][cs_index]
            self.cross_section_bank_coord[_] = self.cross_section_geodf['geometry'][cs_index]
            self.cross_section_tribu[_] = self.cross_section_geodf['cs_tribu'][cs_index]

            dem_list = self.cross_section_geodf['cs_dem'][cs_index]
            dem_list = dem_list.split(',')
            dem_output = []
            if np.mod(len(dem_list), 2) == 0:
                indi = 0
                dem_temp = []
                for _ in dem_list:
                    if indi == 0:
                        dem_temp.append(float(_.split('[')[-1]))
                    elif indi == 1:
                        indi = -1
                        dem_temp.append(float(_.split(']')[0]))
                        dem_output.append(dem_temp)
                        dem_temp = []
                    indi += 1
            else:
                raise Exception(f'Dem of {_} was problematic!')
            dem4df.append(dem_output)
            self.cross_section_dem[_] = self.cross_section_geodf['cs_dem'][cs_index]
        self.cross_section_geodf['cs_dem'] = dem4df

    def from_standard_cross_profiles(self, dem_xlsx_filename: str):

        # Import dem xlsx filename
        if isinstance(dem_xlsx_filename, str):
            if not os.path.exists(dem_xlsx_filename):
                raise ValueError(f'The {str(dem_xlsx_filename)} does not exist!')
            elif dem_xlsx_filename.endswith('.xlsx') or dem_xlsx_filename.endswith('.xls'):
                dem_xlsx_file = pd.read_excel(dem_xlsx_filename)
            elif dem_xlsx_filename.endswith('.csv'):
                dem_xlsx_file = pd.read_csv(dem_xlsx_filename)
            else:
                raise TypeError(f'The dem xlsx file should be a xlsx!')
        else:
            raise TypeError(f'The {str(dem_xlsx_filename)} should be a str!')

        # Process work env
        if self.work_env is None:
            self.work_env = bf.Path(os.path.dirname(dem_xlsx_filename)).path_name

        # Process dem xlsx file
        cs_start_index = dem_xlsx_file.loc[dem_xlsx_file[dem_xlsx_file.keys()[0]] == '序号'].index
        cs_end_index = dem_xlsx_file[dem_xlsx_file.keys()[0]][(dem_xlsx_file[dem_xlsx_file.keys()[0]].str.isnumeric() == False)].index.tolist()
        itr = 0
        for cs_index in cs_start_index:
            try:
                itr += 1
                cs_name_ = dem_xlsx_file[dem_xlsx_file.keys()[0]][cs_index - 3]
                # Index the cross section NAME
                if self.cross_section_name is None:
                    self.cross_section_name = [dem_xlsx_file[dem_xlsx_file.keys()[0]][cs_index - 3]]
                elif cs_name_ not in self.cross_section_name:
                    self.cross_section_name.append(dem_xlsx_file[dem_xlsx_file.keys()[0]][cs_index - 3])

                # Index the cross section DEM
                cs_end_index_temp = [_ for _ in cs_end_index if _ > cs_index]
                cs_end_index_temp = min(cs_end_index_temp)
                if self.cross_section_dem is None:
                    self.cross_section_dem = {cs_name_: np.array(dem_xlsx_file[dem_xlsx_file.keys()[1:3]][cs_index + 1: cs_end_index_temp].astype(np.float32)).tolist()}
                elif cs_name_ not in list(self.cross_section_dem.keys()):
                    self.cross_section_dem[cs_name_] = np.array(dem_xlsx_file[dem_xlsx_file.keys()[1:3]][cs_index + 1: cs_end_index_temp].astype(np.float32)).tolist()

                # Index the cross section distance
                if self.cross_section_distance is None:
                    self.cross_section_distance = {cs_name_: np.float32(dem_xlsx_file[dem_xlsx_file.keys()[1]][cs_end_index[-1] + itr])}
                elif cs_name_ not in list(self.cross_section_distance.keys()):
                    self.cross_section_distance[cs_name_] = np.float32(dem_xlsx_file[dem_xlsx_file.keys()[1]][cs_end_index[-1] + itr])
            except:
                self.issued_cross_section.append(cs_index)

        self._construct_geodf()

    def import_section_coordinates(self, coordinate_files: str, epsg_crs: str):

        # Check whether the cross section name is imported
        if self.cross_section_name is None or len(self.cross_section_name) == 0:
            raise ValueError('Please import the cross section standard information before the coordinates!')

        # Process crs
        if not isinstance(epsg_crs, str):
            raise TypeError('The epsg crs should be a str!')
        else:
            epsg_crs = epsg_crs.lower()
            if not epsg_crs.startswith('epsg:'):
                raise ValueError('The epsg crs should start with epsg!')

        # Load coordinate_files
        if isinstance(coordinate_files, str):
            if not os.path.exists(coordinate_files):
                raise ValueError(f'The {str(coordinate_files)} does not exist!')
            elif coordinate_files.endswith('.xlsx') or coordinate_files.endswith('.xls'):
                coordinate_file_df = pd.read_excel(coordinate_files)
            elif coordinate_files.endswith('.csv'):
                coordinate_file_df = pd.read_csv(coordinate_files)
            else:
                raise TypeError(f'The dem xlsx file should be a xlsx!')
        else:
            raise TypeError(f'The {str(coordinate_files)} should be a str!')

        # Import into the dataframe
        self.cross_section_bank_coord = {}
        for _ in self.cross_section_name:
            if _ in list(coordinate_file_df[coordinate_file_df.keys()[0]]):
                row_temp = coordinate_file_df[coordinate_file_df[coordinate_file_df.keys()[0]] == _].index
                left_bank_coord = (coordinate_file_df.loc[row_temp, coordinate_file_df.keys()[1]].values[0], coordinate_file_df.loc[row_temp, coordinate_file_df.keys()[2]].values[0])
                right_bank_coord = (coordinate_file_df.loc[row_temp, coordinate_file_df.keys()[3]].values[0], coordinate_file_df.loc[row_temp, coordinate_file_df.keys()[4]].values[0])
                self.cross_section_bank_coord[_] = [left_bank_coord, right_bank_coord]
            else:
                self.cross_section_bank_coord[_] = [(np.nan, np.nan), (np.nan, np.nan)]
        self.crs = epsg_crs
        self._construct_geodf()

    def import_section_tributary(self, tributary_files: str):

        # Check whether the cross section name is imported
        if self.cross_section_name is None or len(self.cross_section_name) == 0:
            raise ValueError('Please import the cross section standard information before the tributary!')

        # Load coordinate_files
        if isinstance(tributary_files, str):
            if not os.path.exists(tributary_files):
                raise ValueError(f'The {str(tributary_files)} does not exist!')
            elif tributary_files.endswith('.xlsx') or tributary_files.endswith('.xls'):
                tributary_file_df = pd.read_excel(tributary_files)
            elif tributary_files.endswith('.csv'):
                tributary_file_df = pd.read_csv(tributary_files)
            else:
                raise TypeError(f'The dem xlsx file should be a xlsx!')
        else:
            raise TypeError(f'The {str(tributary_files)} should be a str!')

        # Import tributary inform
        for _ in list(tributary_file_df[tributary_file_df.keys()[0]]):
            if _ in self.cross_section_name:
                pos = list(tributary_file_df[tributary_file_df.keys()[0]]).index(_)
                self.cross_section_tribu[_] = tributary_file_df[tributary_file_df.keys()[1]][pos]

        self._construct_geodf()

    def merged_hydro_inform(self, hydro_ds: HydrometricStationData):

        # Create the hydro inform dic
        self.hydro_inform_dic = {}

        # Detect the datatype
        if not isinstance(hydro_ds, HydrometricStationData):
            raise TypeError('The hydro ds is not a standard hydrometric station data!')

        # Merge hydro inform
        for _ in hydro_ds.cross_section_namelist:
            if _ in self.cross_section_name and ~np.isnan(hydro_ds.water_level_offset[hydro_ds.station_namelist[hydro_ds.cross_section_namelist.index(_)]]):
                wl_offset = hydro_ds.water_level_offset[hydro_ds.station_namelist[hydro_ds.cross_section_namelist.index(_)]]
                self.hydro_inform_dic[_] = hydro_ds.hydrological_inform_dic[hydro_ds.station_namelist[hydro_ds.cross_section_namelist.index(_)]]
                self.hydro_inform_dic[_]['water_level/m'] = self.hydro_inform_dic[_]['water_level/m'] + wl_offset

    def generate_differential_cross_profile(self, cross_section2):

        # Check the type of input
        if self.cross_section_dem is None:
            raise Exception('Please input the cross profiles before further process!')
        elif self.cross_section_bank_coord is None:
            raise Exception('Please input the bank coordinate before further process!')

        if not isinstance(cross_section2, CrossSection):
            raise Exception('The differential cross profile only work for two cross section!!')
        elif cross_section2.cross_section_dem is None:
            raise Exception('Please input the cross profiles before further process!')

        # Differential the rs dem
        self.cross_section_diff = {}
        self.cross_section_diff_geometry = {}
        for _ in self.cross_section_name:
            if _ in cross_section2.cross_section_name:
                diff_list = []
                ref_cross_section_dem = self.cross_section_dem[_]
                new_cross_section_dem = cross_section2.cross_section_dem[_]
                ref_dis, ref_ele = [__[0] for __ in ref_cross_section_dem], [__[1] for __ in ref_cross_section_dem]
                new_dis, new_ele = [__[0] for __ in new_cross_section_dem], [__[1] for __ in new_cross_section_dem]
                dis_sta, dis_end = [max(min(ref_dis), min(new_dis)), min(max(ref_dis), max(new_dis))]
                unique_ref_dis = copy.deepcopy(ref_dis)
                unique_ref_dis.extend(new_dis)
                unique_ref_dis = np.unique(np.array(unique_ref_dis)).tolist()
                for __ in unique_ref_dis:
                    if dis_sta <= __ <= dis_end:
                        ref_ele_t, new_ele_t = np.nan, np.nan

                        if __ in ref_dis:
                            ref_ele_t = ref_ele[ref_dis.index(__)]
                        else:
                            for ___ in range(0, len(ref_dis) - 1):
                                if (ref_dis[___] - __) * (ref_dis[___ + 1] - __) < 0:
                                    ref_ele_t = ref_ele[___] + (ref_ele[___ + 1] - ref_ele[___]) * (__ - ref_dis[___]) / (ref_dis[___ + 1] - ref_dis[___])
                                    break

                        if __ in new_dis:
                            new_ele_t = new_ele[new_dis.index(__)]
                        else:
                            for ___ in range(0, len(new_dis) - 1):
                                if (new_dis[___] - __) * (new_dis[___ + 1] - __) < 0:
                                    new_ele_t = new_ele[___] + (new_ele[___ + 1] - new_ele[___]) * (__ - new_dis[___]) / (new_dis[___ + 1] - new_dis[___])
                                    break

                        if ~np.isnan(ref_ele_t) and ~np.isnan(new_ele_t):
                            diff_list.append([__, new_ele_t - ref_ele_t])

                self.cross_section_diff[_] = diff_list

                line_coord = []
                bank_coord = self.cross_section_bank_coord[_]
                dis_ = distance_between_2points(bank_coord[0], bank_coord[1])
                ref_dis_ = ref_dis[-1] - ref_dis[0]
                new_dis_ = new_dis[-1] - new_dis[0]
                itr_xdim, itr_ydim = (bank_coord[1][0] - bank_coord[0][0]) / dis_, (bank_coord[1][1] - bank_coord[0][1]) / dis_
                offset = self.cross_section_dem[_][0][0] if abs(dis_ - ref_dis_) < abs(dis_ - ref_dis[-1]) else 0

                for dem_index in range(len(diff_list)):
                    dem_dis = diff_list[dem_index][0] - offset
                    line_coord.append((bank_coord[0][0] + dem_dis * itr_xdim, bank_coord[0][1] + dem_dis * itr_ydim, diff_list[dem_index][1]))
                self.cross_section_diff_geometry[_] = LineString(line_coord)

    def compare_inundation_frequency(self, dem_tif: str, year_range: list):

        # Import dem tif
        if isinstance(dem_tif, str):
            if not dem_tif.endswith('.TIF') and not dem_tif.endswith('.tif'):
                raise TypeError('The dem map should be a TIF file')
            else:
                try:
                    ds_temp = gdal.Open(dem_tif)
                    srs_temp = bf.retrieve_srs(ds_temp)
                    if int(srs_temp.split(':')[-1]) != int(self.crs.split(':')[-1]):
                        gdal.Warp('/vsimem/temp1.TIF', ds_temp, )
                        ds_temp = gdal.Open('/vsimem/temp1.TIF')
                    [ul_x, x_res, xt, ul_y, yt, y_res] = ds_temp.GetGeoTransform()
                    arr = ds_temp.GetRasterBand(1).ReadAsArray()
                except:
                    raise ValueError('The dem tif file is problematic!')
        else:
            raise TypeError('The dem map should be a TIF file')

        # Define output path
        output_path = self.work_env + 'output_Fig\\' + dem_tif.split('\\')[-1].split('.')[0] + '_inunfreq\\'
        bf.create_folder(output_path)

        # Define the if compare list
        inunfreq_insitu_list = []
        inunfreq_rs_list = []
        section_list = []
        dem_list = []
        dis_list = []

        # check if hydro inform is imported
        if 'hydro_inform_dic' not in self.__dict__.keys():
            raise ValueError('Please input the hydro inform first!')
        else:
            hydro_section_index = np.array(np.sort([self.cross_section_name.index(_) for _ in self.hydro_inform_dic.keys()])).tolist()
            year_list = [_ for _ in range(year_range[0], year_range[1])]
            for section_name in self.cross_section_name:
                index = self.cross_section_name.index(section_name)
                dis_start_series, dis_end_series = [None for _ in range(year_range[0], year_range[1])], [None for _ in range(year_range[0], year_range[1])]
                wl_start_series, wl_end_series = [[] for _ in range(year_range[0], year_range[1])], [[] for _ in range(year_range[0], year_range[1])]
                start_year_list, end_year_list = [_ for _ in range(year_range[0], year_range[1])], [_ for _ in range(year_range[0], year_range[1])]
                wl_start_index, wl_end_index = [_ for _ in hydro_section_index if _ <= index], [_ for _ in hydro_section_index if _ >= index]
                wl_all = []

                while len(start_year_list) != 0:
                    if len(wl_start_index) == 0:
                        raise Exception('The boundary hydrometric station should have all the data in the year range')
                    start_index = max(wl_start_index)
                    wl_series = self.hydro_inform_dic[self.cross_section_name[start_index]]
                    y_ = 0
                    while y_ < len(start_year_list):
                        year = start_year_list[y_]
                        wl_series_ = wl_series[wl_series['year'] == year]
                        if wl_series_.shape[0] != 0:
                            wl_start_series[year_list.index(year)].extend(np.array(wl_series_['water_level/m']).tolist())
                            dis_start_series[year_list.index(year)] = abs(self.cross_section_distance[section_name] - self.cross_section_distance[self.cross_section_name[start_index]])
                            start_year_list.remove(year)
                            y_ -= 1
                        y_ += 1
                    wl_start_index.remove(start_index)

                while len(end_year_list) != 0:
                    if len(wl_end_index) == 0:
                        raise Exception('The boundary hydrometric station should have all the data in the year range')
                    end_index = min(wl_end_index)
                    wl_series = self.hydro_inform_dic[self.cross_section_name[end_index]]
                    y_ = 0
                    while y_ < len(end_year_list):
                        year = end_year_list[y_]
                        wl_series_ = wl_series[wl_series['year'] == year]
                        if wl_series_.shape[0] != 0:
                            dis_end_series[year_list.index(year)] = abs(self.cross_section_distance[section_name] - self.cross_section_distance[self.cross_section_name[end_index]])
                            wl_end_series[year_list.index(year)].extend(np.array(wl_series_['water_level/m']).tolist())
                            end_year_list.remove(year)
                            y_ -= 1
                        y_ += 1
                    wl_end_index.remove(end_index)

                if len(start_year_list) != 0 or len(end_year_list) != 0:
                    raise Exception('The boundary hydrometric station should have all the data in the year range')

                for _ in range(len(dis_start_series)):
                    if dis_start_series[_] is None or dis_end_series[_] is None:
                        raise Exception(f'The {str(section_name)} is not full!')
                    elif len(wl_start_series[_]) != len(wl_end_series[_]):
                        raise Exception(f'The boundary hydrometric station of {str(section_name)} has not consistent information in year {str(year_list[_])}!')
                    else:
                        if dis_start_series[_] + dis_end_series[_] != 0:
                            wl_temp = np.array(wl_start_series[_]) + (np.array(wl_end_series[_]) - np.array(wl_start_series[_])) * (dis_start_series[_]) / (dis_start_series[_] + dis_end_series[_])
                        else:
                            wl_temp = np.array(wl_start_series[_])
                        wl_all.extend(wl_temp.tolist())

                if self.cross_section_geodf['cs_bank_coord'][index] != [(np.nan, np.nan), (np.nan, np.nan)] and self.cross_section_geodf['cs_bank_coord'][index] is not None:
                    if len(self.cross_section_geodf['cs_dem'][index]) == len(list(self.cross_section_geodf['geometry'][index].coords)):
                        cs_dem_ = [_[1] for _ in self.cross_section_geodf['cs_dem'][index]]
                        cs_channel_ = [_[1] <= min(wl_all) for _ in self.cross_section_geodf['cs_dem'][index]]
                        cs_dem_new = [None for _ in self.cross_section_geodf['cs_dem'][index]]
                        for _ in range(len(cs_dem_)):
                            if cs_channel_[_] is True:
                                cs_dem_new[_] = cs_dem_[_]
                            else:

                                pos_bank, neg_bank = None, None
                                # positive direction
                                for __ in range(len(cs_dem_)):
                                    if _ + __ >= len(cs_dem_):
                                        pos_bank = None
                                        break
                                    elif cs_channel_[_ + __] is True:
                                        if pos_bank is None:
                                            pos_bank = np.nan
                                        break
                                    elif cs_dem_[_ + __] > cs_dem_[_]:
                                        if pos_bank is None:
                                            pos_bank = cs_dem_[_ + __]
                                        else:
                                            pos_bank = max(cs_dem_[_ + __], pos_bank)

                                # negative direction
                                for __ in range(len(cs_dem_)):
                                    if _ - __ < 0:
                                        neg_bank = None
                                        break
                                    elif cs_channel_[_ - __] is True:
                                        if neg_bank is None:
                                            neg_bank = np.nan
                                        break
                                    elif cs_dem_[_ - __] > cs_dem_[_]:
                                        if neg_bank is None:
                                            neg_bank = cs_dem_[_ - __]
                                        else:
                                            neg_bank = max(cs_dem_[_ - __], neg_bank)

                                if np.nan not in [neg_bank, pos_bank]:
                                    if neg_bank is None and pos_bank is None:
                                        raise Exception('Code Error!')
                                    elif neg_bank is None:
                                        cs_dem_new[_] = pos_bank
                                    elif pos_bank is None:
                                        cs_dem_new[_] = neg_bank
                                    else:
                                        cs_dem_new[_] = min(pos_bank, neg_bank)
                                else:
                                    cs_dem_new[_] = cs_dem_[_]

                        rs_if_list = []
                        insitu_if_list = []

                        fig_temp, ax_temp = plt.subplots(figsize=(11, 5), constrained_layout=True)
                        ax_temp.scatter([_ for _ in range(len(cs_dem_))], cs_dem_, marker='s')
                        ax_temp.scatter([_ for _ in range(len(cs_dem_))], cs_dem_new, marker='o')
                        plt.savefig(output_path + self.cross_section_geodf['cs_name'][index] + '_dem.png', dpi=500)
                        plt.close()

                        for __ in range(len(cs_dem_new)):
                            coord_x = self.cross_section_geodf['geometry'][index].coords[__][0]
                            coord_y = self.cross_section_geodf['geometry'][index].coords[__][1]
                            insitu_if_list.append(np.sum(np.array(wl_all) >= cs_dem_new[__]) / len(wl_all))
                            pos_x_new = (coord_x - ul_x) / x_res
                            pos_y_new = (coord_y - ul_y) / y_res
                            pos_x = int(np.floor((coord_x - ul_x) / x_res))
                            pos_y = int(np.floor((coord_y - ul_y) / y_res))
                            if np.isnan(arr[pos_y, pos_x]):
                                rs_if_list.append(np.nan)
                            else:
                                x, y, z, dis, z_dis = [], [], [], [], []
                                for pos_y_temp in range(pos_y - 2, pos_y + 4):
                                    for pos_x_temp in range(pos_x - 2, pos_x + 4):
                                        if ~np.isnan(arr[pos_y_temp, pos_x_temp]):
                                            x.append(pos_x_temp + 0.5)
                                            y.append(pos_y_temp + 0.5)
                                            z.append(arr[pos_y_temp, pos_x_temp])
                                            dis.append(1 / (x_res * distance_between_2points([pos_x_new, pos_y_new], [pos_x_temp, pos_y_temp])))
                                            z_dis.append(arr[pos_y_temp, pos_x_temp] / (x_res * distance_between_2points([pos_x_new, pos_y_new], [pos_x_temp, pos_y_temp])))

                                v2 = np.sum(np.array(z_dis)) / np.sum(np.array(dis))
                                # func = interpolate.CloughTocher2DInterpolator(list(zip(x, y)), z)
                                # v = func(pos_x_new, pos_y_new)
                                if pos_x < arr.shape[1] and pos_y < arr.shape[0]:
                                    rs_if_list.append(v2)
                                else:
                                    rs_if_list.append(np.nan)

                        # Generate FIGURE
                        plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
                        plt.rc('font', size=22)
                        plt.rc('axes', linewidth=2)

                        if len(insitu_if_list) == len(rs_if_list) and len(insitu_if_list) == len(self.cross_section_geodf['cs_dem'][index]):
                            fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
                            offset = self.cross_section_geodf['cs_dem'][index][0][0]
                            insitu_if, insitu_dis, rs_if, rs_dis = [], [], [], []
                            for __ in range(len(self.cross_section_geodf['cs_dem'][index])):
                                insitu_dis.append(self.cross_section_geodf['cs_dem'][index][__][0] - offset)
                                insitu_if.append(insitu_if_list[__])
                                if ~np.isnan(rs_if_list[__]):
                                    inunfreq_rs_list.append(rs_if_list[__])
                                    inunfreq_insitu_list.append(insitu_if_list[__])
                                    section_list.append(section_name)
                                    dis_list.append(self.cross_section_geodf['cs_dem'][index][__][0] - offset)
                                    dem_list.append(cs_dem_[__])
                                    rs_dis.append(self.cross_section_geodf['cs_dem'][index][__][0] - offset)
                                    rs_if.append(rs_if_list[__])
                            ax_temp.set_ylim(-0.05, 1.05)
                            ax_temp.set_xlim(min(insitu_dis), (max(insitu_dis) // 100 + 1) * 100)
                            ax_temp.set_ylabel('Inundation frequency', fontname='Times New Roman', fontsize=24, fontweight='bold')
                            ax_temp.set_xlabel('Distance to left bank/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
                            ax_temp.scatter(insitu_dis, insitu_if, s=9 ** 2, color='none', edgecolor = (0, 0, 0), linewidth=2, marker='o', label='Cross-profile-based')
                            ax_temp.scatter(rs_dis, rs_if, s=10 ** 2, color=(1, 127/256, 14/256),  linewidth=2, marker='.', label='Landsat-derived')
                            ax_temp.legend(fontsize=22)
                            ax_temp.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                            ax_temp.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=22)
                            plt.savefig(output_path + self.cross_section_geodf['cs_name'][index] + '_inunfreq.png', dpi=500)
                            plt.close()
                            fig_temp = None
                            ax_temp = None

            df = pd.DataFrame({'cs_name': section_list, 'insitu_inun_freq': inunfreq_insitu_list, 'rs_inun_freq': inunfreq_rs_list, 'dem': dem_list, 'dis': dis_list})
            rmse = np.sqrt(np.sum((np.array(inunfreq_insitu_list) - np.array(inunfreq_rs_list)) ** 2))
            df.to_csv(output_path + f'inun_all_{str(rmse)[0:6]}.csv', encoding='GB18030')

    def _check_output_information_(self):

        if self.cross_section_geodf is None:
            raise Exception('Please import the basic information of cross section before the export!')
        elif 'geometry' not in list(self.cross_section_geodf.keys()):
            print('The geometry of cross section is not imported!')
        else:
            pass

    def generate_Thalweg(self):

        # Check the import inform
        self._check_output_information_()

        # Generate the thalweg
        self.cross_section_geodf = self.cross_section_geodf.sort_values('cs_distance2dam').reset_index(drop=True)
        Thalweg_list = []
        cs_list = []
        bank_inform = []
        for _ in range(self.cross_section_geodf.shape[0]):
            if self.cross_section_geodf['geometry'][_] is not None and True not in np.isnan(self.cross_section_geodf['geometry'][_].bounds) and self.cross_section_geodf['cs_tribu'][_] != 1:
                left_bank = [self.cross_section_geodf['geometry'][_].coords[0][0], self.cross_section_geodf['geometry'][_].coords[0][1]]
                right_bank = [self.cross_section_geodf['geometry'][_].coords[-1][0], self.cross_section_geodf['geometry'][_].coords[-1][1]]
                dis = np.sqrt((right_bank[0] - left_bank[0]) ** 2 + (right_bank[1] - left_bank[1]) ** 2)
                dem_arr = np.array(self.cross_section_geodf['cs_dem'][_])
                if dem_arr[0, 0] < 0:
                    dis_minus = dem_arr[-1, 0] - dem_arr[0, 0]
                    dis_without_minus = dem_arr[-1, 0]
                    if abs(dis - dis_minus) <= abs(dis_without_minus - dis):
                        dem_temp = copy.deepcopy(dem_arr)
                        dem_temp[:, 0] = dem_temp[:, 0] + abs(dem_arr[0, 0])
                    elif abs(dis - dis_minus) > abs(dis_without_minus - dis):
                        dem_temp = np.delete(dem_arr, np.argwhere(dem_arr[:, 0] < 0), axis=0)
                    else:
                        raise Exception('Code Error')
                elif dem_arr[0, 0] >= 0:
                    dis_plus = dem_arr[-1, 0] - dem_arr[0, 0]
                    dis_without_plus = dem_arr[-1, 0]
                    if abs(dis - dis_plus) <= abs(dis_without_plus - dis):
                        dem_temp = copy.deepcopy(dem_arr)
                        dem_temp[:, 0] = dem_temp[:, 0] - abs(dem_arr[0, 0])
                    elif abs(dis - dis_plus) > abs(dis_without_plus - dis):
                        dem_temp = copy.deepcopy(dem_arr)
                    else:
                        raise Exception('Code Error')
                else:
                    raise Exception('Code Error')

                lowest_pos = np.argwhere(dem_temp[:, 1] == (min(dem_temp[:, 1])))
                if lowest_pos.shape[0] == 1:
                    pos_temp = lowest_pos[0, 0]
                    lowest_dis = dem_temp[pos_temp, 0]
                    lowest_ele = dem_temp[pos_temp, 1]
                elif lowest_pos.shape[0] >= 1:
                    bank_inform = [dem_temp[__, 0] for __ in lowest_pos]
                    bank_inform = [abs(__ - bank_inform[-1]) for __ in lowest_pos]
                    pos_temp = lowest_pos[bank_inform.index(min(bank_inform)), 0]
                    lowest_dis = dem_temp[pos_temp, 0]
                    lowest_ele = dem_temp[pos_temp, 1]
                else:
                    raise Exception('dem is missing')
                Thalweg_list.append([left_bank[0] + lowest_dis * (right_bank[0] - left_bank[0]) / np.sqrt((right_bank[0] - left_bank[0]) ** 2 + (right_bank[1] - left_bank[1]) ** 2),
                                     left_bank[1] + lowest_dis * (right_bank[1] - left_bank[1]) / np.sqrt((right_bank[0] - left_bank[0]) ** 2 + (right_bank[1] - left_bank[1]) ** 2), lowest_ele])
                bank_inform.append(lowest_dis)
                cs_list.append(self.cross_section_geodf['cs_name'][_])
            else:
                pass

        # Initiate Thalweg elevation
        thalweg_ele = Thalweg()
        thalweg_ele.original_cs = self
        thalweg_ele.work_env = self.work_env
        thalweg_ele.Thalweg_cs_name = cs_list
        thalweg_ele.Thalweg_Linestring = LineString(Thalweg_list)
        dic = {'cs_namelist': [cs_list], 'geometry': [thalweg_ele.Thalweg_Linestring]}
        thalweg_ele.Thalweg_geodf = gp.GeoDataFrame(dic, crs=self.cross_section_geodf.crs)
        return thalweg_ele

    def compare_2ddem(self, dem_tif: str, diff_ele=False, inun_freq=None):

        # Check the condition
        if diff_ele is True and (self.cross_section_diff is None or self.cross_section_diff_geometry is None):
            raise Exception('Please calculate the difference before comparison!')

        # Import dem tif
        if isinstance(dem_tif, str):
            if not dem_tif.endswith('.TIF') and not dem_tif.endswith('.tif'):
                raise TypeError('The dem map should be a TIF file')
            else:
                try:
                    ds_temp = gdal.Open(dem_tif)
                    srs_temp = retrieve_srs(ds_temp)
                    if int(srs_temp.split(':')[-1]) != int(self.crs.split(':')[-1]):
                        gdal.Warp('/vsimem/temp1.TIF', ds_temp, )
                        ds_temp = gdal.Open('/vsimem/temp1.TIF')
                    [ul_x, x_res, xt, ul_y, yt, y_res] = ds_temp.GetGeoTransform()
                    arr = ds_temp.GetRasterBand(1).ReadAsArray()
                except:
                    raise ValueError('The dem tif file is problematic!')
        else:
            raise TypeError('The dem map should be a TIF file')

        # Import inun_freq
        if inun_freq is not None and isinstance(inun_freq, str):
            if not inun_freq.endswith('.TIF') and not inun_freq.endswith('.tif'):
                raise TypeError('The inun_freq map should be a TIF file')
            else:
                try:
                    ds_temp = gdal.Open(inun_freq)
                    srs_temp = retrieve_srs(ds_temp)
                    if int(srs_temp.split(':')[-1]) != int(self.crs.split(':')[-1]):
                        gdal.Warp('/vsimem/temp1.TIF', ds_temp, )
                        ds_temp = gdal.Open('/vsimem/temp1.TIF')
                    [ul_x, x_res, xt, ul_y, yt, y_res] = ds_temp.GetGeoTransform()
                    inun_freq_arr = ds_temp.GetRasterBand(1).ReadAsArray()
                    inun_freq_factor = True
                except:
                    raise ValueError('The inun_freq file is problematic!')
        else:
            raise TypeError('The inun_freq should be a TIF file')

        # Generate FIGURE para
        insitu_dem_all, rs_dem_all, section_name_all, rs_dis_all, dem_diff_all = [], [], [], [], []
        inun_freq_all = [] if inun_freq_factor else None
        plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
        plt.rc('font', size=24)
        plt.rc('axes', linewidth=2)
        output_path = self.work_env + 'output_Fig\\' + dem_tif.split('\\')[-1].split('.')[0] + '\\' if diff_ele is not True else self.work_env + 'output_Fig\\' + dem_tif.split('\\')[-1].split('.')[0] + '_diff_ele\\'
        bf.create_folder(output_path)

        # Compare dem
        RS_dem_all = {}
        for _ in range(self.cross_section_geodf.shape[0]):
            if self.cross_section_geodf['cs_bank_coord'][_] != [(np.nan, np.nan), (np.nan, np.nan)] and self.cross_section_geodf['cs_bank_coord'][_] is not None:
                RS_dem = []
                RS_inun_freq = []
                cs_name_ = self.cross_section_geodf['cs_name'][_]
                if diff_ele:
                    if cs_name_ in self.cross_section_diff.keys() and cs_name_ in self.cross_section_diff_geometry.keys():
                        dem_temp = self.cross_section_diff[cs_name_]
                        coords_temp = self.cross_section_diff_geometry[cs_name_]
                    else:
                        dem_temp = None
                elif len(self.cross_section_geodf['cs_dem'][_]) == len(list(self.cross_section_geodf['geometry'][_].coords)):
                    dem_temp = self.cross_section_geodf['cs_dem'][_]
                    coords_temp = self.cross_section_geodf['geometry'][_]
                else:
                    dem_temp = None

                if dem_temp is not None:
                    for __ in range(len(dem_temp)):
                        coord_x = coords_temp.coords[__][0]
                        coord_y = coords_temp.coords[__][1]
                        pos_x_new = (coord_x - ul_x) / x_res
                        pos_y_new = (coord_y - ul_y) / y_res
                        pos_x = int(np.floor((coord_x - ul_x) / x_res))
                        pos_y = int(np.floor((coord_y - ul_y) / y_res))
                        if np.isnan(arr[pos_y, pos_x]):
                            if inun_freq_factor:
                                RS_dem.append([np.nan, np.nan, np.nan])
                            else:
                                RS_dem.append([np.nan, np.nan])
                        else:
                            x, y, z, dis, z_dis = [], [], [], [], []
                            inun_freq_l = [] if inun_freq_factor else None
                            inun_freq_dis = [] if inun_freq_factor else None
                            for pos_y_temp in range(pos_y - 2, pos_y + 4):
                                for pos_x_temp in range(pos_x - 2, pos_x + 4):
                                    if ~np.isnan(arr[pos_y_temp, pos_x_temp]):
                                        x.append(pos_x_temp + 0.5)
                                        y.append(pos_y_temp + 0.5)
                                        z.append(arr[pos_y_temp, pos_x_temp])
                                        dis.append(1 / (x_res * distance_between_2points([pos_x_new, pos_y_new], [pos_x_temp, pos_y_temp])))
                                        z_dis.append(arr[pos_y_temp, pos_x_temp] / (x_res * distance_between_2points([pos_x_new, pos_y_new], [pos_x_temp, pos_y_temp])))
                                        if inun_freq_factor:
                                            inun_freq_l.append(1 / (x_res * distance_between_2points([pos_x_new, pos_y_new], [pos_x_temp, pos_y_temp])))
                                            inun_freq_dis.append(inun_freq_arr[pos_y_temp, pos_x_temp] / (x_res * distance_between_2points([pos_x_new, pos_y_new], [pos_x_temp, pos_y_temp])))

                            v2 = np.sum(np.array(z_dis)) / np.sum(np.array(dis))
                            if inun_freq_factor:
                                v3 = np.sum(np.array(inun_freq_l)) / np.sum(np.array(inun_freq_dis))
                            # func = interpolate.CloughTocher2DInterpolator(list(zip(x, y)), z)
                            # v = func(pos_x_new, pos_y_new)
                            if pos_x < arr.shape[1] and pos_y < arr.shape[0]:
                                if inun_freq_factor:
                                    RS_dem.append([coords_temp.coords[__], v2, v3])
                                else:
                                    RS_dem.append([coords_temp.coords[__], v2])
                            else:
                                if inun_freq_factor:
                                    RS_dem.append([np.nan, np.nan, np.nan])
                                else:
                                    RS_dem.append([np.nan, np.nan])

                    RS_dem_all[cs_name_] = RS_dem
                else:
                    RS_dem_all[cs_name_] = None
            else:
                RS_dem_all[cs_name_] = None

            if RS_dem_all[cs_name_] is not None:
                fig_temp, ax_temp = plt.subplots(figsize=(11, 5), constrained_layout=True)
                offset = dem_temp[0][0]
                insitu_dem, insitu_dis, rs_dem, rs_dis = [], [], [], []

                for __ in range(len(dem_temp)):
                    insitu_dis.append(dem_temp[__][0] - offset)
                    insitu_dem.append(dem_temp[__][1])
                    insitu_dem_all.append(dem_temp[__][1])

                    section_name_all.append(cs_name_)
                    rs_dis_all.append(dem_temp[__][0] - offset)

                    if ~np.isnan(RS_dem_all[cs_name_][__][1]):
                        rs_dis.append(dem_temp[__][0] - offset)
                        rs_dem_all.append(RS_dem_all[cs_name_][__][1])
                        rs_dem.append(RS_dem_all[cs_name_][__][1])
                        dem_diff_all.append(RS_dem_all[cs_name_][__][1] - dem_temp[__][1])
                    else:
                        rs_dem_all.append(np.nan)
                        dem_diff_all.append(np.nan)

                    if inun_freq_factor:
                        inun_freq_all.append(RS_dem_all[cs_name_][__][2])

                ax_temp.scatter(insitu_dis, insitu_dem, s=12**2, color="none", edgecolor=(0, 1, 0), linewidth=3, marker='o')
                ax_temp.scatter(rs_dis, rs_dem, s=12**2, color="none", edgecolor=(1, 0, 0), linewidth=3, marker='s')
                plt.savefig(output_path + cs_name_ + '.png', dpi=500)
                plt.close()
                fig_temp = None
                ax_temp = None

        if inun_freq_factor:
            df = pd.DataFrame({'csname': section_name_all, 'rs_dem': rs_dem_all, 'insitu_dem': insitu_dem_all, 'rs_dis': rs_dis_all, 'inun_freq': inun_freq_all, 'dem_diff': dem_diff_all})
        else:
            df = pd.DataFrame({'csname': section_name_all, 'rs_dem': rs_dem_all, 'insitu_dem': insitu_dem_all, 'rs_dis': rs_dis_all})
        rmse = np.sqrt(np.sum((np.array(rs_dem_all) - np.array(insitu_dem_all)) ** 2))
        df.to_csv(output_path + f'dem_all_{str(rmse)[0:6]}.csv', encoding='GB18030')

    def load_geojson(self, geodf_json: str):

        # Process work env
        self.work_env = bf.Path(os.path.dirname(geodf_json)).path_name

        # Check the csv existence
        if isinstance(geodf_json, str):
            if not os.path.exists(geodf_json):
                raise ValueError(f'The {str(geodf_json)} does not exist!')
            elif geodf_json.endswith('.json'):
                self.cross_section_geodf = gp.read_file(geodf_json)
            else:
                raise TypeError(f'The dem xlsx file should be a xlsx!')
        else:
            raise TypeError(f'The {str(geodf_json)} should be a str!')

        # import from geodf
        self._extract_from_geodf()
        return self

    def to_geojson(self, output_path: str = None):
        self._check_output_information_()

        # Export2geojson
        if output_path is None:
            output_path = self.work_env + 'output_geojson\\'
        else:
            output_path = bf.Path(output_path).path_name
        bf.create_folder(output_path)

        cross_section_temp = copy.deepcopy(self.cross_section_geodf)
        cross_section_temp['cs_dem'] = cross_section_temp['cs_dem'].astype(str)
        cross_section_temp['cs_bank_coord'] = cross_section_temp['cs_bank_coord'].astype(str)
        cross_section_temp.to_file(f'{output_path}CrossSection.json', driver='GeoJSON')
        print('The cross-section information is saved as geojson.')

    def to_csv(self, output_path: str = None):
        self._check_output_information_()
        print('The cross-section information is saved as csv file. Please mentioned the csv file is only for visualisation!')

        # Export2csv for visualisation
        if output_path is None:
            output_path = self.work_env + 'output_csv\\'
        else:
            output_path = bf.Path(output_path).path_name

        bf.create_folder(output_path)
        self.cross_section_geodf.to_csv(f'{output_path}CrossSection.csv', encoding='GB18030')

    def to_shpfile(self, output_path: str = None):

        # Check the import inform
        self._check_output_information_()

        # Export 2 shpfile
        if output_path is None:
            output_path = self.work_env + 'output_shpfile\\'
        else:
            output_path = bf.Path(output_path).path_name

        bf.create_folder(output_path)
        cross_section_geodata_temp = copy.deepcopy(self.cross_section_geodf)
        cross_section_geodata_temp['cs_dem'] = cross_section_geodata_temp['cs_dem'].astype(str)
        cross_section_geodata_temp['cs_bank_coord'] = cross_section_geodata_temp['cs_bank_coord'].astype(str)
        cross_section_geodata_temp['cs_tribu'] = cross_section_geodata_temp['cs_tribu'].astype(np.int16)
        cross_section_geodata_temp.to_file(f'{output_path}cs_combined.shp', encoding='utf-8')
