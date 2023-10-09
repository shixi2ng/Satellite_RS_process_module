import pandas as pd
import numpy as np
import geopandas as gp
import os

import shapely
from shapely import LineString, Geometry, Point
import basic_function as bf
import copy
from datetime import datetime
import traceback
from osgeo import gdal
from River_GIS.utils import *
from tqdm.auto import tqdm
from shapely.ops import nearest_points
import concurrent.futures
from itertools import repeat


class hydrometric_station_data(object):

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

    def load_csvs(self, csv_filelist):
        for _ in csv_filelist:
            pass


class Thelwag(object):

    def __init__(self):

        # Define work env
        self.work_env = None

        # Define the property of cross section
        self.Thelwag_cs_namelist = None
        self.Thelwag_Linestring = None
        self.Thelwag_geodf = None
        self.original_cs = None

        # Define smooth index
        self.smoothed_Thelwag = None
        self.smoothed_cs_index = None

        # Define the property of cross section
        self.crs = None

    def _extract_Thelwag_geodf(self):

        # Check the keys
        try:
            self.crs = self.Thelwag_geodf.crs
        except:
            raise ValueError('The crs of geodf is missing')

        if False in [_ in list(self.Thelwag_geodf.keys()) for _ in ['cs_namelist', 'geometry']]:
            missing_keys = [_ for _ in ['cs_namelist', 'geometry'] if _ not in list(self.Thelwag_geodf.keys())]
            raise KeyError(f'The key {str(missing_keys)} of geodf is missing!')

        # Restruct the thelwag
        if len(self.Thelwag_geodf['cs_namelist'][0]) == self.Thelwag_geodf['geometry'][0].coords.__len__():
            self.Thelwag_cs_namelist = self.Thelwag_geodf['cs_namelist'][0]
            self.Thelwag_Linestring = self.Thelwag_geodf['geometry'][0]
        else:
            raise Exception('The thelwag has inconsistent cross section name and linstring!')

    def _struct_Thelwag_geodf(self):

        if self.Thelwag_cs_namelist is None or self.Thelwag_Linestring is None or len(self.Thelwag_cs_namelist) == 0 or self.Thelwag_Linestring.coords.__len__() == 0:
            raise ValueError('No information concerning the thelwag is imported')
        else:
            if len(self.Thelwag_cs_namelist) == self.Thelwag_Linestring.coords.__len__():
                thelwag_dic = {'cs_namelist': self.Thelwag_cs_namelist}
                self.Thelwag_geodf = gp.GeoDataFrame(thelwag_dic)

                if self.crs is not None:
                    self.Thelwag_geodf = self.Thelwag_geodf.set_crs(self.crs)

    # def load_shapefile(self, shapefile):
    #
    #     # Process work env
    #     self.work_env = bf.Path(os.path.dirname(shapefile)).path_name
    #
    #     # Check the shapefile existence
    #     if isinstance(shapefile, str):
    #         if not os.path.exists(shapefile):
    #             raise ValueError(f'The {str(shapefile)} does not exist!')
    #         elif shapefile.endswith('.shp'):
    #             self.Thelwag_geodf = gp.read_file(shapefile, encoding='utf-8')
    #         else:
    #             raise TypeError(f'The geodf json file should be a json!')
    #     else:
    #         raise TypeError(f'The {str(shapefile)} should be a str!')
    #
    #     # Extract information from geodf
    #     cs_name_temp = self.Thelwag_geodf['cs_namelis'][0]
    #     cs_name = []
    #     for _ in cs_name_temp.split("',"):
    #         cs_name.append(_.split("'")[-1])
    #
    #     self._extract_Thelwag_geodf()
    #     return self

    def load_smooth_Thalweg_shp(self, shpfile):

        # Check the shapefile existence
        if isinstance(shpfile, str):
            if not os.path.exists(shpfile):
                raise ValueError(f'The {str(shpfile)} does not exist!')
            elif shpfile.endswith('.shp'):
                Thelwag_geodf_temp = gp.read_file(shpfile, encoding='utf-8')
            else:
                raise TypeError(f'The geodf json file should be a json!')
        else:
            raise TypeError(f'The {str(shpfile)} should be a str!')

        # Extract information from geodf
        if Thelwag_geodf_temp.shape[0] != 1 or type(Thelwag_geodf_temp['geometry'][0]) != LineString:
            raise ValueError('The shpfile should be a string type')
        else:
            self.smoothed_Thelwag = Thelwag_geodf_temp['geometry'][0]
            self.smoothed_cs_index = []

            for _ in range(len(self.Thelwag_cs_namelist)):
                simplified_thelwag_arr = np.array(self.smoothed_Thelwag.coords)
                if _ == 0:
                    self.smoothed_cs_index.append(0)
                elif _ == len(self.Thelwag_cs_namelist) - 1:
                    self.smoothed_cs_index.append(simplified_thelwag_arr.shape[0] - 1)
                else:
                    cs_line = list(self.original_cs.cross_section_geodf['geometry'][self.original_cs.cross_section_geodf['cs_name'] == self.Thelwag_cs_namelist[_]])[0]
                    intersect = shapely.intersection(cs_line, self.smoothed_Thelwag)
                    if not isinstance(intersect, Point):
                        raise Exception(f'Smooth line is not intersected with cross section {str(self.Thelwag_cs_namelist[_])}')

                    start_vertex = determin_start_vertex_of_point(intersect, self.smoothed_Thelwag)
                    arr_ = np.zeros([1, simplified_thelwag_arr.shape[1]])
                    arr_[0, 0], arr_[0, 1], arr_[0, 2] = intersect.coords[0][0], intersect.coords[0][1], intersect.coords[0][2]
                    simplified_thelwag_arr = np.insert(simplified_thelwag_arr, start_vertex + 1, arr_, axis=0)
                    self.smoothed_cs_index.append(start_vertex + 1)

                    if arr_.shape[1] == 2:
                        smoothed_thelwag_list = [(simplified_thelwag_arr[_, 0], simplified_thelwag_arr[_, 1]) for _ in range(simplified_thelwag_arr.shape[0])]
                    elif arr_.shape[1] == 3:
                        smoothed_thelwag_list = [(simplified_thelwag_arr[_, 0], simplified_thelwag_arr[_, 1], simplified_thelwag_arr[_, 2]) for _ in range(simplified_thelwag_arr.shape[0])]
                    else:
                        raise Exception('Code error!')
                    self.smoothed_Thelwag = LineString(smoothed_thelwag_list)
            # geodf = gp.GeoDataFrame(data=[{'a': 'b'}], geometry=[self.smoothed_Thelwag])
            # geodf.to_file('G:\A_Landsat_veg\Water_level_python\\a.shp')

    def load_geojson(self, geodf_json):

        # Process work env
        self.work_env = bf.Path(os.path.dirname(os.path.dirname(geodf_json))).path_name

        # Check the df json existence
        if isinstance(geodf_json, str):
            if not os.path.exists(geodf_json):
                raise ValueError(f'The {str(geodf_json)} does not exist!')
            elif geodf_json.endswith('.json'):
                self.Thelwag_geodf = gp.read_file(geodf_json)
            else:
                raise TypeError(f'The geodf json file should be a json!')
        else:
            raise TypeError(f'The {str(geodf_json)} should be a str!')

        # Check the cs json existence
        try:
            cs_json = os.path.dirname(geodf_json) + '\\cross_section.json'
        except:
            print(traceback.format_exc())
            raise Exception('The cs json can not be generated!')

        if isinstance(cs_json, str):
            if not os.path.exists(cs_json):
                raise ValueError(f"The cross_section json does not exist!")
            elif cs_json.endswith('.json'):
                self.original_cs = cross_section()
                self.original_cs = self.original_cs.load_geojson(cs_json)
            else:
                raise TypeError(f'The cs_json file should be a json!')
        else:
            raise TypeError(f'The {str(cs_json)} should be a str!')

        # Extract information from geodf
        self._extract_Thelwag_geodf()
        return self

    def to_geojson(self,  output_path: str = None):

        # Define work path
        if output_path is None:
            output_path = self.work_env + 'output_geojson\\'
        else:
            output_path = bf.Path(output_path).path_name
        bf.create_folder(output_path)

        # To geojson
        if isinstance(self.Thelwag_geodf, gp.GeoDataFrame):
            self.Thelwag_geodf['cs_namelist'] = self.Thelwag_geodf['cs_namelist'].astype(str)
            self.Thelwag_geodf.to_file(output_path + 'thelwag.json', driver='GeoJSON')

        if isinstance(self.original_cs, cross_section):
            self.original_cs.to_geojson(output_path)

    def to_shapefile(self,  output_path: str = None):

        # Define work path
        if output_path is None:
            output_path = self.work_env + 'output_shpfile\\'
        else:
            output_path = bf.Path(output_path).path_name
        bf.create_folder(output_path)

        # To shapefile
        if isinstance(self.Thelwag_geodf, gp.GeoDataFrame):
            self.Thelwag_geodf['cs_namelist'] = self.Thelwag_geodf['cs_namelist'].astype(str)
            self.Thelwag_geodf.to_file(output_path + 'thelwag.shp', encoding='utf-8')

        if isinstance(self.original_cs, cross_section):
            self.original_cs.to_shpfile(output_path)

    def merged_hydro_inform(self, hydro_ds: hydrometric_station_data):

        # Create the hydro inform dic
        self.hydro_inform_dic = {}

        # Detect the datatype
        if not isinstance(hydro_ds, hydrometric_station_data):
            raise TypeError('The hydrods is not a standard hydrometric station data!')

        # Merge hydro inform
        for _ in hydro_ds.cross_section_namelist:
            if _ in self.Thelwag_cs_namelist and ~np.isnan(hydro_ds.water_level_offset[hydro_ds.station_namelist[hydro_ds.cross_section_namelist.index(_)]]):
                self.hydro_inform_dic[_] = hydro_ds.hydrological_inform_dic[hydro_ds.station_namelist[hydro_ds.cross_section_namelist.index(_)]]

    def link_inundation_frequency_map(self, inundation_frequency_tif: str, year_range: list = [1900, 2100]):

        # Check if the hydro inform is merged
        if 'hydro_inform_dic' not in self.__dict__.keys():
            raise Exception('Please merged standard hydrometric_station_ds before linkage!')

        # Process cross section inform
        cs_list, year_domain, hydro_pos = [], [], []
        if self.smoothed_Thelwag is None:
            for _ in range(len(self.Thelwag_cs_namelist)):
                if self.Thelwag_cs_namelist[_] in self.hydro_inform_dic.keys():
                    year_domain.append(np.unique(np.array(self.hydro_inform_dic[self.Thelwag_cs_namelist[_]]['year'])).tolist())
                    hydro_pos.append(_)
                    cs_list.append(self.Thelwag_cs_namelist[_])

        elif isinstance(self.smoothed_Thelwag, LineString):
            for _ in range(len(self.Thelwag_cs_namelist)):
                pos = self.smoothed_cs_index[_]
                if self.Thelwag_cs_namelist[_] in self.hydro_inform_dic.keys():
                    year_domain.append(np.unique(np.array(self.hydro_inform_dic[self.Thelwag_cs_namelist[_]]['year'])).tolist())
                    hydro_pos.append(pos)
                    cs_list.append(self.Thelwag_cs_namelist[_])

        else:
            raise TypeError('The smoothed thelwag is not under the correct type')

        # Import Inundation_frequency_tif
        if isinstance(inundation_frequency_tif, str):
            if not inundation_frequency_tif.endswith('.TIF') and not inundation_frequency_tif.endswith('.tif'):
                raise TypeError('The inundation frequency map should be a TIF file')
            else:
                try:
                    ds_temp = gdal.Open(inundation_frequency_tif)
                    srs_temp = retrieve_srs(ds_temp)
                    if int(srs_temp.split(':')[-1]) != self.crs.to_epsg():
                        gdal.Warp('/vsimem/temp1.TIF', ds_temp, )
                        ds_temp = gdal.Open('/vsimem/temp1.TIF')
                    [ul_x, x_res, xt, ul_y, yt, y_res] = ds_temp.GetGeoTransform()
                    arr = ds_temp.GetRasterBand(1).ReadAsArray()
                except:
                    raise ValueError('The inundation frequecy tif file is problematic!')
        else:
            raise TypeError('The inundation frequency map should be a TIF file')

        # Process the inundation frequency map
        ele_arr = np.zeros_like(arr)
        ele_arr[np.isnan(arr)] = np.nan
        inun_arr = np.zeros_like(arr)
        inun_arr[np.isnan(arr)] = np.nan

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
            res = exe.map(frquency_based_elevation, arr_pd_list, repeat(self), repeat(year_range), repeat([ul_x, x_res, ul_y, y_res]), repeat(cs_list), repeat(year_domain), repeat(hydro_pos))

        res_df = None
        res = list(res)
        for result_temp in res:
            if res_df is None:
                res_df = copy.copy(result_temp)
            else:
                res_df = pd.concat([res_df, result_temp])

        res_df = res_df.reset_index(drop=True)
        for _ in range(res_df.shape[0]):
            ele_arr[int(res_df['y'][_]), int(res_df['x'][_])] = res_df['wl'][_]
            inun_arr[int(res_df['y'][_]), int(res_df['x'][_])] = res_df['fr'][_]
        bf.write_raster(ds_temp, ele_arr, self.work_env, 'ele_' + inundation_frequency_tif.split('\\')[-1], raster_datatype=gdal.GDT_Float32)
        bf.write_raster(ds_temp, inun_arr, self.work_env, 'inun_' + inundation_frequency_tif.split('\\')[-1], raster_datatype=gdal.GDT_Float32)


class cross_section(object):

    def __init__(self):

        # Define work env
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

    def from_standard_xlsx(self, dem_xlsx_filename: str):

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
            raise ValueError('Please import the cross section standard information before the coordianates!')

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
            raise ValueError('Please import the cross section standard information before the coordianates!')

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
        thalweg_ele = Thelwag()
        thalweg_ele.original_cs = self
        thalweg_ele.work_env = self.work_env
        thalweg_ele.Thelwag_cs_name = cs_list
        thalweg_ele.Thelwag_Linestring = LineString(Thalweg_list)
        dic = {'cs_namelist': [cs_list], 'geometry': [thalweg_ele.Thelwag_Linestring]}
        thalweg_ele.Thelwag_geodf = gp.GeoDataFrame(dic, crs=self.cross_section_geodf.crs)
        return thalweg_ele

    def import_2d_dem(self, dem_tif: str):

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

        # Compare dem
        for _ in range(self.cross_section_geodf.shape[0]):

            if self.cross_section_bank_coord[_] != [(np.nan, np.nan), (np.nan, np.nan)] and self.cross_section_bank_coord[_] is not None:
                pass



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
        cross_section_temp.to_file(f'{output_path}cross_section.json', driver='GeoJSON')
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
        self.cross_section_geodf.to_csv(f'{output_path}cross_section.csv', encoding='GB18030')

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


if __name__ == '__main__':

    # Cross section construction
    cs1 = cross_section()
    cs1.from_standard_xlsx('G:\A_Landsat_veg\Water_level_python\\cross_section_DEM_2019_all.csv')
    cs1.import_section_coordinates('G:\A_Landsat_veg\Water_level_python\\cross_section_coordinates_wgs84.csv', epsg_crs='epsg:32649')
    cs1.import_section_tributary('G:\A_Landsat_veg\Water_level_python\\cross_section_tributary.xlsx')
    cs1.to_geojson()
    cs1.to_shpfile()
    cs1.to_csv()
    cs1.import_2d_dem('G:\A_Landsat_veg\Water_level_python\\ele_DT_inundation_frequency_posttgd.TIF')

    # Cross section import
    thal1 = cs1.generate_Thalweg()
    thal1.to_shapefile()
    thal1.to_geojson()
    thal1 = Thelwag()
    thal1 = thal1.load_geojson('G:\\A_Landsat_veg\\Water_level_python\\output_geojson\\thelwag.json')
    thal1.load_smooth_Thalweg_shp('G:\\A_Landsat_veg\\Water_level_python\\output_shpfile\\thelwag_smooth.shp')
#
    # Water level import
    wl1 = hydrometric_station_data()
    file_list = bf.file_filter('G:\A_Landsat_veg\Water_level_python\original_water_level\\', ['.xls'])
    corr_temp = pd.read_csv('G:\A_Landsat_veg\Water_level_python\original_water_level\\对应表.csv')
    cs_list, wl_list = [], []
    for file_ in file_list:
        for hs_num in range(corr_temp.shape[0]):
            hs = corr_temp[corr_temp.keys()[1]][hs_num]
            if hs in file_:
                cs_list.append(corr_temp[corr_temp.keys()[0]][hs_num])
                wl_list.append(corr_temp[corr_temp.keys()[2]][hs_num])

    for fn_, cs_, wl_ in zip(file_list, cs_list, wl_list):
        wl1.import_from_standard_excel(fn_, cs_, water_level_offset=wl_)
    wl1.to_csvs()
#
    thal1.merged_hydro_inform(wl1)
    thal1.link_inundation_frequency_map('G:\A_Landsat_veg\Landsat_floodplain_2020_datacube\Inundation_DT_datacube\inun_factor\\DT_inundation_frequency_posttgd.TIF', year_range=[2004, 2019])
    thal1.link_inundation_frequency_map('G:\A_Landsat_veg\Landsat_floodplain_2020_datacube\Inundation_DT_datacube\inun_factor\\DT_inundation_frequency_pretgd.TIF', year_range=[1986, 2004])
