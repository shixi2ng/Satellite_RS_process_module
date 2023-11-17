import pandas as pd
import numpy as np
import geopandas as gp
import os
from shapely import LineString, Geometry, Point
import basic_function as bf
import copy
from datetime import datetime
import traceback
from osgeo import gdal
from tqdm.auto import tqdm
from shapely.ops import nearest_points
from osgeo import osr
import time
import scipy.sparse as sm
from NDsm import NDSparseMatrix
import rasterio.features
from River_GIS.River_centreline import *


def multiple_concept_model(year, thal, ):
    hydrodc1 = HydroDatacube()
    hydrodc1.import_from_matrix(f'G:\\A_Landsat_veg\\Water_level_python\\hydrodatacube\\{str(year)}\\')
    hydrodc1.simplified_conceptual_inundation_model(
        'G:\\A_Landsat_veg\\Water_level_python\\Post_TGD\\ele_pretgd4model.TIF', thal,
        'G:\A_Landsat_veg\Water_level_python\inundation_status\\prewl_predem\\')


def concept_inundation_model(wl_nm, wl_sm, demfile, thalweg, output_filepath):

    # Import datacube
    if demfile.endswith('.tif') or demfile.endswith('.TIF'):
        dem_file_ds = gdal.Open(demfile)
        dem_file_arr = dem_file_ds.GetRasterBand(1).ReadAsArray()
    else:
        raise TypeError('Please input the dem file with right type')

    for _ in range(len(wl_nm)):
        try:
            wl_sm_, wl_nm_ = wl_sm[_], wl_nm[_]
            if not os.path.exists(f'{output_filepath}\\inundation_final\\{str(wl_nm_)}.tif'):
                if wl_sm_.shape[0] != dem_file_arr.shape[0]:
                    wl_sm_ = np.row_stack((wl_sm_.toarray(), np.zeros([dem_file_arr.shape[0] - wl_sm_.shape[0], wl_sm_.shape[1]])))

                if wl_sm_.shape[1] != dem_file_arr.shape[1]:
                    wl_sm_ = np.column_stack((wl_sm_.toarray(), np.zeros([wl_sm_.shape[0], dem_file_arr.shape[1] - wl_sm_.shape[1]])))

                inun_arr = np.array(wl_sm_ > dem_file_arr).astype(np.uint8)
                bf.create_folder(f'{output_filepath}\\inundation_temp\\')
                bf.write_raster(dem_file_ds, inun_arr, f'{output_filepath}\\inundation_temp\\', str(wl_nm_) + '.tif', raster_datatype=gdal.GDT_Byte)

                src_temp = rasterio.open(f'{output_filepath}\\inundation_temp\\{str(wl_nm_)}.tif')
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
                        if thalweg.intersects(shp_file['geometry'][__]):
                            nw_shp_list.append(shp_list[__])
                nw_shp_file = gp.GeoDataFrame.from_features(nw_shp_list)

                bf.create_folder(f'{output_filepath}\\inundation_final\\')
                with rasterio.open(f'{output_filepath}\\inundation_final\\{str(wl_nm_)}.tif', 'w+', **meta) as out:
                    out_arr = out.read(1)

                    # this is where we create a generator of geom, value pairs to use in rasterizing
                    shapes = ((geom, value) for geom, value in zip(nw_shp_file.geometry, nw_shp_file.raster_val))

                    burned = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=src_temp.transform)
                out.write_band(1, burned)
            print(f'The {str(wl_nm_)} is generated!')
        except:
            print(traceback.format_exc())
            print(f'The {str(wl_nm_)} is not generated!!!!!')


def process_hydroinform_df(hydro_inform):
    try:
        yearly_hydroinform_all = []
        with tqdm(total=len(hydro_inform), desc=f'Process hydroinform df', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            for _ in hydro_inform:
                yearly_hydroinform = []
                wl_list = _.split('[')
                for __ in range(2, len(_.split('['))):
                    wl_temp = []
                    wl_ = wl_list[__].split(']')[0].split(', ')
                    for ___ in wl_:
                        if "'" in ___:
                            ___ = ___.split("'")[1]

                        try:
                            wl_temp.append(float(___))
                        except:
                            wl_temp.append(str(___))

                    yearly_hydroinform.append(wl_temp)
                yearly_hydroinform_all.append(yearly_hydroinform)
                pbar.update()
        return yearly_hydroinform_all
    except:
        print(traceback.format_exc())


def generate_hydrodatacube(year, y_list, x_list, hydro_dic, hydro_inform):

    # Define the sparse matrix
    ymax, xmax = int(np.max(y_list)) + 1, int(np.max(x_list)) + 1
    doy_list = [year * 1000 + _ for _ in range(1, datetime(year=year + 1, month=1, day=1).toordinal() - datetime(year=year, month=1, day=1).toordinal() + 1)]
    sm_list = [sm.lil_matrix((ymax, xmax)) for _ in range(len(doy_list))]

    with tqdm(total=len(y_list), desc=f'Generate hydrodatcube', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
        for _ in range(len(y_list)):
            y, x = int(y_list[_]), int(x_list[_])
            wl_start_series = np.array(hydro_dic[hydro_inform[_][1]])
            wl_end_series = np.array(hydro_dic[hydro_inform[_][2]])
            wl_start_dis = hydro_inform[_][3]
            wl_end_dis = hydro_inform[_][4]
            wl_inter = wl_start_series + (wl_end_series - wl_start_series) * wl_start_dis / (wl_start_dis + wl_end_dis)
            for __ in range(len(wl_inter)):
                sm_list[__][y, x] = wl_inter[__]
            pbar.update()
    ND_temp = NDSparseMatrix(*sm_list, SM_namelist=doy_list)
    ND_temp.save('G:\\A_Landsat_veg\\Water_level_python\\hydrodatacube\\')


def retrieve_srs(ds_temp):
    proj = osr.SpatialReference(wkt=ds_temp.GetProjection())
    srs_temp = proj.GetAttrValue('AUTHORITY', 1)
    srs_temp = 'EPSG:' + str(srs_temp)
    return srs_temp


def distance_between_2points(point1, point2):

    # Check the type
    if isinstance(point1, Point):
        point1_x = point1.coords[0][0]
        point1_y = point1.coords[0][1]
    elif isinstance(point1, list):
        point1_x = point1[0]
        point1_y = point1[1]
    elif isinstance(point1, tuple):
        point1_x = point1[0]
        point1_y = point1[1]
    else:
        raise TypeError('Wrong datatype!')

    # Check the type
    if isinstance(point2, Point):
        point2_x = point2.coords[0][0]
        point2_y = point2.coords[0][1]
    elif isinstance(point2, list):
        point2_x = point2[0]
        point2_y = point2[1]
    elif isinstance(point2, tuple):
        point2_x = point2[0]
        point2_y = point2[1]
    else:
        raise TypeError('Wrong datatype!')

    return np.sqrt((point1_x - point2_x) ** 2 + (point1_y - point2_y) ** 2)


def determin_start_vertex_of_point(point1, line: LineString):

    # Check the type
    if isinstance(point1, Point):
        point1_x = point1.coords[0][0]
        point1_y = point1.coords[0][1]
    elif isinstance(point1, list):
        point1_x = point1[0]
        point1_y = point1[1]
    elif isinstance(point1, tuple):
        point1_x = point1[0]
        point1_y = point1[1]
    else:
        raise TypeError('Wrong datatype!')

    start_vertex = None
    line_coords = list(line.coords)
    for _ in range(len(line_coords) - 1):
        if (line_coords[_][0] - point1_x) * (line_coords[_ + 1][0] - point1_x) <= 0:
            if (line_coords[_][1] - point1_y) * (line_coords[_ + 1][1] - point1_y) <= 0:
                point1_y_temp = line_coords[_][1] + (line_coords[_ + 1][1] - line_coords[_][1]) * (point1_x - line_coords[_][0]) / (line_coords[_ + 1][0] - line_coords[_][0])
                if point1_y_temp - point1_y < 0.1:
                    start_vertex = _
                    break
    if start_vertex is None:
        raise Exception('The point is not on the line')
    return start_vertex


def dis2points_via_line(point1, point2, line_temp: LineString, line_distance=None):

    if line_distance is not None:
        if len(line_distance) != len(list(line_temp.coords)):
            raise Exception('Line distance is not consistent with the line string!')

    t1_all, t2_all, t3_all = 0, 0, 0
    t1 = time.time()
    # Check the type
    if isinstance(point1, Point):
        point1_x = point1.coords[0][0]
        point1_y = point1.coords[0][1]
    elif isinstance(point1, list):
        point1_x = point1[0]
        point1_y = point1[1]
    elif isinstance(point1, tuple):
        point1_x = point1[0]
        point1_y = point1[1]
    else:
        raise TypeError('Wrong datatype!')

    # Check the type
    if isinstance(point2, Point):
        point2_x = point2.coords[0][0]
        point2_y = point2.coords[0][1]
    elif isinstance(point2, list):
        point2_x = point2[0]
        point2_y = point2[1]
    elif isinstance(point2, tuple):
        point2_x = point2[0]
        point2_y = point2[1]
    else:
        raise TypeError('Wrong datatype!')

    if not isinstance(line_temp, LineString):
        raise TypeError('The Line should under the shapely.linestring type')
    else:
        line_arr = np.array(line_temp.coords)
    t1_all += time.time()-t1

    t2 = time.time()
    # Process line and points
    point1_index, point2_index = None, None
    # Solution 1
    for _ in range(line_arr.shape[0] - 1):
        if point1_index is None:
            if (line_arr[_, 0] - point1_x) * (line_arr[_ + 1, 0] - point1_x) <= 0:
                if (line_arr[_, 1] - point1_y) * (line_arr[_ + 1, 1] - point1_y) <= 0:
                    point1_y_temp = line_arr[_, 1] + (line_arr[_ + 1, 1] - line_arr[_, 1]) * (point1_x - line_arr[_, 0]) / (line_arr[_ + 1, 0] - line_arr[_, 0])
                    if point1_y_temp - point1_y < 0.1:
                        point1_index = _
        if point2_index is None:
            if (line_arr[_, 0] - point2_x) * (line_arr[_ + 1, 0] - point2_x) <= 0:
                if (line_arr[_, 1] - point2_y) * (line_arr[_ + 1, 1] - point2_y) <= 0:
                    point2_y_temp = line_arr[_, 1] + (line_arr[_ + 1, 1] - line_arr[_, 1]) * (point2_x - line_arr[_, 0]) / (line_arr[_ + 1, 0] - line_arr[_, 0])
                    if point2_y_temp - point2_y < 0.1:
                        point2_index = _
        if point1_index is not None and point2_index is not None:
            break
    # Solution 2
    # itr = abs(line_arr[0, 0] - line_arr[-1, 0]) / line_arr.shape[0]
    # np.argwhere(np.logical_and(point1_x - itr < line_arr[:, 0], line_arr[:, 0] < point1_x + 100))
    #
    t2_all += time.time() - t2

    t3 = time.time()
    # Check if two points both on the line
    if point1_index is None or point2_index is None:
        raise Exception('Make sure both points are on the line')
    else:
        if point1_index > point2_index:
            dis = 0
            for _ in range(point2_index + 1, point1_index + 2):
                if _ == point2_index + 1:
                    dis += distance_between_2points([line_arr[_, 0], line_arr[_, 1]], [point2_x, point2_y])
                elif _ == point1_index + 1:
                    dis += distance_between_2points([line_arr[_ - 1, 0], line_arr[_ - 1, 1]], [point1_x, point1_y])
                else:
                    if line_distance is None:
                        dis += distance_between_2points([line_arr[_, 0], line_arr[_, 1]], [line_arr[_ - 1, 0], line_arr[_ - 1, 1]])
                    else:
                        dis += line_distance[_]
            t3_all += time.time() - t3
            # print(f'{str(t1_all)}s, {str(t2_all)}s, {str(t3_all)}s')
            return dis
        elif point2_index > point1_index:
            dis = 0
            for _ in range(point1_index + 1, point2_index + 2):
                if _ == point2_index + 1:
                    dis += distance_between_2points([line_arr[_ - 1, 0], line_arr[_ - 1, 1]], [point2_x, point2_y])
                elif _ == point1_index + 1:
                    dis += distance_between_2points([line_arr[_, 0], line_arr[_, 1]], [point1_x, point1_y])
                else:
                    if line_distance is None:
                        dis += distance_between_2points([line_arr[_, 0], line_arr[_, 1]], [line_arr[_ - 1, 0], line_arr[_ - 1, 1]])
                    else:
                        dis += line_distance[_]
            t3_all += time.time() - t3
            # print(f'{str(t1_all)}s, {str(t2_all)}s, {str(t3_all)}s')
            return dis
        elif point1_index == point2_index:
            t3_all += time.time() - t3
            # print(f'{str(t1_all)}s, {str(t2_all)}s, {str(t3_all)}s')
            return distance_between_2points([point1_x, point1_y], [point2_x, point2_y])


def frequency_based_elevation(df: pd.DataFrame, thal, year_range, geotransform: list, cs_list: list, year_domain: list, hydro_pos: list, hydro_datacube: bool):

    try:

        # Drop unnes water level
        all_year_list, year_doy = [], []
        for _ in cs_list:
            unne_series = None
            for year in range(year_range[0], year_range[1]):
                if unne_series is None:
                    unne_series = (thal.hydro_inform_dic[_]['year'] != year)
                else:
                    unne_series = unne_series & (thal.hydro_inform_dic[_]['year'] != year)

            thal.hydro_inform_dic[_] = thal.hydro_inform_dic[_].drop(thal.hydro_inform_dic[_][unne_series].index).reset_index(drop=True)
            year_domain[cs_list.index(_)] = np.unique(np.array(thal.hydro_inform_dic[_]['year'])).tolist()

        # Generate year list
        for year_ in range(year_range[0], year_range[1]):
            all_year_list.append(year_)
            year_doy.append(datetime(year=year_ + 1, month=1, day=1).toordinal() - datetime(year=year_, month=1, day=1).toordinal())

        # Define the output data
        ul_x, x_res, ul_y, y_res = geotransform
        df = df.reset_index(drop=True)
        wl, fr, yearly_wl = [], [], []

        with tqdm(total=df.shape[0], desc=f'Process the inundation frequency', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            # t1_all, t2_all, t3_all, t4_all, t5_all, t6_all = 0, 0, 0, 0, 0, 0
            for len_t in range(df.shape[0]):
                y_temp, x_temp, if_value = int(df['y'][len_t]), int(df['x'][len_t]), df['if'][len_t]
                if hydro_datacube:
                    yearly_wl_temp = [[] for _ in range(year_range[0], year_range[1])]

                if ~np.isnan(if_value):
                    coord_x, coord_y = ul_x + (x_temp + 0.5) * x_res, ul_y + (y_temp + 0.5) * y_res
                    wl_all = []
                    hydro_pos_temp = copy.deepcopy(hydro_pos)

                    if thal.smoothed_Thelwag is None:
                        nearest_p = nearest_points(Point([coord_x, coord_y]), thal.Thelwag_Linestring)[1]
                        start_vertex_index = determin_start_vertex_of_point(nearest_p, thal.Thelwag_Linestring)

                        # Determine the hydrostation
                        factor = None
                        hydro_index_list, year_list = [], []
                        for year in range(year_range[0], year_range[1]):
                            start_hydro_index = copy.deepcopy(start_vertex_index)
                            end_hydro_index = copy.deepcopy(start_vertex_index + 1)

                            if start_hydro_index < min(hydro_pos):
                                factor = 1
                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= start_hydro_index]
                                if len(start_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = min(start_hydro_index_list)

                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] > start_hydro_index]
                                if len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    end_hydro_index = min(end_hydro_index_list)

                            elif end_hydro_index > max(hydro_pos):
                                factor = 2
                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= end_hydro_index]
                                if len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    end_hydro_index = max(end_hydro_index_list)

                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] < end_hydro_index]
                                if len(start_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = max(start_hydro_index_list)
                            else:
                                factor = 3
                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= start_hydro_index]
                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= end_hydro_index]

                                if len(start_hydro_index_list) == 0 or len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = max(start_hydro_index_list)
                                    end_hydro_index = min(end_hydro_index_list)

                            year_list.append(year)
                            hydro_index_list.append([start_hydro_index, end_hydro_index])

                        hydro_unique_index_list = np.unique(np.array(hydro_index_list), axis=0).tolist()
                        inform_list = []
                        for _ in hydro_unique_index_list:
                            inform_temp = []
                            # _ 0/1 coord
                            inform_temp.append(Point([thal.Thelwag_Linestring.coords[start_hydro_index][0], thal.Thelwag_Linestring.coords[start_hydro_index][1]]))
                            inform_temp.append(Point([thal.Thelwag_Linestring.coords[end_hydro_index][0], thal.Thelwag_Linestring.coords[end_hydro_index][1]]))
                            # _ 2/3 dis
                            inform_temp.append(dis2points_via_line(nearest_p, inform_temp[0], thal.Thelwag_Linestring))
                            inform_temp.append(dis2points_via_line(nearest_p, inform_temp[1], thal.Thelwag_Linestring))
                            inform_list.append(inform_temp)

                        for year in year_list:
                            ii = year_list.index(year)
                            start_hydro_index = hydro_index_list[ii][0]
                            end_hydro_index = hydro_index_list[ii][1]

                            iii = hydro_unique_index_list.index(hydro_index_list[ii])
                            dis_to_start_station = inform_list[iii][2]
                            dis_to_end_station = inform_list[iii][3]

                            start_cs = cs_list[hydro_pos.index(start_hydro_index)]
                            end_cs = cs_list[hydro_pos.index(end_hydro_index)]

                            wl_start = np.array(thal.hydro_inform_dic[start_cs][thal.hydro_inform_dic[start_cs]['year'] == year]['water_level/m'])
                            wl_end = np.array(thal.hydro_inform_dic[end_cs][thal.hydro_inform_dic[end_cs]['year'] == year]['water_level/m'])

                            if wl_start.shape[0] != wl_end.shape[0]:
                                raise ValueError(f'The water level of {cs_list[end_hydro_index]} and {cs_list[start_hydro_index]} in year {str(year)} is not consistent')
                            else:
                                if factor == 1:
                                    wl_pos = wl_start - (wl_end - wl_start) * dis_to_start_station / (dis_to_end_station - dis_to_start_station)
                                elif factor == 2:
                                    wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station - dis_to_end_station)
                                elif factor == 3:
                                    wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station + dis_to_end_station)

                            wl_all.extend(wl_pos.flatten().tolist())

                    elif isinstance(thal.smoothed_Thelwag, LineString):

                        nearest_p = nearest_points(Point([coord_x, coord_y]), thal.smoothed_Thelwag)[1]
                        start_vertex_index = determin_start_vertex_of_point(nearest_p, thal.smoothed_Thelwag)

                        # Determine the hydrostation
                        # t1 = time.time()
                        factor = None
                        hydro_index_list, year_list = [], []
                        for year in range(year_range[0], year_range[1]):
                            start_hydro_index = copy.deepcopy(start_vertex_index)
                            end_hydro_index = copy.deepcopy(start_vertex_index + 1)

                            if start_hydro_index < min(hydro_pos):
                                factor = 1
                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= start_hydro_index]
                                if len(start_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = min(start_hydro_index_list)

                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] > start_hydro_index]
                                if len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    end_hydro_index = min(end_hydro_index_list)

                            elif end_hydro_index > max(hydro_pos):
                                factor = 2
                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= end_hydro_index]
                                if len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    end_hydro_index = max(end_hydro_index_list)

                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] < end_hydro_index]
                                if len(start_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = max(start_hydro_index_list)

                            else:
                                factor = 3
                                start_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] <= start_hydro_index]
                                end_hydro_index_list = [hydro_pos_temp[i] for i in range(len(hydro_pos_temp)) if year in year_domain[i] and hydro_pos_temp[i] >= end_hydro_index]

                                if len(start_hydro_index_list) == 0 or len(end_hydro_index_list) == 0:
                                    raise Exception('Code Error!')
                                else:
                                    start_hydro_index = max(start_hydro_index_list)
                                    end_hydro_index = min(end_hydro_index_list)

                            year_list.append(year)
                            hydro_index_list.append([start_hydro_index, end_hydro_index])
                        # t1_all += time.time() - t1

                        hydro_unique_index_list = np.unique(np.array(hydro_index_list), axis=0).tolist()
                        for _ in hydro_unique_index_list:

                            # get the hydro station index
                            start_hydro_index = _[0]
                            end_hydro_index = _[1]

                            # Calculate the dis between nearest p with start and end station
                            t2 = time.time()
                            dis_to_start_station = dis2points_via_line(nearest_p, [thal.smoothed_Thelwag.coords[start_hydro_index][0], thal.smoothed_Thelwag.coords[start_hydro_index][1]], thal.smoothed_Thelwag)
                            dis_to_end_station = dis2points_via_line(nearest_p, [thal.smoothed_Thelwag.coords[end_hydro_index][0], thal.smoothed_Thelwag.coords[end_hydro_index][1]], thal.smoothed_Thelwag)
                            # t2_all += time.time() - t2

                            # Retrieve the water series of start and end stations
                            # t3 = time.time()
                            start_cs = cs_list[hydro_pos.index(start_hydro_index)]
                            end_cs = cs_list[hydro_pos.index(end_hydro_index)]

                            year_unique_list = [year_list[__] for __ in range(len(hydro_index_list)) if hydro_index_list[__] == _]
                            if max(year_unique_list) - min(year_unique_list) == len(year_unique_list) - 1:
                                wl_start_series = (thal.hydro_inform_dic[start_cs]['year'] >= min(year_unique_list)) & (thal.hydro_inform_dic[start_cs]['year'] <= max(year_unique_list))
                                wl_end_series = (thal.hydro_inform_dic[end_cs]['year'] >= min(year_unique_list)) & (thal.hydro_inform_dic[end_cs]['year'] <= max(year_unique_list))
                            else:
                                wl_start_series, wl_end_series = None, None
                                for year in year_unique_list:
                                    if wl_start_series is None:
                                        wl_start_series = (thal.hydro_inform_dic[start_cs]['year'] == year)
                                    else:
                                        wl_start_series = wl_start_series | (thal.hydro_inform_dic[start_cs]['year'] == year)

                                    if wl_end_series is None:
                                        wl_end_series = (thal.hydro_inform_dic[end_cs]['year'] == year)
                                    else:
                                        wl_end_series = wl_end_series | (thal.hydro_inform_dic[end_cs]['year'] == year)
                            # t3_all += time.time() - t3

                            # Interpolate the water level
                            # t4 = time.time()
                            wl_start = thal.hydro_inform_dic[start_cs][wl_start_series]['water_level/m'].reset_index(drop=True)
                            wl_end = thal.hydro_inform_dic[end_cs][wl_end_series]['water_level/m'].reset_index(drop=True)

                            if wl_start.shape[0] != wl_end.shape[0]:
                                raise ValueError(f'The water level of {cs_list[end_hydro_index]} and {cs_list[start_hydro_index]} in year {str(year)} is not consistent')
                            else:
                                if factor == 1:
                                    wl_pos = wl_start - (wl_end - wl_start) * dis_to_start_station / (dis_to_end_station - dis_to_start_station)
                                elif factor == 2:
                                    wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station - dis_to_end_station)
                                elif factor == 3:
                                    wl_pos = wl_start + (wl_end - wl_start) * dis_to_start_station / (dis_to_start_station + dis_to_end_station)
                            # t4_all += time.time() - t4

                            wl_pos = list(wl_pos)
                            wl_all.extend(wl_pos)

                            if hydro_datacube:
                                for year_temp in year_unique_list:
                                    yearly_wl_temp[all_year_list.index(year_temp)] = [year_temp, start_cs, end_cs, dis_to_start_station, dis_to_end_station]

                    else:
                        raise Exception('Code Error')

                    t6 = time.time()
                    if if_value == 1:
                        wl.append(np.nan)
                        fr.append(1)
                    else:
                        wl_all = np.sort(np.array(wl_all))
                        inun_freq = np.linspace(1, 1 / wl_all.shape[0], wl_all.shape[0])
                        wl_factor = False
                        for _ in range(inun_freq.shape[0] - 1):
                            if (inun_freq[_] - if_value) * (inun_freq[_ + 1] - if_value) < 0:
                                wl.append(wl_all[_])
                                fr.append(inun_freq[_])
                                wl_factor = True
                                break
                            elif inun_freq[_] - if_value == 0:
                                wl.append(wl_all[_])
                                fr.append(inun_freq[_])
                                wl_factor = True
                                break
                            elif inun_freq[_ + 1] - if_value == 0:
                                wl.append(wl_all[_ + 1])
                                fr.append(inun_freq[_ + 1])
                                wl_factor = True
                                break
                        # for _ in range(wl_unique.shape[0] - 1):
                        #     if_lower = np.sum(wl_all <= wl_unique[_]) / wl_all.shape[0]
                        #     if_higher = np.sum(wl_all <= wl_unique[_ + 1]) / wl_all.shape[0]
                        #     if if_lower <= if_value and if_higher > if_value:
                        #         wl.append(wl_unique[_])
                        #         fr.append(np.sum(wl_all >= wl_unique[_]) / wl_all.shape[0])
                        #         wl_factor = True
                        #         break
                        if wl_factor is False:
                            wl.append(np.nan)
                            fr.append(np.nan)

                    # t6_all += t6 + time.time()
                    # if np.mod(len_t, 5000) == 0:
                    #     print(f'{str(t1_all)}s, {str(t2_all)}s, {str(t3_all)}s, {str(t4_all)}s, {str(t5_all)}s, {str(t6_all)}s')
                elif np.isnan(if_value):
                    wl.append(np.nan)
                    fr.append(np.nan)

                if hydro_datacube:
                    yearly_wl.append(yearly_wl_temp)

                pbar.update()
        df['wl'] = wl
        df['fr'] = fr
        if hydro_datacube:
            df['yearly_wl'] = yearly_wl
        return df
    except:
        print(traceback.format_exc())